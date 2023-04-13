import collections
import json
import os

import numpy as np
from gensim.models import KeyedVectors, Word2Vec
from ray import tune as tuner
from ray.tune.schedulers import ASHAScheduler
from ray.tune.stopper import TrialPlateauStopper
from multiprocessing import cpu_count
from loggers.RecallAtKLogger import RecallAtKLogger

EMBEDDING_FILE = "/home/ana/gensim-data/word2vec-google-news-300/word2vec-google-news-300.gz"


class Word2VecModel:
    def __init__(self, ):
        self._apis = None
        self._bow_apis = None
        self._test = None
        self._train = None
        self._validation = None
        self._apis_bows = {}
        self._word2vec_model = None
    
    def initialize(self, api_info, bow_apis, pretrained, hyperparameters):
        self._apis = api_info
        self._bow_apis = bow_apis
        self._load_word2vec_model(pretrained, hyperparameters, bow_apis)
        #if pretrained:
            #self._load_api_bows()

    def initialize_evaluation(self, api_bows, pretrained, hyperparameters):
        self._word2vec_model = None
        self._apis_bows = {}
        self._load_word2vec_model(pretrained, hyperparameters, [])
        for api, bow in api_bows.items():
            self._apis_bows[api] = [word for word in bow if word in self._word2vec_vocab]

    def _load_word2vec_model(self, pretrained, hyperparameters, data):
        if pretrained:
            self._word2vec_model = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)
            self._word2vec_model.init_sims(replace=True)
            self._word2vec_vocab = list(self._word2vec_model.index_to_key)
        elif hyperparameters:
            self._word2vec_model = Word2Vec(self._bow_apis, **hyperparameters)
        else:
            self._word2vec_model = Word2Vec(window=10, sg=1, hs=0,
                             negative=10,  # for negative sampling
                             alpha=0.03, min_alpha=0.0007,
                             seed=14, vector_size=300)
            self._word2vec_model.build_vocab(data, progress_per=200)
            self._word2vec_model.train(data, total_examples=self._word2vec_model.corpus_count, epochs=20)
            words = [item for list_words in data for item in list_words]
            self._vocab_count = collections.Counter(words)

    def _load_api_bows(self):
        for bow, api in zip(self._bow_apis, self._apis.values()):
            self._apis_bows[api[0]] = [word for word in bow if word in self._word2vec_vocab]

    def train_test_split(self, test_percentage=0.2, random_state=10, save_test=True):
        random_state = np.random.RandomState(random_state)
        keys = list(self._apis.keys())
        random_state.shuffle(keys)
        train_keys = keys[:int(len(keys) * (1 - test_percentage))]
        test_keys = keys[-int(len(keys) * test_percentage):]
        train_apis_bows = dict()
        test_apis_bows = dict()
        train = []
        for key in train_keys:
            train_apis_bows.update({key: self._bow_apis[int(key)]})
            train.append(self._bow_apis[int(key)])
        if save_test:
            for key in test_keys:
                api = self._apis[key][0]
                test_apis_bows[api] = self._bow_apis[int(key)]
            if not os.path.exists("test.json"):
                with open("test.json", 'w') as fp:
                    json.dump(test_apis_bows, fp)
        with open("./info/test/queries.json", 'r') as file:
            validation = json.load(file)
        return train, test_apis_bows, validation

    def train_model(self):
        self._train, self._test, self._validation = self.train_test_split()
        asha_scheduler = ASHAScheduler(max_t=100, grace_period=10)
        stopping_criterion = TrialPlateauStopper(metric='recall_at_k', std=0.002)
        search_space = {
            "vector_size": tuner.grid_search([10, 100, 300, 1200, 1800, 2400, 3000]),
            "window": tuner.grid_search([2, 4, 8, 10]),
            "negative": tuner.grid_search([0, 1, 5, 10]),
            "epochs": tuner.grid_search([10, 25, 35, 50]),
            "workers": cpu_count(),
            "sg": tuner.grid_search([0, 1]),
        }
        analysis = tuner.run(
            self._tune_hyperparameters,
            metric="recall_at_k",
            mode="max",
            local_dir="./results/word2vec",
            scheduler=asha_scheduler,
            verbose=1,
            num_samples=15,
            config=search_space,
        )
        results = analysis.best_dataframe
        results.to_csv("./results/word2vec/results.csv")

    def _tune_hyperparameters(self, hyperparameters):
        ratk_logger = RecallAtKLogger(self._validation, self._test)
        Word2Vec(sentences=self._train, callbacks=[ratk_logger], **hyperparameters)

    def get_predictions(self, query):
        query_bow = list([x for x in query if x in self._word2vec_vocab])
        predictions = []
        for api in self._apis_bows.keys():
            if self._apis_bows[api]:
                predictions.append((api, self._word2vec_model.n_similarity(query_bow, self._apis_bows[api])))
            else:
                predictions.append((api, 0))
        return sorted(predictions, key=lambda item: -item[1])[0:10]

    def evaluate(self, evaluation_metric):
        with open("./info/test/queries.json", 'r') as file:
            validation = json.load(file)
        score = 0
        for item in validation:
            query_item = item["query"].split()
            ground_truth = item["results"]
            try:
                query_bow = list([x for x in query_item if x in self._word2vec_vocab])
                predictions = []
                for api in self._apis_bows.keys():
                    if self._apis_bows[api]:
                        predictions.append((api, self._word2vec_model.n_similarity(query_bow, self._apis_bows[api])))
                    else:
                        predictions.append((api, 0))
                final_predictions = sorted(predictions, key=lambda item: -item[1])[0:10]
            except KeyError:
                pass
            else:
                recommendations = [item for item, distance in final_predictions]
                print(query_item)
                print(recommendations)
                score += evaluation_metric(10, ground_truth, recommendations)[-1]
        score /= len(validation)
        print(score)

    def get_word_count(self):
        total_count = 0
        with open("vocab.txt", "w") as file:
            lines = [word + " " + str(vocab_obj) + '\n' for word, vocab_obj in self._vocab_count.most_common()]
            file.writelines(lines)
        with open("vectors.txt", "w") as file:
            words = list(w for w in self._word2vec_model.wv.index_to_key)
            lines = [word + " " + " ".join(map(str,self._word2vec_model.wv[word])) + '\n' for word in words]
            file.writelines(lines)


    # TODO visualizar https://machinelearningmastery.com/develop-word-embeddings-python-gensim/?

    # def visualize_embeddings(self):
