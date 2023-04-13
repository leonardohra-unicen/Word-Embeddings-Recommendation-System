import json
import os

import numpy as np
from gensim.models import FastText
from gensim.models.fasttext import load_facebook_model
from ray import tune as tuner
from ray.tune.schedulers import ASHAScheduler
from ray.tune.stopper import TrialPlateauStopper
from multiprocessing import cpu_count
from loggers.RecallAtKLogger import RecallAtKLogger

EMBEDDING_FILE = "/home/ana/gensim-data/crawl-300d-2M-subword.bin"


class FastTextModel:
    def __init__(self, ):
        self._apis = None
        self._bow_apis = None
        self._test = None
        self._train = None
        self._validation = None
        self._apis_bows = {}
        self._fasttext_model = None
    
    def initialize(self, api_info, bow_apis, pretrained, hyperparameters):
        self._apis = api_info
        self._bow_apis = bow_apis
        self._load_fasttext_model(pretrained, hyperparameters)
        if pretrained:
            self._load_api_bows()

    def initialize_evaluation(self, api_bows, pretrained, hyperparameters):
        self._fasttext_model = None
        self._apis_bows = {}
        self._load_fasttext_model(pretrained, hyperparameters)
        for api, bow in api_bows.items():
            self._apis_bows[api] = [word for word in bow if word in self._fasttext_vocab]

    def _load_fasttext_model(self, pretrained, hyperparameters):
        if pretrained:
            self._fasttext_model = load_facebook_model(EMBEDDING_FILE)
            self._fasttext_vocab = list(self._fasttext_model.wv.index_to_key)
        elif hyperparameters:
            self._fasttext_model = FastText(self._bow_apis, **hyperparameters)

    def _load_api_bows(self):
        for bow, api in zip(self._bow_apis, self._apis.values()):
            self._apis_bows[api[0]] = [word for word in bow if word in self._fasttext_vocab]

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
            "negative": tuner.grid_search([0, 1, 5, 10, 20]),
            "epochs": tuner.grid_search([10, 25, 35, 50]),
            "workers": cpu_count(),
            "sg": tuner.grid_search([0, 1]),
            "hs": tuner.grid_search([0, 1]),
        }
        analysis = tuner.run(
            self._tune_hyperparameters,
            metric="recall_at_k",
            mode="max",
            local_dir="./results/fasttext",
            scheduler=asha_scheduler,
            stop=stopping_criterion,
            verbose=1,
            num_samples=15,
            config=search_space,
        )
        results = analysis.best_dataframe
        results.to_csv("./results/fasttext/results.csv")

    def _tune_hyperparameters(self, hyperparameters):
        ratk_logger = RecallAtKLogger(self._validation, self._test)
        FastText(sentences=self._train, callbacks=[ratk_logger], **hyperparameters)

    def get_predictions(self, query):
        query_bow = list([x for x in query if x in self._fasttext_vocab])
        predictions = []
        for api in self._apis_bows.keys():
            if self._apis_bows[api]:
                predictions.append((api, self._fasttext_model.wv.n_similarity(query_bow, self._apis_bows[api])))
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
                query_bow = list([x for x in query_item if x in self._fasttext_vocab])
                predictions = []
                for api in self._apis_bows.keys():
                    if self._apis_bows[api]:
                        predictions.append((api, self._fasttext_model.wv.n_similarity(query_bow, self._apis_bows[api])))
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
