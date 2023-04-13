import copy
from gensim.models.callbacks import CallbackAny2Vec
from ray import tune
from evaluation import recall_at_k


class RecallAtKLogger(CallbackAny2Vec):
    '''Report Recall@K metric at end of each epoch

    Computes and reports Recall@K on a validation set with
    a given value of k (number of recommendations to generate).
    '''

    def __init__(self, validation, test_api_bows, k=10, ray=True):
        self._word2vec_vocab = None
        self.epoch = 0
        # Format: [query, [endpoints]
        self.validation = validation
        self.k = k
        self.ray = ray
        self._apis_bows = {}
        self._test_api_bows = test_api_bows
        self._loaded = False

    def on_epoch_end(self, model):
        # make deepcopy of the model and emulate training completion
        mod = copy.deepcopy(model)
        mod._clear_post_train()

        self._word2vec_vocab = list(mod.wv.index_to_key)
        for api, bow in self._test_api_bows.items():
            self._apis_bows[api] = [word for word in bow if word in self._word2vec_vocab]

        # compute the metric we care about on a recommendation task
        # with the validation set using the model's embedding vectors
        recall = 0
        for item in self.validation:
            query_item = item["query"].split()
            ground_truth = item["results"]
            try:
                # get the k most similar items to the query item
                query_bow = list([x for x in query_item if x in self._word2vec_vocab])
                predictions = []
                for api in self._apis_bows.keys():
                    if self._apis_bows[api]:
                        predictions.append((api, mod.wv.n_similarity(query_bow, self._apis_bows[api])))
                    else:
                        predictions.append((api, 0))
            except KeyError:
                pass
            except ZeroDivisionError:
                pass
            else:
                recommendations = sorted(predictions, key=lambda item: -item[1])[0:self.k]
                recall += recall_at_k(self.k, ground_truth, recommendations)[-1]
        recall /= len(self.validation)
        if self.ray:
            tune.report(recall_at_k=recall)
        else:
            print(f"Epoch {self.epoch} -- Recall@{self.k}: {recall}")
        self.epoch += 1
