import os
import sys

import numpy as np
from sklearn.metrics import log_loss
from sklearn.base import BaseEstimator, ClassifierMixin
from sknn import mlp
import logging

logging.basicConfig()

DATA_FILE = os.path.join(os.path.abspath(os.path.dirname(__file__)), './crimes_data.npz')


def get_params(params):
    """
    Values in params are numpy array, we need to get a list out of it
    :param params:
    :return: a new dict
    """
    parsed_params = {}
    for k, v in params.iteritems():
        vv = v
        if isinstance(v, np.ndarray):
            vv = v.tolist()
            if len(vv) == 1:
                vv = vv[0]
            if isinstance(v, basestring) and len(v) == 0:
                vv = None
        elif isinstance(v, list) and len(v) == 1:
            vv = vv[0]

        if isinstance(vv, basestring) and len(vv) == 0:
            vv = None

        parsed_params[k] = vv
    return parsed_params


def create_labels(labels, all_labels):
    lb = np.zeros_like(labels, dtype=np.int)
    for i, v in enumerate(all_labels):
        lb[labels == v] = i
    return lb


def main(job_id, params):
    print job_id, params
    params = get_params(params)
    print job_id, params

    crimes = np.load(DATA_FILE)
    features_train = crimes['features_train']
    all_labels = sorted(list(set(np.unique(crimes['labels_train'])) | set(np.unique(crimes['labels_val']))))
    batch_size = 64

    labels_train = create_labels(crimes['labels_train'], all_labels)
    labels_vals = create_labels(crimes['labels_val'], all_labels)
    labels_full = create_labels(crimes['labels'], all_labels)

    layers = [mlp.Layer('Linear', name='input', units=features_train.shape[1], dropout=params['input_dropout'])]

    for i in range(0, params['layers']):
        layers.append(mlp.Layer('Rectifier', name='hidden_{}'.format(i),
                                units=int(params['hidden_units']),
                                dropout=params['hidden_dropout']))
    layers.append(mlp.Layer('Softmax', dropout=0, units=len(all_labels)))

    model = mlp.Classifier(layers=layers,
                           learning_rate=params['learning_rate'],
                           n_iter=20 * (features_train.shape[0] / batch_size),
                           random_state=42,
                           learning_rule='adagrad',
                           batch_size=batch_size,
                           weight_decay=params['weight_decay'],
                           valid_set=(crimes['features_val'], labels_vals))

    try:
        model.fit(features_train, labels_train)
    except RuntimeError as e:
        if 'diverged' in e.message:
            # super bad
            return 100
        raise

    loss_train = log_loss(labels_train, model.predict_proba(crimes['features_train']))
    loss_val = log_loss(labels_vals, model.predict_proba(crimes['features_val']))
    loss_all = log_loss(labels_full, model.predict_proba(crimes['features']))

    print 'loss_all: ', loss_all
    print 'loss_train: ', loss_train
    print 'loss_val: ', loss_val
    sys.stdout.flush()

    return loss_val


'''
For random search
'''


class NeuralNetworkClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, n_classes=1, batch_size=64, valid_set=None,
                 layers=1, hidden_units=64, input_dropout=0, hidden_dropout=0.5,
                 learning_rate=0.01, weight_decay=0):
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.valid_set = valid_set

        self.layers = layers
        self.hidden_units = hidden_units
        self.input_dropout = input_dropout
        self.hidden_dropout = hidden_dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def fit(self, X, y):
        layers = [mlp.Layer('Linear', name='input', units=X.shape[1], dropout=self.input_dropout)]

        for i in range(0, self.layers):
            layers.append(mlp.Layer('Rectifier', name='hidden_{}'.format(i),
                                    units=int(self.hidden_units), dropout=self.hidden_dropout))
        layers.append(mlp.Layer('Softmax', dropout=0, units=self.n_classes))

        self._model = mlp.Classifier(layers=layers,
                                     learning_rate=self.learning_rate,
                                     n_iter=20 * (X.shape[0] / self.batch_size),
                                     random_state=42,
                                     learning_rule='adagrad',
                                     batch_size=self.batch_size,
                                     weight_decay=self.weight_decay,
                                     valid_set=self.valid_set)
        self._model.fit(X, y)

    def predict(self, X):
        return self._model.predict(X)

    def predict_proba(self, X):
        return self._model.predict_proba(X)

    def get_params(self, deep=True):
        return {'n_classes': self.n_classes, 'batch_size': self.batch_size, 'valid_set': self.valid_set,
                "layers": self.layers, "hidden_units": self.hidden_units,
                'input_dropout': self.input_dropout, 'hidden_dropout': self.hidden_dropout,
                'learning_rate': self.learning_rate, 'weight_decay': self.weight_decay}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def score(self, X, y, sample_weight=None):
        return log_loss(y, self.predict_proba(X), sample_weight=sample_weight)


def report(grid_scores, n_top=3):
    from operator import itemgetter

    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")


def random_search():
    from time import time
    from scipy.stats import uniform as sp_uniform, randint as sp_randint
    from sklearn.grid_search import RandomizedSearchCV

    crimes = np.load(DATA_FILE)
    # features_train = crimes['features_train']
    all_labels = sorted(list(set(np.unique(crimes['labels_train'])) | set(np.unique(crimes['labels_val']))))
    batch_size = 64

    labels_train = create_labels(crimes['labels_train'], all_labels)
    labels_vals = create_labels(crimes['labels_val'], all_labels)
    labels_full = create_labels(crimes['labels'], all_labels)

    param_dist = {'layers': sp_randint(1, 3),
                  "hidden_units": [64, 128, 256],
                  'input_dropout': sp_uniform(0, 0.5),
                  "hidden_dropout": sp_uniform(0, 0.75),
                  "learning_rate": sp_uniform(0.01, 0.1),
                  "weight_decay": sp_uniform(0, 0.01)
                  }

    model = NeuralNetworkClassifier(n_classes=len(all_labels), batch_size=batch_size,
                                    valid_set=(crimes['features_val'], labels_vals))

    n_iter_search = 40
    np.random.seed(42)

    random_searcher = RandomizedSearchCV(model, param_distributions=param_dist, scoring=None,
                                         n_iter=n_iter_search, random_state=42, error_score=100, verbose=5)

    start = time()
    random_searcher.fit(crimes['features_train'], labels_train.ravel())

    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time() - start), n_iter_search))
    report(random_searcher.grid_scores_)

    loss_train = log_loss(labels_train, random_searcher.predict_proba(crimes['features_train']))
    loss_val = log_loss(labels_vals, random_searcher.predict_proba(crimes['features_val']))
    loss_all = log_loss(labels_full, random_searcher.predict_proba(crimes['features']))

    print 'loss_all: ', loss_all
    print 'loss_train: ', loss_train
    print 'loss_val: ', loss_val

    return loss_val

if __name__ == '__main__':
    # main(0, {'input_dropout': 0, 'layers': 1,
    #          'hidden_func': 'Rectifier', 'hidden_units': 128,
    #          'hidden_dropout': 0.5, 'learning_rate': 0.01,
    #          'weight_decay': 0.001})
    random_search()

'''
loss_all:  2.19293237468
loss_train:  2.19113759238
loss_val:  2.19472238042
'''
