import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss

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


def main(job_id, params):
    print job_id, params
    params = get_params(params)
    print job_id, params

    crimes = np.load(DATA_FILE)

    model = RandomForestClassifier(n_estimators=params['n_estimators'],
                                   criterion=params['criterion'],
                                   max_depth=None if params['max_depth'] < 1 else params['max_depth'],
                                   min_samples_split=params['min_samples_split'],
                                   min_samples_leaf=params['min_samples_leaf'],
                                   max_features=params['max_features'],
                                   min_weight_fraction_leaf=0.0,
                                   max_leaf_nodes=None, bootstrap=True, oob_score=False, n_jobs=4,
                                   random_state=42, verbose=0, warm_start=False, class_weight=None)
    model.fit(crimes['features_train'], crimes['labels_train'])
    loss_train = log_loss(crimes['labels_train'], model.predict_proba(crimes['features_train']))
    loss_val = log_loss(crimes['labels_val'], model.predict_proba(crimes['features_val']))
    loss_all = log_loss(crimes['labels'], model.predict_proba(crimes['features']))
    print 'loss_all: ', loss_all
    print 'loss_train: ', loss_train
    print 'loss_val: ', loss_val

    return loss_val


# Utility function to report best scores
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
    from scipy.stats import randint as sp_randint
    from sklearn.grid_search import RandomizedSearchCV

    crimes = np.load(DATA_FILE)

    param_dist = {'n_estimators': sp_randint(1, 150),
                  "criterion": ["gini", "entropy"],
                  'max_depth': sp_randint(1, 40),
                  "min_samples_split": sp_randint(2, 15),
                  "min_samples_leaf": sp_randint(1, 10),
                  "max_features": ['auto', 'sqrt', 'log2', None]
                  }

    model = RandomForestClassifier(min_weight_fraction_leaf=0.0,
                                   max_leaf_nodes=None, bootstrap=True, oob_score=False, n_jobs=4,
                                   random_state=42, verbose=0, warm_start=False, class_weight=None)

    n_iter_search = 40

    random_searcher = RandomizedSearchCV(model, param_distributions=param_dist,
                                         n_iter=n_iter_search, random_state=42)

    start = time()
    random_searcher.fit(crimes['features_train'], crimes['labels_train'].ravel())

    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time() - start), n_iter_search))
    report(random_searcher.grid_scores_)

    loss_train = log_loss(crimes['labels_train'], random_searcher.predict_proba(crimes['features_train']))
    loss_val = log_loss(crimes['labels_val'], random_searcher.predict_proba(crimes['features_val']))
    loss_all = log_loss(crimes['labels'], random_searcher.predict_proba(crimes['features']))
    print 'loss_all: ', loss_all
    print 'loss_train: ', loss_train
    print 'loss_val: ', loss_val

    return loss_val

if __name__ == '__main__':
    random_search()


'''
Vus-MacBook-Pro:sf_crimes vupham$ python crimes_job.py
RandomizedSearchCV took 9103.96 seconds for 40 candidates parameter settings.
Model with rank: 1
Mean validation score: 0.337 (std: 0.001)
Parameters: {'min_samples_leaf': 2, 'n_estimators': 68, 'min_samples_split': 10, 'criterion': 'gini', 'max_features': None, 'max_depth': 14}

Model with rank: 2
Mean validation score: 0.336 (std: 0.001)
Parameters: {'min_samples_leaf': 9, 'n_estimators': 109, 'min_samples_split': 3, 'criterion': 'entropy', 'max_features': None, 'max_depth': 35}

Model with rank: 3
Mean validation score: 0.335 (std: 0.002)
Parameters: {'min_samples_leaf': 6, 'n_estimators': 78, 'min_samples_split': 6, 'criterion': 'gini', 'max_features': None, 'max_depth': 21}

loss_all:  2.08987366098
loss_train:  1.96492968567
loss_val:  2.21481433197
'''





'''
1011 {u'min_samples_leaf': array([7]), u'n_estimators': array([90]), u'max_features': [u''],
u'criterion': [u'gini'], u'min_samples_split': array([12]), u'max_depth': array([14])}
1011 {u'min_samples_leaf': 7, u'n_estimators': 90, u'min_samples_split': 12, u'criterion':
u'gini', u'max_features': None, u'max_depth': 14}

loss_all:  2.09819951454
loss_train:  1.98386436413
loss_val:  2.21253144263
Got result 2.212531

Job file reloaded.
Completed successfully in 410.28 seconds. [2.212531]
setting job 1011 complete
set...
'''