import os
import sys

import numpy as np
from sklearn.metrics import log_loss
import pandas as pd
from sknn import mlp
from keras.layers.advanced_activations import PReLU
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.utils import np_utils
import logging

logging.basicConfig(level=logging.DEBUG)

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

    print 'Done loading data'

    labels_train = create_labels(crimes['labels_train'], all_labels)
    labels_vals = create_labels(crimes['labels_val'], all_labels)
    labels_full = create_labels(crimes['labels'], all_labels)

    print 'Done process labels'

    if False:

        layers = [mlp.Layer('Rectifier', name='input', units=features_train.shape[1], dropout=params['input_dropout'])]

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

        print 'Start fitting'
        try:
            model.fit(features_train, labels_train)
        except RuntimeError as e:
            if 'diverged' in e.message:
                # super bad
                return 100
            raise
    else:

        labels_train = np_utils.to_categorical(labels_train)
        labels_vals = np_utils.to_categorical(labels_vals)
        labels_full = np_utils.to_categorical(labels_full)

        model = Sequential()
        model.add(Dense(input_dim=features_train.shape[1], output_dim=int(params['hidden_units']),
                        init='glorot_uniform'))
        model.add(PReLU(input_shape=(int(params['hidden_units']),)))
        model.add(Dropout(params['input_dropout']))

        for i in range(params['layers']):
            model.add(Dense(input_dim=params['hidden_units'], output_dim=params['hidden_units'], init='glorot_uniform'))
            model.add(PReLU(input_shape=(params['hidden_units'],)))
            model.add(BatchNormalization(input_shape=(params['hidden_units'],)))
            model.add(Dropout(params['hidden_dropout']))

        model.add(Dense(input_dim=params['hidden_units'], output_dim=len(all_labels), init='glorot_uniform'))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam')

        model.fit(features_train, labels_train, nb_epoch=20, batch_size=batch_size,
                  verbose=5, validation_data=(crimes['features_val'], labels_vals))

    loss_train = log_loss(labels_train, model.predict_proba(crimes['features_train']))
    loss_val = log_loss(labels_vals, model.predict_proba(crimes['features_val']))
    loss_all = log_loss(labels_full, model.predict_proba(crimes['features']))

    print 'loss_all: ', loss_all
    print 'loss_train: ', loss_train
    print 'loss_val: ', loss_val
    sys.stdout.flush()

    pred_df = pd.DataFrame(model.predict_proba(crimes['features_test']), columns=all_labels)
    pred_df.to_csv(os.path.join(os.path.abspath(os.path.dirname(__file__)), './crimeSF_NN_logodds.csv'),
                   index_label="Id", na_rep="0")

    return loss_val


if __name__ == '__main__':

    main(0, {'layers': 1, 'hidden_units': 128,
             'input_dropout': 0, 'hidden_dropout': 0.344659,
             'learning_rate': 0.05848, 'weight_decay': 0.000468})
    '''
    main(0, {'layers': 1, 'hidden_units': 64,
             'input_dropout': 0, 'hidden_dropout': 0,
             'learning_rate': 0.01, 'weight_decay': 0.000468})
    '''

'''
loss_all:  2.19293237468
loss_train:  2.19113759238
loss_val:  2.19472238042
'''
