from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import copy
import time
import sys

import pandas as pd
import sigopt.interface

import crimes_job_nn


def run(token):

    conn = sigopt.interface.Connection(client_token=token)

    params = [
        {'name': 'layers', 'type': 'int', 'bounds': {'min': 1, 'max': 3}},
        {'name': 'hidden_units', 'type': 'categorical',
         'categorical_values': [
             {'enum_index': 1, 'name': '64', 'object': 'categorical_value'},
             {'enum_index': 2, 'name': '128', 'object': 'categorical_value'},
             {'enum_index': 3, 'name': '256', 'object': 'categorical_value'}
         ]},
        {'name': 'input_dropout', 'type': 'double', 'bounds': {'min': 1, 'max': 3}},
        {'name': 'hidden_dropout', 'type': 'double', 'bounds': {'min': 1, 'max': 3}, 'precision': 4},
        {'name': 'learning_rate', 'type': 'double', 'bounds': {'min': 1, 'max': 3}, 'precision': 4},
        {'name': 'weight_decay', 'type': 'double', 'bounds': {'min': 1, 'max': 3}, 'precision': 4},
    ]

    experiment = conn.experiments().create(name='crimes_sf', parameters=params)

    trace = []
    history = []
    best_params = {}
    best_score = -9E+9
    max_iters = 40
    try:
        for i in range(max_iters):

            # get suggestion from sigopt
            suggestion = conn.experiments(experiment.id).suggestions().create()
            params = copy.deepcopy(dict(suggestion.assignments))

            # Our template scripts always returns values as if it was minimizing
            # But SigOpt was designed to maximize, so we negate the score here.
            try:
                value = - crimes_job_nn.main(suggestion.id, params)
            except Exception as ex:
                if isinstance(ex, KeyboardInterrupt):
                    raise

                # super bad if there is exception happens
                print(ex)
                value = -9E+9

            ts = time.time()
            trace.append({'ts': ts, 'perf': value, 'job_id': suggestion.id,
                          'job_left': max_iters - i - 1, 'job_pending': 0, 'job_completed': i + 1})
            history_entry = {'job_id': suggestion.id, 'perf': value, 'time': ts}
            history_entry.update(params)
            history.append(history_entry)
            print(history_entry)

            if best_score < value:
                best_score = value
                best_params = copy.deepcopy(params)

            conn.experiments(experiment.id).observations().create(suggestion=suggestion.id, value=value)

    except KeyboardInterrupt:
        pass

    history = pd.DataFrame(history)
    trace = pd.DataFrame(data=trace)

    print(history)
    print(trace)
    print(best_params)

    history.to_csv('history.csv')
    trace.to_csv('trace.csv')


if __name__ == '__main__':
    run(sys.argv[1])
