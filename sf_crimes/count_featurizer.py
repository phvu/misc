import sys
from copy import deepcopy

import numpy as np
import pandas as pd


class CountFeaturizer(object):

    def __init__(self, input_col, label_col, output_col='count_features', min_count_per_cat=2):
        self.input_col = input_col
        self.label_col = label_col
        self.output_col = output_col
        self.min_count_per_cat = min_count_per_cat

    def fit(self, data):
        self.train_set = data
        self.inputs = sorted(data[self.input_col].unique())
        self.labels = sorted(data[self.label_col].unique(), key=lambda x: '{}'.format(x))
        label_counts_tmp = data.groupby([self.label_col]).size()
        label_counts = pd.Series([label_counts_tmp.loc[i] for i in self.labels])
        pair_counts = data.groupby([self.input_col, self.label_col]).size()
        self.input_counts = data.groupby([self.input_col]).size()
        self.logodds = {}
        self.logodds_class = {}
        self.default_logodds = np.log(label_counts / (float(len(data)) - label_counts))
        for inp in self.inputs:
            self.logodds_class[inp] = np.log(self.input_counts[inp] / (float(len(data)) - self.input_counts[inp]))
            self.logodds[inp] = deepcopy(self.default_logodds)
            for cat in pair_counts[inp].keys():
                if self.min_count_per_cat < pair_counts[inp].loc[cat] < self.input_counts[inp]:
                    prob_inp = pair_counts[inp].loc[cat] / float(self.input_counts[inp])
                    self.logodds[inp][self.labels.index(cat)] = np.log(prob_inp) - np.log(1.0 - prob_inp)
            self.logodds[inp] = pd.Series(self.logodds[inp])
            self.logodds[inp].index = range(len(self.labels))

    def transform(self, data):
        new_inputs = sorted(data[self.input_col].unique())
        new_input_counts = data.groupby(self.input_col).size()
        only_new = set(new_inputs + self.inputs) - set(self.inputs)
        in_both = set(new_inputs).intersection(self.inputs)
        new_cnt = 0.0 if data is self.train_set else float(len(data))

        for inp in only_new:
            self.logodds_class[inp] = np.log(new_input_counts[inp] /
                                             (float(len(self.train_set) + new_cnt) - new_input_counts[inp]))
            self.logodds[inp] = deepcopy(self.default_logodds)
            self.logodds[inp].index = range(len(self.labels))
        if data is not self.train_set:
            for inp in in_both:
                prob_inp = (self.input_counts[inp] + new_input_counts[inp]) / float(len(self.train_set) + new_cnt)
                self.logodds_class[inp] = np.log(prob_inp) - np.log(1. - prob_inp)

        for i in range(0, len(self.labels)):
            data['{}_{}'.format(self.output_col, i)] = data[self.input_col].apply(lambda _: self.logodds[_].loc[i])
        data['{}_class'.format(self.output_col)] = data[self.input_col].apply(lambda _: self.logodds_class[_])
        return data

if __name__ == '__main__':
    featurizer = CountFeaturizer(16, 17, 'count_features', 2)
    df = pd.read_csv(sys.argv[1], header=None)
    featurizer.fit(df)
    print(featurizer.transform(df))

"""
import count_featurizer as cf
s = '/Users/vupham/code/pAnalytics/pa/resources/test/airlineBig.csv'
# featurizer = cf.CountFeaturizer(16, 17, 'count_features', 2)
featurizer = cf.CountFeaturizer(16, 1, 'count_features', 2)
import pandas as pd
df = pd.read_csv(s, header=None)
featurizer.fit(df)
r = featurizer.transform(df)
"""