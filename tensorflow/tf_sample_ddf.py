from __future__ import print_function
from __future__ import unicode_literals

from multiprocessing import Process, Queue

import six
import numpy as np
import tensorflow as tf

import arimo


class DDFDataFetcher(Process):

    def __init__(self):
        super(DDFDataFetcher, self).__init__()

    def next_batch(self):
        return None


class DDFRandomDataFetcher(DDFDataFetcher):
    def __init__(self, queue, server='', port=0,
                 username='', password='', ddf_uri='',
                 sample_size=20, batch_size=5, df_to_np_fn=None):
        super(DDFRandomDataFetcher, self).__init__()
        self.queue = queue
        self.server = server
        self.port = port
        self.username = username
        self.password = password
        self.ddf_uri = ddf_uri
        self.sample_size = sample_size
        self.batch_size = batch_size
        self.df_to_np_fn = df_to_np_fn

        if self.df_to_np_fn is None:
            def p(df):
                # take the last column to be the label
                return df.iloc[:, :-1].values, df.iloc[:, -1].values

            self.df_to_np_fn = p

    def run(self):
        s = arimo.connect(self.server, self.port, self.username, self.password)
        ddf = s.get_ddf(self.ddf_uri)
        frac = self.sample_size / float(len(ddf))
        remain_inp, remain_target = None, None

        while True:
            # df = ddf.sample(size=self.sample_size, replace=True)
            # sample() doesn't work on test-pe because of new PR merged
            ddfs = ddf.sample2ddf(fraction=frac, replace=True)
            inp, target = self.df_to_np_fn(ddfs.head(len(ddfs)))

            if remain_inp is not None:
                inp = np.vstack([remain_inp, inp])
            if remain_target is not None:
                target = np.vstack([remain_target, target])

            assert inp.shape[0] == target.shape[0]

            for i in range(0, inp.shape[0], self.batch_size):
                # will block if queue is full
                self.queue.put((inp[i:(i + self.batch_size), :], target[i:(i + self.batch_size), :]))

            if inp.shape[0] % self.batch_size != 0:
                idx = (inp.shape[0] / self.batch_size) * self.batch_size
                remain_inp = inp[idx:, :]
                remain_target = target[idx:, :]

    def next_batch(self):
        return self.queue.get()


class Trainer(object):

    def __init__(self, fetcher, input_size, n_classes):
        self.fetcher = fetcher

        self.x = tf.placeholder(tf.float32, [None, input_size])
        W = tf.Variable(tf.zeros([input_size, n_classes]))
        b = tf.Variable(tf.zeros([n_classes]))
        self.y = tf.nn.softmax(tf.matmul(self.x, W) + b)

        # Define loss and optimizer
        self.y_ = tf.placeholder(tf.float32, [None, n_classes])
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.y), reduction_indices=[1]))
        self.train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    def train(self, iterations):
        tf.initialize_all_variables().run()
        self.fetcher.start()
        for i in range(iterations):
            batch_xs, batch_ys = self.fetcher.next_batch()
            self.train_step.run({self.x: batch_xs, self.y_: batch_ys})
            if i % 2 == 0:
                print('Iteration {}'.format(i))

        correct_prediction = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        batch_xs, batch_ys = self.fetcher.next_batch()
        print('Accuracy on random sample: {}'.format(accuracy.eval({self.x: batch_xs, self.y_: batch_ys})))


if __name__ == '__main__':
    try:
        input = raw_input
    except NameError:
        pass

    q = Queue(20)
    #server = six.text_type(input('Server address: '))
    #port = int(six.text_type(input('Server port: ')))
    #username = six.text_type(input('User name: '))
    #passwd = six.text_type(input('Password: '))
    server, port, username, passwd = 'test-pe.arimo.com', 16000, 'testbot@adatao.com', 'Abc123..'
    ddf_uri = 'ddf://adatao/mtcars_tf'

    def get_batch(df):
        inp = df[['mpg', 'cyl', 'disp', 'hp', 'drat', 'wt', 'qesc', 'vs', 'gear', 'carb']].values
        target_idx = df['am'].values
        n = target_idx.shape[0]
        b = np.zeros((n, 2))
        b[np.arange(n), target_idx] = 1
        return inp, b

    ddf_fetcher = DDFRandomDataFetcher(q, server, port, username, passwd, ddf_uri=ddf_uri,
                                       df_to_np_fn=get_batch)
    sess = tf.InteractiveSession()
    trainer = Trainer(ddf_fetcher, 10, 2)
    trainer.train(20)
    ddf_fetcher.terminate()
