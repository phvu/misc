import math
import time

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

WORKER_TASKS = ['/job:worker/task:0', '/job:worker/task:1']
PS_DEVICE = '/job:ps/task:0'
GRPC_SERVER = "grpc://localhost:2222"

HIDDEN_SIZE = 512
LAYERS = 2
ITERATIONS = 1000
EVAL_EVERY = 50
BATCH_SIZE = 64
L2_REGULARIZER = 1E-5


class Model(object):
    def __init__(self, variable_device, variable_scope, worker_device, worker_name_scope, reuse, optimizer):

        weights = []
        biases = []
        with tf.device(variable_device):
            with tf.variable_scope(variable_scope, reuse=reuse):
                for i in range(0, LAYERS + 1):
                    ws = [784 if i == 0 else HIDDEN_SIZE, 10 if i == LAYERS else HIDDEN_SIZE]

                    # Xavier initialization
                    weights.append(tf.get_variable("weight{}".format(i), ws,
                                                   initializer=tf.truncated_normal_initializer(
                                                       stddev=math.sqrt(3.0 / (ws[0] + ws[1])))))
                    biases.append(tf.get_variable("bias{}".format(i), ws[1:],
                                                  initializer=tf.constant_initializer(value=0, dtype=tf.float32)))
            # tf.get_variable_scope().reuse_variables()

        with tf.device(worker_device):
            with tf.name_scope(worker_name_scope):
                self.inp = tf.placeholder(tf.float32, shape=[None, 784])
                self.labels = tf.placeholder(tf.float32, shape=[None, 10])

                self.hiddens = []
                for i in range(0, LAYERS):
                    x = self.inp if i == 0 else self.hiddens[i - 1]
                    self.hiddens.append(tf.nn.relu(tf.matmul(x, weights[i]) + biases[i]))

                self.predicts = tf.nn.softmax(tf.matmul(self.hiddens[-1], weights[-1]) + biases[-1])

                regularizers = tf.add_n([tf.nn.l2_loss(x) for x in weights + biases])

                self.loss = (-tf.reduce_sum(self.labels * tf.log(tf.maximum(self.predicts, 1E-10))) +
                             L2_REGULARIZER * regularizers)

                self.correct_prediction = tf.equal(tf.argmax(self.predicts, 1), tf.argmax(self.labels, 1))
                self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

                self.grad_op = optimizer.compute_gradients(self.loss)

        self.my_device = worker_device
        self.my_name_scope = worker_name_scope


with tf.Graph().as_default():
    batch_idx = tf.Variable(0, trainable=False)

    # learning_rate = tf.train.exponential_decay(0.01, batch_idx * BATCH_SIZE,
    #                                            decay_steps=1000, decay_rate=0.95, staircase=True)
    # opt = tf.train.MomentumOptimizer(learning_rate, 0.9)
    opt = tf.train.GradientDescentOptimizer(0.01)

    models = []

    for i, w in enumerate(WORKER_TASKS):
        models.append(Model(variable_device=PS_DEVICE,
                            variable_scope='mnist_variables',
                            worker_device=w, worker_name_scope='worker_name_scope_{}'.format(i),
                            reuse=i != 0,
                            optimizer=opt))

    # compute gradients of workers
    all_grads_and_vars = []
    all_losses = []
    for m in models:
        all_grads_and_vars.append(m.grad_op)
        all_losses.append(m.loss)

    # compute average gradient
    if len(all_grads_and_vars) > 1:
        average_grads = []
        for grads_and_vars in zip(*all_grads_and_vars):
            grads = []
            for g, _ in grads_and_vars:
                grads.append(tf.expand_dims(g, 0))

            grad = tf.reduce_mean(tf.concat(0, grads), 0)
            average_grads.append((grad, grads_and_vars[0][1]))
    else:
        average_grads = all_grads_and_vars[0]

    average_loss = tf.add_n(all_losses) / len(all_losses)

    # apply the average gradient
    apply_gradient_op = opt.apply_gradients(average_grads, global_step=batch_idx)

    init_op = tf.initialize_all_variables()

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
    sess_config = tf.ConfigProto(log_device_placement=True, gpu_options=gpu_options)

    with tf.Session(GRPC_SERVER, config=sess_config) as sess:
        sess.run(init_op)

        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

        training_time = time.time()
        for i in range(ITERATIONS):
            feeds = {}
            for m in models:
                # data-parallel
                batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
                feeds[m.inp] = batch_xs
                feeds[m.labels] = batch_ys

            _, avg_loss = sess.run([apply_gradient_op, average_loss], feed_dict=feeds)

            if i % EVAL_EVERY == 0:
                # we only need to evaluate on a single model, because there is only one set of parameters
                m = models[0]
                acc = sess.run(m.accuracy, feed_dict={m.inp: mnist.validation.images,
                                                      m.labels: mnist.validation.labels})
                print 'Step {}: validation accuracy = {}, average training loss = {}'.format(
                    tf.train.global_step(sess, batch_idx), acc, avg_loss)

        training_time = time.time() - training_time
        print 'Training time: {} seconds'.format(training_time)

        m = models[0]
        acc = sess.run(m.accuracy, feed_dict={m.inp: mnist.test.images, m.labels: mnist.test.labels})
        print 'Test accuracy: {}'.format(acc)
