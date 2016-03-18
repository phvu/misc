import math
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


HIDDEN_SIZE = 512
LAYERS = 2

inp = tf.placeholder(tf.float32, shape=[None, 784])
labels = tf.placeholder(tf.float32, shape=[None, 10])

weights = []
biases = []

with tf.variable_scope("mnist_vars"):
    for i in range(0, LAYERS + 1):
        ws = [784 if i == 0 else HIDDEN_SIZE, 10 if i == LAYERS else HIDDEN_SIZE]
        weights.append(tf.get_variable("weight{}".format(i), ws,
                                       initializer=tf.truncated_normal_initializer(
                                           stddev=math.sqrt(3.0 / (ws[0] + ws[1])))))
        biases.append(tf.get_variable("bias{}".format(i), ws[1:],
                                      initializer=tf.constant_initializer(value=0, dtype=tf.float32)))
hiddens = []
for i in range(0, LAYERS):
    hiddens.append(tf.nn.relu(tf.matmul(inp if i == 0 else hiddens[i - 1], weights[i]) + biases[i]))
predicts = tf.nn.softmax(tf.matmul(hiddens[-1], weights[-1]) + biases[-1])

loss = -tf.reduce_sum(labels * tf.log(tf.maximum(predicts, 1E-8)))

correct_prediction = tf.equal(tf.argmax(predicts, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

optimizer = tf.train.GradientDescentOptimizer(0.01)
# optimizer = tf.train.MomentumOptimizer(0.001, momentum=0.5)
# optimizer = tf.train.AdamOptimizer()

# grad_op = optimizer.compute_gradients(loss)

# apply_gradient_op = optimizer.apply_gradients(grad_op)
apply_gradient_op = optimizer.minimize(loss)

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    sess.run(tf.initialize_all_variables())

    print '================== TRAINABLE VARIABLES ================'
    for v in tf.all_variables():
        print v.device, v.name, v.get_shape()

    for i in range(1000):
        feeds = {}
        batch_xs, batch_ys = mnist.train.next_batch(64)
        feeds[inp] = batch_xs
        feeds[labels] = batch_ys

        _, avg_loss = sess.run([apply_gradient_op, loss], feed_dict=feeds)

        if i % 10 == 0:
            acc = sess.run(accuracy, feed_dict={inp: mnist.test.images, labels: mnist.test.labels})
            print 'Step {}: loss = {}, accuracy = {}'.format(i, avg_loss, acc)