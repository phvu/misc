import tensorflow as tf
import numpy as np

c = []
for d in ['/job:ps/task:0', '/job:worker/task:0']:
  with tf.device(d):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3])
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2])
    c.append(tf.matmul(a, b))

with tf.device('/job:ps/task:0'):
  sum = tf.add_n(c)


with tf.Session("grpc://localhost:2222", config=tf.ConfigProto(log_device_placement=True)) as sess:
  r = sess.run(sum)
  assert np.sum(np.asarray([[44., 56.], [98., 128.]]) - r) < 1E-8
  print r
