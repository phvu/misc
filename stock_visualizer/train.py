# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Binary for training translation models and decoding from them.

Running this program without --decode will download the WMT corpus into
the directory specified as --data_dir and tokenize it in a very basic way,
and then start training a model saving checkpoints to --train_dir.

Running with --decode starts an interactive loop so you can see how
the current checkpoint translates English sentences into French.

See the following papers for more information on neural translation models.
 * http://arxiv.org/abs/1409.3215
 * http://arxiv.org/abs/1409.0473
 * http://arxiv.org/abs/1412.2007
"""
from __future__ import division
from __future__ import print_function

import os
import random
import sys
import time

import numpy as np
import tensorflow as tf

import stock_model
import utils

tf.app.flags.DEFINE_integer("data_dims", 28, "Dimension of input data")
tf.app.flags.DEFINE_integer("seq_len", 40, "Length of each series")
tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.99,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 64,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 1024, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model.")
tf.app.flags.DEFINE_string("data_dir", "./data", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "./save", "Training directory.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 10,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("decode", False,
                            "Set to True for interactive decoding.")

FLAGS = tf.app.flags.FLAGS

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
_buckets = [(FLAGS.seq_len + 5, FLAGS.seq_len + 10)]


def create_model(session, forward_only):
    """Create translation model and initialize or load parameters in session."""
    model = stock_model.Model(FLAGS.data_dims, _buckets,
                              FLAGS.size, FLAGS.num_layers, FLAGS.max_gradient_norm, FLAGS.batch_size,
                              FLAGS.learning_rate, FLAGS.learning_rate_decay_factor, forward_only=forward_only)

    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.initialize_all_variables())
    return model


def train():
    with tf.Session() as sess:
        # Create model.
        print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
        model = create_model(sess, False)

        # Read data into buckets and compute their sizes.
        print("Reading development and training data (limit: %d)."
              % FLAGS.max_train_data_size)

        train_set, dev_set, _ = utils.read_data(_buckets, data_dir=FLAGS.data_dir, sequence_len=FLAGS.seq_len)

        train_bucket_sizes = [len(train_set[b]) for b in range(len(_buckets))]
        train_total_size = float(sum(train_bucket_sizes))

        # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
        # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
        # the size if i-th training bucket, as used later.
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                               for i in range(len(train_bucket_sizes))]

        # This is the training loop.
        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []
        while True:
            # Choose a bucket according to data distribution. We pick a random number
            # in [0, 1] and use the corresponding interval in train_buckets_scale.
            random_number_01 = np.random.random_sample()
            bucket_id = min([i for i in range(len(train_buckets_scale))
                             if train_buckets_scale[i] > random_number_01])

            # Get a batch and make a step.
            start_time = time.time()
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                train_set, bucket_id)
            _, step_loss, _, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                            target_weights, bucket_id, False)
            step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
            loss += step_loss / FLAGS.steps_per_checkpoint
            current_step += 1

            # Once in a while, we save checkpoint, print statistics, and run evals.
            if current_step % FLAGS.steps_per_checkpoint == 0:
                # Print statistics for the previous epoch.
                # perplexity = math.exp(loss) if loss < 300 else float('inf')
                print("global step %d learning rate %.4f step-time %.2f loss "
                      "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                                step_time, loss))
                # Decrease learning rate if no improvement was seen over last 3 times.
                if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
                    sess.run(model.learning_rate_decay_op)
                previous_losses.append(loss)
                # Save checkpoint and zero timer and loss.
                checkpoint_path = os.path.join(FLAGS.train_dir, "translate.ckpt")
                model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                step_time, loss = 0.0, 0.0
                # Run evals on development set and print their perplexity.
                for bucket_id in range(len(_buckets)):
                    if len(dev_set[bucket_id]) == 0:
                        print("  eval: empty bucket %d" % (bucket_id))
                        continue
                    encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                        dev_set, bucket_id)
                    _, eval_loss, _, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                                    target_weights, bucket_id, True)
                    # eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
                    print("  eval: bucket %d loss %.2f" % (bucket_id, eval_loss))
                sys.stdout.flush()


def decode():
    print('Decoding')
    with tf.Session() as sess:
        # Create model.
        print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
        model = create_model(sess, False)

        # Read data into buckets and compute their sizes.
        print("Reading development and training data (limit: %d)."
              % FLAGS.max_train_data_size)

        _, _, all_data_labels = utils.read_data(_buckets, data_dir=FLAGS.data_dir, sequence_len=FLAGS.seq_len)

        all_data, all_labels = all_data_labels
        all_states = []
        for bucket_id in range(len(_buckets)):
            if len(all_data[bucket_id]) == 0:
                print("  eval: empty bucket %d" % bucket_id)
                continue
            idx = 0
            total_eval_loss = 0
            i = 0
            total_len = len(all_data[bucket_id])
            while idx < total_len:
                encoder_inputs, decoder_inputs, target_weights, idx = model.get_batch(all_data, bucket_id, idx)
                _, eval_loss, _, states = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True)
                total_eval_loss += eval_loss
                i += 1
                all_states.append(states)
                print('\rBucket {}, item {}/{}'.format(bucket_id, idx, total_len), end='\r')
                sys.stdout.flush()

            print("  eval: bucket %d loss %.2f" % (bucket_id, total_eval_loss / i))
        sys.stdout.flush()
        np.savez(os.path.join(FLAGS.train_dir, 'decoded_states.npz'),
                 states=np.vstack(all_states), labels=all_labels)


def main(_):
    if FLAGS.decode:
        decode()
    else:
        train()


if __name__ == "__main__":
    tf.app.run()
