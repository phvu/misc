from __future__ import division
from __future__ import print_function

import random

import numpy as np
import tensorflow as tf

import utils


def my_basic_rnn_seq2seq(encoder_inputs, decoder_inputs, cell, dtype=tf.float32, scope=None):
  """Modified version of basic_rnn_seq2seq to get the encoder state
  """
  with tf.variable_scope(scope or "my_basic_rnn_seq2seq"):
    _, enc_state = tf.nn.rnn(cell, encoder_inputs, dtype=dtype)
    return tf.nn.seq2seq.rnn_decoder(decoder_inputs, enc_state, cell) + (enc_state, )


def my_model_with_buckets(encoder_inputs, decoder_inputs, targets, weights,
                          buckets, seq2seq, softmax_loss_function=None,
                          per_example_loss=False, name=None):
    """Improved version of model_with_buckets, to take the states
    """
    if len(encoder_inputs) < buckets[-1][0]:
        raise ValueError("Length of encoder_inputs (%d) must be at least that of la"
                         "st bucket (%d)." % (len(encoder_inputs), buckets[-1][0]))
    if len(targets) < buckets[-1][1]:
        raise ValueError("Length of targets (%d) must be at least that of last"
                         "bucket (%d)." % (len(targets), buckets[-1][1]))
    if len(weights) < buckets[-1][1]:
        raise ValueError("Length of weights (%d) must be at least that of last"
                         "bucket (%d)." % (len(weights), buckets[-1][1]))

    all_inputs = encoder_inputs + decoder_inputs + targets + weights
    losses = []
    outputs = []
    states = []
    with tf.op_scope(all_inputs, name, "my_model_with_buckets"):
        for j, bucket in enumerate(buckets):
            with tf.variable_scope(tf.get_variable_scope(), reuse=True if j > 0 else None):
                bucket_outputs, _, bucket_enc_state = seq2seq(encoder_inputs[:bucket[0]], decoder_inputs[:bucket[1]])
                outputs.append(bucket_outputs)
                states.append(bucket_enc_state)
                if per_example_loss:
                    losses.append(tf.nn.seq2seq.sequence_loss_by_example(
                        outputs[-1], targets[:bucket[1]], weights[:bucket[1]],
                        softmax_loss_function=softmax_loss_function))
                else:
                    losses.append(tf.nn.seq2seq.sequence_loss(
                        outputs[-1], targets[:bucket[1]], weights[:bucket[1]],
                        softmax_loss_function=softmax_loss_function))

    return outputs, losses, states


class Model(object):
    def __init__(self, data_dim, buckets, size,
                 num_layers, max_gradient_norm, batch_size, learning_rate,
                 learning_rate_decay_factor, use_lstm=False, forward_only=False):
        """Create the model.

        Args:
          data_dim: size of input and output vectors
          buckets: a list of pairs (I, O), where I specifies maximum input length
            that will be processed in that bucket, and O specifies maximum output
            length. Training instances that have inputs longer than I or outputs
            longer than O will be pushed to the next bucket and padded accordingly.
            We assume that the list is sorted, e.g., [(2, 4), (8, 16)].
          size: number of units in each layer of the model.
          num_layers: number of layers in the model.
          max_gradient_norm: gradients will be clipped to maximally this norm.
          batch_size: the size of the batches used during training;
            the model construction is independent of batch_size, so it can be
            changed after initialization if this is convenient, e.g., for decoding.
          learning_rate: learning rate to start with.
          learning_rate_decay_factor: decay learning rate by this much when needed.
          use_lstm: if true, we use LSTM cells instead of GRU cells.
          forward_only: if set, we do not construct the backward pass in the model.
        """
        self.data_dim = data_dim
        self.buckets = buckets
        self.batch_size = batch_size
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(
            self.learning_rate * learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)

        # Linear mapping for the output
        output_projection = None

        with tf.device("/cpu:0"):
            w = tf.get_variable("proj_w", [size, self.data_dim])
            b = tf.get_variable("proj_b", [self.data_dim])
        output_projection = (w, b)

        def sampled_loss(inputs, labels):
            with tf.device("/cpu:0"):
                linear_map = tf.matmul(inputs, output_projection[0]) + output_projection[1]
                return tf.nn.l2_loss(linear_map - labels)

        l2_loss_function = sampled_loss

        # Create the internal multi-layer cell for our RNN.
        single_cell = tf.nn.rnn_cell.GRUCell(size, input_size=self.data_dim)
        if use_lstm:
            single_cell = tf.nn.rnn_cell.BasicLSTMCell(size, input_size=self.data_dim)
        cell = single_cell
        if num_layers > 1:
            cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * num_layers)

        # The seq2seq function: we use embedding for the input and attention.
        # advances tasks should use embedding_attention_seq2seq
        # here we go with basic_rnn_seq2seq
        def seq2seq_f(encoder_inputs, decoder_inputs):
            return my_basic_rnn_seq2seq(encoder_inputs, decoder_inputs, cell)

        # Feeds for inputs.
        self.encoder_inputs = []
        self.decoder_inputs = []
        self.target_weights = []
        for i in range(buckets[-1][0]):  # Last bucket is the biggest one.
            self.encoder_inputs.append(tf.placeholder(tf.float32, shape=[None, self.data_dim],
                                                      name="encoder{0}".format(i)))
        for i in range(buckets[-1][1] + 1):
            self.decoder_inputs.append(tf.placeholder(tf.float32, shape=[None, self.data_dim],
                                                      name="decoder{0}".format(i)))
            self.target_weights.append(tf.placeholder(tf.float32, shape=[None],
                                                      name="weight{0}".format(i)))

        # Our targets are decoder inputs shifted by one.
        targets = [self.decoder_inputs[i + 1]
                   for i in range(len(self.decoder_inputs) - 1)]

        # Training outputs and losses.
        if forward_only:
            self.outputs, self.losses, self.states = my_model_with_buckets(
                self.encoder_inputs, self.decoder_inputs, targets,
                self.target_weights, buckets, seq2seq_f,
                softmax_loss_function=l2_loss_function)

            # If we use output projection, we need to project outputs for decoding.
            if output_projection is not None:
                for b in range(len(buckets)):
                    self.outputs[b] = [
                        tf.matmul(output, output_projection[0]) + output_projection[1]
                        for output in self.outputs[b]
                        ]
        else:
            self.outputs, self.losses, self.states = my_model_with_buckets(
                self.encoder_inputs, self.decoder_inputs, targets,
                self.target_weights, buckets, seq2seq_f,
                softmax_loss_function=l2_loss_function)

        # Gradients and SGD update operation for training the model.
        params = tf.trainable_variables()
        if not forward_only:
            self.gradient_norms = []
            self.updates = []
            opt = tf.train.GradientDescentOptimizer(self.learning_rate)
            for b in range(len(buckets)):
                gradients = tf.gradients(self.losses[b], params)
                clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                                 max_gradient_norm)
                self.gradient_norms.append(norm)
                self.updates.append(opt.apply_gradients(
                    zip(clipped_gradients, params), global_step=self.global_step))

        self.saver = tf.train.Saver(tf.all_variables())

    def step(self, session, encoder_inputs, decoder_inputs, target_weights,
             bucket_id, forward_only):
        """Run a step of the model feeding the given inputs.

        Args:
          session: tensorflow session to use.
          encoder_inputs: list of numpy int vectors to feed as encoder inputs.
          decoder_inputs: list of numpy int vectors to feed as decoder inputs.
          target_weights: list of numpy float vectors to feed as target weights.
          bucket_id: which bucket of the model to use.
          forward_only: whether to do the backward step or only forward.

        Returns:
          A triple consisting of gradient norm (or None if we did not do backward),
          average perplexity, and the outputs.

        Raises:
          ValueError: if length of encoder_inputs, decoder_inputs, or
            target_weights disagrees with bucket size for the specified bucket_id.
        """
        # Check if the sizes match.
        encoder_size, decoder_size = self.buckets[bucket_id]
        if len(encoder_inputs) != encoder_size:
            raise ValueError("Encoder length must be equal to the one in bucket,"
                             " %d != %d." % (len(encoder_inputs), encoder_size))
        if len(decoder_inputs) != decoder_size:
            raise ValueError("Decoder length must be equal to the one in bucket,"
                             " %d != %d." % (len(decoder_inputs), decoder_size))
        if len(target_weights) != decoder_size:
            raise ValueError("Weights length must be equal to the one in bucket,"
                             " %d != %d." % (len(target_weights), decoder_size))

        # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
        input_feed = {}
        for l in range(encoder_size):
            input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
        for l in range(decoder_size):
            input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
            input_feed[self.target_weights[l].name] = target_weights[l]

        # Since our targets are decoder inputs shifted by one, we need one more.
        last_target = self.decoder_inputs[decoder_size].name
        input_feed[last_target] = np.zeros([self.batch_size, encoder_inputs[0].shape[1]], dtype=np.float32)

        # Output feed: depends on whether we do a backward step or not.
        if not forward_only:
            output_feed = [self.updates[bucket_id],  # Update Op that does SGD.
                           self.gradient_norms[bucket_id],  # Gradient norm.
                           self.losses[bucket_id]]  # Loss for this batch.
        else:
            output_feed = [self.losses[bucket_id], self.states[bucket_id]]  # Loss for this batch.
            for l in range(decoder_size):  # Output logits.
                output_feed.append(self.outputs[bucket_id][l])

        outputs = session.run(output_feed, input_feed)
        if not forward_only:
            return outputs[1], outputs[2], None, None  # Gradient norm, loss, no outputs, no states.
        else:
            return None, outputs[0], outputs[2:], outputs[1]  # No gradient norm, loss, outputs, states.

    def get_batch(self, data, bucket_id, idx=None):
        """Get a random batch of data from the specified bucket, prepare for step.

        To feed data in step(..) it must be a list of batch-major vectors, while
        data here contains single length-major cases. So the main logic of this
        function is to re-index data cases to be in the proper format for feeding.

        Args:
          data: a tuple of size len(self.buckets) in which each element contains
            lists of pairs of input and output data that we use to create a batch.
          bucket_id: integer, which bucket to get the batch for.

        Returns:
          The triple (encoder_inputs, decoder_inputs, target_weights) for
          the constructed batch that has the proper format to call step(...) later.
        """
        encoder_size, decoder_size = self.buckets[bucket_id]
        encoder_inputs, decoder_inputs, weights = [], [], []

        # Get a random batch of encoder and decoder inputs from data,
        # pad them if needed, reverse encoder inputs and add GO to decoder.
        for _ in range(self.batch_size):

            if idx is None:
                encoder_input, decoder_input = random.choice(data[bucket_id])
            else:
                encoder_input, decoder_input = data[bucket_id][idx % len(data[bucket_id])]
                idx += 1

            # Encoder inputs are padded and then reversed.
            if encoder_size > encoder_input.shape[0]:
                encoder_padded = np.vstack((encoder_input,
                                            utils.get_special_symbols('pad', (encoder_size - encoder_input.shape[0],
                                                                              encoder_input.shape[1]))))
                encoder_inputs.append(encoder_padded[::-1, :])
            else:
                encoder_inputs.append(encoder_input[::-1, :])

            # Decoder inputs get an extra "GO" symbol, and are padded then.
            decoder_pad_size = decoder_size - decoder_input.shape[0] - 1
            decoder_padded = np.vstack((utils.get_special_symbols('go', (1, decoder_input.shape[1])), decoder_input))
            if decoder_pad_size > 0:
                decoder_padded = np.vstack((decoder_padded,
                                            utils.get_special_symbols('pad',
                                                                      (decoder_pad_size, decoder_input.shape[1]))))
            decoder_inputs.append(decoder_padded)

            weights.append([1] * (decoder_input.shape[0] + 1) + [0] * decoder_pad_size)

        # Now we create batch-major vectors from the data selected above.
        batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

        # Batch encoder inputs are just re-indexed encoder_inputs.
        for length_idx in range(encoder_size):
            batch_encoder_inputs.append(
                np.array([encoder_inputs[batch_idx][length_idx, :]
                          for batch_idx in range(self.batch_size)], dtype=np.float32))

        # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
        for length_idx in range(decoder_size):
            batch_decoder_inputs.append(
                np.array([decoder_inputs[batch_idx][length_idx, :]
                          for batch_idx in range(self.batch_size)], dtype=np.float32))

            batch_weights.append(np.array([weights[batch_idx][length_idx] for batch_idx in range(self.batch_size)],
                                          dtype=np.float32))

        if idx is not None:
            return batch_encoder_inputs, batch_decoder_inputs, batch_weights, idx
        return batch_encoder_inputs, batch_decoder_inputs, batch_weights
