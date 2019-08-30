#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date   : 2019-08-20
# @Author : KangYu
# @File   : model_for_resave.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf

import utils


flags = tf.app.flags
FLAGS = flags.FLAGS


flags.DEFINE_string('fixed_arc', "0 1 1 2 2 2 3 2 4 2 5 2 6 2", '')
flags.DEFINE_integer('child_hidden_size', 200, '')


def _bidirectional_rnn(O, seq_len,
                       w_prev, w_skip,
                       w_prev_bw, w_skip_bw,
                       input_mask, layer_mask, config):

    _, all_s = _rnn_fn(O, w_prev, w_skip,
                                  input_mask, layer_mask, params=config)

    def _reverse(input_, seq_lengths, seq_axis, batch_axis):
      if seq_lengths is not None:
        return tf.reverse_sequence(
            input=input_,
            seq_lengths=seq_lengths,
            seq_axis=seq_axis,
            batch_axis=batch_axis)
      else:
        return tf.reverse(input_, axis=[seq_axis])

    O_reverse = _reverse(O, seq_len, seq_axis=1, batch_axis=0)

    _, all_s_bw = _rnn_fn(O_reverse, w_prev_bw, w_skip_bw,
                                  input_mask, layer_mask, params=config)

    all_s_bw = _reverse(all_s_bw, seq_len, seq_axis=1, batch_axis=0)

    all_s = tf.concat([all_s, all_s_bw], axis=-1)

    return all_s


def _rnn_fn(x, w_prev, w_skip, input_mask, layer_mask, params):
  """Multi-layer LSTM.

  Args:
    x: [batch_size, num_steps, hidden_size].
    prev_s: [batch_size, hidden_size].
    w_prev: [2 * hidden_size, 2 * hidden_size].
    w_skip: [None, [hidden_size, 2 * hidden_size] * (num_layers-1)].
    input_mask: [batch_size, hidden_size].
    layer_mask: [batch_size, hidden_size].
    params: hyper-params object.

  Returns:
    next_s: [batch_size, hidden_size].
    all_s: [[batch_size, num_steps, hidden_size] * num_layers].
  """

  init_shape = [params.batch_size, params.hidden_size]
  prev_s = tf.zeros(init_shape, dtype=np.float32, name="s")
  num_steps = tf.shape(x)[1]
  fixed_arc = params.fixed_arc
  num_layers = len(fixed_arc) // 2

  all_s = tf.TensorArray(dtype=tf.float32, size=num_steps, infer_shape=False)

  def _condition(step, *unused_args):
    return tf.less(step, num_steps)

  def _body(step, prev_s, all_s):
    """Body fn for `tf.while_loop`."""
    inp = x[:, step, :]
    if layer_mask is not None:
      assert input_mask is not None
      ht = tf.matmul(
          tf.concat([inp * input_mask, prev_s * layer_mask], axis=1), w_prev)
    else:
      ht = tf.matmul(tf.concat([inp, prev_s], axis=1), w_prev)
    h, t = tf.split(ht, 2, axis=1)
    h = tf.tanh(h)
    t = tf.sigmoid(t)
    s = prev_s + t * (h - prev_s)
    layers = [s]

    def _select_function(h, function_id):
      if function_id == 0:
        return tf.tanh(h)
      elif function_id == 1:
        return tf.nn.relu(h)
      elif function_id == 2:
        return tf.sigmoid(h)
      elif function_id == 3:
        return h
      raise ValueError('Unknown func_idx {0}'.format(function_id))

    start_idx = 0
    for layer_id in range(num_layers):
      prev_idx = fixed_arc[start_idx]
      func_idx = fixed_arc[start_idx + 1]
      prev_s = layers[prev_idx]
      if layer_mask is not None:
        ht = tf.matmul(prev_s * layer_mask, w_skip[layer_id])
      else:
        ht = tf.matmul(prev_s, w_skip[layer_id])
      h, t = tf.split(ht, 2, axis=1)

      h = _select_function(h, func_idx)
      t = tf.sigmoid(t)
      s = prev_s + t * (h - prev_s)
      s.set_shape([params.batch_size, params.hidden_size])
      layers.append(s)
      start_idx += 2

    next_s = tf.add_n(layers[1:]) / tf.cast(num_layers, dtype=tf.float32)
    all_s = all_s.write(step, next_s)
    return step + 1, next_s, all_s

  loop_inps = [tf.constant(0, dtype=tf.int32), prev_s, all_s]
  _, next_s, all_s = tf.while_loop(_condition, _body, loop_inps)
  all_s = tf.transpose(all_s.stack(), [1, 0, 2])

  return next_s, all_s



def _set_default_params(params):
  """Set default hyper-parameters."""
  params.add_hparam('alpha', 0.7)  # activation L2 reg
  params.add_hparam('best_valid_ppl_threshold', 5)

  params.add_hparam('batch_size', 64)

  # for dropouts: dropping rate, NOT keeping rate
  params.add_hparam('drop_rate', 0.20)  # voice attention
  params.add_hparam('drop_e', 0.125)  # word
  params.add_hparam('drop_i', 0.175)  # embeddings
  params.add_hparam('drop_x', 0.725)  # input to RNN cells
  params.add_hparam('drop_l', 0.225)  # between layers
  params.add_hparam('drop_o', 0.75)  # output
  params.add_hparam('drop_w', 0.00)  # weight

  assert FLAGS.fixed_arc is not None
  params.add_hparam('fixed_arc', [int(d) for d in FLAGS.fixed_arc.split(' ')])

  params.add_hparam('grad_bound', 0.25)
  params.add_hparam('hidden_size', FLAGS.child_hidden_size)
  params.add_hparam('init_range', 0.05)
  params.add_hparam('learning_rate', 1)
  params.add_hparam('num_train_epochs', 600)

  params.add_hparam('weight_decay', 8e-7)

  params.add_hparam('max_sent_num', 200)
  params.add_hparam('max_sent_len', 100)
  params.add_hparam('w2v_dim', 200)
  params.add_hparam('vocab_sizes', 314041)
  params.add_hparam('w2v_istrain', True)

  # params.add_hparam('train_root', '/workspace/speaker_verification/data/dahai/train/va_widxdahai_tfrecord_200_100/')
  # params.add_hparam('test_dahai_root', '/workspace/speaker_verification/data/dahai/test/va_widxdahai_tfrecord_200_200')
  # params.add_hparam('test_zhikang_root', '/workspace/speaker_verification/data/zhikang/test/va_widxdahai_tfrecord_200_200')
  # params.add_hparam('pretrain_embedding_path', '/workspace/speaker_verification/data/w2v_online/0718_dahaipretrain/w2v_mtrx.npy')

  params.add_hparam('train_root', '/share/kangyu/speaker/dahai/train/va_widxdahai_tfrecord_200_100')
  params.add_hparam('test_dahai_root', '/share/kangyu/speaker/dahai/test/va_widxdahai_tfrecord_200_200')
  params.add_hparam('test_zhikang_root',
                    '/share/kangyu/speaker/zhikang/test/va_widxdahai_tfrecord_200_200')
  params.add_hparam('pretrain_embedding_path',
                    '/share/kangyu/speaker/w2v_mtrx.npy')

  return params



class sentRNN:

    def __init__(self, config, name="fixed_model"):

        self.config = _set_default_params(config)
        self.name = name

        self._build_params()
        self._build_valid()

    def _build_params(self):
        """Create model parameters."""

        print('-' * 80)
        print('Building model params')

        hidden_size = self.config.hidden_size
        fixed_arc = self.config.fixed_arc
        num_layers = len(fixed_arc) // 2

        with tf.variable_scope(self.name):

            self.is_training = tf.placeholder_with_default(False, shape=[], name="is_training")
            self.voice_embed = tf.placeholder(shape=[None, None, 64],
                                         dtype=tf.float32, name="voice_embed")  # shape: (batch, max_sent_num, 64)

            # Build the netword for the words
            self.word_idx = tf.placeholder(
                shape=[None, None, None],
                dtype=tf.int32, name="word_idx")  # shape: (batch, max_sent_num, max_sent_len)
            self.sent_len = tf.placeholder(shape=[None, None],
                                      dtype=tf.int32, name="sent_len")  # shape: (batch, max_sent_num)
            w2v_embedding = tf.Variable(tf.constant(0.0, shape=[self.config.vocab_sizes, self.config.w2v_dim]),
                                        trainable=self.config.w2v_istrain, name="w2i_embedding")
            self.embedding_placeholder = tf.placeholder(tf.float32, [self.config.vocab_sizes, self.config.w2v_dim], name="embedding_placeholder")
            self.embedding_init = w2v_embedding.assign(self.embedding_placeholder)

            word_embed = tf.nn.embedding_lookup(w2v_embedding,
                                                self.word_idx)  # shape: (batch, max_sent_num, max_sent_len, 200)
            # do the mask based on the max_sent_len
            sent_len_mask = tf.sequence_mask(self.sent_len,
                                             maxlen=self.config.max_sent_len)  # (batch, max_sent_num, max_sent_len)
            sent_len_mask = tf.cast(sent_len_mask, tf.float32)
            sent_len_mask = tf.expand_dims(sent_len_mask, -1)  # (batch, max_sent_num, max_sent_len, 1)
            # Here is for the mean pooling
            word_embed = word_embed * sent_len_mask
            word_embed = tf.reduce_sum(word_embed, axis=2)  # (batch, max_sent_num, 200)
            sent_len_temp = self.sent_len + tf.cast(tf.equal(self.sent_len, 0), tf.int32)
            sent_len_temp = tf.cast(tf.expand_dims(sent_len_temp, -1), tf.float32)  # (batch, max_sent_num, 1)
            word_embed = word_embed / sent_len_temp  # (batch, max_sent_num, 200)
            word_embed = tf.layers.dropout(word_embed, rate=self.config.drop_rate, training=self.is_training)

            self.label = tf.placeholder(shape=[None, None],
                                        dtype=tf.int32, name="label")  # shape: (batch, max_sent_num)
            self.seq_len = tf.placeholder(shape=[None], dtype=tf.int32, name="seq_len")  # shape: (batch)

            with tf.variable_scope("Voice_Attention"):
                voice_Q = tf.layers.dense(inputs=self.voice_embed, units=32, activation=None, use_bias=False,
                                          kernel_initializer=tf.truncated_normal_initializer(), name='attention_w')
                voice_K = tf.layers.dense(inputs=self.voice_embed, units=32, activation=None, use_bias=False,
                                          name='attention_w', reuse=True)
                word_V = word_embed  # shape: (batch, max_length, 200)
                voice_A = tf.matmul(voice_Q, voice_K, transpose_b=True)  # shape: (batch, max_length, max_length)
                voice_A = tf.transpose(voice_A, [0, 2, 1])  # transpose to get the correct format
                voice_A = self.mask(voice_A, self.seq_len, mode='add', max_len=self.config.max_sent_num)  # shape: (batch, max_length, max_length)
                voice_A = tf.transpose(voice_A, [0, 2, 1])  # transpose to get the correct format
                self.voice_A = tf.nn.softmax(voice_A, name="voice_A")  # shape: (batch, max_length, max_length)
                voice_O = tf.matmul(self.voice_A, word_V)  # shape: (batch, max_length, 200)
                voice_O = self.mask(voice_O, self.seq_len, mode='mul', max_len=self.config.max_sent_num)  # shape: (batch, max_sent_num, 200)

                O = tf.concat([voice_O, word_V], axis=-1)
                self.O = tf.layers.dense(inputs=O, units=self.config.hidden_size, activation=None, use_bias=False,
                                          kernel_initializer=tf.truncated_normal_initializer(), name='dense_O')

            with tf.variable_scope("enas_rnn_fw"):
                with tf.variable_scope('rnn_cell'):
                    w_prev = tf.get_variable('w_prev', [2 * hidden_size, 2 * hidden_size])

                    w_skip = []
                    for layer_id in range(num_layers):
                        with tf.variable_scope('layer_{}'.format(layer_id)):
                            w = tf.get_variable('w', [hidden_size, 2 * hidden_size])
                            w_skip.append(w)

            with tf.variable_scope("enas_rnn_bw"):
                with tf.variable_scope('rnn_cell'):
                    w_prev_bw = tf.get_variable('w_prev', [2 * hidden_size, 2 * hidden_size])

                    w_skip_bw = []
                    for layer_id in range(num_layers):
                        with tf.variable_scope('layer_{}'.format(layer_id)):
                            w_bw = tf.get_variable('w', [hidden_size, 2 * hidden_size])
                            w_skip_bw.append(w_bw)

            with tf.variable_scope('output'):
                w_out = tf.get_variable('w_out', [2*hidden_size, 2])

        self.num_params = sum([np.prod(v.shape) for v in tf.trainable_variables()
                               if v.name.startswith(self.name)]).value
        print('All children have {0} params'.format(self.num_params))

        num_params_per_child = 0
        for v in tf.trainable_variables():
            if v.name.startswith(self.name):
                if 'rnn_cell' in v.name:
                    num_params_per_child += v.shape[-2].value * v.shape[-1].value
                else:
                    num_params_per_child += np.prod([d.value for d in v.shape])
        print('Each child has {0} params'.format(num_params_per_child))

        self.eval_params = {
            'w_prev': w_prev,
            'w_skip': w_skip,  # [num_functions, layer_id, hidden_size, 2 * hidden_size] * num_layers
            'w_prev_bw': w_prev_bw,
            'w_skip_bw': w_skip_bw,
            'w_out': w_out,
        }


    def _forward(self, model_params):
        """Computes the logits.

        Args:
          x: [batch_size, num_steps], input batch.
          y: [batch_size, num_steps], output batch.
          model_params: a `dict` of params to use.
          init_states: a `dict` of params to use.
          is_training: if `True`, will apply regularizations.

        Returns:
          loss: scalar, cross-entropy loss
        """
        w_prev = model_params['w_prev']
        w_skip = model_params['w_skip']
        w_prev_bw = model_params['w_prev_bw']
        w_skip_bw = model_params['w_skip_bw']
        w_out = model_params['w_out']

        emb = self.O

        input_mask = None
        layer_mask = None

        all_s = _bidirectional_rnn(
                                   O=emb,
                                   seq_len=self.seq_len,
                                   w_prev=w_prev,
                                   w_skip=w_skip,
                                   w_prev_bw=w_prev_bw,
                                   w_skip_bw=w_skip_bw,
                                   input_mask=input_mask,
                                   layer_mask=layer_mask,
                                   config=self.config,
                                  )

        top_s = all_s  # [batch_size, max_sent_num, 2*hidden_size]

        logits = tf.einsum('bnh,hc->bnc', top_s, w_out)  # [batch_size, num_steps, vocabe_size]

        pred_label = tf.cast(tf.argmax(logits, -1), tf.int32)
        self.pred_label = tf.identity(pred_label, name="pred_label")

        return self.pred_label   # loss是纯损失，reg loss里有l2正则损失。


    def _build_valid(self):
        print('Building valid graph')
        _ = self._forward(self.eval_params)


    def parse_helper(self, example_proto):
        '''
        :param example_proto:
        :return:
        '''
        dics = {'voice_embed': tf.FixedLenFeature(shape=(), dtype=tf.string),
                'voice_shape': tf.FixedLenFeature(shape=(2,), dtype=tf.int64),
                'sent_word_idx': tf.FixedLenFeature(shape=(), dtype=tf.string),
                'sent_word_shape': tf.FixedLenFeature(shape=(2,), dtype=tf.int64),
                'sent_word_num': tf.VarLenFeature(dtype=tf.int64),
                'sent_label': tf.VarLenFeature(dtype=tf.int64),
                'length': tf.FixedLenFeature(shape=(), dtype=tf.int64)}
        parsed_example = tf.parse_single_example(example_proto, dics)
        sent_word_num = tf.sparse_tensor_to_dense(parsed_example['sent_word_num'])
        sent_label = tf.sparse_tensor_to_dense(parsed_example['sent_label'])
        sent_word_num = tf.cast(sent_word_num, tf.int32)
        sent_label = tf.cast(sent_label, tf.int32)
        voice_embed = tf.decode_raw(parsed_example['voice_embed'], tf.float32)
        voice_embed = tf.reshape(voice_embed, parsed_example['voice_shape'])
        sent_word_idx = tf.decode_raw(parsed_example['sent_word_idx'], tf.int32)
        sent_word_idx = tf.reshape(sent_word_idx, parsed_example['sent_word_shape'])
        length = tf.cast(parsed_example['length'], tf.int32)
        return voice_embed, sent_word_idx, sent_word_num, sent_label, length

    def mask(self, inputs, seq_len, mode='mul', max_len=None):
        if seq_len == None:
            return inputs
        else:
            mask = tf.cast(tf.sequence_mask(seq_len, max_len), tf.float32)
            for _ in range(len(inputs.shape) - 2):
                mask = tf.expand_dims(mask, 2)
            if mode == 'mul':
                return inputs * mask
            if mode == 'add':
                return inputs - (1 - mask) * 1e12

