#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date   : 2019-08-14
# @Author : KangYu
# @File   : child_sentRNN.py

import os
import numpy as np
import tensorflow as tf
import utils


def _bidirectional_rnn(sample_arc, O, seq_len,
                       prev_s, w_prev, w_skip,
                       prev_s_bw, w_prev_bw, w_skip_bw,
                       input_mask, layer_mask, config):

    _, all_s, var_s = _rnn_fn(sample_arc, O, prev_s, w_prev, w_skip,
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

    _, all_s_bw, var_s_bw = _rnn_fn(sample_arc, O_reverse, prev_s_bw, w_prev_bw, w_skip_bw,
                                  input_mask, layer_mask, params=config)

    all_s_bw = _reverse(all_s_bw, seq_len, seq_axis=1, batch_axis=0)

    all_s = tf.concat([all_s, all_s_bw], axis=-1)

    var_s = var_s + var_s_bw

    return all_s, var_s


def _rnn_fn(sample_arc, x, prev_s, w_prev, w_skip, input_mask, layer_mask,
            params):
  """Multi-layer LSTM.

  Args:
    sample_arc: [num_layers * 2], sequence of tokens representing architecture.
    x: [batch_size, num_steps, hidden_size].
    prev_s: [batch_size, hidden_size].
    w_prev: [2 * hidden_size, 2 * hidden_size].
    w_skip: [None, [hidden_size, 2 * hidden_size] * (num_layers-1)].
    input_mask: `[batch_size, hidden_size]`.
    layer_mask: `[batch_size, hidden_size]`.
    params: hyper-params object.

  Returns:
    next_s: [batch_size, hidden_size].
    all_s: [[batch_size, num_steps, hidden_size] * num_layers].
  """
  batch_size = x.get_shape()[0].value
  num_steps = tf.shape(x)[1]
  num_layers = len(sample_arc) // 2

  all_s = tf.TensorArray(dtype=tf.float32, size=num_steps, infer_shape=False)

  # extract the relevant variables, so that you only do L2-reg on them.
  u_skip = []
  start_idx = 0
  for layer_id in range(num_layers):
    prev_idx = sample_arc[start_idx]
    func_idx = sample_arc[start_idx + 1]
    u_skip.append(w_skip[layer_id][func_idx, prev_idx])  # list of [hidden_size, 2 * hidden_size]
    start_idx += 2 # 每一次取一个连接的层号和一个激活函数序号出来。
  w_skip = u_skip         # [hidden_size, 2 * hidden_size] * num_layers
  var_s = [w_prev] + w_skip[1:]  # [2 * hidden_size, 2 * hidden_size] * 1 + [hidden_size, 2 * hidden_size] * (num_layers-1)。所有变量.

  def _select_function(h, function_id):
    h = tf.stack([tf.tanh(h), tf.nn.relu(h), tf.sigmoid(h), h], axis=0)
    h = h[function_id]
    return h

  def _condition(step, *unused_args):
    return tf.less(step, num_steps)

  def _body(step, prev_s, all_s):   # body中每次循环是训练数据中的一个时间步，即一个rnn cell。所以controller采样出来的是一个rnn cell内部的结构。
    """Body function."""
    inp = x[:, step, :]  # [batch_size, hidden_size]

    # important change: first input uses a tanh()
    if layer_mask is not None:
      assert input_mask is not None
      ht = tf.matmul(tf.concat([inp * input_mask, prev_s * layer_mask],
                               axis=1), w_prev)  # [batch_size, 2*hidden_size]
    else:
      ht = tf.matmul(tf.concat([inp, prev_s], axis=1), w_prev)
    h, t = tf.split(ht, 2, axis=1) # [batch_size, hidden_size], [batch_size, hidden_size]
    h = tf.tanh(h)
    t = tf.sigmoid(t)
    s = prev_s + t * (h - prev_s)  # [batch_size, hidden_size]
    layers = [s]

    start_idx = 0
    used = []
    for layer_id in range(num_layers):
      prev_idx = sample_arc[start_idx]
      func_idx = sample_arc[start_idx + 1]
      used.append(tf.one_hot(prev_idx, depth=num_layers, dtype=tf.int32))
      prev_s = tf.stack(layers, axis=0)[prev_idx] # [batch_size, hidden_size]
      if layer_mask is not None:
        ht = tf.matmul(prev_s * layer_mask, w_skip[layer_id])
      else:
        ht = tf.matmul(prev_s, w_skip[layer_id])  # [batch_size, 2 * hidden_size]
      h, t = tf.split(ht, 2, axis=1)

      h = _select_function(h, func_idx)
      t = tf.sigmoid(t)
      s = prev_s + t * (h - prev_s)
      s.set_shape([batch_size, params.hidden_size])   # [batch_size, hidden_size]
      layers.append(s)   # 保存每一层的输出.
      start_idx += 2

    next_s = tf.add_n(layers[1:]) / tf.cast(num_layers, dtype=tf.float32)  # add_n:输入的列表中的每个元素的对应维度相加。
    all_s = all_s.write(step, next_s)

    return step + 1, next_s, all_s

  loop_inps = [tf.constant(0, dtype=tf.int32), prev_s, all_s]
  _, next_s, all_s = tf.while_loop(_condition, _body, loop_inps)
  all_s = tf.transpose(all_s.stack(), [1, 0, 2])  # [batch_size, num_steps, hidden_size]

  return next_s, all_s, var_s


def _set_default_params(params):
  """Set default hyper-parameters."""
  params.add_hparam('alpha', 0.0)  # activation L2 reg
  params.add_hparam('beta', 1.)  # activation slowness reg
  params.add_hparam('best_valid_ppl_threshold', 5)

  params.add_hparam('batch_size', 64)

  # for dropouts: dropping rate, NOT keeping rate
  params.add_hparam('drop_rate', 0.30)  # voice attention
  params.add_hparam('drop_e', 0.10)  # word
  params.add_hparam('drop_i', 0.20)  # embeddings
  params.add_hparam('drop_x', 0.75)  # input to RNN cells
  params.add_hparam('drop_l', 0.25)  # between layers
  params.add_hparam('drop_o', 0.75)  # output
  params.add_hparam('drop_w', 0.00)  # weight

  params.add_hparam('grad_bound', 1)
  params.add_hparam('hidden_size', 200)
  params.add_hparam('init_range', 0.04)
  params.add_hparam('learning_rate', 1e-3)
  params.add_hparam('num_train_epochs', 600)

  params.add_hparam('weight_decay', 8e-7)

  params.add_hparam('max_sent_num', 200)
  params.add_hparam('max_sent_len', 100)
  params.add_hparam('w2v_dim', 200)
  params.add_hparam('vocab_sizes', 314041)
  params.add_hparam('w2v_istrain', True)

  params.add_hparam('train_root', '/workspace/speaker_verification/data/dahai/train/va_widxdahai_tfrecord_200_100/')
  params.add_hparam('test_dahai_root', '/workspace/speaker_verification/data/dahai/test/va_widxdahai_tfrecord_200_200')
  params.add_hparam('test_zhikang_root', '/workspace/speaker_verification/data/zhikang/test/va_widxdahai_tfrecord_200_200')
  params.add_hparam('pretrain_embedding_path', '/workspace/speaker_verification/data/w2v_online/0718_dahaipretrain/w2v_mtrx.npy')

  return params


class sentRNN:

    def __init__(self, config, controller, name="child"):

        self.config = _set_default_params(config)
        self.name = name
        self.controller = controller
        self.sample_arc = tf.unstack(controller.sample_arc)  # [1] * (2*num_layers)

        w2v_np = np.load(self.config.pretrain_embedding_path)
        self.w2v_np = np.concatenate([np.array([[0.0] * self.config.w2v_dim]), w2v_np], axis=0)

        train_root = self.config.train_root
        train_dataset = tf.data.TFRecordDataset([os.path.join(train_root, x) for x in os.listdir(train_root)])
        parsed_train = train_dataset.map(lambda example_proto: self.parse_helper(example_proto))
        parsed_train = parsed_train.shuffle(10000)
        parsed_train = parsed_train.apply(
            tf.contrib.data.padded_batch_and_drop_remainder(
                batch_size=self.config.batch_size, padded_shapes=(
                    [self.config.max_sent_num, 64], [self.config.max_sent_num, self.config.max_sent_len],
                    [self.config.max_sent_num], [self.config.max_sent_num], [])))
        parsed_train = parsed_train.repeat()
        train_iter = parsed_train.make_one_shot_iterator()
        self.train_next = train_iter.get_next()

        test_zhikang_root = self.config.test_zhikang_root
        test_zhikang_dataset = tf.data.TFRecordDataset(
            [os.path.join(test_zhikang_root, x) for x in os.listdir(test_zhikang_root)])
        parsed_test_zhikang = test_zhikang_dataset.map(lambda example_proto: self.parse_helper(example_proto))
        parsed_test_zhikang = parsed_test_zhikang.shuffle(10000)
        parsed_test_zhikang = parsed_test_zhikang.apply(
            tf.contrib.data.padded_batch_and_drop_remainder(
                batch_size=self.config.batch_size, padded_shapes=(
                    [self.config.max_sent_num, 64], [self.config.max_sent_num, self.config.max_sent_len],
                    [self.config.max_sent_num], [self.config.max_sent_num], [])))
        parsed_test_zhikang = parsed_test_zhikang.repeat()
        test_zhikang_iter = parsed_test_zhikang.make_one_shot_iterator()
        self.test_zhikang_next = test_zhikang_iter.get_next()

        test_dahai_root = self.config.test_dahai_root
        test_dahai_dataset = tf.data.TFRecordDataset(
            [os.path.join(test_dahai_root, x) for x in os.listdir(test_dahai_root)])
        parsed_test_dahai = test_dahai_dataset.map(lambda example_proto: self.parse_helper(example_proto))
        parsed_test_dahai = parsed_test_dahai.shuffle(10000)
        parsed_test_dahai = parsed_test_dahai.apply(
            tf.contrib.data.padded_batch_and_drop_remainder(
                batch_size=self.config.batch_size, padded_shapes=(
                    [self.config.max_sent_num, 64], [self.config.max_sent_num, self.config.max_sent_len],
                    [self.config.max_sent_num], [self.config.max_sent_num], [])))
        parsed_test_dahai = parsed_test_dahai.repeat()
        test_dahai_iter = parsed_test_dahai.make_one_shot_iterator()
        self.test_dahai_next = test_dahai_iter.get_next()

        self._build_params()
        self._build_train()
        self._build_valid()


    def _build_params(self):
        """Create model parameters."""

        print('-' * 80)
        print('Building model params')

        num_functions = self.config.controller_num_functions
        num_layers = self.config.controller_num_layers
        hidden_size = self.config.hidden_size

        with tf.variable_scope(self.name):

            self.is_training = tf.placeholder(dtype=tf.bool, shape=[])
            self.voice_embed = tf.placeholder(shape=[self.config.batch_size, self.config.max_sent_num, 64],
                                         dtype=tf.float32)  # shape: (batch, max_sent_num, 64)

            # Build the netword for the words
            self.word_idx = tf.placeholder(
                shape=[self.config.batch_size, self.config.max_sent_num, self.config.max_sent_len],
                dtype=tf.int32)  # shape: (batch, max_sent_num, max_sent_len)
            self.sent_len = tf.placeholder(shape=[self.config.batch_size, self.config.max_sent_num],
                                      dtype=tf.int32)  # shape: (batch, max_sent_num)
            w2v_embedding = tf.Variable(tf.constant(0.0, shape=[self.config.vocab_sizes, self.config.w2v_dim]),
                                        trainable=self.config.w2v_istrain, name="w2i_embedding")
            self.embedding_placeholder = tf.placeholder(tf.float32, [self.config.vocab_sizes, self.config.w2v_dim])
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

            self.label = tf.placeholder(shape=[self.config.batch_size, self.config.max_sent_num],
                                        dtype=tf.int32)  # shape: (batch, max_sent_num)
            self.seq_len = tf.placeholder(shape=[self.config.batch_size], dtype=tf.int32)  # shape: (batch)

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
                self.voice_A = tf.nn.softmax(voice_A)  # shape: (batch, max_length, max_length)
                voice_O = tf.matmul(self.voice_A, word_V)  # shape: (batch, max_length, 200)
                voice_O = self.mask(voice_O, self.seq_len, mode='mul', max_len=self.config.max_sent_num)  # shape: (batch, max_sent_num, 200)

                O = tf.concat([voice_O, word_V], axis=-1)
                self.O = tf.layers.dense(inputs=O, units=self.config.hidden_size, activation=None, use_bias=False,
                                          kernel_initializer=tf.truncated_normal_initializer(), name='dense_O')

            with tf.variable_scope("enas_rnn_fw"):
                with tf.variable_scope('rnn_cell'):
                    w_prev = tf.get_variable('w_prev', [2 * hidden_size, 2 * hidden_size])
                    i_mask = tf.ones([hidden_size, 2 * hidden_size], dtype=tf.float32)
                    h_mask = self._gen_mask([hidden_size, 2 * hidden_size], self.config.drop_w)
                    mask = tf.concat([i_mask, h_mask], axis=0)
                    dropped_w_prev = w_prev * mask

                    w_skip, dropped_w_skip = [], []
                    for layer_id in range(1, num_layers + 1):
                        with tf.variable_scope('layer_{}'.format(layer_id)):
                            w = tf.get_variable(
                                'w', [num_functions, layer_id, hidden_size, 2 * hidden_size])
                            mask = self._gen_mask([1, 1, hidden_size, 2 * hidden_size],
                                             self.config.drop_w)
                            dropped_w = w * mask
                            w_skip.append(w)
                            dropped_w_skip.append(dropped_w)

                with tf.variable_scope('init_states'):
                    with tf.variable_scope('batch'):
                        init_shape = [self.config.batch_size, hidden_size]
                        batch_prev_s = tf.zeros(init_shape, dtype=np.float32, name="s")

            with tf.variable_scope("enas_rnn_bw"):
                with tf.variable_scope('rnn_cell'):
                    w_prev_bw = tf.get_variable('w_prev', [2 * hidden_size, 2 * hidden_size])
                    i_mask_bw = tf.ones([hidden_size, 2 * hidden_size], dtype=tf.float32)
                    h_mask_bw = self._gen_mask([hidden_size, 2 * hidden_size], self.config.drop_w)
                    mask_bw = tf.concat([i_mask_bw, h_mask_bw], axis=0)
                    dropped_w_prev_bw = w_prev_bw * mask_bw

                    w_skip_bw, dropped_w_skip_bw = [], []
                    for layer_id in range(1, num_layers + 1):
                        with tf.variable_scope('layer_{}'.format(layer_id)):
                            w_bw = tf.get_variable(
                                'w', [num_functions, layer_id, hidden_size, 2 * hidden_size])
                            mask_bw = self._gen_mask([1, 1, hidden_size, 2 * hidden_size],
                                                  self.config.drop_w)
                            dropped_w_bw = w_bw * mask_bw
                            w_skip_bw.append(w_bw)
                            dropped_w_skip_bw.append(dropped_w_bw)

                with tf.variable_scope('init_states'):
                    with tf.variable_scope('batch'):
                        init_shape_bw = [self.config.batch_size, hidden_size]
                        batch_prev_s_bw = tf.zeros(init_shape_bw, dtype=np.float32, name="s")

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

        self.batch_init_states = {
            's': batch_prev_s,
            's_bw': batch_prev_s_bw,
        }
        self.train_params = {
            'w_prev': dropped_w_prev,
            'w_skip': dropped_w_skip,
            'w_prev_bw': dropped_w_prev_bw,
            'w_skip_bw': dropped_w_skip_bw,
            'w_out': w_out,
        }
        self.eval_params = {
            'w_prev': w_prev,
            'w_skip': w_skip,  # [num_functions, layer_id, hidden_size, 2 * hidden_size] * num_layers
            'w_prev_bw': w_prev_bw,
            'w_skip_bw': w_skip_bw,
            'w_out': w_out,
        }

    def _forward(self, model_params, init_states, is_training=False):
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
        prev_s = init_states['s']
        prev_s_bw = init_states['s_bw']

        emb = self.O
        batch_size = self.config.batch_size
        hidden_size = self.config.hidden_size
        sample_arc = self.sample_arc
        if is_training:
            emb = tf.layers.dropout(
                emb, self.config.drop_i, [batch_size, 1, self.config.hidden_size], training=True)  # 控制每个时间步drop掉同样的

            input_mask = self._gen_mask([batch_size, hidden_size], self.config.drop_x)
            layer_mask = self._gen_mask([batch_size, hidden_size], self.config.drop_l)
        else:
            input_mask = None
            layer_mask = None

        all_s, var_s = _bidirectional_rnn(sample_arc=sample_arc,
                                          O=emb,
                                          seq_len=self.seq_len,
                                          prev_s=prev_s,
                                          w_prev=w_prev,
                                          w_skip=w_skip,
                                          prev_s_bw=prev_s_bw,
                                          w_prev_bw=w_prev_bw,
                                          w_skip_bw=w_skip_bw,
                                          input_mask=input_mask,
                                          layer_mask=layer_mask,
                                          config=self.config)

        top_s = all_s  # [batch_size, max_sent_num, 2*hidden_size]
        if is_training:
            top_s = tf.layers.dropout(
                top_s, self.config.drop_o,
                [self.config.batch_size, 1, self.config.hidden_size*2], training=True)  # 添加noise mask，确保时间步上的drop一致。

        logits = tf.einsum('bnh,hc->bnc', top_s, w_out)  # [batch_size, num_steps, vocabe_size]

        # calculate the loss
        label_onehot = tf.one_hot(indices=self.label, depth=2, on_value=1, off_value=0, axis=-1, dtype=tf.int32)
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.reshape(label_onehot, shape=[-1, 2]),
                                                          logits=tf.reshape(logits,
                                                                            shape=[-1,
                                                                                   2]))  # shape: (batch*max_length,1)
        loss = tf.reshape(loss, shape=[self.config.batch_size, -1])  # [batch_size, max_length]

        # add a mask based on whether it is 0
        balance_mask_1 = tf.cast(tf.sequence_mask(self.seq_len, self.config.max_sent_num), tf.float32)
        balance_mask_2 = tf.cast(tf.equal(self.label, 1), tf.float32) * 1
        balance_mask_3 = tf.cast(tf.equal(self.label, 0), tf.float32) * 2
        temp_mask = tf.cast(balance_mask_2 > 0, tf.float32) + tf.cast(balance_mask_3 > 0, tf.float32)
        balance_mask = balance_mask_1 * (balance_mask_2 + balance_mask_3)
        loss = loss * balance_mask
        loss = tf.reduce_sum(loss, axis=1)  # shape: (batch)

        temp_label = tf.expand_dims(self.label, -1)  # [batch, max_length, 1]
        temp_label = tf.matmul(1 - temp_label, temp_label, transpose_b=True) + tf.matmul(temp_label, 1 - temp_label,
                                                                                         transpose_b=True)  # [batch, max_len, max_len]
        constraint_mask_1 = tf.cast(tf.equal(temp_label, 1), tf.float32)  # [batch, max_len, max_len]
        constraint_mask_2 = tf.cast(tf.sequence_mask(self.seq_len, self.config.max_sent_num), tf.float32)  # [batch, max_length]
        constraint_mask_2 = tf.expand_dims(constraint_mask_2, -1)  # [batch, max_length, 1]
        constraint_mask_2 = tf.matmul(constraint_mask_2, constraint_mask_2,
                                      transpose_b=True)  # [batch, max_length, max_length]
        constraint_mask = constraint_mask_1 * constraint_mask_2
        # care about the attentions
        attention_loss = self.voice_A ** 2 * constraint_mask  # [batch, max_length, max_length]
        attention_loss = tf.reduce_sum(attention_loss, axis=[1, 2])  # [batch]

        loss = loss + self.config.alpha * attention_loss

        loss = tf.reduce_mean(loss / tf.cast(tf.reduce_sum(balance_mask, axis=1), dtype=tf.float32))

        reg_loss = loss  # `loss + regularization_terms` is for training only
        if is_training:
            # L2 weight reg
            self.l2_reg_loss = tf.add_n([tf.nn.l2_loss(w ** 2) for w in var_s])
            reg_loss += self.config.weight_decay * self.l2_reg_loss

            # activation L2 reg
            reg_loss += self.config.alpha * tf.reduce_mean(all_s ** 2)

            # activation slowness reg
            reg_loss += self.config.beta * tf.reduce_mean(
                (all_s[:, 1:, :] - all_s[:, :-1, :]) ** 2)

        loss = tf.identity(loss)
        if is_training:
            reg_loss = tf.identity(reg_loss)

        # add the accuracy
        pred_label = tf.cast(tf.argmax(logits, -1), tf.int32)
        correct_label = tf.cast(tf.equal(pred_label, self.label), tf.float32)
        correct_label = correct_label * balance_mask_1 * temp_mask
        accuracy = tf.cast(tf.reduce_sum(correct_label), tf.float32) / tf.cast(
            tf.reduce_sum(temp_mask * balance_mask_1),
            tf.float32)

        return reg_loss, loss, accuracy  # loss是纯损失，reg loss里有l2正则损失。

    def _build_train(self):
        """Build training ops."""
        print('-' * 80)
        print('Building train graph')
        reg_loss, loss, accuracy = self._forward(self.train_params, self.batch_init_states, is_training=True)

        tf_vars = [v for v in tf.trainable_variables()
                   if v.name.startswith(self.name)]
        global_step = tf.train.get_or_create_global_step()
        lr_scale = 1
        learning_rate = utils.get_lr(global_step, self.config) * lr_scale

        if self.config.grad_bound:
            grads = tf.gradients(reg_loss, tf_vars)
            clipped_grads, grad_norm = tf.clip_by_global_norm(grads,
                                                              self.config.grad_bound)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.apply_gradients(zip(clipped_grads, tf_vars),
                                             global_step=global_step)

        self.train_loss = loss
        self.train_acc = accuracy
        self.train_op = train_op
        self.grad_norm = grad_norm
        self.learning_rate = learning_rate

        self.loss_summary = tf.summary.scalar("Child_Loss", self.train_loss)
        self.acc_summary = tf.summary.scalar("Child_Accuracy", self.train_acc)


    def _build_valid(self):
        print('Building valid graph')
        _, loss, accuracy = self._forward(self.eval_params, self.batch_init_states, is_training=False)
        self.valid_loss = loss
        self.rl_loss = loss
        self.valid_acc = accuracy


    def eval_valid(self, sess):
        """Eval 1 round on valid set."""

        all_test_dahai_dataset = tf.data.TFRecordDataset(
            [os.path.join(self.config.test_dahai_root, x) for x in os.listdir(self.config.test_dahai_root)])
        all_parsed_test_dahai = all_test_dahai_dataset.map(lambda example_proto: self.parse_helper(example_proto))
        all_parsed_test_dahai = all_parsed_test_dahai.apply(
            tf.contrib.data.padded_batch_and_drop_remainder(
                batch_size=self.config.batch_size, padded_shapes=(
                    [self.config.max_sent_num, 64], [self.config.max_sent_num,self.config.max_sent_len],
                    [self.config.max_sent_num], [self.config.max_sent_num], []
                )))
        all_test_dahai_iter = all_parsed_test_dahai.make_one_shot_iterator()
        all_test_dahai_next = all_test_dahai_iter.get_next()

        cnt = 0
        total_loss = 0
        total_acc = 0
        while True:
            try:
                pred_voice_embed, pred_word_idx, pred_sent_len, true_label, pred_seq_len = sess.run(
                    all_test_dahai_next)
            except tf.errors.OutOfRangeError:
                break
            loss, acc = sess.run([self.valid_loss, self.valid_acc], feed_dict={self.voice_embed: pred_voice_embed,
                                                self.word_idx: pred_word_idx,
                                                self.sent_len: pred_sent_len,
                                                self.seq_len: pred_seq_len,
                                                self.label: true_label,
                                                self.is_training: False})
            cnt += 1
            total_loss += loss
            total_acc += acc

        valid_ppl = np.exp(total_loss / cnt)
        valid_acc = total_acc / cnt
        print('valid_ppl={0:<.2f}'.format(valid_ppl))
        print('valid_acc={0:<.2f}'.format(valid_acc))

        return valid_ppl


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

    def _gen_mask(self, shape, drop_prob):
        """Generate a droppout mask."""
        keep_prob = 1. - drop_prob
        mask = tf.random_uniform(shape, dtype=tf.float32)
        mask = tf.floor(mask + keep_prob) / keep_prob
        return mask










