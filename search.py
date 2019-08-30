# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Entry point for AWD ENAS search process."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import sys
import time

import numpy as np
import tensorflow as tf
# from tensorflow.python import debug as tf_debug

import child_sentRNN
import controller as ct
import utils
import functools

flags = tf.app.flags
gfile = tf.gfile
FLAGS = flags.FLAGS

flags.DEFINE_boolean('reset_output_dir', False, '')
flags.DEFINE_string('output_dir', None, '')

flags.DEFINE_integer('log_every', 50, '')

#controller hyparams
flags.DEFINE_float('controller_learning_rate', 5e-3, '')
flags.DEFINE_integer('controller_num_layers', 3, '')



def _model_stats():
    """Print trainable variables and total model size."""

    def size(v):
        return functools.reduce(lambda x, y: x * y, v.get_shape().as_list())

    print("Trainable variables")
    for v in tf.trainable_variables():
        print("  %s, %s, %s, %s" % (v.name, v.device, str(v.get_shape()), size(v)))
    print("Total model size: %d" % (sum(size(v) for v in tf.trainable_variables())))


def train(params):
  """Entry train function."""
  g = tf.Graph()
  with g.as_default():

    #Build [train, valid, test] graphs.
    controller = ct.Controller(
        params=params)  # 创建对象的时候就sample了一次。先sample一次，用sample出来的child训练参数训练一个epoch，再用训完的参数回去训练controller。以此循环。
    child = child_sentRNN.sentRNN(params, controller)
    controller.build_trainer(child)

    _model_stats()

    child_merge_summary = tf.summary.merge([child.loss_summary, child.acc_summary])

    ops = {
        'train_op': child.train_op,
        'learning_rate': child.learning_rate,
        'grad_norm': child.grad_norm,
        'train_loss': child.train_loss,
        'dev_loss': child.valid_loss,
        'train_acc': child.train_acc,
        'dev_acc': child.valid_acc,
        'l2_reg_loss': child.l2_reg_loss,
        'global_step': tf.train.get_or_create_global_step(),
        'eval_valid': child.eval_valid,

        "child_merge_summary": child_merge_summary,

        'controller_train_op': controller.train_op,
        'controller_grad_norm': controller.train_op,
        'controller_sample_arc': controller.sample_arc,
        'controller_entropy': controller.sample_entropy,
        'controller_reward': controller.reward,
        'controller_baseline': controller.baseline,
        'controller_optimizer': controller.optimizer,
        'controller_train_fn': controller.train,
    }
    print('-' * 80)
    print('HParams:\n{0}'.format(params.to_json(indent=2, sort_keys=True)))

    run_ops = [
        ops['train_loss'],
        ops['train_acc'],
        ops['l2_reg_loss'],
        ops['grad_norm'],
        ops['learning_rate'],
        ops['train_op'],
        ops["child_merge_summary"],
    ]
    dev_ops = [
        ops['dev_loss'],
        ops['dev_acc'],
        ops["child_merge_summary"],
    ]

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    init = tf.initialize_all_variables()
    sess = tf.Session(config=config)
    sess.run(init)

    child_train_writer = tf.summary.FileWriter(os.path.join(params.output_dir, "logs/child_train"), sess.graph)
    child_zhikang_dev_writer = tf.summary.FileWriter(os.path.join(params.output_dir, "logs/child_test_zhikang"), sess.graph)
    child_dahai_dev_writer = tf.summary.FileWriter(os.path.join(params.output_dir, "logs/child_test_dahai"), sess.graph)

    controller_train_writer = tf.summary.FileWriter(os.path.join(params.output_dir, "logs/controller_train"), sess.graph)

    sess.run(child.embedding_init, feed_dict={child.embedding_placeholder: child.w2v_np})

    ##### read pretrain ckpt #####

    variables = tf.contrib.framework.get_variables_to_restore()
    variables_to_resotre = [v for v in variables if (
                                ("Adam" not in v.name) and
                                (v.name.split('/')[0] == "child") and
                                ("sent_classify_lstm" not in v.name) and
                                ("enas_rnn" not in v.name) and
                                ("w_out" not in v.name)
                            )]
    print("*"*20)
    print("variables_to_resotre:\n")
    for v in variables_to_resotre:
        print(str(v) + "\n")
    print("*" * 20)

    saver = tf.train.Saver(variables_to_resotre)
    saver.restore(sess, params.ckpt_path)
    print("*" * 20 + "\nReading model parameters from %s \n" % params.ckpt_path + "*" * 20)

    ##############################

    saver = tf.train.Saver(max_to_keep=10000)

    train_accum_loss = 0
    zhikang_accum_loss = 0
    dahai_accum_loss = 0
    accum_step = 0
    epoch = 0
    best_valid_ppl = []
    start_time = time.time()
    while True:
        train_voice_embed, train_word_idx, train_sent_len, train_label, train_seq_len = sess.run(child.train_next)
        test_voice_embed, test_word_idx, test_sent_len, test_label, test_seq_len = sess.run(child.test_zhikang_next)
        test_voice_embed_2, test_word_idx_2, test_sent_len_2, test_label_2, test_seq_len_2 = sess.run(child.test_dahai_next)

        train_loss, train_acc, train_l2_reg, train_gn, train_lr, _, train_summary = sess.run(run_ops,
                                                                    feed_dict={child.voice_embed: train_voice_embed,
                                                                    child.word_idx: train_word_idx,
                                                                    child.sent_len: train_sent_len,
                                                                    child.label: train_label,
                                                                    child.seq_len: train_seq_len,
                                                                    child.is_training: True})

        zhikang_dev_loss, zhikang_dev_acc, zhikang_dev_summary = sess.run(dev_ops, feed_dict={child.voice_embed: test_voice_embed,
                                                         child.word_idx: test_word_idx,
                                                         child.sent_len: test_sent_len,
                                                         child.label: test_label,
                                                         child.seq_len: test_seq_len,
                                                         child.is_training: False})

        dahai_dev_loss, dahai_dev_acc, dahai_dev_summary = sess.run(dev_ops, feed_dict={child.voice_embed: test_voice_embed_2,
                                                                         child.word_idx: test_word_idx_2,
                                                                         child.sent_len: test_sent_len_2,
                                                                         child.label: test_label_2,
                                                                         child.seq_len: test_seq_len_2,
                                                                         child.is_training: False})

        train_accum_loss += train_loss
        zhikang_accum_loss += zhikang_dev_loss
        dahai_accum_loss += dahai_dev_loss
        accum_step += 1
        step = sess.run(ops['global_step'])

        if step % 10 == 0:
            child_train_writer.add_summary(train_summary, step)  # write at tensorboard
            child_zhikang_dev_writer.add_summary(zhikang_dev_summary, step)  # write at tensorboard
            child_dahai_dev_writer.add_summary(dahai_dev_summary, step)  # write at tensorboard


        if step % params.log_every == 0:
          train_ppl = np.exp(train_accum_loss / accum_step)
          train_loss_mean = train_accum_loss / accum_step
          mins_so_far = (time.time() - start_time) / 60.
          log_string = 'epoch={0:<5d}'.format(epoch)
          log_string += ' step={0:<7d}'.format(step)
          log_string += ' train_loss={0:<9.2f}'.format(train_loss_mean)
          log_string += ' train_ppl={0:<9.2f}'.format(train_ppl)
          log_string += ' train_acc={0:<9.2f}'.format(train_acc)
          log_string += ' lr={0:<7.2f}'.format(train_lr)
          log_string += ' train_|w|={0:<6.2f}'.format(train_l2_reg)
          log_string += ' train_|g|={0:<6.2f}'.format(train_gn)
          log_string += ' mins={0:<.2f}'.format(mins_so_far)
          print(log_string)

          zhikang_ppl = np.exp(zhikang_accum_loss / accum_step)
          zhikang_loss = zhikang_accum_loss / accum_step
          log_string += ' zhikang_loss={0:<9.2f}'.format(zhikang_loss)
          log_string += ' zhikang_ppl={0:<9.2f}'.format(zhikang_ppl)
          log_string += ' zhikang_acc={0:<9.2f}'.format(zhikang_dev_acc)
          print(log_string)

          dahai_ppl = np.exp(dahai_accum_loss / accum_step)
          dahai_loss = dahai_accum_loss / accum_step
          log_string += ' dahai_dev_loss={0:<9.2f}'.format(dahai_loss)
          log_string += ' dahai_dev_ppl={0:<9.2f}'.format(dahai_ppl)
          log_string += ' dahai_dev_acc={0:<9.2f}'.format(dahai_dev_acc)
          print(log_string)

        if step != 0 and step % params.model_save_steps == 0:
            saver.save(sess, os.path.join(params.output_dir, "./Check_Point/model.ckpt"), global_step=step)


        if epoch >= params.num_train_epochs:
          break


        if step != 0 and step % params.train_controller_every == 0:
          ops['controller_train_fn'](sess=sess, child=child, writer=controller_train_writer)
          epoch += 1
          train_accum_loss = 0
          zhikang_accum_loss = 0
          dahai_accum_loss = 0
          accum_step = 0
          valid_ppl = ops['eval_valid'](sess)
          best_valid_ppl.append(valid_ppl)

    sess.close()


def main():
  np.set_printoptions(precision=3, suppress=True, threshold=int(1e9),
                      linewidth=80)

  print('-' * 80)
  if not gfile.IsDirectory(FLAGS.output_dir):
    print('Path {} does not exist. Creating'.format(FLAGS.output_dir))
    gfile.MakeDirs(FLAGS.output_dir)
  elif FLAGS.reset_output_dir:
    print('Path {} exists. Reseting'.format(FLAGS.output_dir))
    gfile.DeleteRecursively(FLAGS.output_dir)
    gfile.MakeDirs(FLAGS.output_dir)

  print('-' * 80)
  log_file = os.path.join(FLAGS.output_dir, 'stdout')
  print('Logging to {}'.format(log_file))
  sys.stdout = utils.Logger(log_file)

  params = tf.contrib.training.HParams(
      log_every=FLAGS.log_every,
      output_dir=FLAGS.output_dir,
      model_save_steps=2000,
      ckpt_path="/workspace/speaker_verification/enas_experiment/pretrain_sentRNN_for_child/Check_Point/model.ckpt-49",
  )

  params.add_hparam('controller_learning_rate', FLAGS.controller_learning_rate)
  params.add_hparam('controller_num_layers', FLAGS.controller_num_layers)

  train(params)

if __name__ == '__main__':
  main()

