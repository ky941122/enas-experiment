#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date   : 2019-08-20
# @Author : KangYu
# @File   : resave_ckpt.py


import tensorflow as tf
import os

import model_for_resave

flags = tf.app.flags
gfile = tf.gfile
FLAGS = flags.FLAGS

flags.DEFINE_boolean('reset_output_dir', True, '')
flags.DEFINE_string('output_dir', None, '')
flags.DEFINE_string('ckpt_path', None, '')

flags.DEFINE_integer('log_every', 100, '')

output_dir = FLAGS.output_dir

if __name__ == "__main__":

    params = tf.contrib.training.HParams(
        log_every=FLAGS.log_every,
        output_dir=output_dir,
        model_save_steps=2000,
        ckpt_path=FLAGS.ckpt_path,
    )

    g = tf.Graph()
    with g.as_default():
        child = model_for_resave.sentRNN(params)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        init = tf.initialize_all_variables()
        sess = tf.Session(config=config)
        sess.run(init)

        saver = tf.train.Saver(var_list=tf.global_variables())
        saver.restore(sess, params.ckpt_path)
        print("*" * 20 + "\nReading model parameters from %s \n" % params.ckpt_path + "*" * 20)

        saver = tf.train.Saver(max_to_keep=10000)

        saver.save(sess, os.path.join(params.output_dir, "./Check_Point/model.ckpt"))
        print("*" * 20 + "\nSaving model parameters to %s \n" % os.path.join(params.output_dir, "./Check_Point/model.ckpt") + "*" * 20)

