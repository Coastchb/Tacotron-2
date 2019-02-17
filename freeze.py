from tacotron.models import create_model
import tensorflow as tf
from hparams import hparams
from tensorflow.python.framework import graph_util
import os

checkpoint_dir = "data/LJSpeech-1.1/logs-Tacotron/taco_pretrained/"
checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
output_file = "tf.pb_gpu_2"

sess = tf.InteractiveSession()

with tf.variable_scope('Tacotron_model', reuse=tf.AUTO_REUSE) as scope:
      with tf.device("/cpu:0"):
            inputs = tf.placeholder(tf.int32, [1, None], name="text")
            inputs_len = tf.placeholder(tf.int32, [1], name="text_len")
            split_infos = tf.placeholder(tf.int32, shape=(hparams.tacotron_num_gpus, None), name='split_infos')
            model = create_model("Tacotron", hparams)
            model.initialize(inputs, inputs_len, is_training=False, is_evaluating=False, split_infos = split_infos)
            print("#######")
            print(model.tower_mel_outputs)
           # output = model.tower_mel_outputs[0][0]
           # tf.identity(output, "mel_target", )

            saver = tf.train.Saver(tf.global_variables())
            saver.restore(sess, checkpoint_path)

frozen_graph_def = graph_util.convert_variables_to_constants(
      sess, sess.graph_def, ['Tacotron_model/mel_outputs'])

tf.train.write_graph(
      frozen_graph_def,
      os.path.dirname(output_file),
      os.path.basename(output_file),
      as_text=False)