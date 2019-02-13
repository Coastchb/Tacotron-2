from tacotron.models import create_model
import tensorflow as tf
from hparams import hparams
from tensorflow.python.framework import graph_util

checkpoint_path = "data/LJSpeech-1.1/logs-Tacotron-2/taco_pretrained/tacotron_model.ckpt-20000"
#checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
output_file = "tf.pb"
print(checkpoint_path)

sess = tf.InteractiveSession()

inputs = tf.placeholder(tf.int32, [1, None], name="text")
inputs_len = tf.placeholder(tf.int32, [1], name="text_len")
split_infos = tf.placeholder(tf.int32, shape=(hparams.tacotron_num_gpus, None), name='split_infos')
model = create_model("Tacotron", hparams)
model.initialize(inputs, inputs_len, is_training=False, is_evaluating=False, split_infos = split_infos)
print(model.tower_mel_outputs)
output = model.tower_mel_outputs[0][0]
tf.identity(output, "mel_target")

saver = tf.train.Saver(tf.global_variables())
saver.restore(sess, checkpoint_path)

frozen_graph_def = graph_util.convert_variables_to_constants(
      sess, sess.graph_def, ['mel_target'])
tf.train.write_graph(
      frozen_graph_def,
      os.path.dirname(output_file),
      os.path.basename(output_file),
      as_text=False)