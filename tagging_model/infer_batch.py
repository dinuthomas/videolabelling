# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Binary for evaluating Tensorflow models on the YouTube-8M dataset."""

import glob
import json
import os
import time

import eval_util
import losses
import frame_level_models
import video_level_models
import readers
import tensorflow as tf
from tensorflow import app
from tensorflow import flags
from tensorflow import gfile
from tensorflow import logging
import utils
import numpy as np
import pandas as pd
from tensorflow.python import pywrap_tensorflow

FLAGS = flags.FLAGS

if __name__ == "__main__":
  # Dataset flags.
  flags.DEFINE_string("train_dir", "/tmp/yt8m_model/",
                      "The directory to load the model files from. "
                      "The tensorboard metrics files are also saved to this "
                      "directory.")
  flags.DEFINE_string(
      "eval_data_pattern", "",
      "File glob defining the evaluation dataset in tensorflow.SequenceExample "
      "format. The SequenceExamples are expected to have an 'rgb' byte array "
      "sequence feature as well as a 'labels' int64 context feature.")

  # Other flags.
  flags.DEFINE_integer("batch_size", 64,
                       "How many examples to process per batch.")
  flags.DEFINE_integer("num_readers", 8,
                       "How many threads to use for reading input files.")
  flags.DEFINE_boolean("run_once", True, "Whether to run eval only once.")
  flags.DEFINE_integer("top_k", 20, "How many predictions to output per video.")

  flags.DEFINE_integer("n_rounds", 1, "How many rounds of eval to run.")

  flags.DEFINE_boolean("use_EMA", False, "Whether to use EMA shadow variables.")
  flags.DEFINE_boolean("build_only", True, "Whether to build graph, but not evaluate. "
                       "This will build graph without the coordinators.")


def data_loader(FLAGS, batch_size):
    dirs = os.listdir(FLAGS.eval_data_pattern)
    idx = 0
    video_id_array = []
    for file in dirs:
      try:
          with open(FLAGS.eval_data_pattern+'/'+file) as f:
            dict = json.load(f)
      except ValueError as e:
          continue
      id = file.split('.')[0]
      feats = np.append(dict['rgb'][0]['floatList']['value'], dict['audio'][0]['floatList']['value'])
      length = min(len(dict['rgb']), len(dict['audio']))
      for i in range(1, length):
        rgb = np.array(dict['rgb'][i]['floatList']['value'])
        audio = np.array(dict['audio'][i]['floatList']['value'])
        line = np.append(rgb,audio)
        feats = np.vstack((feats,line))
      feats = np.expand_dims(np.vstack((feats, np.zeros((300-length, 1152)))), 0)
      if idx == 0 or (idx)%batch_size == 0:
        all_feats = feats
        video_id_array = []
      else:
        all_feats = np.vstack((all_feats,feats))

      video_id_array.append(id)
      idx += 1
      if (idx)%batch_size == 0 and idx!=0 and len(all_feats)>=16:
        #print(all_feats[-batch_size:,:,:],video_id_array[-batch_size:])
        yield all_feats[-batch_size:,:,:],video_id_array[-batch_size:]
      elif idx>58368:
        yield all_feats, video_id_array
      else: # dinu
        yield all_feats, video_id_array

def find_class_by_name(name, modules):
  """Searches the provided modules for the named class and returns it."""
  modules = [getattr(module, name, None) for module in modules]
  return next(a for a in modules if a)



def get_input_evaluation_tensors(reader,
                                 data_pattern,
                                 batch_size=1024,
                                 num_readers=1):
  """Creates the section of the graph which reads the evaluation data.

  Args:
    reader: A class which parses the training data.
    data_pattern: A 'glob' style path to the data files.
    batch_size: How many examples to process at a time.
    num_readers: How many I/O threads to use.

  Returns:
    A tuple containing the features tensor, labels tensor, and optionally a
    tensor containing the number of frames per video. The exact dimensions
    depend on the reader being used.

  Raises:
    IOError: If no files matching the given pattern were found.
  """
  logging.info("Using batch size of " + str(batch_size) + " for evaluation.")
  with tf.name_scope("eval_input"):
    files = gfile.Glob(data_pattern)
    if not files:
      raise IOError("Unable to find the evaluation files.")
    logging.info("number of evaluation files: " + str(len(files)))
    filename_queue = tf.train.string_input_producer(
        files, shuffle=False, num_epochs=1)
    eval_data = [
        reader.prepare_reader(filename_queue) for _ in range(num_readers)
    ]
    return tf.train.batch_join(
        eval_data,
        batch_size=batch_size,
        capacity=15 * batch_size,
        allow_smaller_final_batch=True,
        enqueue_many=True)


def build_graph(reader,
                model,
                eval_data_pattern,
                label_loss_fn,
                batch_size=1024,
                num_readers=1):
  """Creates the Tensorflow graph for evaluation.

  Args:
    reader: The data file reader. It should inherit from BaseReader.
    model: The core model (e.g. logistic or neural net). It should inherit
           from BaseModel.
    eval_data_pattern: glob path to the evaluation data files.
    label_loss_fn: What kind of loss to apply to the model. It should inherit
                from BaseLoss.
    batch_size: How many examples to process at a time.
    num_readers: How many threads to use for I/O operations.
  """
  global_step = tf.Variable(0, trainable=False, name="global_step")

  if not FLAGS.build_only:
    video_id_batch, model_input_raw, labels_batch, num_frames = get_input_evaluation_tensors(  # pylint: disable=g-line-too-long
        reader,
        eval_data_pattern,
        batch_size=batch_size,
        num_readers=num_readers)
  # TODO: this only works for FRAME image+auido input
  else:
    video_id_batch = tf.placeholder(tf.string, shape=(None, ), name="xvideo_id_batch")
    model_input_raw = tf.placeholder(tf.float32, shape=(None, 300, 1152), name="xmodel_input_raw")
    #labels_batch = tf.placeholder(tf.bool, shape=(None, 3862), name="xlabels_batch")
    num_frames = tf.placeholder(tf.int32, shape=(None), name="xnum_frames")
  if not FLAGS.build_only:
    tf.summary.histogram("model_input_raw", model_input_raw)

  feature_dim = len(model_input_raw.get_shape()) - 1
  
  # Normalize input features.
  model_input = tf.nn.l2_normalize(model_input_raw, feature_dim)

  logging.info("modelinput: " + str(model_input_raw)) # dinu
  logging.info("feature_dim: " + str(feature_dim)) # dinu
  with tf.variable_scope("tower"):
    result = model.create_model(model_input,
                                num_frames=num_frames,
                                vocab_size=reader.num_classes,
                                is_training=False)
    predictions = result["predictions"]

  if FLAGS.n_rounds > 1:
      print("### Using multiple Rouds! ###")
      for i in range(FLAGS.n_rounds - 1):
          with tf.variable_scope("tower", reuse=True):
              result = model.create_model(model_input,
                                          num_frames=num_frames,
                                          vocab_size=reader.num_classes,
                                          is_training=False)
              predictions += result["predictions"]
  predictions = predictions / FLAGS.n_rounds

  tf.summary.histogram("model_activations", predictions)
  '''
  if "loss" in result.keys():
      label_loss = result["loss"]
  else:
      label_loss = label_loss_fn.calculate_loss(predictions, labels_batch)
  '''
  tf.add_to_collection("global_step", global_step)
  #tf.add_to_collection("loss", label_loss)
  tf.add_to_collection("predictions", predictions)
  tf.add_to_collection("input_batch", model_input)
  tf.add_to_collection("input_batch_raw", model_input_raw)
  tf.add_to_collection("video_id_batch", video_id_batch)
  tf.add_to_collection("num_frames", num_frames)
  #tf.add_to_collection("labels", tf.cast(labels_batch, tf.float32))
  tf.add_to_collection("summary_op", tf.summary.merge_all())


def get_latest_checkpoint():
  index_files = glob.glob(os.path.join(FLAGS.train_dir, 'model.ckpt-*.index'))

  # No files
  if not index_files:
    return None


  # Index file path with the maximum step size.
  latest_index_file = sorted(
      [(int(os.path.basename(f).split("-")[-1].split(".")[0]), f)
       for f in index_files])[-1][1]

  # Chop off .index suffix and return
  return latest_index_file[:-6]


def evaluation_loop(video_id_batch, prediction_batch,
                    summary_op, saver, summary_writer, evl_metrics,
                    last_global_step_val, ema_tensors, model_input, video_id, num_frames, feats, id):
  """Run the evaluation loop once.

  Args:
    video_id_batch: a tensor of video ids mini-batch.
    prediction_batch: a tensor of predictions mini-batch.
    label_batch: a tensor of label_batch mini-batch.
    loss: a tensor of loss for the examples in the mini-batch.
    summary_op: a tensor which runs the tensorboard summary operations.
    saver: a tensorflow saver to restore the model.
    summary_writer: a tensorflow summary_writer
    evl_metrics: an EvaluationMetrics object.
    last_global_step_val: the global step used in the previous evaluation.

  Returns:
    The global_step used in the latest model.
  """

  global_step_val = -1
  latest_checkpoint = get_latest_checkpoint()

  with tf.Session() as sess:

    if latest_checkpoint:
      logging.info("Loading checkpoint for eval: " + latest_checkpoint)
      # Restores from checkpoint
      saver.restore(sess, latest_checkpoint)
      # Assuming model_checkpoint_path looks something like:
      # /my-favorite-path/yt8m_train/model.ckpt-0, extract global_step from it.
      global_step_val = os.path.basename(latest_checkpoint).split("-")[-1]

      if FLAGS.use_EMA:
        assert len(ema_tensors) > 0, "Tensors got lost."
        logging.info("####################")
        logging.info("USING EMA VARIABLES.")
        logging.info("####################")

        reader = pywrap_tensorflow.NewCheckpointReader(latest_checkpoint)
        global_vars = tf.global_variables()

        for stensor in ema_tensors:
          destination_t = [x for x in global_vars if x.name == stensor.replace("/ExponentialMovingAverage:", ":")]
          assert len(destination_t) == 1
          destination_t = destination_t[0]
          ema_source = reader.get_tensor(stensor.split(":")[0])
          # Session to take care of
          destination_t.load(ema_source, session=sess)

      # Save model
      logging.info("\nstart save inference_model to file:")
      saver.save(sess, os.path.join(FLAGS.train_dir, "inference_model"))
      logging.info("end save inference_model to file.\n")
      if FLAGS.build_only:
          logging.info("Inference graph built. Existing now.")
          #exit()
    else:
      logging.info("No checkpoint file found.")
      return global_step_val

    if global_step_val == last_global_step_val:
      logging.info("skip this checkpoint global_step_val=%s "
                   "(same as the previous one).", global_step_val)
      return global_step_val

    sess.run([tf.local_variables_initializer()])

    # Start the queue runners.
    fetches = [video_id_batch, prediction_batch, summary_op]

    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(
            sess, coord=coord, daemon=True,
            start=True))
      logging.info("enter eval_once loop global_step_val = %s. ", global_step_val)

      evl_metrics.clear()

      examples_processed = 0
      name = []
      while not coord.should_stop():
        batch_start_time = time.time()
        _, predictions_val, summary_val = sess.run(fetches, feed_dict={video_id: ['11'], model_input: feats, num_frames: [300]})
        idx = np.argsort(-predictions_val, axis=1)[:,:20]
        value = np.vstack(','.join([str(s) for s in predictions_val[i][idx[i]]]) for i in range(len(idx)))
        idx = np.vstack(','.join([str(s) for s in idx[i]]) for i in range(len(idx)))
        print(id)
        pred = pd.DataFrame({'id':id, 'idx':np.squeeze(idx), 'value':np.squeeze(value)})
        pred.to_csv('./data/results.csv', mode='a')
        break
    except tf.errors.OutOfRangeError as e:
      logging.info(
          "Done with batched inference. Now calculating global performance "
          "metrics.")

    except Exception as e:  # pylint: disable=broad-except
      logging.info("Unexpected exception: " + str(e))
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)

    return global_step_val


def evaluate():
  ema_tensors = None

  if FLAGS.use_EMA:
    latest_checkpoint = get_latest_checkpoint()
    assert latest_checkpoint, "No checkpoint found"

    with tf.device("/cpu:0"):
        saver = tf.train.import_meta_graph(latest_checkpoint + ".meta", clear_devices=True)
        # saver.restore(sess, "../trained_models/attention_frames_v0_EMA/model.ckpt-15512")
    xvars = tf.get_collection("ema_vars")
    assert len(xvars) > 0, "No EMA shadow variables found. Did you train with EMA?"
    ema_tensors = list(set([x.name for x in xvars]))
    tf.reset_default_graph()

  tf.set_random_seed(0)  # for reproducibility

  # Write json of flags
  model_flags_path = os.path.join(FLAGS.train_dir, "model_flags.json")
  if not os.path.exists(model_flags_path):
    raise IOError(("Cannot find file %s. Did you run train.py on the same "
                   "--train_dir?") % model_flags_path)
  flags_dict = json.loads(open(model_flags_path).read())
 
  num = 0
  #batch_size = 64 ---> dinu
  batch_size = 64
  train_data = data_loader(FLAGS, batch_size)

  #for _ in range(913): ## 58374/16 ---> dinu
  for _ in range(batch_size): ## 58374/16  
      feats, id = next(train_data)
      id = np.array(id)
      print(feats.shape, id.shape)
      tf.reset_default_graph()
      with tf.Graph().as_default():
        # convert feature_names and feature_sizes to lists of values
        feature_names, feature_sizes = utils.GetListOfFeatureNamesAndSizes(
            flags_dict["feature_names"], flags_dict["feature_sizes"])

        if flags_dict["frame_features"]:
          reader = readers.YT8MFrameFeatureReader(feature_names=feature_names,
                                                  feature_sizes=feature_sizes)
        else:
          reader = readers.YT8MAggregatedFeatureReader(feature_names=feature_names,
                                                      feature_sizes=feature_sizes)

        model = find_class_by_name(flags_dict["model"],
            [frame_level_models, video_level_models])()
        label_loss_fn = find_class_by_name(flags_dict["label_loss"], [losses])()

        if FLAGS.eval_data_pattern == "":
          raise IOError("'eval_data_pattern' was not specified. " + "Nothing to evaluate.")
        

        '''build_graph(
            reader=reader,
            model=model,
            eval_data_pattern=FLAGS.eval_data_pattern,
            label_loss_fn=label_loss_fn,
            num_readers=FLAGS.num_readers,
            batch_size=FLAGS.batch_size)''' #--> dinu 
        build_graph(
            reader=reader,
            model=model,
            eval_data_pattern=FLAGS.eval_data_pattern,
            label_loss_fn=label_loss_fn,
            num_readers=FLAGS.num_readers,
            batch_size=2)

        logging.info("built evaluation graph")
        
        video_id_batch = tf.get_collection("video_id_batch")[0]
        prediction_batch = tf.get_collection("predictions")[0]
        #label_batch = tf.get_collection("labels")[0]
        #loss = tf.get_collection("loss")[0]
        summary_op = tf.get_collection("summary_op")[0]
        #tf.add_to_collection("input_batch", model_input)
        model_input = tf.get_collection("input_batch")[0]
        video_id = tf.get_collection("video_id_batch")[0]
        num_frames = tf.get_collection("num_frames")[0]

        saver = tf.train.Saver(tf.global_variables())
        summary_writer = tf.summary.FileWriter(
            FLAGS.train_dir, graph=tf.get_default_graph())
        
        evl_metrics = eval_util.EvaluationMetrics(reader.num_classes, FLAGS.top_k)

        last_global_step_val = -1
        num += 1
        print('predicted:',(num-1)*batch_size)
        last_global_step_val = evaluation_loop(video_id_batch, prediction_batch, summary_op,
                                              saver, summary_writer, evl_metrics,
                                              last_global_step_val, ema_tensors, model_input, video_id, num_frames, feats, np.array(id))
  


def main(unused_argv):
  logging.set_verbosity(tf.logging.INFO)
  print("tensorflow version: %s" % tf.__version__)

  evaluate()


if __name__ == "__main__":
  app.run()
