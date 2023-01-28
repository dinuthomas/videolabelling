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

"""Contains a collection of util functions for model construction.
"""
import numpy
import tensorflow as tf
from tensorflow import logging
from tensorflow import flags
import tensorflow.contrib.slim as slim

from tensorflow import flags
FLAGS = flags.FLAGS
flags.DEFINE_bool("sample_all", False,
                  "Ensure that all frames are sampled.")

def SampleRandomSequence(model_input, num_frames, num_samples):
  """Samples a random sequence of frames of size num_samples.

  Args:
    model_input: A tensor of size batch_size x max_frames x feature_size
    num_frames: A tensor of size batch_size x 1
    num_samples: A scalar

  Returns:
    `model_input`: A tensor of size batch_size x num_samples x feature_size
  """

  batch_size = tf.shape(model_input)[0]
  frame_index_offset = tf.tile(
      tf.expand_dims(tf.range(num_samples), 0), [batch_size, 1])
  max_start_frame_index = tf.maximum(num_frames - num_samples, 0)
  start_frame_index = tf.cast(
      tf.multiply(
          tf.random_uniform([batch_size, 1]),
          tf.cast(max_start_frame_index + 1, tf.float32)), tf.int32)
  frame_index = tf.minimum(start_frame_index + frame_index_offset,
                           tf.cast(num_frames - 1, tf.int32))
  batch_index = tf.tile(
      tf.expand_dims(tf.range(batch_size), 1), [1, num_samples])
  index = tf.stack([batch_index, frame_index], 2)
  return tf.gather_nd(model_input, index)


def represent_all(model_input, sampled_frames, num_frames, max_frames=300):
    """
    Ensure that each frame is present at least once.
    """
    frame_lens = tf.reshape(num_frames, (-1,))

    mask_set = tf.tile(tf.expand_dims(tf.range(1, max_frames + 1), axis=1), [1, tf.shape(sampled_frames)[0]])
    mask = tf.transpose(tf.logical_not(tf.greater(mask_set, tf.cast(frame_lens, tf.int32))))

    conter_mask = tf.logical_not(mask)

    original = (tf.cast(mask, tf.float32) * tf.transpose(model_input, (2, 0, 1)))
    sampled = (tf.cast(conter_mask, tf.float32) * tf.transpose(sampled_frames, (2, 0, 1)))
    consensus = tf.transpose(original + sampled, (1, 2, 0))
    return consensus

def SampleRandomFrames(model_input, num_frames, num_samples):
  """Samples a random set of frames of size num_samples.

  Args:
    model_input: A tensor of size batch_size x max_frames x feature_size
    num_frames: A tensor of size batch_size x 1
    num_samples: A scalar

  Returns:
    `model_input`: A tensor of size batch_size x num_samples x feature_size
  """
  batch_size = tf.shape(model_input)[0]
  frame_index = tf.cast(
      tf.multiply(
          tf.random_uniform([batch_size, num_samples]),
          tf.tile(tf.cast(num_frames, tf.float32), [1, num_samples])), tf.int32)
  batch_index = tf.tile(
      tf.expand_dims(tf.range(batch_size), 1), [1, num_samples])
  index = tf.stack([batch_index, frame_index], 2)
  if FLAGS.sample_all:
    sampled_frames = tf.gather_nd(model_input, index)
    return represent_all(model_input, sampled_frames, num_frames)
  else:
    return tf.gather_nd(model_input, index)


def FramePooling(frames, method, **unused_params):
  """Pools over the frames of a video.

  Args:
    frames: A tensor with shape [batch_size, num_frames, feature_size].
    method: "average", "max", "attention", or "none".
  Returns:
    A tensor with shape [batch_size, feature_size] for average, max, or
    attention pooling. A tensor with shape [batch_size*num_frames, feature_size]
    for none pooling.

  Raises:
    ValueError: if method is other than "average", "max", "attention", or
    "none".
  """
  if method == "average":
    return tf.reduce_mean(frames, 1)
  elif method == "max":
    return tf.reduce_max(frames, 1)
  elif method == "none":
    feature_size = frames.shape_as_list()[2]
    return tf.reshape(frames, [-1, feature_size])
  else:
    raise ValueError("Unrecognized pooling method: %s" % method)


def shuffle_frames(inpt, n_frames):
    """
    Randomly permutes order of frames, taking into account length of the frames
    inpt: Tensor of frames
    n_frames : tensor of shape batch_size : number of frames for each sample
    """
    print("### FRAME SHUFFLING ENABLED ###")
    max_frames = 300  # maximum number of frames

    order_asign = tf.random_uniform(tf.shape(inpt)[:-1])  # batch x frames, to determine order
    mask_set = tf.tile(tf.expand_dims(tf.range(1, max_frames + 1), axis=1), [1, tf.shape(inpt)[0]])
    mask = tf.transpose(tf.logical_not(tf.greater(mask_set, n_frames)))

    svalues = tf.cast(mask, tf.float32) * order_asign  # sort by these values
    sort_order = tf.contrib.framework.argsort(svalues, direction="DESCENDING")

    inpt_flat = tf.reshape(inpt, (-1, tf.shape(inpt)[-1]))
    flat_loc = tf.tile(tf.expand_dims(tf.range(tf.shape(inpt)[0]), 1),
                       [1, max_frames])  # .eval({inpt: x, n_frames: x_nf})
    flat_loc = tf.reshape(flat_loc * tf.shape(inpt)[1] + sort_order, [-1])
    output = tf.reshape(tf.gather(inpt_flat, flat_loc), tf.shape(inpt))
    return output
