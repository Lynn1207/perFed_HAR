# -*- coding: utf-8 -*-
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""Routine for decoding the CIFAR-10 binary file format."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow.compat.v1 as tf

# Process sensing data "image" of this size, 128*6.
# each chanel like acc_x is 128 length which is sata collected for 2.56s. 

SIGNAL_SIZE=40
axis=40
channels=1

batch_per_user_train=7
batch_per_user_test=2
# Global constants describing the cnnHAR data set.
NUM_CLASSES = 5
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 32*batch_per_user_train
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 32*batch_per_user_test

def read_cnnHAR(filename_queue):

  class CNNHARRecord(object):
    pass
  result = CNNHARRecord()
  
  # Read a record, getting filenames from the filename_queue.  No
  # header or footer in the CIFAR-10 format, so we leave header_bytes
  # and footer_bytes at their default of 0.
  reader = tf.TextLineReader()
  result.key, value = reader.read(filename_queue)
  
  # Convert from a string to a vector of uint8 that is record_bytes long.
  record_defaults = [[1.0] for col in range(SIGNAL_SIZE*axis*channels+1)]# +2 as the col0: subject_id, col1: label
  
  record_bytes = tf.decode_csv(value, record_defaults = record_defaults)
  # The first bytes represent the label, which we convert from uint8->int32.
  if int(sys.argv[1])<4
    result.signal = tf.cast(tf.strided_slice(record_bytes, [1], [SIGNAL_SIZE*axis*channels+1])/255*0.6, tf.float32)
  elif int(sys.argv[1])<7
    result.signal = tf.cast(tf.strided_slice(record_bytes, [1], [SIGNAL_SIZE*axis*channels+1])/255*0.35, tf.float32)
  else:
    result.signal = tf.cast(tf.strided_slice(record_bytes, [1], [SIGNAL_SIZE*axis*channels+1])/255, tf.float32)
  #print('!!!!!!!!!!!!!!!!!!! result.signals', result.signal.get_shape())
  result.signal = tf.reshape(result.signal, [channels, axis, SIGNAL_SIZE])
  #print('!!!!!!!!!!!!!!!!!!! result.signals', result.signal.get_shape())
  # labels-1 cause the logits is defaulted to start with 0~NUM_CLASS-1
  result.label = tf.cast(
      tf.strided_slice(record_bytes, [0], [1])-1, tf.float32)
  #print('!!!!!!!!!!!!!!!!!!! result.label before reshape', result.label)
  result.label = tf.reshape(result.label, [1, 1])
    
  return result



def _generate_image_and_label_batch(signal, label, min_queue_examples,
                                    batch_size, shuffle):
  #print('????????? signal shape BEFORE batch', signal.get_shape())
  num_preprocess_threads = 1
  if shuffle:
    signals, label_batch= tf.train.shuffle_batch(
        [signal, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
  else:
    signals, label_batch= tf.train.batch(
              [signal, label],
              batch_size=batch_size,
              num_threads=num_preprocess_threads,
              capacity=min_queue_examples + 3 * batch_size)
    #print('????????? signal shape AFTER batch reshape', signals.get_shape())
  return signals, label_batch #tf.reshape(label_batch, [batch_size, SIGNAL_SIZE, 1])

def distorted_inputs(data_dir, batch_size):
  """Construct distorted input for CIFAR training using the Reader ops.

  Args:
    data_dir: Path to the CIFAR-10 data directory.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  filename = [os.path.join(data_dir, str(sys.argv[1])+'_train.csv')]
  #if not tf.io.gfile.exists(filename):
    #raise ValueError('Failed to find file: ' + filename)

  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer(filename)
  
  with tf.name_scope('data_augmentation'):
    # Read examples from files in the filename queue.
    read_input = read_cnnHAR(filename_queue)
    signal = tf.transpose(read_input.signal, (2,1,0)) # Singals * numofAxis * channel
    read_input.label.set_shape([1, 1])
    #print('?????????? shape of  the singals:', signal.get_shape())
    
    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             min_fraction_of_examples_in_queue)
    #print ('Filling queue with %d acc_frames before starting to train. '
    #'This will take a few minutes.' % min_queue_examples)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(signal, read_input.label,
                                         min_queue_examples, batch_size,
                                         shuffle=True)

def inputs(eval_data, data_dir, batch_size):

  if not eval_data:
    filenames = [os.path.join(data_dir, 'commset.csv')]
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
  else:
    filenames = [os.path.join(data_dir, str(sys.argv[1])+'_test.csv')]
    num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

  #if not tf.io.gfile.exists(filenames):
      #raise ValueError('Failed to find file: ' + filenames)

  with tf.name_scope('input'):
    # Create a queue that produces the filenames to read.
    filename_queue = tf.train.string_input_producer(filenames)

    read_input = read_cnnHAR(filename_queue)
    signal = tf.transpose(read_input.signal, (2,1,0)) # Singals * numofAxis * channel
    read_input.label.set_shape([1, 1])
    
    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch *
                             min_fraction_of_examples_in_queue)
    #print ('Filling queue with %d acc_frames before starting to test. '
    #                         'This will take a few minutes.' % min_queue_examples)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(signal, read_input.label,
                                         min_queue_examples, batch_size,
                                         shuffle=False)

