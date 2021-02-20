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

"""A binary to train CIFAR-10 using a single GPU.

Accuracy:
cifar10_train.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by cifar10_eval.py.

Speed: With batch_size 128.

System        | Step Time (sec/batch)  |     Accuracy
------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from communication import COMM
from datetime import datetime
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
import sys
import tensorflow.compat.v1 as tf
import numpy as np
from tensorflow.python import debug as tfdbg 
import cnnHAR
import cnnHAR_eval



train_dir = '/home/ubuntu/perFed_HAR/CNN_Human_Activity_Recognition/cnnHAR_check0'+str(sys.argv[1])

num=1 # number of nodes

max_steps = 7*450+1 #400 epoch

log_device_placement = False

log_frequency = 7*90 # log per epoch

batch_size = cnnHAR.batch_size

NUM_CLASSES = cnnHAR.NUM_CLASSES

outer_iter=2


	
def train():
  logLoss=[]
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
  tf.get_logger().setLevel("ERROR")
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default():
    global_step = tf.train.get_or_create_global_step()
    # Get images and labels for CIFAR-10.
    # Force input pipeline to CPU:0 to avoid operations sometimes ending up on
    # GPU and resulting in a slow down.
    #with tf.device('/cpu:'+str(int(sys.argv[1])-1)):
    signals, labels = cnnHAR.distorted_inputs()
    #print('<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>')
      
    # Build a Graph that computes the logits predictions from the
    # inference model.
    #training = tf.placeholder(tf.bool)
    
    logits=cnnHAR.inference(signals)

    loss=cnnHAR.loss(logits, labels)
                                     
    [train_op,paras]= cnnHAR.train(loss, global_step)
  
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    
    # prepare the communication module
    server_addr = "localhost"
    server_port = 9999
    comm = COMM(server_addr,server_port,int(sys.argv[1]))
    
    
    comm.send2server('hello',-1)
    #print("Send Hello")
    print(comm.recvfserver())
    
    
    class _LoggerHook(tf.train.SessionRunHook):
      """Logs loss and runtime."""

      def begin(self):
        self._step = -1
        self._start_time = time.time()

      def before_run(self, run_context):
        self._step += 1
        #print('~~~~~~~~~~~~~~~~before run1~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        #tmp = tf.concat([labels,signals],1)
        return tf.train.SessionRunArgs(loss)# Asks for loss value.

      def after_run(self, run_context, run_values):
        if (self._step-1) % (log_frequency)==0:
          logLoss.append([self._step, run_values.results])
          #format_str = ('*'*3*(int(sys.argv[1])-1)+':step %d=%0.3f')
          #print(format_str % ( self._step, run_values.results))
         
    class _LoggerHook2(tf.train.SessionRunHook):
      """Logs signals."""

      def begin(self):
        self._step = -1

      def before_run(self, run_context):
        self._step += 1
        #print('~~~~~~~~~~~~~~~~before run2~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        return tf.train.SessionRunArgs(logits)  # Asks for logits.

      def after_run(self, run_context, run_values):
        if self._step == max_steps-1:#:
          print('~~~~~~~~~~~~~~~~after run2~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
          ndar = np.array(run_values.results)
          np.savetxt("logits"+str(self._step)+".csv", ndar.reshape(batch_size,NUM_CLASSES), delimiter=",")

    class _LoggerHook3(tf.train.SessionRunHook):
      """Logs labels."""

      def begin(self):
        self._step = -1

      def before_run(self, run_context):
        self._step += 1
        return tf.train.SessionRunArgs(labels)  # Asks for labels.

      def after_run(self, run_context, run_values):
        if self._step == max_steps-1:
          ndar = np.array(run_values.results)
          np.savetxt("labels"+str(self._step)+".csv", ndar.reshape(batch_size,NUM_CLASSES), delimiter=",")

    class _LoggerHook4(tf.train.SessionRunHook):
      """Logs signals."""

      def begin(self):
        self._step = -1

      def before_run(self, run_context):
        self._step += 1
        #print('~~~~~~~~~~~~~~~~before run4~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        return tf.train.SessionRunArgs(paras)  # Asks for signals.

      def after_run(self, run_context, run_values):
        if self._step == max_steps-1:
          paras_v=run_values.results
          cnnHAR_eval.main()

    outer_i = 0
    with tf.train.MonitoredTrainingSession(
          checkpoint_dir=train_dir,
          hooks=[tf.train.StopAtStepHook(last_step=max_steps*outer_iter+1),
                 #tf.train.NanTensorHook(loss),
                 _LoggerHook(),
                 #_LoggerHook2(),
                 _LoggerHook4()],#,save_checkpoint_steps=5000
          config=tf.ConfigProto(
              log_device_placement=log_device_placement),save_checkpoint_steps=log_frequency) as mon_sess:
      while outer_i < outer_iter:
        step=0
      
        while step<max_steps and not mon_sess.should_stop():
          _,all_paras,_=mon_sess.run([train_op,paras,extra_update_ops])
          step+=1

        #get the weights and send to server
        w_flat = np.array([])
        for i in range(len(all_paras)):
          temp = all_paras[i].reshape(-1)
          w_flat=np.concatenate((w_flat, temp), axis=0)
          
        comm.send2server(w_flat,0)
      
        #receive aggregated weights from server
        W_avg = comm.recvOUF()
        W_avg = W_avg.astype(np.float32)
        #assign_model(W_avg)
        with tf.variable_scope('conv1') as scope:
          mon_sess.run(tf.assign(tf.get_variable('weights1'), tf.reshape(W_avg[0:2048],[32, 1, 64]))
          mon_sess.run(tf.assign(tf.get_variable('biases1'), W_avg[2048:2112]))
        with tf.variable_scope('conv2') as scope:
          mon_sess.run(tf.assign(tf.get_default_graph().get_tensor_by_name('weights2'), tf.reshape(W_avg[2112:8256],[3, 64, 32])))
          mon_sess.run(tf.assign(tf.get_default_graph().get_tensor_by_name('biases2'), W_avg[8256:8288]))
        with tf.variable_scope('local2') as scope:
          mon_sess.run(tf.assign(tf.get_default_graph().get_tensor_by_name('weights3'), tf.reshape(W_avg[8288:73824],[64, 1024])))
          mon_sess.run(tf.assign(tf.get_default_graph().get_tensor_by_name('biases3'), W_avg[73824:74848]))
        with tf.variable_scope('local3') as scope:
          mon_sess.run(tf.assign(tf.get_default_graph().get_tensor_by_name('weights4'), tf.reshape(W_avg[74848:599136],[1024, 512])))
          mon_sess.run(tf.assign(tf.get_default_graph().get_tensor_by_name('biases4'), W_avg[599136:599648]))
        with tf.variable_scope('local4') as scope:
          mon_sess.run(tf.assign(tf.get_default_graph().get_tensor_by_name('weights5'), tf.reshape(W_avg[599648:615008],[512, 30])))
          mon_sess.run(tf.assign(tf.get_default_graph().get_tensor_by_name('biases5'), W_avg[615008:615038]))
        with tf.variable_scope('softmax_linear') as scope:
          mon_sess.run(tf.assign(tf.get_default_graph().get_tensor_by_name('weights6'), tf.reshape(W_avg[615038:615218],[30, 6])))
          mon_sess.run(tf.assign(tf.get_default_graph().get_tensor_by_name('biases6'), W_avg[615218:615224]))
        
        outer_i += 1

    #log the train losses
    f = open("log"+str(sys.argv[1])+".txt", "a")
    f.write(str(sys.argv[1])+" train_loss:\n")
    for i in range(len(logLoss)):
      format_str = ("%d=%0.3f\n")
      f.write(format_str % ( logLoss[i][0], logLoss[i][1]))
    f.close()
      
              
      

      

def main(argv=None):  # pylint: disable=unused-argument
#  cifar10.maybe_download_and_extract()
  if tf.gfile.Exists(train_dir):
    tf.gfile.DeleteRecursively(train_dir)
  tf.gfile.MakeDirs(train_dir)
  train()


if __name__ == '__main__':
  tf.app.run()

