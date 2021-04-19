from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import re
import sys
import tarfile
import random

from six.moves import urllib
import tensorflow.compat.v1 as tf
import numpy as np

import cnnHAR_input

#2 baselines, our method: fedper
method="local" #"local", "FedPer"
cur_l=5
num_paras= 200293#l1: 1664; l2: 52896; l3: 163872, l4: 213152; l5: 213797

# Basic model parameters.
batch_size = 32
                          
data_dir = '/home/ubuntu/perFed_HAR/CNN_Human_Activity_Recognition/dataset/'
                    

# Global constants describing the CIFAR-10 data set.
SIGNAL_SIZE = cnnHAR_input.SIGNAL_SIZE
NUM_CLASSES = cnnHAR_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cnnHAR_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cnnHAR_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL


# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999     # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY =100.0     # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.96  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.01     # Initial learning rate.

# If a model is trained with multiple GPUs, prefix all Op names with tower_name
# to differentiate the operations. Note that this prefix is removed from the
# names of the summaries when visualizing a model.
TOWER_NAME = 'tower'

def distorted_inputs():
  """Construct distorted input for CIFAR training using the Reader ops.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.

  Raises:
    ValueError: If no data_dir
  """
  global data_dir
  data_dir = data_dir
  signals, labels= cnnHAR_input.distorted_inputs(data_dir=data_dir,batch_size=batch_size)
  signals = tf.cast(signals, tf.float64)
  labels = tf.cast(labels, tf.float64)
  return signals, labels

 
def inputs(eval_data):
    global data_dir
    data_dir = data_dir
    signals, labels = cnnHAR_input.inputs(eval_data ,data_dir=data_dir,batch_size=batch_size)
    signals = tf.cast(signals, tf.float64)
    labels = tf.cast(labels, tf.float64)
    return signals, labels
  
def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.
  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
  Returns:
    Variable Tensor
  """
  dtype = tf.float64
  var = _variable_on_cpu(
      name,
      shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
  #print(name, var)
  
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  
  return var
  
def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
  
  dtype = tf.float64
  var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var
  
def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measures the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
  # session. This helps the clarity of presentation on tensorboard.
  tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
  tf.summary.histogram(tensor_name + '/activations', x)
  tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def _add_loss_summaries(total_loss):
  """Add summaries for losses in CIFAR-10 model.

  Generates moving average for all losses and associated summaries for
  visualizing the performance of the network.

  Args:
    total_loss: Total loss from loss().
  Returns:
    loss_averages_op: op for generating moving averages of losses.
  """
  # Compute the moving average of all individual losses and the total loss.
  loss_averages = tf.train.ExponentialMovingAverage(0.9999, name='avg')
  losses = tf.get_collection('losses')
  loss_averages_op = loss_averages.apply(losses + [total_loss])
  tf.add_to_collection('losses', total_loss)

  return loss_averages_op
  
def inference(signals):
    #print('<<<<<<<<<<<<<<<<<<<<Shape of singals :',signals.get_shape())
    with tf.variable_scope('conv1') as scope:
           kernel = _variable_with_weight_decay('weights1',
                                                shape=[32, 3, 1, 64],
                                                #shape=[3, 1, 128],
                                                stddev=0.04,
                                                wd=0.009)
           biases = _variable_on_cpu('biases1', [64], tf.constant_initializer(0.0))#!!!
           conv = tf.nn.conv2d(signals, kernel, [1,4,3,1], padding='VALID', data_format='NHWC')
           pre_activation = tf.layers.batch_normalization(tf.nn.bias_add(conv, biases))
           #pre_activation= tf.layers.batch_normalization(tf.nn.bias_add(conv, biases))
           conv1 = tf.nn.relu(pre_activation, name=scope.name)
           #_activation_summary(conv1)
           #print ('<<<<<<<<<<<<<<<<<<<<Shape of conv1 :',conv1.get_shape())
    pool1 = tf.nn.max_pool2d(conv1, ksize=[1,1,1,1], strides=[1,1,1,1],padding='VALID',name='pool1')
    #print ('<<<<<<<<<<<<<<<<<<<<Shape of pool1 :',pool1.get_shape())
    """18x3x64"""
   
    with tf.variable_scope('conv2') as scope:
           kernel = _variable_with_weight_decay('weights2',
                                                shape=[ 4, 1, 64, 32],
                                                #shape=[3, 1, 128],
                                                stddev=0.04,
                                                wd=0.009)
           biases = _variable_on_cpu('biases2', [32], tf.constant_initializer(0.0))#!!!
           conv = tf.nn.conv2d(pool1, kernel, [1,2,1,1], padding='VALID', data_format='NHWC')
           pre_activation=tf.layers.batch_normalization(tf.nn.bias_add(conv, biases))
           conv2 = tf.nn.relu(pre_activation, name=scope.name)
           #_activation_summary(conv2)
           #print ('<<<<<<<<<<<<<<<<<<<<Shape of conv2:',conv2.get_shape()) (8*3*32)
    pool2 = tf.nn.max_pool2d(conv2, ksize=[1,4,1, 1], strides=[1,2,1,1],padding='VALID',name='pool2')
    #print ('<<<<<<<<<<<<<<<<<<<<Shape of pool2 :',pool2.get_shape()) 
    reshape = tf.keras.layers.Flatten()(pool2)
    
    reshape = tf.cast(reshape, tf.float64)
    """32x3x3x32: 32x288"""
    #print ('<<<<<<<<<<<<<<<<<<<<Shape of reshape :',reshape.get_shape()[0], reshape.get_shape()[1])
    dim = reshape.get_shape()[1] 
    
    with tf.variable_scope('local2') as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        weights = _variable_with_weight_decay('weights3', shape=[dim, 384],
                                              stddev=0.04, wd=0.009)
        biases = _variable_on_cpu('biases3', [384], tf.constant_initializer(0.10))
        
        local2 = tf.nn.relu(tf.layers.batch_normalization(tf.matmul(reshape, weights) + biases, name=scope.name))
        #print ('!!!!!!!!!!!!!!!Shape of local2 :', local2.get_shape())
        #_activation_summary(local2)

    with tf.variable_scope('local3') as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        weights = _variable_with_weight_decay('weights4', shape=[384, 192],
                                              stddev=0.04, wd=0.009)#0.004,index)
        biases = _variable_on_cpu('biases4', [192], tf.constant_initializer(0.00))
        
        local3 = tf.nn.relu(tf.layers.batch_normalization(tf.matmul(local2, weights) + biases, name=scope.name))
        #print ('!!!!!!!!!!!!!!!Shape of local3 :', local3.get_shape())
        #_activation_summary(local3)
    '''
    with tf.variable_scope('local4') as scope:
        weights = _variable_with_weight_decay('weights5', shape=[512, 128], stddev=0.04, wd=0.009)
        biases = _variable_on_cpu('biases5', [128], tf.constant_initializer(0.00))
            
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)
        #print ('!!!!!!!!!!!!!!!Shape of local4 :', local4.get_shape())#256
        _activation_summary(local4)
    '''
    with tf.variable_scope('softmax_linear') as scope:
          weights = _variable_with_weight_decay('weights5', [192, NUM_CLASSES],stddev=0.04, wd=0.009)
          biases = _variable_on_cpu('biases5', [NUM_CLASSES],tf.constant_initializer(0.0))
          softmax_linear = tf.nn.softmax(tf.layers.batch_normalization(tf.matmul(local3, weights)+biases,name=scope.name))
          #_activation_summary(softmax_linear)
          #print ('!!!!!!!!!!!!!!!Shape of softmax_linear :', softmax_linear.get_shape())
    
    return softmax_linear
    

def loss(logits, labels):
    labels = tf.cast(labels, tf.int64)
    labels = tf.reshape(labels, [batch_size,1])
    logits = tf.reshape(logits, [batch_size,1,NUM_CLASSES])
    i=0
    loss=0.0
    
    while i<batch_size:
        loss+=-tf.math.log(logits[i,0,labels[i,0]])
        i+=1
    loss=loss/32.0
    
    #print('loss@@@@@@@@@@@@##############',loss)
    '''
    tf.add_to_collection('losses'+index, loss)
    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    total_loss=tf.add_n(tf.get_collection('losses'+index),name='total_loss')
    '''
    return loss
    
    
def train(total_loss, global_step):#index is a string e.g. '_1'
 
 num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / batch_size
 decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)
 # Decay the learning rate exponentially based on the number of steps.
 lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                global_step,
                                decay_steps,#10000,
                                LEARNING_RATE_DECAY_FACTOR,
                                staircase=True)
 #tf.summary.scalar('learning_rate', lr)

 ###### Record the parameters
 '''
 pre_paras=[]
 for var in tf.trainable_variables():
  pre_paras.append(var)
 '''        
 # Generate moving averages of all losses and associated summaries.
 loss_averages_op = _add_loss_summaries(total_loss)
 
  
 # Compute gradients.
 with tf.control_dependencies([loss_averages_op]):
  opt =tf.train.MomentumOptimizer(lr,0.9)#tf.train.AdadeltaOptimizer(lr)
  grads = opt.compute_gradients(total_loss)
  #for i in range(0,len(grads)):
    #print(i)
    #print('<<<<<<<<<<<<<<<<< shape of grads:',grads[i])

# Apply gradients.
 apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

# Track the moving averages of all trainable variables.
 variable_averages = tf.train.ExponentialMovingAverage(
    MOVING_AVERAGE_DECAY, global_step)
 with tf.control_dependencies([apply_gradient_op]):
  variables_averages_op = variable_averages.apply(tf.trainable_variables())
  
 with tf.control_dependencies([variables_averages_op]):
  ###### Record the parameters
  paras=[]
  for var in tf.trainable_variables():
    if var.op.name.find('batch_normalization') == -1 and var.op.name.find('gamma') == -1 and var.op.name.find('beta') == -1:
      paras.append(var)
      #if str(sys.argv[1])=="1":
        #print("Before_merge:", var.op.name)
    
 return  variables_averages_op, paras



def reset_var(W_avg):

  updated_paras=[]
  '''
  for var in tf.trainable_variables():
    if cur_l>0:
      if var.op.name=="conv1/weights1":
        var=tf.assign(var, tf.reshape(W_avg[0:1600],[5, 5, 1, 64]))
        updated_paras.append(var)
      elif var.op.name=="conv1/biases1":
        var=tf.assign(var, tf.reshape(W_avg[1600:1664],[64,]))
        updated_paras.append(var)
      if cur_l>1:
        if var.op.name=="conv2/weights2":
          var=tf.assign(var,tf.reshape(W_avg[1664:52864],[5, 5, 64, 32]))
          updated_paras.append(var)
        elif var.op.name=="conv2/biases2":
          var=tf.assign(var, W_avg[52864:52896])
          updated_paras.append(var)
        if cur_l>2:
          if var.op.name=="local2/weights3":
            var=tf.assign(var,tf.reshape(W_avg[52896:163488],[288, 384]))
            updated_paras.append(var)
          elif var.op.name=="local2/biases3":
            var=tf.assign(var, W_avg[163488:163872])
            updated_paras.append(var)
          if cur_l>3:
            if var.op.name=="local3/weights4":
              var=tf.assign(var,tf.reshape(W_avg[163872:213024],[384, 128]))
              updated_paras.append(var)
            elif var.op.name=="local3/biases4":
              var=tf.assign(var, W_avg[213024:213152])
              updated_paras.append(var)
            if cur_l>4:
              if var.op.name=="softmax_linear/weights6":
                var=tf.assign(var,tf.reshape(W_avg[213152:213792],[128, 5]))
                updated_paras.append(var)
              elif var.op.name=="softmax_linear/biases6":
                var=tf.assign(var, W_avg[213792:213797])
                updated_paras.append(var)
    
    
    #print(var)
  '''
  for var in tf.trainable_variables():
    if cur_l>0:
      if var.op.name=="conv1/weights1":
        var=tf.assign(var, tf.reshape(W_avg[0:6144],[32, 3, 1, 64]))
        updated_paras.append(var)
      elif var.op.name=="conv1/biases1":
        var=tf.assign(var, tf.reshape(W_avg[6144:6208],[64,]))
        updated_paras.append(var)
      if cur_l>1:
        if var.op.name=="conv2/weights2":
          var=tf.assign(var,tf.reshape(W_avg[6208:14400],[4, 1, 64, 32]))
          updated_paras.append(var)
        elif var.op.name=="conv2/biases2":
          var=tf.assign(var, W_avg[14400:14432])
          updated_paras.append(var)
        if cur_l>2:
          if var.op.name=="local2/weights3":
            var=tf.assign(var,tf.reshape(W_avg[14432:125024],[288, 384]))
            updated_paras.append(var)
          elif var.op.name=="local2/biases3":
            var=tf.assign(var, W_avg[125024:125408])
            updated_paras.append(var)
          if cur_l>3:
            if var.op.name=="local3/weights4":
              var=tf.assign(var,tf.reshape(W_avg[125408:199136],[384, 192]))
              updated_paras.append(var)
            elif var.op.name=="local3/biases4":
              var=tf.assign(var, W_avg[199136:199328])
              updated_paras.append(var)
            if cur_l>4:
              if var.op.name=="softmax_linear/weights5":
                var=tf.assign(var,tf.reshape(W_avg[199328:200288],[192, 5]))
                updated_paras.append(var)
              elif var.op.name=="softmax_linear/biases5":
                var=tf.assign(var, W_avg[200288:200293])
                updated_paras.append(var)
              
  
  '''                
  with tf.variable_scope('conv1') as scope:
    weights1=tf.get_variable('weights1')
    weights1.assign(tf.reshape(W_avg[0:2048],[32, 1, 64]))
    bias1=tf.assign(tf.get_variable('biases1'), W_avg[2048:2112])
  with tf.variable_scope('conv2') as scope:
    weights2=tf.assign(tf.get_default_graph().get_tensor_by_name('weights2'), tf.reshape(W_avg[2112:8256],[3, 64, 32]))
    bias2=tf.assign(tf.get_default_graph().get_tensor_by_name('biases2'), W_avg[8256:8288])
  with tf.variable_scope('local2') as scope:
    weights3=tf.assign(tf.get_default_graph().get_tensor_by_name('weights3'), tf.reshape(W_avg[8288:73824],[64, 1024]))
    bias3=tf.assign(tf.get_default_graph().get_tensor_by_name('biases3'), W_avg[73824:74848])
  with tf.variable_scope('local3') as scope:
    weights4=tf.assign(tf.get_default_graph().get_tensor_by_name('weights4'), tf.reshape(W_avg[74848:599136],[1024, 512]))
    bias4=tf.assign(tf.get_default_graph().get_tensor_by_name('biases4'), W_avg[599136:599648])
  with tf.variable_scope('local4') as scope:
    weights5=tf.assign(tf.get_default_graph().get_tensor_by_name('weights5'), tf.reshape(W_avg[599648:615008],[512, 30]))
    bias5=tf.assign(tf.get_default_graph().get_tensor_by_name('biases5'), W_avg[615008:615038])
  with tf.variable_scope('softmax_linear') as scope:
    weights6=tf.assign(tf.get_default_graph().get_tensor_by_name('weights6'), tf.reshape(W_avg[615038:615218],[30, 6]))
    bias6=tf.assign(tf.get_default_graph().get_tensor_by_name('biases6'), W_avg[615218:615224])
  '''

  return updated_paras
  
