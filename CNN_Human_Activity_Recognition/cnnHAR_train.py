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



max_steps = 30

log_device_placement = False

#log_frequency = 5# log per epoch

batch_size = cnnHAR.batch_size

NUM_CLASSES = cnnHAR.NUM_CLASSES

outer_iter=40 #local 8
  
def train():
  w_flat = np.array([])
  logLoss=[]
  logcomm=[]
  #loc_paras=[]
  #gen_paras=[]
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
  tf.get_logger().setLevel("ERROR")
  """Train CNNHAR for a number of steps."""
  with tf.Graph().as_default():
    
    global_step = tf.train.get_or_create_global_step()
    # Get images and labels for CNNHAR.
    # Force input pipeline to CPU:i to simulate federated setting
      
    # Build a Graph that computes the logits predictions from the
    # inference model.
    #training = tf.placeholder(tf.bool)
    signals, labels = cnnHAR.distorted_inputs()
    
    [pre_soft, logits]=cnnHAR.inference(signals)

    loss=cnnHAR.loss(logits, labels)
                                     
    [train_op,paras]= cnnHAR.train(loss, global_step)
  
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    
    W_avg1 = tf.compat.v1.placeholder(tf.float64, shape=(1664,))
    updated_paras1=cnnHAR.reset_var_l1(W_avg1)
    W_avg2 = tf.compat.v1.placeholder(tf.float64, shape=(75424,))
    updated_paras2=cnnHAR.reset_var_l2(W_avg2)
    W_avg3 = tf.compat.v1.placeholder(tf.float64, shape=(130912,))
    updated_paras3=cnnHAR.reset_var_l3(W_avg3)
    W_avg4 = tf.compat.v1.placeholder(tf.float64, shape=(131877,))
    updated_paras4=cnnHAR.reset_var_l4(W_avg4)
    
    # prepare the communication module
    server_addr = "localhost"
    server_port = 9999
    comm = COMM(server_addr,server_port,int(sys.argv[1]))
    
    
    comm.send2server('hello',-1)
    #print("Send Hello")
    comm.recvfserver()
    
    
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
        if (self._step+1)%(max_steps)==0 or self._step==0:
          logLoss.append([self._step, time.time()-self._start_time, run_values.results])
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
        if (self._step+1)%(max_steps)==0 or self._step==0:
          #print(self._step)
          paras_v=run_values.results
          cnnHAR_eval.main(True)
        if (self._step+1)%(max_steps)==0:#self._step+1==max_steps*outer_iter:
          #print("commonset")
          cnnHAR_eval.main(False)

    outer_i = 0
    start_iter=4 #6:20
    cur_layer=0
    intvl=0
    with tf.train.MonitoredTrainingSession(
          checkpoint_dir=train_dir,
          hooks=[tf.train.StopAtStepHook(last_step=max_steps*outer_iter+1),
                 #tf.train.NanTensorHook(loss),
                 _LoggerHook(),
                 #_LoggerHook2(),
                 _LoggerHook4()],#,save_checkpoint_steps=5000
          config=tf.ConfigProto(
              log_device_placement=log_device_placement),save_checkpoint_steps=(max_steps/8)) as mon_sess:
      while outer_i < outer_iter:
        step=0
        while step<max_steps and not mon_sess.should_stop():
          _, all_paras,_=mon_sess.run([train_op,paras, extra_update_ops])
          step+=1
          
        outer_i += 1
        
        intvl+=1 
        '''
        if outer_i>=start_iter:
          #get the weights and send to server
          w_flat = np.array([])
          #depends on how many layer wanna upload to server to share with other users
          #six layers: 2,4,6,8,10,11, or len(all_paras).
          
          if cur_layer<3 and start_iter==intvl:
            cur_layer=min(cur_layer+1,3)
            #print(cur_layer, start_iter)
            start_iter=int(start_iter*0.5)
            intvl=0
          
          for i in range(cur_layer*2):
            temp = all_paras[i].reshape(-1)
            w_flat=np.concatenate((w_flat, temp), axis=0)
            #if outer_i<2 and str(sys.argv[1])=="1":
              #print("Before_merge:", all_paras[i].shape)
            #print("after flatten%%%%%%%%%%%%", w_flat.shape)
          comm.send2server(w_flat,0)

          #receive aggregated weights from server
          W_general = comm.recvOUF()
          #w = tf.cast(W_general, tf.float64)
          #print(W_general.shape)
          logcomm.append([outer_i, w_flat.shape[0], W_general.shape[0]])
            
          if not mon_sess.should_stop():
            if cur_layer>=4:
              updated_paras_v=mon_sess.run(updated_paras4, feed_dict={W_avg4: W_general[0:131877]})
            elif cur_layer>=3:
              updated_paras_v=mon_sess.run(updated_paras3, feed_dict={W_avg3: W_general[0:130912]})
            elif cur_layer>=2:
              updated_paras_v=mon_sess.run(updated_paras2, feed_dict={W_avg2: W_general[0:75424]})
            elif cur_layer>=1:
              updated_paras_v=mon_sess.run(updated_paras1, feed_dict={W_avg1: W_general[0:1664]})
            #if str(sys.argv[1])=="1":
              #print("W_avg:", W_general[0:3])
              #print("After_merge:", updated_paras_v[0].reshape(-1)[0:3])
          #print("Length of updated paras: %d \n"% len(updated_paras_v))
         '''
          
        
        

    #log the train losses
    f = open("/home/ubuntu/perFed_HAR/CNN_Human_Activity_Recognition/results/log_"+cnnHAR.method+str(sys.argv[1])+".txt", "a")
    x = time.strftime("%Y%m%d-%H%M%S")
    f.write(str(sys.argv[1])+", "+x+":\n")
    for i in range(len(logLoss)):
      format_str = ("%d, %0.3f, %0.3f\n")
      f.write(format_str % ( logLoss[i][0], logLoss[i][1], logLoss[i][2]))
    f.close()
    #log the communication 
    f = open("/home/ubuntu/perFed_HAR/CNN_Human_Activity_Recognition/results/log_comm"+cnnHAR.method+str(sys.argv[1])+".txt", "a")
    x = time.strftime("%Y%m%d-%H%M%S")
    f.write(str(sys.argv[1])+", "+x+":\n")
    for i in range(len(logcomm)):
      format_str = ("%d, %d, %d\n")
      f.write(format_str % ( logcomm[i][0], logcomm[i][1], logcomm[i][2]))
    f.close()
    '''
    #debug~~~~~~~~~~
    #print("******************",w_flat.shape)
    f = open("/home/ubuntu/perFed_HAR/CNN_Human_Activity_Recognition/results/log_paras"+cnnHAR.method+str(sys.argv[1])+".txt", "a")
    for i in w_flat:
      f.write("%0.3f\n"%i)
    f.close()
    ''' 
              
      

      

def main(argv=None):  
  if tf.gfile.Exists(train_dir):
    tf.gfile.DeleteRecursively(train_dir)
  tf.gfile.MakeDirs(train_dir)
  train()


if __name__ == '__main__':
  tf.app.run()

