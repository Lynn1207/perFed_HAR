from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
import sys
import numpy as np
import tensorflow.compat.v1 as tf

import cnnHAR

num=1 #number of nodes

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir', '/home/ubuntu/perFed_HAR/CNN_Human_Activity_Recognition/cnnHAR_e0'+str(sys.argv[1]),
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/home/ubuntu/perFed_HAR/CNN_Human_Activity_Recognition/cnnHAR_check0'+str(sys.argv[1]),
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 64,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', True,
                         """Whether to run eval only once.""")

batch_size = 32
NUM_CLASSES = cnnHAR.NUM_CLASSES

def eval_once(is_loc, saver,summary_writer,labels,pre_soft, logits,loss,summary_op):
  
  """Run Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
  """
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/cnnHAR_train/model.ckpt-0,
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
      #print('~~~~~~~~~~~checkpoint file found at step %s'% global_step)
    else:
      print('No checkpoint file found')
      return
    
    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))
      
      num_iter = int(math.ceil(FLAGS.num_examples/batch_size))
      step = 0
      n_acc=0.0
      n_l=0.0
      simpleness=[]
      while step < num_iter and not coord.should_stop():
        step += 1
        #print(str(sys.argv[1])+'~~~~Local test:%d'%(step))
        samplelabels,kl_vec, predictions,precision=sess.run([labels,pre_soft, logits,loss])
        for i in range(0, batch_size):
          if int(samplelabels[i][0][0])==np.argmax(predictions[i]):
            #print(samplelabels[i][0][0], predictions[i], np.argmax(predictions[i]))
            n_acc+=1.0
            simpleness.append(kl_vec[i])
          else:
            simpleness.append(kl_vec[i])
          '''
          if predictions[i][int(samplelabels[i][0][0])]<0.1:
            #print('!!!!!!!!!!!!!!P(sample): ', predictions[i][int(samplelabels[i][0][0])] )
            n_l+=7.0
          else:
          '''
          if is_loc:
            n_l+=-math.log(max(0.0001,predictions[i][int(samplelabels[i][0][0])]))
      
      if is_loc:
        f = open("/home/ubuntu/perFed_HAR/CNN_Human_Activity_Recognition/results/log_test_"+cnnHAR.method+str(sys.argv[1])+".txt", "a")
        #x = time.strftime("%Y%m%d-%H%M%S")
        f.write("%.3f, %.3f\n"% (n_l/FLAGS.num_examples,n_acc/FLAGS.num_examples))
        f.close()
      else:
        f = open("/home/ubuntu/perFed_HAR/CNN_Human_Activity_Recognition/results/log_com_"+cnnHAR.method+str(sys.argv[1])+".txt", "a")
        for i in range(len(simpleness)):
          for j in range(NUM_CLASSES):
            f.write("%0.3f "%simpleness[i][j])
          f.write("\n")
        f.write("\n")
        f.close()
      #print(str(sys.argv[1])+'(locally test)!!!!!!!!!!!!!!!!!!!! average_test loss = %.3f, average_accuracy=%.3f' % (n_l/64,n_acc/64))
      
      summary = tf.Summary()
      summary.ParseFromString(sess.run(summary_op))
      summary.value.add(tag='loss @ 1', simple_value=precision)
      summary_writer.add_summary(summary, global_step)
    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)


def evaluate(is_loc):
  """Eval for a number of steps."""
  with tf.Graph().as_default() as g:
    # Get images and labels for CNNHAR
    
    signals, labels = cnnHAR.inputs(eval_data=is_loc)
    #print('~~~~shape of label:', labels.get_shape())
    # Build a Graph that computes the logits predictions from the

    [pre_soft, logits]=cnnHAR.inference(signals)

    loss=cnnHAR.loss(logits, labels)
  
    
    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        cnnHAR.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    #print('!!!!!!!!!!!!!!!!!!!variables to restore:')
    #print(variables_to_restore)
    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

    while True:
      eval_once(is_loc, saver,summary_writer,labels,pre_soft, logits,loss,summary_op)
      if FLAGS.run_once:
        break
      #time.sleep(FLAGS.eval_interval_secs)


def main(is_loc):  # pylint: disable=unused-argument
  if not tf.gfile.Exists(FLAGS.eval_dir):
    tf.gfile.MakeDirs(FLAGS.eval_dir)
  evaluate(is_loc)

if __name__ == '__main__':
  tf.app.run()
