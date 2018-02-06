import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import evaluation.nn_eval as nn_eval
import evaluation.nn_eval_activity as nn_eval_activity
import evaluation.nn_eval_multitask as nn_eval_multitask
import os 

class Trainer():
    
    def __init__(self,
                 model,
                 data_train, ground_truth_train,
                 data_validation, ground_truth_validation,
                 sess,
                 summary_path,
                 batch_size,
                 checkpoint_path=None,
                 global_step=0,
                 eval_type='change'):
        
        
        #checks if the summary path is exist or not, if so delete it.
        try:
            os.remove(summary_path)
        except OSError:
            pass
        
        self.model = model
        self.data_train = data_train
        self.ground_truth_train = ground_truth_train
        self.data_validation = data_validation
        self.ground_truth_validation = ground_truth_validation
        self.sess = sess
        self.batch_size = batch_size        
        self.checkpoint_path = checkpoint_path
        
        self.summary_writer = tf.summary.FileWriter(summary_path, self.sess.graph)
        self.num_batch = np.ceil(float(self.data_train.shape[0]) / float(self.batch_size))
        
        self.saver = tf.train.Saver()
        
        # Initializing the variables
        init = tf.global_variables_initializer()
     
        self.summary = tf.summary.merge_all()
        self.sess.run(init)
        
        self.global_step = global_step
        
        if eval_type == 'activity':
            self.evaluate_model = nn_eval_activity.evaluate_model
        elif eval_type == 'change':
            self.evaluate_model = nn_eval.evaluate_model
        elif eval_type == 'multitask':
            self.evaluate_model = nn_eval_multitask.evaluate_model
        else:
            raise
    
    def run_debug_ops(self):       
        batch_idx = self.global_step % self.num_batch
        data_idx = np.arange (batch_idx * self.batch_size,
                              min ((batch_idx+1) * self.batch_size,
                                   self.data_train.shape[0])).astype('int32')

        batch_x = self.data_train[data_idx]
        batch_y = self.ground_truth_train[data_idx]
        debug_info = self.sess.run(self.model.debug_info,
                                   feed_dict={self.model.x: batch_x,
                                              self.model.y: batch_y})
    
        print debug_info
        
        
        
    def train(self,
              max_iter,
              train_eval_step,
              validation_eval_step,
              display_step,
              checkpoint_step = None):
        
        # Keep training until reach max iterations
        for _ in range (max_iter):
            if self.global_step % self.num_batch == 0:
                self.data_train ,self.ground_truth_train = self.shuffle(self.data_train, self.ground_truth_train) 
       
            batch_idx = self.global_step % self.num_batch
            data_idx = np.arange (batch_idx * self.batch_size,
                                  min ((batch_idx+1) * self.batch_size,
                                       self.data_train.shape[0])).astype('int32')
            
            batch_x = self.data_train[data_idx]
            batch_y = self.ground_truth_train[data_idx]

            if self.global_step % train_eval_step == 0:
                
                print("Train Batch Evaluation")
                eval_result = self.evaluate_model(self.model, self.sess, batch_x, batch_y)                
                
                #summary np.array
                auc_summary_str_train = tf.Summary(value = [
                        tf.Summary.Value(tag = 'train/'+key, simple_value = eval_result[key])
                        for key in eval_result])
                
                self.summary_writer.add_summary(auc_summary_str_train, global_step = self.global_step)
                self.summary_writer.flush()
            
  
            if self.global_step % validation_eval_step == 0:
                
                print("Validation")
                eval_result = self.evaluate_model(self.model, self.sess, self.data_validation, self.ground_truth_validation)
                                
                #summary np.array
                auc_summary_str_valid = tf.Summary(value = [
                        tf.Summary.Value(tag = 'validation/'+key, simple_value = eval_result[key])
                        for key in eval_result])
                
                self.summary_writer.add_summary(auc_summary_str_valid, global_step = self.global_step)
                self.summary_writer.flush()
            
                
            #optimization part
            _, summary_str, loss, debug_info = self.sess.run((self.model.optimizer, self.summary, self.model.cost, self.model.debug_info),
                                           feed_dict={self.model.x: batch_x,
                                                      self.model.y: batch_y})
            
            #print debug_info
            
            self.summary_writer.add_summary(summary_str, global_step = self.global_step)
            self.summary_writer.flush()
            
            if self.global_step % display_step == 0:
                print "Minibatch Loss= {:.6f}".format(loss)
   
            
            if checkpoint_step is not None and self.global_step % checkpoint_step == 0:
                self.save_model(self.checkpoint_path)

            self.global_step += 1
        print " Optimization Finished!"
    
    
    def save_model(self, path):
        self.saver.save(sess, path, global_step=self.global_step)
        
      
    def shuffle(self, data, gt):
    
        """
        This function shuffle data and ground truth 
        """
       
        
        idx = np.arange(data.shape[0])
        np.random.shuffle(idx)

        data = data[idx]
        gt = gt[idx]

        return data,gt 
            
            
            
            
            
            
                
        
        
        
        
        
        
    