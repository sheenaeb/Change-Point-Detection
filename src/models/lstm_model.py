import tensorflow as tf
import numpy as np

def conv1d(x, w, b, strides = 1, activation = None):
    x = tf.nn.conv1d(x, w, stride=1, padding ='SAME')

    if b is not None:
        
        x = tf.nn.bias_add(x,b)
    if activation is not None:
        x = activation(x)
    return x

    
def maxpool1d(x, k =2):
    temp = tf.expand_dims(x, 1)
    out = tf.nn.max_pool(temp, ksize=[1, 1, k, 1], strides = [1, 1, k ,1],
                         padding = 'SAME')
    return tf.squeeze(out,1)
    
    
class Model:
    
    
    def __init__(self,
                 n_variables,
                 n_classes):
    
        # tf Graph input
        self.x = tf.placeholder("float32", [None, None, n_variables])
        self.y = tf.placeholder("float32", [None, None, n_classes])  

        
    def build_lstm_net(self,
                       lstm_n_hidden,
                       n_classes,
                       learning_rate,
                       pos_weight):

        #RNN
        ## TODO: use "sequence_length"
        
        # Define a lstm cell with tensorflow
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_n_hidden)    #, forget_bias=1.0)

        #make a while loop for lstm
        outputs, _ = tf.nn.dynamic_rnn(lstm_cell, self.x, dtype=tf.float32)   #output shape is [batch_size,t, lstm_n_hidden]

         # Define weights
        weights = {
            'out': tf.Variable(tf.random_normal([lstm_n_hidden, n_classes]))
        }
        
        #define bias
        biases = {
            'out': tf.Variable(tf.random_normal([n_classes]))
        }

        # Linear activation, using rnn inner loop last output
        outputs_flat = tf.reshape (outputs, [-1, lstm_n_hidden])   
        mul_flat = tf.matmul(outputs_flat, weights['out']) + biases['out']
        lin_out = tf.reshape (mul_flat, [-1, tf.shape(self.x)[1], n_classes])
        self.gt = self.y
        
        # downsample the ground truth to have the granularity same as predicted y  
        #self.pooled_y = maxpool1d(self.y, np.prod(pooling_stride))
              
        # Define loss and optimizer
        self.cost = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=lin_out,
                                                                   targets = tf.cast(self.gt, tf.float32),
                                                                   pos_weight = pos_weight))
        tf.summary.scalar('cost', self.cost)
        
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)

        # Evaluate model
        self.prob_of_pos = tf.sigmoid(lin_out)
        
        self.debug_info = {
#            'conv1': conv1,
#            'pool1': pool1,
#            'conv2': conv2,
#            'pool2': pool2,
#            'conv3': conv3,
#            'pool3': pool3,
#            'conv4': conv4,
#            'pool4': pool4,
#            'conv5': conv5,
#            'pool5': tf.reduce_mean(pool5),
             #'pred': tf.reduce_mean(out_cnn),
             #'prob': tf.reduce_mean(self.prob_of_pos),
             #'y': tf.reduce_mean(self.y),
             #'pooled_y': tf.reduce_mean(self.pooled_y),
        }