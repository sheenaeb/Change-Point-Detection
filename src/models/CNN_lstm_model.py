
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

        
    def build_cnn_lstm_net(self,
                           pooling_stride,
                           weights_shape,
                           bias_shape,
                           activation,
                           lstm_n_hidden,
                           n_classes,
                           learning_rate,
                           pos_weight,
                           trainable_conv,
                           trainable_last,
                           variables_weights = None):
        
        if variables_weights is None:
            variables_weights = np.ones((n_classes,), dtype = 'float32')
        else:
            variables_weights = np.reshape(variables_weights, (n_classes,)).astype('float32')
        
        #CNN
        
        self.weights ={}
        self.bias = {}
        for i in range(len(pooling_stride)):
            w = 'wc' + str(i)
            self.weights [w] = tf.get_variable(w,
                                          shape = weights_shape[i],
                                          dtype = tf.float32,
                                          initializer=tf.contrib.layers.xavier_initializer(),
                                          trainable=trainable_conv)
            b = 'bc' + str(i)
            self.bias[b] = tf.get_variable(b,
                                      shape = bias_shape[i],
                                      dtype = tf.float32,
                                      initializer = tf.constant_initializer(0.0),
                                      trainable=trainable_conv)
            
            
        out_cnn = self.x
        for i in range(len(pooling_stride)):
            
            wname = 'wc' + str(i)
            bname= 'bc' + str(i)
            conv = conv1d(out_cnn, self.weights[wname],self.bias[bname], activation = activation)
            
            out_cnn = maxpool1d(conv, k=pooling_stride[i])
        
        #RNN
        
        
        ## TODO: use "sequence_length"
        
        # Define a lstm cell with tensorflow
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(lstm_n_hidden)    #, forget_bias=1.0)

        outputs, _ = tf.nn.dynamic_rnn(lstm_cell, out_cnn, dtype=tf.float32)   #output shape is [batch_size,t, lstm_n_hidden]

         # Define weights
        self.weights_rnn = {
            'out': tf.Variable(tf.random_normal([lstm_n_hidden, n_classes]))
        }
        
        #define bias
        self.bias_rnn = {
            'out': tf.Variable(tf.random_normal([n_classes]))
        }

        # Linear activation, using rnn inner loop last output
        outputs_flat = tf.reshape (outputs, [-1, lstm_n_hidden])   
        mul_flat = tf.matmul(outputs_flat, self.weights_rnn['out']) + self.bias_rnn['out']
        lin_out = tf.reshape (mul_flat, [-1, tf.shape(out_cnn)[1], n_classes])

        # downsample the ground truth to have the granularity same as predicted y  
        self.pooled_y = maxpool1d(self.y, np.prod(pooling_stride))
        self.gt = self.pooled_y
              
        # Define loss and optimizer
        
        
        cost_variables = tf.reduce_mean(tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(logits=lin_out,
                                                     targets = tf.cast(self.pooled_y, tf.float32),
                                                     pos_weight = pos_weight),
            axis = 0
        ), axis = 0)
        
        self.cost = tf.reduce_sum(cost_variables * variables_weights) / tf.reduce_sum(variables_weights)

        tf.summary.scalar('cost', self.cost)
        
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)

        # Evaluate model
        self.prob_of_pos = tf.sigmoid(lin_out)
        
        self.debug_info = {
        }