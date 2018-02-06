import numpy as np
import tensorflow as tf

# Create some wrappers for simplicity
def conv1d(x, w, b, strides=1, activation=None):
    x = tf.nn.conv1d(x, w, stride=1, padding='SAME')
    
    if b is not None:
        x = tf.nn.bias_add(x, b)
    if activation is not None:
        x = activation(x)
    return x


def maxpool1d(x, k=2):
    '''
     tf.nn.max_pool is for the 2d arrays. Here we add one extra dimension and use this
     max_pool function, the we squeeze that dimesntion.
    '''
    temp = tf.expand_dims(x, 1)
    out = tf.nn.max_pool(temp, ksize=[1, 1, k, 1], strides=[1, 1, k, 1],
                         padding='SAME')
    return tf.squeeze(out, 1)

class Model:
    
    def __init__(self, 
                 n_variables,
                 learning_rate,
                 n_classes):
        
        
        # tf Graph input
        self.x = tf.placeholder(tf.float32, [None, None, n_variables])
        self.y = tf.placeholder(tf.float32, [None, None, n_classes])
        self.n_classes = n_classes
        
        

    # Create model
    def build_conv_net(self, pos_weight_values,
                            learning_rate,
                            convolution_levels,
                            pooling_stride,
                            weights_shape,
                            bias_shape,
                            n_classes,
                            activation,
                            trainable_conv,
                            trainable_last,
                            variables_weights = None):
        
        if variables_weights is None:
            variables_weights = np.ones((n_classes,), dtype = 'float32')
        else:
            variables_weights = np.reshape(variables_weights, (n_classes,)).astype('float32')
        
        self.weights = {}
        self.bias = {}
        for i in range(convolution_levels):
            w = 'wc' + str(i)     
            self.weights[w] = tf.get_variable(w, 
                                         shape=weights_shape[i], 
                                         dtype=tf.float32, 
                                         initializer=tf.contrib.layers.xavier_initializer(),
                                         trainable=trainable_conv)
        
            b = 'bc' + str(i)
            self.bias[b] = tf.get_variable(b, 
                                   shape=bias_shape[i] , 
                                   dtype=tf.float32, 
                                   initializer=tf.constant_initializer(0.0),
                                   trainable=trainable_conv)                            
            
        self.weights['wfc'] = tf.get_variable('wfc', 
                                         shape= [1, weights_shape[-1][2], self.n_classes], 
                                         dtype=tf.float32, 
                                         initializer=tf.contrib.layers.xavier_initializer(),
                                         trainable=trainable_last)    
        
        self.bias['bfc'] = tf.get_variable('bfc', 
                                   shape=[self.n_classes], 
                                   dtype=tf.float32, 
                                   initializer=tf.constant_initializer(0.0),
                                   trainable=trainable_last)
        
        # convolution and pooling
        out_cnn = self.x                              
        for i in range(convolution_levels):
         
            wname = 'wc' + str(i)
            bname = 'bc' + str(i)   
            conv = conv1d(out_cnn, self.weights[wname],
                           self.bias[bname],
                           activation = activation)
           
            out_cnn = maxpool1d(conv, k= pooling_stride[i])
            
            
        #fully connected
        out_cnn = conv1d(out_cnn, self.weights['wfc'], self.bias['bfc'])
        self.prob_of_pos = tf.sigmoid(out_cnn)
        
        # downsample the ground truth to have the granularity same as predicted y  
        self.pooled_y = maxpool1d(self.y, np.prod(pooling_stride))
        self.gt = self.pooled_y
        
        
        # Define loss and optimizer
        #self.cost = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets = self.gt,
        #                                                                    logits = out_cnn,
        #                                                                    pos_weight = pos_weight))
                
        pos_weight_tensor = tf.cast(tf.reshape(pos_weight_values, [1,1,-1]), dtype = 'float32')
        
        #print pos_weight_tensor, self.gt, self.prob_of_pos, tf.log(self.prob_of_pos), (1-self.gt), tf.log(1-self.prob_of_pos)
        
        #self.cost = tf.reduce_mean((-1*pos_weight_tensor*self.gt*tf.log(self.prob_of_pos)) - 
        #                           ((1-self.gt)*tf.log(1-self.prob_of_pos)))
        
        x = out_cnn
        z = self.gt
        q = pos_weight_tensor
        l = (1 + (q - 1) * z)
        
        cost_variables = tf.reduce_mean(tf.reduce_mean(
            (1 - z) * x + l * (tf.log(1 + tf.exp(-tf.abs(x))) + tf.maximum(-x, 0)),
            axis = 0
        ), axis = 0)
        
        self.cost = tf.reduce_sum(cost_variables * variables_weights) / tf.reduce_sum(variables_weights)
        
        
        #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)

    
        tf.summary.scalar('cost', self.cost)

    
        self.debug_info = {
#            'pool5': tf.reduce_mean(pool5),
             'pred': tf.reduce_mean(out_cnn),
             'prob': tf.reduce_mean(self.prob_of_pos),
             'y': tf.reduce_mean(self.y),
             'pooled_y': tf.reduce_mean(self.pooled_y),
        }
