import tensorflow as tf

import numpy as np

# Create some wrappers for simplicity
def conv1d(x, w, b = None, stride=1, activation=None):
    x = tf.nn.conv1d(x, w, stride = stride, padding='SAME')
    
    if b is not None:
        x = tf.nn.bias_add(x, b)
    
    if activation is not None:
        x = activation(x)
        
    return x

def maxpool1d(x, k=2):
    # MaxPool2D wrapper
    temp = tf.expand_dims(x, 1)
    out = tf.nn.max_pool(temp, ksize=[1, 1, k, 1], strides=[1, 1, k, 1],
                         padding='SAME')
    return tf.squeeze(out, 1)


class Model:
    
    def __init__(self, 
                 n_variables,
                 learning_rate,
                 n_classes,
                ):
        
        
        # tf Graph input
        self.x = tf.placeholder(tf.float32, [None, None, n_variables])
        self.y = tf.placeholder(tf.float32, [None, None, n_classes])
        self.n_classes = n_classes
        self.n_variables = n_variables
        

    # Create model
    def build_neural_wavelet_layer(self, pos_weight,
                                   learning_rate,
                                   wavelet_levels,
                                   convolution_levels,
                                   pooling_stride,
                                   weights_shape,
                                   bias_shape,
                                   wavelet_kernel_size,
                                   activation,
                                   trainable_wavelet,
                                   trainable_conv,
                                   trainable_last,
                                   variables_weights = None):
        
        if variables_weights is None:
            variables_weights = np.ones((self.n_classes,), dtype = 'float32')
        else:
            variables_weights = np.reshape(variables_weights, (self.n_classes,)).astype('float32')
        
        
        self.weights = {}
        self.bias = {}
        for i in range(convolution_levels):
            w = 'wc' + str(i)     
            self.weights[w] = tf.get_variable(w, 
                                         shape=weights_shape[i], 
                                         dtype=tf.float32, 
                                         initializer=tf.contrib.layers.xavier_initializer(),
                                         trainable=trainable_conv
                                        )
        
            b = 'bc' + str(i)
            self.bias[b] = tf.get_variable(b, 
                                   shape=bias_shape[i] , 
                                   dtype=tf.float32, 
                                   initializer=tf.constant_initializer(0.0),
                                   trainable=trainable_conv
                                     )                            
            
        self.weights['high_pass'] = tf.get_variable('high_pass', 
                                   shape=[self.n_variables, wavelet_kernel_size, 1, 1], 
                                   dtype=tf.float32, 
                                   initializer=tf.contrib.layers.xavier_initializer(),
                                   trainable=trainable_wavelet
                                  )
                                      
        self.weights['low_pass'] = tf.get_variable('low_pass', 
                                   shape=[self.n_variables, wavelet_kernel_size, 1, 1], 
                                   dtype=tf.float32, 
                                   initializer=tf.contrib.layers.xavier_initializer(),
                                   trainable=trainable_wavelet
                                  )
        self.weights['wfc'] = tf.get_variable('wfc', 
                                   shape=[1, weights_shape[-1][2], self.n_classes],
                                   dtype=tf.float32, 
                                   initializer=tf.contrib.layers.xavier_initializer(),
                                   trainable=trainable_last
                                  )
        self.bias['bfc'] = tf.get_variable('bfc', 
                                   shape=[self.n_classes], 
                                   dtype=tf.float32, 
                                   initializer=tf.constant_initializer(0.0),
                                   trainable=trainable_last
                                     )                            
          
        # wavelet 
        wavelet = []
        temp = tf.expand_dims(tf.transpose(self.x, (2, 0, 1)), -1)
       
        for _ in range(wavelet_levels):
            high_pass_res = tf.map_fn(
                lambda (x, w): conv1d(x, w, stride=1),
                (temp, self.weights['high_pass']),
                dtype=tf.float32,
                parallel_iterations=self.n_variables,
            )
            temp = tf.map_fn(
                lambda (x, w): conv1d(x, w, stride=2),
                (temp, self.weights['low_pass']),
                dtype=tf.float32,
                parallel_iterations=self.n_variables,
            )
            wavelet.append(tf.transpose(tf.squeeze(high_pass_res, -1), (1, 2, 0)))

        wavelet_c = []
        for item in wavelet:
            one_wavelet = item
            
            # convolution and pooling
            out_cnn = one_wavelet                            
            for i in range(convolution_levels):
                wname = 'wc' + str(i)
                bname = 'bc' + str(i)      
            
                conv = conv1d(out_cnn, self.weights[wname],
                               self.bias[bname],
                               activation = activation)
        
                out_cnn = maxpool1d(conv, k= pooling_stride[i])
            
            #fully conceted convolution
            fully_cnn = conv1d(out_cnn, self.weights['wfc'],
                                   self.bias['bfc'])    
       
            wavelet_c.append(fully_cnn)
        
        #upsampling each level of wavelet
        pred_ups = [] 
        for i in range(wavelet_levels):
            a = tf.stack((wavelet_c[i], )*(2**i), 2)
            s = tf.shape(a)
            b = tf.reshape(a, [s[0], s[1]* s[2], s[3]])  
            pred_ups.append(b)
            
        pred_ups_stacked = tf.stack(pred_ups, 1)
        pred = tf.reduce_mean(pred_ups_stacked, 1)    #final prediction
        
        #downsample the ground truth to have the granularity same as predicted y  
        self.pooled_y = maxpool1d(self.y, np.prod(pooling_stride)) 
        self.gt = self.pooled_y
        
        
        # Define loss and optimizer
        cost_variables = tf.reduce_mean(tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(targets = self.gt,
                                                     logits = pred,
                                                     pos_weight = pos_weight),
            axis = 0
        ), axis = 0)
        
        self.cost = tf.reduce_sum(cost_variables * variables_weights) / tf.reduce_sum(variables_weights) 
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
    
        tf.summary.scalar('cost', self.cost)
        self.prob_of_pos = tf.sigmoid(pred)

        self.debug_info = {
            'input_shape': tf.shape(self.x),
            #'label_shape': tf.shape(self.y),
            #'wavelet_out_shape': [tf.shape(item) for item in wavelet],
            #'cnn_out_shape': [tf.shape(item) for item in wavelet_c],
        }

