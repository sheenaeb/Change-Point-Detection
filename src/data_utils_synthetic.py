import numpy as np
from hdf5storage import loadmat


def load_mat(path, key):
    return loadmat(path)[key]
    
def slide_window(data, window_size, stride, num_dim_expand=0):
    """
    Inputs:
            data: numpy array 3dim, (batch, length, dim) 
            window_size: int, Time series will be devided to the window_size length if they are too long
            stride: int, Time series will be diveded to different ts with this time series 
            num_dim_expand:  
            
    Outputs:
            timeseries_output: list of numpy.array, times series with window_size of 'window_size'
            info: A list which contains number of real information(non padding) for each time series
    """
    
    #Initializing the output variables 
    info = []
    timeseries_output = []
 
    for item in range(data.shape[0]):
        one_timeseries = data[item]    #this is numpy.ndarray 2dim (signal length,dim)
        
        for _ in range (num_dim_expand):
            one_timeseries = np.expand_dims (one_timeseries, -1)
            
        start = 0
        end = window_size
        len_one_timeseries = one_timeseries.shape[0]
        
        while end <= len_one_timeseries:
            timeseries_output.append(one_timeseries[start: end])
            info.append(window_size)
            start += stride
            end = start + window_size
            
        if start < len_one_timeseries:
            temp = one_timeseries[start: len_one_timeseries]
            temp_pad = np.concatenate ((temp, 
                    np.zeros((start + window_size - len_one_timeseries,) + temp.shape[1:],
                    dtype=temp.dtype)), axis=0)
            
            timeseries_output.append(temp_pad)
            info.append(len_one_timeseries - start)
            
            
    return (info, np.stack(timeseries_output))     
        
        
def preprocess_ground_truth(ground_truth, sequence_length):

    """
        Input:
            ground_truth: np.ndarray dim2, inclues number of change points for each time series
            sequence_length: list, includes length of each time series
            
        Output: 
            timeseries_output: returns a numpy.array 2dim with window_size of corresponded timeseries, everywhere is zero except
            the index of ground truth
            
    """
    
    #initializing the output variables
    timeseries_output = []
    
    for i in range(len(sequence_length)):
        gt_one = np.zeros((sequence_length[i],), dtype = 'int32')
        gt_in_int = ground_truth[i].astype(int) 
        gt_one[gt_in_int] = 1
        timeseries_output.append(gt_one)
    
    
    
    return np.asarray(timeseries_output)

def shuffle(data, gt):
    
    """
    This function shuffle data and ground truth 
    """
    
    idx = np.arange(data.shape[0])
    
    np.random.shuffle(idx)
    
    data = data[idx]
    gt = gt[idx]
    
    return data,gt
    
    
    
    

