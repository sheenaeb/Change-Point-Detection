import numpy as np
from eval_utils import *

def non_max_supression(pred, window, ratio):
    '''
        This function apply non max supression algorithm on pred and return it.
        pred: 2d numpy array, which are probability of change
        '''

    pred = np.copy(pred)
    for i in range(pred.shape[1]):
        temp = np.argmax(pred[:, max(0, i-window): min(i+window+1, pred.shape[1])], axis=1)
        pred[:, i] *= (temp == np.int64(i-max(0,i-window))).astype(np.int64)

    cpd = []
    cpd_val = []
    for i in range(pred.shape[0]):
        non_zero = np.nonzero(pred[i])[0]
        cpd.append(non_zero)
        cpd_val.append(pred[i][non_zero])
        
    #ratio = float(data_num_step) / float(pred.shape[1])    
    cpd_scaled = [(((item) * ratio) + ratio/2) for item in cpd]

    return cpd_val, cpd_scaled

        
    
def ranking(cpd_val, cpd, gt, tolerance):
    '''
       pred: np.array 2d, probability of being classified in true class
       gt: np.array 2d, ground truth
    '''
    
    cpd_val_flatten = np.concatenate(cpd_val)
    cpd_flatten = np.concatenate(cpd)
        
    s = np.concatenate([np.asarray([i]*len(cpd[i])) for i in range(len(cpd))])

    sort_idx = np.argsort(-cpd_val_flatten)
    s_sorted = s[sort_idx]
    cpd_sorted = cpd_flatten[sort_idx]
      
    prec = []
    rec = []
    f1= []
    for i in range(1,len(sort_idx)+1):
        thresholded_cpd = []
        for j in range(len(cpd)):
            thresholded_cpd.append([])
           
        for k in range(i):
            thresholded_cpd[s_sorted[k]].append(cpd_sorted[k])
            
        matches = match_cpd_with_gtruth(thresholded_cpd, gt)
        out = calc_prec_recall(matches, thresholded_cpd, gt, tolerance)
        prec.append(out[0])
        rec.append(out[1])
        f1.append(out[2])
        
    return prec, rec, f1

            
            
            
            
        
        
        