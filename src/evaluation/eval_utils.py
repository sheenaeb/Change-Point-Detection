import numpy as np
import matplotlib.pyplot as plt


def threshold_non_max_supression(pred, threshold, window):
    ''' 
        Input:
             pred: np.array 2d, probability of change for a time series.
             threshold: a float number, to threshold probability of changes.
             window: an integer, for each time t we do non_max supresshion for
                     window of t-window, t+window
            
    '''
    
    pred = np.copy(pred)
    pred[pred<threshold] =0 
    for i in range(pred.shape[1]):
        argmax_res = np.argmax(pred[:,max(0,i-window):min(i+window+1,pred.shape[1])],axis=1)
        pred[:,i] *= (argmax_res == np.int64(i-max(0,i-window))).astype(np.int64)
           
    cpd = []
    for i in range(pred.shape[0]):
        non_zero = np.nonzero(pred[i])[0]
        cpd.append(non_zero)
    
    return cpd
         
def threshold_non_max_supression_(pred, threshold, window):
    ''' 
        Input:
             pred: np.array 2d, probability of change for a time series.
             threshold: a float number, to threshold probability of changes.
             window: an integer, for each time t we do non_max supresshion for
                     window of t-window, t+window
            
    '''
    
    pred = np.copy(pred)
    pred[pred<threshold] =0 
    for i in range(pred.shape[1]):
        argmax_res = np.argmax(pred[:,max(0,i-window):min(i+window+1,pred.shape[1])],axis=1)
        pred[:,i] *= (argmax_res == np.int64(i-max(0,i-window))).astype(np.int64)
           
    cpd = []
    for i in range(pred.shape[0]):
        non_zero = np.nonzero(pred[i])[0]
        cpd.append(non_zero)
    
    return cpd
         
def match_cpd_with_gtruth (cpd, gt):
    '''
    This function returns tuples of one cpd and one gt which are matched together, it means they are the closest
    together.
    '''
    assert (len (cpd) == len (gt))
    
    matches = []
    for i in range (len (cpd)):
        m1 = []
        for t_cpd in cpd[i]:
            best_gt = None
            for t_gt in gt[i]:
                if best_gt is None or abs(t_gt-t_cpd) < min_dist:
                    min_dist = abs(t_gt-t_cpd)
                    best_gt = t_gt
            m1.append ((t_cpd, best_gt))
            
        m2 = []
        for t_gt in gt[i]:
            best_cpd = None
            for t_cpd in cpd[i]:
                if best_cpd is None or abs(t_gt-t_cpd) < min_dist:
                    min_dist = abs(t_gt-t_cpd)
                    best_cpd = t_cpd
            m2.append ((best_cpd, t_gt))
            
        m3 = []
        for m in m1:
            if m in m2:
                m3.append (m)
        matches.append (m3)
        
    return matches
            
def calc_prec_recall (matches, cpd, gt, win_size=None):    
    n_match = 0.
    for mm in matches:
        for m in mm:
            if win_size is None or win_size >= abs(m[0]-m[1]):
                n_match += 1.
                
    n_gt = float(sum([len(item) for item in gt]))
    n_cpd = float(sum([len(item) for item in cpd]))
        
    precision = n_match/n_cpd if n_cpd>0 else 1.
    recall = n_match/n_gt if n_gt>0 else 1.
    f1 = 2*precision*recall/(precision+recall) if precision+recall>0 else 0.
    
    return (precision, recall, f1)
        
def calc_prec_recall_old(cpd, gt, win_size):
    assert (len (cpd) == len (gt))
    
    nom = 0.
    denom = 0.
    for sidx in range (len (cpd)):
        for t1 in cpd[sidx]:
            for t2 in gt[sidx]:  
                if t2 >= t1-win_size and t2 <= t1+win_size:
                    nom += 1.         
                    break
            denom += 1.

    if denom == 0:
        precision = 1.0
    else:
        precision = nom / denom  

    nom = 0.
    denom = 0.
    for sidx in range (len (cpd)):
        for t1 in gt[sidx]:
            for t2 in cpd[sidx]:  
                if t2 >= t1-win_size and t2 <= t1+win_size:
                    nom += 1.         
                    break
            denom += 1.

    if denom == 0:
        recall = 1.0
    else:
        recall = nom / denom  

    return precision, recall
        
    
def plot_pr_curve(prec_recall_list):
        
    maxes = max_for_plot_curve(prec_recall_list)
    
    plt.plot(maxes[:,0],maxes[:,1])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0,1.1])
    plt.xlim([0,1.1])
    plt.grid()
    plt.show()
    
def max_for_plot_curve(prec_recall_list):
    
    precision = np.asarray([item[0] for item in prec_recall_list])    
    recall = np.asarray([item[1] for item in prec_recall_list])    
        
    maxes=[]

    values=np.unique(recall)
    for i in values:
        indexes=np.where(recall==i)[0]

        j=np.max(precision[indexes])
        maxes.append([i,j])

    return np.array(maxes)


def calc_auc(prec_recall_list, gt):
    
    maxes = max_for_plot_curve(prec_recall_list)
        
    return maxes[:,1].sum() / float(sum([len(item) for item in gt]))
    
    
    
   