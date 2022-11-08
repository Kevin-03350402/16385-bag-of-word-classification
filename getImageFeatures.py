import numpy as np
import pickle

def get_image_features(wordMap, dictionarySize):

    # -----fill in your implementation here --------
    h = np.histogram(wordMap[0],bins = np.arange(dictionarySize))[0]

    baselinesum  = np.sum(h)
    baselinesum = int(baselinesum)

    h= h/baselinesum

    length = len(wordMap)
    for i in range (1, length):
        m = wordMap[i]
        new_h = np.histogram(m,bins = np.arange(dictionarySize))[0]
        new_h= new_h/baselinesum
        h = np.vstack((h,new_h))
    
    # ----------------------------------------------
    
    return h

