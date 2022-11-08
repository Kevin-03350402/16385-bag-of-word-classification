import numpy as np
import pickle
import numpy as np
import pickle
from createFilterBank import create_filterbank
def IDF(wordMap, dictionarySize):

    # -----fill in your implementation here --------
    h = np.histogram(wordMap[0],bins = np.arange(dictionarySize))[0]

    freq = np.copy(h)
    freq[freq!=0] = 1




    baselinesum  = np.sum(h)
    baselinesum = int(baselinesum)

    h= h/baselinesum

    length = len(wordMap)
    for i in range (1, length):
        m = wordMap[i]
        new_h = np.histogram(m,bins = np.arange(dictionarySize))[0]
        new_h= new_h/baselinesum
        h = np.vstack((h,new_h))
        temp = np.copy(new_h)
        temp[temp!=0] = 1
        freq  = np.add(freq, temp)
    
    # ----------------------------------------------

    freq = freq.reshape(len(freq),1)
    freq = freq + 1
    w_idf = np.log(length/freq)
 
    h = np.matmul(h,w_idf)



    return (h,w_idf)



f = open('../data/traintest.pkl', 'rb')
meta = pickle.load(f)
f.close()
train_imagenames = meta['train_imagenames']
train_labels = meta['train_labels']
# get all of the train image in pkl form

pkl_file = open('random_dict.pkl', 'rb')
random_dict = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open('harris_dict.pkl', 'rb')
harris_dict = pickle.load(pkl_file)
pkl_file.close()

filterBank = create_filterbank()
K = 100


harris_list = []
for i, path in enumerate(train_imagenames):
    file_name = path[:-4]
    f = open('../data/%s_%s.pkl' % (file_name, 'Harris'),'rb')
    img_pkl = pickle.load(f)
    harris_list.append(img_pkl)
    f.close()


train_feature_harris, w_idf= IDF(harris_list,K)



idf = dict()
idf['w_idf'] = w_idf
idf['train_harris_feature'] = train_feature_harris
idf['labels'] = train_labels 
output = open('IDF.pkl', 'wb')
pickle.dump(idf, output)
output.close()