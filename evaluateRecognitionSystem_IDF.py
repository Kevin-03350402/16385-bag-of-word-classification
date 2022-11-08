import numpy as np
import pickle
from utils import chi2dist
from getImageDistance import get_image_distance
# -----fill in your implementation here --------
#load the test data
f = open('../data/traintest.pkl', 'rb')
meta = pickle.load(f)
f.close()
test_imagenames = meta['test_imagenames']
test_labels = meta['test_labels']

pkl_file = open('harris_dict.pkl', 'rb')
harris_dict = pickle.load(pkl_file)
pkl_file.close()


pkl_file = open('IDF.pkl', 'rb')
idf = pickle.load(pkl_file)
pkl_file.close()


harris_chisq = []
dictionarySize = 100

harris_chisq_matrix= np.zeros((8,8))
def getImageDistance(hist1, histSet, method):
    distance = []
    for th in histSet:
        distance.append(get_image_distance(hist1,th,method))
    distance = np.array(distance)

    return np.argmin(distance)

for i, path in enumerate(test_imagenames):
    file_name = path[:-4]


    f = open('../data/%s_%s.pkl' % (file_name, 'Harris'),'rb')
    harris_input = pickle.load(f)
    f.close()
    # covert to histogram


    harris_hist = np.histogram(harris_input,bins = np.arange(dictionarySize))[0]
    baselinesum  = np.sum(harris_hist)
    baselinesum = int(baselinesum)
    harris_hist = harris_hist/baselinesum
    truelabel = int(test_labels[i])


    

    index_nnhc = getImageDistance(harris_hist , idf['train_harris_feature'], 'chisq')
    nn_harris_chisq = int(idf['labels'][index_nnhc])
    harris_chisq.append(nn_harris_chisq)
    harris_chisq_matrix[truelabel-1,nn_harris_chisq-1]+=1


harris_chisq = np.array(harris_chisq)
truelabel = np.array(test_labels)



harris_chisq_accu = np.sum(np.equal(harris_chisq,truelabel))
print('The accuracy of harris_chisq and confusion matrix is:')
print(harris_chisq_accu/len(harris_chisq))
print(harris_chisq_matrix)

    









    




# ----------------------------------------------
