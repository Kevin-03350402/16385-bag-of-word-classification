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


pkl_file = open('random_dict.pkl', 'rb')
random_dict = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open('harris_dict.pkl', 'rb')
harris_dict = pickle.load(pkl_file)
pkl_file.close()


pkl_file = open('visionRandom.pkl', 'rb')
visionRandom = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open('visionHarris.pkl', 'rb')
visionHarris = pickle.load(pkl_file)
pkl_file.close()

random_euclidean = []
random_chisq = []
harris_euclidean = []
harris_chisq = []
dictionarySize = 100


random_euclidean_matrix = np.zeros((8,8))
random_chisq_matrix = np.zeros((8,8))
harris_euclidean_matrix = np.zeros((8,8))
harris_chisq_matrix= np.zeros((8,8))
def getImageDistance(hist1, histSet, method):
    distance = []
    for th in histSet:
        distance.append(get_image_distance(hist1,th,method))
    distance = np.array(distance)

    return np.argmin(distance)

for i, path in enumerate(test_imagenames):
    file_name = path[:-4]
    f = open('../data/%s_%s.pkl' % (file_name, 'Random'),'rb')
    random_input = pickle.load(f)
    f.close()

    f = open('../data/%s_%s.pkl' % (file_name, 'Harris'),'rb')
    harris_input = pickle.load(f)
    f.close()
    # covert to histogram
    random_hist = np.histogram(random_input,bins = np.arange(dictionarySize))[0]
    baselinesum  = np.sum(random_hist)
    baselinesum = int(baselinesum)
    random_hist  = random_hist/baselinesum

    harris_hist = np.histogram(harris_input,bins = np.arange(dictionarySize))[0]
    baselinesum  = np.sum(harris_hist)
    baselinesum = int(baselinesum)
    harris_hist = harris_hist/baselinesum
    truelabel = int(test_labels[i])


    index_nnre = getImageDistance(random_hist , visionRandom['trainFeatures'], 'euclidean')
    nn_random_euclidean = int(visionRandom['trainLabels'][index_nnre])
    random_euclidean.append(nn_random_euclidean)
    random_euclidean_matrix[truelabel-1,nn_random_euclidean-1]+=1


    index_nnrc = getImageDistance(random_hist , visionRandom['trainFeatures'], 'chisq')
    nn_random_chisq = int(visionRandom['trainLabels'][index_nnrc])
    random_chisq.append(nn_random_chisq)
    random_chisq_matrix[truelabel-1,nn_random_chisq-1] +=1 
    


    index_nnhe = getImageDistance(harris_hist , visionHarris['trainFeatures'], 'euclidean')
    nn_harris_euclidean = int(visionHarris['trainLabels'][index_nnhe])
    harris_euclidean.append(nn_harris_euclidean)
    harris_euclidean_matrix[truelabel-1,nn_harris_euclidean-1] += 1

    index_nnhc = getImageDistance(harris_hist , visionHarris['trainFeatures'], 'chisq')
    nn_harris_chisq = int(visionHarris['trainLabels'][index_nnhc])
    harris_chisq.append(nn_harris_chisq)
    harris_chisq_matrix[truelabel-1,nn_harris_chisq-1]+=1

random_euclidean = np.array(random_euclidean)
random_chisq = np.array(random_chisq)
harris_euclidean = np.array(harris_euclidean)
harris_chisq = np.array(harris_chisq)
truelabel = np.array(test_labels)

random_euclidean_accu = np.sum(np.equal(random_euclidean,truelabel))
print('The accuracy of random_euclidean and confusion matrix with is:')
print(random_euclidean_accu/len(random_euclidean))
print(random_euclidean_matrix)

random_chisq_accu = np.sum(np.equal(random_chisq,truelabel))
print('The accuracy of random_chisq and confusion matrix is:')
print(random_chisq_accu/len(random_chisq))
print(random_chisq_matrix)

harris_euclidean_accu = np.sum(np.equal(harris_euclidean,truelabel))
print('The accuracy of harris_euclidean and confusion matrix is :')
print(harris_euclidean_accu/len(harris_euclidean))
print(harris_euclidean_matrix)

harris_chisq_accu = np.sum(np.equal(harris_chisq,truelabel))
print('The accuracy of harris_chisq and confusion matrix is:')
print(harris_chisq_accu/len(harris_chisq))
print(harris_chisq_matrix)

    









    




# ----------------------------------------------
