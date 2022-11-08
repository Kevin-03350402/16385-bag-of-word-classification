import numpy as np
import pickle
from utils import chi2dist
from getImageDistance import get_image_distance
from statistics import mode
from matplotlib import pyplot as plt
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



dictionarySize = 100


def getImageDistance(hist1, histSet, method):
    distance = []
    for th in histSet:
        distance.append(get_image_distance(hist1,th,method))
    distance = np.array(distance)

    return distance
res_accu = []
for k in range(1,41):
    print(k)
    harris_chisq = []
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




        dl = getImageDistance(harris_hist , visionHarris['trainFeatures'], 'chisq')
        nnd = np.argsort(dl)
        index_nnhc = nnd[:k]
        temp = []
        for ep in range (k):
            nn_harris_chisq = int(visionHarris['trainLabels'][index_nnhc[ep]])
            temp.append(nn_harris_chisq)
        harris_chisq.append(mode(temp))



    harris_chisq = np.array(harris_chisq)
    truelabel = np.array(test_labels)
    harris_chisq_accu = np.sum(np.equal(harris_chisq,truelabel))
    accu = harris_chisq_accu/len(harris_chisq)
    res_accu.append(accu)

x = np.arange(1,41)
y = np.array(res_accu)
print(res_accu)

harris_chisq_matrix= np.zeros((8,8))
hcl = []
for i, path in enumerate(test_imagenames):
        k = 9
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




        dl = getImageDistance(harris_hist , visionHarris['trainFeatures'], 'chisq')
        nnd = np.argsort(dl)
        index_nnhc = nnd[:k]
        temp = []
        for ep in range (k):
            nn_harris_chisq = int(visionHarris['trainLabels'][index_nnhc[ep]])
            temp.append(nn_harris_chisq)
        hcl.append(mode(temp))
        harris_chisq_matrix[truelabel-1,mode(temp)-1]+=1

print('The harris_chisq matrix with k = 9 is:')
print(harris_chisq_matrix)

plt.plot(x,y,label = 'KNN Accuracy')
plt.xlabel("value of k")
plt.ylabel("Accuracy")
leg = plt.legend(loc='upper left')

plt.show()


    









    




# ----------------------------------------------
