import numpy as np
import pickle
from createFilterBank import create_filterbank
from getImageFeatures import get_image_features
import numpy as np
import pickle
from utils import chi2dist
from getImageDistance import get_image_distance
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
from sklearn import metrics
# -----fill in your implementation here --------
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

random_list = []

harris_list = []
for i, path in enumerate(train_imagenames):

    file_name = path[:-4]
    f = open('../data/%s_%s.pkl' % (file_name, 'Random'),'rb')
    img_pkl = pickle.load(f)
    img_pkl = img_pkl.astype(int)
    img_pkl = img_pkl.flatten()
    img_pkl = img_pkl.tolist()
    img_pkl = img_pkl[:20000]
    print(len(random_list))

    
    random_list.append(img_pkl)
    f.close()

    f = open('../data/%s_%s.pkl' % (file_name, 'Harris'),'rb')
    img_pkl = pickle.load(f)
    img_pkl = img_pkl.astype(int)
    img_pkl = img_pkl.flatten()
    img_pkl = img_pkl.tolist()
    img_pkl = img_pkl[:20000]
    
    harris_list.append(img_pkl)
    f.close()

svclassifier_random_linear = SVC(kernel='linear',gamma = 'auto')
svclassifier_random_poly = SVC(kernel='poly',gamma = 'auto')

svclassifier_harris_linear = SVC(kernel='linear',gamma = 'auto')
svclassifier_harris_poly = SVC(kernel='poly',gamma = 'auto')


random_list = np.array(random_list)
train_labels = train_labels.astype(int)
train_labels = train_labels.reshape(len(train_labels),1)
svclassifier_random_linear.fit(random_list , train_labels)
svclassifier_random_poly.fit(random_list , train_labels)


harris_list = np.array(harris_list)
svclassifier_harris_linear.fit(harris_list, train_labels)
svclassifier_harris_poly.fit(harris_list, train_labels)



visionSVM = dict()
visionSVM['random_linear'] = svclassifier_random_linear
visionSVM['random_poly'] = svclassifier_random_poly
visionSVM['harris_linear'] = svclassifier_harris_linear
visionSVM['harris_poly'] = svclassifier_harris_poly
visionSVM['trainLabels'] = train_labels
output = open('visionSVM.pkl', 'wb')
pickle.dump(visionSVM, output)
output.close()


f = open('../data/traintest.pkl', 'rb')
meta = pickle.load(f)
f.close()
test_imagenames = meta['test_imagenames']
test_labels = meta['test_labels']


random_list_test = []

harris_list_test = []
for i, path in enumerate(test_imagenames):

    file_name = path[:-4]
    f = open('../data/%s_%s.pkl' % (file_name, 'Random'),'rb')
    img_pkl = pickle.load(f)
    img_pkl = img_pkl.astype(int)
    img_pkl = img_pkl.flatten()
    img_pkl = img_pkl.tolist()
    img_pkl = img_pkl[:20000]


    
    random_list_test.append(img_pkl)
    f.close()

    f = open('../data/%s_%s.pkl' % (file_name, 'Harris'),'rb')
    img_pkl = pickle.load(f)
    img_pkl = img_pkl.astype(int)
    img_pkl = img_pkl.flatten()
    img_pkl = img_pkl.tolist()
    img_pkl = img_pkl[:20000]
    
    harris_list_test.append(img_pkl)
    f.close()
 # ----------------------------------------------


y_pred_random_linear = svclassifier_random_linear.predict(random_list_test)
y_pred_random_poly = svclassifier_random_poly.predict(random_list_test)

y_pred_harris_linear = svclassifier_harris_linear.predict(harris_list_test)
y_pred_harris_poly = svclassifier_harris_poly.predict(harris_list_test)


print("Accuracy y_pred_random_linear:",metrics.accuracy_score(test_labels, y_pred_random_linear))
print("Accuracy y_pred_random_poly:",metrics.accuracy_score(test_labels, y_pred_random_poly))

print("Accuracy y_pred_harris_linear:",metrics.accuracy_score(test_labels, y_pred_harris_linear))
print("Accuracy y_pred_harris_poly:",metrics.accuracy_score(test_labels, y_pred_harris_poly))