import numpy as np
import pickle
from createFilterBank import create_filterbank
from getImageFeatures import get_image_features
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
    random_list.append(img_pkl)
    f.close()

    f = open('../data/%s_%s.pkl' % (file_name, 'Harris'),'rb')
    img_pkl = pickle.load(f)
    harris_list.append(img_pkl)
    f.close()

train_feature_random = get_image_features(random_list,K)
train_feature_harris = get_image_features(harris_list,K)


visionrandom = dict()
visionrandom['dictionary'] = random_dict
visionrandom['filterBank'] = filterBank
visionrandom['trainFeatures'] = train_feature_random
visionrandom['trainLabels'] = train_labels

output = open('visionRandom.pkl', 'wb')
pickle.dump(visionrandom, output)
output.close()


visionharris = dict()
visionharris['dictionary'] = harris_dict
visionharris['filterBank'] = filterBank
visionharris['trainFeatures'] = train_feature_harris
visionharris['trainLabels'] = train_labels
output = open('visionHarris.pkl', 'wb')
pickle.dump(visionharris, output)
output.close()
 # ----------------------------------------------
