import numpy as np
from scipy.spatial.distance import cdist
from extractFilterResponses import extract_filter_responses
from createFilterBank import create_filterbank
import pickle
import cv2 as cv
from skimage.color import label2rgb
import os

pkl_file = open('random_dict.pkl', 'rb')
random_dict = pickle.load(pkl_file)
pkl_file.close()

pkl_file = open('harris_dict.pkl', 'rb')
harris_dict = pickle.load(pkl_file)
pkl_file.close()


def get_visual_words(I, dictionary, filterBank):

    # -----fill in your implementation here --------
    H = I.shape[0]
    W = I.shape[1]
    wordMap = np.zeros((H,W))
    f_response = extract_filter_responses(I, filterBank)
    for y in range (H):
        for x in range(W):
            vector = f_response[y,x,:]
            vector = np.array(vector)
            vector = vector.reshape((1,len(vector)))
            distance = cdist(vector, dictionary,'euclidean')
            wordMap[y,x] = np.argmin(distance)

            


    # ----------------------------------------------

    return wordMap

img = cv.imread('../data/airport/sun_aetygbcukodnyxkl.jpg')
filterBank = create_filterbank()

res = get_visual_words(img, harris_dict, filterBank)
# note: My dictionary has K = 100 (suggested in q2.3) rather than 50 (writeup example) so the output might be slightly different
nlabel = len(np.unique(res))
colors = [tuple(map(tuple, np.random.rand(1, 3)))[0] for i in range(0, nlabel)]

output = label2rgb(res,colors = colors)
output = ((255*output).astype(int))
path = '../result'
cv.imwrite(os.path.join(path , 'neighbour img.jpg'), output)