import cv2 as cv
import numpy as np
from RGB2Lab import rgb2lab
from utils import *
from createFilterBank import create_filterbank
from RGB2Lab import rgb2lab
from utils import imfilter
import os


def extract_filter_responses(I, filterBank):
    
    I = I.astype(np.float64)
    # extraxt different filters:
    if len(I.shape) == 2:
        I = np.tile(I, (3, 1, 1))

    # -----fill in your implementation here --------
    # convert to lab
    img = rgb2lab(I)
    H = img.shape[0]
    W = img.shape[1]
    n = len(filterBank)
    filterResponses = np.zeros((H,W,3*n))
    for i in range (n):
        filter = filterBank[i]
        filterResponses[:,:,i*3+0] = imfilter(img[:,:,0],filter)
        filterResponses[:,:,i*3+1] = imfilter(img[:,:,1],filter)
        filterResponses[:,:,i*3+2] = imfilter(img[:,:,2],filter)
    # ----------------------------------------------
    
    return filterResponses

im = cv.imread('../data/desert/sun_adpbjcrpyetqykvt.jpg')

filterbank = create_filterbank()

filterResponses = extract_filter_responses(im, filterbank)

path = '../result'

for i in range (3*len(filterbank)):
    img = filterResponses[:,:,i]
    imgmin = np.min(img)
    imgmax = np.max(img)
    p_range = (imgmax-imgmin)
    img += abs(imgmin)
    img/=p_range
    img*= 255
    img = img.astype(int)
    cv.imwrite(os.path.join(path , f'desert img{i}.jpg'), img)
