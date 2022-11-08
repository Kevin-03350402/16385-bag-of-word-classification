import numpy as np
import cv2 as cv
from RGB2Lab import rgb2lab
from utils import *
from createFilterBank import create_filterbank
from RGB2Lab import rgb2lab
from utils import imfilter
import os

def get_random_points(I, alpha):

    # -----fill in your implementation here --------
    H = I.shape[0]
    W = I.shape[1]
    col1y = np.random.randint(H, size=alpha)
    col2x = np.random.randint(W, size=alpha)
    points = np.vstack((col2x,col1y))
    
    points = points.T

    # ----------------------------------------------
    return points

