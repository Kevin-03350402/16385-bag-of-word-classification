import numpy as np
import cv2 as cv
from scipy import ndimage
from utils import imfilter
from scipy.signal import convolve2d
import os
import math
from scipy import signal 
import scipy

def get_harris_points(I, alpha, k):

    if len(I.shape) == 3 and I.shape[2] == 3:
        I = cv.cvtColor(I, cv.COLOR_RGB2GRAY)
    if I.max() > 1.0:
        I = I / 255.0

    # -----fill in your implementation here --------
    sobel = np.array([[1,0,-1]])
    sigma1 = 0.75
    hsize = 2 * math.ceil(3 * sigma1) + 1
    gm1 = scipy.signal.gaussian(hsize,sigma1)
    gmx1 = gm1.reshape(hsize,1)
    gmy1 = gm1.reshape(1,hsize)
    gaussian1 = gmx1*gmy1




    dx = convolve2d(I,sobel,mode = 'same')
    dy = convolve2d(I,sobel.T,mode = 'same')
    dx = convolve2d(dx, gaussian1,mode = 'same')
    dy = convolve2d(dy, gaussian1,mode = 'same')
    dxdx = dx**2
    dxdy = dx*dy
    dydy = dy**2
    sigma2 = 0.2
    hsize = 2 * math.ceil(3 * sigma2) + 1
    gm2 = scipy.signal.gaussian(hsize,sigma2)
    gmx2 = gm2.reshape(hsize,1)
    gmy2 = gm2.reshape(1,hsize)
    gaussian2 = gmx2*gmy2

    dxdx = convolve2d(dxdx, gaussian2,mode = 'same')
    dxdy = convolve2d(dxdy, gaussian2,mode = 'same')
    dydy = convolve2d(dydy, gaussian2,mode = 'same')
    

    detm = np.multiply(dxdx,dydy) - np.multiply(dxdy,dxdy)
    tracem = np.add(dxdx,dydy)
    R = detm - k*(tracem**2)
    rl = R.flatten()

    points = np.argsort(rl)[::-1]
    points = points[len(points)-alpha:]
    xcor = np.unravel_index(points, R.shape)[0]
    ycor = np.unravel_index(points, R.shape)[1]

    res = np.concatenate((np.array(ycor),np.array(xcor)))
    res = res.reshape(2,alpha)

    points = res.T

    # ----------------------------------------------
    
    return points


