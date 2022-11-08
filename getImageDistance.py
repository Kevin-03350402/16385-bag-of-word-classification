import numpy as np
from utils import chi2dist

def get_image_distance(hist1, hist2, method):
    if method == 'euclidean':
        dist  = np.linalg.norm(hist1 - hist2)
    if method == 'chisq':
        dist = chi2dist(hist1, hist2)

    # -----fill in your implementation here --------


    # ----------------------------------------------

    return dist
