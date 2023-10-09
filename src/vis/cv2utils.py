import cv2
import numpy as np

def black_image(size):
    return np.zeros((size[0], size[1], 3), dtype = np.uint8)