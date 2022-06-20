import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import csv



# https://stackoverflow.com/questions/3279560/reverse-colormap-in-matplotlib
from PIL import Image


def return_gaussian(pts):
    sigma = 3
    peaks_img = np.zeros((500, 500))
    peaks_img[np.int_(pts[:, 1]), np.int_(pts[:, 0]) ] = 1
    density_img = cv2.GaussianBlur(peaks_img, (0, 0), sigma)
    return density_img


image = return_gaussian(np.array([[250, 250]]))
plt.imshow(image, cmap='Greys_r')
plt.show()
