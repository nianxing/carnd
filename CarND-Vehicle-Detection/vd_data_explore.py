import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pickle
import cv2
import numpy as np
from lesson_functions import *

cars = glob.glob('./vehicles/*/*.png')
notcars = glob.glob('./non-vehicles/*/*.png')

a_car = plt.imread(cars[13])
not_a_car = plt.imread(noncars[13])

#plt.imshow(a_car)
#plt.imshow(not_a_car)
#plt.show()

#font_size = 15

orient = 9
pixels_per_cell = 8
cells_per_block = 2

colorspace = 'YCrCb'
hog_channels = "ALL"

features, hog_image = features_for_vis(a_car,
                                        cspace = colorspace,
                                        pix_per_cell = pixels_per_cell,
                                        cell_per_block = cells_per_block,
                                        hog_channel = hog_channels
                                       )
fig = plt.figure()
plt.subplot(1,len(hog_image)+1, 1)
plt.imshow(a_car)
plt.title('Car Image')
for idx, hog_img in enumerate(hog_image):
    plt.subplot(1,len(hog_image)+1, idx+2)
    plt.imshow(hog_img, cmap = 'gray')
    plt.title('HOG')


features, hog_image = features_for_vis(not_a_car,
                                        cspace = colorspace,
                                        pix_per_cell = pixels_per_cell,
                                        cell_per_block = cells_per_block,
                                        hog_channel = hog_channels
                                       )
fig = plt.figure()
plt.subplot(1,len(hog_image)+1, 1)
plt.imshow(not_a_car)
plt.title('Not Car')
for idx, hog_img in enumerate(hog_image):
    plt.subplot(1,len(hog_image)+1, idx+2)
    plt.imshow(hog_img, cmap = 'gray')
    plt.title('HOG')

plt.show()
