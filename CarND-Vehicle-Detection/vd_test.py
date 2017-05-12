import glob
import time

import cv2
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from scipy.misc import imresize
from skimage.feature import hog
from skimage.io import imread
from skimage.exposure import adjust_gamma
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from scipy.ndimage.measurements import label
from lesson_functions import *

svc = joblib.load('svc.pkl')
X_scaler = joblib.load('scaler.pkl')

image = mpimg.imread('./test_images/test6.jpg')
draw_image = np.copy(image)


ystart = 400
ystop = 656
scale = 1.5
orient = 9
pix_per_cell = 8
cell_per_block = 2
spatial_size = (32, 32)
hist_bins = 32
cspace = 'YCrCb'
hog_channel = "ALL"
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off

out_img = find_cars(image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)

plt.imshow(out_img)

window_configs = [
  {
    "x_start_stop": [None, None],
    "y_start_stop": [400, 656],
    "xy_window": (256, 256),
    "xy_overlap": (0.5, 0.5)
  },
  {
    "x_start_stop": [None, None],
    "y_start_stop": [400, 656],
    "xy_window": (128, 128),
    "xy_overlap": (0.6, 0.5)
  },
  {
    "x_start_stop": [100, 1280],
    "y_start_stop": [400, 500],
    "xy_window": (96, 96),
    "xy_overlap": (0.7, 0.5)
  },
  {
    "x_start_stop": [500, 1280],
    "y_start_stop": [400, 500],
    "xy_window": (48, 48),
    "xy_overlap": (0.7, 0.5)
  }
]

def process_image_heatmap(img):

  image = mpimg.imread(img)
  image = image.astype(np.float32)/255

  draw_image = mpimg.imread(img)
  window_image = mpimg.imread(img)
  heat = np.zeros_like(image[:,:,0]).astype(np.float)

  all_windows = []
  test_win = []
  for window_config in window_configs:

    x_start_stop = window_config.get('x_start_stop')
    y_start_stop = window_config.get('y_start_stop')
    xy_window = window_config.get('xy_window')
    xy_overlap = window_config.get('xy_overlap')

    windows = slide_window(image, x_start_stop=x_start_stop, y_start_stop=y_start_stop,
                      xy_window=xy_window, xy_overlap=xy_overlap)



    hot_windows = search_windows(image, windows, svc, X_scaler, color_space=cspace,
                          spatial_size=spatial_size, hist_bins=hist_bins,
                          orient=orient, pix_per_cell=pix_per_cell,
                          cell_per_block=cell_per_block,
                          hog_channel=hog_channel, spatial_feat=spatial_feat,
                          hist_feat=hist_feat, hog_feat=hog_feat)
    for wins in windows:
        test_win.append(wins)
    for window in hot_windows:
        all_windows.append(window)

  window_img = draw_boxes(image, test_win, color=(0, 0, 255), thick=6)
  # Add heat to each box in box list
  heat = add_heat(heat,all_windows)

  # Apply threshold to help remove false positives
  heat = apply_threshold(heat,1)

  # Visualize the heatmap when displaying
  heatmap = np.clip(heat, 0, 255)

  # Find final boxes from heatmap using label function
  labels = label(heatmap)
  draw_image = draw_labeled_bboxes(draw_image, labels)

  return draw_image, heatmap, window_img

for img in glob.glob('./test_images/*.jpg'):

    draw_img, heatmap, win = process_image_heatmap(img);
    '''
    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(draw_img)
    plt.title('Car Positions')
    plt.subplot(122)
    plt.imshow(heatmap, cmap='hot')
    plt.title('Heat Map')
    fig.tight_layout()
    '''
    plt.imshow(win)
plt.show()
