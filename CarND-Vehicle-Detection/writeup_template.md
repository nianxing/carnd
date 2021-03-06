**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car.png
[image2]: ./output_images/notcar.png
[image3]: ./output_images/carhog.png
[image4]: ./output_images/notcarhog.png
[image5]: ./output_images/slidewindow.png
[image6]: ./output_images/findcar.png
[image7]: ./output_images/heatmap.png
[video1]: ./project_video_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in lines 31 through 45 of the file called `vd_svm.py`.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]
![alt text][image2]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image3]
![alt text][image4]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and final decide to use :

```
orient = 9
pix_per_cell = 8
cell_per_block = 2
spatial_size = (32, 32)
hist_bins = 32
cspace = 'YCrCb'
hog_channel = "ALL"
```

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I create an array stack of feature vectors of hog features normalize the data, then randomnize it.

I split data into 80% to train and 20% to test. These approaches are in line 34 through 62 in file "vd_svm.py".

I trained a linear SVM using LinearSVC in line 71 through 74 of the file called "vd_svm.py"


###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?
I search window in four different overlapping and window sizes, the window sizes are: 256, 128, 96 and 48; the overlaping is (50%, 50%),(60%, 50%), (70%, 50%),(70%, 50%) respectively.  

![alt text][image5]

the code can be find in code of "vd_test.py", which also implemented in vd_video_pipline.py from line 42 to 112.

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image6]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are one frame of bounding box and  corresponding heatmap:

![alt text][image7]

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I used Linear SVM to do feature extraction and evaluation, it can achieve about 2fps and sometimes have overfit problem.
The algorithm's efficiency and accuracy need to be improve, which with my svm I think more labeled data will add more features to train may improve accuracy. With efficiency, I think there are more solutions, I will try nonlinear svm and neural_network implementations further in future to improve performance. 
