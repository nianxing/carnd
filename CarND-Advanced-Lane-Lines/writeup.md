
**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/figure_2.png "Undistorted"
[image2]: ./output_images/imagetest.png "Road Transformed"
[image3]: ./output_images/image_ud.png "Undistorted test"
[image4]: ./output_images/masked.png "Binary Example"
[image5]: ./output_images/warped.png "Warp Example"
[image6]: ./output_images/fit.png "Fit Visual"
[image7]: ./output_images/result.png "Output"
[video1]: ./project_video_output.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the cali.py.  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![alt text][image1]

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.

I first computed the camera distortion matrix and distortion parameters using the provided chessboard images. Then loaded the camera matrix, and use this to undistort images. Figure below presents the result of undistortion. This part of code is in laneline.py undistort_image(img, mtx, dist) function.
The distorted image is presented below:
![alt text][image2]

And the undistorted image is presented below:
![alt text][image3]

####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
In laneline.py from line 146 to line 157 I first change image from RGB to HSV channel, then give it an threshold to find yellow line, and white line. Then in laneline.py from line 159 to line 172 I applyed sober filter based on grayscaled image to find lanelines. Then I combine them. In laneline.py from line 181 to line 184 I used moving window to remove effects like car on combined image.

The result of the binary image is:

![alt text][image4]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 79 through 84 in the file `laneline.py` .  The `warp_image()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```
  ht_window = np.uint(img_size[0]/1.5)
  hb_window = np.uint(img_size[0])
  c_window = np.uint(img_size[1]/2)
  ctl_window = c_window - .25*np.uint(img_size[1]/2)
  ctr_window = c_window + .25*np.uint(img_size[1]/2)
  cbl_window = c_window - 1*np.uint(img_size[1]/2)
  cbr_window = c_window + 1*np.uint(img_size[1]/2)
  src = np.float32([[cbl_window,hb_window],[cbr_window,hb_window],
                    [ctr_window,ht_window],[ctl_window,ht_window]])
  dst = np.float32([[0,img_size[0]],[img_size[1],img_size[0]],
                [img_size[1],0],[0,0]])

```

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image5]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Orderly, I first did transform part to get a warped image, then I did masking part to get a binary_output of image which only contains recognized lanelines part. Then I fit my lane lines with a 2nd order polynomial in laneline.py from line to line kinda like this:

![alt text][image6]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

In a given curve, the radius of curvature in some point is the radius of the circle that “kisses” it, or osculate it — same tangent and curvature at this point.
This link has a great tutorial on it.
http://www.intmath.com/applications-differentiation/8-radius-curvature.php
The radius of curvature is given by following formula.
Radius of curvature=​​ (1 + (dy/dx)** 2)** 1.5 / abs(d2y /dx2)
Calculate the radius for both lines, left and right, and the chosen point is the base of vehicle, the bottom of image.
x = ay2 + by + c
Taking derivatives, the formula is:
radius = (1 + (2a y_eval+b)** 2)** 1.5 / abs(2a)
in the point where x = 0, represented as y_eval in the formula.
Another consideration. The image is in pixels, the real world is in meters. We have to estimate the real world dimension from the photo.
I’m using the estimative provide by class instructions:
30/720  meters per pixel in y dimension
Applying correction and formula, I get the curvature for each line.
The offset to the center of lane.
We assume the camera is mounted exactly in the center of the car.
Thus, the difference between the center of the image (img_size /2) and the middle point of beginning of lines if the offset (in pixels). This value times conversion factor is the estimate of offset.
which in code laneline.py: line 283, 284.

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 288 through 292 in my code in `laneline.py` by the opencv function cv2.warpPerspective().  Here is an example of my result on a test image:

![alt text][image7]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_output.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

The code is following the tutorial in class step by step.
This pipeline performs bad in challenge_video. because the segmentation performance is not good enough due to lighting condition is not good and ground is not ideal as before. what is more in the harder mode, it is hard for polynomial regression for such a big curve.
In the future, adding more fitting modes for choice might be an important method for me finish the harder challenge mode. What is more, for different condition of road and lighting it might need some learning function to find lane line out if there are enough data.
