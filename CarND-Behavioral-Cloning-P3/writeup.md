**Behavrioal Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./image/NVIDIA_CNN.png "Model Visualization"
[image2]: ./image/center.jpg "center"
[image3]: ./image/left.jpg "left Image"
[image4]: ./image/right.jpg "Right Image"
[image5]: ./image/YUV.png "chopped YUV image"
[image6]: ./image/center.jpg "Normal Image"
[image7]: ./image/flip.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

####3. Submssion code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model Architecture has been employed

My model based on NVIDIA's CNN - Paper:  End to End Learning for Self-Driving Cars.
The Architecture is shown on following image:
![alt text][image1]

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 141,144,150,153).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 42). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 119).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and I record both from center camera, left camera and right camera for me to get more data.

For details about how I created the training data, see the next section.

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to abstract the laneline feature from image without telling it and let the model figure itself to decide how to steer.

My first step was to use a convolution neural network model similar to the NIVIDIA's CNN  I thought this model might be appropriate because I tried several shallow networks(few layers similar to LeNet)  constructed by myself and as data number grows the advantage of NIVIDIA's CNN appear and performs good with fine tuning.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. with both my own CNN and NIVIDIA's CNN, I find it is hard to converge and the validation accuracy rate stay around 50%

Then I tried these two models in simulator, my CNN always tends to go left and drops and swim into the lake, NiVIDIA's CNN tends to go straight and result in climbing up the mountain.  

To improve the behavior of models, I augmented my data and did some treatment of my images. As the data grows and did some pre-processing car performs better and I will intoduce these procesure later.

After I tune my throttle rate to several numbers, at the end of the process, the vehicle is able to drive autonomously around the track without leaving the road with NIVIDIA's CNN model.

####2. Final Model Architecture

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded track one using center lane driving in both center camera, right camera, left camera mode:

![alt text][image2]
![alt text][image3]
![alt text][image4]

I randomly shuffled the data set and put 10% of the data into a validation set.

I preprocess my data by chopping of top 1/3 of the image and change image channel from RGB to YUV:
![alt text][image5]

To augment the training data set, I also flipped images and angles thinking that this would forbid model only learn to drive left because most roads are left turn. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

After the collection process, I had 16074 number of data points.
Here is a [link to my video result](./run3/output.mp4)

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 1000 as evidenced by my validation accuracy reach 98% after 1000 epochs. I used an adam optimizer so that manually training the learning rate wasn't necessary.
