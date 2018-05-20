## Vehicle Detection and Tracking  Udacity CarND T1 Project5
### This project to detect and Track vehicles based on machine learning algorithm. 

---
![](https://github.com/emilkaram/Vehicle-Detection-and-Tracking-Udacity-CarND-T1-Project5/blob/master/images/13.png)

**Vehicle Detection Project**

The goals / steps of this project are the following:

* I performed a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* I also appllied a color transform and append binned color features, as well as histograms of color, to HOG feature vector. 
* Then normalized all features and randomize a selection/splitting for training and testing.
* I implemented a sliding-window technique and used my trained classifier to search for vehicles in images.
* Ran my pipeline on a video stream (started with the test_video.mp4 and later implemented on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimated a bounding box for vehicles detected.


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. I provide a Writeup / README that includes all the rubric points and how I addressed each one.   


### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook (or in lines # through # of the file called `some_file.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![](https://github.com/emilkaram/Vehicle-Detection-and-Tracking-Udacity-CarND-T1-Project5/blob/master/images/1.png)

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

and here are my final parameters:
#parameters
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 8 # HOG cells per block
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off


Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![](https://github.com/emilkaram/Vehicle-Detection-and-Tracking-Udacity-CarND-T1-Project5/blob/master/images/2.png)


#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and...

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using...

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![](https://github.com/emilkaram/Vehicle-Detection-and-Tracking-Udacity-CarND-T1-Project5/blob/master/images/6.png)

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![](https://github.com/emilkaram/Vehicle-Detection-and-Tracking-Udacity-CarND-T1-Project5/blob/master/images/4.png)

![](https://github.com/emilkaram/Vehicle-Detection-and-Tracking-Udacity-CarND-T1-Project5/blob/master/images/4.png)
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)
![](https://github.com/emilkaram/Vehicle-Detection-and-Tracking-Udacity-CarND-T1-Project5/blob/master/project_video_output.mp4)



#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![](https://github.com/emilkaram/Vehicle-Detection-and-Tracking-Udacity-CarND-T1-Project5/blob/master/images/5.png)

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![](https://github.com/emilkaram/Vehicle-Detection-and-Tracking-Udacity-CarND-T1-Project5/blob/master/images/7.png)

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![](https://github.com/emilkaram/Vehicle-Detection-and-Tracking-Udacity-CarND-T1-Project5/blob/master/images/8.png)



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?


When implemented the pipline on the video in the begining i used fixed size sliding windows I got good results for predicting the car position but with jittery and not consistent boxes drwan around the cars , i improve my pipline by using diffrent size sliding windows with diffrent scale factor based on the expected car position also looged some histoy of the boxes so i can keep track of the car movment and end up with very good resulsts as shown in the project video:
![](https://github.com/emilkaram/Vehicle-Detection-and-Tracking-Udacity-CarND-T1-Project5/blob/master/project_video_output.mp4)

only one spot it show false positive near a side road sign borad i may improve this by enahncing my filter thershold.

in Future I will combine  vehicle detection pipeline with the lane finding implementation from my last project

  

