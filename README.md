# Vehicle-Detection-and-Tracking-Udacity-CarND-T1-Project5


### This project to detect and Track vehicles based on machine learning algorithm. 


---
![](https://github.com/emilkaram/Vehicle-Detection-and-Tracking-Udacity-CarND-T1-Project5/blob/master/images/13.png)

**Vehicle Detection Project**

The goals / steps of this project are the following:

* I performed a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier

Here are my training results and accuracy measured on the test dataset

Using: 9 orientations 8 pixels per cell and 8 cells per block

Feature vector length: 4896

19.21 Seconds to train SVC

Test Accuracy of SVC =  0.9865

  
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

#### 1. Here i will explain how I extracted HOG features from the training images.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![](https://github.com/emilkaram/Vehicle-Detection-and-Tracking-Udacity-CarND-T1-Project5/blob/master/images/1.png)

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

and here are my final parameters:

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


Here is an example using the above HOG parameters:


![](https://github.com/emilkaram/Vehicle-Detection-and-Tracking-Udacity-CarND-T1-Project5/blob/master/images/2.png)


#### 2. Here I settled on final choice of HOG parameters.

I tried various combinations of parameters along with spatial and histogram image features and tuned the parmeters till i got a good ML model that can accuratly predit cars in images.

#### 3. Here I will describe how I trained a classifier using HOG features and color features.

I trained a linear SVM using labled car and non car dataset
after extracting features from the images and normiliaze the combined features (HOG , color , spatial ) i split and randamoize my dataset to be 80% training set and 20% testing set

Here are my training results and accuracy measured on the test dataset

Using: 9 orientations 8 pixels per cell and 8 cells per block

Feature vector length: 4896

19.21 Seconds to train SVC

Test Accuracy of SVC =  0.9865


### Sliding Window Search

#### 1. Here I will describe how I implemented a sliding window search and How did I decide what scales to search and how much to overlap windows

I decided first to use fixed size windows with some overlaps(trired diffrent size and overlap ratios) the result was ok but not good enough and consistent with car size and posion in the frame 
Then i decide to use diffrent size windows with a scale.
The scale factor was set on different regions of the image (e.g. small near the horizon, larger in the center).
here are some examples I used for sliding windows:
   
    ystart = 400
    ystop = 464
    scale = 1.0
     
    ystart = 416
    ystop = 480
    scale = 1.0
     
    ystart = 400
    ystop = 496
    scale = 1.5
         
    ystart = 432
    ystop = 528
    scale = 1.5
        
    ystart = 400
    ystop = 528
    scale = 2.0
       
    
    ystart = 432
    ystop = 560
    scale = 2.0
   
    ystart = 400
    ystop = 596
    scale = 3.5
     
    ystart = 464
    ystop = 660
    scale = 3.5
     
Here an expample of sliding window search:

![](https://github.com/emilkaram/Vehicle-Detection-and-Tracking-Udacity-CarND-T1-Project5/blob/master/images/6.png)

#### 2. Here some examples of test image to demonstrate how my pipeline is working:

I used YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![](https://github.com/emilkaram/Vehicle-Detection-and-Tracking-Udacity-CarND-T1-Project5/blob/master/images/4.png)

![](https://github.com/emilkaram/Vehicle-Detection-and-Tracking-Udacity-CarND-T1-Project5/blob/master/images/4.png)
---

### Video Implementation

I used two methods for pipline:
Basic method
Calculating HOG for every sliding window in every frame , the method got a good results but slow , so i used advanced method to speed up the alogrithm and be able to be used in realtime detection

Adanced method
A more efficient method for doing the sliding window approach, one that allows me to only have to extract the Hog features once.
The find_cars funation only has to extract hog features once, for each of a small set of predetermined window sizes (defined by a scale argument), and then can be sub-sampled to get all of its overlaying windows. 
Each window is defined by a scaling factor that impacts the window size. 
The scale factor was set on different regions of the image (e.g. small near the horizon, larger in the center).

#### 1. Here a link to my final video output.  my pipeline performed reasonably well on the entire project video with minimal false positives.
Here's a link to my video result

https://github.com/emilkaram/Vehicle-Detection-and-Tracking-Udacity-CarND-T1-Project5/blob/master/project_video_output.mp4




#### 2. Here I will describe how I implemented filter for false positives and method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  Also loged some histoy of the boxes so i can keep track of the car movement and able to combine boxes in case of detecting multiple cars  this end up with very good resulsts as shown in the project video. 

Here's an example result showing the heatmap and the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:


![](https://github.com/emilkaram/Vehicle-Detection-and-Tracking-Udacity-CarND-T1-Project5/blob/master/images/5.png)


![](https://github.com/emilkaram/Vehicle-Detection-and-Tracking-Udacity-CarND-T1-Project5/blob/master/images/7.png)

### Here the resulting bounding boxes:
![](https://github.com/emilkaram/Vehicle-Detection-and-Tracking-Udacity-CarND-T1-Project5/blob/master/images/8.png)



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?


When implemented the pipline on the video in the begining i used fixed size sliding windows I got good results for predicting the car position but with jittery and not consistent boxes drwan around the cars , i improve my pipline by using diffrent size sliding windows with diffrent scale factor based on the expected car position (e.g. small near the horizon, larger in the center) also loged some histoy of the boxes so i can keep track of the car movement and end up with very good resulsts as shown in the project video:
![](https://github.com/emilkaram/Vehicle-Detection-and-Tracking-Udacity-CarND-T1-Project5/blob/master/project_video_output.mp4)

only one spot it show false positive near a side road sign borad i may improve this by enahncing my filter thershold.

in Future I will combine  vehicle detection pipeline with the lane finding implementation from my last project

  




Disclaimer:Some of the fucntions i used are from Udacity self driving car lectures
