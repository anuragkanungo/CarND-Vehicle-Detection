**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./Car_hog.png
[image2]: ./NonCar_hog.png
[image3]: ./output_images/boxes_test1.jpg
[image4]: ./output_images/boxes_test2.jpg
[image5]: ./output_images/boxes_test3.jpg
[image6]: ./output_images/boxes_test4.jpg
[image7]: ./output_images/boxes_test5.jpg
[image8]: ./output_images/boxes_test6.jpg
[image9]: ./output_images/heat_map_test1.jpg
[image10]: ./output_images/heat_map_test2.jpg
[image11]: ./output_images/heat_map_test3.jpg
[image12]: ./output_images/heat_map_test4.jpg
[image13]: ./output_images/heat_map_test5.jpg
[image14]: ./output_images/heat_map_test6.jpg
[image15]: ./output_images/test1.jpg
[image16]: ./output_images/test2.jpg
[image17]: ./output_images/test3.jpg
[image18]: ./output_images/test4.jpg
[image19]: ./output_images/test5.jpg
[image20]: ./output_images/test6.jpg
[video1]: ./output_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

All of the code is in the `detect.py` file which has sections:
- Parameters
- Lesson Functions
- Modified Lesson Functions
- Added Functions and Pipeline Invocation


### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I started by reading in all the `vehicle` and `non-vehicle` images and calculate number of samples for each.

```
Number of car samples: 8792
Number of non car samples: 8968
```

Based on this the total number of samples are balanced so I don't have to worry about balancing the dataset.

Using the default get_hog_feature lesson function, I explored various color spaces such as YUV, LUV and YCrCb and tweaked the `orientations`, `pixels_per_cell`, and `cells_per_block` parameters. Also I visualized all the three channels.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=32`, `pixels_per_cell=(16, 16)` and `cells_per_block=(2, 2)`:

For Car Image:


![alt text][image1]

For NonCar Image:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

Based on various combinations on parameters, I went through the full pipeline to see which parameters works the best to detect the vehicles, also I looked through the hog features image to determine which has a better intensity and gradients. Mostly I add tweak the color space and hog channels. 


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained using a linear SVM with `hinge` loss to train. Before training I extracted all the car and non car features and generated the train/test dataset such that I can test the classifier performance. Based on the training I achieved :
```
Classifier Accuracy: 0.995213963963964
```

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I used the basic sliding window approach instead of scaled hog features sampling. To determine the best overlap I set the ystart/ystop parameters and overlap percentage and the xy_window size in the `slide_window` function. I found this parameters `xy_window=(128, 128), xy_overlap=(0.85, 0.85)` gave the best result.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

To optimize the performance of the classifier I have to improve the feature extraction by tweaking color space in hog features and hog channels. Also I enabled spatial binning and histogram features.

Here are some of the boxes:

![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]
![alt text][image7]
![alt text][image8]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:
![alt text][image9]
![alt text][image10]
![alt text][image11]
![alt text][image12]
![alt text][image13]
![alt text][image14]


### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image15]
![alt text][image16]
![alt text][image17]
![alt text][image18]
![alt text][image19]
![alt text][image20]
---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

