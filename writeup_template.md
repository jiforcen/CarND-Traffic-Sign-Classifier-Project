#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/ImagesExamples.png "Example images per class"
[image2]: ./examples/HistogramOriginal.png "Histogram of the train data set"
[image3]: ./examples/HistogramAugmented.png "Histogram of augmented train dataset"
[image4]: ./examples/ImageTransformation.png "Image equalization and transformation"
[image5]: ./examples/NewImages.png "New Images"
[image6]: ./examples/ProbNIm1.png "Traffic Sign 1"
[image7]: ./examples/ProbNIm2.png "Traffic Sign 2"
[image8]: ./examples/ProbNIm3.png "Traffic Sign 3"
[image9]: ./examples/ProbNIm4.png "Traffic Sign 4"
[image10]: ./examples/ProbNIm5.png "Traffic Sign 5"
[image11]: ./examples/ProbNIm6.png "Traffic Sign 6"
[image12]: ./examples/ProbNIm7.png "Traffic Sign 7"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/jiforcen/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is [32, 32]
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

In cell three is showed one random image of the Train Set per class, so we can see that different signals of the dataset.

![alt text][image1]

Here is an exploratory visualization of the data set. It is a bar chart showing the number of images per class. We can see that number of images per class are not balanced.


![alt text][image2]


###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this steps is contained in cells from five to eight of the IPython notebook.

In the dataset are images with different conditions of luminance, to make easy for the neural network images are corrected to similar conditions. That is the reason because the histogram of luminance of each image is equalized. For this task Images are converted from RGB Colorspace to YUV. Then Y component is equalized. (Functions: cv2.cvtColor and cv2.equalizeHist)

Also as we can see in the previous histogram, number of images in classes are not balanced, that may cause a biased result trough the majority class. So I have create a function to augmentate data (Cell 5) inspired in opencv functions of the next link [geometric transformations](#http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html)
(Functions used: cv2.getRotationMatrix2D, cv2.getAffineTransform and cv2.warpAffine)

An example of both techniques is shown in cell 10.

![alt text][image4]

In the image before we can see a random image, random image equalized and this random image equalized and transformed.

In cell 12 we augmentate the data wit less images than the minimun value and save it into a new file. In next executions this images can be recovered to avoid this slow task. As we can see in the code images are equalized before transform to avoid consider in the histogram black areas introduced in the transformation.

Now we can check the histogram again (cell xx) and see the new distribution of images per class.

![alt text][image3]

Finaly train, valid and test data is normalized (cell xx) between -1 and 1, to give to the neural network images with uniform distributions.

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)



####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the seventh cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 30x30x32 	|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 28x28x32 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, outputs 14x14x32                	|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 12x12x64 	|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 10x10x64 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 8x8x128 	|
| Convolution 3x3     	| 1x1 stride, same padding, outputs 6x6x128 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 3x3x128              	|
| Convolution 3x3	    | etc.      									|
| Fully connected		| input 1152, output 400.        				|
| RELU					|												|
| Fully connected		| input 400, output 84.           				|
| RELU					|												|
| Fully connected		| input 84, output 43.           				|


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the eigth cell of the ipython notebook. 

To train the model, I used an ....

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image5]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			                        |     Prediction	        					| 
|:-------------------------------------:|:---------------------------------------------:| 
| Children crossing     		        | Children crossing  							| 
| Roundabout mandatory     			    | Roundabout mandatory 							|
| Keep right					        | Keep right									|
| Stop	      		                    | Stop					 			          	|
| Speed limit (60km/h)		            | Speed limit (60km/h)      					|
| Pedestrians		                    | Pedestrians      						    	|
| Right-of-way at the next intersection	| Right-of-way at the next intersection    		|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 
