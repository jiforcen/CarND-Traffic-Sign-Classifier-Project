#**Traffic Sign Recognition** 

##Writeup - Juan Ignacio Forcén

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
[image6]: ./examples/ProbNim1.png "Traffic Sign 1"
[image7]: ./examples/ProbNim2.png "Traffic Sign 2"
[image8]: ./examples/ProbNim3.png "Traffic Sign 3"
[image9]: ./examples/ProbNim4.png "Traffic Sign 4"
[image10]: ./examples/ProbNim5.png "Traffic Sign 5"
[image11]: ./examples/ProbNim6.png "Traffic Sign 6"
[image12]: ./examples/ProbNim7.png "Traffic Sign 7"

[image20]: ./examples/ResultsNN.png "Results CNN"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####This submission includes a reference to the project code.

Here is a link to my [project code](https://github.com/jiforcen/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799.
* The size of test set is 12630.
* The shape of a traffic sign image is [32, 32].
* The number of unique classes/labels in the data set is 43.

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

In cell three of the IPython notebook is the code for visualize one random image of each class of the trainset.

![alt text][image1]

This bar chart showing the number of images per class is generated in cell four. We can see that number of images per class are not balanced.


![alt text][image2]


###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.
####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

Code for this steps is contained in cells from five to eight of the IPython notebook.

In the dataset are images with different conditions of luminance, to make easy for the neural network, images are corrected to similar conditions. To achieve that histogram of luminance of each image is equalized. For this task Images are converted from RGB Colorspace to YUV. Then Y component is equalized. (Functions: cv2.cvtColor and cv2.equalizeHist)

Also as we can see in the previous histogram, number of images in classes are not balanced, that may cause a biased result of the neural network trough the majority class. So I have create a function to augmentate data (Cell 5) inspired in opencv functions of the next link [geometric transformations](http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html)
(Functions used: cv2.getRotationMatrix2D, cv2.getAffineTransform and cv2.warpAffine). That functions returns a radomly transformed image from a previous one.

An example of both techniques with a randomi image is shown in cell 6.

![alt text][image4]

In the images before we can see a random image, random image equalized and this random image equalized and transformed.

In cell 8 data is augmented. New images are created in classes with less images than a minimun stablished with the variable('min_nImages'). This images are created with the function explained before.

This new train set is saved into a new file. In next cycle this images can be recovered to avoid the slow task of generate images. As we can see in the code, images are equalized before data augmentation, this is to avoid consider in the histogram black areas introduced by the transformation.

Now we can check the bar chart of images per clas again (cell 9), and see that new distribution of images per class.

![alt text][image3]

In cell 10 we can see the new dimensions of images sets. Now Trainset have 58860 images against the initial 34799 images.

Finaly train, valid and test data is normalized (cell 11) between -1 and 1, to give to the neural network images with uniform distributions.

####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the cell 13 of the ipython notebook. 

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

The code for define and train the model is located from cell 12 to 17. Concretely code for training the model is in cell 17.

To train the model, as a start point parameteres of LeNet practice were used, later a lot of different parameters were tested through different experiments. Finally next parameters were selected:

EPOCHS = 15
BATCH_SIZE = 100
mu = 0
sigma = 0.05  
learning_rate = 0.001

AdamOptimizer was used, also GradientDescentOptimizer was tried to change learning rate. But finally it was consider that AdamOptimizer work better changing learning rate automatically.

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

Code for calculating the accuracy of the model is located in cell 18 of the Ipython notebook.

First architecture tried was LeNet because it was implemented in the initial example. It was necessary to add dropout layers to prevent overfitting, because the inital neural network tends to overfit quickly, parameter keep_prob was adjusted experimentally at 0.65. 

Later a lot of different architectures were tried. Adding layers, adding second stage with different configurations (like in the article recomended in the project), changing the size of the convolutions, feeding images with and without color...
The best results were obtained with the proposed architecture.

Hyperparameters epochs, batch_size, mu, sigma, keep_prob and rate were adjusted through lot of experiments until determine the parameters exposed before.

The key was add one layer more, use convoution size of 3x3 instead of 5x5 and use features of 32, 64 and 128 in the layers 1, 2 and 3.

My final model results were:
* training set accuracy of 0.998
* validation set accuracy of 0.991
* test set accuracy of 0.974

Results can vary also depending on the parameters used for augment images.

![alt text][image20]


###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

In cell 19 we can seven signal from the web, here also:

![alt text][image5]

The seventh signals from the web have good conditions of brightnes, contrast and without oclusions. Only small changes in brightness are in images like the stop signal, which have bit more light in the bottom. So is expected to obtain good results in the model´s prediction in all the signals.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located cell 22 of the Ipython notebook.

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

The model was able to correctly guess all traffic signs, which gives an accuracy of 100%.

In test set this model have an accuracy of 97.4%, while in train set and validation set have more than 99%. After obtain this high accuracy in images with very different conditions in bright, contrast and size is normal have a 100% of accuracy clasifying this signals from internet which have very good conditions.

As it was expected the model classify correctly signals with standard conditions.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

In cell 25 we can see the softmax probabilities of each image, in cell 26 we can see also as bar chart representation.

For the first image, the model gives more probability to "Children crossing", as second option is "Bicycles crossing", but with a probability 6 times less.

![alt text][image6]


| Class number	| Description	  | Softmax probabilities|
|:-------------:|:-------------:|:--------------------:|
|28	|Children crossing|30.23|
|29	|Bicycles crossing|5.81|
|20	|Dangerous curve to the right|-4.69|
|15	|No vehicles|-6.42|
|27	|Pedestrians|-11.80|

For the second image, the model is completely sure that signal is "Roundabout mandatory", because is the unique option with positive probability.

![alt text][image7]

| Class number	| Description	  | Softmax probabilities|
|:-------------:|:-------------:|:--------------------:|
|40	|Roundabout mandatory|16.33|
|1	|Speed limit (30km/h)|-6.24|
|38|	Keep right|-7.23|
|35|	Ahead only|-7.37|
|11|	Right-of-way at the next intersection|-7.58|


In the the trird image, the model is also sure that signal is "Keep right", because is the unique option with positive probability.

![alt text][image8]

| Class number	| Description	  | Softmax probabilities|
|:-------------:|:-------------:|:--------------------:|
|38	|Keep right|18.95|
|34	|Turn left ahead|-3.16|
|40	|Roundabout mandatory|-3.21|
|12	|Priority road|-6.28|
|0	|Speed limit (20km/h)|-6.46|

In the the fourth image, the model is also sure that signal is "Stop", because is the unique option with positive probability.

![alt text][image9]

| Class number	| Description	  | Softmax probabilities|
|:-------------:|:-------------:|:--------------------:|
|14	|Stop|20.42|
|5	|Speed limit (80km/h)|-0.22|
|20	|Dangerous curve to the right|-6.16|
|6	|End of speed limit (80km/h)|-7.67|
|29	|Bicycles crossing|-7.97|

In the the fifth image, the model predict that signal is "Speed limit (60km/h)", as second option with less than a half of probability is "Speed limit (80km/h)" which is quite similar.

![alt text][image10]

| Class number	| Description	  | Softmax probabilities|
|:-------------:|:-------------:|:--------------------:|
|3	|Speed limit (60km/h)|7.92|
|5	|Speed limit (80km/h)|3.13|
|2	|Speed limit (50km/h)|-0.77|
|28	|Children crossing|-1.77|
|31	|Wild animals crossing|-5.60|

In the the sixth image, the model predict that signal is "Pedestrians", as second option with moreless five times less of probability is "Road narrows on the right" which can be similar.

![alt text][image11]

| Class number	| Description	  | Softmax probabilities|
|:-------------:|:-------------:|:--------------------:|
|27	|Pedestrians|14.06|
|24	|Road narrows on the right|3.11|
|11	|Right-of-way at the next intersection|2.72|
|19	|Dangerous curve to the left|1.98|
|26	|Traffic signals|0.78|

Finally in the the seventh image, the model predict that signal is "Right-of-way at the next intersection".

![alt text][image12]

| Class number	| Description	  | Softmax probabilities|
|:-------------:|:-------------:|:--------------------:|
|11	|Right-of-way at the next intersection|22.90|
|30	|Beware of ice/snow|4.27|
|27	|Pedestrians|0.61|
|18	|General caution|-1.99|
|28	|Children crossing|-8.03|

