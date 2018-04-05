# **Traffic Sign Recognition** 


**Build a Traffic Sign Recognition Project**


This is my writeup on my submission for the traffic sign classifier project as part of the Udacity Self Driving Car Nanodegree. The project code can be found here: [project code](https://github.com/abdullaayyad96/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./Images/sample_data.jpg "Visualization"
[image2]: ./Images/training_set_bar.jpg "Training Set Distribution"
[image3]: ./Images/data_aug_sample.jpg " Augmentation"
[image4]: ./Images/training_augmented_Set_bar.jpg "Distribution after Augmentation"
[image5]: ./Images/data_gray_sample.jpg "Grayscaling"
[image6]: ./Images/testing_set_sample.jpg "Testing Results"
[image7]: ./Images/additional_images.jpg "Additional Images for testing"
[image8]: ./Images/addtional_images_result.jpg "Classification results for additional images"
[image9]: ./Images/softmax_probabilities.jpg "Softmax probabilites for the Additional images"
[image10]: ./Images/vis_sample_img.jpg "Sample Image for NN Visualization"
[image11]: ./Images/first_covnet_lay.jpg "First CNN Layer Visualization"
[image12]: ./Images/second_covnet_lay.jpg "Second CNN Layer Visualization"


## Rubric Points
### This project was implemented to meet the rubric found here: [rubric points](https://review.udacity.com/#!/rubrics/481/view) 


### Data Set Summary & Exploration

#### 1. Summary of the data set. 

Python 3.0  built in functions as well as Numpy library were used to calculate basic statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43


#### 2. Visualization of the dataset.

Samples from the training set can be seen below:

![alt text][image1]

The bar chart below shows the distribution of the training set data across the 43 possible classes. It can easily be observed that some labels are more represented than others.

![alt text][image2]

### Design and Test a Model Architecture

#### 1. Preprocessing the data. 

##### 1.1 Augmentation

As a first step, I decided to apply augmentation due to the high degree of differance in represenation between different classes. Three techniques were used to generate additional data: Prespective transformation, translation and rotation. These methods are randomly applied in order to have a minimum number of training samples for each class. A sample outcome of the augmentation process can be seen here:

![alt text][image3]

Additionally, the bar chart below shows the new distribution of training set upon augmentation:

![alt_text][image4]

##### 1.2 Shuffling 

Prior to proceeding with preprocessing and training process, training data were shuffled. It is worth noting that training data is also shuffled during the training process making the shuffling here slightly redundant. 

##### 1.3 Grayscaling

The third step is grayscaling the training set; since grayscaling was observed to enhance the performance of the NN. The following figure shows few samples of the grayscaling process.

[!alt_text][image5]

##### 1.4 Grayscaling

As a last step, the training, validation and testing set images were normalized as normalization showed great perfomance enhancements. The normalization method was straigh forward by using the following equation: (pixel_value - 128)/128

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | etc.      									|
| Fully connected		| etc.        									|
| Softmax				| etc.        									|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

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

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


