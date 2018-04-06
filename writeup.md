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
[image3]: ./Images/data_aug_sample.jpg "Augmentation"
[image4]: ./Images/training_augmented_set_bar.jpg "Distribution after Augmentation"
[image5]: ./Images/data_gray_sample.jpg "Grayscaling"
[image6]: ./Images/testing_set_sample.jpg "Testing Results"
[image7]: ./Images/additional_images.jpg "Additional Images for testing"
[image8]: ./Images/additional_images_result.jpg "Classification results for additional images"
[image9]: ./Images/softmax_probabilities.jpg "Softmax probabilites for the Additional images"
[image10]: ./Images/vis_sample_img.jpg "Sample Image for NN Visualization"
[image11]: ./Images/first_covnet_layer.png "First CNN Layer Visualization"
[image12]: ./Images/second_covnet_layer.png "Second CNN Layer Visualization"


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

As a first step, I decided to apply augmentation due to the high degree of differance in represenation between different classes. Three techniques were used to generate additional data: Prespective transformation, translation and rotation. The aforementiond augmentation methods were applied using the openCV library and were randomly applied in order to have a minimum number of training samples for each class. A sample outcome of the augmentation process can be seen here:

![alt text][image3]

Additionally, the bar chart below shows the new distribution of training set upon augmentation:

![alt_text][image4]

##### 1.2 Shuffling 

Prior to proceeding with preprocessing and training process, training data were shuffled. It is worth noting that training data is also shuffled during the training process making the shuffling here slightly redundant. 

##### 1.3 Grayscaling

The third step is grayscaling the training set; since grayscaling was observed to enhance the performance of the NN. The following figure shows few samples of the grayscaling process.

![alt_text][image5]

##### 1.4 Grayscaling

As a last step, the training, validation and testing set images were normalized as normalization showed great perfomance enhancements. The normalization method was straigh forward by using the following equation: (pixel_value - 128)/128

#### 2. Model architecture

In my code, I've implemented and tested two architecture. The first is the LeNet architecture with the addition of Dropout stage for the fully connected layers as can be seen below:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| HxWxC image   							| 
| Convolution     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 window, 2x2 stride,  outputs 14x14x6 				|
| Convolution 	    | 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU   |               |
| Max pooling	      	| 2x2 window, 2x2 stride,  outputs 5x5x16 				|
| Flatten       |   outputs 400   |
| Fully connected		|  outputs 120    |
| RELU      |          |
| Dropout   |         |
| Fully connected		|  outputs 84    |
| RELU      |          |
| Dropout   |         |
| Classifier		|  outputs (number of classes)    |


Another architecture was also implemented. This one follows a similar architecture to the one described in this [paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) In which the outout of two convolutional layers are fed into fully connected layer as seen below:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| HxWxC image   							| 
| Convolution     	| 1x1 stride, valid padding, outputs 28x28x30 	|
| RELU					|												|
| Max pooling	      	| 3x3 window, 2x2 stride,  outputs 14x14x30 				|
| Convolution 	    | 1x1 stride, valid padding, outputs 10x10x64 	|
| RELU    |               |
| Max pooling	      	| 2x2 stride,  outputs 5x5x64 				|
| Flatten       |   outputs: 5880 + 1600   |
| Concatenate | outputs 7480  |
| Fully connected		|  outputs 120    |
| RELU      |          |
| Dropout   |         |
| Classifier		|  outputs (number of classes)    |

Upon testing, the second architecture yielded better results thus it was chosen for the final design. The LeNet architecture is still present in the code for referance and possible future enhancements.

#### 3. Model Optimizer & Parameters

The optimizer utilized in this project is the standard [Adam Optimizer](https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer) included in the tensor flow library.
The hyperparameters were tuned in a trial and error method and the final used parameters are:
Batch size: 128
EPOCHS: 20
Learning rate: 0.0005

Past commits prior to "Deleted unnecessary files" Include HTML files indicating the performance under different parameters.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.996
* validation set accuracy of 0.966
* test set accuracy of 0.948

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

The figure below shows random samples of the testing set along with the NN model output:

![alt_text][image6]

### Testing the Model on New Images

#### 1. Choose new German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Below are the German traffic signs I found on the web:

![alt text][image7]

The images used are fairly clear in order to better understand the strengths/weakness of NN in classifying different types of signs rather than estimation under differernt conditions.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

![alt_text][image8]

The model was able to correctly guess 8 of the 10 traffic signs, which gives an accuracy of 80%. This is not an ideal accuracy compared to the testing accuracy of 94.8%. Nonetheless, The mistakes that occured were in detecting the actual numbers on the speed signs even though it identified the images as speed signs. This clearly indicates the main shortcoming of the NN model as number classification as it performed perfectly otherwise. Possible improvement that can be done in the future is seperately applying the LeNet architecture on images that were identified as speed signs. 

#### 3. Describe how certain the model is when predicting on each of the  new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 28th and 29th cell of the Ipython notebook. The softmax probabilities for classifications can be seen below:

![alt_text][image9]

The model is fairly certain of all classification even for the incorrect cases. In general, the poorest performance is  the speed sign number identification, which might indicate the unsuitability of the model for number identification.


#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

A random image from the training set was used to visualize the convolutional layers of the NN model. Below are the sample image used along with the output of the first and second convolutional layers:

![alt_text][image10]

* First covnet:
![alt_text][image11]

* Second covnet:
![alt_text][image12]




