import cv2
import random
import math
import numpy as np


def augment_img(image2aug, method_selector=-1):
    #Applies one of three methods of image augmentation and returns the augmented image
	
	'''
		method_selector options:
		1 - perspective transformation
		2- image translation
		3- image rotation
	'''	
	if (method_selector == -1):
		#random msg to select which selector to use
		method_selector = random.randint(1,3)
		
	image_shape = image2aug.shape

	if(method_selector == 1):	
		#Apply prespective transfromation
		
		#getting random vertices of the image
		x1 = random.randint(0, math.ceil(image_shape[0]/8))
		x2 = random.randint(math.ceil(7 * image_shape[0]/8), image_shape[0])
		y1 = random.randint(0, math.ceil(image_shape[1]/8))
		y2 = random.randint(math.ceil(7 * image_shape[1]/8), image_shape[0])
		
		#applying prespective transformation
		pts1 = np.float32([[x1,y1],[x2,y1],[x1,y2],[x2,y2]])
		pts2 = np.float32([[0,0],[image_shape[0],0],[0,image_shape[1]],[image_shape[0],image_shape[1]]])
		M = cv2.getPerspectiveTransform(pts1,pts2)
		output_img = cv2.warpPerspective(image2aug,M,(image_shape[0],image_shape[1]))
		
	elif(method_selector == 2):
		#apply translation
		
		#getting random shifts for the image
		shift_x = random.randint(-math.ceil(image_shape[0]/8), math.ceil(image_shape[0]/8))
		shift_y = random.randint(-math.ceil(image_shape[1]/8), math.ceil(image_shape[1]/8))
		
		#applying translation
		M = np.float32([[1,0,shift_x],[0,1,shift_y]])
		output_img = cv2.warpAffine(image2aug,M,(image_shape[0],image_shape[1]))
		
	elif(method_selector == 3):
		#apply rotation
		
		#obtaining random angle to rotate
		angle2rotate = random.randint(-5, 5)
		
		#applying rotation
		M = cv2.getRotationMatrix2D((math.ceil(image_shape[0]/2),math.ceil(image_shape[1]/2)),angle2rotate,1)
		output_img = cv2.warpAffine(image2aug,M,(image_shape[0],image_shape[1]))
		
	return output_img
	
	
def rgb2gray(rgb_list):
	#Converts a list of rgb images to grayscale
	
	if (type(rgb_list) is not list) and (type(rgb_list) is not np.ndarray):
		rgb_list = [rgb_list]

	imgGray_list = []

	for img in rgb_list:
		grayCh = np.zeros([img.shape[0], img.shape[1], 1])
		grayCh[:,:,0] = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		imgGray_list.append(grayCh)
		
	return imgGray_list
	
def normalize(img_list, mean=128, std=128):
    #normalizes the pixel values of a list of images
	
	if (type(img_list) != list) and (type(img_list) != np.ndarray):                 
		img_list = [img_list]
		
	norm_list = []

	for img in img_list:
		norm_list.append(np.divide(img - mean, std))
		
	return norm_list