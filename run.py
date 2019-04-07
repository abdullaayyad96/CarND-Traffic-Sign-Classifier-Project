import cv2
import os
import csv
import sys
import matplotlib.image as mpimg
import tensorflow as tf
import numpy as np
from image_edit import *

NN_model_meta = 'model/Traffic_Sign_Classifier_LeNet.meta'
NN_input_shape = [32, 32, 1]
CSV_file_dir = 'signnames.csv'

def load_classes(CSV_file):
	#Read the csv file with the meaning of each label
	
	#obtaining messages meaning from csv file
	signnames = csv.DictReader(open(CSV_file))
	signnameslist = []                            
	for row in signnames:
		signnameslist.append(row['SignName'])
		
	return signnameslist
	
	
def load_images(input_dir):
	#load jpgimages in a directory
	
	img_list = [] #list to store all images
	file_names = [] #list with all the filenames
	
	if (input_dir[-3:].upper() == "JPG"):	
		#image mode
		#reading images
		img = mpimg.imread(input_dir)		
		#resizing images to the shape of the training set
		img = cv2.resize(img, (NN_input_shape[0],NN_input_shape[1]))
		img_list.append(img)
		file_names.append(input_dir)
		
	else:	
		#read images in directory		
		file_names = os.listdir(input_dir)
		for file in file_names:
			if (file[-3:].upper() != "JPG"):
				file_names.remove(file)

		#rading images
		for file in file_names:
			img = mpimg.imread(input_dir + file)
			img = cv2.resize(img, (NN_input_shape[0],NN_input_shape[1]))
			img_list.append(img)
			
	return img_list, file_names
	
	
def NN_run(input_imgs):
	#run inference on trained NN model
	
	imgs_n = normalize(rgb2gray(input_imgs))
	
	#run evaluating model
	with tf.Session() as sess:
		saver = tf.train.import_meta_graph(NN_model_meta)
		saver.restore(sess, tf.train.latest_checkpoint('model')) 
		
		graph = tf.get_default_graph()
		x = graph.get_tensor_by_name('placeholder_x:0') 
		keep_prob = graph.get_tensor_by_name('keep_prob:0')
		logits = graph.get_tensor_by_name('logits_out:0')
			
		output = sess.run(logits, feed_dict={x: imgs_n, keep_prob: 1})
		
	return output
	
	
def classify(logits):
	#converting the logits obtained to a message list
	
	signnameslist = load_classes(CSV_file_dir)
	
	output_msg = []
	for i in range(logits.shape[0]):
		output_msg.append(signnameslist[np.argmax(logits[i])])
		
	return output_msg

	
def save_output(image_list, class_list, output_dir):

	print('output_classes are:')
	if len(image_list) == len(class_list):
		for n in range(len(image_list)):
			print(str(n) + ':\t' + class_list[n])
			
			mpimg.imsave((output_dir + str(n) + '_' + class_list[n] + '.jpg'), image_list[n])
	else:
		print("length of image list and output class list is not equal")
		sys.exit()			
		
		
def main(argvs):
	
	input_dir = ''	
	output_dir = './'
	
	#read input & output directories
	if (len(argvs) >= 2):	
		input_dir = argvs[1]
		
		if (len(argvs) >= 3):
			output_dir = argvs[2]
		
		if (input_dir[-1] != "/") and (input_dir[-3:].upper() != "JPG"):
			input_dir += "/"
		
		if (output_dir[-1] != "/") and (output_dir[-3:].upper() != "JPG"):
			output_dir += "/"
	else:
		print("2 arguments are required, only %s were provided" % (len(argvs)-1))
		sys.exit()	
		
	img_list, file_names = load_images(input_dir)	
	
	logits = NN_run(img_list)
	
	output_classes = classify(logits)
	
	save_output(img_list, output_classes, output_dir)

if __name__ == "__main__":
	main(sys.argv)
	sys.exit()