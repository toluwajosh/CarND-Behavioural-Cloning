#!/usr/bin/env python
"""
Steering angle prediction model
author: Joshua Owoyemi
for Udacity Self-driving car behavioural cloning project (project 3)
"""
## Import libraries
import os
import argparse
import json
import cv2
import numpy as np
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU, MaxPooling2D
from keras.regularizers import l2
from keras.layers.convolutional import Convolution2D

import matplotlib.pyplot as plt

# data paths
session_data = '../simulator-linux/session_data/driving_log.csv'
udacity_data = '../simulator-linux/udacity_data/data/driving_log_edit.csv'
path_prefix = '../simulator-linux/udacity_data/data/'

# define functions
def augment_brightness(image):
	"""
	apply random brightness on the image
	"""
	image = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
	random_bright = .25+np.random.uniform()
	
	# scaling up or down the V channel of HSV
	image[:,:,2] = image[:,:,2]*random_bright
	return image

def trans_image(image,steer,trans_range, trans_y=False):
	"""
	translate image and compensate for the translation on the steering angle
	"""
	
	rows, cols, chan = image.shape
	
	# horizontal translation with 0.008 steering compensation per pixel
	tr_x = trans_range*np.random.uniform()-trans_range/2
	steer_ang = steer + tr_x/trans_range*.4
	
	# option to disable vertical translation (vertical translation not necessary)
	if trans_y:
		tr_y = 40*np.random.uniform()-40/2
	else:
		tr_y = 0
	
	Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
	image_tr = cv2.warpAffine(image,Trans_M,(cols,rows))
	
	return image_tr,steer_ang


def crop_image(image, y1, y2, x1, x2):
	"""
	crop image into respective size
	give: the crop extent
	"""
	return image[y1:y2, x1:x2]


def im_process(image, steer_ang, train=True):
    """
    Apply processing to image
    """    
    # image size
    im_y = image.shape[0]
    im_x = image.shape[1]
    
    # translate image and compensate for steering angle
    trans_range = 50
    image, steer_ang = trans_image(image, steer_ang, trans_range) # , trans_y=True
    
    # crop image region of interest
    image = crop_image(image, 20, 140, 0+trans_range, im_x-trans_range)
    
    # flip image (randomly)
    if np.random.uniform()>= 0.5: #and abs(steer_ang) > 0.1
        image = cv2.flip(image, 1)
        steer_ang = -steer_ang
    
    # augment brightness
    image = augment_brightness(image)
    
    # perturb steering with a bias
    # steer_ang += np.random.normal(loc=0,scale=0.2)
    
    # image = cv2.cvtColor(image,cv2.COLOR_HSV2RGB)
    
    return image, steer_ang

## data generator
def generate_batch_samples(path, batch_size=128, path_prefix=''):
	while 1:
		batch_x, batch_y = [], []
		
		f = open(path)
		for line in f:
			
			# select image
			data_id = line.split(',')
			cam_view = np.random.choice(['center', 'left', 'right'])
			if cam_view == 'left':
				## left image
				image = plt.imread(path_prefix+data_id[1].strip())
				image, steering_angle = im_process(image, float(data_id[3])+.25)

			elif cam_view == 'center':
				## centre image
				image = plt.imread(path_prefix+data_id[0].strip())
				image, steering_angle = im_process(image, float(data_id[3]))

			elif cam_view == 'right':
				## right image
				image = plt.imread(path_prefix+data_id[2].strip())
				image, steering_angle = im_process(image, float(data_id[3])-.25)
			
			# resize image
			image = cv2.resize(image, (200,66))
			
			batch_x.append(np.reshape(image, (1,66,200,3)))
			batch_y.append(np.array([[steering_angle]]))
			
			if len(batch_x) == batch_size:
				# shuffle batch
				batch_x, batch_y, = shuffle(batch_x, batch_y, random_state=0)
				
				yield (np.vstack(batch_x), np.vstack(batch_y))
				batch_x, batch_y = [], []
	f.close()

def generate_batch_validate(path, batch_size=128, path_prefix=''):
	while 1:
		batch_x, batch_y = [], []
		
		f = open(path)
		for line in f:
			
			data_id = line.split(',')
			cam_view = np.random.choice(['center', 'left', 'right'])

			## use only center image for validation
			image = plt.imread(path_prefix+data_id[0].strip())
			steering_angle = float(data_id[3])
			
			# crop region of interest and resize to model input size
			image = crop_image(image, 20, 140, 50, 270)
			image = cv2.resize(image, (200,66))
			
			# change colourspace
			image = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
			
			batch_x.append(np.reshape(image, (1,66,200,3)))
			batch_y.append(np.array([[steering_angle]]))
			
			if len(batch_x) == batch_size:
				yield (np.vstack(batch_x), np.vstack(batch_y))
				batch_x, batch_y = [], []
	f.close()


## Model architecture
def nvidia_model(time_len=1):
	ch, row, col = 3, 66, 200  # camera format
	INIT='glorot_uniform' # 'he_normal', glorot_uniform
	keep_prob = 0.2
	reg_val = 0.01
	
	model = Sequential()
	model.add(Lambda(lambda x: x/127.5 - 1.,
			input_shape=(row, col, ch),
			output_shape=(row, col, ch)))
	model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode="valid", init=INIT, W_regularizer=l2(reg_val)))
	# W_regularizer=l2(reg_val)
	model.add(ELU())
	model.add(Dropout(keep_prob))

	model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="valid", init=INIT))
	model.add(ELU())
	model.add(Dropout(keep_prob))
	
	model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode="valid", init=INIT))
	model.add(ELU())
	model.add(Dropout(keep_prob))

	model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="valid", init=INIT))
	model.add(ELU())
	model.add(Dropout(keep_prob))

	model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="valid", init=INIT))
	model.add(ELU())
	model.add(Dropout(keep_prob))
	
	model.add(Flatten())

	model.add(Dense(100))
	model.add(ELU())
	model.add(Dropout(0.2))
	
	model.add(Dense(50))
	model.add(ELU())
	model.add(Dropout(0.2))
	
	model.add(Dense(10))
	model.add(ELU())
	
	model.add(Dense(1))

	model.compile(optimizer="adam", loss="mse") # , metrics=['accuracy']
	
	return model

# try to load a previously saved model
model_path = 'model_best'

# create model
model = nvidia_model()

try:
	model.load_weights(model_path+'.h5')
#     model.load_weights('checkpoints/nvidia_yuv_glorot_model-03-0.0187')
	
except IOError:
	print ('no previous model found....\n')


# Callbacks
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback

# a callback to save a list of the losses over each batch during training
class LossHistory(Callback):
	def on_train_begin(self, logs={}):
		self.train_loss = []

	def on_batch_end(self, batch, logs={}):
		self.train_loss.append(logs.get('loss'))


# a callback to save a list of the accuracies over each batch during training
class AccHistory(Callback):
	def on_train_begin(self, logs={}):
		self.train_acc = []
		
	def on_batch_end(self, batch, logs={}):
		self.train_acc.append(logs.get('acc'))

loss_hist = LossHistory()
acc_hist = AccHistory()
early_stop = EarlyStopping(monitor='val_loss', patience=3, 
						   verbose=0, mode='min')
checkpoint = ModelCheckpoint('checkpoints/'+model_path+'-{epoch:02d}-{val_loss:.4f}', 
							 monitor='val_loss',verbose=0, save_best_only=True, 
							 save_weights_only=False, mode='auto')



## Training and validation
EPOCHS = 30

# initialize generators
my_samples_gen = generate_batch_samples(session_data, path_prefix='', batch_size=128)
u_samples_gen = generate_batch_validate(udacity_data, path_prefix=path_prefix, batch_size=200)

# train model
model.fit_generator(
	my_samples_gen,
	samples_per_epoch=128*188, nb_epoch=EPOCHS,
	validation_data=u_samples_gen,
	nb_val_samples=24000,
	callbacks=[early_stop, checkpoint]
)

# save model
print("Saving model weights and configuration file...")

# if not os.path.exists("./outputs/steering_model"):
#     os.makedirs("./outputs/steering_model")

model.save_weights(model_path+'.h5', True)
with open(model_path+'.json', 'w') as outfile:
	json.dump(model.to_json(), outfile)