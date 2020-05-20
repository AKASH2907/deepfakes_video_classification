from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
import os.path
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.pooling import AveragePooling2D, MaxPooling2D
from keras.applications.resnet50 import ResNet50
from keras.layers.core import Dropout, Flatten, Dense
from keras.layers import Input
from keras.models import Model
from keras.optimizers import SGD, Adam, Nadam
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
from keras.utils import np_utils, plot_model
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.xception import Xception
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import warnings
import imageio.core.util
from facenet_pytorch import MTCNN
from PIL import Image
import pandas as pd
import cv2
import math
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.pooling import AveragePooling2D, MaxPooling2D
from keras.applications.resnet50 import ResNet50
from keras.layers.core import Dropout, Flatten, Dense
from keras.layers import Input, LSTM, Bidirectional, TimeDistributed
from keras.models import Model
from keras.optimizers import SGD, Adam, Nadam
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import pickle
from keras import utils
from keras.utils import np_utils, plot_model
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.xception import Xception
from random import shuffle
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation, GRU
import numpy as np
import time
from keras import backend as K
from keras.engine import InputSpec
from keras.engine.topology import Layer
import numpy as np


test_data = pd.read_csv('test_vids_label.csv')

videos = test_data["vids_list"]
true_labels = test_data["label"]

def ignore_warnings(*args, **kwargs):
    pass

imageio.core.util._precision_warn = ignore_warnings

# Create face detector
mtcnn = MTCNN(margin=40, select_largest=False, post_process=False, device='cuda:0')

testAug = ImageDataGenerator(rescale=1.0/255.0)


# def xception_model():
baseModel = Xception(weights="imagenet", include_top=False, input_shape=(160, 160, 3))
headModel = baseModel.output
headModel = MaxPooling2D(pool_size=(3, 3))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(512, activation="relu", kernel_initializer='he_uniform')(headModel)
headModel = Dense(512, activation="relu", kernel_initializer='he_uniform')(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(512, activation="relu", kernel_initializer='he_uniform')(headModel)
headModel = Dropout(0.5)(headModel)
predictions = Dense(2, activation="softmax", kernel_initializer='he_uniform')(headModel)
model = Model(inputs=baseModel.input, outputs=predictions)

for layer in baseModel.layers:
	layer.trainable = True

# opt = SGD(lr=1e-4, momentum=0.9, decay=1e-4 / EPOCHS)
# opt = Adam(lr=3e-4)
optimizer = Nadam(lr=0.002,
              beta_1=0.9,
              beta_2=0.999,
              epsilon=1e-08,
              schedule_decay=0.004)
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
	# return model


def lstm_model():
	main_input = Input(shape=(25, 2048), name="main_input")
	headModel = Bidirectional(GRU(512))(main_input)
	headModel = Dropout(0.2)(headModel)
	headModel = Dense(512, activation="relu", kernel_initializer='he_uniform')(headModel)
	predictions = Dense(2, activation="softmax", kernel_initializer='he_uniform')(headModel)
	model = Model(inputs=main_input, outputs=predictions)
	return model

# model = xception_model()


model.load_weights("trained_wts/xception_50_160.hdf5")
model_2 = Model(inputs=baseModel.input, outputs=baseModel.output)

model_lstm = lstm_model()

model_lstm.load_weights("xception_lstm.hdf5")

print("Weights loaded...")

y_pred = []
y_probabs = []
c = 0

for i in videos:
	# print(i)
	cap = cv2.VideoCapture(i)
	batches = []
	frameRate = cap.get(5) #frame rate
	mounting = 0
	while(cap.isOpened()):
		frameId = cap.get(1) #current frame number
		ret, frame = cap.read()
		if (ret != True):
			break
		# if (frame.any() != None):
		# 	frame = cv2.resize(frame, (320, 320), interpolation = cv2.INTER_AREA)			
		if ((frameId % math.floor(frameRate)) == 0 and frame.any() != None):
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			frame = Image.fromarray(frame)
			face = mtcnn(frame)
			
			try:
				# print(face.shape)
				# print(face)
				face = face.permute(1, 2, 0).int().numpy()
				# face = np.transpose(face, (1, 2, 0))
				# print(face.shape)
				# if mounting<16:
				batches.append(face)
			except AttributeError:
				print("Image Skipping")
			mounting+=1
	# print(mounting)
	batches = np.asarray(batches).astype('float32')
	batches /= 255
	# print(batches.shape)

	predictions = model_2.predict(batches)
	# print(predictions.shape)
	predictions = np.reshape(predictions, (predictions.shape[0], 
											predictions.shape[1]*predictions.shape[2],
											predictions.shape[3]))
	
	preds = model_lstm.predict(predictions)
	# print(preds)
	
	# Predict the output of each frame
	# axis =1 along the row and axis=0 along the column
	# print(predictions.argmax(1))
	pred_mean = np.mean(preds, axis=0)
	# print(pred_mean)
	# print(pred_mean)
	y_probabs+=[pred_mean]
	# print(y_probabs)
	y_pred+=[pred_mean.argmax(0)]
	
	cap.release()

	if c%10==0:
		print(c, "Done....")
	c+=1

# print(true_labels)
# print(y_pred)
# print(y_probabs)
print("Accuracy Score:", accuracy_score(true_labels, y_pred))
print("Precision Score", precision_score(true_labels, y_pred))
print("Recall Score:", recall_score(true_labels, y_pred))
print("F1 Score:", f1_score(true_labels, y_pred))

np.save("lstm_preds.npy", y_pred)
np.save("lstm_probabs.npy", y_probabs)
