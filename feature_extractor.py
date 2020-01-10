from keras.applications.inception_v3 import InceptionV3
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
from keras.optimizers import SGD, Adam
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
from keras.utils import np_utils, plot_model
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.xception import Xception, preprocess_input
from random import shuffle
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation
import numpy as np
from keras.preprocessing import image
import pandas as pd
import cv2


base_model = Xception(weights='imagenet', include_top=False, input_shape=(160, 160, 3))
# print(base_model.summary())
# model = Model(inputs=base_model.input, outputs=base_model.get_layer('block13_sepconv1').output)

data = pd.read_csv("train_faces_shuffled_vids.csv")

images = data["images_list"]
labels = data["label"]

train_data = []
train_label = []

count = 0

for img_path in images[:1]:
	# img = image.load_img(img_path, target_size=(224, 224))
	img = image.load_img(img_path)
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	features = base_model.predict(x)

	print(features.shape)
	features = features.reshape(features.shape[0], 
								features.shape[1] * features.shape[2],
								features.shape[3])
	print(features.shape)
# 	train_data.append(img)
# 	# train_label.append(j)
# 	train_label+=[j]

# 	if count%10000==0:
# 		print("Number of files done:", count)
# 	count+=1

# train_data = np.array(train_data)
# train_label = np.array(train_label)
# # lb = LabelBinarizer()
# # train_label = lb.fit_transform(train_label)
# train_label = np_utils.to_categorical(train_label)
# print(train_label)
# print(train_data.shape, train_label.shape)

# np.save('train_data_10k.npy', train_data)
# np.save('train_label_10k.npy', train_label)

# print("Files saved....")

