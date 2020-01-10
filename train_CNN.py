from keras.preprocessing.image import ImageDataGenerator
from keras.layers.pooling import AveragePooling2D, MaxPooling2D
from keras.applications.resnet50 import ResNet50
from keras.layers.core import Dropout, Flatten, Dense
from keras.layers import Input
from keras.models import Model
from keras.optimizers import SGD, Adam, Nadam
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import pickle
from keras.utils import np_utils, plot_model
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.xception import Xception
from random import shuffle
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation
import numpy as np

EPOCHS = 50

# Dataset subset loading
train_data = np.load('train_subsets/data/train_data_VII.npy')
train_label = np.load('train_subsets/labels/train_label_VII.npy')
print("Dataset Loaded...")

trainX, valX, trainY, valY = train_test_split(train_data, train_label, shuffle=False)
print(trainX.shape, valX.shape, trainY.shape, valY.shape)

# Training data augmentation
trainAug = ImageDataGenerator(
	rescale=1.0/255.0,
	rotation_range=30,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

valAug = ImageDataGenerator(rescale=1.0/255.0)

# Defining model
baseModel = Xception(weights="imagenet", include_top=False, input_shape=(75, 75, 3))
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

# Print total number of trainable parameters
trainable_count = int(
    np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
non_trainable_count = int(
    np.sum([K.count_params(p) for p in set(model.non_trainable_weights)]))

print('Total params: {:,}'.format(trainable_count + non_trainable_count))
print('Trainable params: {:,}'.format(trainable_count))
print('Non-trainable params: {:,}'.format(non_trainable_count))

optimizer = Nadam(lr=0.002,
                  beta_1=0.9,
                  beta_2=0.999,
                  epsilon=1e-08,
                  schedule_decay=0.004)
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

# Callbacks
model_checkpoint = ModelCheckpoint('xception_50_75.hdf5', monitor='val_loss',verbose=1, save_best_only=True,
	save_weights_only=True)
stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0)

print("Training is going to start in 3... 2... 1... ")

H = model.fit_generator(
	trainAug.flow(trainX, trainY, batch_size=64),
	steps_per_epoch=len(trainX) // 64,
	validation_data=valAug.flow(valX, valY),
	validation_steps=len(valX) // 64,
	epochs=EPOCHS,
	callbacks=[model_checkpoint, stopping]
	)

