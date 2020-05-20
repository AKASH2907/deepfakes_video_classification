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
from keras import utils
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.xception import Xception
from keras import backend as K
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
import numpy as np
import time
from keras import backend as K
from keras.engine import InputSpec
from keras.engine.topology import Layer
import numpy as np


def ignore_warnings(*args, **kwargs):
    pass


def cnn_model(model_name, img_size, weights):
    """
    Model definition using Xception net architecture
    """
    input_size = (img_size, img_size, 3)
    if model_name == "xception":
        baseModel = Xception(
            weights="imagenet",
            include_top=False,
            input_shape=(img_size, img_size, 3)
        )
    elif model_name == "iv3":
        baseModel = InceptionV3(
            weights="imagenet",
            include_top=False,
            input_shape=(img_size, img_size, 3)
        )
    elif model_name == "irv2":
        baseModel = InceptionResNetV2(
            weights="imagenet",
            include_top=False,
            input_shape=(img_size, img_size, 3)
        )
    elif model_name == "resnet":
        baseModel = ResNet50(
            weights="imagenet",
            include_top=False,
            input_shape=(img_size, img_size, 3)
        )
    elif model_name == "nasnet":
        baseModel = NASNetLarge(
            weights="imagenet",
            include_top=False,
            input_shape=(img_size, img_size, 3)
        )
    elif model_name == "ef0":
        baseModel = EfficientNetB0(
            input_size,
            weights="imagenet",
            include_top=False
        )
    elif model_name == "ef5":
        baseModel = EfficientNetB5(
            input_size,
            weights="imagenet",
            include_top=False
        )

    headModel = baseModel.output
    headModel = GlobalAveragePooling2D()(headModel)
    headModel = Dense(512, activation="relu", kernel_initializer="he_uniform")(
        headModel
    )
    headModel = Dropout(0.4)(headModel)
    # headModel = Dense(512, activation="relu", kernel_initializer="he_uniform")(
    #     headModel
    # )
    # headModel = Dropout(0.5)(headModel)
    headModel = Dropout(0.5)(headModel)
    predictions = Dense(
        2,
        activation="softmax",
        kernel_initializer="he_uniform")(
        headModel
    )
    model = Model(inputs=baseModel.input, outputs=predictions)

    model.load_weights("trained_wts/" + weights + ".hdf5")
    print("Weights loaded...")
    model_lstm = Model(
        inputs=baseModel.input,
        outputs=model.get_layer("fc1").output
    )

    for layer in baseModel.layers:
        layer.trainable = True

    optimizer = Nadam(
        lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004
    )
    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"]
    )
    return model_lstm


class TemporalMaxPooling(Layer):
    """
    This pooling layer accepts the temporal sequence output by a recurrent layer
    and performs temporal pooling, looking at only the non-masked portion of the sequence.
    The pooling layer converts the entire variable-length hidden vector sequence
    into a single hidden vector.

    Modified from https://github.com/fchollet/keras/issues/2151 so code also
    works on tensorflow backend. Updated syntax to match Keras 2.0 spec.

    Args:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        3D tensor with shape: `(samples, steps, features)`.

        input shape: (nb_samples, nb_timesteps, nb_features)
        output shape: (nb_samples, nb_features)

    Examples:
        > x = Bidirectional(GRU(128, return_sequences=True))(x)
        > x = TemporalMaxPooling()(x)
    """

    def __init__(self, **kwargs):
        super(TemporalMaxPooling, self).__init__(**kwargs)
        self.supports_masking = True
        self.input_spec = InputSpec(ndim=3)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])

    def call(self, x, mask=None):
        if mask is None:
            mask = K.sum(K.ones_like(x), axis=-1)

        # if masked, set to large negative value so we ignore it
        # when taking max of the sequence
        # K.switch with tensorflow backend is less useful than Theano's
        if K._BACKEND == "tensorflow":
            mask = K.expand_dims(mask, axis=-1)
            mask = K.tile(mask, (1, 1, K.int_shape(x)[2]))
            masked_data = K.tf.where(
                K.equal(mask, K.zeros_like(mask)), K.ones_like(x) * -np.inf, x
            )  # if masked assume value is -inf
            return K.max(masked_data, axis=1)
        else:  # theano backend
            mask = mask.dimshuffle(0, 1, "x")
            masked_data = K.switch(K.eq(mask, 0), -np.inf, x)
            return masked_data.max(axis=1)

    def compute_mask(self, input, mask):
        # do not pass the mask to the next layers
        return None


def lstm_model(train_data):
    # Model definition
    main_input = Input(
        shape=(train_data.shape[1],
               train_data.shape[2]),
        name="main_input"
    )
    # headModel = Bidirectional(LSTM(256, return_sequences=True))(main_input)
    headModel = LSTM(32)(main_input)
    # headModel = TemporalMaxPooling()(headModel)
    # headModel = TimeDistributed(Dense(512))(headModel)
    # # headModel = Bidirectional(LSTM(512, dropout=0.2))(main_input)
    # headModel = LSTM(256)(headModel)
    predictions = Dense(
        2,
        activation="softmax",
        kernel_initializer="he_uniform"
    )(headModel)
    model = Model(inputs=main_input, outputs=predictions)

    # Model compilation
    # opt = SGD(lr=1e-4, momentum=0.9, decay=1e-4 / EPOCHS)
    optimizer = Nadam(
        lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004
    )
    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"]
    )

    return model


def main():
	start = time.time()

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-seq",
        "--seq_length",
        required=True,
        type=int,
        help="Number of frames to be taken into consideration",
    )
    ap.add_argument(
        "-m",
        "--model_name",
        required=True,
        type=str,
        help="Imagenet model to train",
        default="xception",
    )
    ap.add_argument(
        "-w",
        "--load_weights_name",
        required=True,
        type=str,
        help="Model wieghts name"
    )
    ap.add_argument(
        "-im_size",
        "--image_size",
        required=True,
        type=int,
        help="Batch size",
        default=224,
    )
    args = ap.parse_args()

    # MTCNN face extraction from frames
    imageio.core.util._precision_warn = ignore_warnings

    # Create face detector
    mtcnn = MTCNN(
        margin=40,
        select_largest=False,
        post_process=False,
        device="cuda:0"
    )

	test_data = pd.read_csv('test_vids_label.csv')

	videos = test_data["vids_list"]
	true_labels = test_data["label"]

    # Loading model for feature extraction
    model = cnn_model(
        model_name=args.model_name,
        img_size=args.image_size,
        weights=args.load_weights_name
    )

	model_lstm = lstm_model()

	model_lstm.load_weights("xception_lstm.hdf5")

	print("Weights loaded...")

	y_pred = []
	y_probabs = []
	c = 0

	for video in videos:
		cap = cv2.VideoCapture(video)
		batches = []

		while cap.isOpened() and len(batches) < args.seq_length:
			ret, frame = cap.read()
			if not ret:
				break
			
            h, w, _ = frame.shape
            if h >= 1080 and w >= 1920:
                frame = cv2.resize(
                    frame,
                    (640, 480),
                    interpolation=cv2.INTER_AREA
                )

			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			frame = Image.fromarray(frame)
			face = mtcnn(frame)
			
			try:
				face = face.permute(1, 2, 0).int().numpy()
				batches.append(face)
			except AttributeError:
				print("Image Skipping")

		batches = np.asarray(batches).astype('float32')
		batches /= 255

		feature_vec = model_2.predict(batches)
		
		preds = model_lstm.predict(feature_vec)
		
		# Predict the output of each frame
		# axis =1 along the row and axis=0 along the column
		pred_mean = np.mean(preds, axis=0)
		y_probabs+=[pred_mean]
		y_pred+=[pred_mean.argmax(0)]
		
		cap.release()

		if c%10==0:
			print(c, "Done....")
		c+=1

	print("Accuracy Score:", accuracy_score(true_labels, y_pred))
	print("Precision Score", precision_score(true_labels, y_pred))
	print("Recall Score:", recall_score(true_labels, y_pred))
	print("F1 Score:", f1_score(true_labels, y_pred))

	np.save("lstm_preds.npy", y_pred)
	np.save("lstm_probabs.npy", y_probabs)


if __name__ == '__main__':
	main()
