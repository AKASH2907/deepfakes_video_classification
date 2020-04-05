from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.applications.xception import Xception
import numpy as np
import imageio.core.util
from facenet_pytorch import MTCNN
from PIL import Image
import cv2
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dropout, Flatten, Dense
from keras.optimizers import Nadam
from os import listdir
import glob
from os.path import join
import argparse

ap = argparse.ArgumentParser()
ap.add_argument(
    "-seq",
    "--seq_length",
    required=True,
    type=int,
    help="Number of frames to be taken into consideration",
)

args = ap.parse_args()

train_dir = "./train"
sub_directories = listdir(train_dir)

videos = []

for i in sub_directories:
    videos += glob.glob(join(train_dir, i, "*.mp4"))


def ignore_warnings(*args, **kwargs):
    pass


imageio.core.util._precision_warn = ignore_warnings

# Create face detector
mtcnn = MTCNN(margin=40, select_largest=False, post_process=False, device="cuda:0")

testAug = ImageDataGenerator(rescale=1.0 / 255.0)


# def xception_model():
baseModel = Xception(weights="imagenet", include_top=False, input_shape=(160, 160, 3))
headModel = baseModel.output
headModel = MaxPooling2D(pool_size=(3, 3))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(512, activation="relu", kernel_initializer="he_uniform", name="fc1")(
    headModel
)
headModel = Dense(512, activation="relu", kernel_initializer="he_uniform")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(512, activation="relu", kernel_initializer="he_uniform")(headModel)
headModel = Dropout(0.5)(headModel)
predictions = Dense(2, activation="softmax", kernel_initializer="he_uniform")(headModel)
model = Model(inputs=baseModel.input, outputs=predictions)

for layer in baseModel.layers:
    layer.trainable = True

optimizer = Nadam(
    lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004
)
model.compile(
    loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
)
model.load_weights("trained_wts/xception_50_160.hdf5")
model_2 = Model(inputs=baseModel.input, outputs=model.get_layer("fc1").output)

print("Weights loaded...")

features = []
counter = 0
labels = []

for video in videos[:]:
    cap = cv2.VideoCapture(video)
    labels += [int(video.split("/")[-2])]

    batches = []
    sequence_length = 0
    while cap.isOpened() and sequence_length < args.seq_length:
        frameId = cap.get(1)  # current frame number
        ret, frame = cap.read()
        if ret != True:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        face = mtcnn(frame)

        try:
            face = face.permute(1, 2, 0).int().numpy()
            batches.append(face)
        except AttributeError:
            print("Image Skipping")
        sequence_length += 1
    batches = np.asarray(batches).astype("float32")
    batches /= 255

    predictions = model_2.predict(batches)

    features += [predictions]

    cap.release()

    if counter % 100 == 0:
        print("Number of videos done:", counter)
    counter += 1

features = np.array(features)
labels = np.array(labels)

print(features.shape, labels.shape)

np.save("lstm_25f_data.npy", features)
np.save("lstm_25f_labels.npy", labels)
