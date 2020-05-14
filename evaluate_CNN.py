from keras.models import Model
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dropout, Flatten, Dense
from keras.optimizers import Nadam
from keras.applications.xception import Xception
import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)
import imageio.core.util
from facenet_pytorch import MTCNN
from PIL import Image
import pandas as pd
import cv2
import math
import time


def ignore_warnings(*args, **kwargs):
    pass


def cnn_model():
    baseModel = Xception(
        weights="imagenet", include_top=False, input_shape=(160, 160, 3)
    )
    headModel = baseModel.output
    headModel = MaxPooling2D(pool_size=(3, 3))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(512, activation="relu", kernel_initializer="he_uniform")(
        headModel
    )
    headModel = Dense(512, activation="relu", kernel_initializer="he_uniform")(
        headModel
    )
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(512, activation="relu", kernel_initializer="he_uniform")(
        headModel
    )
    headModel = Dropout(0.5)(headModel)
    predictions = Dense(
        2,
        activation="softmax",
        kernel_initializer="he_uniform")(
        headModel
    )
    model = Model(inputs=baseModel.input, outputs=predictions)

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
    return model


def main():
    start = time.time()

    # Read video labels from csv file
    test_data = pd.read_csv("test_vids_label.csv")

    videos = test_data["vids_list"]
    true_labels = test_data["label"]

    # Suppress unncessary warnings
    imageio.core.util._precision_warn = ignore_warnings

    # Create face detector
    mtcnn = MTCNN(
        margin=40,
        select_largest=False,
        post_process=False,
        device="cuda:0"
    )

    # Loading model weights
    model = cnn_model()
    model.load_weights("trained_wts/xception_50_160.hdf5")
    print("Weights loaded...")

    y_predictions = []
    y_probabilities = []
    videos_done = 0

    for video in videos:
        cap = cv2.VideoCapture(video)
        batches = []

        # Number of frames taken into consideration for each video
        while (cap.isOpened() and len(batches)<25):
            frameId = cap.get(1)  # current frame number
            ret, frame = cap.read()
            if ret is not True:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            face = mtcnn(frame)

            try:
                face = face.permute(1, 2, 0).int().numpy()
                batches.append(face)
            except AttributeError:
                print("Image Skipping")

        batches = np.asarray(batches).astype("float32")
        batches /= 255

        predictions = model.predict(batches)
        # Predict the output of each frame
        # axis =1 along the row and axis=0 along the column
        predictions_mean = np.mean(predictions, axis=0)
        y_probabilities += [predictions_mean]
        y_predictions += [predictions_mean.argmax(0)]

        cap.release()

        if videos_done % 10 == 0:
            print("Number of videos done:", videos_done)
        videos_done += 1

    print("Accuracy Score:", accuracy_score(true_labels, y_predictions))
    print("Precision Score:", precision_score(true_labels, y_predictions))
    print("Recall Score:", recall_score(true_labels, y_predictions))
    print("F1 Score:", f1_score(true_labels, y_predictions))

    # Saving predictions and probabilities for further calculation
    # of AUC scores.
    np.save("Y_predictions.npy", y_predictions)
    np.save("Y_probabilities.npy", y_probabilities)

    end = time.time()
    dur = end - start

    if dur < 60:
        print("Execution Time:", dur, "seconds")
    elif dur > 60 and dur < 3600:
        dur = dur / 60
        print("Execution Time:", dur, "minutes")
    else:
        dur = dur / (60 * 60)
        print("Execution Time:", dur, "hours")


if __name__ == "__main__":
    main()
