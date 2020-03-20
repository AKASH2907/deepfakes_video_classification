from keras.layers import Activation
from keras.regularizers import l2
from keras.models import Model
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv3D, MaxPooling3D, ZeroPadding3D
from keras.layers import Input

# from schedules import onetenth_4_8_12
import numpy as np
import cv2
import time
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


def ignore_warnings(*args, **kwargs):
    pass


imageio.core.util._precision_warn = ignore_warnings

# Create face detector
mtcnn = MTCNN(
    margin=40,
    select_largest=False,
    post_process=False,
    device="cuda:0"
)


def conv3d_model():
    input_shape = (32, 112, 112, 3)
    weight_decay = 0.005
    nb_classes = 2

    inputs = Input(input_shape)
    x = Conv3D(
        64,
        (3, 3, 3),
        strides=(1, 1, 1),
        padding="same",
        activation="relu",
        kernel_regularizer=l2(weight_decay),
    )(inputs)
    x = MaxPooling3D((2, 2, 1), strides=(2, 2, 1), padding="same")(x)

    x = Conv3D(
        128,
        (3, 3, 3),
        strides=(1, 1, 1),
        padding="same",
        activation="relu",
        kernel_regularizer=l2(weight_decay),
    )(x)
    x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding="same")(x)

    x = Conv3D(
        128,
        (3, 3, 3),
        strides=(1, 1, 1),
        padding="same",
        activation="relu",
        kernel_regularizer=l2(weight_decay),
    )(x)
    x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding="same")(x)

    x = Conv3D(
        256,
        (3, 3, 3),
        strides=(1, 1, 1),
        padding="same",
        activation="relu",
        kernel_regularizer=l2(weight_decay),
    )(x)
    x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding="same")(x)

    x = Conv3D(
        256,
        (3, 3, 3),
        strides=(1, 1, 1),
        padding="same",
        activation="relu",
        kernel_regularizer=l2(weight_decay),
    )(x)
    x = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding="same")(x)

    x = Flatten()(x)
    x = Dense(2048, activation="relu", kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)
    x = Dense(2048, activation="relu", kernel_regularizer=l2(weight_decay))(x)
    x = Dropout(0.5)(x)
    x = Dense(nb_classes, kernel_regularizer=l2(weight_decay))(x)
    x = Activation("softmax")(x)

    model = Model(inputs, x)
    return model


def c3d_model(summary=False):
    """ Return the Keras model of the network
    """
    main_input = Input(shape=(32, 112, 112, 3), name="main_input")
    # 1st layer group
    x = Conv3D(
        64,
        kernel_size=(3, 3, 3),
        activation="relu",
        padding="same",
        name="conv1",
        strides=(1, 1, 1),
    )(main_input)
    x = MaxPooling3D(
        pool_size=(1, 2, 2), strides=(1, 2, 2), padding="valid", name="pool1"
    )(x)
    # 2nd layer group
    x = Conv3D(
        128,
        kernel_size=(3, 3, 3),
        activation="relu",
        padding="same",
        name="conv2",
        strides=(1, 1, 1),
    )(x)
    x = MaxPooling3D(
        pool_size=(2, 2, 2), strides=(2, 2, 2), padding="valid", name="pool2"
    )(x)
    # 3rd layer group
    x = Conv3D(
        256,
        kernel_size=(3, 3, 3),
        activation="relu",
        padding="same",
        name="conv3a",
        strides=(1, 1, 1),
    )(x)
    x = Conv3D(
        256,
        kernel_size=(3, 3, 3),
        activation="relu",
        padding="same",
        name="conv3b",
        strides=(1, 1, 1),
    )(x)
    x = MaxPooling3D(
        pool_size=(2, 2, 2), strides=(2, 2, 2), padding="valid", name="pool3"
    )(x)
    # 4th layer group
    x = Conv3D(
        512,
        kernel_size=(3, 3, 3),
        activation="relu",
        padding="same",
        name="conv4a",
        strides=(1, 1, 1),
    )(x)
    x = Conv3D(
        512,
        kernel_size=(3, 3, 3),
        activation="relu",
        padding="same",
        name="conv4b",
        strides=(1, 1, 1),
    )(x)
    x = MaxPooling3D(
        pool_size=(2, 2, 2), strides=(2, 2, 2), padding="valid", name="pool4"
    )(x)
    # 5th layer group
    x = Conv3D(
        512,
        kernel_size=(3, 3, 3),
        activation="relu",
        padding="same",
        name="conv5a",
        strides=(1, 1, 1),
    )(x)
    x = Conv3D(
        512,
        kernel_size=(3, 3, 3),
        activation="relu",
        padding="same",
        name="conv5b",
        strides=(1, 1, 1),
    )(x)
    x = ZeroPadding3D(padding=(0, 1, 1))(x)
    x = MaxPooling3D(
        pool_size=(2, 2, 2), strides=(2, 2, 2), padding="valid", name="pool5"
    )(x)
    x = Flatten()(x)
    # FC layers group
    x = Dense(2048, activation="relu", name="fc6")(x)
    x = Dropout(0.5)(x)
    x = Dense(2048, activation="relu", name="fc7")(x)
    x = Dropout(0.5)(x)
    predictions = Dense(2, activation="softmax", name="fc8")(x)

    model = Model(inputs=main_input, outputs=predictions)
    if summary:
        print(model.summary())
    return model


def process_batch(video_paths, num_frames=16):
    num = len(video_paths)
    batch = np.zeros((num, 32, 112, 112, 3), dtype="float32")
    labels = np.zeros(num, dtype="int")
    for i in range(num):
        cap = cv2.VideoCapture(video_paths[i])
        batches = []
        counter = 0
        while cap.isOpened():
            frameId = cap.get(1)
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            face = mtcnn(frame)
            try:
                face = face.permute(1, 2, 0).float().numpy()
                face = cv2.resize(face, (171, 128))
                batch[i][counter][:][:][:] = face[8:120, 30:142, :]
                batches.append(face)
            except AttributeError:
                print("Image Skipping")
            if counter == 31:
                break
            counter += 1
        cap.release()
        label = video_paths[i].split("/")[1]
        label = int(label)
        labels[i] = label
    return batch, labels


def preprocess(inputs):
    # inputs[..., 0] -= 99.9
    # inputs[..., 1] -= 92.1
    # inputs[..., 2] -= 82.6
    # inputs[..., 0] /= 65.8
    # inputs[..., 1] /= 62.3
    # inputs[..., 2] /= 60.3
    inputs /= 255.0
    return inputs


def generator_test_batch(test_vid_list, batch_size, num_classes):
    num = len(test_vid_list)
    while True:
        for i in range(int(num / batch_size)):
            a = i * batch_size
            b = (i + 1) * batch_size
            y_test, y_labels = process_batch(test_vid_list[a:b])
            x = preprocess(y_test)
            y = np_utils.to_categorical(np.array(y_labels), num_classes)
            yield x, y


def main():
    test_data = pd.read_csv("test_vids_label.csv")
    test_vids_list = test_data["vids_list"]
    test_vids_list = np.array(test_vids_list)
    true_labels = test_data["label"]

    start = time.time()

    model = c3d_model()
    lr = 0.005
    sgd = SGD(lr=lr, momentum=0.9, nesterov=True)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=sgd,
        metrics=["accuracy"]
    )
    model.load_weights("results/weights_c3d.h5")
    print("Weights loaded...")

    num_classes = 2
    batch_size = 16
    probabs = model.predict_generator(
        generator_test_batch(test_vids_list, batch_size, num_classes),
        steps=len(test_vids_list) // batch_size,
        verbose=1,
    )
    np.save("C3D_probabs.npy", probabs)
    print(probabs)
    y_pred = probabs.argmax(1)
    np.save("C3D_preds.npy", y_pred)

    print("Accuracy Score:", accuracy_score(true_labels, y_pred))
    print("Precision Score", precision_score(true_labels, y_pred))
    print("Recall Score:", recall_score(true_labels, y_pred))
    print("F1 Score:", f1_score(true_labels, y_pred))
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
