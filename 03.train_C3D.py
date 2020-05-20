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
import os
import random
import time
import argparse
import matplotlib.pyplot as plt
from os.path import join
from os import listdir
from random import shuffle


def conv3d_model(batch_size):
    input_shape = (batch_size, 112, 112, 3)
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


def c3d_model(batch_size):
    """ Return the Keras model of the network
    """
    main_input = Input(shape=(batch_size, 112, 112, 3), name="main_input")
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
    return model


def plot_history(history, result_dir):
    plt.plot(history.history["accuracy"], marker=".")
    plt.plot(history.history["val_accuracy"], marker=".")
    plt.title("model accuracy")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.grid()
    plt.legend(["accuracy", "val_accuracy"], loc="lower right")
    plt.savefig(os.path.join(result_dir, "model_accuracy.png"))
    plt.close()

    plt.plot(history.history["loss"], marker=".")
    plt.plot(history.history["val_loss"], marker=".")
    plt.title("model loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.grid()
    plt.legend(["loss", "val_loss"], loc="upper right")
    plt.savefig(os.path.join(result_dir, "model_loss.png"))
    plt.close()


def save_history(history, result_dir):
    loss = history.history["loss"]
    acc = history.history["acc"]
    val_loss = history.history["val_loss"]
    val_acc = history.history["val_acc"]
    nb_epoch = len(acc)

    with open(os.path.join(result_dir, "result.txt"), "w") as fp:
        fp.write("epoch\tloss\tacc\tval_loss\tval_acc\n")
        for i in range(nb_epoch):
            fp.write(
                "{}\t{}\t{}\t{}\t{}\n".format(
                    i, loss[i], acc[i], val_loss[i], val_acc[i]
                )
            )
        fp.close()


def process_batch(video_paths, batch_size, train=True):
    num = len(video_paths)
    batch = np.zeros((num, batch_size, 112, 112, 3), dtype="float32")
    labels = np.zeros(num, dtype="int")
    for i in range(num):
        # path = video_paths[i].split(" ")[0]
        path = video_paths[i]
        label = video_paths[i].split("/")[1]
        # symbol = video_paths[i].split(" ")[1]
        # label = label.strip("\n")
        label = int(label)
        # symbol = int(symbol) - 1
        imgs = os.listdir(path)
        imgs.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
        if train:
            crop_x = random.randint(0, 15)
            crop_y = random.randint(0, 58)
            is_flip = random.randint(0, 1)

            for j in range(batch_size):
                img = imgs[j]
                image = cv2.imread(path + "/" + img)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (171, 128))
                if is_flip == 1:
                    image = cv2.flip(image, 1)
                batch[i][j][:][:][:] = image[
                    crop_x: crop_x + 112, crop_y: crop_y + 112, :
                ]
            labels[i] = label
        else:
            for j in range(batch_size):
                img = imgs[j]
                image = cv2.imread(path + "/" + img)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (171, 128))
                batch[i][j][:][:][:] = image[8:120, 30:142, :]
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


def generator_train_batch(train_vid_list, batch_size, num_classes):
    num = len(train_vid_list)
    while True:
        for i in range(int(num / batch_size)):
            a = i * batch_size
            b = (i + 1) * batch_size
            x_train, x_labels = process_batch(
                train_vid_list[a:b],
                batch_size,
                train=True
            )
            x = preprocess(x_train)
            y = np_utils.to_categorical(np.array(x_labels), num_classes)
            yield x, y


def generator_val_batch(val_vid_list, batch_size, num_classes):
    num = len(val_vid_list)
    while True:
        for i in range(int(num / batch_size)):
            a = i * batch_size
            b = (i + 1) * batch_size
            y_test, y_labels = process_batch(
                val_vid_list[a:b],
                train=False
            )
            x = preprocess(y_test)
            y = np_utils.to_categorical(np.array(y_labels), num_classes)
            yield x, y


def main():
    start = time.time()

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-e", "--epochs", required=True, type=int,
        help="Number of epochs", default=25
    )
    ap.add_argument(
        "-m", "--model_name", required=True, type=str,
        help="Imagenet model to train", default="c3d"
    )
    ap.add_argument(
        "-w",
        "--weights_save_name",
        required=True,
        type=str,
        help="Model wieghts name"
    )
    ap.add_argument(
        "-b", "--batch_size", required=True, type=int,
        help="Batch size", default=32
    )
    args = ap.parse_args()

    # Video cropped faces train path
    train_path = ["train_faces_c40/1", "train_faces_c40/0"]

    list_1 = [join(train_path[0], x) for x in listdir(train_path[0])]
    list_0 = [join(train_path[1], x) for x in listdir(train_path[1])]

    for i in range(1):
        # for i in range(len(list_0)//len(list_1)):
        vid_list = list_1 + list_0[i * (len(list_1)): (i + 1) * (len(list_1))]
        print(len(vid_list))
        shuffle(vid_list)

        # Distrbution of training data as 80/20 according to FF++ paper
        train_vid_list = vid_list[: int(0.8 * len(vid_list))]
        val_vid_list = vid_list[int(0.8 * len(vid_list)):]

    num_classes = 2
    batch_size = args.batch_size
    epochs = args.epochs

    # Model choice can be added more
    if args.model_name == "c3d":
        model = c3d_model(batch_size=args.batch_size)
    else:
        model = conv3d_model(batch_size=args.batch_size)

    lr = 0.005
    sgd = SGD(lr=lr, momentum=0.9, nesterov=True)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=sgd,
        metrics=["accuracy"]
    )

    # Model fitting
    history = model.fit_generator(
        generator_train_batch(train_vid_list, batch_size, num_classes),
        steps_per_epoch=len(train_vid_list) // batch_size,
        epochs=epochs,
        # callbacks=[onetenth_4_8_12(lr)],
        validation_data=generator_val_batch(
            val_vid_list,
            batch_size,
            num_classes
        ),
        validation_steps=len(val_vid_list) // batch_size,
        verbose=1,
    )

    # Make results directory
    if not os.path.exists("results/"):
        os.mkdir("results/")
    plot_history(history, "results/")
    save_history(history, "results/")
    model.save_weights("results/" + args.weights_save_name + ".hdf5")

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
