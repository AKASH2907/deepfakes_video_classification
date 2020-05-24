from keras.layers import GlobalAveragePooling2D
from keras.layers.core import Dropout, Dense
from keras.models import Model
import numpy as np
import imageio.core.util
from facenet_pytorch import MTCNN
from PIL import Image
import cv2
from keras.optimizers import Nadam
from keras.applications.xception import Xception
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.nasnet import NASNetLarge
from keras_efficientnets import EfficientNetB5, EfficientNetB0
from random import shuffle
from os import listdir
import glob
from os.path import join
import argparse


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
    headModel = Dense(512, activation="relu", kernel_initializer="he_uniform", name="fc1")(
        headModel
    )
    headModel = Dropout(0.4)(headModel)
    # headModel = Dense(512, activation="relu", kernel_initializer="he_uniform")(
    #     headModel
    # )
    # headModel = Dropout(0.5)(headModel)
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


def main():

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

    train_dir = "./train_c23/"
    sub_directories = listdir(train_dir)

    videos = []

    for i in sub_directories:
        videos += glob.glob(join(train_dir, i, "*.mp4"))

    shuffle(videos)

    # Loading model for feature extraction
    model = cnn_model(
        model_name=args.model_name,
        img_size=args.image_size,
        weights=args.load_weights_name
    )

    features = []
    counter = 0
    labels = []

    for video in videos:
        cap = cv2.VideoCapture(video)
        labels += [int(video.split("/")[-2])]

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

        cap.release()
        batches = np.array(batches).astype("float32")
        batches /= 255

        # fc layer feature generation
        predictions = model.predict(batches)

        features += [predictions]

        if counter % 50 == 0:
            print("Number of videos done:", counter)
        counter += 1

    features = np.array(features)
    labels = np.array(labels)

    print(features.shape, labels.shape)

    np.save("lstm_40fpv_data.npy", features)
    np.save("lstm_40fpv_labels.npy", labels)


if __name__ == "__main__":
    main()
