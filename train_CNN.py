from keras.preprocessing.image import ImageDataGenerator
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dropout, Flatten, Dense
from keras.models import Model
from keras.optimizers import Nadam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.callbacks import CSVLogger
from sklearn.model_selection import train_test_split
from keras.applications.xception import Xception
from matplotlib import pyplot as plt
from keras import backend as K
import numpy as np
import time
from os.path import exists
from os import makedirs

EPOCHS = 50


def cnn_model():
    """
    Model definition using Xception net architecture
    """
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
    # Training dataset loading
    train_data = np.load("train_data.npy")
    train_label = np.load("train_label.npy")

    print("Dataset Loaded...")

    # Train and validation split
    trainX, valX, trainY, valY = train_test_split(
        train_data, train_label, test_size=0.1, shuffle=False
    )
    print(trainX.shape, valX.shape, trainY.shape, valY.shape)

    # Train nad validation image data generator
    trainAug = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=30,
        zoom_range=0.15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest",
    )

    valAug = ImageDataGenerator(rescale=1.0 / 255.0)

    model = cnn_model()

    # Number of trainable and non-trainable parameters
    trainable_count = int(
        np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    )
    non_trainable_count = int(
        np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])
    )

    print("Total params: {:,}".format(trainable_count + non_trainable_count))
    print("Trainable params: {:,}".format(trainable_count))
    print("Non-trainable params: {:,}".format(non_trainable_count))

    if not exists("./trained_wts"):
        makedirs("./trained_wts")
    if not exists("./training_logs"):
        makedirs("./training_logs")
    if not exists("./plots"):
        makedirs("./plots")

    # Keras backend
    model_checkpoint = ModelCheckpoint(
        "trained_wts/xception_best.hdf5",
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
    )

    stopping = EarlyStopping(monitor="val_loss", patience=10, verbose=0)
    csv_logger = CSVLogger("training_logs/xception.log", separator=",",
        append=True,)

    print("Training is going to start in 3... 2... 1... ")

    # Model Training
    H = model.fit_generator(
        trainAug.flow(trainX, trainY, batch_size=32),
        steps_per_epoch=len(trainX) // 32,
        validation_data=valAug.flow(valX, valY),
        validation_steps=len(valX) // 32,
        epochs=EPOCHS,
        callbacks=[model_checkpoint, stopping, csv_logger],
    )

    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    N = EPOCHS
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on Santa/Not Santa")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig("plots/training_plot.png")

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
