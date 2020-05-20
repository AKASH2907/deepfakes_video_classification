import cv2
import argparse
import numpy as np
from keras.utils import np_utils
import glob
from os.path import join
from os import listdir
from random import shuffle

ap = argparse.ArgumentParser()
ap.add_argument(
    "-img_size",
    "--img_size",
    required=True,
    type=int,
    help="Resize face image",
    default=160,
)
ap.add_argument(
    "-fpv",
    "--frames_per_video",
    required=True,
    type=int,
    help="Number of frames per video to consider",
    default=25,
)
args = ap.parse_args()

train_path = ["train_face/1", "train_face/0"]

list_1 = [join(train_path[0], x) for x in listdir(train_path[0])]
list_0 = [join(train_path[1], x) for x in listdir(train_path[1])]

c = 0

for i in range(len(list_0) // len(list_1)):
    vid_list = list_1 + list_0[i * (len(list_1)): (i + 1) * (len(list_1))]
    shuffle(vid_list)

    train_data = []
    train_label = []

    count = 0

    images = []
    labels = []

    counter = 0

    for x in vid_list:
        img = glob.glob(join(x, "*.jpg"))
        img.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
        images += img[: args.frames_per_video]
        label = [k.split("/")[1] for k in img]
        labels += label[: args.frames_per_video]

        if counter % 1000 == 0:
            print("Number of files done:", counter)
        counter += 1

    print("Number of lists done --> {}".format(files_name[i]))

    for j, k in zip(images, labels):

        img = cv2.imread(j)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(
            img, (args.img_size, args.img_size), interpolation=cv2.INTER_AREA
        )
        train_data.append(img)
        train_label += [k]

        if count % 10000 == 0:
            print("Number of files done:", count)
        count += 1

    train_data = np.array(train_data)
    train_label = np.array(train_label)
    train_label = np_utils.to_categorical(train_label)
    print(train_data.shape, train_label.shape)

    np.save("train_data_" + str(args.frames_per_video) + "_c40.npy", train_data)
    np.save("train_label_" + str(args.frames_per_video) + "_c40.npy", train_label)

    print("Files saved number....", files_name[i])
