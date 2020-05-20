from facenet_pytorch import MTCNN
import cv2
from PIL import Image

from os import listdir, makedirs
import glob
from os.path import join, exists
from skimage.io import imsave
import imageio.core.util


def ignore_warnings(*args, **kwargs):
    pass


imageio.core.util._precision_warn = ignore_warnings

# Create face detector
# If you want to change the default size of image saved from 160, you can
# uncomment the second line and set the parameter accordingly.
mtcnn = MTCNN(
    margin=40,
    select_largest=False,
    post_process=False,
    device="cuda:0"
)
# mtcnn = MTCNN(margin=40, select_largest=False, post_process=False,
# device='cuda:0', image_size=256)

# Directory containing images respective to each video
source_frames_folders = ["../train_frames/0/"]
# Destination location where faces cropped out from images will be saved
dest_faces_folder = "../train_face/0/"


for i in source_frames_folders:
    counter = 0
    for j in listdir(i):
        imgs = glob.glob(join(i, j, "*.jpg"))
        if counter % 1000 == 0:
            print("Number of videos done:", counter)
        if not exists(join(dest_faces_folder, j)):
            makedirs(join(dest_faces_folder, j))
        for k in imgs:
            frame = cv2.imread(k)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            face = mtcnn(frame)

            try:
                imsave(
                    join(dest_faces_folder, j, k.split("/")[-1]),
                    face.permute(1, 2, 0).int().numpy(),
                )
            except AttributeError:
                print("Image skipping")
        counter += 1
