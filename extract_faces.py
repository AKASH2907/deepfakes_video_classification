from facenet_pytorch import MTCNN
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
# from tqdm.notebook import tqdm
from os import listdir, makedirs
import glob
from os.path import join, sep, split, exists
from skimage.io import imsave
from skimage import img_as_ubyte
import warnings
import imageio.core.util

def ignore_warnings(*args, **kwargs):
    pass

imageio.core.util._precision_warn = ignore_warnings

# Create face detector
# If you want to change the default size of image saved from 160, you can uncomment 
# the second line and set the parameter accordingly.
mtcnn = MTCNN(margin=40, select_largest=False, post_process=False, device='cuda:0')
# mtcnn = MTCNN(margin=40, select_largest=False, post_process=False, device='cuda:0', image_size=256)

# Directory containing images respective to each video
train_vids = ["./train_frames/1/"]
# Destination location where faces cropped out from images will be saved
dest_folder = "./train_face/1/"


for i in train_vids:
	counter = 0
	skips = 0
	for j in listdir(i):
		imgs = glob.glob(join(i, j, '*.jpg'))
		if counter%1000==0:
			print("Number of videos done...", counter)
		if not exists(join(dest_folder, j)):
			makedirs(join(dest_folder, j))
		for k in imgs:
			# print(k)
			frame = cv2.imread(k)
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			frame = Image.fromarray(frame)
			face = mtcnn(frame)

			try:
				imsave(join(dest_folder, j, k.split('/')[-1]), face.permute(1, 2, 0).int().numpy())
			except AttributeError:
				skips+=1
				print("Image skipping")
		counter+=1
print("Number of images skipeed:", skips)