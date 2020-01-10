import pandas as pd
import glob
import os
from os.path import isfile, join, split
from os import rename, listdir, rename, makedirs
from random import shuffle
# from sklearn.utils import shuffle

train_path = ["train_face/1", "train_face/0"]
files_name = ["I", "II", "III", "IV", "V", "VI", "VII"]

list_1 = [join(train_path[0], x) for x in listdir(train_path[0])]
list_0 = [join(train_path[1], x) for x in listdir(train_path[1])]
print(len(list_0)//len(list_1))

c = 0

for i in range(len(list_0)//len(list_1)):
	vid_list = list_1 + list_0[i*(len(list_1)):(i+1)*(len(list_1))]
	print(len(vid_list))
	shuffle(vid_list)

	images = []
	labels = []

	counter = 0

	for x in vid_list:
		img = glob.glob(join(x, '*.jpg'))
		img.sort(key=lambda f:int(''.join(filter(str.isdigit, f))))
		images+=img
		label = [k.split('/')[1] for k in img]
		labels+=label

		if counter%1000==0:
			print("Number of files done:", counter)
		counter+=1

	print("Number of lists done --> {}".format(files_name[i]))


	data = {
		'images_list': images,
		'label': labels
		}

	df = pd.DataFrame(data)
	df.to_csv("train_subset_" + files_name[i] + ".csv", index=False)