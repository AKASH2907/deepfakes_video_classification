import numpy as np
from keras.preprocessing import image
import pandas as pd
from keras_facenet import FaceNet

# Read csv file contain image face paths
data = pd.read_csv("train_faces_160.csv")

images = data["images_list"]
labels = data["label"]

train_data = []
train_label = []
count = 0

embedder = FaceNet()


for (img_path, label) in zip(images, labels):
    # img = image.load_img(img_path)
    x = image.img_to_array(img_path)
    x = np.expand_dims(x, axis=0)
    embeddings = embedder.embeddings(x)
    train_data.append(embeddings)
    train_label += [label.argmax(1)]

    if count % 10000 == 0:
        print("Number of files done:", count)
    count += 1

train_data = np.array(train_data)
train_label = np.array(train_label)

np.save("train_data_facenet_embeddings.npy", train_data)
np.save("train_label_facenet_embeddings.npy", train_label)
print("Files saved....")


# Testing part
# for i in images[:5]:
# 	x = i
# 	print(x.shape)
# 	embs = embedder.embeddings(x)
# 	print(embs.shape)
# 	train_data.append(embs)
# 	if count%10==0:
# 		print("Number of files done:", count)
# 	count+=1
