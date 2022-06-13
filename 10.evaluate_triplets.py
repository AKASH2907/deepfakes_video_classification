import argparse
import pickle
import pandas as pd
import warnings
import imageio.core.util

from PIL import Image
import pandas as pd
import cv2
import math
## for Model definition/training
from keras.models import Model, load_model
from keras.layers import Input, Flatten, Dense, concatenate,  Dropout
from keras.optimizers import Adam, Nadam
from keras.applications.xception import Xception
from keras import backend as K
# from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
from keras.layers.pooling import MaxPooling2D
from keras import utils

## required for semi-hard triplet loss:
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import dtypes
import tensorflow as tf

## for visualizing 
import matplotlib.pyplot as plt, numpy as np
from sklearn.decomposition import PCA
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, auc, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.patheffects as PathEffects
from facenet_pytorch import MTCNN
from keras_facenet import FaceNet


def ignore_warnings(*args, **kwargs):
	pass

imageio.core.util._precision_warn = ignore_warnings

# Create face detector
mtcnn = MTCNN(margin=40, select_largest=False, post_process=False, device='cuda:0')

def pairwise_distance(feature, squared=False):
	"""Computes the pairwise distance matrix with numerical stability.

	output[i, j] = || feature[i, :] - feature[j, :] ||_2

	Args:
	  feature: 2-D Tensor of size [number of data, feature dimension].
	  squared: Boolean, whether or not to square the pairwise distances.

	Returns:
	  pairwise_distances: 2-D Tensor of size [number of data, number of data].
	"""
	pairwise_distances_squared = math_ops.add(
		math_ops.reduce_sum(math_ops.square(feature), axis=[1], keepdims=True),
		math_ops.reduce_sum(
			math_ops.square(array_ops.transpose(feature)),
			axis=[0],
			keepdims=True)) - 2.0 * math_ops.matmul(feature,
													array_ops.transpose(feature))

	# Deal with numerical inaccuracies. Set small negatives to zero.
	pairwise_distances_squared = math_ops.maximum(pairwise_distances_squared, 0.0)
	# Get the mask where the zero distances are at.
	error_mask = math_ops.less_equal(pairwise_distances_squared, 0.0)

	# Optionally take the sqrt.
	if squared:
		pairwise_distances = pairwise_distances_squared
	else:
		pairwise_distances = math_ops.sqrt(
			pairwise_distances_squared + math_ops.to_float(error_mask) * 1e-16)

	# Undo conditionally adding 1e-16.
	pairwise_distances = math_ops.multiply(
		pairwise_distances, math_ops.to_float(math_ops.logical_not(error_mask)))

	num_data = array_ops.shape(feature)[0]
	# Explicitly set diagonals to zero.
	mask_offdiagonals = array_ops.ones_like(pairwise_distances) - array_ops.diag(
		array_ops.ones([num_data]))
	pairwise_distances = math_ops.multiply(pairwise_distances, mask_offdiagonals)
	return pairwise_distances

def masked_maximum(data, mask, dim=1):
	"""Computes the axis wise maximum over chosen elements.

	Args:
	  data: 2-D float `Tensor` of size [n, m].
	  mask: 2-D Boolean `Tensor` of size [n, m].
	  dim: The dimension over which to compute the maximum.

	Returns:
	  masked_maximums: N-D `Tensor`.
		The maximized dimension is of size 1 after the operation.
	"""
	axis_minimums = math_ops.reduce_min(data, dim, keepdims=True)
	masked_maximums = math_ops.reduce_max(
		math_ops.multiply(data - axis_minimums, mask), dim,
		keepdims=True) + axis_minimums
	return masked_maximums

def masked_minimum(data, mask, dim=1):
	"""Computes the axis wise minimum over chosen elements.

	Args:
	  data: 2-D float `Tensor` of size [n, m].
	  mask: 2-D Boolean `Tensor` of size [n, m].
	  dim: The dimension over which to compute the minimum.

	Returns:
	  masked_minimums: N-D `Tensor`.
		The minimized dimension is of size 1 after the operation.
	"""
	axis_maximums = math_ops.reduce_max(data, dim, keepdims=True)
	masked_minimums = math_ops.reduce_min(
		math_ops.multiply(data - axis_maximums, mask), dim,
		keepdims=True) + axis_maximums
	return masked_minimums

def triplet_loss_adapted_from_tf(y_true, y_pred):
	del y_true
	margin = 1.
	labels = y_pred[:, :1]

 
	labels = tf.cast(labels, dtype='int32')

	embeddings = y_pred[:, 1:]

	### Code from Tensorflow function [tf.contrib.losses.metric_learning.triplet_semihard_loss] starts here:
	
	# Reshape [batch_size] label tensor to a [batch_size, 1] label tensor.
	# lshape=array_ops.shape(labels)
	# assert lshape.shape == 1
	# labels = array_ops.reshape(labels, [lshape[0], 1])

	# Build pairwise squared distance matrix.
	pdist_matrix = pairwise_distance(embeddings, squared=True)
	# Build pairwise binary adjacency matrix.
	adjacency = math_ops.equal(labels, array_ops.transpose(labels))
	# Invert so we can select negatives only.
	adjacency_not = math_ops.logical_not(adjacency)

	# global batch_size  
	batch_size = array_ops.size(labels) # was 'array_ops.size(labels)'

	# Compute the mask.
	pdist_matrix_tile = array_ops.tile(pdist_matrix, [batch_size, 1])
	mask = math_ops.logical_and(
		array_ops.tile(adjacency_not, [batch_size, 1]),
		math_ops.greater(
			pdist_matrix_tile, array_ops.reshape(
				array_ops.transpose(pdist_matrix), [-1, 1])))
	mask_final = array_ops.reshape(
		math_ops.greater(
			math_ops.reduce_sum(
				math_ops.cast(mask, dtype=dtypes.float32), 1, keepdims=True),
			0.0), [batch_size, batch_size])
	mask_final = array_ops.transpose(mask_final)

	adjacency_not = math_ops.cast(adjacency_not, dtype=dtypes.float32)
	mask = math_ops.cast(mask, dtype=dtypes.float32)

	# negatives_outside: smallest D_an where D_an > D_ap.
	negatives_outside = array_ops.reshape(
		masked_minimum(pdist_matrix_tile, mask), [batch_size, batch_size])
	negatives_outside = array_ops.transpose(negatives_outside)

	# negatives_inside: largest D_an.
	negatives_inside = array_ops.tile(
		masked_maximum(pdist_matrix, adjacency_not), [1, batch_size])
	semi_hard_negatives = array_ops.where(
		mask_final, negatives_outside, negatives_inside)

	loss_mat = math_ops.add(margin, pdist_matrix - semi_hard_negatives)

	mask_positives = math_ops.cast(
		adjacency, dtype=dtypes.float32) - array_ops.diag(
		array_ops.ones([batch_size]))

	# In lifted-struct, the authors multiply 0.5 for upper triangular
	#   in semihard, they take all positive pairs except the diagonal.
	num_positives = math_ops.reduce_sum(mask_positives)

	semi_hard_triplet_loss_distance = math_ops.truediv(
		math_ops.reduce_sum(
			math_ops.maximum(
				math_ops.multiply(loss_mat, mask_positives), 0.0)),
		num_positives,
		name='triplet_semihard_loss')
	
	### Code from Tensorflow function semi-hard triplet loss ENDS here.
	return semi_hard_triplet_loss_distance


def create_base_network(image_input_shape, embedding_size):
	"""
	Base network to be shared (eq. to feature extraction).
	"""
	main_input = Input(shape=(512, ))
	x = Dense(256, activation='relu', kernel_initializer='he_uniform')(main_input)
	x = Dropout(0.1)(x)
	x = Dense(256, activation='relu', kernel_initializer='he_uniform')(x)
	x = Dropout(0.1)(x)
	x = Dense(128, activation='relu', kernel_initializer='he_uniform')(x)
	x = Dropout(0.1)(x)
	y = Dense(embedding_size)(x)
	base_network = Model(main_input, y)
	return base_network


if __name__ == "__main__":
	# in case this scriot is called from another file, let's make sure it doesn't start training the network...
	embedding_size = 64
	input_image_shape = (512, )
	test_data = pd.read_csv('ff++/test_vids_label.csv')

	embedder = FaceNet()

	videos = test_data["vids_list"]
	true_labels = test_data["label"]
	print("Dataset Loaded...")
	print(len(videos), len(true_labels))
	# Test the network
	# creating an empty network
	testing_embeddings = create_base_network(input_image_shape,
											 embedding_size=embedding_size)

	model = load_model("triplets/triplets_semi_hard.hdf5",
		custom_objects={'triplet_loss_adapted_from_tf':triplet_loss_adapted_from_tf})

	# Grabbing the weights from the trained network
	for layer_target, layer_source in zip(testing_embeddings.layers, model.layers[2].layers):
		weights = layer_source.get_weights()
		layer_target.set_weights(weights)
		del weights 
	print("Model Loaded...")

	y_predictions = []
	y_probabilities = []
	c= 0

	# test_data = np.load("test_embs.npy")
	test_label = np.load("test_labels.npy")
	y_test_onehot = utils.to_categorical(test_label)

	for i in videos[:]:
		cap = cv2.VideoCapture(i)
		batches = []
		mounting = 0
		while(cap.isOpened() and mounting<25):
			frameId = cap.get(1) #current frame number
			ret, frame = cap.read()
			if (ret != True):
				break		
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			frame = Image.fromarray(frame)
			face = mtcnn(frame)
			
			try:
				face = face.permute(1, 2, 0).int().numpy()
				batches.append(face)
			except AttributeError:
				print("Image Skipping")
			mounting+=1

		batches = np.asarray(batches).astype('float32')
		print(batches.shape)

		embeddings = embedder.embeddings(batches)
		x_test_after = testing_embeddings.predict(embeddings)
		x_test = testing_embeddings.predict(embeddings)
		
		# print("Embeddings after training")
		sgd = linear_model.SGDClassifier(max_iter=50, tol=None)
		with open('triplets/sgd_classifier.pkl', 'rb') as fid:
			sgd_loaded = pickle.load(fid)
		y_pred = sgd_loaded.predict(x_test)
		y_probabs = sgd_loaded.predict_proba(x_test)

		pred_mean = np.mean(y_pred, axis=0)
		probab_mean = np.mean(y_probabs, axis=0)
		probab_mean = 1 - probab_mean

		y_probabilities +=[probab_mean]
		# print(pred_mean)
		if pred_mean>0.5:
			y_predictions+=[0]
		else:
			y_predictions+=[1]

	y_probabilities = np.array(y_probabilities)
	# print(y_probabilities[:, 1])
	fpr, tpr, threshold = roc_curve(test_label, y_probabilities[:, 1])
	fnr = 1 - tpr
	eer_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
	# EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
	# print(EER)
	EER = fnr[np.nanargmin(np.absolute((fnr - fpr)))]
	print(EER)
	roc_auc = auc(fpr, tpr)
	print("AUC Score:", roc_auc)
	plt.title('Receiver Operating Characteristic')
	plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
	plt.grid()
	plt.legend(loc = 'lower right')
	plt.plot([0, 1], [0, 1],'r--')
	plt.xlim([0, 1])
	plt.ylim([0, 1])
	plt.ylabel('True Positive Rate')
	plt.xlabel('False Positive Rate')
	plt.title('ROC Curve of kNN')
	plt.savefig("AUC-ROC Score")
	
	print("Accuracy:", accuracy_score(test_label, y_predictions))
	print("Precision:", precision_score(test_label, y_predictions))
	print("Recall:", recall_score(test_label, y_predictions))
	print("F1 score:", f1_score(test_label, y_predictions))


