import argparse
import pickle

## for Model definition/training
from keras.models import Model, load_model
from keras.layers import Input, Flatten, Dense, concatenate,  Dropout
from keras.optimizers import Adam, Nadam
from keras.applications.xception import Xception
# from keras.applications.resnet_v2 import ResNet50V2
from keras import backend as K
# from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint
from keras.layers.pooling import MaxPooling2D

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
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.patheffects as PathEffects

flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]


def scatter(x, labels, subtitle=None):
	# We choose a color palette with seaborn.
	palette = np.array(sns.color_palette(flatui, 2))

	# We create a scatter plot.
	f = plt.figure(figsize=(8, 8))
	ax = plt.subplot(aspect="equal")
	sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=40, c=palette[labels.astype(np.int)])
	plt.xlim(-25, 25)
	plt.ylim(-25, 25)
	ax.axis("off")
	ax.axis("tight")

	# We add the labels for each digit.
	txts = []
	for i in range(2):
		# Position of each label.
		xtext, ytext = np.median(x[labels == i, :], axis=0)
		txt = ax.text(xtext, ytext, str(i), fontsize=24)
		txt.set_path_effects(
			[PathEffects.Stroke(linewidth=5, foreground="w"), PathEffects.Normal()]
		)
		txts.append(txt)

	if subtitle != None:
		plt.suptitle(subtitle)

	plt.savefig(subtitle)


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

def triplets_loss(y_true, y_pred):
	
#     embeddings = K.cast(embeddings, 'float32')
#     with sess.as_default():
#         print(embeddings.eval())
	
	embeddings = y_pred
	anchor_positive = embeddings[:10]
	negative = embeddings[10:]
#     print(anchor_positive)

	# Compute pairwise distance between all of anchor-positive
	dot_product = K.dot(anchor_positive, K.transpose(anchor_positive))
	square = K.square(anchor_positive)
	a_p_distance = K.reshape(K.sum(square, axis=1), (-1,1)) - 2.*dot_product  + K.sum(K.transpose(square), axis=0) + 1e-6
	a_p_distance = K.maximum(a_p_distance, 0.0) ## Numerical stability
#     with K.get_session().as_default():
#         print(a_p_distance.eval())
#     print("Pairwise shape: ", a_p_distance)
#     print("Negative shape: ", negative)

	# Compute distance between anchor and negative
	dot_product_2 = K.dot(anchor_positive, K.transpose(negative))
	negative_square = K.square(negative)
	a_n_distance = K.reshape(K.sum(square, axis=1), (-1,1)) - 2.*dot_product_2  + K.sum(K.transpose(negative_square), axis=0)  + 1e-6
	a_n_distance = K.maximum(a_n_distance, 0.0) ## Numerical stability
	
	hard_negative = K.reshape(K.min(a_n_distance, axis=1), (-1, 1))
	
	distance = (a_p_distance - hard_negative + 0.2)
	loss = K.mean(K.maximum(distance, 0.0))/(2.)

#     with K.get_session().as_default():
#             print(loss.eval())
			
	return loss

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

	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-f", "--train_flag", required=True, type=str,
		help="Do you want to train the model??")
	ap.add_argument("-e", "--epochs", type=int, default=10)
	args = ap.parse_args()

	batch_size = 32
	epochs = args.epochs
	# train_flag = args["train_flag"]  # either     True or False
	train_flag = args.train_flag
	# print(train_flag)

	embedding_size = 64

	no_of_components = 2  # for visualization -> PCA.fit_transform()

	step = 10

	# The data, split between train and test sets
	train_data = np.load("embs_data_25f.npy")
	train_label = np.load("embs_label_25f.npy")
	# train_label = train_label.argmax(1)
	print("Dataset Loaded...")

	x_train, x_test, y_train, y_test = train_test_split(train_data, train_label, 
												  test_size=0.1, stratify=train_label,
												  random_state=34)
	print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

	input_image_shape = (512, )

	x_val = x_test[:2000, :]
	y_val = y_test[:2000]
	x_test = x_test[2000:, :]
	y_test = y_test[2000:]
	
	# Network training...
	if train_flag == "True":
		base_network = create_base_network(input_image_shape, embedding_size)
		for layer in base_network.layers:
			if layer.name.endswith('bn'):
				# print(layer.name)
				layer.trainable=False
		base_network.summary()

		input_images = Input(shape=input_image_shape, name='input_image') # input layer for images
		input_labels = Input(shape=(1,), name='input_label')    # input layer for labels
		embeddings = base_network([input_images])               # output of network -> embeddings
		labels_plus_embeddings = concatenate([input_labels, embeddings])  # concatenating the labels + embeddings

		# Defining a model with inputs (images, labels) and outputs (labels_plus_embeddings)
		model = Model(inputs=[input_images, input_labels],
					  outputs=labels_plus_embeddings)
		# model.summary()
		plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

		# train session
		optimizer = Adam(lr=3e-4)

		model.compile(loss=triplet_loss_adapted_from_tf,
					  optimizer=optimizer)


		# Uses 'dummy' embeddings + dummy gt labels. Will be removed as soon as loaded, to free memory
		dummy_gt_train = np.zeros((len(x_train), embedding_size + 1))
		dummy_gt_val = np.zeros((len(x_val), embedding_size + 1))

		H = model.fit(
			x=[x_train,y_train],
			y=dummy_gt_train,
			batch_size=batch_size,
			epochs=epochs,
			validation_data=([x_val, y_val], dummy_gt_val)
			# callbacks=callbacks_list
			)
		model.save("triplets_semi_hard.hdf5")
		
	else:
		model_tr = load_model("triplets_semi_hard.hdf5",
			custom_objects={'triplet_loss_adapted_from_tf':triplet_loss_adapted_from_tf})
		print("model loaded")
	
		# Test the network
		# creating an empty network
		testing_embeddings = create_base_network(input_image_shape,
												 embedding_size=embedding_size)
		x_train_before = testing_embeddings.predict(x_train)
		x_test_before = testing_embeddings.predict(x_test)

		print("Embeddings before training")
		sgd = linear_model.SGDClassifier(max_iter=50, tol=None)
		sgd.fit(x_train_before, y_train)
		Y_pred = sgd.predict(x_test_before)
		acc_sgd = accuracy_score(y_test, Y_pred)
		print("SGD Acc:", acc_sgd)

		# Grabbing the weights from the trained network
		for layer_target, layer_source in zip(testing_embeddings.layers, model_tr.layers[2].layers):
			weights = layer_source.get_weights()
			layer_target.set_weights(weights)
			del weights        

		x_train_after = testing_embeddings.predict(x_train)
		x_test_after = testing_embeddings.predict(x_test)

		print("Embeddings after training")
		sgd = linear_model.SGDClassifier(max_iter=50, tol=None, loss="log")
		
		sgd.fit(x_train_after, y_train)
		Y_pred = sgd.predict(x_test_after)
		acc_sgd = accuracy_score(y_test, Y_pred)
		print("SGD Acc:", acc_sgd)
		with open('sgd_classifier.pkl', 'wb') as fid:
			pickle.dump(sgd, fid)

		rf = RandomForestClassifier(n_estimators=100)
		rf.fit(x_train_after, y_train)
		y_pred = rf.predict(x_test_after)
		acc_rf = accuracy_score(y_test, y_pred)
		print("RF Acc:", acc_rf)
		with open('rf_classifier.pkl', 'wb') as fid:
			pickle.dump(rf, fid) 

		logreg = LogisticRegression()
		logreg.fit(x_train_after, y_train)
		y_pred = logreg.predict(x_test_after)
		acc_lg = accuracy_score(y_test, y_pred)
		print("LG Acc:", acc_lg)

		knn = KNeighborsClassifier(n_neighbors=10)
		knn.fit(x_train_after, y_train)
		y_pred = knn.predict(x_test_after)
		acc_knn = accuracy_score(y_test, y_pred)
		print("KNN Acc:", acc_knn)

		perceptron = Perceptron(max_iter=15)
		perceptron.fit(x_train_after, y_train)
		y_pred = perceptron.predict(x_test_after)
		acc_per = accuracy_score(y_test, y_pred)
		print("Perceptron Acc:", acc_per)

		dt = DecisionTreeClassifier()
		dt.fit(x_train_after, y_train)
		y_pred = dt.predict(x_test_after)
		acc_dt = accuracy_score(y_test, y_pred)
		print("DT Acc:", acc_dt)


		embed = "TSNE"

		if embed=="TSNE":
			tsne = TSNE()
			train_tsne_embeds_before_train = tsne.fit_transform(x_train_before[:512])
			train_tsne_embeds_after_train = tsne.fit_transform(x_train_after[:512])

			# val_tsne_bf4_train = tsne.fit_transform()

			scatter(train_tsne_embeds_before_train, y_train[:512], "Training Data Before TNN")
			scatter(train_tsne_embeds_after_train, y_train[:512], "Training Data After TNN")

		else:

			dict_embeddings = {}
			dict_gray = {}
			test_class_labels = np.unique(np.array(y_test))

			pca = PCA(n_components=no_of_components)
			decomposed_embeddings = pca.fit_transform(x_embeddings)
			print(decomposed_embeddings.shape)
			print(decomposed_embeddings[y_test == 1].shape)
			# x_test_reshaped = np.reshape(x_test, (len(x_test), 28 * 28))
			decomposed_gray = pca.fit_transform(x_embeddings_before_train)
			
			fig = plt.figure(figsize=(16, 8))
			for label in test_class_labels:
				decomposed_embeddings_class = decomposed_embeddings[y_test == label]
				# print("After train")
				# print(decomposed_embeddings_class.shape)
				# print(decomposed_embeddings_class[::step, 1], decomposed_embeddings_class[::step, 0])
				decomposed_gray_class = decomposed_gray[y_test == label]
				# print("Before train")
				# print(decomposed_gray_class.shape)

				plt.subplot(1,2,1)
				plt.scatter(decomposed_gray_class[::step,1], decomposed_gray_class[::step,0],label=str(label))
				plt.title('before training (embeddings)')
				plt.legend()
				# plt.savefig('before training')

				plt.subplot(1,2,2)
				plt.scatter(decomposed_embeddings_class[::step, 1], decomposed_embeddings_class[::step, 0], label=str(label))
				plt.title('after @%d epochs' % epochs)
				plt.legend()
			plt.savefig('final learning')

			# plt.show()
