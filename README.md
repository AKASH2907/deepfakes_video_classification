# deepfakes_classification
[![arXiv](https://img.shields.io/badge/arXiv-2003.08645-b31b1b.svg)](http://arxiv.org/abs/2003.08645)

This repository provides the official Python implementation of [Deepfakes Detection with Metric Learning](http://arxiv.org/abs/2003.08645) accepted at 8th International Workshop on Biometrics and Forensics. Medium blog post is shared here: [deepfakes-classification-via-metric-learning](https://medium.com/@akash29/deepfakes-classification-via-metric-learning-89fa5179c920)

<p align="center">
  <img src="https://user-images.githubusercontent.com/22872200/75561975-de8dee00-5a6d-11ea-8131-cab5cc736993.png">
</p>

## Table of Contents

- [Requirements](#requirements)
- [Celeb-DF](#celeb-df)
- [Face-forensics](#face-forensics)
- [Results](#results)
- [Citation](#citation)
- [Notes](#notes)

## Requirements

Tested on Python 3.6.x and Keras 2.3.0 with TF backend version 1.14.0.
* Numpy (1.16.4)
* OpenCV (4.1.0)
* Pandas (0.25.3)
* Scikit-learn (0.22.1)
* facenet-pytorch (2.0.1)
* PyTorch (1.2.0)

## Getting Started

* Install the required dependencies:
 ```javascript
 pip install -r requirements.txt
 ```
* [frames_extraction.py](https://github.com/AKASH2907/deepfakes_video_classification/blob/master/frames_extraction.py) - Extract frames from videos
* [face_extraction.py](https://github.com/AKASH2907/deepfakes_video_classification/blob/master/face_extraction.py) - Extract faces from frames/videos for training purpose
* [train_data_prepare.py](https://github.com/AKASH2907/deepfakes_video_classification/blob/master/train_data_prepare.py) - Selecting the first n frames per video and then saving it to numpy file for CNN training
 ```
 python 00.train_data_prepare.py -img_size 160 -fpv 25
 
-  [-img_size] IMG_SIZE, Resize face image size
-  [-fpv] FRAMES_PER_VIDEO, Number of frames per video to consider

 ```
* [train_CNN.py](https://github.com/AKASH2907/deepfakes_video_classification/blob/master/01.train_CNN.py) - Train on 2D CNN - XceptionNet model
 ```
 python 01.train_CNN.py -e 20 -m xception -w xception_best -b 32 -im_size 160
 
-  [-e] EPOCHS, Number of epochs
-  [-m] MODEL_NAME, Imagenet architecture
-  [-w] WEIGHTS_SAVE_NAME, Model weights name
-  [-b] BATCH_SIZE, Batch size
-  [-im_size] IMAGE_SIZE, Input image dimension
 ```
* [evaluate_CNN.py](https://github.com/AKASH2907/deepfakes_video_classification/blob/master/02.evaluate_CNN.py) - Evaluate the testing video accuracy using CNN

```
python 02.evaluate_CNN.py -m ef0 -w ef0.hdf5 -im_size 160

- [-m] MODEL_NAME, Imagenet architecture, ef0=EfficientNet0
- [-w] LOAD_WEIGHTS_NAME, Load saved weights
- [-im_size] IMAGE_SIZE, Input image dimension
```
* [train_C3D.py](https://github.com/AKASH2907/deepfakes_video_classification/blob/master/03.train_C3D.py) - Train on Convolutional 3D architecture
 ```
 python 03.train_C3D.py -e 15 -m c3d -b 32 -im_size 160
 
-  [-e] EPOCHS, Number of epochs
-  [-m] MODEL_NAME, conv3d or c3d model
-  [-w] WEIGHTS_SAVE_NAME, Model weights name
-  [-b] BATCH_SIZE, Batch size
 ```
 
* [evaluate_C3D.py](https://github.com/AKASH2907/deepfakes_video_classification/blob/master/04.evaluate_C3D.py) - Evaluate videos using 3D CNN architecture

```
python 04.evaluate_C3D.py -m c3d -w trained_wts/weights_c3d.h5 -b 32

- [-m] MODEL_NAME, Imagenet architecture, c3d/conv3d
- [-w] LOAD_WEIGHTS_NAME, Load saved weights
- [-b] BATCH_SIZE, Batch size
```

* [feature_extractor.py](https://github.com/AKASH2907/deepfakes_video_classification/blob/master/05.lstm_features.py) - Extract features for recurrence networks
```
python 05.lstm_features.py -seq 25 -m ef0 -w ef0 -im_size 160

- [-seq] SEQ_LENGTH, Number of frames to consider per video
- [-m] MODEL_NAME, Imagenet architecture, ef0=EfficientNet0
- [-w] LOAD_WEIGHTS_NAME, Load saved weights
- [-im_size] IMAGE_SIZE, Input image dimension 
```

* [train_CNN_LSTM.py](https://github.com/AKASH2907/deepfakes_video_classification/blob/master/06.train_CNN_LSTM.py) - Train the LSTM/GRU (BiDirectional)/ Temporal models
 ```
 python 06.train_CNN_LSTM.py -e 15 -b 32 -w lstm_wts
 
-  [-e] EPOCHS, Number of epochs
-  [-w] WEIGHTS_SAVE_NAME, Model weights name
-  [-b] BATCH_SIZE, Batch size
 ```

* [evaluate_CNN_LSTM.py](https://github.com/AKASH2907/deepfakes_video_classification/blob/master/07.evaluate_CNN_LSTM.py) - Evaluate the recurrence models
```
python 07.evaluate_CNN_LSTM.py -seq 25 -m ef0 -w ef0 -im_size 160

- [-seq] SEQ_LENGTH, Number of frames to consider per video
- [-m] MODEL_NAME, Imagenet architecture, ef0=EfficientNet0
- [-w] LOAD_WEIGHTS_NAME, Load saved weights
- [-im_size] IMAGE_SIZE, Input image dimension 
```

* [facenet_embeddings.py](https://github.com/AKASH2907/deepfakes_video_classification/blob/master/08.facenet_embeddings.py) - Generate face embedding vectors of each face

* [train_triplets_semi_hard.py](https://github.com/AKASH2907/deepfakes_video_classification/blob/master/09.train_triplets_semi_hard.py) - Train the triplets of face embedding vectors and then train ML classifiers such as SGD, Random Forest, etc. to classify feature vectors

```
python 10.train_triplets_semi_hard.py -f True -e 20

- [-f] TRAIN_FLAG, True-> Train else test
- [-e] EPOCHS, Number of epochs
```

* [evaluate_triplets.py](https://github.com/AKASH2907/deepfakes_video_classification/blob/master/10.evaluate_triplets.py) - Evaluate the testing video embeddings uing trained ML classifiers

## Celeb-DF
It contains high resolution videos, with 5299/712 training distribution and 340/178 videos in testing distribution as real/fake videos. With frame rate 5, there are approximately 70K frames generated. 

Although Celeb-DF face quality is better than FaceForensics++ c-40 videos, training directly on whole frames is not useful. Therefore, we extracted faces from frames and then utilised that for classification. Data imbalance plays a huge role that affects the weights of network. In our case, it was 7:1. We applied bagging and boosting algorithm. So, the dataset was divided into 7 chunks of 1400 videos approximately: 700 fake and 700 real. It was trained on each distribution and then performance was boosted by max voting all the predictions.

Frames contains a lot of noise and we have to focus on face. We used facenet model to extract faces from the whole video (can be done directly using videos or after extraction of frames), and then we trained XceptionNet for 50 epochs with EarlyStopping (patience=10) and ModelCheckpoint to save only the best mdoel by tracking the val_loss. We achieve the accuracy of 96% and after boosting accuracy improves to 98%.

<p align="center">
  <img src="https://user-images.githubusercontent.com/22872200/74857763-29bb4900-536a-11ea-8562-61ded44123c1.png">
  </br> <caption> TSNE plot before and after training using frames only (2D-CNN- Xception) </caption>
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/22872200/80180961-887a9880-8621-11ea-8381-2579c11f323b.png">
  </br><caption> TSNE plot before and after training using Triplet Network </caption>
</p>


<p align="center">
  <img src="https://user-images.githubusercontent.com/22872200/75562309-5d832680-5a6e-11ea-8d80-cf7e4eb327cf.png">
  </br><caption> Grad CAM Activation maps </caption>
</p>

## Face-forensics

FaceForensics++ dataset contains four types of forgeries:
* Face2Face
* FaceSwap
* Deepfakes
* Neural Texture

It contains 1000 manipulated videos of each type and 1000 real videos on which these 4 manipulations have been done. 

<p align="center">
  <img src="https://user-images.githubusercontent.com/22872200/80180728-05594280-8621-11ea-97f4-f7ff17482a5a.png">
  </br><caption> Final Architecture </caption>
</p>


<p align="center">
  <img src="https://user-images.githubusercontent.com/22872200/75562036-f4031800-5a6d-11ea-9a2a-c34d693b0fca.png">
  </br><caption> TSNE plot of FaceForensics++ dataset </caption>
</p>

## Results

<p align="center">
  <img src="https://user-images.githubusercontent.com/22872200/77188792-8db56680-6afc-11ea-8323-9f2275da1a89.png">
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/22872200/77188833-9e65dc80-6afc-11ea-9072-2e836d6bce58.png">
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/22872200/77188862-a9b90800-6afc-11ea-8e6d-0749815625a5.png">
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/22872200/77188893-b89fba80-6afc-11ea-9689-398bd1b268cd.png">
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/22872200/77188925-c6554000-6afc-11ea-8bf3-d3bb595d82fc.png">
</p>


## Citation
If you find this work useful, please consider citing the following paper:

 ```
@INPROCEEDINGS{9107962,
  author={A. {Kumar} and A. {Bhavsar} and R. {Verma}},
  booktitle={2020 8th International Workshop on Biometrics and Forensics (IWBF)}, 
  title={Detecting Deepfakes with Metric Learning}, 
  year={2020},
  volume={},
  number={},
  pages={1-6},}
```

## Notes
For FaceForensics++ and Celeb-DF dataset, contact the authors of the dataset. The dataset can't be shared with the third party. You need to accept the terms on their pages. Then, they will provide you with the access.

Codes have been formatted in PEP-8 styling so that it's easy to read and reproducible to all. If any errors you face in the repo, please raise a issue. (Any place where I should explain more) I'll be happy to resolve it as soon as possible.
