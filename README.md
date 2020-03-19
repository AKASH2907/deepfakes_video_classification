# deepfakes_classification
This repository provides the official Python implementation of Deepfakes Detection with Metric Learning.
<p align="center">
  <img src="https://user-images.githubusercontent.com/22872200/75561975-de8dee00-5a6d-11ea-8131-cab5cc736993.png">
</p>

## Dependencies

Tested on Python 3.6.x and Keras 2.3.0 with TF backend version 1.14.0.
* Numpy (1.16.4)
* OpenCV (4.1.0)
* Pandas (0.25.3)
* Scikit-learn (0.22.1)
* facenet-pytorch (2.0.1)
* PyTorch (1.2.0)

## Celeb-DF
It contains high resolution videos. 
Working on Celeb-DF dataset and XceptionNet using Imagenet weights.

In videos, with 5299/712 training distribution and 340/178 videos in testing distribution as real/fake videos. With frame rate 5, there are approximately 70K frames generated. 

2 scenarios with ImageNet: 
1) Loading ImageNet weights. Finetuning only final fc layers. (2-3M params)
2) Loading ImageNet weights. Training whole network. (22M params)

Although Celeb-DF face quality is better than FaceForensics++ c-40 videos, training directly on whole frames is not useful. Data imbalance plays a huge role that affects the weights of network. So, the dataset was divided into 7chunks of 1400 videos approximately: 700 fake and 700 real.

Frames contains a lot of noise and we have to focus on face. We used facenet model to extract faces from the whole video (can be done directly using videos or after extraction of frames), and then we trained XceptionNet for 50 epochs with EarlyStopping (patience=10) and ModelCheckpoint to save only the best mdoel by tracking the val_loss. We achieve the accuracy of 96% and after applying max voting the accuracy got boosted to 98%.

TSNE plot before and after training:

<p align="center">
  <img src="https://user-images.githubusercontent.com/22872200/74857763-29bb4900-536a-11ea-8562-61ded44123c1.png">
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/22872200/77095375-ce957880-6a33-11ea-9f4f-defd002326f6.png">
</p>

Grad CAM Activation maps:

<p align="center">
  <img src="https://user-images.githubusercontent.com/22872200/75562309-5d832680-5a6e-11ea-8d80-cf7e4eb327cf.png">
</p>

## FaceForensics++

FaceForensics++ dataset contains four types of forgeries:
* Face2Face
* FaceSwap
* Deepfakes
* Neural Texture

It contains 1000 manipulated videos of each type and 1000 real videos on which these 4 manipulations have been done. 

<p align="center">
  <img src="https://user-images.githubusercontent.com/22872200/75562036-f4031800-5a6d-11ea-9a2a-c34d693b0fca.png">
</p>

## Results

## Citation
If you find this work useful, please consider citing the following paper:


