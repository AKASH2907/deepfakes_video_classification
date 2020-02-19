# deepfakes_classification

## Celeb-DF
Classifying deepfakes in High resolution videos.

Working on Celeb-DF dataset and XceptionNet using Imagenet weights.

In videos, with 5299/712 training distribution and 340/178 videos in testing distribution as real/fake videos. With frame rate 5, there are approximately 70K frames generated. 

2 scenarios with ImageNet: 
1) Loading ImageNet weights. Finetuning only final fc layers. (2-3M params)
2) Loading ImageNet weights. Training whole network. (22M params)

Although Celeb-DF face quality is better than FaceForensics++ c-40 videos, training directly on whole frames is not useful. Data imbalance plays a huge role that affects the weights of network. So, the dataset was divided into 7chunks of 1400 videos approximately: 700 fake and 700 real.

Frames contains a lot of noise and we have to focus on face. We used facenet model to extract faces from the whole video (can be done directly using videos or after extraction of frames), and then we trained XceptionNet for 50 epochs with EarlyStopping (patience=10) and ModelCheckpoint to save only the best mdoel by tracking the val_loss. We achieve the accuracy of 96% and after applying max voting the accuracy got boosted to 98%.

TSNE plot before and after training:

![celeb_df_tsne](https://user-images.githubusercontent.com/22872200/74857763-29bb4900-536a-11ea-8562-61ded44123c1.png)

## FaceForensics++

FaceForensics++ dataset contains four types of forgeries:
* Face2Face
* FaceSwap
* Deepfakes
* Neural Texture

It contains 1000 manipulated videos 


