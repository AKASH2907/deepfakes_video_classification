# deepfakes_classification

## Celeb-DF
Classifying deepfakes in High resolution videos.

Working on Celeb-DF dataset and XceptionNet using Imagenet weights.

In videos, with 5299/712 training distribution and 340/178 videos in testing distribution as real/fake videos. With frame rate 5, there are approximately 70K frames generated. 

2 scenarios with ImageNet: 
1) Loading ImageNet weights. Finetuning only final fc layers. (2-3M params)
2) Loading ImageNet weights. Training whole network. (22M params)

Although Celeb-DF face quality is better than FaceForensics++ c-40 videos, training directly on whole frames is not useful.

Then, we used facenet model to extract faces from the whole video (can be done directly using videos or after extraction of frames), and then we trained XceptionNet for 50 epochs with EarlyStopping (patience=10) and ModelCheckpoint to save only the best mdoel by tracking the val_loss.
