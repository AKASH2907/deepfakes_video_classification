# deepfakes_classification
Classifying deepfakes in HIgh resolution videos.

Working on Celeb-DF dataset and XceptionNet using Imagenet weights.

In videos, with 5299/712 training distribution and 340/178 videos in testing distribution, with frame rate there are approximately, 70,000 frames ca be generated. 

4 scenarios with ImageNet: 
1) Loading ImageNet weights. Finetuning only final fc layers.
2) Loading ImageNet weights. Training whole network.
3) Loading No weights.
4) Loading Random initialization.

In the last two scenarios,  a very large number of epochs would be required to optimize it perfectly. 

Now, if we do finetune by adding final few layers, let's say we have 2 to 3 million parameters and 70,000 images. It will overfit the model and also since thre's a huge data imbalance, direct training will not help network learn the tampering on the image faces.

Although Celeb-DF face quality is better than FaceForensics++ c-40 videos, training directly via frames is not useful.

Then, we used facenet model to extract faces from the whole video (can be done directly using videos or after extraction of frames), and then we trained XceptionNet for 50 epochs with EarlyStopping (patience=10) and ModelCheckpoint to save only the best mdoel by tracking the val_loss.
