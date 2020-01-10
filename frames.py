import cv2
import math
from os import listdir, makedirs
from os.path import join, sep, split, exists
import glob

train_path = ["test/0", "test/1"]

for i in train_path:
	vids = glob.glob(join(i, '*.mp4'))
	folder = i.split('/')[1]
	# vids = listdir(i)
	# print(len(vids))
	j = 0
	for v in vids:
		cap = cv2.VideoCapture(v)
		# print(cap.isOpened())
		# print(split(v)[0])
		vid = v.split('/')[-1]
		vid = vid.split('.')[0]
		# print(vid)
		frameRate = cap.get(5) #frame rate
		# if not exists("test_vids/"+ folder + "/video_" + str(int(j))):
		# 	makedirs("test_vids/"+ folder + "/video_" + str(int(j)))
		while(cap.isOpened()):
			frameId = cap.get(1) #current frame number
			ret, frame = cap.read()
			# print(frame.shape)
			if (ret != True):
				break
			if (frame.any() != None):
				frame = cv2.resize(frame, (320, 320), interpolation = cv2.INTER_AREA)
			
			if ((frameId % math.floor(frameRate)) == 0):
				# filename = 'train_frames/' + folder + '/' + vid + "_image_" +  str(int(frameId)) + ".jpg"
				filename = "test_vids/" + folder + "/video_" + str(int(j)) + "/image_" + str(int(frameId / math.floor(frameRate))+1) + ".jpg"
				# print(filename)
				cv2.imwrite(filename, frame)
		cap.release()

		if j%100==0:
			print(j, "Done....")
		j+=1