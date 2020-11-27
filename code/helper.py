import numpy as np
import argparse
import cv2
import os

def load_video(video_path):
	video = cv2.VideoCapture(video_path)
	success, frame = video.read()

	height, width, channels = frame.shape
	num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
	frames = np.zeros((num_frames, height, width, channels))

	count = 0
	while success:
		frames[count] = frame
		success, frame = video.read()
		count += 1

	frames = frames[:count]
	frames = frames / 255
	return frames