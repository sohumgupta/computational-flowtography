import numpy as np
import argparse
import cv2
import os

def load_tracking_points(file_path):
	data_path = '../data'
	points_path = os.path.join(data_path, file_path)
	points_file = open(points_path, 'r') 
	points = points_file.readlines() 
	points = np.array([[int(s.split(",")[1]), int(s.split(",")[0])] for s in points])
	return points

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

## write out video frames to a file, specified by path
def write_video(frames, path, fps = 29.97):
	#this codec is hard coded for .mp4 or .MP4 file extensions. Use MJPG for .avi. If there's a bug in this file, it's probably this line
	videoWriter = cv2.VideoWriter(path,cv2.VideoWriter_fourcc(*'mp4v'), fps, (frames[0].shape[1],frames[0].shape[0]))
	if videoWriter is None:
		return
	
	for frame in frames:
		videoWriter.write((frame).astype('uint8')) #needs to be converted to uint8 for the codec
	
	if videoWriter is not None:
		print("Successfully wrote out to video", path)
		videoWriter.release()

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])