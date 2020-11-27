import numpy as np
import argparse
import cv2
import os
from helper import load_video
from visualization import show_optical_flow

def parse_args():
	parser = argparse.ArgumentParser(description='Calculating and using optical flow for videos!')
	parser.add_argument('data', type=str, help='Name of input video (in data directory)')
	# parser.add_argument('--object-tracking', 
    #                 help='Optional argument for object tracking')
	args = parser.parse_args()
	return args

def main():
	args = parse_args()

	data_path = '../data'
	video_path = os.path.join(data_path, args.data)
	video_frames = load_video(video_path)

	video_frames = video_frames[0:5]

	flow = np.random.rand(video_frames.shape[0], video_frames.shape[1] // 10, video_frames.shape[2] // 10, 2)
	show_optical_flow(video_frames, flow * 5, (10, 10))

if __name__ =="__main__":
	main()
