import numpy as np
import argparse
import cv2
import os
from helper import load_video
from visualization import show_optical_flow
from flow import naive_optical_flow

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
	frames = load_video(video_path)
	frames = frames[0:10]

	patch_size = (10, 10)
	flow = naive_optical_flow(frames, patch_size)
	show_optical_flow(frames, flow, patch_size)

if __name__ =="__main__":
	main()
