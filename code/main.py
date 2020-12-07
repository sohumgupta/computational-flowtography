import numpy as np
import argparse
import cv2
import os
from helper import load_video, write_video, load_tracking_points
from visualization import show_optical_flow, show_object_tracking, heatmap
from lucas_kanade import lucas_kanade, pyramid_lucas_kanade
from naive_flow import naive_optical_flow, naive_object_tracking

def parse_args():
	parser = argparse.ArgumentParser(description='Calculating and using optical flow for videos!')
	parser.add_argument('data', type=str, help='Name of input video (in data directory)')
	parser.add_argument('algorithm', type=str, help='Algorithm to use for calculating optical flow (NAIVE or LUCAS_KANADE)')
	parser.add_argument('--tracking', type=str, help='Name of input .txt file with tracking points (in data directory)')
	parser.add_argument('--output_path', type=str, help="path to output video")
	parser.add_argument('--heatmap', action='store_true',
                    help='Visualize with a heatmap')
	args = parser.parse_args()
	if (args.algorithm not in ['NAIVE', 'LUCAS_KANADE']):
		parser.error("Algorithm must be one of NAIVE or LUCAS_KANADE.")
	return args

def main():
	args = parse_args()

	data_path = '../data'
	video_path = os.path.join(data_path, args.data)
	frames = load_video(video_path)

	if args.algorithm == 'NAIVE':
		print(f"Using naive algorithm.")
		object_tracking = naive_object_tracking
		optical_flow = naive_optical_flow
	elif args.algorithm == 'LUCAS_KANADE':
		print(f"Using Lucas-Kanade algorithm.")
		object_tracking = pyramid_lucas_kanade
		optical_flow = lucas_kanade
	
	if (args.tracking):
		frames = frames[0:50]

		patch_size = 7
		points = load_tracking_points(args.tracking)
		# points = points[0:1]
		flow = object_tracking(frames, np.copy(points), patch_size)
		out_frames = show_object_tracking(frames, flow, points)
	else:
		frames = frames[0:10]

		patch_size = 3
		stride = 5
		flow = optical_flow(frames, patch_size, stride)
		out_frames = show_optical_flow(frames, flow, stride)

	if(args.heatmap):
		out_frames = heatmap(flow)
	
	if (args.output_path is not None):
		write_video(out_frames, args.output_path)

if __name__ =="__main__":
	main()
