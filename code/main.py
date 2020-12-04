import numpy as np
import argparse
import cv2
import os
from helper import load_video, write_video
from visualization import show_optical_flow, show_object_tracking, heatmap
from lucas_kanade import lucas_kanade
from naive_flow import naive_optical_flow, naive_object_tracking

def parse_args():
	parser = argparse.ArgumentParser(description='Calculating and using optical flow for videos!')
	parser.add_argument('data', type=str, help='Name of input video (in data directory)')
	parser.add_argument('algorithm', type=str, help='Algorithm to use for calculating optical flow (NAIVE or LUCAS_KANADE)')
	parser.add_argument('--tracking', action='store_true',
                    help='Optional argument for object tracking')
	parser.add_argument('--track_x', type=int, help="x-position for object to track")
	parser.add_argument('--track_y', type=int, help="y-position for object to track")
	parser.add_argument('--output_path', type=str, help="path to output video")
	parser.add_argument('--heatmap', action='store_true',
                    help='Visualize with a heatmap')
	args = parser.parse_args()
	if (args.tracking and not (args.track_x and args.track_y)):
		parser.error("--tracking requires --track_x and --track_y.")
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
		object_tracking = naive_object_tracking # change this eventually
		optical_flow = lucas_kanade
	

	if (args.tracking):
		frames = frames[0:300]
		patch_size = 5
		flow = object_tracking(frames, (args.track_y, args.track_x), patch_size)
		out_frames = show_object_tracking(frames, flow, (args.track_y, args.track_x))
	else:
		frames = frames[0:10]
		patch_size = 5
		stride = 5
		flow = optical_flow(frames, patch_size, stride)
		# out_frames = show_optical_flow(frames, flow, stride)

	if(args.heatmap):
		out_frames = heatmap(flow)
	
	if (args.output_path is not None):
		write_video(out_frames, args.output_path)
		# print("frames:", out_frames)
		# write_video(out_frames, args.output_path)

if __name__ =="__main__":
	main()
