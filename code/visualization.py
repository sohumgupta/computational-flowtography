import numpy as np
import cv2
import matplotlib
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
import time

def show_optical_flow(frames, flow, strides):
	num_frames, height, width, channels = frames.shape

	plt.ion()
	for (i, frame) in enumerate(frames):
		plt.clf()
		plt.imshow(frame)

		if i < num_frames:
			cur_flow = flow[i]
			for i in range(cur_flow.shape[0]):
				for j in range(cur_flow.shape[1]):
					if (cur_flow[i, j, 0] == 0 and cur_flow[i, j, 1] == 0): continue
					plt.arrow(j * strides[0] + strides[0] // 2, i * strides[1] + strides[1] // 2, cur_flow[i, j, 0], cur_flow[i, j, 1], color="red")

		plt.pause(1)
		
	plt.ioff()
	plt.show()

def show_object_tracking(frames, flow, location):
	num_frames, height, width, channels = frames.shape

	plt.ion()
	cur_location = location
	for (i, frame) in enumerate(frames):
		darkening = 0.6
		plt.imshow(frame * darkening)

		if i < num_frames:
			cur_flow = flow[i]
			arrow_color = matplotlib.colors.hsv_to_rgb([i / num_frames, 1, 1])
			plt.arrow(cur_location[1], cur_location[0], cur_flow[0], cur_flow[1], color=arrow_color)
			cur_location = (cur_location[0] + cur_flow[1], cur_location[1] + cur_flow[0])

		plt.pause(1)
		
	plt.ioff()
	plt.show()