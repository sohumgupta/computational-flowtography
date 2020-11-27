import numpy as np
import cv2
import matplotlib
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt

def show_optical_flow(frames, flow, strides):
	num_frames, height, width, channels = frames.shape

	plt.ion()
	for (i, frame) in enumerate(frames):
		plt.clf()
		plt.imshow(frame)

		cur_flow = flow[i]
		for i in range(cur_flow.shape[0]):
			for j in range(cur_flow.shape[1]):
				plt.arrow(j * strides[0], i * strides[1], cur_flow[i, j, 0], cur_flow[i, j, 1])

		plt.pause(1/60)
		
	plt.ioff()
	plt.show()