import numpy as np
import cv2
import matplotlib
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
import time

import io

def show_optical_flow(frames, flow, strides):
	output_frames = []

	num_frames, height, width, _ = frames.shape

	# plt.ion() 
	for (i, frame) in enumerate(frames):
		fig = plt.figure()

		plt.clf()
		plt.imshow(frame)

		if i < num_frames:
			cur_flow = flow[i]
			for row in range(cur_flow.shape[0]):
				for col in range(cur_flow.shape[1]):
					if (cur_flow[row, col, 0] == 0 and cur_flow[row, col, 1] == 0): continue
					start_x, start_y = col * strides[0] + strides[0] // 2, row * strides[1] + strides[1] // 2
					flow_x, flow_y = cur_flow[row, col, 0], cur_flow[row, col, 1]
					if (flow_x > .1): flow_x = .1
					if (flow_x < -.1): flow_x = -.1

					if (flow_y > .1): flow_y = .1
					if (flow_y < -.1): flow_y = -.1
					plt.arrow(start_x, start_y, flow_x * 100, flow_y * 100, color="red")
		plot_img_np = get_img_from_fig(fig).astype('float32')
		output_frames.append(plot_img_np)		
		# plt.pause(1)
	# plt.ioff()
	# plt.show()

	return output_frames

def show_object_tracking(frames, flow, location):
	output_frames = []

	num_frames, height, width, channels = frames.shape

	# plt.ion()
	cur_location = location
	for (i, frame) in enumerate(frames):
		fig = plt.figure()

		darkening = 0.6
		plt.imshow(frame * darkening)

		if i < num_frames:
			cur_flow = flow[i]
			arrow_color = matplotlib.colors.hsv_to_rgb([i / num_frames, 1, 1])
			plt.arrow(cur_location[1], cur_location[0], cur_flow[0], cur_flow[1], color=arrow_color)
			cur_location = (cur_location[0] + cur_flow[1], cur_location[1] + cur_flow[0])
		
		plot_img_np = get_img_from_fig(fig).astype('float32')
		output_frames.append(plot_img_np)	
		
	# plt.ioff()

	return output_frames

def heatmap(flowList):
	for flow in flowList:
		normalizedFlow = flow / np.amax(flow)
		print("max:", np.amax(flow))
		print("min:", np.amin(flow))
		colorMap = np.zeros((flow.shape[0], flow.shape[1], 3))
		colorMap[:,:,0] = normalizedFlow[:,:,0]
		colorMap[:,:,1] = normalizedFlow[:,:,1]
		cv2.imshow("flow", colorMap)
		cv2.waitKey(0)


#taken from https://stackoverflow.com/questions/7821518/matplotlib-save-plot-to-numpy-array
# define a function which returns an image as numpy array from figure
def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img