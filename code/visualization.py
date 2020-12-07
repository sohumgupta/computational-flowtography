import numpy as np
import cv2
import matplotlib
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
import time

import io

def show_optical_flow(frames, flow, stride):
	output_frames = []
	epsilon = 1e-3
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
					if (abs(cur_flow[row, col, 0]) < epsilon and abs(cur_flow[row, col, 1]) < epsilon): continue
					start_x, start_y = col * stride + stride // 2, row * stride + stride // 2
					flow_x, flow_y = cur_flow[row, col, 1], cur_flow[row, col, 0]
					plt.arrow(start_x, start_y, flow_x, flow_y, color="red", head_width=0.5, head_length=0.5)
		plot_img_np = get_img_from_fig(fig).astype('float32')
		output_frames.append(plot_img_np)
		# plt.pause(1)
	# plt.ioff()
	# plt.show()

	return output_frames

def show_object_tracking(frames, flow, locations):
	output_frames = []
	num_frames, height, width, channels = frames.shape

	plt.ion()
	cur_locations = np.array(locations, dtype=np.float32)
	for (i, frame) in enumerate(frames):
		fig = plt.figure()

		darkening = 0.5
		plt.imshow(frame * darkening)

		if i < num_frames:
			arrow_color = matplotlib.colors.hsv_to_rgb([i / num_frames, 1, 1])
			for p in range(len(cur_locations)):
				cur_flow = flow[p, i]
				cur_point = cur_locations[p]
				plt.arrow(
					int(np.rint(cur_point[1])), int(np.rint(cur_point[0])),
					-cur_flow[1], cur_flow[0], 
					color=arrow_color
				)
				cur_locations[p] += [cur_flow[1], cur_flow[0]]
		
		plot_img_np = get_img_from_fig(fig).astype('float32')
		output_frames.append(plot_img_np)	
		# plt.pause(.00001)
		
	# plt.ioff()
	# plt.show()

	return output_frames

def heatmap(flowList):
	frames = []
	for flow in flowList:
		normalizedFlow = (flow) / 0.4
		colorMap = np.zeros((flow.shape[0], flow.shape[1], 3))

		#g = y and b = x visualization
		colorMap[:,:,0] = normalizedFlow[:,:,0]
		colorMap[:,:,1] = normalizedFlow[:,:,1]
		
		#euclid dist visualization
		#colorMap[:,:,1] = np.sqrt(np.square(normalizedFlow[:,:,1]) + np.square(normalizedFlow[:,:,0]))


		# angle visualization
		# mag, ang = cv2.cartToPolar(normalizedFlow[:,:,0], normalizedFlow[:,:,1])
		# hsv[...,0] = ang*180/np.pi/2
		# hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
		# rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

		frames.append(colorMap * 255)
	return frames



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