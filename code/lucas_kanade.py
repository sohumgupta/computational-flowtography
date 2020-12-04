import numpy as np
import cv2
import sys
from scipy.signal import convolve2d
from scipy import ndimage
from helper import rgb2gray

import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import lsqr

def construct_lucas_kanade_matrices(cur_x, cur_y, cur_t, i, j, patch_size, stride):
	half_patch = patch_size // 2

	i_range_s, i_range_e = max(0, (i * stride - half_patch)), (i * stride + (patch_size - half_patch))
	j_range_s, j_range_e = max(0, (j * stride - half_patch)), (j * stride + (patch_size - half_patch))
	
	cur_patch_x = cur_x[i_range_s:i_range_e, j_range_s:j_range_e]
	cur_patch_y = cur_y[i_range_s:i_range_e, j_range_s:j_range_e]
	cur_patch_t = cur_t[i_range_s:i_range_e, j_range_s:j_range_e]

	A = np.stack([cur_patch_x.flatten(), cur_patch_y.flatten()], axis=1)
	b = np.reshape(cur_patch_t.flatten(), (-1, 1))
	return A, b

def get_image_gradients(cur_frame, next_frame, patch_size):
	cur_x = cv2.Sobel(cur_frame, cv2.CV_64F, 1, 0, ksize=patch_size)
	cur_y = cv2.Sobel(cur_frame, cv2.CV_64F, 0, 1, ksize=patch_size)
	kernel_t = np.ones((patch_size, patch_size)) / (patch_size * patch_size)
	cur_t = convolve2d(cur_frame, kernel_t, boundary='symm', mode='same') + convolve2d(next_frame, -kernel_t, boundary='symm', mode='same')
	return cur_x, cur_y, cur_t

def filter_flow(frame_flow, MED_FILTER_SIZE):
	frame_flow[:,:,0] = ndimage.median_filter(frame_flow[:,:,0], size=MED_FILTER_SIZE)
	frame_flow[:,:,1] = ndimage.median_filter(frame_flow[:,:,1], size=MED_FILTER_SIZE)
	return frame_flow

def clamp_flow(flow, clamp_low=-0.1, clamp_high=0.1):
	flow[flow < clamp_low] = clamp_low
	flow[flow > clamp_high] = clamp_high
	return flow

def lucas_kanade(frames, patch_size=5, stride=1):
	num_frames, height, width, channels = frames.shape
	flow = np.zeros((num_frames, height // stride, width // stride, 2))

	for f in range(num_frames-1):
		print(f"Tracking location for frame {f} -> {f+1}...")

		cur_frame, next_frame = rgb2gray(frames[f]), rgb2gray(frames[f+1])
		cur_x, cur_y, cur_t = get_image_gradients(cur_frame, next_frame, patch_size)
		num_equations = patch_size * patch_size
		num_pixels = (width // stride) * (height // stride)

		A = np.zeros((num_equations * num_pixels, 2 * num_pixels))
		b = np.zeros((num_equations * num_pixels, 1))
		
		cur_pixel = 0
		for i in range(height // stride):
			for j in range(width // stride):
				cur_A, cur_b = construct_lucas_kanade_matrices(cur_x, cur_y, cur_t, i, j, patch_size, stride)
				x = np.linalg.lstsq(cur_A, cur_b, rcond=None)[0]
				u, v = x[0:2, 0]
				flow[f, i, j] = [u, v]

		# median filter u and v coordinates for each frame
		# flow[f] = filter_flow(flow[f], MED_FILTER_SIZE=7)

	# flow = clamp_flow(flow)
	return flow

def lucas_kanade_slow(frames, patch_size=5, stride=1):
	num_frames, height, width, channels = frames.shape
	flow = np.zeros((num_frames, height // stride, width // stride, 2))

	for f in range(num_frames-1):
		print(f"Tracking location for frame {f} -> {f+1}...")

		cur_frame, next_frame = rgb2gray(frames[f]), rgb2gray(frames[f+1])
		cur_x, cur_y, cur_t = get_image_gradients(cur_frame, next_frame, patch_size)
		num_equations = patch_size * patch_size
		num_pixels = (width // stride) * (height // stride)

		A = np.zeros((num_equations * num_pixels, 2 * num_pixels))
		b = np.zeros((num_equations * num_pixels, 1))
		
		cur_pixel = 0
		for i in range(height // stride):
			for j in range(width // stride):
				cur_A, cur_b = construct_lucas_kanade_matrices(cur_x, cur_y, cur_t, i, j, patch_size, stride)
				if cur_A.shape[0] != num_equations:
					cur_A = np.pad(cur_A, ((0, num_equations - cur_A.shape[0]), (0, 0)))
					cur_b = np.pad(cur_b, ((0, num_equations - cur_b.shape[0]), (0, 0)))

				A[cur_pixel*num_equations:(cur_pixel + 1)*num_equations, cur_pixel*2:(cur_pixel+1)*2] = cur_A
				b[cur_pixel*num_equations:(cur_pixel + 1)*num_equations] = cur_b
				
				cur_pixel += 1
		
		A = sp.csr_matrix(A)
		x = lsqr(A, b)[0]
		x = np.reshape(x, (height // stride, width // stride, 2))
		flow[f] = x

		# median filter u and v coordinates for each frame
		# flow[f] = filter_flow(flow[f], MED_FILTER_SIZE=7)

	# flow = clamp_flow(flow)
	return flow