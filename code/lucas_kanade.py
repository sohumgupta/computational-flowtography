import numpy as np
import cv2
import sys
from scipy.signal import convolve2d
from scipy import ndimage
from helper import rgb2gray

import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import lsqr


import matplotlib
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt


def construct_lucas_kanade_matrices(cur_x, cur_y, cur_t, i, j, patch_size, stride):
	half_patch = patch_size // 2

	i_range_s, i_range_e = max(0, (i * stride - half_patch)), (i * stride + half_patch + 1)
	j_range_s, j_range_e = max(0, (j * stride - half_patch)), (j * stride + half_patch + 1)
	
	cur_patch_x = cur_x[i_range_s:i_range_e, j_range_s:j_range_e].flatten()
	cur_patch_y = cur_y[i_range_s:i_range_e, j_range_s:j_range_e].flatten()
	cur_patch_t = cur_t[i_range_s:i_range_e, j_range_s:j_range_e].flatten()

	A = np.stack([cur_patch_y, cur_patch_x], axis=1)
	b = -cur_patch_t
	return A, b

def get_image_gradients(cur_frame, next_frame, patch_size):
	kernel_t = np.ones((patch_size, patch_size)) / (patch_size * patch_size)
	cur_t = convolve2d(next_frame, kernel_t, boundary='symm', mode='same') + convolve2d(cur_frame, -kernel_t, boundary='symm', mode='same')
	
	cur_x = cv2.Sobel(cur_frame, cv2.CV_64F, 1, 0, ksize=patch_size)
	cur_y = cv2.Sobel(cur_frame, cv2.CV_64F, 0, 1, ksize=patch_size)

	return cur_x, cur_y, cur_t

def filter_flow(frame_flow, MED_FILTER_SIZE):
	frame_flow[:,:,0] = ndimage.median_filter(frame_flow[:,:,0], size=MED_FILTER_SIZE)
	frame_flow[:,:,1] = ndimage.median_filter(frame_flow[:,:,1], size=MED_FILTER_SIZE)
	return frame_flow

def clamp_flow(flow, clamp_low=-0.1, clamp_high=0.1):
	flow[flow < clamp_low] = clamp_low
	flow[flow > clamp_high] = clamp_high
	return flow

def check_threshold(A, tau=0.01):
	AtA = np.matmul(np.transpose(A), A)
	eigs, _ = np.linalg.eig(AtA)
	return tau <= np.min(eigs)

def single_frame_lucas_kanade(cur_frame, next_frame, patch_size, stride):
	height, width = cur_frame.shape
	flow = np.zeros((height // stride, width // stride, 2))

	cur_x, cur_y, cur_t = get_image_gradients(cur_frame, next_frame, patch_size)
	cur_pixel = 0
	for i in range(height // stride):
		for j in range(width // stride):
			A, b = construct_lucas_kanade_matrices(cur_x, cur_y, cur_t, i, j, patch_size, stride)
			if not check_threshold(A): 
				continue
			x = np.linalg.lstsq(A, b, rcond=None)[0]
			flow[i, j] = x * 50

	return flow

def lucas_kanade(frames, patch_size=5, stride=1):
	num_frames, height, width, channels = frames.shape
	flow = np.zeros((num_frames, height // stride, width // stride, 2))

	for f in range(num_frames-1):
		print(f"Computing optical flow for frame {f} -> {f+1}...")

		cur_frame, next_frame = rgb2gray(frames[f]), rgb2gray(frames[f+1])
		cur_flow = single_frame_lucas_kanade(cur_frame, next_frame, patch_size, stride)

		flow[f] = filter_flow(cur_flow, MED_FILTER_SIZE=5) # median filter u and v coordinates for each frame

	return flow

def image_pyramid(image, num_levels=4):
	scales = [2 ** i for i in range(num_levels)][::-1]
	images = [cv2.resize(image, (image.shape[1] // scale, image.shape[0] // scale)) for scale in scales]
	return images

def sparse_single_frame_lucas_kanade(cur_frame, next_frame, point, patch_size):
	cur_x, cur_y, cur_t = get_image_gradients(cur_frame, next_frame, patch_size)

	A, b = construct_lucas_kanade_matrices(cur_x, cur_y, cur_t, point[0], point[1], patch_size, 1)
	if not check_threshold(A): 
		return np.array([0, 0])
	x = np.linalg.lstsq(A, b, rcond=None)[0]
	return x

def translate_image(image, dy, dx):
	dx, dy = int(np.rint(dx)), int(np.rint(dy))
	translated = np.zeros(image.shape)

	if dy == 0: translated = image
	elif dy > 0: translated[dy:] = image[:-dy]
	else: translated[:dy] = image[-dy:]

	if dx == 0: translated = translated
	elif dx > 0: translated[:, dx:] = translated[:, :-dx]
	else: translated[:, :dx] = translated[:, -dx:]

	return translated

def pyramid_lucas_kanade(frames, points, patch_size=5):
	num_frames, num_points = frames.shape[0], points.shape[0]
	flow = np.zeros((num_points, num_frames, 2))

	for p in range(len(points)):
		cur_point = points[p]
		print(f"Tracking point ({cur_point[1]},{cur_point[0]})...")

		for f in range(num_frames-1):
			# print(cur_point)
			cur_flow = np.array([0.0, 0.0])
			cur_frame, next_frame = rgb2gray(frames[f]), rgb2gray(frames[f+1])
			num_levels = 4
			cur_frame_pyramid, next_frame_pyramid = image_pyramid(cur_frame, num_levels), image_pyramid(next_frame, num_levels)
			cur_point = cur_point / 2**num_levels

			for i in range(num_levels):
				cur_point *= 2
				cur_frame_level, next_frame_level = cur_frame_pyramid[i], next_frame_pyramid[i]
				warped_next_frame_level = translate_image(next_frame_level, cur_flow[0], cur_flow[1])
				cur_level_flow = sparse_single_frame_lucas_kanade(cur_frame_level, warped_next_frame_level, cur_point.astype(int), patch_size)
				# cur_level_flow = np.array(cur_level_flow) * 1000
				cur_flow = (cur_flow + cur_level_flow) * 2
				# print(cur_flow)

			cur_point += cur_flow
			flow[p, f] = cur_flow
			# print()
			
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