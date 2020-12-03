import numpy as np
import cv2
import sys
from scipy.signal import convolve2d

def ssd(a, b):
	return np.sum(np.square(a - b))

def naive_optical_flow(frames, patch_size):
	num_frames, height, width, channels = frames.shape
	num_patches = (height // patch_size, width // patch_size)
	flow = np.zeros((num_frames, num_patches[0], num_patches[1], 2))

	for f in range(num_frames-1):
		print(f"\nCalculating for frame {f} -> {f+1}...")
		cur_frame, next_frame = frames[f], frames[f+1]
		for y in range(num_patches[0]):
			for x in range(num_patches[1]):
				sys.stdout.write(f"\rFinding best match for patch ({y + 1}/{num_patches[0]}, {x + 1}/{num_patches[1]})")
				cur_patch = cur_frame[
					y*patch_size:(y+1)*patch_size, 
					x*patch_size:(x+1)*patch_size
				]
				best_patch, best_ssd = (0, 0), float('inf')
				for i in range(-patch_size // 2, patch_size // 2, 1):
					for j in range(-patch_size // 2, patch_size // 2):
						next_patch = next_frame[
							max(0, y*patch_size + i):y*patch_size + i + patch_size, 
							max(0, x*patch_size + j):x*patch_size + j + patch_size
						]
						if next_patch.shape != cur_patch.shape: continue
						error = ssd(cur_patch, next_patch)
						if error < best_ssd: 
							best_patch, best_ssd = (j, i), error
				flow[f, y, x] = np.array(best_patch)
				sys.stdout.flush()
	
	return flow

def naive_object_tracking(frames, location, patch_size):
	num_frames, height, width, channels = frames.shape
	flow = np.zeros((num_frames, 2))

	cur_location = location
	for f in range(num_frames-1):
		print(f"Tracking location for frame {f} -> {f+1}...")
		cur_frame, next_frame = frames[f], frames[f+1]

		cur_patch = cur_frame[
			cur_location[0]:cur_location[0]+patch_size,
			cur_location[1]:cur_location[1]+patch_size
		]

		best_patch, best_ssd = (0, 0), float('inf')
		for i in range(-patch_size // 2, patch_size // 2):
			for j in range(-patch_size // 2, patch_size // 2):
				next_patch = next_frame[
					max(0, cur_location[0] + i):cur_location[0] + i + patch_size, 
					max(0, cur_location[1] + j):cur_location[1] + j + patch_size
				]
				if next_patch.shape != cur_patch.shape: continue
				error = ssd(cur_patch, next_patch)
				if error < best_ssd: 
					best_patch, best_ssd = (j, i), error
		flow[f] = np.array(best_patch)
		cur_location = (cur_location[0] + best_patch[1], cur_location[1] + best_patch[0])
	return flow

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def lucas_kanade(frames, patch_size=5, stride=1):
	num_frames, height, width, channels = frames.shape
	flow = np.zeros((num_frames, height // stride, width // stride, 2))

	for f in range(num_frames-1):
		print(f"\nTracking location for frame {f} -> {f+1}...")
		cur_frame, next_frame = frames[f], frames[f+1]
		cur_frame, next_frame = rgb2gray(cur_frame), rgb2gray(next_frame)

		cur_x = cv2.Sobel(cur_frame, cv2.CV_64F, 1, 0, ksize=patch_size)
		cur_y = cv2.Sobel(cur_frame, cv2.CV_64F, 0, 1, ksize=patch_size)
		kernel_t = np.ones((patch_size, patch_size)) / (patch_size * patch_size)
		cur_t = convolve2d(cur_frame, kernel_t, boundary='symm', mode='same') + convolve2d(next_frame, -kernel_t, boundary='symm', mode='same')

		half_patch = patch_size // 2
		for i in range(height // stride):
			for j in range(width // stride):
				sys.stdout.write(f"\rFinding best match for pixel ({(i + 1) * stride}/{height}, {(j + 1) * stride}/{width})")

				i_range_s, i_range_e = max(0, (i * stride - half_patch)), (i * stride + (patch_size - half_patch))
				j_range_s, j_range_e = max(0, (j * stride - half_patch)), (j * stride + (patch_size - half_patch))
				
				cur_patch_x = cur_x[i_range_s:i_range_e, j_range_s:j_range_e]
				cur_patch_y = cur_y[i_range_s:i_range_e, j_range_s:j_range_e]
				cur_patch_t = cur_t[i_range_s:i_range_e, j_range_s:j_range_e]

				A = np.stack([cur_patch_x.flatten(), cur_patch_y.flatten()], axis=1)
				b = np.reshape(cur_patch_t.flatten(), (-1, 1))
				x = np.linalg.lstsq(A, b, rcond=None)[0]
				u, v = x[0:2, 0]
				flow[f, i, j] = [u, v]
				sys.stdout.flush()
	
	print("\n")
	return flow