import numpy as np
import cv2
import sys
from scipy.signal import convolve2d
from scipy import ndimage

def ssd(a, b):
	return np.sum(np.square(a - b))

def naive_optical_flow(frames, patch_size, stride):
	num_frames, height, width, channels = frames.shape
	num_patches = (height // stride, width // stride)
	flow = np.zeros((num_frames, num_patches[0], num_patches[1], 2))

	for f in range(num_frames-1):
		print(f"\nCalculating for frame {f} -> {f+1}...")
		cur_frame, next_frame = frames[f], frames[f+1]
		for y in range(num_patches[0]):
			for x in range(num_patches[1]):
				sys.stdout.write(f"\rFinding best match for patch ({y + 1}/{num_patches[0]}, {x + 1}/{num_patches[1]})")
				cur_patch = cur_frame[
					y*stride:(y+1)*stride, 
					x*stride:(x+1)*stride
				]
				best_patch, best_ssd = (0, 0), float('inf')
				for i in range(-patch_size // 2, patch_size // 2, 1):
					for j in range(-patch_size // 2, patch_size // 2):
						next_patch = next_frame[
							max(0, y*stride + i):y*stride + i + patch_size, 
							max(0, x*stride + j):x*stride + j + patch_size
						]
						if next_patch.shape != cur_patch.shape: continue
						error = ssd(cur_patch, next_patch)
						if error < best_ssd: 
							best_patch, best_ssd = (j, i), error
				flow[f, y, x] = np.array(best_patch)
				sys.stdout.flush()
	
	return flow

def naive_object_tracking(frames, locations, patch_size):
	num_frames, height, width, channels = frames.shape
	flow = np.zeros((num_frames, 2))

	cur_location = locations[0]
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