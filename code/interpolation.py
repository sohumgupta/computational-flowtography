import numpy as np
import glob
import argparse
import os
import sys
from scipy.ndimage import map_coordinates
from scipy.ndimage.morphology import binary_dilation
import cv2

  # Parse arguments
arg_parser = argparse.ArgumentParser(description="Frame Interpolation")
arg_parser.add_argument("-i", "--in", type=str, default=os.path.join("..", "frame_int_data"),
                        help='''Input data directory''')
arg_parser.add_argument("-o", "--out", type=str, default="./frame_int_results",
                        help="Directory in which to save output images (default: ./frame_int_results)")
arg_parser.add_argument("-m", "--media", type=str, default="video",
                        help="Directory in which to save output images (default: ./frame_int_results)")

args = vars(arg_parser.parse_args())
media = args["media"]

# paths
input_dir = os.path.join("..", "frame_int_data", args["in"])
output_vid = "../frame_int_results/%s/result.avi" % (args["in"])
og_output_vid = "../frame_int_results/%s/og.avi" % (args["in"])
output_dir = "../frame_int_results/%s" % args["in"]

# prepare video or image capture
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)
if media == "video":
    cap = cv2.VideoCapture(input_dir)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"input FPS: {fps}")
    ret, old_frame = cap.read()
elif media == "images":
    path_name = "frame"
    fps = 60
    filenames = glob.glob(f"{input_dir}/*.png")
    filenames.sort()
    images = [cv2.imread(img) for img in filenames]
    old_frame = images[0]
    n_images = len(images)

# video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_vid ,fourcc, int(fps)*1.5,( old_frame.shape[1], old_frame.shape[0]))
img_array = []
og_img_array = []
img_array.append(old_frame)
og_img_array.append(old_frame)

img_count = 1
frame_count = 1
while True:
    print(f"frame count: {frame_count}") 
    frame_count += 1
    if media == "video":
        ret, frame = cap.read()
        if ret == False:
            break
    elif media == "images":
        frame = images[img_count]
        img_count+=1
        if img_count == n_images:
            break


    # prepare images for flow
    I0 = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    I1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # calculate dense flow
    u0 = cv2.calcOpticalFlowFarneback(I0, I1, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # 1. forward warp flow u_0 to time t to give u_t
    #    where u_1(round(x+t*u_0(x))) = u_0(x)
    ut = np.empty(u0.shape)
    ut[:] = np.NaN
    # ut = np.zeros(u0.shape)

    t = 0.5
    height = u0.shape[0]
    width = u0.shape[1]

    x, y = np.meshgrid(range(width), range(height))
    xt = np.round(x + t*u0[...,0])
    yt = np.round(y + t*u0[...,1])
    for j in range(height):
        for i in range(width):
            new_j = int(yt[j,i])
            new_i = int(xt[j,i])
            if new_i >=0 and new_i<width and new_j>=0 and new_j<height:
                ut[new_j, new_i] = u0[j, i]

    # 2. Fill any holes in u_t using outside-in filling strategy
    h_low = 0
    h_high = width-1
    v_low = 0
    v_high = height-1
    fill = np.empty((2))
    fill[:] = np.NaN
    
    while True:

        if h_low >= h_high or v_low >= v_high :
            break
        for i in range(h_low, h_high, 1):
            if np.isnan(ut[v_low, i]).any(): ut[v_low, i] = fill
            else: fill =  ut[v_low, i]
        for i in range(v_low, v_high, 1):
            if np.isnan(ut[i, h_high]).any(): 
                ut[i, h_high] = fill
            else: fill =  ut[i, h_high]
        for i in range( h_high, h_low, -1):
            if np.isnan(ut[v_high, i]).any(): 
                ut[v_high, i] = fill
            else: fill =  ut[v_high, i]
        for i in range(v_high, v_low, -1):
            if np.isnan(ut[i, h_low]).any(): ut[i, h_low] = fill
            else: fill =  ut[i, h_low]
        h_low += 1
        h_high -= 1
        v_low += 1
        v_high -= 1

    # 3. Estimate occlusion masks 
    # o1[x] = 1 for pixels that is not targeted after splatting
    # forward warp I0 to t=1
    u1 = np.zeros(u0.shape)
    # u1[:] = np.NaN
    # map warped points
    xt1 = np.round(x + u0[...,0])
    yt1 = np.round(y + u0[...,1])
    o0 = np.zeros(xt1.shape, dtype=np.uint8)
    o1 = np.zeros(yt1.shape, dtype=np.uint8)

    for j in range(height):
        for i in range(width): # TODO: Add ordering independence?
            # for each pixel, retrieve warping
            jt1 = int(yt1[j,i])
            it1 = int(xt1[j,i])
            # calculate u1 only if not out of image region
            if (it1 >=0 and it1<width and jt1>=0 and jt1<height):
                u1[jt1, it1] = u0[j, i]
            # else mark o1 = 1
            else: # doesn't make sense to exclude by x and by y seperately?
                o1[j,i] = 1
    
    # calculate o0
    # o0[x] = 1 if    |u0[x] - u1(x + u0[x])| > 0.5
    for j in range(height):
        for i in range(width):
            u0x = u0[j, i, 0] 
            u0y = u0[j, i, 1] 
            diff_x = abs(u0x - u1[j, i, 0])
            diff_y = abs(u0y - u1[j, i, 1])

            if diff_x > 0.5  and diff_y > 0.5: 
                o0[j, i] = 1         
    # dilate occlusion masks once.
    # o0 = binary_dilation(o0).astype(o0.dtype)
    # o1 = binary_dilation(o1).astype(o1.dtype)

    # 4. Compute the colors of the interpolated pixels with consideration on occlusion
    output = np.zeros(frame.shape, dtype=np.uint8)

    x0 = x - t* ut[..., 0]
    y0 = y - t* ut[..., 1]
    x1 = x - (1-t)* ut[..., 0]
    y1 = y - (1-t)* ut[..., 1]
    x0 = np.clip(x0, 0, width-1)
    x1 = np.clip(x1, 0, width-1)
    y0 = np.clip(y0, 0, height-1)
    y1 = np.clip(y1, 0, height-1)
    
    # where 1, use the image pixel    
    I0_Ois1 = cv2.bitwise_and(old_frame, old_frame, mask= o0)
    I1_Ois1 = cv2.bitwise_and(frame, frame, mask=o1)

    # where both 0, interpolate
    x0_Ois0 = cv2.bitwise_and(x0, x0, mask=(1-o0))
    y0_Ois0 = cv2.bitwise_and(y0, y0, mask=(1-o0))
    x1_Ois0 = cv2.bitwise_and(x1, x1, mask=(1-o1))
    y1_Ois0 = cv2.bitwise_and(y1, y1, mask=(1-o1))

    # bilinear interpolation with occlusion consideration
    for c in range(frame.shape[2]):
        interpolation = (1-t)*map_coordinates(old_frame[...,c], (y0, x0)) +  t*map_coordinates(frame[..., c], (y1, x1))
        interpolation = cv2.bitwise_and(interpolation, interpolation, mask=(1-o0))
        output[..., c] = cv2.bitwise_and(interpolation, interpolation, mask=(1-o1)) + I0_Ois1[...,c]+ I1_Ois1[...,c] 
    old_frame = frame.copy()
    img_array.append(output)
    img_array.append(old_frame)
    og_img_array.append(old_frame)

# save image and create video
for i in range(len(img_array)):
    out.write(img_array[i])
    output_path = "%s/res_img%02d.jpg" % (output_dir, i+1)
    cv2.imwrite(output_path, img_array[i])
out.release()

if media == "video": cap.release()

# save original video 
og_out = cv2.VideoWriter(og_output_vid, fourcc, fps,( old_frame.shape[1], old_frame.shape[0]))
for i in range(len(og_img_array)):
    og_out.write(og_img_array[i])
og_out.release()

