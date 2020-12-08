import numpy as np
import cv2

cap = cv2.VideoCapture('/Users/Maceo/Desktop/CompPhoto/computational-flowtography/data/Run.mp4')

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 1,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0,255,(100,3))

# Take first frame
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# Parameters for Stabilization
crop_factor = 2
window_X = old_frame.shape[1] // (2 * crop_factor)
window_Y = old_frame.shape[0] // (2 * crop_factor)

# Make mask to find object
mask_points = (950,475) # Will make this easier to change eventually
mask = np.zeros_like(old_gray)
mask[mask_points[1]-50:mask_points[1]+50, mask_points[0] - 50:mask_points[0] + 50] = 255

# Find the point we want to track
p0 = cv2.goodFeaturesToTrack(old_gray, mask = mask, **feature_params)

# Create Video Writer
out = cv2.VideoWriter('../results/running.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, 
    (window_X * 2,window_Y * 2))


while(1):
    ret,frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    good_new = p1[st==1]
    #good_old = p0[st==1]

    # draw the tracks
    # for i,(new,old) in enumerate(zip(good_new,good_old)):
    #     a,b = new.ravel()
    #     c,d = old.ravel()
    #     mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
    #     frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
    
    # Temporarily hardcoded for single point, can expand for "scaling" later.
    new_x = int(good_new[0][0])
    new_y = int(good_new[0][1])

    # Crop & center around single point
    centered = frame[new_y - window_Y:new_y+window_Y, new_x - window_X:new_x + window_X]
    
    out.write(centered)
    cv2.imshow('frame',centered)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)

cv2.destroyAllWindows()
cap.release()
out.release()