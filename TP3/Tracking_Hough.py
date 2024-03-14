import numpy as np
import cv2
import math
from collections import defaultdict

roi_defined = False
channel = 2
thr = 30

def nothing(x):
    pass

def grad_orientation(image):

    gradx, grady = np.gradient(image)

    # Gradient orientation ( Index of the vote )
    orientation = np.arctan2(grady, gradx)

    return orientation

def grad_norme(image):

    gradx, grady = np.gradient(image)

    # Gradient norm
    norme = np.sqrt(gradx**2 + grady**2)

    return norme

def grad_threshold(image, seuil):
    norme = grad_norme(image)
    orientation = grad_orientation(image)
    index = np.where(norme < seuil)
    norme[index] = 0

    return norme, orientation

def R_table(image, centro, seuil):

    norme, orientation = grad_threshold(image, seuil)
    orientation = orientation*90//math.pi

    r_table = defaultdict(list) # Create a empty dictionary

    for (i, j), value in np.ndenumerate(norme):
        if value:
            # Associate relative position with gradient orientation in the R-table
            # key = orientation, value = relative position
            r_table[orientation[i, j]].append((centro[0] - j,centro[1] - i))
        
    return r_table

def hough_transform(r_table, norme, orientation):
    # Create a empty matrix
    vote_map = np.zeros_like(orientation)
    orientation = orientation*90//math.pi

    for (i, j), value in np.ndenumerate(norme):
        if value and orientation[i, j] in r_table: 
            for pos in r_table[orientation[i, j]]:
                if 0 < i + pos[1] < orientation.shape[0] and 0 < j + pos[0] < orientation.shape[1]:
                    # Increment the accumulator
                    vote_map[int(i + pos[1]), int(j + pos[0])] += 1

    return vote_map 

def define_ROI(event, x, y, flags, param):
    global r,c,w,h,roi_defined
    # if the left mouse button was clicked, 
    # record the starting ROI coordinates 
    if event == cv2.EVENT_LBUTTONDOWN:
        r, c = x, y
        roi_defined = False
    # if the left mouse button was released,
    # record the ROI coordinates and dimensions
    elif event == cv2.EVENT_LBUTTONUP:
        r2, c2 = x, y
        h = abs(r2-r)
        w = abs(c2-c)
        r = min(r,r2)
        c = min(c,c2)  
        roi_defined = True

#cap = cv2.VideoCapture('Test-Videos/VOT-Ball.mp4')
cap = cv2.VideoCapture('Test-Videos/Antoine_Mug.mp4')
#cap = cv2.VideoCapture('Test-Videos/VOT-Woman.mp4')
#cap = cv2.VideoCapture('Test-Videos/VOT-Sunshade.mp4')


# take first frame of the video
ret,frame = cap.read()
# load the image, clone it, and setup the mouse callback function
clone = frame.copy()
cv2.namedWindow("First image")
cv2.setMouseCallback("First image", define_ROI)
 
# keep looping until the 'q' key is pressed
while True:
    # display the image and wait for a keypress
    cv2.imshow("First image", frame)
    key = cv2.waitKey(1) & 0xFF
    # if the ROI is defined, draw it!
    if (roi_defined):
        # draw a green rectangle around the region of interest
        cv2.rectangle(frame, (r,c), (r+h,c+w), (0, 255, 0), 2)
    # else reset the image...
    else:
        frame = clone.copy()
    # if the 'q' key is pressed, break from the loop
    if key == ord("q"):
        break

track_window = (r,c,h,w)
# set up the ROI for tracking
roi = frame[c:c+w, r:r+h]

hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
h_roi = hsv_roi[:,:,channel]

roi_center = np.array([int(h//2), int(w//2)])
r_table = R_table(h_roi, roi_center, thr)


cpt = 1
while(1):
    ret ,frame = cap.read()
    if ret == True:
        
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h_frame = hsv_frame[:,:,channel]

        norme, orientation = grad_threshold(h_frame, thr)

        ## Display selection of the voting pixels

        #orientation_col = cv2.cvtColor(orientation, cv2.COLOR_GRAY2BGR)
        
        #orientation_col[index[0],index[1],0] = 0
        #orientation_col[index[0],index[1],1] = 0
        #orientation_col[index[0],index[1],2] = 255
        
        #cv2.imshow('Gradient orientation', orientation)
        #cv2.imshow('Selected orientations', orientation_col)

        vote_map = hough_transform(r_table, norme, orientation)

        vote_map = np.uint8(vote_map)
        vote_map = cv2.normalize(vote_map, vote_map, 0, 255, cv2.NORM_MINMAX)

        Mpx, Mpy = np.unravel_index(vote_map.argmax(), vote_map.shape)
        r = Mpx - (h // 2)
        c = Mpy - (w // 2)

        frame_tracked = cv2.rectangle(frame, (r, c), (r + h, c + w), (255, 0, 0), 2)

        #Plotting all images
        cv2.imshow('Gradient norm', norme)  
        cv2.imshow('Sequence', frame_tracked)
        cv2.imshow('Vote map', vote_map)

        k = cv2.waitKey(60) & 0xff
        if k == ord('a'):
            break
        elif k == ord('s'):
            cv2.imwrite('Frame_%04d.png'%cpt,frame_tracked)
        cpt += 1
    else:
        break

cv2.destroyAllWindows()
cap.release()