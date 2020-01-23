
# USAGE
# python main.py 

################ Import Packages ################
from webcamvideostream import WebcamVideoStream
from fps import FPS
from processframe import FindObject,BoundingBox,Draw,Coordinates3d
import numpy.linalg as la
import argparse
import imutils
import math
import cv2 as cv
import numpy as np
import datetime


################ Basic Variables initialisation ################
font = cv.FONT_HERSHEY_SIMPLEX

screen_ratio = 3/4

# Minimum area of an object to be recognized (to avoid noise)
min_area = 20

# Range of green color in HSV
lower = np.array([40,100,30])
upper = np.array([100,255,255])

video_file1 = 'videos/example_05_c2.avi'
video_file2 = 'videos/example_05.avi'
# video_file1 = None
# video_file2 = None

pause = False 

frame_w = 600
frame_h = int(frame_w * screen_ratio)



################ Camera initialisation ################
# Resolution
res_x1 = 640
res_y1 = 480
res_x2 = 640
res_y2 = 480

# Angle of View
aov_x1 = 67
aov_y1 = 48
aov_x2 = 67
aov_y2 = 48

# Position in 3D of the Camera's
c1_x = 0
c1_y = 0
c1_z = 0
c2_x = c1_x + 4
c2_y = 0
c2_z = 0



################ START VIDEOSTREAM ################
print("[INFO] sampling frames from webcam...")

# Check if we have a video or a webcam
if (video_file1 is not None) and (video_file2 is not None):
	stream1 = cv.VideoCapture(video_file1)
	stream2 = cv.VideoCapture(video_file2)
	fps1 = stream1.get(cv.CAP_PROP_FPS)
	fps2 = stream2.get(cv.CAP_PROP_FPS)
# otherwise, we are reading from webcams
else:
	stream1 = WebcamVideoStream(src=0).start()
	stream2 = WebcamVideoStream(src=0).start()
# Start framecount
streamfps = FPS().start()
streamfps = FPS().start()

# Storage of 2d coordinates of each camera
list2dX_1 = []
list2dY_1 = []
list2dX_2 = []
list2dY_2 = []

# Storage of calculated 3d coordinates
list3dX = []
list3dY = []
list3dZ = []

average_x = 0
average_y = 0

######################## START CAPTURING ########################
# loop over every frame
while(True):
	key = cv.waitKey(20) & 0xFF
	#if key== ord("c"): crop = True # Crop only to the region of interest
	if key == ord("p"): P = np.diag([100,100,100,100])**2 # Make the filter less uncertain
	if key == ord("q") or key == 27: break # quitting when ESCAPE or q is pressed
	if key == ord(" "): pause =not pause # pause when spacebar is pressed, unpause when pressed again
	if(pause): continue



	################ GRAB FRAME AND PROCESS ################
	# grab the frame from the stream and resize it
	(grabbed1, frame1) = stream1.read()
	(grabbed2, frame2) = stream2.read()
	# Check if the frames have been grabbed
	if grabbed1 is False or grabbed2 is False:
		# Video has ended, but we still want to watch the end result
		continue
	# Check if the frames are not None or empty
	if not(isinstance(frame1, np.ndarray)) or not(isinstance(frame2, np.ndarray)):
		print("[INFO] Oops, something went wrong with the frames")
		break
	frame1 = imutils.resize(frame1, width=frame_w)
	frame2 = imutils.resize(frame2, width=frame_w)
	graph3d = np.zeros((frame_h,frame_w,3), np.uint8)



	################ FIND GREEN PIXELS ################
	# Convert BGR to HSV
	hsv1 = cv.cvtColor(frame1, cv.COLOR_BGR2HSV)
	hsv2 = cv.cvtColor(frame2, cv.COLOR_BGR2HSV)
    # Threshold the HSV image to get only green colors
	mask1 = cv.inRange(hsv1, lower, upper)
	mask2 = cv.inRange(hsv2, lower, upper)



	################ Find object ################
	# get contours, find biggest contour (which is the ball), noise filter
	contour1_ball = FindObject(mask1, lower,upper, min_area)
	contour2_ball = FindObject(mask2, lower,upper, min_area)

	if contour1_ball is not None or contour2_ball is not None: # Check if there is any green object 
		(x1, y1, w1, h1) = BoundingBox(contour1_ball)
		(x2, y2, w2, h2) = BoundingBox(contour2_ball)

		list2dX_1.append(x1)
		list2dY_1.append(y1)
		list2dX_2.append(x2)
		list2dY_2.append(y2)

		if len(list3dX)!=0:
			average_x = sum(list3dX) / len(list3dX)
			average_y = sum(list3dY) / len(list3dY)

		############ Calculate 3D position ############
		xo,yo,zo = Coordinates3d(x1, y1, x2, y2, 	# Position of ball camera 1 and 2
					res_x1,res_y1,res_x2,res_y2, 	# Resolution of screen camera 1 and 2
					aov_x1,aov_y1,aov_x2,aov_y2, 	# Field of View camera 1 and 2
					c1_x,c1_y,c1_z,c2_x,c2_y,c2_z,	# Camera Position camera 1 and 2
					average_x, average_y)

		if xo is not None:
			list3dX.append(xo)
			list3dY.append(yo)
			list3dZ.append(zo)
	


	################ DRAW ################
	if list2dX_1: # If there is any measurement, draw measurements and coordinates
		frame1 = Draw(frame1,x1,y1,w1,h1)
		frame2 = Draw(frame2,x2,y2,w2,h2)
		
		
		
		# Draw every measurement so far on the frames
		for n in range(len(list2dX_1)): 
			cv.circle(frame1,(int(list2dX_1[n]),int(list2dY_1[n])),3, (0, 255, 0),-1)
		for n in range(len(list2dX_2)): 
			cv.circle(frame2,(int(list2dX_2[n]),int(list2dY_2[n])),3, (0, 255, 0),-1)
	
		# Make visual representation of 3d position on a new frame
		for n in range(len(list3dX)): 
			cv.circle(graph3d,(list2dX_1[n], int(4* list3dX[n]+frame_h/2+90)),3, (0, 220, 0),-1)
			cv.circle(graph3d,(list2dX_1[n], int(4*-list3dY[n]+frame_h/2+90)),3, (220, 50, 50),-1)
			cv.circle(graph3d,(list2dX_1[n],-int(4* list3dZ[n]-frame_h/2-90)),3, (00, 0, 220),-1)
	
	# Horizontal axis ( at zero )
	cv.line(graph3d, (0,int(frame_w/2)), (frame_w,int(frame_w/2)), (220, 220, 220), thickness=1, lineType=8, shift=0)

	# Show which color is which dimension
	cv.putText(graph3d, format("X"), (10, 20), font, 0.5, (0, 255, 0), 2)
	cv.putText(graph3d, format("Y"), (30, 20), font, 0.5, (255, 0, 0), 2)
	cv.putText(graph3d, format("Z"), (50, 20), font, 0.5, (0, 0, 255), 2)



	########## DISPLAY ##########
	cv.imshow("Frame 1", frame1)
	cv.imshow("Frame 2", frame2)
	cv.imshow("Position", graph3d)
	if not(grabbed1 is False or grabbed2 is False):
		# update the FPS counter
		streamfps.update()
		fps = streamfps.fps_now()


################### CLEARING UP ###################
# stop the timer and display FPS information
streamfps.stop()
print("[INFO] elasped time: {:.2f}".format(streamfps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps))
print(list3dZ)
# do a bit of cleanup
stream1.release()
stream2.release()
cv.destroyAllWindows()