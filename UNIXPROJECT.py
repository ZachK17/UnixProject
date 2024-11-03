from gpiozero import LED,MotionSensor
import numpy as np
import cv2
import time
video = cv2.VideoCapture(0)

#GPIO PIN 
sensorPin = 17
timeS = 0

#Determine if true or false for motion
sensor = MotionSensor(sensorPin)
currentstate = False

while True:
	currentstate = sensor.motion_detected
	#To capture frames
	ret, frame = video.read()
	if(currentstate == True):
		print("Motion")
		#Can change 100 for time and the timeS in if statement below
		while timeS != 100:
			ret, frame = video.read()
			#To display frames
			cv2.imshow("frame",frame)
			timeS += 1 	
			if cv2.waitKey(1) == ord('q') or timeS == 90:
				timeS = 0
				break

	else:
		print("No Motion Detected")

	time.sleep(0.40)

#To close camera down
video.release()
cv2.destroyAllWindows()
