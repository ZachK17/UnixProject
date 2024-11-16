from gpiozero import LED,MotionSensor
import numpy as np
import cv2
import time
# Set up libraries and overall settings
import RPi.GPIO as GPIO  # Imports the standard Raspberry Pi GPIO library
from time import sleep   # Imports sleep (aka wait or pause) into the program
GPIO.setmode(GPIO.BOARD) # Sets the pin numbering system to use the physical layout

# Set up pin 27 for PWM
GPIO.setup(27,GPIO.OUT)  # Sets up pin 11 to an output (instead of an input)
p = GPIO.PWM(11, 50)     # Sets up pin 11 as a PWM pin
p.start(0)               # Starts running PWM on the pin and sets it to 0

video = cv2.VideoCapture(0)

#GPIO PIN 
sensorPin = 17
timeS = 0

# Load the face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize video capture
cap = cv2.VideoCapture(0)

# Initialize background subtractor for motion detection
fgbg = cv2.createBackgroundSubtractorMOG2()


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
			# Motion detection
			fgmask = fgbg.apply(frame)
			
			# Find contours in the motion mask
			contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

			# Check for motion
			if contours:
				# Face detection
				gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
				faces = face_cascade.detectMultiScale(gray, 1.3, 5)

				# Draw rectangles around faces
				for (x, y, w, h) in faces:
					cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
					#if match :
					#cv2.putText(frame, 'MATCH' , (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
					#if it doesnt match
					#cv2.putText(frame, 'NO MATCH' , (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
			# Display the frame
			cv2.imshow('Frame', frame)
			
			timeS += 1 	
			if cv2.waitKey(1) == ord('q') or timeS == 90:
				timeS = 0
				break
			if cv2.waitKey(1) == ord('u'):
				# Move the servo back and forth
				p.ChangeDutyCycle(3)     # Changes the pulse width to 3 (so moves the servo)
				sleep(1)                 # Wait 1 second
				p.ChangeDutyCycle(12)    # Changes the pulse width to 12 (so moves the servo)
				sleep(1)
	else:
		print("No Motion Detected")

	time.sleep(0.40)

# Clean up everything
p.stop()                 # At the end of the program, stop the PWM
GPIO.cleanup()           # Resets the GPIO pins back to defaults

#To close camera down
video.release()
cv2.destroyAllWindows()
