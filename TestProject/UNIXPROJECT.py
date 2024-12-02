from gpiozero import LED,MotionSensor
import numpy as np
import cv2
import time
import os
# Set up libraries and overall settings
import RPi.GPIO as GPIO  # Imports the standard Raspberry Pi GPIO library
from time import sleep   # Imports sleep (aka wait or pause) into the program
GPIO.setmode(GPIO.BOARD) # Sets the pin numbering system to use the physical layout

# Set up pin 27 for PWM
GPIO.setup(11,GPIO.OUT)  # Sets up pin 11 to an output (instead of an input)
p = GPIO.PWM(11, 50)     # Sets up pin 11 as a PWM pin
p.start(0)               # Starts running PWM on the pin and sets it to 0

video = cv2.VideoCapture(0)

#GPIO PIN 
sensorPin = 17
timeS = 0

# Load OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to compare two images using histogram comparison
def compare_images(image1, image2):
    # Convert images to grayscale
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    # Compute histograms
    hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
    
    # Normalize histograms
    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)
    
    # Compare histograms using correlation method
    correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    
    # Return a match score (1 is perfect match)
    return correlation

# Load the reference images from the folder
image_folder =  "C:\\Users\\zachr\\OneDrive\\Desktop\\TestProject\\Knownfaces" # Change to your folder path
reference_images = []
image_files = os.listdir(image_folder)

for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    if os.path.isfile(image_path):
        img = cv2.imread(image_path)
        reference_images.append((img, image_file))  # Store image with its filename

# Start capturing the live feed
cap = cv2.VideoCapture(0)  # 0 is usually the default webcam


# Initialize background subtractor for motion detection
fgbg = cv2.createBackgroundSubtractorMOG2()


#Determine if true or false for motion
sensor = MotionSensor(sensorPin)
currentstate = False
#currentstate = sensor.motion_detected

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw rectangles around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Crop the face from the frame
        face_roi = frame[y:y + h, x:x + w]

        # Compare the cropped face with reference images
        matched = False
        for reference_image, filename in reference_images:
            similarity = compare_images(face_roi, reference_image)
            if similarity > 0.8:  # You can adjust the threshold for a better match
                matched = True
                #cv2.putText(frame, f"Match!{filename}",(x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                if(filename == "zach1.jpg" or filename == "zach2.jpg" or filename == "zach3.jpg"):
                    cv2.putText(frame, f"Hi Zach!{filename}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    p.ChangeDutyCycle(3)     # Changes the pulse width to 3 (so moves the servo)
                    sleep(1)                 # Wait 1 second
                    p.ChangeDutyCycle(12)    # Changes the pulse width to 12 (so moves the servo)
                    sleep(1)
                    break  # Stop searching once a match is found
                if(filename == "asani2.jpg" or filename == "asani3.jpg" or filename == "asani4.jpg"):
                    cv2.putText(frame, f"Hi Asani!{filename}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    break  # Stop searching once a match is found
                if(filename == "makenna1.jpg" or filename == "makenna2.jpg" or filename == "makenna3.jpg" or filename == "makenna4.jpg" or filename == "makenna5.jpg" or filename == "makenna6.jpg"):
                    cv2.putText(frame, f"Hi MaKenna!{filename}",(x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    
                    break  # Stop searching once a match is found

        # If no match is found, display no match
        if not matched:
            cv2.putText(frame, "No Match Found", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Display the live feed
    cv2.imshow('Live Feed', frame)

    # Press 'q' to exit the live feed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()