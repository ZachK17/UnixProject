import cv2
import numpy as np
import os
from gpiozero import MotionSensor, AngularServo
import time
import threading  # Import threading to handle servo movement timing independently
from rpi_lcd import LCD  # Importing LCD from rpi_lcd

    #Servo Motor 
myGPIO=27
SERVO_DELAY_SEC = 0.001 
myCorrection=0.0
maxPW=(2.5+myCorrection)/1000
minPW=(0.5-myCorrection)/1000
servo =  AngularServo(myGPIO,initial_angle=0,min_angle=0, max_angle=180,min_pulse_width=minPW,max_pulse_width=maxPW) 
def loop():
    # GPIO PIN for Motion Sensor
    sensorPin = 17
    sensor = MotionSensor(sensorPin)
    
    #0 Initialize the LCD
    lcd = LCD()

    # Clear the LCD screen
    lcd.clear()

    # Load OpenCV face detector
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


    # Load the reference images from subfolders within the folder
    image_folder = "/home/bert459/Downloads/UnixProject-main/TestProject/Knownfaces"  # Change to your folder path
    reference_images = []

    # Walk through the folder and subfolders to read all images
    for root, dirs, files in os.walk(image_folder):
        for image_file in files:
            if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                image_path = os.path.join(root, image_file)
                img = cv2.imread(image_path)
                folder_name = os.path.basename(root)  # Get the folder name where the image is located
                reference_images.append((img, folder_name))  # Store image and folder name

    # Start capturing the live feed
    cap = cv2.VideoCapture(0)  # 0 is usually the default webcam

    # Variable to keep track of the camera feed state and the time limit
    camera_active = False
    start_time = None

    # Function to handle the servo movement to 180 degrees and back to 0
    def move_servo_to_180():
        # Move to 180 degrees
        for angle in range(0, 181, 1):   # make servo rotate from 0 to 180 deg
            servo.angle = angle
            time.sleep(SERVO_DELAY_SEC)
        time.sleep(5)
        for angle in range(180, -1, -1): # make servo rotate from 180 to 0 deg
            servo.angle = angle
            time.sleep(SERVO_DELAY_SEC)

    # Function to display folder name on LCD for 5 seconds
    def display_name_on_lcd(name):
        lcd.clear()  # Clear the display
        lcd.text(f"Welcome {name}!", 1)  # Display folder name on the first line of the LCD
        time.sleep(5)  # Wait for 5 seconds
        lcd.clear()  # Clear the display after 5 seconds

    # Function to compare two images using histogram comparison
    def compare_images(image1, image2):
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        
        hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
        
        cv2.normalize(hist1, hist1)
        cv2.normalize(hist2, hist2)
        
        correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        return correlation
        
    while True:
        lcd.clear()  # Clear the display after 5 seconds
        # Wait for motion to be detected
        sensor.wait_for_motion()

        # When motion is detected, start the camera feed for 20 seconds
        camera_active = True
        start_time = time.time()

        while camera_active:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the frame
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            # Draw rectangles around faces and compare with reference images
            matched = False  # Start with the assumption that no match is found
            highest_similarity = 0  # To track the highest similarity
            best_folder_name = ""  # To store the folder name with highest similarity

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # Crop the face from the frame
                face_roi = frame[y:y + h, x:x + w]

                # Compare the cropped face with reference images
                for reference_image, folder_name in reference_images:
                    similarity = compare_images(face_roi, reference_image)
                    if similarity > highest_similarity:  # Track the highest similarity
                        highest_similarity = similarity
                        best_folder_name = folder_name

                # If the highest similarity exceeds a threshold, perform actions
                if highest_similarity > 0.423:  # You can adjust the threshold for a better match

                    if cv2.waitKey(1) & 0xFF == ord('d'):
                        # Start a separate thread to move the servo and display on LCD
                        threading.Thread(target=move_servo_to_180).start()
                        threading.Thread(target=display_name_on_lcd, args=(best_folder_name,)).start()
                    
                    matched = True

                    # Display the folder name (main folder) and similarity percentage on the frame
                    similarity_text = f"Similarity: {highest_similarity * 100:.2f}%"
                    cv2.putText(frame, f"Welcome {best_folder_name}!", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    cv2.putText(frame, similarity_text, (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    break

            # If no match is found, display the highest similarity found
            if not matched:
                similarity_text = f"No match found. Similarity: {highest_similarity * 100:.2f}%"
                cv2.putText(frame, similarity_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                

            # Display the live feed
            cv2.imshow('Live Feed', frame)

            # Check if 20 seconds have passed or if 'q' is pressed to exit
            if time.time() - start_time > 20:
                camera_active = False  # Stop the camera feed after 20 seconds

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Wait for motion again to trigger the camera
        sensor.wait_for_motion()

    # Clear the LCD screen
    lcd.clear()

    # Release the capture and close any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':     # Program entrance
    print ('Program is starting...')
    try:
        loop()
    except KeyboardInterrupt:  # Press ctrl-c to end the program.
        print("Ending program")
