import cv2
import os
import numpy as np

# Function to detect faces in an image
def detect_faces(image):
    # Load pre-trained Haar Cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Convert the image to grayscale for face detection
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    return faces, gray_image

# Function to compare two faces using ORB feature matching
def compare_faces(face1, face2):
    # Initialize ORB detector
    orb = cv2.ORB_create()
    
    # Compute keypoints and descriptors for the two faces
    kp1, des1 = orb.detectAndCompute(face1, None)
    kp2, des2 = orb.detectAndCompute(face2, None)
    
    if des1 is not None and des2 is not None:
        # Match the descriptors using Brute-Force Matcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        
        # Calculate the total distance (sum of match distances)
        total_distance = sum([match.distance for match in matches])
        
        # Set a threshold for matching
        threshold = 200  # You can tweak this threshold for your use case
        if total_distance < threshold:
            return True
    return False

# Function to compare an image with faces in a folder
def compare_faces_in_folder(image_path, folder_path):
    # Load the input image
    input_image = cv2.imread(image_path)
    if input_image is None:
        print(f"Could not load the input image from {image_path}")
        return
    
    # Detect faces in the input image
    input_faces, input_gray_image = detect_faces(input_image)
    
    if len(input_faces) == 0:
        print("No faces detected in the input image.")
        return

    # Loop over all images in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # Load each image in the folder
        folder_image = cv2.imread(file_path)
        if folder_image is None:
            continue
        
        # Detect faces in the folder image
        folder_faces, folder_gray_image = detect_faces(folder_image)
        
        if len(folder_faces) == 0:
            continue
        
        # Compare each face in the input image with each face in the folder image
        for (x, y, w, h) in input_faces:
            input_face = input_gray_image[y:y+h, x:x+w]
            
            for (fx, fy, fw, fh) in folder_faces:
                folder_face = folder_gray_image[fy:fy+fh, fx:fx+fw]
                
                # Compare the faces using ORB feature matching
                if compare_faces(input_face, folder_face):
                    print(f"Match found with face in {filename}")
                    if(filename == "elvis2.jpg" or filename == "elvis3.jpg" or filename == "elvis4.jpg" or filename == "elvis5.jpg" or filename == "elvis6.jpg"):
                        print("Welcome elvis")
                    if(filename == "face1.JPG" or filename == "face2.JPG" or filename == "face3.JPG" or filename == "face4.JPG"):
                        print("Welcome joel")
                    return
                else:
                    print(f"No match with face in {filename}")

    print("This is not a match with any face in the folder.")


image_path = "C:\\Users\\zachr\\OneDrive\\Desktop\\asani.jpg"  # Path to the image you want to compare
folder_path =  "C:\\Users\\zachr\\OneDrive\\Desktop\\Knownfaces" # Folder containing other images to compare with


compare_faces_in_folder(image_path, folder_path)


#image_path = "C:\\Users\\zachr\\OneDrive\\Desktop\\elvis1.jpg"  # Path to the image you want to compare
#folder_path =  "C:\\Users\\zachr\\OneDrive\\Desktop\\Knownfaces" # Folder containing other images to compare with

#image_path = "C:\\Users\\zachr\\OneDrive\\Desktop\\Joel\\joel.jpg"  # Path to the image you want to compare
#folder_path = "C:\\Users\\zachr\\OneDrive\\Desktop\\Joel2"  # Folder containing other images to compare with
