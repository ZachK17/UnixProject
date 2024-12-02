import cv2

def detect_face_and_draw_rectangle(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Load the Haar Cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the image
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
image_path = "C:\\Users\\zachr\\OneDrive\\Desktop\\IMG_5412.jpg"
detect_face_and_draw_rectangle(image_path)