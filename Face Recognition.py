#!/usr/bin/env python
# coding: utf-8

# ### Detect livenes

# In[9]:


## return true for real and false for fake

import cv2
import dlib
import numpy as np

# Load the face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Function to detect facial landmarks
def get_landmarks(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    if len(rects) == 0:
        return None

    shape = predictor(gray, rects[0])
    landmarks = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
    return landmarks

# Function to calculate eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    # Compute the euclidean distances between the two sets of vertical eye landmarks
    A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
    B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))

    # Compute the euclidean distance between the horizontal eye landmarks
    C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))

    # Compute the eye aspect ratio
    ear = (A + B) / (2 * C)
    return ear

# Function to draw a frame around the detected face
def draw_frame(image, landmarks):
    overlay = image.copy()
    for point in landmarks:
        cv2.circle(overlay, point, 2, (0, 255, 0), -1)

    alpha = 0.5
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

# Main loop for liveness detection
def detect_liveness():
    cap = cv2.VideoCapture(0)
    blink_counter = 0
    is_blinking = False
    liveness = False
    start_time = cv2.getTickCount()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        landmarks = get_landmarks(frame)
        if landmarks is None:
            continue

        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)

        avg_ear = (left_ear + right_ear) / 2

        if avg_ear < 0.2:
            if not is_blinking:
                blink_counter += 1
                is_blinking = True
        else:
            is_blinking = False
        
        draw_frame(frame, landmarks)

        if blink_counter >= 1 or cv2.getTickCount() - start_time >= 15 * cv2.getTickFrequency():
            liveness = True
            cv2.putText(frame, "Liveness: Real", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Face Liveness Detection", frame)
            break
        else:
            liveness = False
            cv2.putText(frame, "Liveness: Fake", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Face Liveness Detection", frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Liveness detected :" ,liveness)
    return liveness

# Run the liveness detection
#liveness_status = detect_liveness()
#print("Liveness Status:", liveness_status)


# ### Face Recognition

# In[12]:


import cv2
import pymongo
import face_recognition

# Connect to MongoDB
client = pymongo.MongoClient("mongodb://localhost:27017")
db = client["face_recognition"]
collection = db["face recognition"]

# Function to compare face encodings
def compare_face_encodings(face_encodings, face_encodings_list):
    results = face_recognition.compare_faces(face_encodings_list, face_encodings)
    if True in results:
        index = results.index(True)
        person = collection.find_one({"id": index + 1})
        return person["name"]
    else:
        return None

# Function to capture an image using a webcam
def capture_image():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        return None
    cap.release()
    return frame

# Function to extract face encodings from an image
def extract_face_encodings(image):
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)
    return face_encodings

# Main program
def main():
    if detect_liveness():
        # Capture an image using the webcam
        image = capture_image()

        # Check if the image was captured successfully
        if image is None:
            return

        # Extract face encodings from the captured image
        face_encodings = extract_face_encodings(image)

        # Check if any face encodings were extracted
        if len(face_encodings) == 0:
            print("No face detected in the captured image")
            return

        # Get the face encodings list from the database
        cursor = collection.find()
        face_encodings_list = [person["encodings"] for person in cursor]

        # Compare the face encodings with the database
        name = compare_face_encodings(face_encodings[0], face_encodings_list)

        if name is not None:
            print("Person is verified.")
            print("Name of person is : ", name)
        else:
            print("Person is not verified. Kindly register yourself")

if __name__ == '__main__':
    main()

