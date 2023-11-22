import cv2
import numpy as np
import os
haar_cascade = cv2.CascadeClassifier('haar_face.xml')
# people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield','Jimmy Falon' 'Madonna', 'Mindy Kaling']
# people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield','Jimmy Fallon', 'Madonna', 'Mindy Kaling']
people=[]
DIR = r'opencv-course\Resources\Faces\train'

for char in os.listdir(DIR):
    people.append(char[:])
print(people)
# face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

face_recognizer.read('face_trained.yml')

# Load the video capture object
cap = cv2.VideoCapture('Jen Law.mp4')
frame=[]
count=1
while True:
    count+=1
    # print(count)
    # Read a frame from the video
    ret, img = cap.read()

    if not ret:
        break  # Break the loop if the video has ended

    # Convert the frame to grayscale for face recognition
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces_rect = haar_cascade.detectMultiScale(gray_image, 2.1, 4, 4)

    for (x, y, w, h) in faces_rect:
        face_roi = gray_image[y:y + h, x:x + w]  # Extract the face region

        # Perform face recognition on the detected face
        label, confidence = face_recognizer.predict(face_roi)

        # Draw a rectangle around the face and display the name of the person
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, f'{people[label]} - {confidence}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        frame.append(img)
    # Display the resulting frame
    cv2.imshow('Face Recognition', img)

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# for f in frame:
#       cv2.imshow('Face Recognition', img)
#       cv2.waitKey(0)
#     # Break the loop if the 'q' key is pressed
#     #   if cv2.waitKey(1) & 0xFF == ord('q'):
#     #         break
# # Release the video capture object
cap.release()
cv2.destroyAllWindows()
