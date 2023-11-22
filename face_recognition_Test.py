import cv2
import numpy as np
import os
haar_cascade=cv2.CascadeClassifier('haar_face.xml')
# people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield', 'Madonna', 'Mindy Kaling']
people=[]
DIR = r'opencv-course\Resources\Faces\train'
for char in os.listdir(DIR):
    people.append(char[:])
# features=np.load('features.npy',allow_pickle=True)
# labels=np.load('labels.npy',allow_pickle=True )
face_recognizer=cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')
img=cv2.imread(r'C:\Users\ashok\Desktop\resources\Programming stuff\image processing\opencv-course\Resources\Faces\train\Tom Cruise\002_6749a2c4.jpg')
gray_image=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# cv2.imshow("Original image", img)
# Detecting the face in the image 
faces_rect=haar_cascade.detectMultiScale(gray_image,2.1,1,4)
for (x,y,w,h) in faces_rect:
        face_roi=gray_image[y:y+h,x:x+w]
        label,confidence=face_recognizer.predict(face_roi)
        cv2.putText(img,str(people[label]),(20,20),cv2.FONT_HERSHEY_COMPLEX,1.1 ,(0,255,255))
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.imshow(f" {people[label]} with {confidence}", img)
        
    
cv2.waitKey(0)
cv2.destroyAllWindows()      
        

