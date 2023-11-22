# Here we shall be training the  model on a couple of celebrity photos and then testing the results 
import cv2
import os 
import numpy as np
from tqdm import tqdm
# from cv2 import face_LBPHFaceRecognizer
# import face_recognition
haar_cascade=cv2.CascadeClassifier('haar_face.xml')

# people = ['Ben Afflek', 'Elton John', 'Jerry Seinfield','Jimmy Fallon', 'Madonna', 'Mindy Kaling']
people=[]
DIR = r'opencv-course\Resources\Faces\train'
for char in os.listdir(DIR):
    people.append(char[:])
features=[]
labels=[]
def face_crop(img):
    face_cropped=None
    face_rect=haar_cascade.detectMultiScale(img,1.1,minNeighbors=5)
    if(len(face_rect)!=0):
       
        
        if(len(face_rect)!=0):
            for (x,y,w,h) in face_rect:
                face_cropped=img[y:y+h,x:x+w]
            # face_cropped = img[face_rect[0][1]:face_rect[0][1]+face_rect[0][3],face_rect[0][0]:face_rect[0][0]+face_rect[0][2]]
            # # cv2.imshow("face",face_cropped)
            # cv2.waitKey(0)
                # face_cropped=np.array(face_cropped)
                # resized_img=cv2.resize(face_cropped,(180,180))
                return face_cropped
def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        # print("path: ", path)
        label = people.index(person)

        for img in tqdm(os.listdir(path)):
            
            img_path = os.path.join(path, img)

            img_array = cv2.imread(img_path, 0)
            if img_array is None:
                continue

            faces_roi = face_crop(img_array)
            if faces_roi is not None:  # Add this check
                features.append(faces_roi)
                labels.append(label)
# By adding the if faces_roi is not None: check, you'll skip the images where no face is detected, which should resolve the assertion error you were encountering during training.






create_train()
print('Training done ---------------')
features = np.array(features, dtype='object')
labels = np.array(labels)

face_recognizer = cv2.face.LBPHFaceRecognizer_create()




# Train the Recognizer on the features list and the labels list
epochs =20
for epoch in tqdm(range(epochs)):
    if len(features) > 0 and len(labels) > 0:
        try:
            face_recognizer.train(features, labels)
            # print('Training done ---------------')
        except Exception as e:
            print(f"ERROR:{e}")
    else:
        print('Error: Empty data for training.')

    # Print the current epoch number
    # print(f"Epoch {epoch+1}/{epochs} completed")

face_recognizer.save('face_trained.yml')
np.save('features.npy', features)
np.save('labels.npy', labels)



