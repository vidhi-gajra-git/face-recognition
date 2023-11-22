import cv2
img=cv2.imread('opencv-course/Resources/Photos/lady.jpg')
# img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow("actual",img)

haar_cascade=cv2.CascadeClassifier('haar_face.xml')
faces_rect=haar_cascade.detectMultiScale(img,scaleFactor=1.1,minNeighbors=3)    
for (x,y,w,h) in faces_rect:
    cv2.rectangle(img, (x,y),(x+w,y+h), (0,255,0),thickness=2)
cv2.imshow("face detected ", img)
cv2.waitKey(0)

# cv2.waitKey(0)
# DETECTING MORE NUMBER OF PEOPLE 
img=cv2.imread('opencv-course/Resources/Photos/group 2.jpg')
# img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow("actual",img)

haar_cascade=cv2.CascadeClassifier('haar_face.xml')
faces_rect=haar_cascade.detectMultiScale(img,scaleFactor=1.1,minNeighbors=3)    
for (x,y,w,h) in faces_rect:
    cv2.rectangle(img, (x,y),(x+w,y+h), (0,255,0),thickness=2)
cv2.imshow("face detected ", img)
cv2.waitKey(0)
cv2.destroyAllWindows()