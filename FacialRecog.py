import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
webcam = cv2.VideoCapture(0)
while True:
    try:
        ret,frame = webcam.read()
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces=face_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)
        for(x,y,w,h) in faces:
            photo=frame[y:y+h,x:x+w]
            #I think we put the line here
            width=x+w
            height=y+h
            frame=photo
            #cv2.rectangle(frame,(x,y),(width,height),(255,0,0),2)
        cv2.waitKey(0)
        cv2.imshow("Capturing", frame)
        key = cv2.waitKey(1)
    except(KeyboardInterrupt):
        webcam.release()
        cv2.destroyAllWindows()
        break
webcam.release()
cv2.destroyAllWindows()