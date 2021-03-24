#1. Implement Face Detection and Eyes Detection using HaarCascade Classifier in a Video Frame.

import numpy as np
import cv2

# Loading the required XML Classfiers

face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')

# For Videos from live camera feeds

# Capturing a live video 

cap = cv2.VideoCapture("http://192.168.43.1:4747/video")

width  = int(cap.get(3)) #Getting width of the captured frame in int
height = int(cap.get(4)) #Getting height of the captured frame in int
fps = int(cap.get(5)) #Getting frame rate of the captured frame rate in int

fourcc = cv2.VideoWriter_fourcc(*'XVID') #Specifying FourCC code 
    
#Creating a videowriter object to write the video in a given output file and specifying  its frame rate, width and height
out = cv2.VideoWriter('Output/output_live_2.mp4', fourcc, fps, (width, height))

while True:
    
    #Read frames from Camera
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.6, 5)

    for (x,y,w,h) in faces:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,255),2)
        
    # Display image in a window
    cv2.imshow('img', img) 
    
    # Writing image to an output file
    out.write(img)
    
    #Stop if Esc key is pressed
    k = cv2.waitKey(30) & 0xff    
    if k==27:
        break

# Release the video capture and video write objects            
cap.release()
out.release()

# Closes all the frames
cv2.destroyAllWindows()