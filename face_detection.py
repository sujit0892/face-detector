import cv2
import numpy as np

cascade = cv2.CascadeClassifier('cascade_files/haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('cascade_files/haarcascade_eye.xml')

if cascade.empty():
    raise IOError('Unable to load cascade')

cap = cv2.VideoCapture(0)

while True:
      ret, frame = cap.read()
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      face_rects = cascade.detectMultiScale(gray, 1.3, 5)
      for (x,y,w,h) in face_rects:
          cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)
          roi_gray = gray[y:y+h,x:x+h]
          roi_color = frame[y:y+h,x:x+h]
      eye_rects = eye_cascade.detectMultiScale(roi_gray)
      for (x_eye, y_eye,w_eye,h_eye) in eye_rects:
          cv2.rectangle(roi_color, (x_eye,y_eye), (x_eye+w_eye, y_eye+h_eye), (0,255,0), 3)
      
      cv2.imshow('Face Detector', frame)
      if cv2.waitKey(1) & 0xFF==ord('q'):
         break;

cap.release()
cv2.destroyAllWindows()
