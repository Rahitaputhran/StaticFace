import cv2
import mediapipe as mp
import time

from fontTools.merge.util import current_time

mp_face_detection=mp.solutions.face_detection
face_detection=mp_face_detection.FaceDetection(min_detection_confidence=0.5)

cap=cv2.VideoCapture(0)

saved_face=None
start_time=time.time()
capture_duration=5
while cap.isOpened():
    success,frame=cap.read()
    frame=cv2.flip(frame,1)
    if not success:
        break
    frame_rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    result=face_detection.process(frame_rgb)
    current_time=time.time()

    if result.detections and (current_time - start_time < capture_duration):
        saved_face=frame.copy()
    if result.detections:
        cv2.imshow('Face Projection', frame)
    else:
        if saved_face is not None:
            cv2.imshow('Face Projection', saved_face)
        else:
            cv2.imshow('Face Projection', frame)
    cv2.waitKey(1)