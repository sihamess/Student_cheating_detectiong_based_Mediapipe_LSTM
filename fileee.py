import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
import math 
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 03:00:36 2020
@author: hp
"""
import math
from face_detector import get_face_detector, find_faces,draw_faces
from face_landmarksja import get_landmark_model, detect_marks
#from videoprocessnvvvv30  import draw_styled_landmarks,extract_keypoints
from face_landmarks import draw_marks,detect_markss
mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results
def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
def draw_styled_landmarks(image, results):
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    return pose
def list_marks(frame,faces):
    marks=np.zeros(136)
    for face in faces:
        marks = detect_marks(frame, landmark_model, face)
    return marks
def draw(frame,faces):
    for face in faces:
        val=detect_markss(frame, landmark_model, face)
        draw_marks(frame, val, color=(0, 255, 0))
def frame_capture(file): 
 #print(file)
 chaine=file[13:-4]
 #print(chaine)
 DATA_PATH = os.path.join('C:/Users/sihal/Desktop/3test de cheating app/generate_files7') 
 count=0
 
  # Playing video from file:
 vid_capture = cv2.VideoCapture(file)
 frame_rate = vid_capture.get(cv2.CAP_PROP_FPS) #video frame rate
 frames_per_second=4
 if frames_per_second > frame_rate or frames_per_second == -1:
    frames_per_second = frame_rate
 with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while(vid_capture.isOpened()):
        
        ret, frame = vid_capture.read()
        if ret == True:
            image, results = mediapipe_detection(frame, holistic)
            faces = find_faces(frame, face_model)
            marks=list_marks(frame,faces)
            draw_faces(image,faces)
            #draw(image,faces)
            
            
            draw_styled_landmarks(image, results)
            
            #print(count % (math.floor(frame_rate/frames_per_second)))
            if count % (math.floor(frame_rate/frames_per_second)) == 0:
                pose=extract_keypoints(results)
                print(chaine)
                filename ="frame%d" % count
                npy_path = os.path.join(DATA_PATH,chaine+filename)
                #print(marks)
                
                data=np.concatenate([pose,marks])
                #print(np.size(data))
                #print(np.size(data))
                print(np.shape(data))
                np.save(npy_path, data)
            count+=1
            cv2.imshow('Frame',image)
            key = cv2.waitKey(20)
            if key == ord('q'):
                break
        else:
            break
#Release the video capture object
    vid_capture.release()
    cv2.destroyAllWindows()  



start=time.time()
for file in os.listdir("dataset/tout"):
    #if file.endswith(".mp4)"): 
    path=os.path.join("dataset/tout", file)
    face_model = get_face_detector()
    landmark_model = get_landmark_model()
  
    #DATA_PATH = os.path.join('C:/Users/sihal/Desktop/2test de cheating app/data_saved') 
    #count=0
    frame_capture(path)
end=time.time()
print("le temps ecoulé est",end-start)
