from dambbell_module import arm_tracker
from stand_sit_module import foot_tracker
import mediapipe as mp
import numpy as np
import math
import cv2

import stand_sit_module

cap = cv2.VideoCapture('demo3.webm')

# variables
count = 0
side = "left"
direction = ""
min_degree = 100
max_degree = 170
previous_degree = 0
upward, downward = False, False

# arm = arm_tracker(side=side)
foot = foot_tracker()

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
   
size = (frame_width, frame_height)

fourcc = cv2.VideoWriter_fourcc('F','M','P','4')
# fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
out = cv2.VideoWriter('outputs/output3.avi', fourcc, 10.0, size)

while True:
    _, frame = cap.read()
    h, w, _ = frame.shape
    
    frame = foot.track_foot(frame, h, w)
    
    try:
        frame = foot.vectors_angle(frame)
        frame = foot.draw_stats(frame)
        frame = foot.count_movement(frame, previous_degree=previous_degree)
    except Exception as e:
        print(e)
    
    out.write(frame)
    cv2.imshow("Result", frame)

    k = cv2.waitKey(5)
    if k == 27:
        break
 
cap.release()
out.release()
cv2.destroyAllWindows()
