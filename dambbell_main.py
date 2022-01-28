from dambbell_module import arm_tracker
from stand_sit_module import foot_tracker
import mediapipe as mp
import numpy as np
import math
import cv2

# cap = cv2.VideoCapture('demo2.webm')
cap = cv2.VideoCapture('demo.mp4')

# variables
count = 0
side = "left"
direction = ""
min_degree = 70
max_degree = 160
previous_degree = 0
upward, downward = False, False

arm = arm_tracker(side=side)

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
   
size = (frame_width, frame_height)

fourcc = cv2.VideoWriter_fourcc('F','M','P','4')
# fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
out = cv2.VideoWriter('outputs/output2.avi', fourcc, 10.0, size)

while True:
    ret, frame = cap.read()
    h, w, _ = frame.shape
    
    if ret == True:
        frame = arm.track_arm(frame, h, w)
        
        try:
            frame = arm.vectors_angle(frame)
            frame = arm.draw_stats(frame)
            frame = arm.count_dambbell(frame)
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
