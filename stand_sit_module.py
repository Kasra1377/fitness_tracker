from dambbell_module import arm_tracker
import mediapipe as mp
import numpy as np
import math
import cv2

class foot_tracker(arm_tracker):
    def __init__(self, static_image_mode=False, model_complexity=1, enable_segmentation=False,
                    min_detection_confidence=0.5, side="left", min_degree=100, max_degree=170,
                    downward=False, upward=False, direction="", count=0):

        super().__init__(static_image_mode=static_image_mode, model_complexity=model_complexity, enable_segmentation=enable_segmentation,
                    min_detection_confidence=min_detection_confidence, side=side, min_degree=min_degree, max_degree=max_degree,
                    downward=downward, upward=upward, direction="", count=count)

        
    def body_direction(self, frame):
        if self.direction=="Stand":
            cv2.putText(frame, self.direction ,(100, 200), 
                cv2.FONT_HERSHEY_PLAIN, 4, (0, 255, 0), 4)
        
        if self.direction=="Sit":
            cv2.putText(frame, self.direction ,(100, 200), 
                cv2.FONT_HERSHEY_PLAIN, 4, (0, 255, 0), 4)
        return frame
    
    def count_movement(self, frame, previous_degree=0):
        
        self.current_degree = self.degree
        if self.current_degree > previous_degree:
            self.upward = False
        elif self.current_degree < previous_degree:
            self.downward = False

        if self.degree > self.max_degree:
            self.downward = True
            self.direction = "Sit"       
        elif self.degree < self.min_degree:
            self.upward = True
            self.direction = "Stand"
        
        # print("Direction : ",self.direction)
        #print("Upward : ",self.upward)
        if self.downward and self.upward:
            self.count += 1
            self.upward, self.downward = False, False
            previous_degree = self.degree
        # print(self.degree)

        frame = self.body_direction(frame)

        cv2.putText(frame, "{}".format(int(self.count)) ,(100, 100), 
                    cv2.FONT_HERSHEY_PLAIN, 4, (0, 255, 0), 4)
        # print("Downward : ", downward)
        # print("Upward : ", upward)
        return frame

    def track_foot(self, frame, h, w):
        # define a list to save the corresponding points in it
        self.points = []

        self.height = h
        self.width = w

        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(frameRGB)

        if self.results.pose_landmarks:
            # if the module detects one or more hands then
            # loop through attributes of each hand
            # draw hand landmarks and connection between them
            # on the input frame 
            self.mp_drawing.draw_landmarks(
                frame,
                self.results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                )
            
            # since every hand landmark has a unique id and location,
            # so we have to loop through them
            for id, lm in enumerate(self.results.pose_landmarks.landmark):

                x, y = int(lm.x * self.width), int(lm.y * self.height)

                if self.side == "left":
                    ids = [23, 25, 27]
                else:
                    ids = [24, 26, 28]

                try:
                    for i in ids:
                        if id == i:
                            self.points.append((x, y))
                            cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)
                    frame = self.draw_line(frame)
                except Exception as e:
                    print(e)

        return frame