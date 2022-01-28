import mediapipe as mp
import numpy as np
import math
import cv2

class arm_tracker():
    def __init__(self, static_image_mode=False, model_complexity=1, enable_segmentation=False,
                    min_detection_confidence=0.5, side="left", min_degree=70, max_degree=160,
                    downward=False, upward=False, direction="", count=0):
        
        '''
        Args:
            static_image_mode : If set to false, the solution treats the input images as a video
            stream. It will try to detect the most prominent person in the very first images, and
            upon a successful detection further localizes the pose landmarks. In subsequent images,
            it then simply tracks those landmarks without invoking another detection until it loses
            track, on reducing computation and latency. If set to true, person detection runs every
            input image, ideal for processing a batch of static, possibly unrelated, images. Default
            to false.
            
            model_complexity : Complexity of the pose landmark model: 0, 1 or 2. Landmark accuracy as
            well as inference latency generally go up with the model complexity. Default to 1.

            enable_segmentation : If set to true, in addition to the pose landmarks the solution also
            generates the segmentation mask. Default to false.

            min_detection_confidence : Minimum confidence value ([0.0, 1.0]) from the person-detection
            model for the detection to be considered successful. Default to 0.5.
            
            side : The side of body arm that the program must consider. Default to left.

            min_degree : If the angle between Forearm and Arm is less than this value, then the program
            set the upward value to True. Default to 70. 
            
            max_degree : If the angle between Forearm and Arm is greater than this value, then the program
            set the downward value to True. Default to 160.

            downward : Describes that the exerciser lower the dambbell sufficiently. Default to False. 

            upward : Describes that the exerciser raise the dambbell sufficiently. Default to False.

            direction : The direction of the dambbell that the player must take. Default to an empty string.

            count : 

        '''
        
        self.side = side
        self.count = count
        self.upward = upward
        self.downward = downward
        self.direction = direction
        self.min_degree = min_degree
        self.max_degree = max_degree
        # self.previous_degree = previous_degree
        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.enable_segmentation = enable_segmentation
        self.min_detection_confidence = min_detection_confidence

        self.mp_pose = mp.solutions.pose
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_drawing = mp.solutions.drawing_utils

        self.pose = self.mp_pose.Pose(static_image_mode=self.static_image_mode,
        model_complexity=self.model_complexity,
        enable_segmentation=self.enable_segmentation,
        min_detection_confidence=self.min_detection_confidence)

    # define a function that draws two lines on Forearm and Arm of the player 
    def draw_line(self, frame):
        cv2.line(frame, self.points[1], self.points[0], (0,255,0), 3)
        cv2.line(frame, self.points[1], self.points[2], (0,255,0), 3)
        return frame

    # this function specifies the direction of the dambell that the player
    # must lift
    def dambbell_direction(self, frame):
        if self.direction=="Raise":
            cv2.putText(frame, self.direction ,(100, 200), 
                cv2.FONT_HERSHEY_PLAIN, 4, (0, 255, 0), 4)
        
        if self.direction=="Lower":
            cv2.putText(frame, self.direction ,(100, 200), 
                cv2.FONT_HERSHEY_PLAIN, 4, (0, 255, 0), 4)
        return frame

    def count_dambbell(self, frame, previous_degree=0):
        
        self.current_degree = self.degree
        if self.current_degree > previous_degree:
            self.upward = False
        elif self.current_degree < previous_degree:
            self.downward = False

        if self.degree > self.max_degree:
            self.downward = True
            self.direction = "Raise"       
        elif self.degree < self.min_degree:
            self.upward = True
            self.direction = "Lower"

        if self.downward and self.upward:
            self.count += 1
            self.upward, self.downward = False, False
            previous_degree = self.degree

        frame = self.dambbell_direction(frame)

        cv2.putText(frame, "{}".format(int(self.count)) ,(100, 100), 
                    cv2.FONT_HERSHEY_PLAIN, 4, (0, 255, 0), 4)
        # print("Downward : ", downward)
        # print("Upward : ", upward)
        return frame

    def draw_stats(self, frame):
        cv2.rectangle(frame, (self.width - 60, 120) , (self.width - 20, self.height - 100),
                (0, 255, 0), 3)
        
        x = int(self.degree)
        x = self.max_degree if x > self.max_degree else x
        x = self.min_degree if x < self.min_degree else x

        x = np.abs(x - self.min_degree)
        percentage = (x / np.abs(self.max_degree - self.min_degree)) * 100

        barHeight = x / np.abs(self.max_degree - self.min_degree)
        barHeight = (1 - barHeight) * (self.height - 220)
        
        cv2.rectangle(frame, (self.width - 60, int(120 + barHeight)) , (self.width - 20, self.height - 100),
                (0, 255, 0), -1)

        cv2.putText(frame, "{}%".format(int(percentage)) ,(self.width - 80, self.height - 70), 
            cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
        
        return frame
    
    def vectors_angle(self, frame):
        vec1 = (self.points[0][0] - self.points[1][0],  self.points[0][1] - self.points[1][1])
        vec2 = (self.points[2][0] - self.points[1][0],  self.points[2][1] - self.points[1][1])
        
        rad = np.arccos(np.dot(vec1, vec2) /((np.linalg.norm(vec1) * np.linalg.norm(vec2))))
        self.degree = np.degrees(rad)

        cv2.putText(frame, "{}".format(int(self.degree)) ,(self.points[1][0] + 20, self.points[1][1] + 20), 
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
        
        originVec = (5, 0)
        vec1Angle = np.degrees(np.arccos(np.dot(vec1, originVec) / (np.linalg.norm(vec1) * np.linalg.norm(originVec))))
        # vec2Angle = np.degrees(np.arccos(np.dot(vec2, originVec) / (np.linalg.norm(vec2) * np.linalg.norm(originVec))))
        
        if self.side == "left":
            cv2.ellipse(frame, center=self.points[1], axes=(30, 30), angle=360 - vec1Angle , startAngle=0, endAngle= -self.degree, color=(0,255,0), thickness=2)
        else:
            cv2.ellipse(frame, center=self.points[1], axes=(30, 30), angle=360 - vec1Angle , startAngle=0, endAngle=self.degree, color=(0,255,0), thickness=2)
        
        return frame

    def track_arm(self, frame, h, w):
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
                    ids = [11, 13, 15]
                else:
                    ids = [12, 14, 16]

                try:
                    for i in ids:
                        if id == i:
                            self.points.append((x, y))
                            cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)
                    frame = self.draw_line(frame)
                except Exception as e:
                    print(e)

        return frame