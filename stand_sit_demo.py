import mediapipe as mp
import numpy as np
import math
import cv2

mp_pose = mp.solutions.pose
mp_drawing_styles = mp.solutions.drawing_styles
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5)

def draw_line(frame, points):
        cv2.line(frame, points[1], points[0], (0,255,0), 3)
        cv2.line(frame, points[1], points[2], (0,255,0), 3)
        return frame

def body_direction(frame, direction):
    if direction=="Stand":
        cv2.putText(frame, direction ,(100, 200), 
            cv2.FONT_HERSHEY_PLAIN, 4, (0, 255, 0), 4)
    
    if direction=="Sit":
        cv2.putText(frame, direction ,(100, 200), 
            cv2.FONT_HERSHEY_PLAIN, 4, (0, 255, 0), 4)
    return frame

def count_movement(frame ,degree, direction, min_degree, max_degree, count=0, upward=False,
                        downward=False, previous_degree=0):
        
        current_degree = degree
        if current_degree > previous_degree:
            upward = False
        elif current_degree < previous_degree:
            downward = False

        if degree > max_degree:
            downward = True
            direction = "Sit"       
        elif degree < min_degree:
            upward = True
            direction = "Stand"
        
        print("Direction : ",direction)
        #print("Upward : ",upward)
        if downward and upward:
            count += 1
            upward, downward = False, False
            previous_degree = degree
        print(degree)

        frame = body_direction(frame, direction)

        cv2.putText(frame, "{}".format(int(count)) ,(100, 100), 
                    cv2.FONT_HERSHEY_PLAIN, 4, (0, 255, 0), 4)
        # print("Downward : ", downward)
        # print("Upward : ", upward)
        return frame, upward, downward, direction, count

def draw_stats(frame, degree, width, height ,max_degree, min_degree):

    cv2.rectangle(frame, (width - 60, 120) , (width - 20, height - 100),
            (0, 255, 0), 3)
    
    x = int(degree)
    x = max_degree if x > max_degree else x
    x = min_degree if x < min_degree else x

    x = np.abs(x - min_degree)
    percentage = (x / np.abs(max_degree - min_degree)) * 100

    barHeight = x / np.abs(max_degree - min_degree)
    barHeight = (1 - barHeight) * (height - 220)
    cv2.rectangle(frame, (width - 60, int(120 + barHeight)) , (width - 20, height - 100),
            (0, 255, 0), -1)

    cv2.putText(frame, "{}%".format(int(percentage)) ,(width - 80, height - 70), 
        cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
    return frame

def vectors_angle(frame ,points, side):

    vec1 = (points[0][0] - points[1][0],  points[0][1] - points[1][1])
    vec2 = (points[2][0] - points[1][0],  points[2][1] - points[1][1])
    
    rad = np.arccos(np.dot(vec1, vec2) /((np.linalg.norm(vec1) * np.linalg.norm(vec2))))
    degree = np.degrees(rad)

    cv2.putText(frame, "{}".format(int(degree)) ,(points[1][0] + 20, points[1][1] + 20), 
                cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
    
    originVec = (5, 0)
    vec1Angle = np.degrees(np.arccos(np.dot(vec1, originVec) / (np.linalg.norm(vec1) * np.linalg.norm(originVec))))
    vec2Angle = np.degrees(np.arccos(np.dot(vec2, originVec) / (np.linalg.norm(vec2) * np.linalg.norm(originVec))))
    
    if side == "left":
        cv2.ellipse(frame, center=points[1], axes=(30, 30), angle=360 - vec1Angle , startAngle=0, endAngle= -degree, color=(0,255,0), thickness=2)
    else:
        cv2.ellipse(frame, center=points[1], axes=(30, 30), angle=360 - vec1Angle , startAngle=0, endAngle=degree, color=(0,255,0), thickness=2)
    
    return frame, degree


def track_foot(frame, h, w, side="right"):
    # define a list to save the corresponding points in it
    points = []
    height = h
    width = w

    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frameRGB)

    if results.pose_landmarks:
        # if the module detects one or more hands then
        # loop through attributes of each hand
        # draw hand landmarks and connection between them
        # on the input frame 
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            )
        
        # since every hand landmark has a unique id and location,
        # so we have to loop through them
        for id, lm in enumerate(results.pose_landmarks.landmark):

            x, y = int(lm.x * width), int(lm.y * height)

            if side == "left":
                ids = [23, 25, 27]
            else:
                ids = [24, 26, 28]

            try:
                for i in ids:
                    if id == i:
                        points.append((x, y))
                        cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)
                frame = draw_line(frame, points)
            except Exception as e:
                print(e)

    return frame, points