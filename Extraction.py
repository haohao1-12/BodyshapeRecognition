from unittest import result
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    #Recolor image
    frame = cv2.imread('pose1.jpg')
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Make Detection
    results = pose.process(image)

    # Recolor back to BGR
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Extract landmarks
    try:
        landmarks = results.pose_landmarks.landmark
        print(landmarks)
    except:
        pass

    # Rendering 
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                     mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                     mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

    cv2.imwrite("output1.jpg", image)
    plt.imshow(image)

