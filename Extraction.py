from unittest import result
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

class frontExtract:
    
    
    def __init__(self,image):
        with mp_pose.Pose(min_detection_confidence=0.9, min_tracking_confidence=0.9) as pose:
            #Recolor image
            frame = cv2.imread(image)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make Detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract landmarks
            
            self.landmarks = results.pose_landmarks.landmark
                #print(landmarks)
            

        # Rendering 
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

            #cv2.imwrite("output11.jpg", image)

        self.features = {}
        #print(self.landmarks[13])
        #print(self.landmarks[15])
        #print(self.landmarks[mp_pose.PoseLandmark.NOSE.value])

        #self.features = {}
        #plt.imshow(image)

        #print('------------------')

        

    ##print(type(landmarks[0].x))
    ## Process the keypoints
    def exe(self):
        point = []
        
        for i in range(len(self.landmarks)):
            x = self.landmarks[i].x
            y = self.landmarks[i].y
            #z = self.landmarks[i].z
            point.append(np.array([x,y]))

        #print(point)

        self.features['lefteye_len'] = np.linalg.norm(point[1]-point[3])*10
        self.features['righteye_len']= np.linalg.norm(point[4]-point[6])*10
        #self.features['mouth_len'] = np.linalg.norm(point[10]-point[9])
        #self.features['0-11'] = np.linalg.norm(point[0]-point[11])
        #self.features['0-12'] = np.linalg.norm(point[0]-point[12])
        self.features['0-27'] = np.linalg.norm(point[0]-point[5])*10
        self.features['0-28'] = np.linalg.norm(point[0]-point[2])*10
        self.features['0-25'] = np.linalg.norm(point[0]-point[25])
        self.features['0-26'] = np.linalg.norm(point[0]-point[26])
        self.features['0-5'] = np.linalg.norm(point[0]-point[5])
        self.features['0-2'] = np.linalg.norm(point[0]-point[2])
        self.features['0-7'] = np.linalg.norm(point[0]-point[7])*10
        self.features['0-8'] = np.linalg.norm(point[0]-point[8])*10
        #self.features['0-10'] = np.linalg.norm(point[0]-point[10])
        #self.features['0-9'] = np.linalg.norm(point[0]-point[9])
        self.features['0-24'] = np.linalg.norm(point[0]-point[24])
        self.features['0-23'] = np.linalg.norm(point[0]-point[23])
        self.features['shoulder_len'] = np.linalg.norm(point[12]-point[11])
        #self.features['lefttorso_len'] = np.linalg.norm(point[11]-point[23])
        #self.features['righttorso_len'] = np.linalg.norm(point[12]-point[24])
        self.features['hip_len'] = np.linalg.norm(point[24]-point[23])
        #self.features['leftupperarm_len'] = np.linalg.norm(point[11]-point[13])
        #self.features['leftforearm_len'] = np.linalg.norm(point[13]-point[15])
        self.features['rightupperarm_len'] = np.linalg.norm(point[12]-point[14])
        #self.features['rightforearm_len'] = np.linalg.norm(point[14]-point[16])
        #self.features['leftthigh_len'] = np.linalg.norm(point[23]-point[25])
        self.features['rightthigh_len'] = np.linalg.norm(point[24]-point[26])
        #self.features['leftshin_len'] = np.linalg.norm(point[25]-point[27])
        self.features['rightshin_len'] = np.linalg.norm(point[26]-point[28])
        #self.features['leftfeet_len'] = np.linalg.norm(point[29]-point[31])

        return self.features

class sideExtract:
    
    
    def __init__(self,image):
        with mp_pose.Pose(min_detection_confidence=0.9, min_tracking_confidence=0.9) as pose:
            #Recolor image
            frame = cv2.imread(image)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Make Detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract landmarks
            
            self.landmarks = results.pose_landmarks.landmark
                #print(landmarks)
            

        # Rendering 
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

            #cv2.imwrite("output11.jpg", image)

        self.features = {}
        #print(self.landmarks[13])
        #print(self.landmarks[15])
        #self.features = {}
        #plt.imshow(image)

        #print('------------------')

        

    ##print(type(landmarks[0].x))
    ## Process the keypoints
    def exe(self):
        point = []
        
        for i in range(len(self.landmarks)):
            x = self.landmarks[i].x
            y = self.landmarks[i].y
            #z = self.landmarks[i].z
            point.append(np.array([x,y]))

        #print(point)

        
        self.features['leftforearm_len'] = np.linalg.norm(point[15]-point[13])
        #self.features['leftfeet_len'] = np.linalg.norm(point[29]-point[31])
        self.features['leftupperarm_len'] = np.linalg.norm(point[11]-point[13])
        self.features['leftthigh_len'] = np.linalg.norm(point[23]-point[25])
        self.features['leftshin_len'] = np.linalg.norm(point[25]-point[27])
        self.features['11-27'] = np.linalg.norm(point[11]-point[27])

        #self.features['27-29'] = np.linalg.norm(point[27]-point[29])
        #self.features['27-31'] = np.linalg.norm(point[27]-point[31])

        return self.features
    

### This is for test
'''
example = frontExtract('pose1.jpg')
print(example.exe())

example2 = sideExtract('side1.jpg')
print(example2.exe())
'''

