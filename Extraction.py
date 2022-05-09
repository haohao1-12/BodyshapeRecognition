from unittest import result
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

class frontExtract:
    
    
    def __init__(self,image):
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
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

            cv2.imwrite("output11.jpg", image)

        self.features = {}

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
            z = self.landmarks[i].z
            point.append(np.array((x,y,z)))

        #print(point)

        self.features['lefteye_len'] = np.linalg.norm(point[1]-point[3])
        self.features['righteye_len']= np.linalg.norm(point[4]-point[6])
        self.features['mouth_len'] = np.linalg.norm(point[10]-point[9])
        self.features['shoulder_len'] = np.linalg.norm(point[12]-point[11])
        self.features['lefttorso_len'] = np.linalg.norm(point[11]-point[23])
        self.features['righttorso_len'] = np.linalg.norm(point[12]-point[24])
        self.features['hip_len'] = np.linalg.norm(point[24]-point[23])
        self.features['leftupperarm_len'] = np.linalg.norm(point[11]-point[13])
        #self.features['leftforearm_len'] = np.linalg.norm(point[13]-point[15])
        self.features['rightupperarm_len'] = np.linalg.norm(point[12]-point[14])
        #self.features['rightforearm_len'] = np.linalg.norm(point[14]-point[16])
        self.features['leftthigh_len'] = np.linalg.norm(point[23]-point[25])
        self.features['rightthigh_len'] = np.linalg.norm(point[24]-point[26])
        self.features['leftshin_len'] = np.linalg.norm(point[25]-point[27])
        self.features['leftshin_len'] = np.linalg.norm(point[26]-point[28])
        #self.features['leftfeet_len'] = np.linalg.norm(point[29]-point[31])

        return self.features

class sideExtract:
    
    
    def __init__(self,image):
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
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

            cv2.imwrite("output11.jpg", image)

        self.features = {}

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
            z = self.landmarks[i].z
            point.append(np.array((x,y,z)))

        #print(point)

        
        self.features['leftforearm_len'] = np.linalg.norm(point[13]-point[15])
        self.features['leftfeet_len'] = np.linalg.norm(point[29]-point[31])

        return self.features
    


#example = Extract('pose1.jpg')
#print(example.exe()['lefteye_len'])

