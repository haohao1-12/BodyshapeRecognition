from Extraction import *
import os
from DataBase import DataBase
import numpy as np
import scipy.stats

answer = [1,2,3,4,5,6,7,8,9,10]
output_index = []
rate = 0
db = DataBase()
files = os.listdir('D:/File/BodyshapeRecognition/Picture/training')


for k in range(10):
    example1 = frontExtract('D:/File/BodyshapeRecognition/Picture/training'+'/'+files[2*k])
    example2 = sideExtract('D:/File/BodyshapeRecognition/Picture/training'+'/'+files[2*k+1])
    correlation_rate = {}
    
    test_features = []
    for i in example1.exe().values():
        test_features.append(i)
    for j in example2.exe().values():
        test_features.append(j)

    for x in range(len(db)):
        #correlation_rate[x+1] = abs(scipy.stats.pearsonr(np.array(db[x]), np.array(test_features))[0])
        #np.linalg.norm(np.array(db[x])-np.array(test_features))
        correlation_rate[x+1] = np.linalg.norm(np.array(db[x])-np.array(test_features))

    output_index.append(min(correlation_rate, key=correlation_rate.get))

for l in range(len(answer)):
    if answer[l] == output_index[l]:
        rate = rate + 1

print(rate)   