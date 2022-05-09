from Extraction import *
import os
from DataBase import DataBase
import numpy as np
import scipy.stats

answer = [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 44]
output_index = []
rate = 0

db = DataBase()
files = os.listdir('D:/File/BodyshapeRecognition/Picture/test')


for k in range(int(len(files)/2)):
    example1 = sideExtract('D:/File/BodyshapeRecognition/Picture/test'+'/'+files[2*k])
    example2 = frontExtract('D:/File/BodyshapeRecognition/Picture/test'+'/'+files[2*k+1])
    correlation_rate = {}

    test_features = []
    for i in example2.exe().values():
        test_features.append(i)
    for j in example1.exe().values():
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
'''
print(correlation_rate)
sorted = dict(sorted(correlation_rate.items(), key=lambda x: x[1]))
print(sorted)
'''

