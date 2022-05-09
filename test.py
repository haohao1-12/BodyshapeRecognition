from Extraction import *
import os
from DataBase import DataBase
import numpy as np

db = DataBase()
correlation_rate = {}
example1 = sideExtract('D:/File/BodyshapeRecognition/Picture/test/DSC00167.JPG')
example2 = frontExtract('D:/File/BodyshapeRecognition/Picture/test/DSC00168.JPG')
test_features = []
for i in example2.exe().values():
    test_features.append(i)
for j in example1.exe().values():
    test_features.append(j)

for k in range(len(db)):
    correlation_rate[k+1] = np.linalg.norm(np.array(test_features)-np.array(db[k]))

print(correlation_rate)
sorted = dict(sorted(correlation_rate.items(), key=lambda x: x[1]))
print(sorted)


