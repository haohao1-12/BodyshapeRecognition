from matplotlib import axes
#from matplotlib.pyplot import axis
from sklearn.decomposition import DictionaryLearning
from Extraction import *
import os
from DataBase import DataBase
import numpy as np
import scipy.stats
answer = [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 44]

##output_index = []
#rate = 5

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

    #output_index.append(min(correlation_rate, key=correlation_rate.get))
    colors = ['g']*44
    output = min(correlation_rate, key=correlation_rate.get)

    if output == answer[k]:
        colors[output-1] = 'b'
        
    elif output != answer[k]:
        colors[output-1] = 'r'
        colors[answer[k]-1] = 'b'

    #colors[min(correlation_rate, key=correlation_rate.get)-1] = 'r'
    plt.xlim(xmin=0.5, xmax = 44.5)
    plt.bar(list(correlation_rate.keys()), correlation_rate.values(), color=colors)
    plt.title('Distance between the subject and samples')
    plt.xlabel('Sample index')
    plt.ylabel('Euclidean distance')
    plt.savefig('D:/File/BodyshapeRecognition/Picture/histogram'+'/'+files[2*k])