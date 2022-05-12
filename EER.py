from cProfile import label
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

FAR = []
FRR = []
for threshold in np.arange(0,0.5,0.02):
    FA = 0
    A = 0
    FR = 0
    R = 0

    ## Walk through all the test images
    for k in range(int(len(files)/2)):
        example1 = sideExtract('D:/File/BodyshapeRecognition/Picture/test'+'/'+files[2*k])
        example2 = frontExtract('D:/File/BodyshapeRecognition/Picture/test'+'/'+files[2*k+1])
        #correlation_rate = {}

        ## Generate the features of the test image
        test_features = []
        for i in example2.exe().values():
            test_features.append(i)
        for j in example1.exe().values():
            test_features.append(j)
        ## ----------------------------------------

        ## Compare and Calculate
        for x in range(len(db)):
            #correlation_rate[x+1] = abs(scipy.stats.pearsonr(np.array(db[x]), np.array(test_features))[0])
            #np.linalg.norm(np.array(db[x])-np.array(test_features))
            correlation_rate = np.linalg.norm(np.array(db[x])-np.array(test_features))
            if correlation_rate <= threshold:
                A = A+1
                if x+1 != answer[k]:
                    FA = FA+1    
            else:
                R = R+1
                if x+1 == answer[k]:
                    FR = FR+1
        ##------------------------------------------
    
    FAR.append(FA/(11*43))

    
    FRR.append(FR/11)

FAR = np.array(FAR)
FRR = np.array(FRR)

plt.plot(np.arange(0,0.5,0.02), FAR, 'r', label="FAR") 
plt.plot(np.arange(0,0.5,0.02), FRR, 'b', label="FRR")
plt.legend(loc="upper left")
plt.title('FAR and FRR')
plt.xlabel('threshold')
plt.ylabel('Rate')
plt.show()  
plt.savefig('D:/File/BodyshapeRecognition/Picture/EER')

    


'''
    for l in range(len(answer)):
        if answer[l] == output_index[l]:
            rate = rate + 1

    print(rate)
'''
