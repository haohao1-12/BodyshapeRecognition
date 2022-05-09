from Extraction import *
import os
def DataBase():
    # DataBase is a list
    DB = []
    files = os.listdir('D:/File/BodyshapeRecognition/Picture/training')

    for k in range(int(len(files)/2)):
        example1 = frontExtract('D:/File/BodyshapeRecognition/Picture/training'+'/'+files[2*k])
        example2 = sideExtract('D:/File/BodyshapeRecognition/Picture/training'+'/'+files[2*k+1])
        selected_features = []

        #print(example.exe()['lefteye_len'])
        for i in example1.exe().values():
            selected_features.append(i)
        for j in example2.exe().values():
            selected_features.append(j)

        DB.append(selected_features)
    return DB

    #print(DB)
    #print(len(DB))



    #print(files)
