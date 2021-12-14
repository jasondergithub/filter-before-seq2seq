import pandas as pd
import numpy as np
import os
import json
keywordSet = {}
count = 1
for dirName, dum, fileNames in os.walk('../data/Keywords'):
    for file in fileNames:
        path = os.path.join(dirName, file)
        print(path)
        file1 = pd.read_excel(path, header=None)
        for j in range(len(file1)):
            for i in list(file1.loc[j].dropna()):
                keywordSet[i] = count
            count+=1

#save dictionary
tf = open('../dict/keywordSet.json', 'w')
json.dump(keywordSet, tf)
tf.close()