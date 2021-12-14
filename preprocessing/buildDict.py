'''
把keywords資料夾，底下的三種keyword檔案： 化學、農作物、蟲害，另外加上台灣的地區名稱，彙整成一個字典
之後讀取txt檔時，可以將這四種關鍵選出來，放到lsit中
'''
import pandas as pd
import numpy as np
import json
import os
keywords_dict = {}

# 寫入字典

for dirName, dum, fileNames in os.walk('../data/Keywords'):
    for file in fileNames:
        path = os.path.join(dirName, file)
        print(path)
        file1 = pd.read_excel(path, header=None)
        for j in range(len(file1)):
            for i in list(file1.loc[j].dropna()):
                keywords_dict[i] = i

keywords_dict['台北'] = '臺北'
keywords_dict['臺北'] = '臺北'
keywords_dict['新北'] = '新北'
keywords_dict['桃園'] = '桃園'
keywords_dict['台中'] = '臺中'
keywords_dict['臺中'] = '臺中'
keywords_dict['台南'] = '臺南'
keywords_dict['臺南'] = '臺南'
keywords_dict['高雄'] = '高雄'
keywords_dict['新竹'] = '新竹'
keywords_dict['苗栗'] = '苗栗'
keywords_dict['彰化'] = '彰化'
keywords_dict['南投'] = '南投'
keywords_dict['雲林'] = '雲林'
keywords_dict['嘉義'] = '嘉義'
keywords_dict['屏東'] = '屏東'
keywords_dict['宜蘭'] = '宜蘭'
keywords_dict['花蓮'] = '花蓮'
keywords_dict['臺東'] = '臺東'
keywords_dict['台東'] = '臺東'
keywords_dict['澎湖'] = '澎湖'
keywords_dict['金門'] = '金門'
keywords_dict['連江'] = '連江'
keywords_dict['基隆'] = '基隆'

# delete from dictionary a nan key and space
keywords_dict = {k: v for k, v in keywords_dict.items() if k==k}
keywords_dict = {k: v for k, v in keywords_dict.items() if v != ' '}

# example of finding key words in article
article = "梅雨季來臨，文旦黑點病易發生，請注意病徵，以及早加強防治措施。 \
5月已進入梅雨季節，近日連續降雨，為文旦黑點病開始感染的時機，往年文旦在經過4-6月的春雨及梅雨季後，原來長得亮麗的果實外表，會開始出現許多小黑點，現在文旦已開始進入中果期，花蓮區農業改良場呼籲應注意防治。\
  除冬季清園作業外，在4-8月時應每月施用一次56%貝芬硫可 濕性粉劑800倍、或22.7% 硫 水懸劑1000倍、或80%鋅錳乃浦可濕性粉劑500倍、或33%鋅錳乃浦水懸劑500倍等政府核准登記使用之藥劑防治，並依登記使用方法使用，尤其雨前及雨後要特別加強防治，若遇連續降雨時則可利用間歇時分區進行施藥以即時達到防治效果。 \
  "
keys_list = list(keywords_dict)
keyword_list = []
for key in keys_list:
    value = keywords_dict[key]
    #print(value)
    if article.find(value) != -1:
        keyword_list.append(value)

print(keyword_list)
print(keywords_dict['腈硫醌可濕性粉劑'])
#save dictionary
tf = open('../dict/keywords_dict.json', 'w')
json.dump(keywords_dict, tf)
tf.close()