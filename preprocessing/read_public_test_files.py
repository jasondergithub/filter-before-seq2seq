import os
import json

# load keywords directory 
tf = open('../dict/keywords_dict.json', 'r')
keywords = json.load(tf)
keys_list = list(keywords)
keyword_list = []

# iterate files in the given directory
directory = r'C:\Users\劉碩元\Desktop\seq2seq\publicTestData'
sentence = ''
count = 0
for filename in os.listdir(directory):
    with open('../publicTestData/' + filename, 'r', encoding='utf-8') as f:
        for line in f:
            sentence += line[:-1]
            count += 1
            if count == 2:
                break   
    with open('../publicTestData/' + filename, 'r', encoding='utf-8') as f:
        text = f.read()

    for key in keys_list:
        value = keywords[key]
        if text.find(value) != -1:
            keyword_list.append(value)

    if sentence[-1] != '。':
        sentence += '。'
    sentence += ','.join(keyword_list)
    sentence += '。'

    f = open('../public_processed_test_files/' + filename, 'w', encoding='UTF-8')
    f.write(sentence)
    f.close()
    keyword_list.clear()
    count = 0
    sentence = ''    