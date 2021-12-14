import pickle
import json

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return len(lst3)


with open('../dict/private_test.txt', 'rb') as fp:
     test_table = pickle.load(fp)


tf = open('../dict/keywordSet.json', 'r')
keywordSet = json.load(tf)
keys_list = list(keywordSet)

temp1 = []
temp2 = []
encode_prediction_list = []
for i in range(len(test_table)):
  with open('../private_processed_test_files/' + str(test_table[i][0]) + '.txt', 'r', encoding='UTF-8') as text1:
    file1 = text1.read()

  with open('../private_processed_test_files/' + str(test_table[i][1]) + '.txt', 'r', encoding='UTF-8') as text2:
    file2 = text2.read()  
  
  for key in keys_list:
    value = keywordSet[key]

    if file1.find(key) != -1:
      if value not in temp1:
        temp1.append(value)
    if file2.find(key) != -1:
      if value not in temp2:
        temp2.append(value)

  encode_prediction_list.append((temp1, temp2))
  temp1 = []
  temp2 = []


filter_table = []
for i in range(len(test_table)):
  num = intersection(encode_prediction_list[i][0], encode_prediction_list[i][1])
  
  if len(encode_prediction_list[i][0]) != 0:
    if num/len(encode_prediction_list[i][0]) > 0.8:
      filter_table.append(test_table[i])

with open("../dict/filter_test.txt",  "wb") as fp:
    pickle.dump(filter_table, fp)