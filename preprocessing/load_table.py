import pickle
import random

with open("../dict/relatedTable.txt", "rb") as fp:
    related_table = pickle.load(fp)

with open("../dict/unrelatedTable.txt", "rb") as fp:
    unrelated_table = pickle.load(fp)

name_list = ['table1.txt', 'table2.txt', 'table3.txt', 'table4.txt', 'table5.txt', 'table6.txt', 
'table7.txt', 'table8.txt', 'table9.txt', 'table10.txt', 'table11.txt', 'table12.txt', 'table13.txt', 'table14.txt',
'table15.txt', 'table16.txt', 'table17.txt', 'table18.txt', 'table19.txt', 'table20.txt']
# for i in range(20):
#     if (i+1) % 5 == 0:
#         subtable = random.sample(unrelated_table, 2762)
#     else:
#         subtable = random.sample(unrelated_table, 1381)    
#     subtable = subtable + related_table
#     random.shuffle(subtable)
#     with open("../table/" + name_list[i], "wb") as fp:
#         pickle.dump(subtable, fp)

subtable = random.sample(unrelated_table, 6905) #8286
for i in range(3): #4
    subtable += related_table

random.shuffle(subtable)
with open("../table/table100.txt",  "wb") as fp:
    pickle.dump(subtable, fp)