import pickle

with open('../table/table4.txt', 'rb') as fp:
    table = pickle.load(fp)

print(table[0])
print(table[0][0])