import itertools 
import os
import pickle

total_articles = []
total_combinations = []

directory = r'../private_processed_test_files'
for filename in os.listdir(directory):
    total_articles.append(int(filename[:-4]))

#print(len(total_articles))
amount = len(total_articles)
for i in range(amount):
    article_no = []
    list_pop = total_articles.pop(0)
    article_no.append(list_pop)
    combination = list(itertools.product(article_no, total_articles))
    total_combinations += combination
    total_articles.append(list_pop)

with open("../dict/private_test.txt", "wb") as fp:
    pickle.dump(total_combinations, fp)

# print(len(total_combinations))