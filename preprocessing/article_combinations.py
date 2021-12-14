import itertools 
import os
import pandas as pd
import pickle

total_articles = []
total_combinations = []

directory = r'../processed_files'
for filename in os.listdir(directory):
    total_articles.append(int(filename[:-4]))

amount = len(total_articles)
for i in range(amount):
    article_no = []
    list_pop = total_articles.pop(0)
    article_no.append(list_pop)
    combination = list(itertools.product(article_no, total_articles))
    total_combinations += combination
    total_articles.append(list_pop)

related_df = pd.read_csv('../data/TrainLabel.csv')
first_stage_related_table = []
for i in range(len(related_df)):
    tuples = (related_df.iloc[i, 0], related_df.iloc[i, 1])
    first_stage_related_table.append(tuples)

diff = lambda l1,l2: [x for x in l1 if x not in l2]
first_stage_unrelated_table = diff(total_combinations, first_stage_related_table)

for i in range(len(first_stage_related_table)):
    first_stage_related_table[i] = first_stage_related_table[i] + (1,)

for i in range(len(first_stage_unrelated_table)):
    first_stage_unrelated_table[i] = first_stage_unrelated_table[i] + (0,)

with open("../dict/relatedTable.txt", "wb") as fp:
    pickle.dump(first_stage_related_table, fp)
with open("../dict/unrelatedTable.txt", "wb") as fp:
    pickle.dump(first_stage_unrelated_table, fp)

#print(len(total_combinations))
