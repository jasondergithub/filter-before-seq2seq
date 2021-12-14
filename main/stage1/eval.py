import config
import data_loader
import engine
import pickle
import torch
import numpy as np

from stage1_dataset import ArticleClassificationDataset
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification

first_stage_predictions = []

testset = ArticleClassificationDataset('test')
test_data_loader = DataLoader(testset, batch_size=1)

device = torch.device(config.DEVICE)
model =  BertForSequenceClassification(config.bert_config)
model.load_state_dict(torch.load(config.MODEL_PATH))
model.to(device)

outputs = engine.eval_fn(test_data_loader, model, device)
outputs = np.array(outputs) >= 0.8

with open('../../dict/first_stage_public_test.txt', 'rb') as fp:
     table = pickle.load(fp)

for i, score in enumerate(outputs):
    if score==1:
        first_stage_predictions.append((table[i][0], table[i][1]))

with open("../../dict/first_stage_predictions.txt", "wb") as fp:
    pickle.dump(first_stage_predictions, fp)