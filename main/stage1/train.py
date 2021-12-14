import config
import engine
import pickle
import data_loader
import numpy as np
import torch
import torch.nn as nn

from stage1_dataset import ArticleClassificationDataset
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn import metrics
from sklearn.metrics import f1_score

def run_train(tableNumber):
    trainset = ArticleClassificationDataset('train', tableNumber)
    train_data_loader = DataLoader(trainset, batch_size=config.BATCH_SIZE, collate_fn=data_loader.create_mini_batch)

    device = torch.device(config.DEVICE)
    model =  BertForSequenceClassification(config.bert_config)
    if tableNumber > 1:
        model.load_state_dict(torch.load(config.MODEL_PATH))
    model.to(device)
    
    # I am not familiar with this part yet
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]


    with open('../../table/table'+ str(tableNumber) +'.txt', 'rb') as fp:
        table = pickle.load(fp)
    length_df = len(table)
    num_train_steps = int(length_df / config.BATCH_SIZE * config.EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )
    #########
    
    #best_accuracy = 0

    for epoch in range(config.EPOCHS):
        outputs, targets, loss = engine.train_fn(train_data_loader, model, optimizer, device, scheduler)
        print(f'Epoch:{epoch+1}, Loss:{loss:.4f}')

        outputs = np.array(outputs) >= 0.8
        accuracy = metrics.accuracy_score(targets, outputs)
        fScore = f1_score(targets, outputs)
        print(f"After training {epoch+1} epoch(s), Accuracy Score = {accuracy}")
        print(f"After training {epoch+1} epoch(s), F1 Score = {fScore}")
    #save model
    torch.save(model.state_dict(), config.MODEL_PATH)
    
if __name__ == "__main__":
    for i in range(11):
        print('-------------------------------------')
        print(f'Table : {i+1}')
        # if (i+1) == 5:
        #     config.EPOCHS = 1
        run_train(i+1)
        print('-------------------------------------')