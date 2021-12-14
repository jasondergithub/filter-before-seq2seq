import config
import engine
import pickle
import data_loader
import numpy as np
import torch
import torch.nn as nn

from encoderDataset import ReferenceDataset
from decoderDataset import TestDataset
from Model import Seq2Seq, Encoder, Decoder
from torch.utils.data import DataLoader
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn import metrics
from sklearn.metrics import f1_score

def run_train(tableNumber):
    encoder_trainset = ReferenceDataset('train', tableNumber)
    encoder_data_loader = DataLoader(encoder_trainset, batch_size=config.BATCH_SIZE, collate_fn=data_loader.create_encoder_batch)

    decoder_trainset = TestDataset('train', tableNumber)
    decoder_data_loader = DataLoader(decoder_trainset, batch_size=config.BATCH_SIZE, collate_fn=data_loader.create_decoder_batch)

    device = torch.device(config.DEVICE)
    
    encoder = Encoder()
    encoder.to(device)
    decoder = Decoder()
    decoder.to(device)

    model = Seq2Seq(encoder, decoder)
    # if tableNumber > 1:
    #     model.load_state_dict(torch.load(config.MODEL_PATH))
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


    with open('../table/table'+ str(tableNumber) +'.txt', 'rb') as fp:
        table = pickle.load(fp)
    length_df = len(table)
    num_train_steps = int(length_df / config.BATCH_SIZE * config.EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )
    #########

    for epoch in range(config.EPOCHS):
        outputs, targets, loss = engine.train_fn(encoder_data_loader, decoder_data_loader, model, optimizer, device, scheduler)
        print(f'Epoch:{epoch+1}, Loss:{loss:.4f}')

        outputs = np.array(outputs) >= 0.8
        accuracy = metrics.accuracy_score(targets, outputs)
        fScore = f1_score(targets, outputs)
        print(f"After training {epoch+1} epoch(s), Accuracy Score = {accuracy}")
        print(f"After training {epoch+1} epoch(s), F1 Score = {fScore}")
    #save model
    torch.save(model.state_dict(), config.MODEL_PATH)

if __name__ == "__main__":

    print('-------------------------------------')
    #config.EPOCHS = 5
    run_train(100)
    print('-------------------------------------')