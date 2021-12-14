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

predictions = []

encoder_trainset = ReferenceDataset('test')
encoder_data_loader = DataLoader(encoder_trainset, batch_size=config.BATCH_SIZE, collate_fn=data_loader.create_encoder_batch)

decoder_trainset = TestDataset('test')
decoder_data_loader = DataLoader(decoder_trainset, batch_size=config.BATCH_SIZE, collate_fn=data_loader.create_decoder_batch)

device = torch.device(config.DEVICE)

encoder = Encoder()
encoder.to(device)
decoder = Decoder()
decoder.to(device)

model = Seq2Seq(encoder, decoder)
model.load_state_dict(torch.load(config.MODEL_PATH))
model.to(device)


outputs = engine.eval_fn(encoder_data_loader, decoder_data_loader, model, device)
outputs = np.array(outputs) >= 0.8

with open('../dict/private_test.txt', 'rb') as fp:
     table = pickle.load(fp)

for i, score in enumerate(outputs):
    if score==1:
        predictions.append((table[i][0], table[i][1]))

with open("../dict/final_predictions.txt", "wb") as fp:
    pickle.dump(predictions, fp)