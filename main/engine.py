import pickle
from tqdm import tqdm
from encoderDataset import ReferenceDataset
from decoderDataset import TestDataset
import data_loader
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def loss_fn(outputs, targets):
    return nn.BCEWithLogitsLoss()(outputs, targets)

def train_fn(src_loader, trg_loader, model, optimizer, device, scheduler):
    
    model.train()
    fin_targets = []
    fin_outputs = []
    running_loss = 0    

    for (src_data, trg_data) in tqdm(zip(src_loader, trg_loader)):
        
        src_tokens_tensor = src_data['tokens_tensor']
        src_segments_tensor = src_data['segments_tensor']
        src_masks_tensor = src_data['masks_tensor']
        
        src_tokens_tensor = src_tokens_tensor.to(device, dtype=torch.long)
        src_segments_tensor = src_segments_tensor.to(device, dtype=torch.long)
        src_masks_tensor = src_masks_tensor.to(device, dtype=torch.long)

        reference = [src_tokens_tensor, src_segments_tensor, src_masks_tensor]

        trg_tokens_tensor = trg_data['tokens_tensor']
        trg_segments_tensor = trg_data['segments_tensor']
        trg_masks_tensor = trg_data['masks_tensor']
        trg_label = trg_data['target']

        trg_tokens_tensor = trg_tokens_tensor.to(device, dtype=torch.long)
        trg_segments_tensor =trg_segments_tensor.to(device, dtype=torch.long)
        trg_masks_tensor = trg_masks_tensor.to(device, dtype=torch.long)   
        trg_label_tensor = trg_label.to(device, dtype=torch.float)

        target = [trg_tokens_tensor, trg_segments_tensor, trg_masks_tensor]

        optimizer.zero_grad()
        outputs = model(reference, target)
        outputs = torch.squeeze(outputs, 1)

        loss = loss_fn(outputs, trg_label_tensor)
        loss.backward()
        optimizer.step()
        scheduler.step()

        fin_targets.extend(trg_label_tensor.cpu().detach().numpy().tolist())
        fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
        running_loss += loss.item()

    return fin_outputs, fin_targets, running_loss 

def eval_fn(encoder_loader, decoder_loader, model, device):
    model.eval()
    fin_outputs = []
    with torch.no_grad():   
         for (src_data, trg_data) in tqdm(zip(encoder_loader, decoder_loader)):
            src_tokens_tensor = src_data['tokens_tensor']
            src_segments_tensor = src_data['segments_tensor']
            src_masks_tensor = src_data['masks_tensor']
        
            src_tokens_tensor = src_tokens_tensor.to(device, dtype=torch.long)
            src_segments_tensor = src_segments_tensor.to(device, dtype=torch.long)
            src_masks_tensor = src_masks_tensor.to(device, dtype=torch.long)
            reference = [src_tokens_tensor, src_segments_tensor, src_masks_tensor]

            trg_tokens_tensor = trg_data['tokens_tensor']
            trg_segments_tensor = trg_data['segments_tensor']
            trg_masks_tensor = trg_data['masks_tensor']

            trg_tokens_tensor = trg_tokens_tensor.to(device, dtype=torch.long)
            trg_segments_tensor =trg_segments_tensor.to(device, dtype=torch.long)
            trg_masks_tensor = trg_masks_tensor.to(device, dtype=torch.long)   
            target = [trg_tokens_tensor, trg_segments_tensor, trg_masks_tensor] 

            outputs = model(reference, target)
            outputs = torch.squeeze(outputs, 1) 

            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())                              
    
    return fin_outputs