'''
可以回傳mini-batch的DataLoader
製作mask tensor
'''
import pickle
import config
import torch

from torch.utils import data
from encoderDataset import ReferenceDataset
from decoderDataset import TestDataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


def create_decoder_batch(samples):
    tokens_tensor = [s['tokens_tensor'] for s in samples]
    segments_tensor = [s['segments_tensor'] for s in samples]
    masks_tensor = [s['masks_tensor'] for s in samples]
    target_tensor = torch.stack([s['target'] for s in samples])

    tokens_tensors = pad_sequence(tokens_tensor, batch_first=True)                                      
    segments_tensors = pad_sequence(segments_tensor, batch_first=True)
    masks_tensors = pad_sequence(masks_tensor, batch_first=True)
                                                                    
    return {
        'tokens_tensor' : tokens_tensors,
        'segments_tensor': segments_tensors,
        'masks_tensor' : masks_tensors,
        'target' : target_tensor
    }

def create_encoder_batch(samples):
    tokens_tensor = [s['tokens_tensor'] for s in samples]
    segments_tensor = [s['segments_tensor'] for s in samples]
    masks_tensor = [s['masks_tensor'] for s in samples]

    tokens_tensors = pad_sequence(tokens_tensor, batch_first=True)                                      
    segments_tensors = pad_sequence(segments_tensor, batch_first=True)
    masks_tensors = pad_sequence(masks_tensor, batch_first=True)
                                                                    
    return {
        'tokens_tensor' : tokens_tensors,
        'segments_tensor': segments_tensors,
        'masks_tensor' : masks_tensors,
    }

'''
BATCH_SIZE = 32
trainloader = DataLoader(dataset.trainset, batch_size=BATCH_SIZE, collate_fn=create_mini_batch)

data = next(iter(trainloader))
tokens_tensors, segments_tensors, masks_tensors = data

print(f"""
tokens_tensors.shape   = {tokens_tensors.shape} 
{tokens_tensors}
------------------------
segments_tensors.shape = {segments_tensors.shape}
{segments_tensors}
------------------------
masks_tensors.shape    = {masks_tensors.shape}
{masks_tensors}
""")
'''    

# with open('../table/table'+ str(1) +'.txt', 'rb') as fp:
#     table = pickle.load(fp)
# print(table[0:4])

# encoder_trainset = ReferenceDataset('train', 1)
# encoder_data_loader = DataLoader(encoder_trainset, batch_size=4, collate_fn=create_encoder_batch)
# data = next(iter(encoder_data_loader))

# print(f"""
# tokens_tensors.shape   = {data['tokens_tensor'].shape} 
# {data['tokens_tensor']}
# ------------------------
# segments_tensors.shape = {data['segments_tensor'].shape}
# {data['segments_tensor']}
# ------------------------
# masks_tensors.shape    = {data['masks_tensor'].shape}
# {data['masks_tensor']}
# """)


# decoder_trainset = TestDataset('train', 1)
# decoder_data_loader = DataLoader(decoder_trainset, batch_size=16, collate_fn=create_decoder_batch)
# data = next(iter(decoder_data_loader))
# print(f"""
# tokens_tensors.shape   = {data['tokens_tensor'].shape} 
# {data['tokens_tensor']}
# ------------------------
# segments_tensors.shape = {data['segments_tensor'].shape}
# {data['segments_tensor']}
# ------------------------
# masks_tensors.shape    = {data['masks_tensor'].shape}
# {data['masks_tensor']}
# ------------------------
# label_tensors.shape = {data['target'].shape}
# {data['target']}
# """)