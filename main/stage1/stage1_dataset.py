import pickle
import pandas as pd
import torch
import config

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class ArticleClassificationDataset(Dataset):
    def __init__(self, mode, tableNumber=1):
        assert mode in ['train', 'test']
        self.mode = mode
        if self.mode == 'train':
            with open('../../table/table'+ str(tableNumber) +'.txt', 'rb') as fp:
                table = pickle.load(fp)
        else:
            with open('../../dict/first_stage_public_test.txt', 'rb') as fp:
                table = pickle.load(fp)
        self.pairingTable =table 
        self.len = len(table)
        self.tokenizer = config.tokenizer
    
    def __getitem__(self, index):
        if self.mode == 'train':
            num1 = self.pairingTable[index][0]
            num2 = self.pairingTable[index][1]
            label = self.pairingTable[index][2]
            with open('../../processed_files/' + str(num1) + '.txt', 'r', encoding='UTF-8') as text1:
                file1 = text1.read()
            with open('../../processed_files/' + str(num2) + '.txt', 'r', encoding='UTF-8') as text2:
                file2 = text2.read()

            inputs = self.tokenizer.encode_plus(
                file1,
                file2,
                add_special_tokens = True,
                max_length = 1500,
                truncation=True,
                #pad_to_max_length = True
                padding = True
            ) 

            tokens_tensor = inputs["input_ids"]
            segments_tensor = inputs["token_type_ids"]
            masks_tensor = inputs["attention_mask"]
        
            return {
                'tokens_tensor' : torch.tensor(tokens_tensor, dtype=torch.long),
                'segments_tensor': torch.tensor(segments_tensor, dtype=torch.long),
                'masks_tensor' : torch.tensor(masks_tensor, dtype=torch.long),
                'target' : torch.tensor(label, dtype=torch.float)
            }
        else:
            num1 = self.pairingTable[index][0]
            num2 = self.pairingTable[index][1]
            with open('../../public_processed_test_files/' + str(num1) + '.txt', 'r', encoding='UTF-8') as text1:
                file1 = text1.read()
            with open('../../public_processed_test_files/' + str(num2) + '.txt', 'r', encoding='UTF-8') as text2:
                file2 = text2.read()    

            inputs = self.tokenizer.encode_plus(
                file1,
                file2,
                add_special_tokens = True,
                max_length = 1500,
                truncation=True,
                #pad_to_max_length = True
                padding = True
            )

            tokens_tensor = inputs["input_ids"]
            segments_tensor = inputs["token_type_ids"]
            masks_tensor = inputs["attention_mask"]                    

            return {
                'tokens_tensor' : torch.tensor(tokens_tensor, dtype=torch.long),
                'segments_tensor': torch.tensor(segments_tensor, dtype=torch.long),
                'masks_tensor' : torch.tensor(masks_tensor, dtype=torch.long)
            }  

    def __len__(self):
        return self.len


'''
trainset = ArticleClassificationDataset('train', 1)
token_dict = trainset[0]

tokens = config.tokenizer.convert_ids_to_tokens(token_dict['tokens_tensor'].tolist())
combined_text = "".join(tokens)

with open('../table/table'+ str(1) +'.txt', 'rb') as fp:
    table = pickle.load(fp)

with open('../processed_files/' + str(table[0][0]) + '.txt', 'r', encoding='UTF-8') as text1:
    file1 = text1.read()
with open('../processed_files/' + str(table[0][1]) + '.txt', 'r', encoding='UTF-8') as text2:
    file2 = text2.read()

# 渲染前後差異，毫無反應就是個 print。可以直接看輸出結果
print(f"""[原始文本]
句子 1：{file1}
句子 2：{file2}

--------------------

[Dataset 回傳的 tensors]
tokens_tensor  ：{token_dict['tokens_tensor']}

segments_tensor：{token_dict['segments_tensor']}

masks_tensor : {token_dict['masks_tensor']}

label_tensor : {token_dict['target']}
--------------------

[還原 tokens_tensors]
{combined_text}
""")

# trainset = ArticleClassificationDataset('train', 1)
# token_dict = trainset[0]

print(f"length of seg: {len(token_dict['segments_tensor'])}")
print(f"length of mask: {len(token_dict['masks_tensor'])}")

token_dict = trainset[1]

print(f"length of seg: {len(token_dict['segments_tensor'])}")
print(f"length of mask: {len(token_dict['masks_tensor'])}")
'''

# trainset = ArticleClassificationDataset('train', 1)
# train_data_loader = DataLoader(trainset, batch_size=config.BATCH_SIZE)
# data = next(iter(train_data_loader))

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
# label_ids.shape        = {data['target'].shape}
# {data['target']}
# """)