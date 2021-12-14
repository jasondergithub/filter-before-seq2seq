import pickle
import pandas as pd
import torch
import config

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class TestDataset(Dataset):
    def __init__(self, mode, tableNumber=1):
        assert mode in ['train', 'test']
        self.mode = mode
        if self.mode == 'train':
            with open('../table/table'+ str(tableNumber) +'.txt', 'rb') as fp:
                table = pickle.load(fp)
        else:
            with open('../dict/private_test.txt', 'rb') as fp:
                table = pickle.load(fp)
        self.pairingTable =table 
        self.len = len(table)
        self.tokenizer = config.tokenizer
    
    def __getitem__(self, index):
        if self.mode == 'train':
            num1 = self.pairingTable[index][0]
            label = self.pairingTable[index][2]

            with open('../processed_files/' + str(num1) + '.txt', 'r', encoding='UTF-8') as text1:
                file1 = text1.read()

            inputs = self.tokenizer.encode_plus(
                file1,
                None,
                add_special_tokens = True,
                max_length = 1500,
                truncation=True,
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

            with open('../private_processed_test_files/' + str(num1) + '.txt', 'r', encoding='UTF-8') as text1:
                file1 = text1.read()    

            inputs = self.tokenizer.encode_plus(
                file1,
                None,
                add_special_tokens = True,
                max_length = 1500,
                truncation=True,
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


# trainset =  TestDataset('train', 1)
# token_dict = trainset[0]

# tokens = config.tokenizer.convert_ids_to_tokens(token_dict['tokens_tensor'].tolist())
# combined_text = "".join(tokens)

# with open('../table/table'+ str(1) +'.txt', 'rb') as fp:
#     table = pickle.load(fp)

# print('test number: ', table[0][0])
# with open('../processed_files/' + str(table[0][1]) + '.txt', 'r', encoding='UTF-8') as text2:
#     file2 = text2.read()

# # 渲染前後差異，毫無反應就是個 print。可以直接看輸出結果
# print(f"""[原始文本]
# 句子 2：{file2}

# --------------------

# [Dataset 回傳的 tensors]
# tokens_tensor  ：{token_dict['tokens_tensor']}

# segments_tensor：{token_dict['segments_tensor']}

# masks_tensor : {token_dict['masks_tensor']}

# labels_tensor : {token_dict['target']}

# --------------------

# [還原 tokens_tensors]
# {combined_text}
# """)