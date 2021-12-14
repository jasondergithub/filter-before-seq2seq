import config
import torch
import torch.nn as nn

from transformers import BertModel

class Encoder(nn.Module):
    def __init__(self):
        
        super(Encoder, self).__init__()
        self.bert = BertModel(config.bert_config)
        self.linear = nn.Linear(768, 512)
        self.relu = nn.ReLU(True)
    
    def forward(self, tokens_tensors, segments_tensors, masks_tensors):
        
        outputs = self.bert(input_ids = tokens_tensors, token_type_ids = segments_tensors, 
            attention_mask = masks_tensors)
        
        last_hidden_states = outputs.last_hidden_state
        l_list = [torch.reshape(last_hidden_states[i][0], (1, 1, -1)) for i in range(len(last_hidden_states))] 
        h_cls = torch.cat(l_list, dim=0) #only takes CLS as repr. which is [batch, 1, 768]

        out = self.linear(h_cls)
        out = self.relu(out)

        return out

class Decoder(nn.Module):
    def __init__(self):

        super(Decoder, self).__init__()
        self.bert = BertModel(config.bert_config)
        self.linear1 = nn.Linear(768, 512)
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=512, nhead=2, batch_first=True), num_layers=4)
        self.linear2 = nn.Linear(512, 128)
        self.linear3 = nn.Linear(128, 1)
        self.drop = nn.Dropout(0.3)
        self.relu = nn.ReLU(True)

    def forward(self, tokens_tensors, segments_tensors, masks_tensors, hidden):
        
        outputs = self.bert(input_ids = tokens_tensors, token_type_ids = segments_tensors, 
            attention_mask = masks_tensors)
                
        last_hidden_states = outputs.last_hidden_state
        l_list = [torch.reshape(last_hidden_states[i][0], (1, 1, -1)) for i in range(len(last_hidden_states))] 
        h_cls = torch.cat(l_list, dim=0) #only takes CLS as repr. which is [batch, 1, 768]

        out = self.linear1(h_cls)
        out = self.relu(out)

        decoderOutput = self.transformer_decoder(out, hidden)
        decoderOutput = torch.squeeze(decoderOutput, 1) # [batch, 768]
        
        decoderOutput= self.relu(decoderOutput)
        decoderOutput= self.drop(decoderOutput)

        decoderOutput= self.linear2(decoderOutput)
        decoderOutput= self.relu(decoderOutput)
        decoderOutput= self.linear3(decoderOutput)

        return decoderOutput

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, reference_data, trg_data):

        hidden_output = self.encoder(reference_data[0], reference_data[1], reference_data[2])
        output = self.decoder(trg_data[0], trg_data[1], trg_data[2], hidden_output)
        
        return output