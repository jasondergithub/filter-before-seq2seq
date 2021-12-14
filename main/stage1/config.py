from transformers import BertConfig, BertTokenizer
from transformers.models import bert

DEVICE = "cuda"
EPOCHS = 2
BATCH_SIZE = 16
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
MODEL_PATH = "../../dict/saved_model.pth"

bert_config = BertConfig.from_pretrained('bert-base-chinese')
bert_config.num_labels = 1
bert_config.max_position_embeddings = 1500
bert_config.num_hidden_layers = 6
bert_config.num_attention_heads = 4
bert_config.output_hidden_states=True