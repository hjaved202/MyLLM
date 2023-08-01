# Import relevant libraries
import torch
from src import dataset_gen as dg
from pathlib import Path

# Check if Apple Silicon available
device = torch.device('mps') if torch.has_mps else 'cpu'

# Read in file
text_dataset = 'tiny_shakespeare'
text_loader = dg.TextDatasetLoad(hugging_face_dataset=text_dataset)
train_text, val_text, test_text = text_loader.load_huggingface_dataset()
# filepath = Path().cwd().parent.joinpath('data/tiny-shakespeare.txt')
# data = text_loader.load_textfile_dataset(filepath)

# Tokenisation
tokenizer = dg.TextTokenizer(train_text)  # extract vocab and corresponding encodings from train dataset only

# Train-val-test data split
train_data = torch.tensor(tokenizer.token_encoder(train_text['text'][0]), device=device)
val_data = torch.tensor(tokenizer.token_encoder(val_text['text'][0]), device=device)
test_data = torch.tensor(tokenizer.token_encoder(test_text['text'][0]), device=device)

# Mini-batches
context_length = 8  # maximum context length the transformer
