# Import relevant libraries
import torch
from src import dataset_gen as dg
from pathlib import Path

# Check if Apple Silicon available
device = torch.device('mps') if torch.has_mps else 'cpu'

# Read in file
train_data, val_data, test_data = dg.load_tiny_shakespeare()
filepath = Path().cwd().parent.joinpath('data/tiny-shakespeare.txt')
# data = dg.load_textfile_dataset(filepath)

# Tokenisation
tokenizer = dg.TextTokenizer(train_data)  # extract vocab and corresponding encodings from train dataset only

# Train-val-test data split
train_data = torch.tensor(tokenizer.token_encoder(train_data['text'][0]), device=device)
val_data = torch.tensor(tokenizer.token_encoder(val_data['text'][0]), device=device)
test_data = torch.tensor(tokenizer.token_encoder(test_data['text'][0]), device=device)

# Mini-batches
context_length = 8  # maximum context length the transformer
