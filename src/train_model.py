# Import relevant libraries
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader  # PyTorch dataset and loader classes
from src import dataset_gen as dg
from src import bigram_model as model

# Check if Apple Silicon available
# device = torch.device('mps') if torch.has_mps else 'cpu'  # TODO: figure out how to use Apple mps
device = 'cpu'
torch.manual_seed(150)  # set random seed for reproducibility


# --
# Data
# --

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

# Problem setup
context_length = 8  # maximum context length to consider in prediction

# Mini-batches
# batch_size = 16  # number of samples to compute forward and backward pass for in a mini-batch
# context_length = 8  # maximum context length the transformer
# dataloader = dg.MiniBatchLoader(block_size=context_length, batch_size=batch_size, shuffle=True)
# xb, yb = dataloader.get_batch(train_data)

train_dataset = dg.TextSequenceDataset(train_data, context_length)
val_dataset = dg.TextSequenceDataset(val_data, context_length)
x_samp, y_samp = train_dataset[15]  # example of an input and corresponding target sequence


# --
# Model
# --

# Model load
baseline_model = model.BigramLanguageModel(tokenizer.vocab_len, embedding_size=128).to(device)

# Model prediction and performance assessment (untrained model)
xb = x_samp.unsqueeze(dim=0)  # model designed to take in batched data
yb = y_samp.unsqueeze(dim=0)

pred = baseline_model(xb)

loss = baseline_model.loss(pred, yb)
print(f"Cross-entropy loss for next token prediction, randomly initialised bigram model: {loss}")
print(f"Expected loss with vocab size of {tokenizer.vocab_len} is -log(1/{tokenizer.vocab_len})= {torch.log(torch.tensor(tokenizer.vocab_len))}")

# Generative model performance
generated_text = baseline_model.generate(xb, 100)  # next ten tokens generated for xb batch of data
print(f"Example next 100 token generation for:\n{''.join(tokenizer.token_decoder(generated_text[0].tolist()))}")


# --
# Training
# --

num_epochs = 1  # number of passes to perform (as mini-batches randomly generated, cannot cycle through all data)
batch_size = 32
lr = 1e-2  # for smaller models can afford for learning rates to be higher (also using momentum optimizers)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size, shuffle=True, drop_last=True)  # ensures each batch size will be the same size by dropping the last batch if it is smaller
val_loader = DataLoader(val_dataset, batch_size, shuffle=True, drop_last=True)  # will loop through batches to test the trained model

# Optimizer and loss
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(baseline_model.parameters(), lr=lr)  # parameters attribute inherited from nn.module
loss_tracker = []
running_loss = 0  # to calculate mean loss across a few mini-batches

for epoch in range(num_epochs):
    for batch_idx, (data, labels) in enumerate(train_loader):

        # Forward pass through network and compute loss
        pred = baseline_model(data)
        loss = baseline_model.loss(pred, labels)

        # Backwards pass - backpropagation and zero grading
        optimizer.zero_grad(set_to_none=True)  # grads don't accumulate + set them to None not 0 to save on memory
        loss.backward()  # calculates dl/dw for all weights (and trainable params)

        # Gradient descent weight update
        optimizer.step()  # note optimizer object has access to model parameters that are trainable + learning rate

        loss_tracker.append(loss.item())

        if batch_idx % 100 == 0:
            print(f"Progress: {100*batch_idx/(len(train_dataset)/batch_size):.2f}%  Loss: {loss.item():.2f}")

# Plot loss against number of iterations
plt.plot(loss_tracker)

# Generate text with a trained model
generated_text = baseline_model.generate(xb, 300)  # next ten tokens generated for xb batch of data
print(f"Example next 300 token generation for:\n{''.join(tokenizer.token_decoder(generated_text[0].tolist()))}")
