# Import relevant libraries
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
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

# Mini-batches
batch_size = 16  # number of samples to compute forward and backward pass for in a mini-batch
context_length = 8  # maximum context length the transformer

dataloader = dg.MiniBatchLoader(block_size=context_length, batch_size=batch_size, shuffle=True)
xb, yb = dataloader.get_batch(train_data)


# --
# Model
# --

# Model load
baseline_model = model.BigramLanguageModel(tokenizer.vocab_len, embedding_size=128)

# Model prediction and performance assessment (untrained model)
pred = baseline_model(xb)

loss = baseline_model.loss(pred, yb)
print(f"Cross-entropy loss for next token prediction, randomly initialised bigram model: {loss}")
print(f"Expected loss with vocab size of {tokenizer.vocab_len} is -log(1/{tokenizer.vocab_len})= {torch.log(torch.tensor(tokenizer.vocab_len))}")

# Generative model performance
generated_text = baseline_model.generate(xb, 100)  # next ten tokens generated for xb batch of data
print(f"Example next 100 token generation for:\n{''.join(tokenizer.token_decoder(generated_text[3].tolist()))}")


# --
# Training
# --
num_iterations = 10000  # number of passes to perform (as mini-batches randomly generated, cannot cycle through all data)
lr = 1e-2  # for smaller models can afford for learning rates to be higher (also using momentum optimizers)

# Optimizer and loss
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(baseline_model.parameters(), lr=lr)  # parameters attribute may be inherited from nn.module
loss_tracker = []

for step in range(num_iterations):
    # Load batch of data
    xb, yb = dataloader.get_batch(train_data)

    # Forward pass through network and compute loss
    pred = baseline_model(xb)
    loss = baseline_model.loss(pred, yb)

    # Backwards pass - backpropagation and zero grading
    optimizer.zero_grad(set_to_none=True)  # grads don't accumulate + set them to None instead of 0 to save on memory
    loss.backward()  # calculates dl/dw for all weights (and trainable params)

    # Gradient descent weight update
    optimizer.step()  # note optimizer object has access to model parameters that are trainable + learning rate

    loss_tracker.append(loss.item())

# Plot loss against number of iterations
plt.plot(loss_tracker)

# Generate text with a trained model
generated_text = baseline_model.generate(xb, 300)  # next ten tokens generated for xb batch of data
print(f"Example next 300 token generation for:\n{''.join(tokenizer.token_decoder(generated_text[3].tolist()))}")
