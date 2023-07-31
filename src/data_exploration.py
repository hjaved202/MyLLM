# Import relevant libraries
from src import dataset_gen as dg
from pathlib import Path

# Read in raw text file
filepath = Path().cwd().parent.joinpath('data/tiny-shakespeare.txt')
raw_data = dg.TextDatasetLoad(filepath=filepath).load_textfile()

# Corpus length
print(f"Corpus length: {len(raw_data)} characters\n")
print("---")
print("First 500 characters")
print(raw_data[:500])

# Character-level tokenisation
vocab = sorted(list(set(raw_data)))  # character level vocabulary
print(f"Vocabulary:\n{''.join(vocab)}")
print(f"\nVocab size:{len(vocab)}")

# Tokenisation
tokenizer = dg.TextTokenizer(raw_data)
print(raw_data[0:33])
print(tokenizer.token_encoder(raw_data[0:33]))
print(''.join(tokenizer.token_decoder(tokenizer.token_encoder(raw_data[0:33]))))

# Dealing with out-of-vocab examples
print(f"\nFor out of vocabulary elements, e.g. '{'*, (, )'}', the encoding is: {tokenizer.token_encoder('*)(')}\n")

# Setting up prediction problem
context_length = 8  # how many preceding tokens used to predict next token (also known as block length)
encoded_data = tokenizer.token_encoder(raw_data[0:33])
x = encoded_data[0:context_length]  # example training sequence
y = encoded_data[1:context_length+1]  # corresponding target sequence

# Context length
for t in range(context_length):
    print(f"When the input sequence is: {encoded_data[0:t+1]}, the target is: {encoded_data[t+1]}")
