# Import relevant libraries
from src import dataset_gen as dg
from pathlib import Path

# Read in raw text file
filepath = Path().cwd().parent.joinpath('data/tiny-shakespeare.txt')
raw_data = dg.load_textfile_dataset(filepath)

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
