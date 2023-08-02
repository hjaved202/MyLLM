"""
Script contains model definitions
"""

# Import relevant libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(150)  # for reproducibility


class BigramLanguageModel(nn.Module):
    """
    Predict the next token based on only the previous token. Index of current token maps to a learnable embedding that
    can directly be trained to be the classification head/logit scores for predicting the next token
    """
    def __init__(self, vocab_size: int, embedding_size: int):
        super(BigramLanguageModel, self).__init__()  # super calls the init of nn.Module (the parent class)
        # super().__init__()  # equivalent to the above
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.token_embedding_lookup = nn.Embedding(num_embeddings=vocab_size, embedding_dim=vocab_size)  # index of vocab maps to a learnable embedding vector

    def forward(self, idx_sequence):
        # idx and targets are tensors of dimension [batch x sequence]
        logits = self.token_embedding_lookup(idx_sequence)  # scores for next token in the sequence [batch x seq x embedding]
        return logits

    @staticmethod
    def loss(logits, targets):
        """
        Method computes the cross entropy loss between prediction and targets
        """
        B, T, C = logits.shape  # batch x sequence x embedding size
        logits = logits.view(B*T, C)
        targets = targets.view(B*T)
        l = F.cross_entropy(logits, targets)
        return l

    def generate(self, idx_sequence, max_token_gen: int):
        """
        Method generates token sequences based on given input sequence (in case of bigram single token)
        NB - for a bigram model a bit redundant to be inputting in a sequence?
        """
        for _ in range(max_token_gen):
            logits = self.forward(idx_sequence)
            logits = logits[:, -1, :]  # take the next token prediction from last token in the sequence [B x C]
            probs = F.softmax(logits, dim=1)  # logits score per mini-batch sample softmax-ed [B x C]
            # idx_next = torch.argmax(probs, dim=1)  # pick the predicted token with the highest probability
            idx_next = torch.multinomial(probs, num_samples=1)  # sample from probs distribution and return next token idx [B x 1]
            idx_sequence = torch.cat((idx_sequence, idx_next), dim=1)  # append token indices to sequence [B x T+1]

        return idx_sequence
