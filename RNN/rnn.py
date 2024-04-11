import torch
from torch.nn import functional as F

text = '''
    hello world
    ...
'''

chars = sorted(set(text))
stoi = {c:i for i, c in enumerate(chars)}
itos = {i:c for i, c in enumerate(chars)}
def encode(s):
    return [stoi[x] for x in s]
def decode(l):
    return ''.join([itos[i] for i in l])

data = torch.tensor(encode(text))

vocab_size = len(chars)
hidden_size = 2**5

# -------------------------------
# RNN
# Hidden = tanh(X @ W_xh + Hidden @ W_hh + b_h)
# Output = Hidden @ W_o + b_o
# -------------------------------
# Inputs Sequence: (N, vocab_size)
inputs = F.one_hot(data, num_classes=vocab_size).type(torch.float32)
# Hidden Layer:
W_xh = torch.randn(vocab_size, hidden_size)
Hidden = torch.randn(1, hidden_size)
W_hh = torch.randn(hidden_size, hidden_size)
b_h = torch.randn(1, hidden_size)
# Output Layer:
W_o = torch.randn(hidden_size, 1)
b_o = torch.randn(1, 1)

# rnn
output = []
for X in inputs:
    # X: (1, vocab_size)
    X = X.view(1, -1)
    Hidden = (X@W_xh + Hidden@W_hh + b_h) #(1, hidden_size)
    out = Hidden @ W_o + b_o
    output.append(out)

print(torch.cat(output, dim=1)) # result