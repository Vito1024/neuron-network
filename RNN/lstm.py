import torch
from torch import nn
from torch.nn import functional as F

# LSTM
# 记忆单元和三个门控结构（输入门、遗忘门、输出门
# Reference: https://zh.d2l.ai/chapter_recurrent-modern/lstm.html


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
num_hiddens = 2**5
num_inputs = num_outputs = vocab_size

# 输入门参数
W_xi = torch.randn(num_inputs, num_hiddens)
W_hi = torch.randn(num_hiddens, num_hiddens)
b_i = torch.zeros(num_hiddens)

# 遗忘门参数
W_xf = torch.randn(num_inputs, num_hiddens)
W_hf = torch.randn(num_hiddens, num_hiddens)
b_f = torch.zeros(num_hiddens)

# 输出门参数
W_xo = torch.randn(num_inputs, num_hiddens)
W_ho = torch.randn(num_hiddens, num_hiddens)
b_o = torch.zeros(num_hiddens)

# 候选记忆单元
W_xc = torch.randn(num_inputs, num_hiddens)
W_hc = torch.randn(num_hiddens, num_hiddens)
b_c = torch.zeros(num_hiddens)

# 输出层参数
W_hq = torch.randn(num_hiddens, num_outputs)
b_q = torch.zeros(num_outputs)

H = torch.randn(1, num_hiddens)
C = torch.randn(1, num_hiddens)
C_tilda = torch.randn(1, num_hiddens) # 候选记忆

inputs = F.one_hot(data, num_classes=vocab_size).type(torch.float32)
outputs = []
# lstm
for X in inputs:
    # X: (1, vocab_size)
    I = torch.sigmoid((X @ W_xi) + (H @ W_hi) + b_i) # (1, num_hiddens)
    F = torch.sigmoid((X @ W_xf) + (H @ W_hf) + b_f) # (1, num_hiddens)
    O = torch.sigmoid((X @ W_xo) + (H @ W_ho) + b_c) # (1, num_hiddens)
    C = F * C + I * C_tilda
    H = O * torch.tanh(C)
    Y = (H @ W_hq) + b_q
    outputs.append(Y)

print(torch.cat(outputs, dim=0))
