import torch, torch.nn as nn

rnn = nn.RNN(input_size=50, hidden_size=64, batch_first=True)   # e.g. 50-d GloVe vectors in
x = torch.randn(8, 12, 50)            # batch of 8 tickets, 12 words each, 50 numbers per word
outputs, h_last = rnn(x)

print("outputs:", tuple(outputs.shape))   # one hidden state per word
print("h_last :", tuple(h_last.shape))    # final summary of each ticket
print("weights:", {n: tuple(p.shape) for n, p in rnn.named_parameters()})
