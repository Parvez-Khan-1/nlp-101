import torch.nn as nn
for name, cls in [("RNN", nn.RNN), ("GRU", nn.GRU), ("LSTM", nn.LSTM)]:
    m = cls(input_size=100, hidden_size=128)
    print(f"{name:<5} parameters = {sum(p.numel() for p in m.parameters()):,}")
