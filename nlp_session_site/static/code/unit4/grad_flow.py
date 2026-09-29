import torch, torch.nn as nn

torch.manual_seed(0)
T, D, H = 50, 16, 64                        # 50 time steps, 16-dim inputs, 64 hidden units

def lstm_forget_bias(b):
    m = nn.LSTM(D, H)
    for name, p in m.named_parameters():    # PyTorch gate order: input, forget, cell, output
        if "bias" in name:
            p.data[H:2*H] = b / 2           # two bias vectors are added, so split b between them
    return m

def gradient_per_step(cell):
    x = torch.randn(T, 1, D, requires_grad=True)   # one sequence
    out, _ = cell(x)
    out[-1].sum().backward()                       # loss depends only on the LAST output
    return x.grad.norm(dim=(1, 2))                 # how much each input step influenced it

models = {"RNN": lambda: nn.RNN(D, H), "GRU": lambda: nn.GRU(D, H),
          "LSTM": lambda: nn.LSTM(D, H), "LSTM (forget bias 3)": lambda: lstm_forget_bias(3.0)}
curves = {}
for name, make in models.items():
    g = torch.stack([gradient_per_step(make()) for _ in range(20)]).mean(0)   # average of 20 random models
    curves[name] = (g / g[-1]).tolist()            # relative to the most recent word
    print(f"{name:<21} 10 steps back: {curves[name][39]:.1e}   25 back: {curves[name][24]:.1e}   49 back: {curves[name][0]:.1e}")
import json; json.dump(curves, open("grad_curves.json", "w"))
