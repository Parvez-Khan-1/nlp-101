import torch, torch.nn as nn
torch.manual_seed(0)

def batch(n, T):
    """Sequences of random digits 0-9. The label is the FIRST digit, seen T steps ago."""
    x = torch.randint(0, 10, (n, T))
    return x, x[:, 0]

def lstm_forget_bias(b):
    def make(inp, hid, batch_first=True):
        m = nn.LSTM(inp, hid, batch_first=batch_first)
        for name, p in m.named_parameters():      # gate order: input, forget, cell, output
            if "bias" in name:
                p.data[hid:2*hid] = b / 2
        return m
    return make

class Model(nn.Module):
    def __init__(self, cell):
        super().__init__()
        self.emb = nn.Embedding(10, 16)
        self.rnn = cell(16, 64, batch_first=True)
        self.out = nn.Linear(64, 10)
    def forward(self, x):
        h, _ = self.rnn(self.emb(x))
        return self.out(h[:, -1])                   # predict from the final hidden state only

def train_and_test(cell, T, steps=3000):
    m = Model(cell); opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    for _ in range(steps):
        x, y = batch(64, T)
        loss = nn.functional.cross_entropy(m(x), y)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(m.parameters(), 1.0)   # gradient clipping: standard practice
        opt.step()
    x, y = batch(2000, T)
    return (m(x).argmax(1) == y).float().mean().item()

cells = {"RNN": nn.RNN, "GRU": nn.GRU, "LSTM": nn.LSTM, "LSTM+bias": lstm_forget_bias(3.0)}
for T in [5, 20, 50]:
    scores = "   ".join(f"{name} {train_and_test(c, T):>4.0%}" for name, c in cells.items())
    print(f"first digit, {T:>2} steps back:  {scores}")
