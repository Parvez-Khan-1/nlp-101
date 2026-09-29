import torch, torch.nn as nn, random
torch.manual_seed(0); random.seed(0)

names = open("dishes.txt").read().split("\n")[:-1]           # ~200 real Indian dish names
chars = sorted(set("".join(names))) + ["<end>"]
idx = {c: i for i, c in enumerate(chars)}

class CharLSTM(nn.Module):
    def __init__(self, vocab, emb=32, hidden=128):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb)
        self.lstm = nn.LSTM(emb, hidden, num_layers=2, batch_first=True, dropout=0.2)
        self.out = nn.Linear(hidden, vocab)
    def forward(self, x, state=None):
        h, state = self.lstm(self.emb(x), state)
        return self.out(h), state

model = CharLSTM(len(chars))
opt = torch.optim.Adam(model.parameters(), lr=3e-3)

def encode(name):
    return [idx[c] for c in name] + [idx["<end>"]]

for step in range(2000):                                     # ~1-2 minutes on a laptop CPU
    seq = torch.tensor([encode(random.choice(names))])
    logits, _ = model(seq[:, :-1])                           # predict each next character
    loss = nn.functional.cross_entropy(logits[0], seq[0, 1:])
    opt.zero_grad(); loss.backward(); opt.step()

@torch.no_grad()
def generate(start, temperature=0.8, max_len=30):
    x, state, out = torch.tensor([[idx[c] for c in start]]), None, start
    logits, state = model(x, state)
    for _ in range(max_len):
        probs = torch.softmax(logits[0, -1] / temperature, dim=0)
        c = torch.multinomial(probs, 1).item()
        if chars[c] == "<end>":
            break
        out += chars[c]
        logits, state = model(torch.tensor([[c]]), state)
    return out

torch.manual_seed(42)
for start in ["p", "ch", "m", "tandoori ", "kashmiri ", "mango ", "schezwan ", "butter "]:
    dish = generate(start, temperature=1.0)
    print(f"{start!r:<12} -> {dish:<28} {'(in training data)' if dish in names else 'NEW'}")
