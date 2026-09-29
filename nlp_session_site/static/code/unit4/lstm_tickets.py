import torch, torch.nn as nn, pandas as pd, time
from sklearn.model_selection import train_test_split
torch.manual_seed(0)

df = pd.read_csv("support_tickets.csv")
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.25, stratify=df["label"], random_state=42)   # same split as Unit 2

vocab = {w: i + 1 for i, w in enumerate(sorted({w for t in X_train for w in t.split()}))}  # 0 = padding / unknown
labels = sorted(df["label"].unique())
def encode(texts):
    seqs = [[vocab.get(w, 0) for w in t.split()] for t in texts]
    L = max(map(len, seqs))
    return torch.tensor([s + [0] * (L - len(s)) for s in seqs])

class TicketLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(len(vocab) + 1, 32, padding_idx=0)
        self.lstm = nn.LSTM(32, 64, batch_first=True, bidirectional=True)
        self.out = nn.Linear(128, len(labels))
    def forward(self, x):
        h, _ = self.lstm(self.emb(x))
        return self.out(h.max(dim=1).values)          # max-pool over time

Xtr, ytr = encode(X_train), torch.tensor([labels.index(l) for l in y_train])
Xte, yte = encode(X_test), torch.tensor([labels.index(l) for l in y_test])
model = TicketLSTM(); opt = torch.optim.Adam(model.parameters(), lr=5e-3)
for epoch in range(60):
    loss = nn.functional.cross_entropy(model(Xtr), ytr)
    opt.zero_grad(); loss.backward(); opt.step()

model.eval()
with torch.no_grad():
    acc = (model(Xte).argmax(1) == yte).float().mean().item()
    t = time.perf_counter()
    for _ in range(100): model(encode(["order kab aayega bahut late"]))
    ms = (time.perf_counter() - t) * 10
print(f"BiLSTM accuracy = {acc:.2f}   latency = {ms:.2f} ms/ticket   parameters = {sum(p.numel() for p in model.parameters()):,}")
