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


class TicketCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(len(vocab) + 1, 32, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(32, 32, k, padding=k // 2) for k in (2, 3)])   # 2- and 3-word filters
        self.out = nn.Linear(64, len(labels))
    def forward(self, x):
        e = self.emb(x).transpose(1, 2)                                   # (batch, channels, time)
        pooled = [torch.relu(c(e)).max(dim=2).values for c in self.convs]  # max-pool over time
        return self.out(torch.cat(pooled, dim=1))

Xtr, ytr = encode(X_train), torch.tensor([labels.index(l) for l in y_train])
Xte, yte = encode(X_test), torch.tensor([labels.index(l) for l in y_test])
model = TicketCNN(); opt = torch.optim.Adam(model.parameters(), lr=5e-3)
for epoch in range(60):
    loss = nn.functional.cross_entropy(model(Xtr), ytr)
    opt.zero_grad(); loss.backward(); opt.step()
model.eval()
with torch.no_grad():
    acc = (model(Xte).argmax(1) == yte).float().mean().item()
    t = time.perf_counter()
    for _ in range(100): model(encode(["order kab aayega bahut late"]))
    ms = (time.perf_counter() - t) * 10
print(f"CNN accuracy = {acc:.2f}   latency = {ms:.2f} ms/ticket   parameters = {sum(p.numel() for p in model.parameters()):,}")
