import random, datetime, torch, torch.nn as nn
random.seed(0); torch.manual_seed(0)

# ---------- 1. Data: messy human dates -> ISO format (a real data-cleaning task) ----------
MONTHS = ["january", "february", "march", "april", "may", "june", "july",
          "august", "september", "october", "november", "december"]
def messy(d):
    m, m3 = MONTHS[d.month - 1], MONTHS[d.month - 1][:3]
    suf = "th" if 11 <= d.day <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(d.day % 10, "th")
    return random.choice([
        f"{d.day} {m3} {d.year}", f"{d.day} {m} {d.year}", f"{m3} {d.day}, {d.year}",
        f"{d.day}{suf} {m} {d.year}", f"{d.day:02d}/{d.month:02d}/{d.year}", f"{d.day}-{d.month}-{d.year}",
        f"{m} {d.day} {d.year}", f"{d.day:02d}.{d.month:02d}.{d.year}", f"{d.day}{suf} of {m}, {d.year}",
    ])
def sample():
    d = datetime.date(2000, 1, 1) + datetime.timedelta(days=random.randint(0, 12000))
    return messy(d), d.isoformat()

train = [sample() for _ in range(20000)]
SRC = sorted(set("".join(s for s, _ in train))) ; TGT = list("0123456789-") + ["<s>", "</s>"]
si = {c: i + 1 for i, c in enumerate(SRC)}; ti = {c: i for i, c in enumerate(TGT)}

def enc_src(batch):
    L = max(len(s) for s in batch)
    return torch.tensor([[si.get(c, 0) for c in s] + [0] * (L - len(s)) for s in batch])
def enc_tgt(batch):
    return torch.tensor([[ti["<s>"]] + [ti[c] for c in t] + [ti["</s>"]] for t in batch])

# ---------- 2. Model: GRU encoder + GRU decoder with (Luong-style) attention ----------
class Seq2Seq(nn.Module):
    def __init__(self, H=128, attention=True):
        super().__init__()
        self.attention = attention
        self.src_emb = nn.Embedding(len(SRC) + 1, 32, padding_idx=0)
        self.tgt_emb = nn.Embedding(len(TGT), 32)
        self.encoder = nn.GRU(32, H, batch_first=True)
        self.decoder = nn.GRU(32, H, batch_first=True)
        self.out = nn.Linear(2 * H if attention else H, len(TGT))

    def forward(self, src, tgt_in):
        enc_out, h = self.encoder(self.src_emb(src))                 # enc_out: one vector per source char
        dec_out, _ = self.decoder(self.tgt_emb(tgt_in), h)          # decoder starts from the encoder summary
        if not self.attention:
            return self.out(dec_out), None
        scores = dec_out @ enc_out.transpose(1, 2)                   # each output step scores every input char
        scores = scores.masked_fill((src == 0).unsqueeze(1), -1e9)   # ignore padding
        weights = scores.softmax(-1)                                 # attention weights
        context = weights @ enc_out                                  # weighted mix of input chars
        return self.out(torch.cat([dec_out, context], -1)), weights

    @torch.no_grad()
    def translate(self, text):
        src, y, out = enc_src([text]), torch.tensor([[ti["<s>"]]]), ""
        for _ in range(12):                                          # greedy decoding, one char at a time
            logits, w = self(src, y)
            nxt = logits[0, -1].argmax().item()
            if TGT[nxt] == "</s>": break
            out += TGT[nxt]; y = torch.cat([y, torch.tensor([[nxt]])], 1)
        return out, w

def train_model(attention, epochs=3):
    model = Seq2Seq(attention=attention); opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    for epoch in range(epochs):
        random.shuffle(train)
        for i in range(0, len(train), 64):
            b = train[i:i + 64]
            src, tgt = enc_src([s for s, _ in b]), enc_tgt([t for _, t in b])
            logits, _ = model(src, tgt[:, :-1])                      # teacher forcing
            loss = nn.functional.cross_entropy(logits.reshape(-1, len(TGT)), tgt[:, 1:].reshape(-1))
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    return model

test = [sample() for _ in range(1000)]
for attention in [False, True]:
    model = train_model(attention)
    acc = sum(model.translate(s)[0] == t for s, t in test) / len(test)
    print(f"attention={attention!s:<5}  exact-match accuracy on 1,000 new dates = {acc:.1%}")

for text in ["26 sep 2026", "october 2, 2019", "7th of march, 2011", "05.11.2003", "12 sitambar 2026"]:
    print(f"{text!r:<22} -> {model.translate(text)[0]}")
torch.save(model.state_dict(), "dates_attention.pt")
