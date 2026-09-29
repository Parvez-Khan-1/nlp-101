import numpy as np, gensim.downloader as api

glove = api.load("glove-wiki-gigaword-50")

def self_attention(words):
    X = np.array([glove[w] for w in words])            # (n_words, 50): one static vector per word
    Q, K, V = X, X, X                                   # a real Transformer learns 3 projection matrices here
    scores = Q @ K.T / np.sqrt(X.shape[1])              # how relevant is each word to each other word
    weights = np.exp(scores) / np.exp(scores).sum(1, keepdims=True)   # softmax per row
    return weights, weights @ V                          # new vector = weighted mix of all words

def cos(a, b): return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))

for sentence in ["i sat on the river bank", "the bank approved my loan"]:
    words = sentence.split()
    weights, out = self_attention(words)
    bank = out[words.index("bank")]                      # "bank" after attending to its context
    top = sorted(zip(words, weights[words.index("bank")]), key=lambda x: -x[1])[:3]
    print(f"{sentence!r}")
    print("   bank attends to:", [(w, round(float(p), 2)) for w, p in top])
    print(f"   similarity to 'water' = {cos(bank, glove['water']):.2f}   to 'money' = {cos(bank, glove['money']):.2f}")
print(f"static GloVe 'bank':  water = {cos(glove['bank'], glove['water']):.2f}   money = {cos(glove['bank'], glove['money']):.2f}")
