from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

reference = "the food arrived cold and the delivery was late".split()
candidates = {
    "exact copy":         "the food arrived cold and the delivery was late",
    "same meaning":       "the meal came cold and delivery was delayed",
    "word salad":         "the delivery the food was late cold and arrived",
    "wrong meaning":      "the food arrived hot and the delivery was early",
}
smooth = SmoothingFunction().method1
for name, cand in candidates.items():
    score = sentence_bleu([reference], cand.split(), smoothing_function=smooth)
    print(f"{name:<14} BLEU = {score:.2f}   {cand}")
