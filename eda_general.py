import re
import math
from collections import Counter, defaultdict
import numpy as np

# ==========================
# CONFIG
# ==========================
TOKEN_REGEX = re.compile(
    r"https?://\S+|www\.\S+|[\w]+\.[\w]+|[\w]+['’`]\w+|@\w+|#\w+|\w+|[^\s\w]+"
)

END_PUNCS = {".", "!", "?", "..."}


# ==========================
# UTILS
# ==========================
def tokenize(text):
    return TOKEN_REGEX.findall(text.lower())


def entropy(counter):
    total = sum(counter.values())
    if total == 0:
        return 0
    return -sum((c / total) * math.log2(c / total) for c in counter.values())


# ==========================
# MAIN EDA FUNCTION
# ==========================
def eda_language_generation(path, name="DATASET"):
    print(f"\n{'='*60}")
    print(f"EDA FOR {name}")
    print(f"{'='*60}")

    sentences = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line:
                sentences.append(line)

    print(f"\nTotal sentences: {len(sentences)}")

    tokenized = [tokenize(s) for s in sentences if tokenize(s)]

    # ==========================
    # 1. Sentence Length
    # ==========================
    lengths = [len(toks) for toks in tokenized]
    print("\n--- Sentence Length ---")
    print(f"Min: {min(lengths)}")
    print(f"Max: {max(lengths)}")
    print(f"Mean: {np.mean(lengths):.2f}")
    print(f"Median: {np.median(lengths)}")
    print(
        "Suggested length range:",
        f"{np.percentile(lengths, 20):.0f} - {np.percentile(lengths, 85):.0f}",
    )

    # ==========================
    # 2. Punctuation Profile
    # ==========================
    punc_counter = Counter()
    for s in sentences:
        s = s.strip()
        if s.endswith("..."):
            punc_counter["..."] += 1
        elif s and s[-1] in END_PUNCS:
            punc_counter[s[-1]] += 1

    print("\n--- Sentence Ending Punctuation ---")
    for k, v in punc_counter.most_common():
        print(f"{k}: {v} ({v/len(sentences)*100:.1f}%)")

    # ==========================
    # 3. Sentence Starters (First 1–2 tokens)
    # ==========================
    starter_1 = Counter()
    starter_2 = Counter()

    for toks in tokenized:
        if len(toks) >= 1:
            starter_1[toks[0]] += 1
        if len(toks) >= 2:
            starter_2[" ".join(toks[:2])] += 1

    print("\n--- Top Sentence Starters (1 token) ---")
    for w, c in starter_1.most_common(10):
        print(f"{w}: {c}")

    print("\n--- Top Sentence Starters (2 tokens) ---")
    for w, c in starter_2.most_common(10):
        print(f"{w}: {c}")

    # ==========================
    # 4. N-gram Stats
    # ==========================
    unigram = Counter()
    bigram = Counter()
    trigram = Counter()

    for toks in tokenized:
        unigram.update(toks)
        for i in range(len(toks) - 1):
            bigram[(toks[i], toks[i + 1])] += 1
        for i in range(len(toks) - 2):
            trigram[(toks[i], toks[i + 1], toks[i + 2])] += 1

    print("\n--- Top Unigrams ---")
    for w, c in unigram.most_common(15):
        print(w, c)

    print("\n--- Top Bigrams ---")
    for (w1, w2), c in bigram.most_common(10):
        print(f"{w1} {w2}: {c}")

    print("\n--- Top Trigrams ---")
    for (w1, w2, w3), c in trigram.most_common(10):
        print(f"{w1} {w2} {w3}: {c}")

    # ==========================
    # 5. Entropy (Style diversity)
    # ==========================
    print("\n--- Entropy ---")
    print(f"Unigram entropy: {entropy(unigram):.2f}")
    print(f"Bigram entropy: {entropy(bigram):.2f}")
    print(f"Trigram entropy: {entropy(trigram):.2f}")

    print("\n[Interpretation]")
    print("- Low entropy → repetitive / rigid")
    print("- High entropy → noisy / incoherent")
    print("- Goal: medium entropy (style + fluency)")


# ==========================
# RUN
# ==========================
if __name__ == "__main__":
    eda_language_generation("1.txt", "LABEL 1 (NEGATIVE / COMPLAINT)")
    eda_language_generation("2.txt", "LABEL 2 (POSITIVE / PRAISE)")
