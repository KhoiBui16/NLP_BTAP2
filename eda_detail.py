import re
from collections import Counter, defaultdict

# ==============================================================================
# 1. CẤU HÌNH (GIỐNG HỆT FILE MODEL.PY ĐỂ ĐỒNG BỘ)
# ==============================================================================
TOKEN_REGEX = re.compile(r"[\w]+['’`]\w+|[\w]+|[^\s\w]")
CLEAN_REGEX = re.compile(r"\]+\]")

STOPWORDS = {
    "the",
    "to",
    "and",
    "a",
    "of",
    "is",
    "in",
    "it",
    "for",
    "my",
    "i",
    "you",
    "that",
    "with",
    "on",
    "this",
    "app",
    "be",
    "so",
    "have",
    "but",
    "not",
    "are",
    "just",
    "at",
    "was",
    "me",
    "up",
    "all",
    "day",
    "out",
    "if",
    "can",
    "get",
    "like",
    "do",
    "no",
    "we",
    "from",
    "go",
    "about",
    "an",
    "one",
    "what",
    "now",
    "will",
    "your",
    "or",
    "as",
    "time",
    "when",
    "u",
}


# ==============================================================================
# 2. CÁC HÀM XỬ LÝ
# ==============================================================================
def load_and_clean(filepath):
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        # Đọc từng dòng, clean rác, và tokenize y hệt model chính
        sentences = []
        all_tokens = []
        for line in f:
            clean_line = CLEAN_REGEX.sub("", line.strip().lower())
            if not clean_line:
                continue
            tokens = TOKEN_REGEX.findall(clean_line)
            if tokens:
                sentences.append(tokens)
                all_tokens.extend(tokens)
        return sentences, all_tokens


def get_optimized_lengths(sentences_1, sentences_2):
    """Tìm khoảng độ dài phổ biến nhất (Median range)"""
    lens = [len(s) for s in sentences_1 + sentences_2]
    c = Counter(lens)

    # Lọc bỏ các độ dài quá ngắn (<5) hoặc quá dài (>30)
    valid_lens = {k: v for k, v in c.items() if 10 <= k <= 25}

    # Lấy Top 10 độ dài phổ biến nhất
    top_lens = sorted([k for k, v in Counter(valid_lens).most_common(10)])

    # Cắt bớt phần đuôi quá dài để tập trung vào vùng an toàn (như model 0.973)
    # Lấy khoảng giữa tập trung nhất
    final_opts = [l for l in top_lens if l <= 20]
    return final_opts


def get_best_starters(sentences, top_n=7):
    """Tìm 2 từ đầu câu phổ biến nhất"""
    starters = []
    for s in sentences:
        if len(s) >= 2:
            # Lưu dưới dạng tuple để hash được trong Counter
            starters.append(tuple(s[:2]))

    # Đếm tần suất
    counts = Counter(starters)
    total = sum(counts.values())

    result = []
    # Lấy top N phổ biến nhất
    for phrase, count in counts.most_common(top_n):
        # Tính trọng số dựa trên độ phổ biến, làm tròn đẹp
        # Logic: Càng phổ biến trọng số càng cao (0.25, 0.2, 0.15...)
        raw_prob = count / total
        weight = 0.10  # Mặc định
        if raw_prob > 0.05:
            weight = 0.25
        elif raw_prob > 0.03:
            weight = 0.20
        elif raw_prob > 0.01:
            weight = 0.15

        result.append((list(phrase), weight))

    # Luôn thêm fallback <START>
    result.append((["<START>", "<START>"], 0.10))
    return result


def get_distinctive_keywords(target_tokens, other_tokens, limit=35):
    """
    Tìm từ khóa ĐẶC TRƯNG: Xuất hiện nhiều bên này nhưng ÍT bên kia.
    Công thức: Score = (Tần suất bên Target) - (Tần suất bên Other * Hệ số phạt)
    """
    c_target = Counter(target_tokens)
    c_other = Counter(other_tokens)

    scores = {}

    for word, freq in c_target.items():
        # Bỏ qua từ ngắn, stopwords, hoặc từ chứa ký tự lạ
        if len(word) < 3 or word in STOPWORDS:
            continue
        if not re.match(r"^[a-z']+$", word):
            continue  # Chỉ lấy chữ cái

        freq_other = c_other.get(word, 0)

        # Chiến thuật: Phạt cực nặng nếu từ đó xuất hiện ở file kia
        # Để đảm bảo từ khóa là "Độc quyền" (Exclusive)
        score = freq - (freq_other * 10)

        if score > 0:
            scores[word] = score

    # Lấy top từ có điểm cao nhất
    top_keywords = [k for k, v in Counter(scores).most_common(limit)]
    return set(top_keywords)


# ==============================================================================
# 3. MAIN RUN
# ==============================================================================
def main():
    print("Đang phân tích dữ liệu 1.txt và 2.txt...")
    sents1, toks1 = load_and_clean("1.txt")
    sents2, toks2 = load_and_clean("2.txt")

    print("\n" + "=" * 60)
    print("KẾT QUẢ EDA TỐI ƯU (COPY VÀO MODEL.PY)")
    print("=" * 60)

    # 1. LENGTHS
    lengths = get_optimized_lengths(sents1, sents2)
    print(
        f"\n# TẬP TRUNG VÀO ĐỘ DÀI CHUẨN ({min(lengths)}-{max(lengths)}) sau khi eda data"
    )
    print(f"LENGTH_OPTS = {lengths}")

    # 2. STARTERS
    print("\n# STARTERS: DATA-DRIVEN CHUẨN theo phân bố thực tế từ eda")
    print("STARTERS_LM1 = [")
    for s, w in get_best_starters(sents1):
        print(f"    ({s}, {w}),")
    print("]")

    print("\nSTARTERS_LM2 = [")
    for s, w in get_best_starters(sents2):
        print(f"    ({s}, {w}),")
    print("]")

    # 3. KEYWORDS
    kw1 = get_distinctive_keywords(toks1, toks2)
    kw2 = get_distinctive_keywords(toks2, toks1)

    print("\n# KEYWORDS (BỘ DATA-DRIVEN XỊN)")
    # Format in đẹp để copy
    print("KEYWORDS_LM1 = {")
    sorted_kw1 = sorted(list(kw1))
    for i in range(0, len(sorted_kw1), 8):
        print("    " + ", ".join([f'"{w}"' for w in sorted_kw1[i : i + 8]]) + ",")
    print("}")

    print("\nKEYWORDS_LM2 = {")
    sorted_kw2 = sorted(list(kw2))
    for i in range(0, len(sorted_kw2), 8):
        print("    " + ", ".join([f'"{w}"' for w in sorted_kw2[i : i + 8]]) + ",")
    print("}")


if __name__ == "__main__":
    main()
