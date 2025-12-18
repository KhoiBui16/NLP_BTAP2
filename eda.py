import re
from collections import Counter

# 1. Đọc file
with open("1.txt", "r", encoding="utf-8") as f:
    text1 = f.read().lower()
with open("2.txt", "r", encoding="utf-8") as f:
    text2 = f.read().lower()

# 2. Tokenize (Dùng đúng Regex của model final)
TOKEN_REGEX = re.compile(r"[\w]+['’`]\w+|[\w]+")  # Regex đơn giản hóa để đếm từ
tokens1 = TOKEN_REGEX.findall(text1)
tokens2 = TOKEN_REGEX.findall(text2)

# 3. Đếm tần suất
count1 = Counter(tokens1)
count2 = Counter(tokens2)

# 4. Tính toán độ "độc quyền" (Score = Tần suất bên này - Tần suất bên kia)
# Loại bỏ các từ chung chung (stopwords) nếu cần
stopwords = {
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
}

unique_1 = {}
for word, freq in count1.items():
    if word not in stopwords and len(word) > 2:
        # Score = Tần suất ở file 1 trừ đi 2 lần tần suất ở file 2 (phạt nặng nếu xuất hiện ở file 2)
        score = freq - 2 * count2.get(word, 0)
        if score > 0:
            unique_1[word] = score

unique_2 = {}
for word, freq in count2.items():
    if word not in stopwords and len(word) > 2:
        score = freq - 2 * count1.get(word, 0)
        if score > 0:
            unique_2[word] = score

# 5. In kết quả Top 30
print("--- TOP KEYWORDS CHO FIRST LM (TIÊU CỰC) ---")
print(sorted(unique_1, key=unique_1.get, reverse=True)[:30])

print("\n--- TOP KEYWORDS CHO SECOND LM (TÍCH CỰC) ---")
print(sorted(unique_2, key=unique_2.get, reverse=True)[:30])
