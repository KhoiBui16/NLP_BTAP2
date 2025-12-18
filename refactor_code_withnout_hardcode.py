import pickle
import random
import re
from collections import defaultdict, Counter
import math

# CONFIGURATION
TOKEN_REGEX = re.compile(r"[\w]+['’`]\w+|[\w]+|[^\s\w]")
CLEAN_REGEX = re.compile(r"\]+\]")

# Lamba weights for interpolation
LAMBDA_3 = 0.75
LAMBDA_2 = 0.20
LAMBDA_1 = 0.05


class BaseModel:
    def __init__(self):
        self.unigram = Counter()
        self.bigram = defaultdict(Counter)
        self.trigram = defaultdict(Counter)
        self.total_tokens = 0
        self.generated_history = set()

        self.dynamic_keywords = set()
        self.dynamic_starters = []
        self.dynamic_lengths = []

    def _data_extraction(self, clean_data):
        # Trích xuất tokens từ data
        all_sentences_tokens = []
        all_tokens = []

        for s in clean_data:
            tokens = TOKEN_REGEX.findall(s.lower())
            if not tokens:
                continue
            all_sentences_tokens.append(tokens)
            all_tokens.extend(tokens)

        return all_sentences_tokens, all_tokens

    def _feature_selection(self, all_sentences_tokens, all_tokens):
        # Chọn lọc feature và dynamic parameters từ dữ liệu theo thống kê

        # --- 1. XỬ LÝ ĐỘ DÀI (LENGTH_OPTS) ---
        # dùng IQR (Interquartile Range) để tìm vùng tập trung
        lengths = [len(s) for s in all_sentences_tokens]
        if lengths:
            lengths.sort()
            n = len(lengths)
            q1 = lengths[n // 4]
            q3 = lengths[(3 * n) // 4]
            # Lấy dải phổ biến nhất giữa Q1 và Q3 (vùng dữ liệu tập trung nhất) => Độ dài chuẩn của câu hay xuất hiện nhiều
            self.dynamic_lengths = [l for l in range(q1, q3 + 1)]

            # Fallback nếu dải quá hẹp
            if len(self.dynamic_lengths) < 3:
                avg = int(sum(lengths) / n)
                self.dynamic_lengths = [avg - 2, avg - 1, avg, avg + 1, avg + 2]
        else:
            self.dynamic_lengths = [12, 14, 15, 16, 17, 18]

        # --- 2. XỬ LÝ STARTERS (STARTERS_LM) ---
        # Logic: Lấy Top Starters phổ biến nhất
        # Format: (['word1', 'word2'], weight)
        starters_list = []
        for tokens in all_sentences_tokens:
            if len(tokens) >= 2:
                starters_list.append(tuple(tokens[:2]))
            elif len(tokens) == 1:
                starters_list.append((tokens[0], "<START>"))

        starter_counts = Counter(starters_list)
        total_s = sum(starter_counts.values())

        # Lấy top 8 starters
        top_starters = starter_counts.most_common(8)

        self.dynamic_starters = []
        if top_starters:
            for (w1, w2), count in top_starters:
                # Tính weight dựa trên tần suất thực tế
                weight = round(count / total_s, 2)
                # Boost weight lên một chút để tổng ~ 1.0 hoặc chuẩn hóa sau
                if weight < 0.05:
                    weight = 0.05  # Minimum weight
                self.dynamic_starters.append(([w1, w2], weight))

            # Luôn thêm <START> <START> như một fallback an toàn
            self.dynamic_starters.append((["<START>", "<START>"], 0.10))
        else:
            self.dynamic_starters = [(["<START>", "<START>"], 1.0)]

        # --- 3. XỬ LÝ KEYWORDS (KEYWORDS_LM) ---
        # Tìm "Top Keywords" đặc trưng.
        # Vấn đề: Làm sao biết từ nào là stopword (như 'the', 'is') mà không hardcode list stopword?
        # Giải pháp: Stopwords thường là những từ xuất hiện NHIỀU NHẤT trong TOÀN BỘ tập dữ liệu (cả 1 và 2).
        # Nhưng ở đây ta chỉ thấy 1 file tại 1 thời điểm.
        # Heuristic cải tiến: Stopwords là những từ có tần suất cực cao (top 10-20 từ đầu).
        # Keywords là những từ phổ biến tiếp theo (rank 20-100) và có độ dài >= 3.

        word_counts = Counter(all_tokens)

        # Sắp xếp từ phổ biến nhất xuống thấp nhất
        sorted_words = word_counts.most_common()

        # Bỏ qua Top 15 từ phổ biến nhất (coi là stopwords nội tại: the, i, to, and...)
        # Đây là cách "học" stopword từ chính dữ liệu
        potential_keywords = sorted_words[15:]

        final_keywords = set()
        count = 0
        for word, freq in potential_keywords:
            # Logic lọc rác từ EDA:
            # 1. Độ dài > 2 (bỏ từ ngắn vô nghĩa)
            # 2. Không chứa ký tự lạ (chỉ lấy chữ cái)
            if len(word) > 2 and word.isalpha():
                final_keywords.add(word)
                count += 1
            if count >= 40:  # Lấy top 40 keywords đặc trưng
                break

        self.dynamic_keywords = final_keywords

    def fit(self, data):
        clean_data = [CLEAN_REGEX.sub("", s) for s in data]

        # --- GIAI ĐOẠN 1: TRÍCH XUẤT & HỌC THAM SỐ (EDA TỰ ĐỘNG) ---
        # Thay vì hardcode, ta gọi hàm để "nhìn" dữ liệu và rút ra tham số
        s_tokens, all_toks = self._data_extraction(clean_data)
        self._feature_selection(s_tokens, all_toks)

        # --- GIAI ĐOẠN 2: HUẤN LUYỆN N-GRAM (CORE MODEL) ---
        for tokens in s_tokens:
            # Logic cũ để build model ngram
            if not tokens:
                continue

            # Đếm unigram
            tokens_with_end = tokens + ["<END>"]
            self.total_tokens += len(tokens_with_end)
            self.unigram.update(tokens_with_end)

            # Đếm bigram, trigram
            padded = ["<START>", "<START>"] + tokens_with_end
            for i in range(len(padded) - 2):
                w1, w2, w3 = padded[i], padded[i + 1], padded[i + 2]
                if w3 == "<START>":
                    continue
                if w2 != "<START>":
                    self.bigram[w2][w3] += 1
                self.trigram[(w1, w2)][w3] += 1

    def _get_next_word_distribution(self, w1, w2):
        candidates = set()
        if (w1, w2) in self.trigram:
            candidates.update(self.trigram[(w1, w2)].keys())
        if w2 in self.bigram:
            candidates.update(self.bigram[w2].keys())

        if not candidates:
            candidates.update(k for k, v in self.unigram.most_common(20))

        dist = {}
        c_tri = sum(self.trigram[(w1, w2)].values()) if (w1, w2) in self.trigram else 0
        c_bi = sum(self.bigram[w2].values()) if w2 in self.bigram else 0

        for w in candidates:
            p_tri = (self.trigram[(w1, w2)][w] / c_tri) if c_tri > 0 else 0
            p_bi = (self.bigram[w2][w] / c_bi) if c_bi > 0 else 0
            p_uni = (
                (self.unigram[w] / self.total_tokens) if self.total_tokens > 0 else 0
            )
            dist[w] = (LAMBDA_3 * p_tri) + (LAMBDA_2 * p_bi) + (LAMBDA_1 * p_uni)

        return dist

    def _post_process(self, tokens):
        text = " ".join(tokens)
        text = re.sub(r"\s+([.,!?;:])", r"\1", text)
        text = re.sub(r"\(\s+", "(", text)
        text = re.sub(r"\s+\)", ")", text)
        text = re.sub(r" ' ", "'", text)

        if len(text) > 0:
            text = text[0].upper() + text[1:]

        def capitalize_match(match):
            return match.group(0).upper()

        text = re.sub(r"([.!?]\s+)([a-z])", capitalize_match, text)
        text = re.sub(r"\b(i)\b", "I", text)
        text = re.sub(r"\b(i`m)\b", "I`m", text)
        return text

    def generate(self):
        # SỬ DỤNG THAM SỐ ĐỘNG ĐÃ HỌC TỪ FIT
        starters_config = self.dynamic_starters
        filter_keywords = self.dynamic_keywords
        length_opts = self.dynamic_lengths

        BEST_OF_N = 30
        MAX_RETRIES = 120

        candidates = []

        for _ in range(MAX_RETRIES):
            # 1. Chọn Starter (từ dynamic list)
            if not starters_config:
                w1, w2 = "<START>", "<START>"
            else:
                s_opts = [x[0] for x in starters_config]
                s_w = [x[1] for x in starters_config]
                # Fallback nếu list rỗng hoặc lỗi
                if not s_opts:
                    w1, w2 = "<START>", "<START>"
                else:
                    w1, w2 = random.choices(s_opts, weights=s_w, k=1)[0]

            # 2. Chọn độ dài (từ dynamic list)
            target_len = random.choice(length_opts) if length_opts else 15

            ret = []
            if w1 != "<START>":
                ret.extend([w1, w2])

            sentence_log_prob = 0
            valid_sentence = True

            for _ in range(target_len):
                dist = self._get_next_word_distribution(w1, w2)
                if not dist:
                    valid_sentence = False
                    break

                words = list(dist.keys())
                probs = list(dist.values())

                next_word = random.choices(words, weights=probs, k=1)[0]

                if dist[next_word] > 0:
                    sentence_log_prob += math.log(dist[next_word])
                else:
                    sentence_log_prob -= 20

                if next_word == "<END>":
                    if len(ret) < 6:  # Chặn câu quá ngắn (như logic cũ)
                        valid_sentence = False
                    break

                ret.append(next_word)
                w1, w2 = w2, next_word

            if not valid_sentence or not ret:
                continue

            refined = self._post_process(ret)

            if refined in self.generated_history:
                continue

            # Filter keywords (Dynamic Check)
            if filter_keywords:
                # Fail-fast: Nếu không có keyword, bỏ qua ngay
                if not any(kw in refined.lower() for kw in filter_keywords):
                    continue

            avg_log_prob = sentence_log_prob / len(ret)
            kw_count = sum(1 for kw in filter_keywords if kw in refined.lower())

            final_score = avg_log_prob + (kw_count * 4.0)

            candidates.append((final_score, refined))

            if len(candidates) >= BEST_OF_N:
                break

        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            best = candidates[0][1]
            self.generated_history.add(best)
            return best

        return "I am sorry."

    def _save_to_file(self, filename):
        temp = self.generated_history
        self.generated_history = set()
        try:
            with open(filename, "wb") as f:
                pickle.dump(self, f)
        finally:
            self.generated_history = temp


# SUBCLASSES
class FirstLM(BaseModel):
    def __init__(self):
        super().__init__()

    def save(self):
        self._save_to_file("FirstLM.mdl")


class SecondLM(BaseModel):
    def __init__(self):
        super().__init__()

    def save(self):
        self._save_to_file("SecondLM.mdl")
