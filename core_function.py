class BaseModel:
    def __init__(self):
        self.unigram = Counter()
        self.bigram = defaultdict(Counter)
        self.trigram = defaultdict(Counter)
        self.total_tokens = 0
        self.generated_history = set()

    def fit(self, data):
        clean_data = [CLEAN_REGEX.sub("", s) for s in data]
        for s in clean_data:
            tokens = TOKEN_REGEX.findall(s.lower())
            if not tokens:
                continue
            tokens.append("<END>")
            self.total_tokens += len(tokens)
            self.unigram.update(tokens)

            padded = ["<START>", "<START>"] + tokens
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

        # Fallback nếu bí từ
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

            # logic interpolation
            dist[w] = (LAMBDA_3 * p_tri) + (LAMBDA_2 * p_bi) + (LAMBDA_1 * p_uni)

        return dist

    def _post_process(self, tokens):
        # Basic cleanup
        text = " ".join(tokens)
        text = re.sub(r"\s+([.,!?;:])", r"\1", text)
        text = re.sub(r"\(\s+", "(", text)
        text = re.sub(r"\s+\)", ")", text)
        text = re.sub(r" ' ", "'", text)

        if len(text) > 0:
            text = text[0].upper() + text[1:]

        def capitalize_match(match):
            # Capitalize the letter after sentence-ending punctuation
            return match.group(0).upper()

        # Capitalize first letter after sentence-ending punctuation
        text = re.sub(r"([.!?]\s+)([a-z])", capitalize_match, text)
        text = re.sub(r"\b(i)\b", "I", text)
        text = re.sub(r"\b(i`m)\b", "I`m", text)
        return text

    def generate(self, filter_keywords=None, starters_config=None):
        # TĂNG CƯỜNG ĐỘ TÌM KIẾM CỦA MÔ HÌNH (AGGRESSIVE SEARCH)
        BEST_OF_N = 30  # Số lượng ứng viên để chọn câu tốt nhất
        MAX_RETRIES = 120  # Cho phép sai tối đa 4 lần để tìm được 1 câu (120/30 = 4)

        candidates = []

        for _ in range(MAX_RETRIES):
            # 1. Chọn Starter
            s_opts = [x[0] for x in starters_config]
            s_w = [x[1] for x in starters_config]
            w1, w2 = random.choices(s_opts, weights=s_w, k=1)[0]

            target_len = random.choice(LENGTH_OPTS)
            ret = []
            if w1 != "<START>":
                ret.extend([w1, w2])

            sentence_log_prob = 0
            valid_sentence = True

            # 2. Sinh từ
            for _ in range(target_len):
                dist = self._get_next_word_distribution(w1, w2)
                if not dist:
                    valid_sentence = False
                    break

                words = list(dist.keys())
                probs = list(dist.values())

                next_word = random.choices(words, weights=probs, k=1)[0]

                # Tính Log-Prob
                if dist[next_word] > 0:
                    sentence_log_prob += math.log(dist[next_word])
                else:
                    sentence_log_prob -= 20

                if next_word == "<END>":
                    if len(ret) < 6:
                        valid_sentence = False  # Chặn câu quá ngắn
                    break

                ret.append(next_word)
                w1, w2 = w2, next_word

            if not valid_sentence or not ret:
                continue

            refined = self._post_process(ret)

            # 3. Filters history & keywords
            if refined in self.generated_history:
                continue
            if filter_keywords:
                # Fail-fast: Nếu không có keyword, bỏ qua ngay lập tức
                if not any(kw in refined.lower() for kw in filter_keywords):
                    continue

            # 4. SCORING
            avg_log_prob = sentence_log_prob / len(ret)
            kw_count = sum(1 for kw in filter_keywords if kw in refined.lower())

            # 5. TÍNH ĐIỂM CUỐI CÙNG VỚI TĂNG CƯỜNG KEYWORD & AVG LOG PROB
            final_score = avg_log_prob + (kw_count * 4.0)

            candidates.append((final_score, refined))

            if len(candidates) >= BEST_OF_N:
                break

        if candidates:
            # Chọn câu điểm cao nhất (The Chosen One)
            candidates.sort(key=lambda x: x[0], reverse=True)
            best = candidates[0][1]
            self.generated_history.add(best)
            return best

        return "I am sorry."
