import random
from collections import defaultdict

class BigramModel:
    def __init__(self, corpus):
        """
        corpus: list of strings (sentences)
        """
        self.bigram_counts = defaultdict(lambda: defaultdict(int))
        self.unigram_counts = defaultdict(int)
        self.vocab = set()
        self._train(corpus)
    
    def _train(self, corpus):
        for sentence in corpus:
            words = sentence.split()
            # 添加起始和结束标记
            words = ['<s>'] + words + ['</s>']
            for i in range(len(words)-1):
                w1, w2 = words[i], words[i+1]
                self.bigram_counts[w1][w2] += 1
                self.unigram_counts[w1] += 1
                self.vocab.add(w1)
                self.vocab.add(w2)
    
    def generate_text(self, start_word, length=10):
        """生成文本，从 start_word 开始，生成 length 个词"""
        if start_word not in self.vocab:
            return f"Start word '{start_word}' not in vocabulary."
        current = start_word
        result = [current]
        for _ in range(length):
            if current not in self.bigram_counts:
                break
            # 根据当前词的下一个词分布采样
            next_word_probs = self.bigram_counts[current]
            total = sum(next_word_probs.values())
            if total == 0:
                break
            # 随机选择下一个词（按频率加权）
            rand = random.randint(0, total-1)
            cumsum = 0
            next_word = None
            for w, cnt in next_word_probs.items():
                cumsum += cnt
                if rand < cumsum:
                    next_word = w
                    break
            if next_word == '</s>':
                break
            result.append(next_word)
            current = next_word
        return ' '.join(result)