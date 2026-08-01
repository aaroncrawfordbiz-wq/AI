"""
Tokenizer: converts text <-> lists of integers, because neural nets only eat
numbers. Ours is character-level: every distinct character in the training
text gets an id. (Real GPTs use sub-word pieces — same idea, bigger alphabet.)
"""

import numpy as np


class CharTokenizer:
    def __init__(self, chars):
        self.chars = list(chars)
        self.ctoi = {c: i for i, c in enumerate(self.chars)}

    @classmethod
    def from_text(cls, text):
        return cls(sorted(set(text)))

    @property
    def vocab_size(self):
        return len(self.chars)

    def encode(self, text):
        # characters the model has never seen are skipped rather than crashing
        return np.array([self.ctoi[c] for c in text if c in self.ctoi],
                        dtype=np.int64)

    def decode(self, ids):
        return "".join(self.chars[int(i)] for i in ids)
