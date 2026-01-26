"""
Tokenizer for the KEMP language model.

This module provides ToyTokenizer which converts text to token IDs and back,
handling basic punctuation splitting and unknown tokens.
"""
import re

class ToyTokenizer:
    """Converts text to token IDs and back"""
    
    def __init__(self, corpus: str) -> None:

        self.id_to_word = {}
        self.word_to_id = {}

        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', corpus)                 
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        unique_words = sorted(list(set(preprocessed)))
        unique_words.extend(["<|pad|>", "<|endoftext|>", "<|unk|>"])

        self.vocab_size = len(unique_words)

        for i, word in enumerate(unique_words):
            self.id_to_word[i] = word
            self.word_to_id[word] = i
    
    def tokenize(self, text: str) -> list[int]:
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
                                
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        # if s not in self.word_to_id:

        ids = [self.word_to_id.get(s, self.word_to_id['<|unk|>']) for s in preprocessed]

        return ids
        
    
    def detokenize(self, ids: list[int]) -> str:
        text = " ".join([self.id_to_word[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text
