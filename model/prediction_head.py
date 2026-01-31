"""
Prediction head module
"""
import random
import math


class PredictionHead:
    """Converts embeddings to word probability distributions"""
    
    def __init__(self, vocab_size: int, emb_dim: int) -> None:
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.weight_matrix = [[random.uniform(-0.01, 0.01) for _ in range(self.vocab_size)] 
                              for _ in range(self.emb_dim)]
        self.b = [0.0 for _ in range(self.vocab_size)]
        
    def predict(self, embedding: list[float]) -> list[float]:
        if not isinstance(embedding, list) or not isinstance(embedding[0], float):
            raise ValueError("embedding is not list[float]")
        scores = [0.0 for _ in range(self.vocab_size)]
        for i in range(self.vocab_size):
            total = self.b[i]
            for j in range(self.emb_dim):
                total += (embedding[j] * self.weight_matrix[j][i])
            scores[i] = total
        
        m = max(scores)
        exp_scores = [math.exp(s - m) for s in scores]
        total_exp = sum(exp_scores)
        probabilities = [exp / total_exp for exp in exp_scores]
        return probabilities
    
