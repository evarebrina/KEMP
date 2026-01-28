"""
Embedding layers for the KEMP language model.

This module provides:
- EmbeddingMatrix: converts token IDs to dense embeddings
- PositionalEmbedding: adds positional information to embeddings
- Embeddings: combines token and positional embeddings
"""
import random, math


class EmbeddingMatrix:
    """Converts token IDs to dense embeddings"""
    
    def __init__(self, voc: int, dimen: int = 16) -> None:
        self.vocab_size = voc
        self.emb_dim = dimen
        self.emb_matrix = [[random.uniform(-0.01, 0.01) for _ in range(self.emb_dim)]
                            for _ in range(self.vocab_size)]

    def embed(self, tokens: list[int]) -> list[list[float]]:
        """
        Returns an array of embbeddings for tokens.
        
        :param self: Description
        :param tokens: Description
        """
        res = []
        for token in tokens:
            res.append(self.lookup(token))
        return res

    def lookup(self, token_id: int) -> list[float]:
        """Returns a copy to avoid unintentional overwrite"""
        return [i for i in self.emb_matrix[token_id]]

    def add_inplace_to_row(self, token_id: int, grad_vec: list[float], scale: float) -> None:
        row = self.emb_matrix[token_id]
        for i in range(self.emb_dim):
            row[i] += scale * grad_vec[i]

class PositionalEmbedding:
    """Adds positional information to embeddings"""
    
    def __init__(self, max_len: int, dim: int) -> None:
        self.max_len = max_len
        self.dim = dim
        self.rows = self._generate_positional_encoding(max_len, dim)
        
    def _generate_positional_encoding(self, max_len: int, dim: int) -> list[list[float]]:
        """Generate positional encodings using sine and cosine functions"""
        rows = []
        for pos in range(max_len):
            row = []
            for i in range(dim):
                if i % 2 == 0:  # Even indices: sine
                    row.append(math.sin(pos / (10000 ** (i / dim))))
                else:  # Odd indices: cosine
                    row.append(math.cos(pos / (10000 ** ((i - 1) / dim))))
            rows.append(row)
        return rows

    def get(self, pos: int) -> list[float]:
        return self.rows[pos]

    def add_inplace_to_row(self, pos: int, grad_vec: list[float], scale: float) -> None:
        pass
        # No need to train positional embedding

class Embeddings:
    """Combines token and positional embeddings"""
    
    def __init__(self, vocab_size: int, emb_dim: int, max_len: int) -> None:
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.max_len = max_len
        self.token_emb = EmbeddingMatrix(vocab_size, emb_dim)
        self.pos_emb = PositionalEmbedding(max_len, emb_dim)
    
    def embed(self, token_ids: list[int]) -> list[list[float]]:
        """Get token embeddings"""
        return self.token_emb.embed(token_ids)
    
    def add_positional_embeddings(self, token_embeddings: list[list[float]]) -> list[list[float]]:
        """Add positional embeddings to token embeddings"""
        pos_embeddings = []
        dim = len(token_embeddings[0])
        for pos in range(len(token_embeddings)):
            token_emb = token_embeddings[pos]
            pos_emb = self.pos_emb.get(pos)
            combined = [token_emb[i] + pos_emb[i] for i in range(dim)]
            pos_embeddings.append(combined)
        return pos_embeddings

