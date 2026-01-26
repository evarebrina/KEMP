import random


class EmbeddingMatrix:
    """Converts token IDs to dense embeddings"""
    
    def __init__(self, voc, dimen=16):
        self.vocab_size = voc
        self.emb_dim = dimen
        self.emb_matrix = [[random.uniform(-0.01, 0.01) for _ in range(self.emb_dim)]
                            for _ in range(self.vocab_size)]

    def embed(self, tokens: list[int]):
        """
        Returns an array of embbeddings for tokens.
        
        :param self: Description
        :param tokens: Description
        """
        res = []
        for token in tokens:
            res.append(self.lookup(token))
        return res

    def lookup(self, token_id: int):
        """Returns a copy to avoid unintentional overwrite"""
        return [i for i in self.emb_matrix[token_id]]

    def add_inplace_to_row(self, token_id, grad_vec, scale):
        row = self.emb_matrix[token_id]
        for i in range(self.emb_dim):
            row[i] += scale * grad_vec[i]

class PositionalEmbedding:
    """Adds positional information to embeddings"""
    
    def __init__(self, max_len, dim):
        self.max_len = max_len
        self.dim = dim
        self.rows = [[random.uniform(-0.01, 0.01) for _ in range(dim)]
                     for _ in range(max_len)]

    def get(self, pos):
        return self.rows[pos]

    def add_inplace_to_row(self, pos, grad_vec, scale):
        row = self.rows[pos]
        for i in range(self.dim):
            row[i] += scale * grad_vec[i]

class Embeddings:
    """Combines token and positional embeddings"""
    
    def __init__(self, vocab_size, emb_dim, max_len):
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.max_len = max_len
        self.token_emb = EmbeddingMatrix(vocab_size, emb_dim)
        self.pos_emb = PositionalEmbedding(max_len, emb_dim)
    
    def embed(self, token_ids):
        """Get token embeddings"""
        return self.token_emb.embed(token_ids)
    
    def add_positional_embeddings(self, token_embeddings):
        """Add positional embeddings to token embeddings"""
        pos_embeddings = []
        dim = len(token_embeddings[0])
        for pos in range(len(token_embeddings)):
            token_emb = token_embeddings[pos]
            pos_emb = self.pos_emb.get(pos)
            combined = [token_emb[i] + pos_emb[i] for i in range(dim)]
            pos_embeddings.append(combined)
        return pos_embeddings

