"""
Self-attention mechanism for the KEMP language model.

This module implements:
- SimpleSelfAttention: self-attention layer with Q, K, V projections
- attention: scaled dot-product attention forward pass
- attention_backward: backpropagation through attention
- projection_backward: backpropagation through linear projections
"""
import random, math
from embeddings import Embeddings

class SimpleSelfAttention:
    """Self-attention layer with Q, K, V projections"""
    
    def __init__(self, vocab_size: int, emb_dim: int = 16, max_len: int = 32, lr: float = 0.05) -> None:
        self.vocab_size = vocab_size
        self.dim = emb_dim
        self.max_len = max_len
        self.lr = lr

        # Embeddings
        self.embeddings = Embeddings(vocab_size, emb_dim, max_len)

        # Q/K/V projections: [emb_dim x emb_dim]
        self.Wq = [[random.uniform(-0.01, 0.01) for _ in range(emb_dim)]
                    for _ in range(emb_dim)]
        self.Wk = [[random.uniform(-0.01, 0.01) for _ in range(emb_dim)]
                    for _ in range(emb_dim)]
        self.Wv = [[random.uniform(-0.01, 0.01) for _ in range(emb_dim)]
                    for _ in range(emb_dim)]

    def multiply(self, A: list[list[float]], B: list[list[float]]) -> list[list[float]] | None:
        """Matrix multiplication"""
        if len(A[0]) != len(B):
            return None
        res = []
        for i in range(len(A)):
            row = []
            for j in range(len(B[0])):
                total = 0
                for k in range(len(B)):
                    total += A[i][k] * B[k][j]
                row.append(total)
            res.append(row)
        return res

    def forward(self, embeddings: list[list[float]]) -> list[list[float]]:
        # Add positional embeddings
        pos_embeddings = self.embeddings.add_positional_embeddings(embeddings)
        
        Q = self.multiply(pos_embeddings, self.Wq)
        K = self.multiply(pos_embeddings, self.Wk)
        V = self.multiply(pos_embeddings, self.Wv)
        attended = attention(Q, K, V)
        return attended

def attention(Q: list[list[float]], K: list[list[float]], V: list[list[float]]) -> list[list[float]]:
    """
    Scaled dot-product attention mechanism
    Q: queries (list of embeddings)
    K: keys (list of embeddings)
    V: values (list of embeddings)
    returns: attended outputs
    """
    dim = len(V[0])
    scale = 1.0 / math.sqrt(dim)  # Scaling factor for numerical stability
    
    outputs = []
    for i in range(len(Q)):
        scores = []
        for j in range(len(K)):
            score = dot(Q[i], K[j]) * scale  # Apply scaling
            scores.append(score)

        weights = softmax(scores)

        out = [0.0] * dim
        for j in range(len(V)):
            for d in range(dim):
                out[d] += weights[j] * V[j][d]
        outputs.append(out)
    return outputs

def attention_backward(Q: list[list[float]], K: list[list[float]], V: list[list[float]], 
                      grad_outputs: list[list[float]]) -> tuple[list[list[float]], list[list[float]], list[list[float]]]:
    """Backpropagate through attenton
    
    Q, K, V: The queries, keys, and values from forward pass
    grad_outputs: Gradient flowing back (list of gradients for each position)
    
    Returns: grad_V, grad_K, grad_Q (gradients for values, keys, and queries)"""
    seq_len = len(Q)
    dim = len(V[0])
    scale = 1.0 / math.sqrt(dim)  # Same scaling factor as forward pass

    # Init gradients
    grad_V = [[0.0] * dim for _ in range(seq_len)]
    grad_K = [[0.0] * dim for _ in range(seq_len)]
    grad_Q = [[0.0] * dim for _ in range(seq_len)]

    # For each query position
    for i in range(seq_len):
        # Recalculate attention weihts (same as forward pass)
        scores = []
        for j in range(seq_len):
            score = dot(Q[i], K[j]) * scale  # Apply scaling
            scores.append(score)
        
        weights = softmax(scores)

        # Gradient from output to V
        # output[i] = sum(weights[j] x V[j])
        # So: grad_V[j] += weights[j] x grad_output[i]
        for j in range(seq_len):
            for d in range(dim):
                grad_V[j][d] += weights[j] * grad_outputs[i][d]

        # Gradients from output tp attention weights
        # This is trickier - we need to backprop through softmax
        grad_weights = [0.0] * seq_len
        for j in range(seq_len):
            # How much does changing weight[j] affect the output?
            for d in range(dim): # grad_outputs[i] x V[j]
                grad_weights[j] += grad_outputs[i][d] * V[j][d]
            
        # Gradient through softmax
        # softmax derivative: softmax[i] x (grad[i] - sum(softmax[j] x grad[j]))
        sum_weighted_grad = sum(weights[j] * grad_weights[j] for j in range(seq_len))
        grad_scores = [weights[j] * (grad_weights[j] - sum_weighted_grad)
                      for j in range(seq_len)]
        
        # Gradient from scores to Q and K
        # score[j] = dot(Q[i], K[j]) * scale
        for j in range(seq_len):
            for d in range(dim):
                # grad_Q[i] += grad_scores[j] * K[j] * scale
                grad_Q[i][d] += grad_scores[j] * K[j][d] * scale
                # grad_K[j] += grad_scores[j] * Q[i] * scale
                grad_K[j][d] += grad_scores[j] * Q[i][d] * scale
        
    return grad_V, grad_K, grad_Q

def dot(A: list[float], B: list[float]) -> float:
    """Simple dot product between two vectors"""
    if len(A) != len(B):
        raise ValueError("Vectors need to have the same length")
    s = 0
    for i in range(len(A)):
        s += A[i] * B[i]
    return s

def softmax(xs: list[float]) -> list[float]:
    """Softmax normalization"""
    m = max(xs)
    exps = []
    for x in xs:
        exps.append(math.exp(x - m))
    total = sum(exps)
    return [e / total for e in exps]

def projection_backward(inputs: list[list[float]], W: list[list[float]], 
                       grad_output: list[list[float]]) -> tuple[list[list[float]], list[list[float]]]:
    """
    Backprop through: output = inputs × W
    
    inputs: Original input vectors (seq_len × dim)
    W: Weight matrix (dim × dim)
    grad_output: Gradient from next layer (seq_len × dim)
    
    Returns: grad_W, grad_inputs
    """
    seq_len = len(inputs)
    dim = len(W)
    
    # Initialize gradients
    grad_W = [[0.0] * dim for _ in range(dim)]
    grad_inputs = [[0.0] * dim for _ in range(seq_len)]
    
    # For each position in sequence
    for pos in range(seq_len):
        # Gradient for W: grad_W[i][j] += input[pos][j] × grad_output[pos][i]
        for i in range(dim):
            for j in range(dim):
                grad_W[i][j] += inputs[pos][j] * grad_output[pos][i]
        
        # Gradient for inputs: grad_input[pos][j] += W[i][j] × grad_output[pos][i]
        for j in range(dim):
            for i in range(dim):
                grad_inputs[pos][j] += W[i][j] * grad_output[pos][i]
    
    return grad_W, grad_inputs
