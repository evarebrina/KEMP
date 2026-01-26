import random
import math
import json

from tokenizer import ToyTokenizer
from embeddings import Embeddings

class SimpleSelfAttention:
    """Self-attention layer with Q, K, V projections"""
    
    def __init__(self, vocab_size, emb_dim=16, max_len=32, lr=0.05):
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

    def multiply(self, A, B):
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

    def forward(self, embeddings):
        # Add positional embeddings
        pos_embeddings = self.embeddings.add_positional_embeddings(embeddings)
        
        Q = self.multiply(pos_embeddings, self.Wq)
        K = self.multiply(pos_embeddings, self.Wk)
        V = self.multiply(pos_embeddings, self.Wv)
        attended = attention(Q, K, V)
        return attended

class PredictionHead:
    """Converts embeddings to word probability distributions"""
    
    def __init__(self, vocab_size, emb_dim):
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.weight_matrix = [[random.uniform(-0.01, 0.01) for _ in range(self.vocab_size)] 
                              for _ in range(self.emb_dim)]
        self.b = [0.0 for _ in range(self.vocab_size)]
        
    def predict(self, embedding: list[float]):
        if not isinstance(embedding, list) or not isinstance(embedding[0], float):
            raise ValueError("embedding is not list[float]")
        scores = [0.0 for _ in range(self.vocab_size)]
        for i in range(self.vocab_size):
            total = self.b[i]
            for j in range(self.emb_dim):
                total += (embedding[j] * self.weight_matrix[j][i])
            scores[i] = total
        
        exp_scores = [math.exp(s) for s in scores]
        total_exp = sum(exp_scores)
        probabilities = [exp / total_exp for exp in exp_scores]
        return probabilities


def dot(A, B):
    """Simple dot product between two vectors"""
    if len(A) != len(B):
        raise ValueError("Vectors need to have the same length")
    s = 0
    for i in range(len(A)):
        s += A[i] * B[i]
    return s

def softmax(xs):
    """Softmax normalization"""
    m = max(xs)
    exps = []
    for x in xs:
        exps.append(math.exp(x - m))
    total = sum(exps)
    return [e / total for e in exps]

def attention(Q, K, V):
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

def attention_backward(Q, K, V, grad_outputs):
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

def forward_pass(input_tokens, attention_layer, pred_head):
    """Execute forward pass through the model"""
    embedded = attention_layer.embeddings.embed(input_tokens)
    pos_embeddings = attention_layer.embeddings.add_positional_embeddings(embedded)
    
    Q = attention_layer.multiply(pos_embeddings, attention_layer.Wq)
    K = attention_layer.multiply(pos_embeddings, attention_layer.Wk)
    V = attention_layer.multiply(pos_embeddings, attention_layer.Wv)
    
    attended = attention(Q, K, V)
    last_embedding = attended[-1]
    prediction = pred_head.predict(last_embedding)
    
    return prediction, last_embedding, Q, K, V, pos_embeddings

def compute_prediction_head_gradients(pred_head, prediction, target_token, last_embedding, learning_rate):
    """Compute and apply gradients for prediction head"""
    embedding_gradient = [0.0] * len(last_embedding)
    gradient_correct = 1.0 - prediction[target_token]
    
    # Accumilate embedding gradients from all tokens
    for i in range(len(pred_head.weight_matrix)):
        embedding_gradient[i] += gradient_correct * pred_head.weight_matrix[i][target_token]
        for wrong_word_id in range(len(prediction)):
            if wrong_word_id != target_token:
                embedding_gradient[i] -= prediction[wrong_word_id] * pred_head.weight_matrix[i][wrong_word_id]

    # Update PredictionHead weight matrix for each dimesion:
    for i in range(len(pred_head.weight_matrix)):
        # Update the weights for target token
        pred_head.weight_matrix[i][target_token] += learning_rate * gradient_correct * last_embedding[i]
        # Update the weights for wrong tokens
        for wrong_word_id in range(len(prediction)):
            if wrong_word_id != target_token:
                pred_head.weight_matrix[i][wrong_word_id] -= learning_rate * prediction[wrong_word_id] * last_embedding[i]
                embedding_gradient[i] -= prediction[wrong_word_id] * pred_head.weight_matrix[i][wrong_word_id]
    
    return embedding_gradient

def update_attention_weights(attention_layer, grad_Wq, grad_Wk, grad_Wv, learning_rate):
    """Update Q, K, V projection matrices"""
    attn_lr = learning_rate * CONFIG['attn_lr_factor']
    for i in range(attention_layer.dim):
        for j in range(attention_layer.dim):
            attention_layer.Wv[i][j] += attn_lr * grad_Wv[i][j]
            attention_layer.Wk[i][j] += attn_lr * grad_Wk[i][j]
            attention_layer.Wq[i][j] += attn_lr * grad_Wq[i][j]

def update_embeddings(attention_layer, input_tokens, grad_embeddings, learning_rate):
    """Update token and positional embeddings"""
    emb_lr = learning_rate * CONFIG['emb_lr_factor']
    for pos, token_id in enumerate(input_tokens):
        attention_layer.embeddings.token_emb.add_inplace_to_row(token_id, grad_embeddings[pos], emb_lr)
        attention_layer.embeddings.pos_emb.add_inplace_to_row(pos, grad_embeddings[pos], emb_lr)

def train_one_example_with_attention(input_tokens, target_token, attention_layer, 
                                     pred_head, learning_rate=0.1):
    """Full training with attention backprop"""
    
    # Forward pass
    prediction, last_embedding, Q, K, V, pos_embeddings = forward_pass(
        input_tokens, attention_layer, pred_head
    )
    
    # Backward pass - prediction head
    embedding_gradient = compute_prediction_head_gradients(
        pred_head, prediction, target_token, last_embedding, learning_rate
    )
    
    # Prepare gradient for attention backward pass
    grad_attended = [[0.0] * attention_layer.dim for _ in range(len(input_tokens))]
    grad_attended[-1] = embedding_gradient
    
    # Backprop through attention
    grad_V, grad_K, grad_Q = attention_backward(Q, K, V, grad_attended)
    
    # Backprop through projections
    grad_Wv, grad_pos_from_V = projection_backward(pos_embeddings, attention_layer.Wv, grad_V)
    grad_Wk, grad_pos_from_K = projection_backward(pos_embeddings, attention_layer.Wk, grad_K)
    grad_Wq, grad_pos_from_Q = projection_backward(pos_embeddings, attention_layer.Wq, grad_Q)
    
    # Update attention weights
    update_attention_weights(attention_layer, grad_Wq, grad_Wk, grad_Wv, learning_rate)
    
    # Combine gradients for embeddings
    grad_embeddings = [[0.0] * attention_layer.dim for _ in range(len(input_tokens))]
    for pos in range(len(input_tokens)):
        for d in range(attention_layer.dim):
            grad_embeddings[pos][d] = (grad_pos_from_V[pos][d] + 
                                       grad_pos_from_K[pos][d] + 
                                       grad_pos_from_Q[pos][d])
    
    # Update embeddings
    update_embeddings(attention_layer, input_tokens, grad_embeddings, learning_rate)

def projection_backward(inputs, W, grad_output):
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

def cross_entropy_loss(predictions: list[float], target_idx):
    """
    Calculate how wrong were we
    
    :param predictions: generated predictions
    :param target_idx: Description
    """
    # print("cross enytopy =========")
    # print(-math.log(max(predictions[target_idx], 1e-10)))
    # print("cross entropy ^^^^^^^^^^^")
    return -math.log(max(predictions[target_idx], 1e-10))

def generate_a_token(embedding: list[float], pred_head: PredictionHead):
    predictions = pred_head.predict(embedding)
    best_prediction_id = predictions.index(max(predictions))
    return best_prediction_id

def sample_token(predictions: list[float], temperature=1.0):
    """
    Sample a token based on probability distribution
    
    :param predictions: probability distribution over vocabulary
    :param temperature: controls randomness (higher = more random)
    """
    if temperature == 0:
        return predictions.index(max(predictions))
    
    # Apply temperature
    scaled = [p ** (1.0 / temperature) for p in predictions]
    total = sum(scaled)
    probs = [p / total for p in scaled]
    
    # Sample from distribution
    r = random.random()
    cumulative = 0
    for i, p in enumerate(probs):
        cumulative += p
        if r <= cumulative:
            return i
    return len(probs) - 1

def save_model(filename: str, attention_layer: SimpleSelfAttention, pred_head: PredictionHead, tokenizer: ToyTokenizer):
    """Save all model weights and tokenizer vocab"""
    checkpoint = {
        'embedding_matrix': attention_layer.embeddings.token_emb.emb_matrix,
        'positional_embeddings': attention_layer.embeddings.pos_emb.rows,
        'Wq': attention_layer.Wq,
        'Wk': attention_layer.Wk,
        'Wv': attention_layer.Wv,
        'pred_head_weights': pred_head.weight_matrix,
        'pred_head_bias': pred_head.b,
        'vocab': tokenizer.word_to_id,
        'id_to_word': tokenizer.id_to_word,
    }
    with open(filename, 'w') as f:
        json.dump(checkpoint, f)

def load_model(filename):
    """Load checkpoint and reconstruct model"""
    try:
        with open(filename, 'r') as f:
            checkpoint = json.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Model file '{filename}' not found.") from e
    except (OSError, IOError) as e:
        raise IOError(f"Error reading model file '{filename}': {e}") from e
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON content in model file '{filename}': {e}") from e

    # Reconstruct tokenizer
    tokenizer = ToyTokenizer("")  # Empty init
    tokenizer.word_to_id = checkpoint['vocab']
    tokenizer.id_to_word = {int(k): v for k, v in checkpoint['id_to_word'].items()}
    tokenizer.vocab_size = len(tokenizer.word_to_id)

    # Validate embedding matrix dimensions against tokenizer vocab size
    embedding_matrix = checkpoint.get('embedding_matrix')
    if not embedding_matrix:
        raise ValueError("Invalid checkpoint: 'embedding_matrix' is missing or empty.")
    if not isinstance(checkpoint, dict):
        raise ValueError("Invalid checkpoint format: expected a JSON object.")

    required_keys = [
        'embedding_matrix',
        'positional_embeddings',
        'Wq',
        'Wk',
        'Wv',
        'pred_head_weights',
        'pred_head_bias',
        'vocab',
        'id_to_word',
    ]
    missing = [k for k in required_keys if k not in checkpoint]
    if missing:
        raise ValueError(f"Invalid checkpoint: missing keys {missing}")

    embedding_matrix = checkpoint['embedding_matrix']
    if not isinstance(embedding_matrix, list) or not embedding_matrix:
        raise ValueError("Invalid checkpoint: 'embedding_matrix' must be a non-empty list.")
    first_row = embedding_matrix[0]
    try:
        emb_dim = len(first_row)
    except TypeError as exc:
        raise ValueError("Invalid checkpoint: 'embedding_matrix' rows must be indexable sequences.") from exc
    if emb_dim <= 0:
        raise ValueError("Invalid checkpoint: 'embedding_matrix' rows must have positive length.")

    pos_embeddings = checkpoint['positional_embeddings']
    if not isinstance(pos_embeddings, list) or not pos_embeddings:
        raise ValueError("Invalid checkpoint: 'positional_embeddings' must be a non-empty list.")
    max_len = len(pos_embeddings)

    vocab = checkpoint['vocab']
    if not isinstance(vocab, dict) or not vocab:
        raise ValueError("Invalid checkpoint: 'vocab' must be a non-empty mapping.")

    # Reconstruct tokenizer
    tokenizer = ToyTokenizer("")  # Empty init
    tokenizer.word_to_id = vocab
    tokenizer.id_to_word = {int(k): v for k, v in checkpoint['id_to_word'].items()}
    tokenizer.vocab_size = len(tokenizer.word_to_id)

    # Reconstruct model
    attention_layer = SimpleSelfAttention(tokenizer.vocab_size, emb_dim, max_len)
    attention_layer.embeddings.token_emb.emb_matrix = embedding_matrix
    attention_layer.embeddings.pos_emb.rows = pos_embeddings
    attention_layer.Wq = checkpoint['Wq']
    attention_layer.Wk = checkpoint['Wk']
    attention_layer.Wv = checkpoint['Wv']

    pred_head = PredictionHead(tokenizer.vocab_size, emb_dim)
    pred_head.weight_matrix = checkpoint['pred_head_weights']
    pred_head.b = checkpoint['pred_head_bias']

    return attention_layer, pred_head, tokenizer

# ============================================================================
# HYPERPARAMETERS
# ============================================================================

CONFIG = {
    # Model architecture
    'emb_dim': 64,              # Embedding dimension
    'max_len': 32,              # Maximum context window
    
    # Training
    'epochs': 3,                # Number of training epochs
    'learning_rate': 0.1,       # Base learning rate
    'emb_lr_factor': 0.1,       # Embedding learning rate multiplier (0.01 of base)
    'attn_lr_factor': 0.01,     # Attention learning rate multiplier (0.001 of base)
    'num_samples_per_epoch': 5000,  # Training samples per epoch
    
    # Inference
    'temperature': 0.8,         # Sampling temperature (higher = more random)
    'n_predictions': 30,        # Number of tokens to generate
    
    # Data
    'corpus_file': './cat_corpus.txt'
}

