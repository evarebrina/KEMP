"""
KEMP - Kindergarten Educational Model for Predictions

This module contains the core training and inference logic for a simple
transformer-based language model. It implements:
- PredictionHead: converts embeddings to vocabulary probability distributions
- Forward and backward pass functions for training
- Model checkpoint saving/loading
- Hyperparameter configuration
"""
import random
import math
import json

from tokenizer import ToyTokenizer
from attention import SimpleSelfAttention, attention, attention_backward, projection_backward

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


def forward_pass(input_tokens: list[int], attention_layer: SimpleSelfAttention, 
                pred_head: PredictionHead) -> tuple[list[float], list[float], list[list[float]], 
                                                     list[list[float]], list[list[float]], list[list[float]]]:
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

def compute_prediction_head_gradients(pred_head: PredictionHead, prediction: list[float], 
                                     target_token: int, last_embedding: list[float], 
                                     learning_rate: float) -> list[float]:
    """Compute and apply gradients for prediction head"""
    embedding_gradient = [0.0] * len(last_embedding)
    gradient_correct = 1.0 - prediction[target_token]
    
    # Accumulate embedding gradients from all tokens
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
    
    return embedding_gradient

def update_attention_weights(attention_layer: SimpleSelfAttention, grad_Wq: list[list[float]], 
                            grad_Wk: list[list[float]], grad_Wv: list[list[float]], 
                            learning_rate: float) -> None:
    """Update Q, K, V projection matrices"""
    attn_lr = learning_rate * CONFIG['attn_lr_factor']
    for i in range(attention_layer.dim):
        for j in range(attention_layer.dim):
            attention_layer.Wv[i][j] += attn_lr * grad_Wv[i][j]
            attention_layer.Wk[i][j] += attn_lr * grad_Wk[i][j]
            attention_layer.Wq[i][j] += attn_lr * grad_Wq[i][j]

def update_embeddings(attention_layer: SimpleSelfAttention, input_tokens: list[int], 
                     grad_embeddings: list[list[float]], learning_rate: float) -> None:
    """Update token and positional embeddings"""
    emb_lr = learning_rate * CONFIG['emb_lr_factor']
    for pos, token_id in enumerate(input_tokens):
        attention_layer.embeddings.token_emb.add_inplace_to_row(token_id, grad_embeddings[pos], emb_lr)
        attention_layer.embeddings.pos_emb.add_inplace_to_row(pos, grad_embeddings[pos], emb_lr)

def train_one_example_with_attention(input_tokens: list[int], target_token: int, 
                                     attention_layer: SimpleSelfAttention, 
                                     pred_head: PredictionHead, learning_rate: float = 0.1) -> None:
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

def cross_entropy_loss(predictions: list[float], target_idx: int) -> float:
    """
    Calculate how wrong were we
    
    :param predictions: generated predictions
    :param target_idx: Description
    """
    # print("cross enytopy =========")
    # print(-math.log(max(predictions[target_idx], 1e-10)))
    # print("cross entropy ^^^^^^^^^^^")
    return -math.log(max(predictions[target_idx], 1e-10))

def generate_a_token(embedding: list[float], pred_head: PredictionHead) -> int:
    predictions = pred_head.predict(embedding)
    best_prediction_id = predictions.index(max(predictions))
    return best_prediction_id

def sample_token(predictions: list[float], temperature: float = 1.0) -> int:
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

def save_model(filename: str, attention_layer: SimpleSelfAttention, 
              pred_head: PredictionHead, tokenizer: ToyTokenizer) -> None:
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

def load_model(filename: str) -> tuple[SimpleSelfAttention, PredictionHead, ToyTokenizer]:
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
    'num_samples_per_epoch': 100,  # Training samples per epoch
    
    # Inference
    'temperature': 0.8,         # Sampling temperature (higher = more random)
    'n_predictions': 30,        # Number of tokens to generate
    
    # Data
    'corpus_file': './cat_corpus.txt'
}

