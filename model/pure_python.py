"""
Pure original Python implementation of KEMP
(refactored into a class)
"""
import random
import math
import pickle
from attention import SimpleSelfAttention, attention, attention_backward, projection_backward
from config import CONFIG

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
    

class SimpleKEMPModel:

    def __init__(self, vocab_size: int, config: dict, tokenizer=None):
        self.tokenizer = tokenizer
        self.config = config
        self.vocab_size = vocab_size
        self.attention_layer = SimpleSelfAttention(
            vocab_size, 
            config['emb_dim'], 
            max_len=config['max_len']
            
        )
        self.pred_head = PredictionHead(vocab_size, config['emb_dim'])
        # Default pad_id, will be overridden if tokenizer provided
        self.pad_id = 0

    def forward_pass(self, input_tokens: list[int]) -> tuple[list[float], list[float], list[list[float]], 
                                                        list[list[float]], list[list[float]], list[list[float]]]:
        """Execute forward pass through the model"""
        embedded = self.attention_layer.embeddings.embed(input_tokens)
        pos_embeddings = self.attention_layer.embeddings.add_positional_embeddings(embedded)
        
        Q = self.attention_layer.multiply(pos_embeddings, self.attention_layer.Wq)
        K = self.attention_layer.multiply(pos_embeddings, self.attention_layer.Wk)
        V = self.attention_layer.multiply(pos_embeddings, self.attention_layer.Wv)
        
        attended = attention(Q, K, V)
        last_embedding = attended[-1]
        prediction = self.pred_head.predict(last_embedding)
        
        return prediction, last_embedding, Q, K, V, pos_embeddings
    
    def train_one_example_with_attention(self, input_tokens: list[int], target_token: int, 
                                         learning_rate: float = 0.1) -> None:
        """Full training with attention backprop"""
        
        # Forward pass
        prediction, last_embedding, Q, K, V, pos_embeddings = self.forward_pass(
            input_tokens
        )
        
        # Backward pass - prediction head
        embedding_gradient = self.compute_prediction_head_gradients(
            prediction, target_token, last_embedding, learning_rate
        )
        
        # Prepare gradient for attention backward pass
        grad_attended = [[0.0] * self.attention_layer.dim for _ in range(len(input_tokens))]
        grad_attended[-1] = embedding_gradient
        
        # Backprop through attention
        grad_V, grad_K, grad_Q = attention_backward(Q, K, V, grad_attended)
        
        # Backprop through projections
        grad_Wv, grad_pos_from_V = projection_backward(pos_embeddings, self.attention_layer.Wv, grad_V)
        grad_Wk, grad_pos_from_K = projection_backward(pos_embeddings, self.attention_layer.Wk, grad_K)
        grad_Wq, grad_pos_from_Q = projection_backward(pos_embeddings, self.attention_layer.Wq, grad_Q)
        
        # Update attention weights
        self.update_attention_weights(grad_Wq, grad_Wk, grad_Wv, learning_rate)
        
        # Combine gradients for embeddings
        grad_embeddings = [[0.0] * self.attention_layer.dim for _ in range(len(input_tokens))]
        for pos in range(len(input_tokens)):
            for d in range(self.attention_layer.dim):
                grad_embeddings[pos][d] = (grad_pos_from_V[pos][d] + 
                                        grad_pos_from_K[pos][d] + 
                                        grad_pos_from_Q[pos][d])
        
        # Update embeddings
        self.update_embeddings(input_tokens, grad_embeddings, learning_rate)


    def compute_prediction_head_gradients(self, prediction: list[float], 
                                        target_token: int, last_embedding: list[float], 
                                        learning_rate: float) -> list[float]:
        """Compute and apply gradients for prediction head"""
        emb_dim = len(last_embedding)
        vocab_size = len(prediction)
        # 1) delta = p - y where y is one-hot
        delta = prediction[:]
        delta[target_token] -= 1.0  # p_i - y_i

        # 2) dL/dx = W @ delta
        embedding_gradient = [0.0] * emb_dim
        for j in range(emb_dim):
            s = 0.0
            row = self.pred_head.weight_matrix[j]  # vocab_size length
            for i in range(vocab_size):
                s += row[i] * delta[i]
            embedding_gradient[j] = s
        
        # 3) Update head weights
        for j in range(emb_dim):
            row = self.pred_head.weight_matrix[j]
            for i in range(vocab_size):
                row[i] -= learning_rate * delta[i] * last_embedding[j]

        # 4) Update head biases
        for i in range(vocab_size):
            self.pred_head.b[i] -= learning_rate * delta[i]
        
        return embedding_gradient

    def update_attention_weights(self, grad_Wq: list[list[float]], 
                                grad_Wk: list[list[float]], grad_Wv: list[list[float]], 
                                learning_rate: float) -> None:
        """Update Q, K, V projection matrices"""
        attn_lr = learning_rate * CONFIG['attn_lr_factor']
        for i in range(self.attention_layer.dim):
            for j in range(self.attention_layer.dim):
                self.attention_layer.Wv[i][j] += attn_lr * grad_Wv[i][j]
                self.attention_layer.Wk[i][j] += attn_lr * grad_Wk[i][j]
                self.attention_layer.Wq[i][j] += attn_lr * grad_Wq[i][j]

    def update_embeddings(self, input_tokens: list[int], 
                         grad_embeddings: list[list[float]], learning_rate: float) -> None:
        """Update token and positional embeddings"""
        emb_lr = learning_rate * CONFIG['emb_lr_factor']
        for pos, token_id in enumerate(input_tokens):
            self.attention_layer.embeddings.token_emb.add_inplace_to_row(token_id, grad_embeddings[pos], emb_lr)
            self.attention_layer.embeddings.pos_emb.add_inplace_to_row(pos, grad_embeddings[pos], emb_lr)

    def save_model(self, filename: str) -> None:
        """Save all model weights and tokenizer vocab"""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load_model(cls, filename: str = "weights_1.plk") -> "SimpleKEMPModel":
        """Load checkpoint and reconstruct model"""
        with open(filename, 'rb') as f:
            model = pickle.load(f)
            return model

