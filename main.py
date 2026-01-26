import random
import math
import re


class ToyTokenizer:
    """Converts text to token IDs and back"""
    
    def __init__(self,  corpus:str):

        self.id_to_word = {}
        self.word_to_id = {}

        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', corpus)                 
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        unique_words = sorted(list(set(preprocessed)))
        unique_words.extend(["", "<|unk|>"])

        self.vocab_size = len(unique_words)

        for i, word in enumerate(unique_words):
            self.id_to_word[i] = word
            self.word_to_id[word] = i
    
    def tokenize(self, text: str):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
                                
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        # if s not in self.word_to_id:

        ids = [self.word_to_id.get(s, self.word_to_id['<|unk|>']) for s in preprocessed]

        return ids
        
    
    def detokenize(self, ids):
        text = " ".join([self.id_to_word[i] for i in ids])
        # Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text

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

class SimpleSelfAttention:
    """Self-attention layer with Q, K, V projections"""
    
    def __init__(self, vocab_size, emb_dim=16, max_len=32, lr=0.05):
        self.vocab_size = vocab_size
        self.dim = emb_dim
        self.max_len = max_len
        self.lr = lr

        self.E = EmbeddingMatrix(vocab_size, emb_dim)
        self.P = PositionalEmbedding(max_len, emb_dim)

        # Q/K/V projections: [emb_dim x emb_dim]
        self.Wq = [[random.uniform(-0.01, 0.01) for _ in range(emb_dim)]
                    for _ in range(emb_dim)]
        self.Wk = [[random.uniform(-0.01, 0.01) for _ in range(emb_dim)]
                    for _ in range(emb_dim)]
        self.Wv = [[random.uniform(-0.01, 0.01) for _ in range(emb_dim)]
                    for _ in range(emb_dim)]

        # Output Layer: emb_dim -> vocab_size
        self.Wo = [[random.uniform(-0.01, 0.01) for _ in range(vocab_size)]
                    for _ in range(emb_dim)]
        self.bo = [0.0 for _ in range(vocab_size)]

    def project(self, vec, W):
        """Multiply vector by weight matrix"""
        out = [0.0] * len(W)
        for i in range(len(W)):
            for j in range(len(W[i])):
                out[i] += vec[j] * W[i][j]
        return out

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
        pos_embeddings = []
        for pos in range(len(embeddings)):
            token_emb = embeddings[pos]
            pos_emb = self.P.get(pos)
            # Add positional encoding to token embedding
            combined = [token_emb[i] + pos_emb[i] for i in range(self.dim)]
            pos_embeddings.append(combined)
        
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
    outputs = []
    for i in range(len(Q)):
        scores = []
        for j in range(len(K)):
            score = dot(Q[i], K[j])
            scores.append(score)

        weights = softmax(scores)

        dim = len(V[0])
        out = [0.0] * dim
        for j in range(len(V)):
            for d in range(dim):
                out[d] += weights[j] * V[j][d]
        outputs.append(out)
    return outputs

def train_one_example(input_tokens: list, target_token, last_embedding, prediction, pred_head, pos_emb, emb_mat, learning_rate=0.5):
    """Train on a single sentence with embedding updates"""
   
    # Step 1 Calclate gradient for the last embedding
    # This tells us "how should we change the embedding to improve the prediction?"
    embedding_gradient = [0.0] * len(last_embedding)

    # Gradient from correct token (push UP)
    gradient_correct = 1.0 - prediction[target_token] # we substract the predicted probability of the target word from 1
    # so if the model predicted a high probability--we will change weights slightly, if low--stronger   
    # push right token up. gradient--how much to push. if we were not very wrong--push a little
    for i in range(len(pred_head.weight_matrix)):
        # Update the prediction head weights
        pred_head.weight_matrix[i][target_token] += learning_rate * gradient_correct * last_embedding[i]
        # Accumulate gradient for embedding
        embedding_gradient[i] += gradient_correct * pred_head.weight_matrix[i][target_token]
    
    # Gradient from wrong tokens (push DOWN)
    for wrong_word_id in range(len(prediction)):
        if wrong_word_id != target_token:
            for i in range(len(pred_head.weight_matrix)):
                # Update prediction head weights
                pred_head.weight_matrix[i][wrong_word_id] -= learning_rate * prediction[wrong_word_id] * last_embedding[i]
                # Accumulate gradient for embedding
                embedding_gradient[i] -= prediction[wrong_word_id] * pred_head.weight_matrix[i][wrong_word_id]
    
    # Step 2: Update the embedding for the last token
    # Use smaller ;earning rate cause embeddings are more sensetive
    emb_lr = learning_rate * 0.1
    # The last embedding came from the last input token
    
    for pos, token_id in enumerate(input_tokens):
        decay = 0.5 ** (len(input_tokens) - pos - 1)
        emb_mat.add_inplace_to_row(token_id, embedding_gradient, learning_rate * 0.1 * decay)
    
    # Step 3: Update positional embeddings
    for pos in range(len(input_tokens)):
        decay = 0.5 ** (len(input_tokens) - pos - 1)
        pos_emb.add_inplace_to_row(pos, embedding_gradient, emb_lr * decay)

    # TODO: Step 4: Update weights

def attention_backward(Q, K, V, grad_outputs):
    """Backpropagate through attenton
    
    Q, K, V: The queries, keys, and values from forward pass
    grad_outputs: Gradient flowing back (list of gradients for each position)
    
    Returns: grad_V, grad_K, grad_Q (gradients for values, keys, and queries)"""
    seq_len = len(Q)
    dim = len(V[0])

    # Init gradients
    grad_V = [[0.0] * dim for _ in range(seq_len)]
    grad_K = [[0.0] * dim for _ in range(seq_len)]
    grad_Q = [[0.0] * dim for _ in range(seq_len)]

    # For each query position
    for i in range(seq_len):
        # Recalculate attention weihts (same as forward pass)
        scores = []
        for j in range(seq_len):
            score = dot(Q[i], K[j])
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
        # score[j] = dot(Q[i], K[j])
        for j in range(seq_len):
            for d in range(dim):
                # grad_Q[i] += grad_scores[j] x K[j]
                grad_Q[i][d] += grad_scores[j] * K[j][d]
                # grad_K[j] += grad_scores[j] x Q[i]
                grad_K[j][d] += grad_scores[j] * Q[i][d]
        
    return grad_V, grad_K, grad_Q

def train_one_example_with_attention(input_tokens, target_token, attention_layer, 
                                     pred_head, learning_rate=0.1):
    """Full training with attention backprop"""
    
    # === FORWARD PASS ===
    embedded = attention_layer.E.embed(input_tokens)
    
    # Add positional embeddings
    pos_embeddings = []
    for pos in range(len(embedded)):
        token_emb = embedded[pos]
        pos_emb = attention_layer.P.get(pos)
        combined = [token_emb[i] + pos_emb[i] for i in range(attention_layer.dim)]
        pos_embeddings.append(combined)
    
    # Project to Q, K, V
    Q = attention_layer.multiply(pos_embeddings, attention_layer.Wq)
    K = attention_layer.multiply(pos_embeddings, attention_layer.Wk)
    V = attention_layer.multiply(pos_embeddings, attention_layer.Wv)
    
    # Apply attention
    attended = attention(Q, K, V)
    last_embedding = attended[-1]
    
    # Prediction
    prediction = pred_head.predict(last_embedding)
    
    # === BACKWARD PASS ===
    
    # Step 1: Gradient from prediction head (you already have this)
    embedding_gradient = [0.0] * len(last_embedding)
    gradient_correct = 1.0 - prediction[target_token]
    
    for i in range(len(pred_head.weight_matrix)):
        pred_head.weight_matrix[i][target_token] += learning_rate * gradient_correct * last_embedding[i]
        embedding_gradient[i] += gradient_correct * pred_head.weight_matrix[i][target_token]
    
    for wrong_word_id in range(len(prediction)):
        if wrong_word_id != target_token:
            for i in range(len(pred_head.weight_matrix)):
                pred_head.weight_matrix[i][wrong_word_id] -= learning_rate * prediction[wrong_word_id] * last_embedding[i]
                embedding_gradient[i] -= prediction[wrong_word_id] * pred_head.weight_matrix[i][wrong_word_id]
    
    # Step 2: Create gradient for all positions (only last position has gradient)
    grad_attended = [[0.0] * attention_layer.dim for _ in range(len(input_tokens))]
    grad_attended[-1] = embedding_gradient  # Only last position gets gradient
    
    # Step 3: Backprop through attention
    grad_V, grad_K, grad_Q = attention_backward(Q, K, V, grad_attended)
    
    # Step 4: Backprop through projections and update Wv, Wk, Wq
    grad_Wv, grad_pos_from_V = projection_backward(pos_embeddings, attention_layer.Wv, grad_V)
    grad_Wk, grad_pos_from_K = projection_backward(pos_embeddings, attention_layer.Wk, grad_K)
    grad_Wq, grad_pos_from_Q = projection_backward(pos_embeddings, attention_layer.Wq, grad_Q)
    
    # Update attention weights
    attn_lr = learning_rate * 0.01  # Smaller learning rate for attention
    for i in range(attention_layer.dim):
        for j in range(attention_layer.dim):
            attention_layer.Wv[i][j] += attn_lr * grad_Wv[i][j]
            attention_layer.Wk[i][j] += attn_lr * grad_Wk[i][j]
            attention_layer.Wq[i][j] += attn_lr * grad_Wq[i][j]
    
    # Step 5: Combine gradients for embeddings
    grad_embeddings = [[0.0] * attention_layer.dim for _ in range(len(input_tokens))]
    for pos in range(len(input_tokens)):
        for d in range(attention_layer.dim):
            grad_embeddings[pos][d] = (grad_pos_from_V[pos][d] + 
                                       grad_pos_from_K[pos][d] + 
                                       grad_pos_from_Q[pos][d])
    
    # Step 6: Update embeddings and positional embeddings
    emb_lr = learning_rate * 0.1
    for pos, token_id in enumerate(input_tokens):
        attention_layer.E.add_inplace_to_row(token_id, grad_embeddings[pos], emb_lr)
        attention_layer.P.add_inplace_to_row(pos, grad_embeddings[pos], emb_lr)

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

# Training data

with open(CONFIG['corpus_file'], "r", encoding="utf-8") as f:
    raw_text = f.read()

# Tokenize the entire corpus once
print(raw_text[:99])
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(preprocessed[:30])
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
print(vocab_size)


# Initialize KEMP (your tiny transformer named after your boyfriend!)
t = ToyTokenizer(raw_text)
attention_layer = SimpleSelfAttention(t.vocab_size, CONFIG['emb_dim'], max_len=CONFIG['max_len'])
# Use the attention layer's embedding matrix
emb_mat = attention_layer.E
pos_emb = attention_layer.P
pred_head = PredictionHead(t.vocab_size, CONFIG['emb_dim'])

# Tokenize entire corpus for training
all_tokens = t.tokenize(raw_text.lower())
print(f"Total tokens: {len(all_tokens)}")

print("Training phase...")
for epoch in range(CONFIG['epochs']):
    epoch_loss = 0
    correct = 0
    total = 0

    # Sample random positions instead of using every token (much faster)
    num_samples = min(5000, len(all_tokens) - CONFIG['max_len'])  # Train on 5000 random samples per epoch
    sample_positions = random.sample(range(CONFIG['max_len'], len(all_tokens)), num_samples)
    
    for i in sample_positions:
        # Take previous context_window-1 tokens as input
        input_tokens = all_tokens[i - (CONFIG['max_len'] - 1):i]
        target_token = all_tokens[i]

        embedded = emb_mat.embed(input_tokens)
        attended = attention_layer.forward(embedded)
        last_embedding = attended[-1]

        prediction = pred_head.predict(last_embedding)
        loss = cross_entropy_loss(prediction, target_token)
        epoch_loss += loss

        # Track accuracy
        predicted_token = prediction.index(max(prediction))
        if predicted_token == target_token:
            correct += 1
        total += 1

        # Update weights
        train_one_example_with_attention(input_tokens, target_token, attention_layer, pred_head)
    
    avg_loss = epoch_loss / total
    accuracy = correct / total * 100
    print(f"Epoch {epoch + 1} - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.1f}%")

# Main loop
try:
    while True:
        prompt = ''
        while prompt == '':
            prompt = input("Starting word: ")
            n_preds = 30 #int(input("Length in words: "))
        try:
            tokens = t.tokenize(prompt.lower())
        finally:
            while prompt == '':
                prompt = input("Starting word: ")
                n_preds = 30 #int(input("Length in words: "))
        if not tokens:
            continue
        result = tokens

        for i in range(n_preds):
            # Use only the last max_len-1 tokens to stay within position embeddings
            context = result[-(attention_layer.max_len - 1):]
            embedded = emb_mat.embed(context)
            attended = attention_layer.forward(embedded)
            last_embedding = attended[-1]
            
            # Use sampling instead of argmax for diversity
            prediction = pred_head.predict(last_embedding)
            next_token = sample_token(prediction, CONFIG['temperature'])
            result.append(next_token)
            
        print("KEMP: " + t.detokenize(result))
finally:
    pass