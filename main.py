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
        unique_words.extend(["<|endoftext|>", "<|unk|>"])

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


        # for s in preprocessed:
        #     if s in self.word_to_id:
        #         ids.append(self.word_to_id[s])
        #     else:
        #         raise ValueError(f"Unknown word! {s}")
        
        # for s in preprocessed:
        #     if s in self.word_to_id:
        #         ids.append(self.word_to_id[s])
        #     else:
        #         # add to the vocab
        #         new_word = s
        #         new_word_id = self.vocab_size
        #         self.id_to_word[new_word_id] = new_word
        #         self.word_to_id[new_word] = new_word_id
        #         # increase vocabl size
        #         self.vocab_size += 1
        #         # add the newly added token to the final sequence 
        #         ids.append(new_word_id)
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
        Q = self.multiply(embeddings, self.Wq)
        K = self.multiply(embeddings, self.Wk)
        V = self.multiply(embeddings, self.Wv)
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

def train_one_example(input_tokens: list, target_token, last_embedding, prediction, pred_head, learning_rate=0.5):
    """Train on a single sentence"""
    
    input_tokens
    target_token
    last_embedding
    prediction
    predicted_token = prediction.index(max(prediction))

    # print("Before training:")
    # print(tokenizer.detokenize([predicted_token]))
    
    # push right token up. gradient--how much to push. if weere not very wronf--push a little
    for i in range(len(pred_head.weight_matrix)):
        gradient = 1.0 - prediction[target_token] # how wrong we were (0 to 1)
        pred_head.weight_matrix[i][target_token] += learning_rate * gradient * last_embedding[i]
    
    # push wrong tokens down
    for wrong_word_id in range(len(prediction)):
        if wrong_word_id != target_token:
            for i in range(len(pred_head.weight_matrix)):
                pred_head.weight_matrix[i][wrong_word_id] -= learning_rate * prediction[wrong_word_id] * last_embedding[i]
    
def cross_entropy_loss(predictions: list[float], target_idx):
    """
    Calculate how wrong were we
    
    :param predictions: generated predictions
    :param target_idx: Description
    """
    return -math.log(max(predictions[target_idx], 1e-10))

def generate_a_token(embedding: list[float], pred_head: PredictionHead):
    predictions = pred_head.predict(embedding)
    best_prediction_id = predictions.index(max(predictions))
    return best_prediction_id

# Training data

with open("./cat_corpus.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
cat_sentences = [
    "the cat sat on the mat",
    "my cat loves to nap",
    "cats have soft fur",
    "a cat chased a mouse",
    "the black cat purred loudly",
    "cats enjoy playing with yarn",
    "my cat eats fish for dinner",
    "cats can see in the dark",
    "the orange cat climbed the tree",
    "cats make me happy",
    "a cat has sharp claws",
    "cats drink milk sometimes",
    "the cat slept all day",
    "cats are good pets",
    "my cat likes to cuddle",
    "cats hunt small birds",
    "the cat washed its face",
    "cats have nine lives",
    "a cat meowed at the door",
    "cats jump very high",
    "the cat ran fast",
    "cats have whiskers for sensing",
    "my cat has green eyes",
    "cats are curious animals",
    "the cat found a warm spot",
    "cats purr when content",
    "a cat scratched the sofa",
    "cats groom themselves often",
    "the cat watched the birds",
    "cats are independent creatures",
    "my cat follows me everywhere",
    "cats like cardboard boxes",
    "the cat caught a mouse",
    "cats have retractable claws",
    "a cat slept in the sun",
    "cats communicate with meows",
    "the cat rubbed against my leg",
    "cats can be very playful",
    "my cat hides under the bed",
    "cats are clean animals"
]
cat_corpus = " ".join(cat_sentences)
print(raw_text[:99])
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(preprocessed[:30])
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
print(vocab_size)


dimensions = 64
epochs_no = 3

# Initialize KEMP (your tiny transformer named after your boyfriend!)
t = ToyTokenizer(raw_text)
emb_mat = EmbeddingMatrix(t.vocab_size, dimen=dimensions)
pred_head = PredictionHead(t.vocab_size, dimensions)
attention_layer = SimpleSelfAttention(t.vocab_size, dimensions)



print("Training phase...")
for epoch in range(epochs_no):
    epoch_loss = 0
    correct = 0
    total = 0

    for sentence in cat_sentences: 
        tokens = t.tokenize(sentence)
        input_tokens = tokens[:-1]
        target_token = tokens[-1]

        embedded = emb_mat.embed(input_tokens)
        attended = attention_layer.forward(embedded)
        last_embedding = attended[-1]

        prediction = pred_head.predict(last_embedding)
        loss = cross_entropy_loss(prediction,target_token)
        epoch_loss += loss

        # Track accuracy
        predicted_token = prediction.index(max(prediction))
        if predicted_token == target_token:
            correct += 1
        total += 1

        # Update weights
        train_one_example(input_tokens, target_token, last_embedding, prediction,pred_head, learning_rate=0.1)

    avg_loss = epoch_loss / len(cat_sentences)
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

        embeddings = emb_mat.embed(tokens)[-1]
        for i in range(n_preds):
            embedded = emb_mat.embed(result)
            attended = attention_layer.forward(embedded)
            last_embedding = attended[-1]
            next_token = generate_a_token(last_embedding, pred_head)
            result.append(next_token)
            
        print("KEMP: " + t.detokenize(result))
finally:
    pass