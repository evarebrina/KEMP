"""
Training script for the KEMP language model.

This script:
- Loads and preprocesses the training corpus
- Initializes the model components (tokenizer, attention layer, prediction head)
- Runs the training loop with sampling-based mini-batches
- Saves the trained model weights
"""
import random, re
from main import (PredictionHead,
                  cross_entropy_loss, train_one_example_with_attention, save_model, forward_pass,
                  CONFIG,)
from attention import SimpleSelfAttention
from tokenizer import ToyTokenizer


# ============================================================================
# TRAINING DATA SETUP
# ============================================================================

# Training data
with open(CONFIG['corpus_file'], "r", encoding="utf-8") as f:
    raw_text = f.read()

# Initialize KEMP
tokenizer = ToyTokenizer(raw_text)
attention_layer = SimpleSelfAttention(
    tokenizer.vocab_size, 
    CONFIG['emb_dim'], 
    max_len=CONFIG['max_len']
)
pred_head = PredictionHead(tokenizer.vocab_size, CONFIG['emb_dim'])


# Tokenize entire corpus for training
all_tokens = tokenizer.tokenize(raw_text.lower())
print(f"Total tokens: {len(all_tokens)}")

# ============================================================================
# TRAINING LOOP
# ============================================================================

print("Training phase...")
for epoch in range(CONFIG['epochs']):
    epoch_loss = 0
    correct = 0
    total = 0

    # Sample random positions instead of using every token (much faster)
    num_samples = min(CONFIG['num_samples_per_epoch'], len(all_tokens) - CONFIG['max_len'])
    sample_positions = random.sample(range(CONFIG['max_len'], len(all_tokens)), num_samples)
    
    for i in sample_positions:
        # Take previous context_window-1 tokens as input
        input_tokens = all_tokens[i - (CONFIG['max_len'] - 1):i]
        target_token = all_tokens[i]

        prediction, last_embedding, Q, K, V, pos_embeddings = forward_pass(
            input_tokens, 
            attention_layer, 
            pred_head
        )

        loss = cross_entropy_loss(prediction, target_token)
        epoch_loss += loss

        # Track accuracy
        predicted_token = prediction.index(max(prediction))
        if predicted_token == target_token:
            correct += 1
        total += 1

        # Update weights
        train_one_example_with_attention(
            input_tokens, target_token, attention_layer, pred_head, 
            learning_rate=CONFIG['learning_rate']
        )
    
    avg_loss = epoch_loss / total
    accuracy = correct / total * 100
    print(f"Epoch {epoch + 1} - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.1f}%")

# Save model
save_model('weights.json', attention_layer, pred_head, tokenizer)
