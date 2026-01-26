"""
Inference script for the KEMP language model.

This script:
- Loads a trained model from checkpoint
- Accepts user prompts in an interactive loop
- Generates text continuations using the trained model
"""
from main import (
                  CONFIG,
                  load_model, sample_token
                  )

# ============================================================================
# INFERENCE LOOP
# ============================================================================

attention_layer, pred_head, tokenizer = load_model('weights.json')
pad_id = tokenizer.word_to_id["<|pad|>"]
while True:
    prompt = ''
    while prompt == '':
        prompt = input("Starting word: ")
    
    tokens = tokenizer.tokenize(prompt)
    if not tokens:
        continue
    
    result = tokens
    n_preds = CONFIG['n_predictions']

    for i in range(n_preds):
        # Use only the last max_len-1 tokens to stay within position embeddings
        context = result[-(attention_layer.max_len - 1):]
        if len(context) < attention_layer.max_len - 1:
            context = [pad_id] * (attention_layer.max_len - 1 - len(context)) + context
        # embedded = emb_mat.embed(context)
        embedded = attention_layer.embeddings.token_emb.embed(context)
        attended = attention_layer.forward(embedded)
        last_embedding = attended[-1]
        
        # Use sampling instead of argmax for diversity
        prediction = pred_head.predict(last_embedding)
        next_token = sample_token(prediction, CONFIG['temperature'])
        result.append(next_token)
        
    print("KEMP: " + tokenizer.detokenize(result))