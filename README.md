# KEMP - Minimal GPT from Scratch

A transformer-based language model implemented in pure Python (no deep learning libraries).

## Overview

This project implements a minimal GPT (Generative Pre-trained Transformer) to understand transformer architecture from first principles. Everything is built from scratch including:

- **Tokenizer**: Word-level tokenization with unknown token handling
- **Embeddings**: Token and positional embeddings
- **Self-Attention**: Scaled dot-product attention with Q/K/V projections
- **Training**: Cross-entropy loss with gradient descent

## Features

- ✅ Pure Python implementation (no PyTorch/TensorFlow)
- ✅ Self-attention mechanism with Q/K/V matrices
- ✅ Positional embeddings for sequence ordering
- ✅ Text generation capabilities
- ✅ Training loop with loss tracking

## Current Status

**Stage**: Early training phase
- Single attention layer functional
- Prediction head training implemented
- Successfully learns patterns from training data
- Can generate text based on learned patterns

## Usage

```bash
python main.py
```

The model will train on the cat corpus and then enter interactive mode where you can provide prompts for text generation.

## Architecture

- **Embedding Dimension**: 64
- **Vocabulary**: Dynamic based on training corpus
- **Context Length**: Up to 32 tokens
- **Layers**: 1 attention layer (minimal architecture)

## Training Data

- `cat_corpus.txt`: Educational text about cats
- `LightNovels.txt`: Additional training corpus

## Next Steps

- [ ] Implement full backpropagation through all layers
- [ ] Add causal attention masking
- [ ] Stack multiple transformer layers
- [ ] Improve text generation with temperature/sampling
- [ ] Add layer normalization

## Learning Goals

Understanding transformer architecture by:
1. Implementing matrix operations from scratch
2. Building attention mechanisms without frameworks
3. Training with manual gradient calculations
4. Debugging at the fundamental level

---

*Built as a learning project to understand transformers from the ground up.*
