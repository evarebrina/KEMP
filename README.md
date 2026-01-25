# KEMP - Minimal GPT Implementation

A minimal GPT (Generative Pre-trained Transformer) implementation in pure Python with **no external libraries** (no PyTorch, no TensorFlow, no NumPy).

## 🎯 Project Goal

Learn transformer architecture from first principles by implementing every component manually.

## ✨ Features

- **Custom Tokenizer**: Word-level tokenization with unknown token handling
- **Embedding Layer**: Token and positional embeddings
- **Self-Attention**: Single attention layer with Q/K/V projections
- **Training Loop**: Cross-entropy loss with gradient descent
- **Text Generation**: Next-token prediction

## 🏗️ Architecture

```
Input Text → Tokenizer → Embeddings → Self-Attention → Prediction Head → Output
```

- **Embedding dimension**: 64
- **Vocabulary size**: ~dynamic (based on corpus)
- **Layers**: 1 attention layer (planned to expand)

## 🚀 Usage

```bash
python main.py
```

The model trains on cat-themed sentences, then enters interactive mode:
- Enter a starting prompt
- Model generates next 30 tokens

## 📊 Training Data

- `cat_corpus.txt`: Primary training corpus about cats
- Built-in cat sentences for quick training

## 🛠️ Current Status

**Implemented:**
- ✅ Tokenization and vocabulary building
- ✅ Embedding matrices
- ✅ Self-attention mechanism
- ✅ Softmax and cross-entropy loss
- ✅ Basic training loop

**In Progress:**
- 🔄 Full backpropagation through all layers
- 🔄 Causal attention masking
- 🔄 Multiple transformer blocks

**Planned:**
- ⏳ Multi-head attention
- ⏳ Layer normalization
- ⏳ Better text generation (temperature, sampling)
- ⏳ Training visualization

## 📚 What I Learned

- How attention mechanisms work at the matrix level
- Why gradient descent works for neural networks
- The importance of proper weight initialization
- Trade-offs between model complexity and training time

## 🎓 Educational Purpose

This project demonstrates understanding of:
- Transformer architecture (Attention is All You Need)
- Linear algebra operations (dot products, matrix multiplication)
- Optimization through gradient descent
- Neural network fundamentals

## 📝 License

Educational project - feel free to use for learning!

## 🙏 Inspiration

Named after someone special who inspired this learning journey.
