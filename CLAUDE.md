# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

KEMP is a minimal GPT-style transformer language model implemented in **pure Python** — no PyTorch, TensorFlow, NumPy, or any other third-party library. Every piece (tokenizer, embeddings, self-attention, backprop, training loop) is hand-written with plain lists of lists as tensors, for learning purposes. Correctness and clarity of the from-scratch math matter more than performance here.

There is no build system, no dependency manifest, and no test suite — the only dependencies are Python 3 stdlib modules (`random`, `math`, `json`, `re`, `pickle`).

## Commands

- Train a model: `python trainer.py` (reads `cat_corpus.txt`, trains for `CONFIG['epochs']` epochs, writes `weights.json`)
- Run inference: `python inference.py` (loads `weights.json`, then prompts interactively for a starting word and generates a continuation)
- No linter, formatter, or test runner is configured. There are no test files in the repo.
- `python main.py` does **not** run anything by itself — `main.py` only defines shared classes/functions/`CONFIG` and has no `if __name__ == "__main__"` block. Ignore the README's `python main.py` usage instructions; use `trainer.py` / `inference.py` instead.
- `.github/workflows/django.yml` is a stale/generic CI template (`python manage.py test`, `requirements.txt`) left over from a project scaffold. There is no Django app, no `manage.py`, and no `requirements.txt` in this repo, so this workflow will fail if triggered — it does not reflect how this project is actually built or tested.

## Architecture

The model is a single-layer decoder-only transformer, wired together across five modules that each own one piece of the forward/backward pass:

- `tokenizer.py` — `ToyTokenizer`: whitespace/punctuation word-level tokenizer built from whatever corpus it's constructed with. Adds `<|pad|>`, `<|endoftext|>`, `<|unk|>` to the vocab. Supports pickling via `save`/`load` for reuse without retokenizing.
- `embeddings.py` — `EmbeddingMatrix` (token id → learned vector, trainable), `PositionalEmbedding` (fixed sinusoidal encoding, **not** trained — `add_inplace_to_row` is a no-op), and `Embeddings`, which combines the two.
- `attention.py` — `SimpleSelfAttention` holds the `Wq`/`Wk`/`Wv` projection matrices and an `Embeddings` instance; `attention()` is scaled dot-product attention (no causal mask — every position attends to every other position, including "future" ones); `attention_backward()` and `projection_backward()` are the hand-derived gradients for attention and for the `input × W` projections respectively.
- `main.py` — the glue layer and single source of truth for hyperparameters (`CONFIG` dict at the bottom of the file). Contains:
  - `PredictionHead` — final linear + softmax layer mapping the last position's embedding to vocab logits/probabilities.
  - `forward_pass()` — runs embedding → positional add → Q/K/V projection → attention → prediction for one input sequence, returning everything backward passes need (Q, K, V, pos_embeddings, last_embedding).
  - `compute_prediction_head_gradients()`, `update_attention_weights()`, `update_embeddings()`, `train_one_example_with_attention()` — manual backprop and SGD-style weight updates. There is no autograd; every gradient path is written out explicitly and threaded manually between modules.
  - `save_model()` / `load_model()` — checkpoint everything (embedding matrix, positional embeddings, Wq/Wk/Wv, prediction head weights/bias, tokenizer vocab) to/from a single JSON file, with explicit validation of required keys when loading.
  - `cross_entropy_loss()`, `generate_a_token()` (argmax), `sample_token()` (temperature-based sampling) — inference-time helpers.
- `trainer.py` — the training entry point: loads the corpus named in `CONFIG['corpus_file']`, builds a fresh tokenizer/attention layer/prediction head from it, and trains by sampling random windows of *random length* (padded with `<|pad|>` on the left up to `CONFIG['max_len'] - 1`) rather than iterating every token — this keeps training fast and teaches the model to handle variable-length context. Saves to `weights.json` at the end.
- `inference.py` — the interactive generation entry point: loads `weights.json`, then in a loop reads a prompt, tokenizes it, and autoregressively samples `CONFIG['n_predictions']` more tokens, always truncating/padding context to `max_len - 1` tokens before each forward pass.

### Data flow for one training step

`tokens → EmbeddingMatrix.embed → + PositionalEmbedding → ×Wq/Wk/Wv → attention() → last position's vector → PredictionHead.predict → softmax probs → cross_entropy_loss` against the target token, then gradients flow back in the exact reverse order (`compute_prediction_head_gradients → attention_backward → projection_backward` ×3 for Q/K/V → embedding + positional gradient updates), all coordinated by `train_one_example_with_attention()` in `main.py`.

### Key conventions

- All tensors are plain `list`/`list[list[float]]` — no NumPy. Matrix multiply is `SimpleSelfAttention.multiply`.
- Weights are initialized with small uniform random values (`random.uniform(-0.01, 0.01)`).
- Hyperparameters live in one place: `CONFIG` in `main.py`. Change values there rather than hardcoding new ones in `trainer.py`/`inference.py`.
- Learning rates for embeddings and attention weights are the base `learning_rate` scaled by `emb_lr_factor`/`attn_lr_factor` in `CONFIG` — see `update_embeddings`/`update_attention_weights`.
- Attention here is **unmasked** (bidirectional) even though the model is used autoregressively at inference — see the README's "Next Steps" (causal masking, multiple layers, layer norm) for known gaps, not yet implemented.
- Model checkpoints (`weights.json`) and the training corpus (`*.txt`) are gitignored (`.gitignore` excludes `*.json` and `*.txt`), except `cat_corpus.txt`, which is force-tracked as the canonical small training corpus committed to the repo.
