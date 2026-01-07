# Sequence Labelling and Transformer Language Modeling

This repository contains two related NLP components:

1) Sequence Labelling (notebook-driven)
2) Transformer Language Modeling (Python package + scripts)

The goal is to provide a minimal, reproducible setup to train a small Transformer LM and run inference, along with a separate sequence labelling workflow in a Jupyter notebook.

## Repository Structure

- Sequence Labelling/
	- project.ipynb — sequence labelling experiments notebook
	- test_pred_cnn_lstm_crf.txt, test_pred_cnn_lstm.txt, test_pred_lstm.txt — saved predictions
- Transformer Language Modeling/
	- pyproject.toml — project configuration and dependencies
	- data/input_prompts.txt — prompts for inference
	- gpt2_tokenizer/ — tokenizer files (config.json, merges.txt, tokenizer.json, vocab.json)
	- src/ — language model code
		- model.py — `TransformerLM` implementation
		- train_lm.py — training loop and CLI
		- inference.py — top‑p sampling inference CLI
		- tokenizer.py — GPT‑2 compatible tokenizer wrapper
		- bpe.py, utils.py — helpers

## Requirements

- Python 3.11–3.13 (per `pyproject.toml`)
- macOS: PyTorch is pinned to a compatible version for Intel Macs; Apple Silicon uses the general pin.
- Recommended: `uv` for environment management.

## Setup

Using `uv` (recommended):

```sh
cd "Transformer Language Modeling"
uv sync
```

Then run any script with:

```sh
uv run src/train_lm.py --help
uv run src/inference.py --help
```

Alternative using `venv` + `pip`:

```sh
cd "Transformer Language Modeling"
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

## Transformer LM: Training

The training script expects tokenized datasets stored as contiguous NumPy arrays of integer token IDs (e.g., GPT‑2 IDs) saved as `.npy` or via `np.memmap`.

Key arguments in `src/train_lm.py`:

- `--train_path` and `--valid_path`: paths to tokenized data (uint16 IDs). Defaults point to course storage; change these to local files.
- `--checkpoint_path`: directory to save checkpoints (default: `checkpoints`).
- `--context_length`, `--d_model`, `--num_layers`, `--num_heads`, `--d_ff`, `--rope_theta`: model/config.
- `--total_tokens_processed`, `--batch_size`, `--lr`: training budget and hyperparameters.
- `--device`: `cuda` or `cpu`.

Outputs:

- `checkpoints/transformer_lm.pt` and a `config.json` via `utils.save_model()` when a new best validation loss is found.
- `losses.png` showing training and validation curves.

## Transformer LM: Inference

`src/inference.py` performs temperature + nucleus (top‑p) sampling over prompts in `data/input_prompts.txt` and writes `completions.json`.

Prerequisites:

- A trained checkpoint in `checkpoints/` (filename `transformer_lm.pt` + `config.json`).
- GPT‑2 tokenizer files in `Transformer Language Modeling/gpt2_tokenizer/` (already provided).

Run:

```sh
uv run src/inference.py \
	--model_path checkpoints \
	--tokenizer_path "gpt2_tokenizer" \
	--temperature 0.8 \
	--top_p 0.95 \
	--device cpu \
	--max_length 256
```

Results are saved to `completions.json`.

Notes:

- The script uses Hugging Face `AutoTokenizer` to load the local GPT‑2 tokenizer files.
- Ensure `max_length` does not exceed the training `context_length`.

## Data and Tokenization

- For training, prepare token ID arrays compatible with GPT‑2 vocab (default `vocab_size=50257`). You can use `tokenizer.py` or `tiktoken` to produce IDs.
- If you modify special tokens, adjust `tokenizer.py` to include them, and re‑serialize your vocab/merges accordingly.

## Sequence Labelling

- Open `Sequence Labelling/project.ipynb` in VS Code or Jupyter and run cells in order.
- The text files in the same directory (`test_pred_*.txt`) are example prediction dumps from different architectures.
- If datasets are required, adjust notebook paths and environment cells accordingly.
