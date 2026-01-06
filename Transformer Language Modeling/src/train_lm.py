import os
import numpy as np
import json
import random
import argparse
from tqdm import tqdm
import torch
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn import CrossEntropyLoss
from torch.nn.utils.clip_grad import clip_grad_norm_
from project3_src.model import TransformerLM
from project3_src.utils import save_model, load_model
import matplotlib.pyplot as plt


def evaluate(model: torch.nn.Module, dev_data: np.array, num_iterations: int, batch_size: int, context_length: int, device: str):
    """
    Evaluate the model on the development data.

    Args:
        model: The model to evaluate.
        dev_data: 1D numpy array of integer token IDs in the development data.
        num_iterations: The number of iterations to evaluate the model.
        batch_size: Desired batch size to sample.
        context_length: Desired context length of each sampled example.
        device: PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.

    Returns:
        The average loss of the model on the development data.
        The perplexity of the model on the development data.
    """
    model.eval()
    loss = 0
    cross_entropy = CrossEntropyLoss(reduction='sum')
    total_tokens = 0
    with torch.no_grad():
        for i in range(num_iterations):
            dev_x, dev_y = load_data(dev_data, batch_size, context_length, device)
            logits = model(dev_x)
            loss += cross_entropy(logits.view(-1, logits.size(-1)), dev_y.view(-1))
            total_tokens += dev_y.numel()
    loss = loss.item() / total_tokens
    perplexity = np.exp(loss)
    model.train()
    return loss, perplexity


def plot_learning_curve(train_losses, train_steps_logged, val_losses, val_steps_logged):
    plt.figure(figsize=(12, 6))
    plt.plot(train_steps_logged, train_losses, label='Training Loss', color='blue')
    plt.plot(val_steps_logged, val_losses, label='Validation Loss', color='red', marker='o', linestyle='--')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curve Over Steps')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('losses.png')


def load_data(dataset: np.array, batch_size: int, context_length: int, device: str):
    """
    Args:
        dataset: 1D numpy array of integer token IDs in the dataset.
        batch_size: Desired batch size to sample.
        context_length: Desired context length of each sampled example.
        device: PyTorch device string (e.g., 'cpu' or 'cuda:0') indicating the device
            to place the sampled input sequences and labels on.
    Returns:
        Tuple of torch.LongTensors of shape (batch_size, context_length). The first tuple item
        is the sampled input sequences, and the second tuple item is the corresponding
        language modeling labels.
    """
    # Dataset is a 1D array of token ids. Sample batch_size random sequences of length context_length.
    # Each example x is dataset[i : i+context_length] and target y is dataset[i+1 : i+1+context_length]
    assert dataset.ndim == 1, "dataset must be a 1D array of token ids"
    n = len(dataset)
    if n < context_length + 1:
        raise ValueError("Dataset too small for the requested context length")

    # randint upper bound is exclusive; valid start indices are [0, n - context_length]
    starts = np.random.randint(0, n - context_length, size=batch_size)
    x_batch = np.stack([dataset[s : s + context_length] for s in starts]).astype(np.int64)
    y_batch = np.stack([dataset[s + 1 : s + 1 + context_length] for s in starts]).astype(np.int64)

    x_tensor = torch.from_numpy(x_batch).long().to(device)
    y_tensor = torch.from_numpy(y_batch).long().to(device)
    return x_tensor, y_tensor


def train(args: argparse.Namespace):
    """
    Train the model.
    """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_config = {
        'vocab_size': args.vocab_size,
        'context_length': args.context_length,
        'd_model': args.d_model,
        'num_layers': args.num_layers,
        'num_heads': args.num_heads,
        'd_ff': args.d_ff,
        'rope_theta': args.rope_theta,
    }
    device = args.device

    train_data = np.memmap(args.train_path, dtype=np.uint16)
    valid_data = np.memmap(args.valid_path, dtype=np.uint16)

    model = TransformerLM(**model_config)
    model.train()
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr)
    loss_fn = CrossEntropyLoss()
    total_steps = args.total_tokens_processed // (args.batch_size * args.context_length)

    # initialize variables for logging
    train_losses = []
    val_losses = []
    train_steps_logged = []
    val_steps_logged = []
    best_valid_loss = float('inf')
    best_valid_perplexity = float('inf')
    train_loss_sum = 0.0
    train_loss_count = 0
    valid_loss, loss = None, None
    perplexity = None

    for step in tqdm(range(total_steps)):

        # Sample a batch
        x_batch, y_batch = load_data(train_data, args.batch_size, args.context_length, device)

        optimizer.zero_grad()
        logits = model(x_batch)
        # logits: (batch, seq, vocab)
        loss = loss_fn(logits.view(-1, logits.size(-1)), y_batch.view(-1))
        loss.backward()
        # gradient clipping
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()


        # log train loss
        train_loss_sum += loss.item()
        train_loss_count += 1
        if step > 0 and step % args.log_train_interval == 0:
            avg_interval_train_loss = train_loss_sum / train_loss_count
            train_losses.append(avg_interval_train_loss)
            train_steps_logged.append(step)
            train_loss_sum = 0.0
            train_loss_count = 0

        # log validation loss
        if (step > 0 and step % args.log_valid_interval == 0) or (step == total_steps - 1):

            valid_loss, perplexity = evaluate(model, valid_data, args.eval_iterations, args.batch_size, args.context_length, device)

            val_losses.append(valid_loss)
            val_steps_logged.append(step)
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_valid_perplexity = perplexity
                save_model(model=model, config=model_config, path=args.checkpoint_path, filename="transformer_lm")

    print("Best validation loss:", best_valid_loss)
    print("Best validation perplexity:", best_valid_perplexity)
    plot_learning_curve(train_losses, train_steps_logged, val_losses, val_steps_logged)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="/storage/ice-shared/cs7650/project3/tokenized_data/gpt2_train.npy")
    parser.add_argument("--valid_path", type=str, default="/storage/ice-shared/cs7650/project3/tokenized_data/gpt2_valid.npy")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints")
    parser.add_argument("--vocab_size", type=int, default=50257)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--total_tokens_processed", type=int, default=40000000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_iterations", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log_train_interval", type=int, default=100)
    parser.add_argument("--log_valid_interval", type=int, default=500)

    args = parser.parse_args()
    train(args)