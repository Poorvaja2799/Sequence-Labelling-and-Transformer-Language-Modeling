import torch
import json
import argparse
from transformers import AutoTokenizer
from torch.nn import functional as F
from project3_src.utils import load_model
from tqdm import tqdm


def apply_temperature(logits: torch.Tensor, temperature: float):
    """
    Apply temperature to logits and compute the probability distribution.
    Formula: 
        P(x) = exp(logits / temperature) / sum(exp(logits / temperature))
    Args:
        logits: Tensor of shape (1, vocab_size) containing the model logits of the last token.
        temperature: The temperature to use for sampling.

    Returns:
        distribution: Tensor of shape (1, vocab_size) probabilities of generating the next token.
    """
    if temperature <= 0.0:
        raise ValueError("temperature must be > 0")

    # logits shape: (batch=1, vocab_size) or (vocab_size,) -- ensure we softmax over last dim
    scaled = logits / float(temperature)
    probs = F.softmax(scaled, dim=-1)
    return probs


def truncate_distribution(probs: torch.Tensor, top_p: float):
    """
    The function selects the smallest set of tokens whose cumulative
    probability mass >= `top_p`, sets the probabilities of the remaining
    tokens to zero, and renormalizes the distribution so that the sum of
    the probabilities is 1.
    Return the truncated and normalized distribution

    Args:
        probs: Tensor of shape (1, vocab_size). The probabilities of the tokens.
        top_p: Float in (0, 1]. The top p value to use for sampling.

    Returns:
        truncated_distribution: Tensor of shape (1, vocab_size). The truncated and normalized distribution to sample the next token.
    """
    if not (0.0 < top_p <= 1.0):
        raise ValueError("top_p must be in (0, 1]")

    # If top_p == 1.0, nothing to do
    if top_p >= 1.0:
        return probs

    # Work per-batch; here batch size is expected to be 1 but implement generally
    batch_size, vocab_size = probs.shape
    device = probs.device
    truncated = torch.zeros_like(probs)

    for i in range(batch_size):
        p = probs[i]
        # sort probabilities descending
        sorted_p, sorted_idx = torch.sort(p, descending=True)
        cumsum = torch.cumsum(sorted_p, dim=0)
        # find smallest index where cumulative >= top_p
        above_idx = torch.nonzero(cumsum >= top_p, as_tuple=False)
        if above_idx.numel() == 0:
            # keep all
            keep = sorted_idx
        else:
            last = int(above_idx[0].item())
            keep = sorted_idx[: last + 1]

        mask = torch.zeros(vocab_size, device=device, dtype=torch.bool)
        mask[keep] = True

        p_masked = p * mask.float()
        total = p_masked.sum()
        if total.item() == 0.0:
            # fallback: return original probs (shouldn't happen unless numerical issues)
            truncated[i] = p
        else:
            truncated[i] = p_masked / total

    return truncated


def top_p_sampling(
    model: torch.nn.Module, 
    input_ids: torch.Tensor, 
    max_new_tokens: int, 
    eos_token_id: int, 
    temperature: float, 
    top_p: float):
    """
    Generate text using temperature and nucleus (top-p) sampling (batch_size = 1).
    This function should implement the following steps:
    1. Compute model logits for the last token
    2. Apply temperature to logits and normalize (call `apply_temperature` function)
    3. Truncate distribution to top-p tokens (call `truncate_distribution` function)
    4. Sample the next token from the truncated distribution
    5. Repeat until max_new_tokens reached or `eos_token_id` is sampled.

    Args:
        model: The model to use for generation.
        input_ids: Tensor of shape (1, sequence_length). The input ids of the input prompt.
        (for this assignment, your code only needs to support the batch size of 1).
        max_new_tokens: The maximum number of new tokens to generate.
        eos_token_id: End-of-sequence token ID.
        temperature: The temperature to use for sampling.
        top_p: Float in (0, 1]. The top p value to use for sampling.

    Returns:
        Tensor of shape (1, sequence_length + max_new_tokens). The generated token IDs.
    """
    device = input_ids.device
    # work on a copy so we don't mutate caller tensors
    generated = input_ids.clone().to(device)

    with torch.no_grad():
        for _ in range(int(max_new_tokens)):
            output = model(generated)
            # model may return logits or a tuple (logits, ...)
            logits = output[0] if isinstance(output, (tuple, list)) else output

            # take logits for last token
            last_logits = logits[:, -1, :]

            # apply temperature -> probabilities
            probs = apply_temperature(last_logits, temperature)

            # truncate to top-p
            probs = truncate_distribution(probs, top_p)

            # sample next token (batch size = 1 supported)
            next_token = torch.multinomial(probs, num_samples=1)

            # append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)

            # stop if EOS token generated
            if next_token.item() == int(eos_token_id):
                break

    return generated


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="checkpoints")
    parser.add_argument("--tokenizer_path", type=str, default="gpt2_tokenizer")
    parser.add_argument("--eos_token", type=str, default="<|endoftext|>")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--top_p", type=float, required=True)
    parser.add_argument("--max_length", type=int, default=256, help="this should be <= the context length used in training, change this if you use a different context length")
    args = parser.parse_args()

    input_path = "data/input_prompts.txt"
    output_path = "completions.json"

    with open(input_path, 'r') as f:
        input_prompts = f.read().strip().splitlines()

    MAX_LENGTH = args.max_length

    model, config = load_model(path=args.model_path, filename="transformer_lm", device=args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    model.eval()
    model.to(args.device)
    output_completions = []
    eos_token_id = tokenizer.convert_tokens_to_ids(args.eos_token)
    for input_prompt in tqdm(input_prompts):
        input_ids = tokenizer.encode(input_prompt, return_tensors="pt", add_special_tokens=False).to(args.device)

        max_new_tokens = MAX_LENGTH - input_ids.shape[1]
        output_ids = top_p_sampling(
            model=model, input_ids=input_ids, 
            max_new_tokens=max_new_tokens, 
            eos_token_id=eos_token_id, 
            temperature=args.temperature,
            top_p=args.top_p)
        completion = tokenizer.decode(output_ids.squeeze(0).tolist(), skip_special_tokens=True)
        output_completions.append(completion)

    with open(output_path, 'w') as f:
        json.dump(output_completions, f, indent=4)
    print(f"Completions saved to {output_path}")