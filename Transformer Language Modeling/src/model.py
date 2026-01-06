import torch
import torch.nn as nn
from einops import rearrange
import os


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        """
        Args:
            d_model: Hidden dimension of the model
            eps: Epsilon value for numerical stability
        """
        super().__init__()
        # Learnable scale parameter (affine weight). Initialized to ones.
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, sequence_length, d_model)
        Returns:
            Output tensor of shape (batch_size, sequence_length, d_model)
        """
        # Preserve original dtype and upcast to float32 for stable computation
        in_dtype = x.dtype
        x = x.to(torch.float32)

        # Compute RMS over the last (feature) dimension
        # rms = sqrt(mean(x**2, dim=-1, keepdim=True) + eps)
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

        # shape for weight: (1, 1, ..., d_model) where number of leading 1s = x.ndim - 1
        expand_shape = (1,) * (x.ndim - 1) + (-1,)
        w = self.weight.view(expand_shape).to(torch.float32)

        result = x / rms * w

        # Return in the original input dtype
        return result.to(in_dtype)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        """
        Args:
            d_model: Hidden dimension of the model
            d_ff: Hidden dimension of the feedforward layer
        """
        super().__init__()
        # Implement SwiGLU-style feedforward to match reference weight names:
        # w1: (d_model -> d_ff), w3: (d_model -> d_ff), w2: (d_ff -> d_model)
        # This matches the reference snapshots and allows adapter weight copying.
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, sequence_length, d_model)
        Returns:
            Output tensor of shape (batch_size, sequence_length, d_model)
        """
        # x: (..., seq_len, d_model)
        # SwiGLU: gated = SiLU(x @ W1^T) * (x @ W3^T); out = gated @ W2^T
        x1 = self.w1(x)
        x3 = self.w3(x)
        gated = self.act(x1) * x3
        return self.w2(gated)


def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor):
    """
    Your implementation should support: (1) both 3d and 4d tensors for Q, K, and V, (2) an optional user provided bolean mask.
    Args:
        Q: Query tensor of shape (batch_size, ..., sequence_length, d_k)
        K: Key tensor of shape (batch_size, ..., sequence_length, d_k)
        V: Value tensor of shape (batch_size, ..., sequence_length, d_v)
        mask: Mask tensor of shape (sequence_length, sequence_length)
    Returns:
        Output tensor of shape (batch_size, ..., sequence_length, d_v)
    """
    # Ensure float for numerically stable softmax
    orig_dtype = Q.dtype
    Q = Q.to(torch.float32)
    K = K.to(torch.float32)
    V = V.to(torch.float32)

    # Q, K, V can be (..., seq, d_k) or (..., heads, seq, d_k)
    # We'll handle both by working with the last two dims for matmul.
    d_k = Q.shape[-1]
    scale = 1.0 / (d_k ** 0.5)

    # Compute attention scores
    # scores shape: (..., query_len, key_len)
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

    if mask is not None:
        # mask is boolean with shape broadcastable to scores
        scores = scores.masked_fill(~mask, float('-inf'))

    attn = torch.softmax(scores, dim=-1)

    out = torch.matmul(attn, V)
    return out.to(orig_dtype)


class MultiheadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int=None, theta: float=None):
        """
        Following (Vaswani et al., 2017), set dk = dv = dmodel/h.
        Args:
            d_model: Hidden dimension of the model
            num_heads: Number of attention heads
            max_seq_len: Maximum sequence length that will be inputted
            theta: Theta value for the RoPE (None for no RoPE)
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        # projection weights (no bias to match reference weight-only format)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        # alias for compatibility with reference state_dict naming
        self.output_proj = self.o_proj
        # RoPE support optional; not implemented here but placeholder attribute
        self.max_seq_len = max_seq_len
        self.theta = theta
    
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, token_positions: torch.Tensor=None):
        """
        Args:
            Q: Query tensor of shape (batch_size, ..., sequence_length, d_k)
            K: Key tensor of shape (batch_size, ..., sequence_length, d_k)
            V: Value tensor of shape (batch_size, ..., sequence_length, d_v)
            token_positions: Tensor of shape (batch_size, ...) Optional tensor with the positions of the tokens
        Returns:
            Output tensor of shape (batch_size, ..., sequence_length, d_v)
        """
        # Q,K,V expected shape: (..., seq_len, d_model)
        # Project
        q = self.q_proj(Q)
        k = self.k_proj(K)
        v = self.v_proj(V)

        # reshape to (batch_flat, heads, seq, d_head) for stable batched matmuls
        # We'll flatten any leading dimensions into a single batch dimension so we can
        # use standard view/transpose ops (this is robust and fast).
        seq_len = Q.shape[-2]
        d_model = Q.shape[-1]

        def split_heads_flat(x):
            # x: (..., seq, d_model) -> (batch_flat, heads, seq, d_head), orig_leading
            orig_leading = x.shape[:-2]
            x_flat = x.reshape(-1, seq_len, d_model)  # (batch_flat, seq, d_model)
            # now split
            x_heads = x_flat.reshape(x_flat.shape[0], seq_len, self.num_heads, self.d_head).transpose(1, 2)
            # x_heads: (batch_flat, heads, seq, d_head)
            return x_heads, orig_leading

        qh, _ = split_heads_flat(q)
        kh, _ = split_heads_flat(k)
        vh, orig_leading = split_heads_flat(v)

        # Optional RoPE (rotary) support: apply rotary embeddings to q and k.
        # The tests pass `theta` (e.g., 10000.0) via the adapter when RoPE is required.
        if self.theta is not None or token_positions is not None:
            # compute positions: accept token_positions shaped like (1, seq) or (batch, seq)
            if token_positions is None:
                pos_seq = torch.arange(seq_len, device=Q.device)
            else:
                # token_positions may have leading batch dim; reduce to sequence positions
                pos_seq = token_positions.reshape(-1)[-seq_len:]

            # rotary uses pairs of dims; require d_head to be even
            assert self.d_head % 2 == 0, "d_head must be even for RoPE"
            # compute inverse frequencies
            inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.d_head, 2, device=Q.device).float() / self.d_head))
            # pos_seq: (seq,) -> (seq, 1) * (d_head/2,) -> (seq, d_head/2)
            sinusoid_inp = torch.einsum("p,d->pd", pos_seq.float(), inv_freq)
            sin = torch.sin(sinusoid_inp)
            cos = torch.cos(sinusoid_inp)

            # reshape sin/cos to broadcast to (batch_flat, heads, seq, d_head/2)
            sin = sin.unsqueeze(0).unsqueeze(0)  # (1,1,seq,d_head/2) after reshape
            cos = cos.unsqueeze(0).unsqueeze(0)

            # helper to apply RoPE to tensor of shape (batch_flat, heads, seq, d_head)
            def apply_rope(x):
                x1 = x[..., 0::2]
                x2 = x[..., 1::2]
                # x1,x2 shape: (batch_flat, heads, seq, d_head/2)
                x1_rot = x1 * cos - x2 * sin
                x2_rot = x1 * sin + x2 * cos
                # interleave
                x_rot = torch.stack([x1_rot, x2_rot], dim=-1).reshape(x.shape)
                return x_rot

            qh = apply_rope(qh)
            kh = apply_rope(kh)

        # By default, apply a causal (lower-triangular) attention mask so that tokens
        # can only attend to current and previous positions. This matches the
        # autoregressive behavior expected by the reference implementation.
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=Q.device))
        out_h = scaled_dot_product_attention(qh, kh, vh, mask=causal_mask)

        # out_h shape: (batch_flat, heads, seq, d_head)
        # move heads back and collapse: (batch_flat, seq, heads * d_head)
        out_flat = out_h.transpose(1, 2).reshape(out_h.shape[0], seq_len, d_model)

        # restore original leading dims: (..., seq, d_model)
        out = out_flat.reshape(*orig_leading, seq_len, d_model)
        
        return self.o_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int=None, theta: float=None):
        """
        Args:
            d_model:  The dimensionality of the Transformer block input
            num_heads: Number of attention heads
            d_ff: Dimensionality of the feed-forward inner layer
            max_seq_len: Maximum sequence length that will be inputted
            theta: Theta value for the RoPE (None for no RoPE)
        """
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = MultiheadAttention(d_model, num_heads, max_seq_len=max_seq_len, theta=theta)
        self.ln2 = RMSNorm(d_model)
        self.ffn = PositionwiseFeedForward(d_model, d_ff)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor=None):
        """
        Args:
            x: Input tensor of shape (batch_size, ..., sequence_length, d_model)
            token_positions: Tensor of shape (batch_size, ...) Optional tensor with the positions of the tokens
        Returns:
            Output tensor of shape (batch_size, ..., sequence_length, d_model)
        """
        # Pre-norm transformer block
        # x: (..., seq, d_model)
        resid = x
        x = self.ln1(x)
        x_attn = self.attn(x, x, x, token_positions)
        x = x_attn + resid

        resid = x
        x = self.ln2(x)
        x_ffn = self.ffn(x)
        x = x_ffn + resid
        return x


class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, d_model: int, num_layers: int, num_heads: int, d_ff: int, rope_theta: float):
        """
        Args:
            vocab_size: Size of the vocabulary
            context_length: The maximum number of tokens to process at once
            d_model:  The dimensionality of the model embeddings and sublayer outputs
            num_layers: The number of Transformer layers to use.
            num_heads: Number of heads to use in multi-headed attention. `d_model` must be evenly divisible by `num_heads`.
            d_ff: Dimensionality of the feed-forward inner layer
            rope_theta: RoPE Theta parameter (None for no RoPE)
        """
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([TransformerBlock(d_model, num_heads, d_ff, max_seq_len=context_length, theta=rope_theta) for _ in range(num_layers)])
        self.ln_final = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor=None):
        """
        Args:
            x: Input tensor of shape (batch_size, ..., sequence_length, d_model)
            token_positions: Tensor of shape (batch_size, ...) Optional tensor with the positions of the tokens
        Returns:
            Output tensor of shape (batch_size, ..., sequence_length, d_model)
        """
        # x is token indices: (batch, seq)
        # embed
        emb = self.token_embeddings(x)
        h = emb

        # If RoPE is expected (layers were constructed with a theta), and the
        # caller did not provide explicit token positions, create a default
        # position vector [0,1,...,seq-1] on the correct device. This mirrors
        # typical LM implementations where positional info is implicit.
        if token_positions is None and len(self.layers) > 0 and getattr(self.layers[0].attn, "theta", None) is not None:
            seq_len = h.shape[-2]
            token_positions = torch.arange(seq_len, device=h.device).unsqueeze(0)

        for layer in self.layers:
            h = layer(h, token_positions)
        h = self.ln_final(h)
        logits = self.lm_head(h)
        return logits
