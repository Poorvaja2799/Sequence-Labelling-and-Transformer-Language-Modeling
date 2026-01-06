
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Dict, Tuple
import os
import regex as re  # ensure we use the 'regex' module

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: List[str] | None = None,
    debug: bool = False,
):
    """
    Train a deterministic, reasonably efficient byte-level BPE tokenizer.

    Args:
        input_path:
            Path to a UTF-8 text file containing training data.
        vocab_size:
            Desired total vocabulary size (including bytes AND special tokens).
        special_tokens:
            List of special token strings to be added to the vocabulary.
            These are reserved tokens and are *not* merged.
        debug:
            If True, returns (vocab, merges, debug_trace) where debug_trace
            can be any useful diagnostics; otherwise returns (vocab, merges).

    Returns:
        vocab: dict[int, bytes]
            Mapping from token id -> token bytes.
        merges: list[tuple[bytes, bytes]]
            Each merge is a pair (token1_bytes, token2_bytes) in the order
            they were created.
    """
    if special_tokens is None:
        special_tokens = []

    text = Path(input_path).read_text(encoding="utf-8")

    id_to_bytes: List[bytes] = []
    special_token_to_id: Dict[str, int] = {}

    for tok in special_tokens:
        b = tok.encode("utf-8")
        special_token_to_id[tok] = len(id_to_bytes)
        id_to_bytes.append(b)

    byte_offset = len(id_to_bytes)  # where raw-byte tokens start

    for i in range(256):
        id_to_bytes.append(bytes([i]))

    # If requested vocab_size is <= base vocab, just truncate.
    base_vocab_size = len(id_to_bytes)
    if vocab_size <= base_vocab_size:
        vocab = {i: id_to_bytes[i] for i in range(vocab_size)}
        merges: List[Tuple[bytes, bytes]] = []
        if debug:
            return vocab, merges, {}
        return vocab, merges


    token_pattern = re.compile(PAT)

    # Split the raw text around special tokens (if any). We treat special
    # tokens as reserved, atomic tokens; they do NOT participate in merges.
    segments: List[tuple[str, bool, str | None]] = []
    if special_tokens:
        st_pattern = re.compile("|".join(re.escape(st) for st in special_tokens))
        pos = 0
        for m in st_pattern.finditer(text):
            if m.start() > pos:
                segments.append((text[pos:m.start()], False, None))
            segments.append((m.group(0), True, m.group(0)))
            pos = m.end()
        if pos < len(text):
            segments.append((text[pos:], False, None))
    else:
        segments.append((text, False, None))

    # word -> frequency, where word is a tuple of integer token IDs
    words: Dict[Tuple[int, ...], int] = {}

    for seg, is_special, st in segments:
        if is_special:
            # Represent each special token as a single-id "word"
            sid = special_token_to_id[st]  # type: ignore[arg-type]
            w = (sid,)
            words[w] = words.get(w, 0) + 1
        else:
            # Normal text: GPT-2 style pre-tokenization
            for tok in token_pattern.findall(seg):
                tok_bytes = tok.encode("utf-8")
                if not tok_bytes:
                    continue
                # Initial segmentation: each byte is its own token
                ids = tuple(byte_offset + b for b in tok_bytes)
                words[ids] = words.get(ids, 0) + 1

    # If corpus is empty after pretokenization, just return base vocab.
    if not words:
        vocab = {i: id_to_bytes[i] for i in range(len(id_to_bytes))}
        merges: List[Tuple[bytes, bytes]] = []
        if debug:
            return vocab, merges, {}
        return vocab, merges

    pair_counts: Counter[Tuple[int, int]] = Counter()
    pair_to_words: Dict[Tuple[int, int], set[Tuple[int, ...]]] = defaultdict(set)

    for w, freq in words.items():
        if len(w) < 2:
            continue
        for i in range(len(w) - 1):
            p = (w[i], w[i + 1])
            pair_counts[p] += freq
            pair_to_words[p].add(w)

    merges: List[Tuple[bytes, bytes]] = []

    # Helper for deterministic tie-breaking:
    #   1. Highest count
    #   2. Lexicographically larger first token (by its bytes)
    #   3. Lexicographically larger second token (by its bytes)
    def pair_score(p: Tuple[int, int]) -> tuple[int, bytes, bytes]:
        return (pair_counts[p], id_to_bytes[p[0]], id_to_bytes[p[1]])

    debug_trace = {"num_steps": 0} if debug else None

    target_vocab_size = vocab_size

    while len(id_to_bytes) < target_vocab_size and pair_counts:
        # Select best pair by count + deterministic lex tie-breaking
        best_pair = max(pair_counts.keys(), key=pair_score)
        best_count = pair_counts[best_pair]
        if best_count <= 0:
            break

        a, b = best_pair

        # Create a new token for this merge
        new_id = len(id_to_bytes)
        new_bytes = id_to_bytes[a] + id_to_bytes[b]
        id_to_bytes.append(new_bytes)
        merges.append((id_to_bytes[a], id_to_bytes[b]))

        # Words currently containing this pair
        affected_words = pair_to_words.pop(best_pair, set())
        if not affected_words:
            # No words actually contain it (should be rare, but be safe)
            pair_counts.pop(best_pair, None)
            continue

        for w in list(affected_words):
            freq = words.pop(w)

            # Decrement counts for all pairs in the old word and remove
            # the old word from pair_to_words.
            if len(w) >= 2:
                for i in range(len(w) - 1):
                    p = (w[i], w[i + 1])
                    pair_counts[p] -= freq
                    if pair_counts[p] <= 0:
                        pair_counts.pop(p, None)
                    s = pair_to_words.get(p)
                    if s is not None:
                        s.discard(w)
                        if not s:
                            pair_to_words.pop(p, None)

            # Build the new word by merging every occurrence of (a, b)
            new_word_ids: List[int] = []
            i = 0
            while i < len(w):
                if i < len(w) - 1 and w[i] == a and w[i + 1] == b:
                    new_word_ids.append(new_id)
                    i += 2
                else:
                    new_word_ids.append(w[i])
                    i += 1
            new_w = tuple(new_word_ids)

            # Update word frequencies
            words[new_w] = words.get(new_w, 0) + freq

            # Update pair stats for the new word
            if len(new_w) >= 2:
                for i in range(len(new_w) - 1):
                    p = (new_w[i], new_w[i + 1])
                    pair_counts[p] += freq
                    pair_to_words[p].add(new_w)

        # Remove best_pair from counts (it no longer exists as a pair)
        pair_counts.pop(best_pair, None)

        if debug:
            debug_trace["num_steps"] += 1  # type: ignore[index]

    vocab = {i: id_to_bytes[i] for i in range(len(id_to_bytes))}

    if debug:
        return vocab, merges, debug_trace
    return vocab, merges
