"""Utility functions to compute metrics for generated text."""

import tiktoken
import torch
from torch import nn

from utils.generation import generate


def compute_distinct_n(generated_texts: list[str], enc: tiktoken.Encoding, n: int) -> float:
    """Compute distinct-n metric for generated texts.

    distinct-n = number of unique n-grams / total number of n-grams

    Args:
        generated_texts (list[str]): List of generated text strings.
        enc (tiktoken.Encoding): Tokenizer encoding to tokenize text.
        n (int): N-gram size (1, 2, or 3).

    Returns:
        float: distinct-n score.
    """
    all_ngrams = []
    for text in generated_texts:
        # tokenize text
        tokens = enc.encode(text)
        # extract n-grams
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i : i + n])
            all_ngrams.append(ngram)

    if len(all_ngrams) == 0:
        return 0.0
    unique_ngrams = len(set(all_ngrams))
    return unique_ngrams / len(all_ngrams)


def compute_diversity(
    enc: tiktoken.Encoding,
    model: nn.Module,
    prompts: list[str],
    n: int,
    max_new_tokens: int = 50,
    top_p: float = 0.9,
) -> dict[str, float]:
    """Evaluate diversity metrics (distinct-1, distinct-2, distinct-3) by generating text.

    Args:
        enc (tiktoken.Encoding): The tokenizer encoding.
        model (nn.Module): The language model to use for generation.
        prompts (list[str]): List of prompt texts to generate from.
        n (int): Maximum n-gram size to compute distinct-n metrics for.
        max_new_tokens (int): Maximum number of new tokens to generate per prompt.
        top_p (float): Nucleus sampling probability for generation.

    Returns:
        dict[str, float]: Dictionary containing distinct-1, distinct-2, distinct-3 metrics.
    """
    model.eval()
    generated_texts = []
    with torch.no_grad():
        for prompt in prompts:
            # Generate text using nucleus sampling
            generated_text, _ = generate(
                model,
                enc,
                prompt,
                max_new_tokens=max_new_tokens,
                top_p=top_p,
                monosemantic_analysis=False,
            )
            # Extract only the generated part (remove prompt)
            generated_part = generated_text[len(prompt) :].strip()
            if generated_part:
                generated_texts.append(generated_part)

    if len(generated_texts) == 0:
        return {f"distinct_{i}": 0.0 for i in range(1, n + 1)}

    return {f"distinct_{i}": compute_distinct_n(generated_texts, enc, n=i) for i in range(1, n + 1)}
