"""Utility functions for text generation with language models."""

import tiktoken
import torch
import torch.nn as nn


def nucleus_sampling(logits: torch.Tensor, p: float = 0.95) -> torch.Tensor:
    """Perform nucleus (top-p) sampling to select the next token.

    Args:
        logits (torch.Tensor): The logits from the model for the next token.
        p (float): Cumulative probability threshold for nucleus sampling.

    Returns:
        torch.Tensor: The selected token ID.
    """
    # TODO: Implement nucleus sampling logic here. For now, we return the argmax as a placeholder.
    return torch.argmax(logits).item()


def monosemantic_analysis(token_id: int, model: nn.Module, enc: tiktoken.Encoding, top_n: int) -> list:
    """Perform monosemantic analysis for a given token.

    Args:
        token_id (int): The token ID to analyze.
        model (nn.Module): The language model.
        enc (tiktoken.Encoding): The tokenizer encoding.
        top_n (int): Number of nearest neighbors to retrieve.

    Returns:
        list: List of tuples containing (distance, token_id) of nearest neighbors.
    """
    # TODO: Implement monosemantic analysis logic here. For now, we return an empty list as a placeholder.
    return []


def generate(
    model: nn.Module,
    enc: tiktoken.Encoding,
    prompt_text: str,
    max_new_tokens: int = 20,
    top_p: float | None = None,
    monosemantic_analysis: bool = False,
    top_n: int = 5,
) -> tuple[str, str]:
    """Generate text using the provided language model.

    - We keep a growing list 'context_tokens'.
    - At each step, we feed the entire context as (seq_len,1) to model(...).
    - We get model(...)->(seq_len,1,vocab_size). We take the final step's logits => logits[-1,0,:].
    - We pick next token (greedy or top-p), append to context_tokens.
    - Optionally do monosemantic analysis on that newly generated token.

    Args:
        model (nn.Module): The language model to use for generation.
        enc (tiktoken.Encoding): The tokenizer encoding.
        prompt_text (str): The initial text prompt to start generation.
        max_new_tokens (int): Maximum number of new tokens to generate.
        top_p (float | None): If provided, use nucleus sampling with this probability.
        monosemantic_analysis (bool): Whether to perform monosemantic analysis.
        top_n (int): Number of nearest neighbors to retrieve for monosemantic analysis.

    Returns:
        tuple[str, str]: A tuple containing the final generated text and the annotated text.
    """
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        # encode prompt
        context_tokens = enc.encode(prompt_text)
        annotation_list = []

        for _ in range(max_new_tokens):
            # prepare input tensor
            seq_tensor = torch.tensor(context_tokens, dtype=torch.long, device=device).unsqueeze(1)
            logits_seq = model(seq_tensor)  # (seq_len,1,vocab_size)
            # get logits for the next token
            next_logits = logits_seq[-1, 0, :]  # shape (vocab_size,)

            if top_p is None:
                # greedy
                chosen_token = torch.argmax(next_logits).item()
            else:
                # nucleus sampling
                chosen_token = nucleus_sampling(next_logits, p=top_p)
            # append chosen token to context
            context_tokens.append(chosen_token)

            if monosemantic_analysis:  # perform monosemantic analysis
                # get nearest neighbors
                neighbors = monosemantic_analysis(chosen_token, model, enc, top_n)
                annotation_list.append((chosen_token, neighbors))
            else:
                annotation_list.append((chosen_token, []))

    # decode final text and annotated text
    final_text = enc.decode(context_tokens)
    prefix_text = enc.decode(context_tokens[:-max_new_tokens])
    annotated_strs = [prefix_text]
    for tid, neighs in annotation_list:
        token_str = enc.decode([tid])
        if neighs:
            # annotate with nearest neighbors
            neighbor_strs = [f"{enc.decode([x[1]])}" for x in neighs]
            annotated = f"{token_str}[NN={neighbor_strs}]"
        else:
            annotated = token_str
        annotated_strs.append(annotated)

    annotated_text = "".join(annotated_strs)
    return final_text, annotated_text
