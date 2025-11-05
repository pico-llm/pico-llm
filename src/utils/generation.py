"""Utility functions for text generation with language models."""

import tiktoken
import torch
import torch.nn as nn


def nucleus_sampling(logits: torch.Tensor, p: float = 0.95) -> int:
    """Perform nucleus (top-p) sampling to select the next token.

    Args:
        logits (torch.Tensor): The logits from the model for the next token.
        p (float): Cumulative probability threshold for nucleus sampling.

    Returns:
        int: The selected token ID.
    """
    # convert logits to probabilities
    probs = torch.softmax(logits, dim=-1)
    # sort the probabilities and their corresponding token indices
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    # compute cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    # determine the cutoff index where cumulative probability exceeds p
    cutoff_index = torch.searchsorted(cumulative_probs, p).item()
    # keep tokens up to and including the cutoff index (nucleus)
    # we add 1 because we want to include the token that crosses the threshold
    nucleus_size = max(cutoff_index + 1, 1)
    # extract the nucleus probabilities and indices
    nucleus_probs = sorted_probs[:nucleus_size].clone()
    nucleus_indices = sorted_indices[:nucleus_size]
    # re-normalize the nucleus probabilities
    nucleus_probs /= nucleus_probs.sum()
    # sample from the nucleus
    chosen_index = torch.multinomial(nucleus_probs, num_samples=1).item()
    # return the corresponding token ID
    return nucleus_indices[chosen_index].item()


def monosemantic_analysis_for_token(token_id: int, model: nn.Module, enc: tiktoken.Encoding, top_n: int) -> list:
    """Perform monosemantic analysis for a given token.

    Args:
        token_id (int): The token ID to analyze.
        model (nn.Module): The language model.
        enc (tiktoken.Encoding): The tokenizer encoding.
        top_n (int): Number of nearest neighbors to retrieve.

    Returns:
        list: List of tuples containing (distance, token_id) of nearest neighbors.
    """
    if not hasattr(model, "embedding"):
        raise ValueError("Model does not support token embeddings for monosemantic analysis.")

    embedding_layer = model.embedding
    vocab_size = enc.n_vocab
    device = next(model.parameters()).device
    with torch.no_grad():
        # get the embedding vector for the given token
        token_tensor = torch.tensor([token_id], dtype=torch.long, device=device)
        token_embedding = embedding_layer(token_tensor).squeeze(0)  # shape (embed_size,)
        # normalize for cosine similarity
        token_embedding = torch.nn.functional.normalize(token_embedding, dim=0)
        # get all embeddings
        all_token_ids = torch.arange(vocab_size, dtype=torch.long, device=device)
        all_embeddings = embedding_layer(all_token_ids)  # shape (vocab_size, embed_size)
        all_embeddings = torch.nn.functional.normalize(all_embeddings, dim=1)
        # compute cosine similarities
        similarities = torch.matmul(all_embeddings, token_embedding)  # shape (vocab_size,)
        # get top_n nearest neighbors (excluding the token itself)
        similarities[token_id] = -1.0  # exclude self
        # get top_n indices
        top_similarities, top_indices = torch.topk(similarities, k=top_n, largest=True)
        # convert similarities to distances
        top_distances = 1.0 - top_similarities
        return [(top_distances[i].item(), top_indices[i].item()) for i in range(top_n)]


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
    was_training = model.training
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
                neighbors = monosemantic_analysis_for_token(chosen_token, model, enc, top_n)
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
    model.train(was_training)
    return final_text, annotated_text
