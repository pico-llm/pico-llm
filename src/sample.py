"""Script to sample sentences from trained Hugging Face models."""

import argparse
import json
import os
import random

import numpy as np
import tiktoken
import torch
from huggingface_hub import HfApi
from tqdm.auto import tqdm

import dataset
import models
import utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample sentences from the HF models.")
    parser.add_argument(
        "--encoding-name",
        type=str,
        default="gpt2",
        help="Name of the tiktoken encoding to use. Default='gpt2'.",
    )
    parser.add_argument(
        "--input-files",
        nargs="*",
        default=None,
        help="Optional list of text files to mix in as data sources. Each line is one example (up to block_size).",
    )
    parser.add_argument(
        "--dataset-subset-size",
        type=int,
        help="Number of dataset sequences to use. Default=None (use all data).",
    )
    parser.add_argument(
        "--block-size", type=int, default=1024, help="Maximum sequence length for each example. Default=1024."
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        choices=["fixed", "mixed"],
        default="fixed",
        help="Type of dataset to use: 'fixed' for deterministic splits, 'mixed' for random sampling. Default='fixed'.",
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.9, help="Ratio of data to use for training. Default=0.9."
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.05, help="Ratio of data to use for validation. Default=0.05."
    )
    parser.add_argument(
        "--number",
        "-n",
        type=int,
        default=10,
        help="Number of sentences to sample.",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training. Default=16.")
    parser.add_argument(
        "--top-p",
        nargs="+",
        type=float,
        default=[0.9, 1.0],
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
        help="Maximum number of new tokens to generate during training samples. Default=100.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="sampled_sentences.json",
        help="Output file to save the sampled sentences.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run the sampling on.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()
    print(args)

    # set device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # set random seed for reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # init tokenizer/encoding
    enc = tiktoken.get_encoding(args.encoding_name)
    vocab_size = enc.n_vocab

    # load test dataloader
    _, _, test_dataloader = dataset.create_dataloaders(
        dataset_subset_size=args.dataset_subset_size,
        input_files=args.input_files,
        block_size=args.block_size,
        enc=enc,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        dataset_type=args.dataset_type,
        seed=args.seed,
    )

    # create list of prompts from test dataset
    prompts = list()
    # sample random number of sequences
    indices = random.sample(range(len(test_dataloader.dataset)), args.number)
    for idx in indices:
        # get a sequence from the test dataset
        sequence = test_dataloader.dataset[idx]
        # use first K tokens as prompt
        prompt_length = min(10, len(sequence))  # TODO: Add as argument
        seq = sequence[:prompt_length].tolist()
        prompts.append(enc.decode(seq))
    print(f"Created {len(prompts)} prompts from test dataset")

    # initialize HF API
    api = HfApi(token=os.getenv("HF_TOKEN"))

    # list models by pico-llm
    all_models = list(api.list_models(author="pico-llm", limit=None))

    # collect generated samples from each model
    samples = dict()
    # models x number of sequences per model x (top-p values + 1 for greedy)
    progress_bar = tqdm(total=len(list(all_models)) * args.number * (len(args.top_p) + 1), desc="Sampling")
    for m in all_models:
        try:
            progress_bar.write(f"Sampling from model: {m.modelId}")
            # determine model type from modelId
            if "transformer" in m.modelId or "gpt2" in m.modelId:
                args.model = "transformer"
            elif "lstm" in m.modelId:
                args.model = "lstm"
            elif "kgram" in m.modelId:
                args.model = "kgram_mlp"
            else:
                progress_bar.write(f"Skipping model: {m.modelId}")
                continue
            # set checkpoint to HF model repo
            args.checkpoint = m.modelId
            model = models.init_model(args, vocab_size, device)
            model.eval()

            # create a list to hold samples for this model
            samples[m.modelId] = list()

            # generate samples for each prompt
            for prompt_text in prompts:
                # create a dictionary to hold samples for this prompt
                prompt_samples = {"prompt_text": prompt_text, "completions": dict()}
                # greedy sampling
                generated_text_greedy, generated_annotated_greedy = utils.generate(
                    model=model,
                    enc=enc,
                    prompt_text=prompt_text,
                    max_new_tokens=args.max_new_tokens,
                    top_p=None,
                    monosemantic_analysis=True,
                    top_n=3,
                )
                prompt_samples["completions"]["greedy"] = {
                    "text": generated_text_greedy,
                    "annotated": generated_annotated_greedy,
                }
                progress_bar.update(1)

                # nucleus sampling for each top-p value
                for top_p in args.top_p:
                    generated_text_top_p, generated_annotated_top_p = utils.generate(
                        model=model,
                        enc=enc,
                        prompt_text=prompt_text,
                        max_new_tokens=args.max_new_tokens,
                        top_p=top_p,
                        monosemantic_analysis=True,
                        top_n=3,
                    )
                    prompt_samples["completions"][f"top_p={top_p}"] = {
                        "text": generated_text_top_p,
                        "annotated": generated_annotated_top_p,
                    }
                    progress_bar.update(1)

                # add prompt samples to model samples
                samples[m.modelId].append(prompt_samples)

            # clean up model to free memory
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            progress_bar.write(f"Error sampling from model {m.modelId}: {e}")
            continue

    progress_bar.close()

    # save samples to disk
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=4, ensure_ascii=False)
    print(f"Saved sampled sentences to {args.output}")
