"""Main entry point for pico-llm training and evaluation."""

import random

import numpy as np
import tiktoken
import torch

import dataset
import models
import trainer
import utils

if __name__ == "__main__":
    # set up argument parser and parse args
    args = utils.parse_args()
    # TODO: Move all prints to logs
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
    print(f"Vocab size: {vocab_size}")

    # create train dataloader
    train_dataloader = dataset.create_train_dataloader(
        tinystories_weight=args.tinystories_weight,
        train_subset_size=args.train_subset_size,
        input_files=args.input_files,
        block_size=args.block_size,
        enc=enc,
        batch_size=args.batch_size,
    )

    # initialize model
    model = models.init_model(args, vocab_size, device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Instantiated {args.model} / params: {total_params//10**6}M")

    # initialize trainer
    trainer = trainer.Trainer(
        model=model,
        learning_rate=args.learning_rate,
        optimizer_class=args.optimizer,
        scheduler_class=args.scheduler,
        device=device,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        output_dir=args.output_dir,
    )
    print(trainer)



