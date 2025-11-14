"""Main entry point for pico-llm training."""

import random

import numpy as np
import tiktoken
import torch

import dataset
import models
import training
import utils

if __name__ == "__main__":
    # set up argument parser and parse args
    args = utils.parse_args()
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

    # create train/val/test dataloaders
    train_dataloader, val_dataloader, test_dataloader = dataset.create_dataloaders(
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

    # initialize model
    model = models.init_model(args, vocab_size, device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Instantiated {args.model} / params: {total_params / 10**6:.2f}M")

    # initialize trainer
    trainer = training.init_trainer(model, train_dataloader, args)

    # train the model
    trainer.train(
        enc=enc,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        num_epochs=args.num_epochs,
        save_dir=args.save_dir,
        use_wandb=args.use_wandb,
        wandb_entity=args.wandb_entity,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
        upload_model_to_hub=args.upload_model_to_hub,
        repo_id=args.repo_id,
        log_interval_steps=args.log_interval_steps,
        save_interval_steps=args.save_interval_steps,
        save_latest=args.save_latest,
        save_best=args.save_best,
        loss_metric_for_best_model=args.loss_metric_for_best_model,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        top_p=args.top_p,
        monosemantic_analysis=args.monosemantic_analysis,
    )
