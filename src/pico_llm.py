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

    # create train/val/test dataloaders with deterministic 80/10/10 split
    train_dataloader, val_dataloader, test_dataloader = dataset.create_dataloaders(
        tinystories_weight=args.tinystories_weight,
        dataset_subset_size=args.dataset_subset_size,
        input_files=args.input_files,
        block_size=args.block_size,
        enc=enc,
        batch_size=args.batch_size,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
    )

    # initialize model
    model = models.init_model(args, vocab_size, device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Instantiated {args.model} / params: {total_params // 10**6}M")

    # initialize trainer
    trainer = training.init_trainer(model, train_dataloader, args)

    # train the model
    trainer.train(
        enc=enc,
        train_dataloader=train_dataloader,
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
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        top_p=args.top_p,
        monosemantic_analysis=args.monosemantic_analysis,
        val_dataloader=val_dataloader,
        eval_interval_epochs=1,
    )

    # Evaluate on test set at the end
    print("\nEvaluating on test set...")
    test_metrics = trainer.evaluate(test_dataloader, device)
    print(f"Test NLL Loss: {test_metrics['loss']:.4f}")
    print(f"Test Perplexity: {test_metrics['perplexity']:.4f}")

    # Compute diversity metrics on test set
    print("\nComputing diversity metrics...")
    # Extract prompts from test set (first 50 sequences, use first 10 tokens as prompt)
    test_prompts = []
    test_dataset = test_dataloader.dataset
    num_prompts = min(50, len(test_dataset))
    
    for i in range(num_prompts):
        tokens = test_dataset[i].tolist()
        # Use first 10 tokens as prompt (or fewer if sequence is shorter)
        prompt_length = min(10, len(tokens) - 1)
        if prompt_length > 0:
            prompt_tokens = tokens[:prompt_length]
            prompt_text = enc.decode(prompt_tokens)
            test_prompts.append(prompt_text)

    diversity_metrics = trainer.evaluate_diversity(
        enc=enc,
        prompts=test_prompts,
        max_new_tokens=50,
        top_p=0.9,
    )
    print(f"Distinct-1: {diversity_metrics['distinct_1']:.4f}")
    print(f"Distinct-2: {diversity_metrics['distinct_2']:.4f}")
    print(f"Distinct-3: {diversity_metrics['distinct_3']:.4f}")
