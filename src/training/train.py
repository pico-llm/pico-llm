"""Training module for language models."""

import json
from pathlib import Path

import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import utils

from .base import BaseTrainer


class Trainer(BaseTrainer):
    """Trainer class for language model training."""

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float,
        optimizer_class: str = "adamw",
        scheduler_class: str = "cosine",
        **kwargs: dict,
    ) -> "Trainer":
        """Initialize the Trainer.

        Args:
            model (nn.Module): The language model to be trained.
            learning_rate (float): Learning rate for the optimizer.
            optimizer_class (str): Optimizer type.
            scheduler_class (str): Learning rate scheduler type.
            **kwargs (dict): Additional arguments for the base trainer.

        Returns:
            Trainer: An instance of the Trainer class.
        """
        super().__init__(model, learning_rate, optimizer_class, scheduler_class, **kwargs)

    def get_model_attrs(self, model: nn.Module) -> dict:
        """Get model attributes for logging.

        Args:
            model (nn.Module): The language model.

        Returns:
            dict: A dictionary of model attributes.
        """
        attrs = model.__dict__.copy()
        attrs.update({"model_type": type(model).__name__})
        attrs.update({"model_class": model.__class__.__name__})
        attrs.update({"model_module": model.__class__.__module__})
        attrs.update({"total_params": sum(p.numel() for p in model.parameters())})
        return attrs

    @staticmethod
    def compute_next_token_loss(logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        """Compute next-token prediction loss.

        Next-token prediction => we shift target by 1.

        Args:
            logits (torch.Tensor): Model output logits of shape (seq_len, batch, vocab_size).
            tokens (torch.Tensor): Input tokens of shape (seq_len, batch).

        Returns:
            torch.Tensor: Computed cross-entropy loss.
        """
        seq_len, _, vocab_size = logits.shape
        if seq_len < 2:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        preds = logits[:-1, :, :]  # (seq_len-1, batch, vocab_size)
        gold = tokens[1:, :]  # (seq_len-1, batch)
        preds = preds.reshape(-1, vocab_size)
        gold = gold.reshape(-1)
        return F.cross_entropy(preds, gold)

    def evaluate(self, dataloader: DataLoader, device: torch.device) -> dict[str, float]:
        """Run evaluation on the given dataloader.

        Args:
            dataloader (DataLoader): DataLoader for evaluation data.
            device (torch.device): Device to run evaluation on.

        Returns:
            dict[str, float]: Dictionary containing evaluation metrics (loss, perplexity).
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch_tokens in dataloader:
                batch_tokens = batch_tokens.to(device)  # (seq_len, batch)
                logits = self.model(batch_tokens)  # (seq_len, batch, vocab_size)
                loss = self.compute_next_token_loss(logits, batch_tokens)
                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        return {"loss": avg_loss, "perplexity": perplexity}

    def train(  # noqa: C901
        self,
        enc: tiktoken.Encoding,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader | None,
        test_dataloader: DataLoader | None,
        num_epochs: int,
        save_dir: str = "./saved_models",
        use_wandb: bool = False,
        wandb_entity: str = None,
        wandb_project: str = None,
        wandb_name: str = None,
        upload_model_to_hub: bool = False,
        repo_id: str = None,
        log_interval_steps: int = 100,
        save_interval_steps: int = 200,
        save_latest: bool = False,
        save_best: bool = True,
        loss_metric_for_best_model: str = "val",
        prompt: str = "Once upon",
        max_new_tokens: int = 50,
        top_p: float = 0.9,
        monosemantic_analysis: bool = False,
    ) -> None:
        """Train the language model.

        Args:
            enc (tiktoken.Encoding): The tokenizer encoding.
            train_dataloader (DataLoader): DataLoader for training data.
            val_dataloader (DataLoader | None): DataLoader for validation data.
            test_dataloader (DataLoader | None): DataLoader for test data.
            num_epochs (int): Number of training epochs.
            save_dir (str): Directory to save model checkpoints.
            use_wandb (bool): Whether to use Weights & Biases for logging.
            wandb_entity (str): Weights & Biases entity name.
            wandb_project (str): Weights & Biases project name.
            wandb_name (str): Weights & Biases run name.
            upload_model_to_hub (bool): Whether to upload the model to Hugging Face Hub
            repo_id (str): Hugging Face Hub repository ID.
            log_interval_steps (int): Steps interval for logging training loss.
            save_interval_steps (int): Steps interval for saving model checkpoints.
            save_latest (bool): If True, overwrite the latest checkpoint instead of saving per save steps.
            save_best (bool): If True, track and save the best model checkpoint.
            loss_metric_for_best_model (str): Metric to use for best model tracking ("val" or "train").
            prompt (str): Prompt text for generating sample outputs during training.
            max_new_tokens (int): Maximum number of new tokens to generate for sample outputs.
            top_p (float): Nucleus sampling probability for generating sample outputs.
            monosemantic_analysis (bool): Whether to perform monosemantic analysis during generation.

        Returns:
            None
        """
        # initialize logging (e.g., Weights & Biases)
        if use_wandb:
            self.init_wandb(
                entity=wandb_entity,
                project=wandb_project,
                name=wandb_name,
                config=self.get_model_attrs(self.model),
            )

        # clone Hugging Face Hub configuration
        if upload_model_to_hub:
            self.init_hf_api()

        # Set up tracking variables
        global_step = 0
        steps_per_epoch = len(train_dataloader)
        total_steps = num_epochs * steps_per_epoch
        progress_bar = tqdm(total=total_steps)
        best_loss = float("inf") if save_best else None

        # reset model to training mode
        self.model.train()
        self.model.zero_grad()
        torch.cuda.empty_cache()
        device = next(self.model.parameters()).device

        # training loop
        for epoch in range(num_epochs):
            total_epoch_loss = 0.0
            epoch_step = 0
            for batch_tokens in train_dataloader:
                # track steps
                epoch_step += 1
                global_step += 1

                # zero gradients
                self.model.train()
                self.optimizer.zero_grad()

                # move batch to device
                batch_tokens = batch_tokens.to(device)  # (seq_len, batch)

                # create log dict
                log_dict = {
                    "epoch": float(f"{epoch + (epoch_step / steps_per_epoch):.2f}"),
                    "lr": self.optimizer.param_groups[0]["lr"],
                }

                # forward pass
                logits = self.model(batch_tokens)  # (seq_len, batch, vocab_size)
                loss = self.compute_next_token_loss(logits, batch_tokens)

                # backward pass
                loss.backward()
                total_epoch_loss += loss.item()
                log_dict["loss"] = loss.item()
                log_dict["perplexity"] = torch.exp(loss).item()

                # clip gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                # optimizer step
                self.optimizer.step()

                # step the scheduler (note: we are stepping every batch)
                self.scheduler.step()

                # log metrics
                if global_step % log_interval_steps == 0:
                    # generate sample text using greedy decoding
                    text_greedy, annotated_greedy = utils.generate(
                        self.model,
                        enc,
                        prompt,
                        max_new_tokens,
                        top_p=None,
                        monosemantic_analysis=monosemantic_analysis,
                    )
                    # generate sample text using nucleus sampling with top_p
                    text_nucleus_top_p, annotated_nucleus_top_p = utils.generate(
                        self.model,
                        enc,
                        prompt,
                        max_new_tokens,
                        top_p=top_p,
                        monosemantic_analysis=monosemantic_analysis,
                    )
                    # generate sample text using nucleus sampling with top_p=1.0
                    text_nucleus_top_p_1, annotated_nucleus_top_p_1 = utils.generate(
                        self.model,
                        enc,
                        prompt,
                        max_new_tokens,
                        top_p=1.0,
                        monosemantic_analysis=monosemantic_analysis,
                    )

                    log_str = json.dumps(
                        {
                            "epoch": float(f"{epoch + (epoch_step / steps_per_epoch):.2f}"),
                            "step": global_step,
                            "loss": loss.item(),
                            "perplexity": torch.exp(loss).item(),
                            "lr": self.optimizer.param_groups[0]["lr"],
                        }
                    )
                    progress_bar.write(log_str)
                    if self.wandb_writer is not None:
                        self.write_losses_to_wandb(global_step, log_dict)
                        self.write_decoded_sentences_to_wandb(
                            global_step,
                            prompt,
                            [text_greedy, text_nucleus_top_p, text_nucleus_top_p_1],
                            [annotated_greedy, annotated_nucleus_top_p, annotated_nucleus_top_p_1],
                            top_p=[None, top_p, 1.0],
                        )

                # update progress bar
                progress_bar.set_postfix(
                    {
                        "epoch": float(f"{epoch + (epoch_step / steps_per_epoch):.2f}"),
                        "loss": f"{loss.item():.4f}",
                    }
                )
                progress_bar.update(1)

                # save model checkpoint
                if global_step % save_interval_steps == 0:
                    if not save_latest:
                        checkpoint_path = Path(save_dir) / f"step_{global_step}"
                    else:
                        checkpoint_path = Path(save_dir) / "latest"

                    self.save_pretrained(save_directory=checkpoint_path)
                    if upload_model_to_hub:
                        self.push_to_hub(
                            repo_id=repo_id,
                            commit_message=f"Training Step {global_step}",
                        )

            # end of epoch logging
            avg_epoch_loss = total_epoch_loss / epoch_step
            avg_epoch_perplexity = torch.exp(torch.tensor(avg_epoch_loss)).item()
            log_dict_epoch = {
                "epoch": epoch + 1,
                "step": global_step,
                "train_loss": avg_epoch_loss,
                "train_perplexity": avg_epoch_perplexity,
                "lr": self.optimizer.param_groups[0]["lr"],
            }
            # evaluate on validation set if provided
            val_metrics = None
            if val_dataloader is not None:
                val_metrics = self.evaluate(val_dataloader, device)
                log_dict_epoch.update(
                    {
                        "val_loss": val_metrics["loss"],
                        "val_perplexity": val_metrics["perplexity"],
                    }
                )

            log_str = json.dumps(log_dict_epoch)
            progress_bar.write(log_str)
            if self.wandb_writer is not None:
                wandb_log_dict = {
                    "train_loss": avg_epoch_loss,
                    "train_perplexity": avg_epoch_perplexity,
                }
                if val_metrics is not None:
                    wandb_log_dict.update(
                        {
                            "val_loss": val_metrics["loss"],
                            "val_perplexity": val_metrics["perplexity"],
                        }
                    )
                self.write_losses_to_wandb(global_step, wandb_log_dict)

            # save best model checkpoint based on specified metric
            if save_best and best_loss is not None:
                if loss_metric_for_best_model == "train":
                    current_loss = avg_epoch_loss
                elif loss_metric_for_best_model == "val" and val_dataloader is not None:
                    current_loss = val_metrics["loss"]
                else:
                    raise ValueError("Invalid loss_metric_for_best_model or missing val_loader.")

                if current_loss < best_loss:
                    best_loss = current_loss
                    checkpoint_path = Path(save_dir) / "best_model"
                    self.save_pretrained(save_directory=checkpoint_path)
                    if upload_model_to_hub:
                        self.push_to_hub(
                            repo_id=repo_id,
                            commit_message=f"Best model at Step {global_step}",
                        )

        # close progress bar
        progress_bar.close()

        # final model save
        self.save_pretrained(save_directory=Path(save_dir) / "final_model")
        if upload_model_to_hub:
            self.push_to_hub(
                repo_id=repo_id,
                commit_message=f"Final model at Step {global_step}",
            )

        # evaluate on test set if provided
        if test_dataloader is not None:
            test_metrics = self.evaluate(test_dataloader, device)
            log_dict = {
                "test_loss": test_metrics["loss"],
                "test_perplexity": test_metrics["perplexity"],
            }
            # compute diversity metrics on test set
            diversity_metrics = utils.compute_diversity(
                enc,
                self.model,
                test_dataloader.dataset,
                n=5,
                max_new_tokens=max_new_tokens,
                top_p=top_p,
            )
            test_metrics.update(diversity_metrics)
            progress_bar.write(json.dumps(test_metrics))
            if self.wandb_writer is not None:
                self.write_losses_to_wandb(global_step, test_metrics)

        # close wandb writer
        if self.wandb_writer is not None:
            self.wandb_writer.finish()
