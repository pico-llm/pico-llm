"""Training module for language models."""

import json

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
        seq_len, batch_size, vocab_size = logits.shape
        if seq_len < 2:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        preds = logits[:-1, :, :]  # (seq_len-1, batch, vocab_size)
        gold = tokens[1:, :]  # (seq_len-1, batch)
        preds = preds.reshape(-1, vocab_size)
        gold = gold.reshape(-1)
        return F.cross_entropy(preds, gold)

    def train(  # noqa: C901
        self,
        model: nn.Module,
        enc: tiktoken.Encoding,
        train_dataloader: DataLoader,
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
        save_model_name: str = "model",
        save_latest: bool = False,
        save_best: bool = True,
        prompt: str = "Once upon",
        max_new_tokens: int = 50,
        top_p: float = 0.9,
        monosemantic_analysis: bool = False,
    ) -> None:
        """Train the language model.

        Args:
            model (nn.Module): The language model to be trained.
            enc (tiktoken.Encoding): The tokenizer encoding.
            train_dataloader (DataLoader): DataLoader for training data.
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
            save_model_name (str): Base name for the saved model file.
            save_latest (bool): If True, overwrite the latest checkpoint instead of saving per save steps.
            save_best (bool): If True, track and save the best model checkpoint based on training loss.
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
                config=self.get_model_attrs(model),
            )

        # clone Hugging Face Hub repository if needed
        if upload_model_to_hub:
            self.init_hf_api()
            self.clone_hub_repository_into_save_dir(repo_id, save_dir)

        # Set up tracking variables
        global_step = 0
        steps_per_epoch = len(train_dataloader)
        total_steps = num_epochs * steps_per_epoch
        progress_bar = tqdm(total=total_steps)
        best_train_loss = float("inf") if save_best else None

        # reset model to training mode
        model.train()
        model.zero_grad()
        torch.cuda.empty_cache()
        device = next(model.parameters()).device

        # training loop
        for epoch in range(num_epochs):
            total_epoch_loss = 0.0
            epoch_step = 0
            for batch_idx, batch_tokens in enumerate(train_dataloader):
                # track steps
                epoch_step += 1
                global_step += 1

                # zero gradients
                model.train()
                self.optimizer.zero_grad()

                # move batch to device
                batch_tokens = batch_tokens.to(device)  # (seq_len, batch)

                # create log dict
                log_dict = {
                    "epoch": float(f"{epoch + (epoch_step / steps_per_epoch):.2f}"),
                    "lr": self.optimizer.param_groups[0]["lr"],
                }

                # forward pass
                logits = model(batch_tokens)  # (seq_len, batch, vocab_size)
                loss = self.compute_next_token_loss(logits, batch_tokens)

                # backward pass
                loss.backward()
                total_epoch_loss += loss.item()
                log_dict["loss"] = loss.item()

                # clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # optimizer step
                self.optimizer.step()

                # log metrics
                if global_step % log_interval_steps == 0:
                    # generate sample text using greedy decoding
                    text_greedy, annotated_greedy = utils.generate(
                        model,
                        enc,
                        prompt,
                        max_new_tokens,
                        top_p=None,
                        monosemantic_analysis=monosemantic_analysis,
                    )
                    # generate sample text using nucleus sampling with top_p
                    text_nucleus_top_p, annotated_nucleus_top_p = utils.generate(
                        model,
                        enc,
                        prompt,
                        max_new_tokens,
                        top_p=top_p,
                        monosemantic_analysis=monosemantic_analysis,
                    )
                    # generate sample text using nucleus sampling with top_p=1.0
                    text_nucleus_top_p_1, annotated_nucleus_top_p_1 = utils.generate(
                        model,
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
                            "lr": self.optimizer.param_groups[0]["lr"],
                            "text_greedy": text_greedy,
                            "annotated_greedy": annotated_greedy,
                            f"text_nucleus_{top_p}": text_nucleus_top_p,
                            f"annotated_nucleus_{top_p}": annotated_nucleus_top_p,
                            "text_nucleus_1.0": text_nucleus_top_p_1,
                            "annotated_nucleus_1.0": annotated_nucleus_top_p_1,
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
                    self.save_training_state_locally(
                        save_dir, model, global_step, save_model_name, save_latest=save_latest
                    )
                    if upload_model_to_hub:
                        self.upload_model_to_hub(repo_id, save_dir, global_step)

            # end of epoch logging
            avg_epoch_loss = total_epoch_loss / epoch_step
            log_str = json.dumps(
                {
                    "epoch": epoch + 1,
                    "step": global_step,
                    "avg_loss": avg_epoch_loss,
                    "lr": self.optimizer.param_groups[0]["lr"],
                }
            )
            progress_bar.write(log_str)
            if self.wandb_writer is not None:
                self.write_losses_to_wandb(global_step, {"avg_loss": avg_epoch_loss})

            # save best model checkpoint based on training loss
            if save_best and best_train_loss is not None and avg_epoch_loss < best_train_loss:
                best_train_loss = avg_epoch_loss
                self.save_training_state_locally(
                    save_dir, model, global_step, f"{save_model_name}_best", save_latest=True
                )
                if upload_model_to_hub:
                    self.upload_model_to_hub(repo_id, save_dir, global_step)

            # step the scheduler
            self.scheduler.step()

        # close progress bar
        progress_bar.close()

        # final model save
        self.save_training_state_locally(save_dir, model, global_step, save_model_name, save_latest=save_latest)
        if upload_model_to_hub:
            self.upload_model_to_hub(repo_id, save_dir, global_step)
