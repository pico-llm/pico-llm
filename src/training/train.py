"""Training module for language models."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
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

    def evaluate(self, dataloader: DataLoader, device: torch.device) -> dict[str, float]:
        """Evaluate the model on a dataloader.

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

                # Loss is already averaged over all tokens in the batch
                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        return {"loss": avg_loss, "perplexity": perplexity}

    @staticmethod
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
            # Tokenize text using the same tokenizer
            tokens = enc.encode(text)
            # Extract n-grams
            for i in range(len(tokens) - n + 1):
                ngram = tuple(tokens[i : i + n])
                all_ngrams.append(ngram)

        if len(all_ngrams) == 0:
            return 0.0

        unique_ngrams = len(set(all_ngrams))
        return unique_ngrams / len(all_ngrams)

    def evaluate_diversity(
        self,
        enc: tiktoken.Encoding,
        prompts: list[str],
        max_new_tokens: int = 50,
        top_p: float = 0.9,
    ) -> dict[str, float]:
        """Evaluate diversity metrics (distinct-1, distinct-2, distinct-3) by generating text.

        Args:
            enc (tiktoken.Encoding): The tokenizer encoding.
            prompts (list[str]): List of prompt texts to generate from.
            max_new_tokens (int): Maximum number of new tokens to generate per prompt.
            top_p (float): Nucleus sampling probability for generation.

        Returns:
            dict[str, float]: Dictionary containing distinct-1, distinct-2, distinct-3 metrics.
        """
        self.model.eval()
        device = next(self.model.parameters()).device
        generated_texts = []

        with torch.no_grad():
            for prompt in prompts:
                # Generate text using nucleus sampling
                generated_text, _ = utils.generate(
                    self.model,
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
            return {"distinct_1": 0.0, "distinct_2": 0.0, "distinct_3": 0.0}

        distinct_1 = self.compute_distinct_n(generated_texts, enc, n=1)
        distinct_2 = self.compute_distinct_n(generated_texts, enc, n=2)
        distinct_3 = self.compute_distinct_n(generated_texts, enc, n=3)

        return {"distinct_1": distinct_1, "distinct_2": distinct_2, "distinct_3": distinct_3}

    @staticmethod
    def save_loss_plot(
        save_dir: str | Path,
        train_epochs: list[int],
        train_losses: list[float],
        val_epochs: list[int],
        val_losses: list[float],
    ) -> Path:
        """Save loss curves for training and validation NLL."""
        save_dir_path = Path(save_dir)
        save_dir_path.mkdir(parents=True, exist_ok=True)
        plot_path = save_dir_path / "loss_curve.png"

        plt.figure(figsize=(8, 5))
        if train_losses:
            plt.plot(train_epochs, train_losses, marker="o", label="Train NLL")
        if val_losses:
            plt.plot(val_epochs, val_losses, marker="s", label="Val NLL")

        plt.xlabel("Epoch")
        plt.ylabel("NLL Loss")
        plt.title("Training and Validation NLL Loss")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

        return plot_path

    def train(  # noqa: C901
        self,
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
        save_latest: bool = False,
        save_best: bool = True,
        prompt: str = "Once upon",
        max_new_tokens: int = 50,
        top_p: float = 0.9,
        monosemantic_analysis: bool = False,
        val_dataloader: DataLoader | None = None,
        eval_interval_epochs: int = 1,
    ) -> None:
        """Train the language model.

        Args:
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
            save_latest (bool): If True, overwrite the latest checkpoint instead of saving per save steps.
            save_best (bool): If True, track and save the best model checkpoint based on validation loss (if available) or training loss.
            prompt (str): Prompt text for generating sample outputs during training.
            max_new_tokens (int): Maximum number of new tokens to generate for sample outputs.
            top_p (float): Nucleus sampling probability for generating sample outputs.
            monosemantic_analysis (bool): Whether to perform monosemantic analysis during generation.
            val_dataloader (DataLoader | None): Optional DataLoader for validation data.
            eval_interval_epochs (int): Evaluate on validation set every N epochs. Default=1.

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

        train_loss_history: list[float] = []
        train_epoch_indices: list[int] = []
        val_loss_history: list[float] = []
        val_epoch_indices: list[int] = []

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
            log_dict_epoch = {
                "epoch": epoch + 1,
                "step": global_step,
                "train_loss_avg": avg_epoch_loss,  # NLL loss
                "train_perplexity": torch.exp(torch.tensor(avg_epoch_loss)).item(),
                "lr": self.optimizer.param_groups[0]["lr"],
            }
            train_loss_history.append(avg_epoch_loss)
            train_epoch_indices.append(epoch + 1)

            # Evaluate on validation set if provided
            if val_dataloader is not None and (epoch + 1) % eval_interval_epochs == 0:
                val_metrics = self.evaluate(val_dataloader, device)
                log_dict_epoch.update(
                    {
                        "val_loss": val_metrics["loss"],  # NLL loss
                        "val_perplexity": val_metrics["perplexity"],
                    }
                )
                val_loss_history.append(val_metrics["loss"])
                val_epoch_indices.append(epoch + 1)

            log_str = json.dumps(log_dict_epoch)
            progress_bar.write(log_str)
            if self.wandb_writer is not None:
                wandb_log_dict = {"train_loss_avg": avg_epoch_loss, "train_perplexity": log_dict_epoch["train_perplexity"]}
                if val_dataloader is not None and (epoch + 1) % eval_interval_epochs == 0:
                    wandb_log_dict.update({"val_loss": val_metrics["loss"], "val_perplexity": val_metrics["perplexity"]})
                self.write_losses_to_wandb(global_step, wandb_log_dict)

            # save best model checkpoint based on NLL loss (validation loss if available, otherwise training loss)
            if save_best and best_loss is not None:
                # Use validation NLL loss if available, otherwise use training NLL loss
                if val_dataloader is not None and (epoch + 1) % eval_interval_epochs == 0:
                    current_loss = val_metrics["loss"]  # NLL loss
                else:
                    current_loss = avg_epoch_loss  # NLL loss

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

        # save loss curve plot
        loss_plot_path = self.save_loss_plot(
            save_dir,
            train_epoch_indices,
            train_loss_history,
            val_epoch_indices,
            val_loss_history,
        )
        print(f"Saved loss curves to {loss_plot_path}")

        # final model save
        self.save_pretrained(save_directory=Path(save_dir) / "final_model")
        if upload_model_to_hub:
            self.push_to_hub(
                repo_id=repo_id,
                commit_message=f"Final model at Step {global_step}",
            )
