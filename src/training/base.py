"""Base trainer class for model management and logging."""

import os
from pathlib import Path

import torch
import torch.nn as nn
from huggingface_hub import HfApi

import wandb


class BaseTrainer:
    """Base trainer class for model management and logging."""

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float,
        optimizer_class: str = "adamw",
        scheduler_class: str = "cosine",
        **kwargs: dict,
    ) -> "BaseTrainer":
        """Initialize the BaseTrainer.

        Args:
            model (nn.Module): The model to be trained.
            learning_rate (float): Learning rate for the optimizer.
            optimizer_class (str): The optimizer class to use.
            scheduler_class (str): The scheduler class to use.
            **kwargs (dict): Additional keyword arguments for scheduler initialization.

        Returns:
            BaseTrainer: An instance of the BaseTrainer class.
        """
        self.hf_api = None
        self.wandb_writer = None
        self.wandb_table = None
        self.optimizer = None
        self.scheduler = None
        self.optimizer_class = None
        self.scheduler_class = None
        self.learning_rate = learning_rate
        self.init_optimizer(model, learning_rate, optimizer_class)
        self.init_scheduler(scheduler_class, **kwargs)

    def init_optimizer(
        self,
        model: nn.Module,
        learning_rate: float,
        optimizer_class: str,
    ) -> None:
        """Initialize the optimizer.

        Args:
            model (nn.Module): The model to be optimized.
            learning_rate (float): Learning rate for the optimizer.
            optimizer_class (str): The optimizer class to use.

        Returns:
            None
        """
        self.optimizer_class = optimizer_class
        if self.optimizer_class == "adamw":
            self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        elif self.optimizer_class == "adam":
            self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        elif self.optimizer_class == "sgd":
            self.optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        else:
            raise NotImplementedError

    def init_scheduler(self, scheduler_class: str, **kwargs: dict) -> None:
        """Initialize the learning rate scheduler.

        Args:
            scheduler_class (str): The scheduler class to use.
            **kwargs (dict): Additional keyword arguments for scheduler initialization.

        Returns:
            None
        """
        self.scheduler_class = scheduler_class

        # Set up decreasing lr scheduler
        if self.scheduler_class == "plateau":
            decreasing_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.1,
                patience=1,
            )
        elif self.scheduler_class == "cosine":
            decreasing_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=kwargs.get("num_epochs"),
                eta_min=kwargs.get("eta_min", 1e-6),
            )
        elif self.scheduler_class == "exponential":
            decreasing_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=kwargs.get("gamma", 0.95),
            )
        else:
            raise NotImplementedError

        # set up warmup scheduler if specified
        warmup_ratio = kwargs.get("warmup_ratio", 0.0)
        if warmup_ratio > 0.0:
            warmup_epochs = int(kwargs.get("num_epochs") * warmup_ratio)
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
            )
            self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                self.optimizer, schedulers=[warmup_scheduler, decreasing_lr_scheduler], milestones=[warmup_epochs]
            )
        else:
            self.scheduler = decreasing_lr_scheduler

    def init_hf_api(self) -> None:
        """Initialize the Hugging Face API client."""
        self.hf_api = HfApi(token=os.getenv("HF_TOKEN"))

    def init_wandb(
        self,
        entity: str = None,
        project: str = None,
        name: str = None,
        config: dict = None,
    ) -> None:
        """Initialize Weights & Biases (wandb) for experiment tracking.

        Args:
            entity (str): The wandb entity (user or team) to log under.
            project (str): The wandb project name.
            name (str): The wandb run name.
            config (dict): Configuration dictionary to log with wandb.

        Returns:
            None
        """
        wandb.login(key=os.getenv("WANDB_API_KEY"))
        self.wandb_writer = wandb.init(
            entity=entity,
            project=project,
            name=name,
            config=config,
            allow_val_change=True,
        )

    def write_losses_to_wandb(self, step: int, losses: dict) -> None:
        """Log losses to Weights & Biases (wandb).

        Args:
            step (int): Current training step.
            losses (dict): Dictionary of loss values to log.

        Returns:
            None
        """
        if self.wandb_writer is not None:
            self.wandb_writer.log(losses, step=step)

    def write_decoded_sentences_to_wandb(
        self, step: int, prompt: str, completions: list[str], annotations: list[str], top_p: list[float | None]
    ) -> None:
        """Log decoded sentences to Weights & Biases (wandb).

        Args:
            step (int): Current training step.
            prompt (str): The input prompt used for generation.
            completions (list[str]): List of generated completions.
            annotations (list[str]): List of annotated completions.
            top_p (list[float | None]): List of top-p values used for generation.

        Returns:
            None
        """
        if self.wandb_writer is not None:
            if self.wandb_table is None:
                columns = ["step", "top_p", "prompt", "completion", "annotation"]
                self.wandb_table = wandb.Table(columns=columns, log_mode="INCREMENTAL")
            for completion, annotation, p in zip(completions, annotations, top_p):
                self.wandb_table.add_data(step, p, prompt, completion, annotation)
            self.wandb_writer.log({"examples": self.wandb_table}, step=step)

    def save(self, path: str | Path) -> None:
        """Save the trainer state to a checkpoint.

        Args:
            path (str | Path): Path to save the checkpoint.

        Returns:
            None
        """
        if not Path(path).parent.exists():
            raise FileNotFoundError
        torch.save(
            {
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "optimizer_class": self.optimizer_class,
                "scheduler_class": self.scheduler_class,
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        """Load the trainer state from a checkpoint.

        Args:
            path (str | Path): Path to the checkpoint file.

        Returns:
            None
        """
        if not Path(path).exists():
            raise FileNotFoundError
        checkpoint = torch.load(path)
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.optimizer_class = checkpoint["optimizer_class"]
        self.scheduler_class = checkpoint["scheduler_class"]

    def save_training_state_locally(
        self,
        path: str,
        model: nn.Module,
        step: int,
        save_model_name: str = "model",
        save_latest: bool = False,
    ) -> None:
        """Save the model and trainer state locally.

        Args:
            path (str): Directory path to save the model and trainer state.
            model (nn.Module): The model to be saved.
            step (int): Current training step.
            save_model_name (str): Base name for the saved model file.
            save_latest (bool): If True, overwrite the latest checkpoint instead of saving per step.

        Returns:
            None
        """
        if not Path(path).exists():
            Path(path).mkdir(parents=True)

        if not save_latest:
            save_model_name = f"{save_model_name}_{step}.pt"
        else:
            save_model_name = f"{save_model_name}.pt"
        model_save_path = Path(path).joinpath(save_model_name)
        trainer_save_path = Path(path).joinpath("trainer.pt")
        torch.save(model.state_dict(), model_save_path)
        self.save(trainer_save_path)

    def upload_model_to_hub(self, repo_id: str, path: str, step: int, commit_message: str | None = None) -> None:
        """Upload the model folder to Hugging Face Hub.

        Args:
            repo_id (str): The identifier of the repository on Hugging Face Hub.
            path (str): The local directory path of the model to be uploaded.
            step (int): Current training step.
            commit_message (str | None): Commit message for the upload.

        Returns:
            None
        """
        self.hf_api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=path,
            commit_message=f"Step: {step}" if commit_message is None else commit_message,
        )

    def clone_hub_repository_into_save_dir(self, repo_id: str, path: str) -> None:
        """Clone a Hugging Face Hub repository into the specified local directory.

        Args:
            repo_id (str): The identifier of the repository on Hugging Face Hub.
            path (str): The local directory path where the repository will be cloned.
        """
        self.hf_api.create_repo(repo_id=repo_id, exist_ok=True, private=False)
        self.hf_api.snapshot_download(repo_id=repo_id, local_dir=path)
