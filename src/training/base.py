"""Base trainer class for model management and logging."""

import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import wandb
from huggingface_hub import HfApi, ModelHubMixin, hf_hub_download


class BaseTrainer(ModelHubMixin):
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
        super().__init__()

        self.hf_api = None
        self.wandb_writer = None
        self.wandb_table = None
        self.optimizer = None
        self.scheduler = None
        self.optimizer_class = None
        self.scheduler_class = None
        self.model = model
        self.learning_rate = learning_rate
        self._init_optimizer(model, learning_rate, optimizer_class, **kwargs)
        self._init_scheduler(scheduler_class, **kwargs)

    def _init_optimizer(
        self,
        model: nn.Module,
        learning_rate: float,
        optimizer_class: str,
        **kwargs: dict,
    ) -> None:
        """Initialize the optimizer.

        Args:
            model (nn.Module): The model to be optimized.
            learning_rate (float): Learning rate for the optimizer.
            optimizer_class (str): The optimizer class to use.
            **kwargs (dict): Additional keyword arguments for optimizer initialization.

        Returns:
            None
        """
        self.optimizer_class = optimizer_class
        if self.optimizer_class == "adamw":
            self.optimizer = torch.optim.AdamW(
                model.parameters(), lr=learning_rate, weight_decay=kwargs.get("weight_decay", 0.05)
            )
        elif self.optimizer_class == "adam":
            self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        elif self.optimizer_class == "sgd":
            self.optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        else:
            raise NotImplementedError

    def _init_scheduler(self, scheduler_class: str, **kwargs: dict) -> None:
        """Initialize the learning rate scheduler.

        Args:
            scheduler_class (str): The scheduler class to use.
            **kwargs (dict): Additional keyword arguments for scheduler initialization.

        Returns:
            None
        """
        self.scheduler_class = scheduler_class

        if self.scheduler_class == "cosine":
            decreasing_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=kwargs.get("num_steps"),
                eta_min=kwargs.get("eta_min", 1e-8),
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
            period = kwargs.get("num_steps")
            warmup_period = int(period * warmup_ratio)
            print(f"Using warmup for {warmup_period} steps out of {period} total steps.")
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_period
            )
            self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, decreasing_lr_scheduler],
                milestones=[warmup_period],
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

    def _save_pretrained(self, save_directory: Path) -> None:
        """Save trainer state to directory.

        This method is called by save_pretrained() from ModelHubMixin.

        Args:
            save_directory (Path): Directory to save trainer state.
        """
        torch.save(
            {
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "optimizer_class": self.optimizer_class,
                "scheduler_class": self.scheduler_class,
                "learning_rate": self.learning_rate,
            },
            save_directory / "trainer_state.pt",
        )
        with open(save_directory / "training_config.json", "w") as f:
            json.dump(self.__dict__, f, indent=2, default=str)

        original_state_dict = self.model.state_dict()
        cloned_state_dict = {k: v.clone().contiguous() for k, v in original_state_dict.items()}
        original_state_dict_fn = self.model.state_dict

        def _get_cloned_state_dict() -> dict:
            """Get cloned state dict to ensure contiguous tensors."""
            return cloned_state_dict

        self.model.state_dict = _get_cloned_state_dict

        try:
            self.model.save_pretrained(save_directory)
        finally:
            self.model.state_dict = original_state_dict_fn

    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str,
        model: nn.Module,
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        force_download: bool = False,
        proxies: dict | None = None,
        local_files_only: bool = False,
        token: str | bool | None = None,
        **kwargs: dict,
    ) -> "BaseTrainer":
        """Load trainer from pretrained.

        This method is called by from_pretrained() from ModelHubMixin.

        Args:
            model_id (str): Model ID on HuggingFace Hub.
            model (nn.Module): The model instance.
            revision (str | None): Specific model version to use.
            cache_dir (str | Path | None): Directory to cache the downloaded model.
            force_download (bool): Whether to force re-download of model files.
            proxies (dict | None): Proxy settings for downloading.
            local_files_only (bool): Whether to only use local files.
            token (str | bool | None): Authentication token for private models.
            **kwargs: Additional arguments.

        Returns:
            BaseTrainer: Loaded trainer instance.
        """
        # Check if model_id is a local path
        local_path = Path(model_id)
        if local_path.exists() and local_path.is_dir():
            # Load from local directory
            trainer_state_path = local_path / "trainer_state.pt"
            if not trainer_state_path.exists():
                raise FileNotFoundError(
                    f"trainer_state.pt not found in {local_path}. "
                    "Make sure the directory contains a saved trainer state."
                )
        else:
            # Download trainer state from HuggingFace Hub
            trainer_state_path = hf_hub_download(
                repo_id=model_id,
                filename="trainer_state.pt",
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                proxies=proxies,
                local_files_only=local_files_only,
                token=token,
            )

        # Load trainer state
        trainer_state = torch.load(trainer_state_path)

        # Create new trainer instance
        trainer = cls(
            model=model,
            learning_rate=trainer_state["learning_rate"],
            optimizer_class=trainer_state["optimizer_class"],
            scheduler_class=trainer_state["scheduler_class"],
            **kwargs,
        )

        trainer.optimizer.load_state_dict(trainer_state["optimizer"])
        trainer.scheduler.load_state_dict(trainer_state["scheduler"])
        trainer.optimizer_class = trainer_state["optimizer_class"]
        trainer.scheduler_class = trainer_state["scheduler_class"]
        trainer.learning_rate = trainer_state["learning_rate"]
        return trainer

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
