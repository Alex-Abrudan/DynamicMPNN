import abc
from typing import Any, Callable, Dict, List, Literal, Optional, Union
from time import time
import hydra
import pytorch_lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from beartype import beartype as typechecker
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from torch_geometric.data import Batch, Data

from dynamicmpnn.modules.models.utils import get_loss


class BaseModel(L.LightningModule, abc.ABC):
    config: DictConfig
    losses: Dict[str, Callable]
    metric_names: List[str]

    @abc.abstractmethod
    def forward(self, batch: Batch) -> torch.Tensor:
        ...

    @abc.abstractmethod
    def training_step(self, batch: Batch, batch_idx: torch.Tensor) -> torch.Tensor:
        ...

    @abc.abstractmethod
    def validation_step(self, batch: Batch, batch_idx: torch.Tensor) -> torch.Tensor:
        ...

    @abc.abstractmethod
    def test_step(self, batch: Batch, batch_idx: torch.Tensor) -> torch.Tensor:
        ...

    def get_labels(self, batch: Union[Data, Batch]) -> torch.Tensor:
        return batch.seq

    def configure_optimizers(self):
        logger.info("Instantiating optimiser...")
        optimiser = hydra.utils.instantiate(self.config.optimiser)["optimizer"]
        optimiser = optimiser(self.parameters())

        if self.config.get("scheduler"):
            logger.info("Instantiating scheduler...")
            scheduler = hydra.utils.instantiate(self.config.scheduler, optimiser)
            scheduler = OmegaConf.to_container(scheduler)
            scheduler["scheduler"] = scheduler["scheduler"](optimizer=optimiser)
            return {"optimizer": optimiser, "lr_scheduler": scheduler}
        return optimiser

    def configure_losses(self, loss_dict: Dict[str, str]) -> Dict[str, Callable]:
        return {
            k: get_loss(
                v,
                smoothing=self.config.task.label_smoothing,
                ignore_index=self.config.task.get("ignore_index", -100),
            )
            for k, v in loss_dict.items()
        }

    def configure_metrics(self):
        METRICS = {"accuracy", "perplexity", "custom_perplexity", "sequence_recovery"}
        STAGES = {"train": "metrics_train", "val": "metrics_val", "test": "metrics_test"}

        metric_names = set()
        for stage, config_key in STAGES.items():
            stage_metrics = getattr(self.config.metrics, config_key)
            for metric_name, metric_conf in stage_metrics.items():
                if metric_name == '_self_':
                    continue
                if metric_conf.get('_target_') is None:
                    metric_names.add(metric_name)
                    continue
                if metric_name in METRICS:
                    metric_names.add(metric_name)
                    try:
                        metric = hydra.utils.instantiate(metric_conf)
                        setattr(self, f"{stage}_{metric_name}", metric)
                    except Exception as e:
                        logger.warning(f"Failed to instantiate metric {metric_name}: {e}")

        self.metric_names = list(metric_names)


class DynamicMPNNModule(BaseModel):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.config = cfg
        self._batch_load_time = None
        self.batch_size = cfg.dataset.datamodule.batch_size

        logger.info("Instantiating encoder...")
        self.GNN_model: nn.Module = hydra.utils.instantiate(cfg.model)
        logger.info(self.GNN_model)

        logger.info("Instantiating losses...")
        self.losses = self.configure_losses(cfg.task.losses)

        logger.info("Configuring metrics...")
        self.configure_metrics()
        self.save_hyperparameters()

    @typechecker
    def forward(self, batch: Union[Batch, Data]) -> tuple:
        return self.GNN_model(batch)

    def _compute_eval_metrics(self, batch: Union[Batch, Data]) -> tuple:
        """Shared evaluation logic for val/test steps."""
        y_hat_samples, y_hat_logits, final_indices, batch_indices = self.GNN_model.sample(
            batch, return_logits=True
        )

        y_true = self.get_labels(batch)[final_indices]
        n_samples = y_hat_logits.shape[0]
        num_classes = y_hat_logits.shape[2]

        losses, recoveries, perplexities = [], [], []

        for idx in range(batch_indices.max() + 1):
            mask = batch_indices == idx
            y_true_sample = y_true[mask]
            y_hat_sample = y_hat_samples[:, mask]
            logits_sample = y_hat_logits[:, mask]

            y_target = y_true_sample.unsqueeze(0).expand(n_samples, -1)
            preds_flat = logits_sample.view(-1, num_classes)
            targets_flat = y_target.reshape(-1)

            loss = self.losses['residue_type'](preds_flat, targets_flat)
            recovery = y_hat_sample.eq(y_target).float().mean()

            per_token_ce = F.cross_entropy(preds_flat, targets_flat.long(), reduction="none")
            per_token_ce = per_token_ce.view(n_samples, y_true_sample.shape[0])
            perplexity = torch.exp(per_token_ce.mean(dim=1)).mean()

            losses.append(loss)
            recoveries.append(recovery)
            perplexities.append(perplexity)

        return (
            torch.stack(losses).mean(),
            torch.stack(recoveries).mean(),
            torch.stack(perplexities).mean(),
        )

    def training_step(self, batch: Union[Batch, Data], batch_idx: int) -> Optional[torch.Tensor]:
        if batch is None:
            return None

        try:
            y_true = self.get_labels(batch)
            y_hat_logits, valid_mask = self(batch)

            loss = self.losses['residue_type'](y_hat_logits[valid_mask], y_true[valid_mask].long())
            self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)

            with torch.no_grad():
                preds = torch.argmax(y_hat_logits, dim=1)
                self.train_accuracy(preds[valid_mask], y_true[valid_mask])
                self.log('train/accuracy', self.train_accuracy, on_step=True, on_epoch=True)

            return loss

        except RuntimeError as e:
            if "out of memory" in str(e):
                for p in self.parameters():
                    if p.grad is not None:
                        del p.grad
                torch.cuda.empty_cache()
                logger.warning(f"OOM in train on rank {self.global_rank}, batch {batch_idx}")
                return None
            raise

    def validation_step(self, batch: Union[Batch, Data], batch_idx: int) -> Optional[torch.Tensor]:
        if batch is None:
            return None

        self.GNN_model.eval()
        with torch.no_grad():
            try:
                loss, recovery, perplexity = self._compute_eval_metrics(batch)
                self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
                self.log('val/sequence_recovery', recovery, on_step=False, on_epoch=True, prog_bar=True)
                self.log('val/perplexity', perplexity, on_step=False, on_epoch=True, prog_bar=True)
                return loss

            except RuntimeError as e:
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                    logger.warning(f"OOM in val on rank {self.global_rank}, batch {batch_idx}")
                    return None
                raise

    def test_step(self, batch: Union[Batch, Data], batch_idx: int) -> torch.Tensor:
        self.GNN_model.eval()
        with torch.no_grad():
            loss, recovery, perplexity = self._compute_eval_metrics(batch)
            self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log('val/sequence_recovery', recovery, on_step=False, on_epoch=True, prog_bar=True)
            self.log('val/perplexity', perplexity, on_step=False, on_epoch=True, prog_bar=True)
            return loss

    def on_train_start(self):
        self._batch_load_time = time()
        super().on_train_start()

    def on_train_batch_start(self, batch, batch_idx):
        if self._batch_load_time is not None:
            self.log("batch_load_time", time() - self._batch_load_time, sync_dist=False)
        return super().on_train_batch_start(batch, batch_idx)

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None, **kwargs):
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure, **kwargs)
        self._batch_load_time = time()

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, new_cfg=None, **kwargs):
        model = super().load_from_checkpoint(checkpoint_path, **kwargs)
        if new_cfg is not None:
            model.config = new_cfg
            model.configure_metrics()
        return model

    def backward(self, loss: torch.Tensor, *args: Any, **kwargs: Dict[str, Any]):
        try:
            loss.backward(*args, **kwargs)
        except RuntimeError as e:
            if "out of memory" in str(e):
                for p in self.trainer.model.parameters():
                    if p.grad is not None:
                        del p.grad
                torch.cuda.empty_cache()
                logger.warning(f"OOM in backward on rank {self.global_rank}")
            else:
                raise
