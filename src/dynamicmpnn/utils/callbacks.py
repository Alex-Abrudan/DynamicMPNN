# Inspired from https://github.com/a-r-j/ProteinWorkshop

import time
from typing import List

import hydra
import torch
import torch.distributed as dist
from pytorch_lightning.callbacks import Callback
from loguru import logger
from omegaconf import DictConfig, OmegaConf


class DDPDebugCallback(Callback):
    """Logs rank info at key training lifecycle points to debug DDP hangs."""

    def _log_rank(self, event: str):
        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1
        ts = time.strftime("%H:%M:%S")
        logger.info(f"[DDP DEBUG] {ts} rank={rank}/{world_size} | {event}")

    def on_fit_start(self, trainer, pl_module):
        self._log_rank("on_fit_start")

    def on_train_epoch_start(self, trainer, pl_module):
        self._log_rank(f"on_train_epoch_start (epoch={trainer.current_epoch})")

    def on_train_epoch_end(self, trainer, pl_module):
        self._log_rank(f"on_train_epoch_end (epoch={trainer.current_epoch})")

    def on_validation_start(self, trainer, pl_module):
        self._log_rank(f"on_validation_start (epoch={trainer.current_epoch})")

    def on_validation_epoch_start(self, trainer, pl_module):
        self._log_rank(f"on_validation_epoch_start (epoch={trainer.current_epoch})")

    def on_validation_epoch_end(self, trainer, pl_module):
        self._log_rank(f"on_validation_epoch_end (epoch={trainer.current_epoch})")

    def on_validation_end(self, trainer, pl_module):
        self._log_rank(f"on_validation_end (epoch={trainer.current_epoch})")

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if batch_idx % 100 == 0:
            self._log_rank(f"train_batch_start batch_idx={batch_idx} (epoch={trainer.current_epoch})")

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        self._log_rank(f"val_batch_start batch_idx={batch_idx} (epoch={trainer.current_epoch})")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self._log_rank(f"val_batch_end batch_idx={batch_idx} (epoch={trainer.current_epoch})")


def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    """
    Instantiates callbacks from Hydra config.
    """
    callbacks: List[Callback] = []

    # Always add DDP debug callback for hang diagnosis
    logger.info("Adding DDPDebugCallback for DDP hang diagnosis")
    callbacks.append(DDPDebugCallback())

    if not callbacks_cfg:
        logger.warning("Callbacks config is empty.")
        return callbacks
    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    # Now instantiate the callbacks with the modified config
    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            logger.info(f"Instantiating callback <{cb_conf._target_}>")
            logger.info(f"Callback config: {OmegaConf.to_yaml(cb_conf)}")  # Add this line for debugging
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks
