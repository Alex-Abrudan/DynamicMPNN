import copy
import sys
import os
from typing import List, Optional
from pathlib import Path
import graphein
import hydra
import pytorch_lightning as L
import lovely_tensors as lt
import torch_geometric
from graphein.protein.tensor.dataloader import ProteinDataLoader
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import Logger
from loguru import logger as log
from omegaconf import DictConfig, OmegaConf
from dotenv import load_dotenv
from hydra import compose, initialize

from dynamicmpnn import constants, utils, register_custom_omegaconf_resolvers
from dynamicmpnn.modules.models.lightning_module import DynamicMPNNModule

load_dotenv(constants.REPO_ROOT / ".env", override=False)
load_dotenv(constants.REPO_ROOT / ".env.local", override=False)

graphein.verbose(False)
lt.monkey_patch()

import torch
from pytorch_lightning import Trainer

print(f"=== Process Debug Info ===")
print(f"PID: {os.getpid()}")
print(f"SLURM_PROCID: {os.environ.get('SLURM_PROCID', 'not set')}")
print(f"SLURM_LOCALID: {os.environ.get('SLURM_LOCALID', 'not set')}")
print(f"SLURM_NTASKS: {os.environ.get('SLURM_NTASKS', 'not set')}")
print(f"LOCAL_RANK: {os.environ.get('LOCAL_RANK', 'not set')}")
print(f"RANK: {os.environ.get('RANK', 'not set')}")
print(f"WORLD_SIZE: {os.environ.get('WORLD_SIZE', 'not set')}")
print(f"CUDA devices visible: {torch.cuda.device_count()}")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")


def _num_training_steps(train_dataset: ProteinDataLoader, trainer: L.Trainer) -> int:
    """
    Returns total training steps inferred from datamodule and devices.

    :param train_dataset: Training dataloader
    :type train_dataset: ProteinDataLoader
    :param trainer: Lightning trainer
    :type trainer: L.Trainer
    :return: Total number of training steps
    :rtype: int
    """
    if trainer.max_steps != -1:
        return trainer.max_steps

    dataset_size = (
        trainer.limit_train_batches
        if trainer.limit_train_batches not in {0, 1}
        else len(train_dataset) * train_dataset.batch_size
    )

    log.info(f"Dataset size: {dataset_size}")

    num_devices = max(1, trainer.num_devices)
    effective_batch_size = train_dataset.batch_size * trainer.accumulate_grad_batches * num_devices
    return (dataset_size // effective_batch_size) * trainer.max_epochs


def train_model(cfg: DictConfig):
    L.seed_everything(cfg.seed)

    log.info(f"Instantiating datamodule: <{cfg.dataset.datamodule._target_}...")

    datamodule: L.LightningDataModule = hydra.utils.instantiate(
        cfg.dataset.datamodule,
        cfg_features=cfg.features,
    )
    
    ckpt_path = cfg.get("ckpt_path")
    cfg_to_use = cfg # Default to the new config from YAML files

    if ckpt_path:
        log.info(f"Resuming from checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        checkpoint_cfg = OmegaConf.create(checkpoint["hyper_parameters"]["cfg"])

        OmegaConf.set_struct(checkpoint_cfg, False)
        
        log.info("Merging new config into the checkpoint's config to allow for overrides.")
        # This is the key change: new config values will overwrite old ones.
        cfg_to_use = OmegaConf.merge(checkpoint_cfg, cfg)

        # Surgically remove problematic legacy callbacks if they exist
        if "callbacks" in cfg_to_use and "stop_on_nan" in cfg_to_use.callbacks:
            log.warning("Found and removed legacy 'stop_on_nan' callback from config.")
            del cfg_to_use.callbacks.stop_on_nan

        precision = cfg_to_use.trainer.get("precision")

        if precision == 16:
            if torch.cuda.is_bf16_supported():
                cfg_to_use.trainer.precision = "bf16-mixed"
                log.info("Hardware supports bfloat16. Upgrading precision from '16-mixed' to 'bf16-mixed'. 🚀")

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = utils.callbacks.instantiate_callbacks(cfg_to_use.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.loggers.instantiate_loggers(cfg_to_use.get("logger"))

    log.info("Instantiating trainer...")
    trainer: L.Trainer = hydra.utils.instantiate(cfg_to_use.trainer, callbacks=callbacks, logger=logger)

    if cfg_to_use.get("scheduler"):
        if (
            cfg_to_use.scheduler.scheduler._target_ == "flash.core.optimizers.LinearWarmupCosineAnnealingLR"
            and cfg_to_use.scheduler.interval == "step"
        ):
            datamodule.setup()  # type: ignore
            num_steps = _num_training_steps(datamodule.train_dataloader(), trainer)
            log.info(f"Setting number of training steps in scheduler to: {num_steps}")
            cfg_to_use.scheduler.scheduler.warmup_epochs = num_steps / trainer.max_epochs
            cfg_to_use.scheduler.scheduler.max_epochs = num_steps
            log.info(cfg_to_use.scheduler)

    log.info("Instantiating model...")
    model: L.LightningModule = DynamicMPNNModule(cfg_to_use)

    object_dict = {
        "cfg": cfg_to_use,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.logging_utils.log_hyperparameters(object_dict)

    if cfg_to_use.get("compile"):
        log.info("Compiling model!")
        model = torch_geometric.compile(model, dynamic=True)

    if cfg_to_use.get("task_name") == "train":
        log.info("Starting training!")
        try:
            trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg_to_use.get("ckpt_path"))
        except Exception as e:
            log.error(f"Error during training: {e}")
            raise e

    if cfg_to_use.get("test"):
        log.info("Starting testing!")
        if hasattr(datamodule, "test_dataset_names"):
            splits = datamodule.test_dataset_names
            wandb_logger = copy.deepcopy(trainer.logger)
            for i, split in enumerate(splits):
                dataloader = datamodule.test_dataloader(split)
                trainer.logger = False
                log.info(f"Testing on {split} ({i+1} / {len(splits)})...")
                results = trainer.test(model=model, dataloaders=dataloader, ckpt_path="best")[0]
                results = {f"{k}/{split}": v for k, v in results.items()}
                log.info(f"{split}: {results}")
                wandb_logger.log_metrics(results)
        else:
            trainer.test(model=model, datamodule=datamodule, ckpt_path="best")

@hydra.main(
    version_base="1.3",
    config_path=str(constants.HYDRA_CONFIG_PATH),
    config_name="dynamicmpnn",
)
def _main(cfg: DictConfig) -> None:
    """Load and validate the hydra config."""
    utils.extras(cfg)
    #cfg = config.validate_config(cfg)
    train_model(cfg)

    if cfg.trainer.devices > 1:
        trainer = Trainer(accelerator='gpu', devices=4, strategy='ddp_find_unused_parameters_false')
        print(f"\n=== Trainer Info ===")
        print(f"Trainer.num_devices: {trainer.num_devices}")
        print(f"Trainer.device_ids: {trainer.device_ids}")
        print(f"Strategy class: {type(trainer.strategy)}")

        # Check if using distributed
        if hasattr(trainer.strategy, 'world_size'):
            print(f"Strategy world_size: {trainer.strategy.world_size}")
        if hasattr(trainer.strategy, 'local_rank'):
            print(f"Strategy local_rank: {trainer.strategy.local_rank}")

if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    register_custom_omegaconf_resolvers()
    #dist.init_process_group(backend="nccl")
    _main()  # type: ignore