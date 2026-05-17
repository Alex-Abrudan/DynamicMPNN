import os
import random
import time
from typing import Optional

from omegaconf import DictConfig
from loguru import logger
from pathlib import Path
import pandas as pd

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from dynamicmpnn.datamodules.sampler import safe_collate
from dynamicmpnn.datamodules.base import ProteinDataModule
from dynamicmpnn.datamodules.pt_dataset import PTFileDataset
from dynamicmpnn import constants


def _log_ddp(event: str):
    """Log event with DDP rank info."""
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    ts = time.strftime("%H:%M:%S")
    logger.info(f"[DDP DATAMODULE] {ts} rank={rank}/{world_size} | {event}")


def _resolve_repo_path(path_value: Optional[str]) -> Optional[Path]:
    if path_value is None:
        return None

    path = Path(path_value)
    if path.is_absolute():
        return path

    return constants.REPO_ROOT / path


class MultiConfProteinDataModule(ProteinDataModule):
    def __init__(
        self,
        cfg_features: DictConfig,
        processed_dir: Optional[str] = None,
        val_processed_dir: Optional[str] = None,
        test_processed_dir: Optional[str] = None,
        val_splitting_csv: Optional[str] = None,
        cluster_sampling_csv: Optional[str] = None,
        in_memory: bool = False,
        num_workers: int = 32,
        batch_size: int = 8,
        short_dataset: bool = False,
    ) -> None:
        super().__init__()

        self.processed_dir = _resolve_repo_path(processed_dir)
        os.makedirs(self.processed_dir, exist_ok=True)

        self.val_processed_dir = _resolve_repo_path(val_processed_dir) if val_processed_dir else self.processed_dir
        self.test_processed_dir = _resolve_repo_path(test_processed_dir) if test_processed_dir else self.processed_dir

        self.in_memory = in_memory
        self.cfg_features = cfg_features
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.short_dataset = short_dataset
        self.val_splitting_csv = _resolve_repo_path(val_splitting_csv)
        self.cluster_sampling_csv = _resolve_repo_path(cluster_sampling_csv)

        # Cluster pool for per-epoch sampling (loaded once)
        self._cluster_pool: Optional[dict] = None
        self._cluster_pool_df: Optional[pd.DataFrame] = None

    def _load_cluster_pool(self):
        """Load cluster pool CSV once."""
        if self._cluster_pool is not None:
            return
        csv_path = self.cluster_sampling_csv
        if not csv_path.exists():
            raise FileNotFoundError(f"cluster_sampling_csv not found: {csv_path}")
        df = pd.read_csv(csv_path)
        self._cluster_pool_df = df
        self._cluster_pool = df.groupby("cluster30")["clus_80"].apply(list).to_dict()
        has_mintm = "min_tm" in df.columns
        logger.info(f"Loaded cluster pool: {len(self._cluster_pool):,} cluster30 groups, "
                    f"{len(df):,} total pairs. min_tm column present: {has_mintm}")

    def _sample_train_from_clusters(self, epoch: int = 0) -> list:
        """Sample one clus_80 per cluster30 each epoch (deterministic across ranks)."""
        self._load_cluster_pool()

        rng_state = random.getstate()
        random.seed(42 + epoch)

        sampled = [random.choice(members) for members in self._cluster_pool.values()]

        random.setstate(rng_state)

        # Filter to existing .pt files
        valid = [x for x in sampled if (self.processed_dir / f"{x}.pt").exists()]
        logger.info(f"Epoch {epoch} cluster sampling: {len(valid):,} / {len(sampled):,} clus_80 have .pt files")
        return valid

    def setup(self, stage: Optional[str] = None):
        """Setup train/val/test splits."""
        _log_ddp(f"setup() called with stage={stage}")
        self.get_splits()
        _log_ddp(f"setup() complete: train={len(self.splits['train'])}, val={len(self.splits['val'])}")

    def get_splits(self):
        self.splits = {"train": [], "val": [], "test": []}

        val_csv_path = self.val_splitting_csv
        if Path(val_csv_path).exists():
            logger.info(f"Loading validation split from: {val_csv_path}")
            df_val = pd.read_csv(val_csv_path)
            self.splits["val"] = df_val["pdb_auth"].tolist()
            logger.info(f"Loaded {len(self.splits['val'])} samples for validation.")
        else:
            logger.error(f"Validation splitting file not found at: {val_csv_path}")

        csv_path = self.cluster_sampling_csv
        if csv_path.exists():
            logger.info(f"Loading cluster pool from: {csv_path}")
            df = pd.read_csv(csv_path)
            cluster_pool = df.groupby("cluster30")["clus_80"].apply(list).to_dict()

            # Sample one per cluster30 (deterministic seed for reproducibility)
            random.seed(42)
            sampled = [random.choice(members) for members in cluster_pool.values()]

            # Filter to existing .pt files
            valid = [x for x in sampled if (self.processed_dir / f"{x}.pt").exists()]
            self.splits["train"] = valid
            logger.info(f"Cluster sampling: {len(valid):,} / {len(sampled):,} sampled clus_80 have .pt files.")
        else:
            logger.error(f"cluster_sampling_csv not found: {csv_path}")

    def exclude_pdbs(self):
        pass

    def parse_labels(self):
        pass

    def parse_dataset(self, split: str):
        return self.splits[split]

    def _maybe_shorten_pdb_codes(self, pdb_codes: list, split: str) -> list:
        if not self.short_dataset:
            return pdb_codes

        if split == "train":
            short_length_pdb = max(1, len(pdb_codes) // 1000)
            shortened = pdb_codes[:short_length_pdb]
            logger.info(
                f"short_dataset=True: using {len(shortened):,} / {len(pdb_codes):,} {split} samples"
            )
            return shortened

        return pdb_codes

    def _get_dataset(self, split: str, return_pdb_codes: bool = False):
        pdb_codes = self._maybe_shorten_pdb_codes(self.parse_dataset(split), split)

        if split == "val":
            split_processed_dir = self.val_processed_dir
        elif split == "test":
            split_processed_dir = self.test_processed_dir
        else:
            split_processed_dir = self.processed_dir

        valid_pdb_codes = []
        missing_files = []

        for pdb_code in pdb_codes:
            if (split_processed_dir / f"{pdb_code}.pt").exists():
                valid_pdb_codes.append(pdb_code)
            else:
                missing_files.append(pdb_code)

        if missing_files:
            logger.warning(
                f"Processed files not found for {len(missing_files)} PDB codes in {split_processed_dir}:"
            )
            for pdb_code in missing_files:
                logger.warning(f"Missing: {pdb_code}.pt")
            logger.warning("Run the process_PDBFlex.py script first for these files.")

        return PTFileDataset(
            pdb_codes=valid_pdb_codes,
            cfg_features=self.cfg_features,
            split=split,
            processed_dir=split_processed_dir,
            in_memory=self.in_memory,
            return_pdb_codes=return_pdb_codes,
        )

    def train_dataset(self, return_pdb_codes: bool = False) -> PTFileDataset:
        return self._get_dataset("train", return_pdb_codes)

    def val_dataset(self, return_pdb_codes: bool = False) -> PTFileDataset:
        return self._get_dataset("val", return_pdb_codes)

    def test_dataset(self, return_pdb_codes: bool = False) -> PTFileDataset:
        return self._get_dataset("test", return_pdb_codes)

    def train_dataloader(self) -> DataLoader:
        _log_ddp("train_dataloader() called")

        # Per-epoch cluster sampling (when reload_dataloaders_every_n_epochs=1)
        if self.cluster_sampling_csv and hasattr(self, 'trainer') and self.trainer is not None:
            epoch = self.trainer.current_epoch
            _log_ddp(f"Per-epoch cluster sampling for epoch {epoch}")
            pdb_codes = self._maybe_shorten_pdb_codes(
                self._sample_train_from_clusters(epoch=epoch), split="train"
            )

            dataset = PTFileDataset(
                pdb_codes=pdb_codes,
                cfg_features=self.cfg_features,
                split="train",
                processed_dir=self.processed_dir,
                in_memory=self.in_memory,
            )
        else:
            dataset = self.train_dataset()

        _log_ddp(f"train_dataloader() dataset created, len={len(dataset)}")

        loader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            drop_last=True,
            collate_fn=safe_collate,
        )
        _log_ddp("train_dataloader() returning DataLoader")
        return loader

    def val_dataloader(self) -> DataLoader:
        _log_ddp("val_dataloader() called")
        dataset = self.val_dataset()
        _log_ddp(f"val_dataloader() dataset created, len={len(dataset)}")

        loader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            drop_last=True,
            collate_fn=safe_collate,
        )
        _log_ddp("val_dataloader() returning DataLoader")
        return loader

    def test_dataloader(self) -> DataLoader:
        dataset = self.test_dataset()

        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=False,
            collate_fn=safe_collate,
        )
