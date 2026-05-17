from collections import defaultdict
import os
import numpy as np
from pathlib import Path
from loguru import logger
import torch
from torch_geometric.data import Data
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)
from omegaconf import DictConfig
from tqdm import tqdm
from hydra.utils import instantiate

from dynamicmpnn.types import DISTANCE_EPS


class PTFileDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        pdb_codes: List[str],
        cfg_features: DictConfig,
        processed_dir: Path,
        split: str,
        in_memory: bool = False,
        return_pdb_codes: bool = False,
        ):

        self.pdb_codes = pdb_codes
        self.split = split
        self.in_memory = in_memory
        self._processed_dir = processed_dir
        self._indices = list(range(len(self.pdb_codes)))
        self.return_pdb_codes = return_pdb_codes

        # cfg_features may be either:
        # 1. Already instantiated by Hydra's recursive instantiation (a featurizer object)
        # 2. A DictConfig with _target_ that needs to be instantiated
        if hasattr(cfg_features, '__call__'):
            # Already instantiated - use directly, but update split
            self.featuriser = cfg_features
            self.featuriser.split = self.split
        else:
            # DictConfig - instantiate it
            self.featuriser = instantiate(
                cfg_features,
                split=self.split,
                device="cpu",
                distance_eps=DISTANCE_EPS,
            )

        if self.in_memory:
            logger.info("Reading data into memory")
            self.data = []
            valid_max_index = 0

            for i, pdb_code in enumerate(tqdm(self.pdb_codes)):
                protein_data = torch.load(self.processed_dir / f"{pdb_code}.pt")
                featurised_data = self.featuriser(protein_data, pdb_code=pdb_code)

                if featurised_data is not None:
                    self.data.append(featurised_data)
                    valid_max_index += 1
                else:
                    logger.warning(f"Skipping {pdb_code} due to insufficient valid coordinates.")

            self._indices = list(range(valid_max_index))

    @property
    def processed_dir(self):
        return self._processed_dir

    @processed_dir.setter
    def processed_dir(self, value):
        self._processed_dir = value

    def __len__(self) -> int:
        """Return length of the dataset."""
        return len(self._indices)

    def __getitem__(self, idx):
        real_idx = self._indices[idx]
        
        try:
            if self.in_memory:
                if self.return_pdb_codes:
                    return self.data[real_idx], self.pdb_codes[real_idx]
                else:
                    return self.data[real_idx]  # Already loaded as dict
            
            else:
                pyg_object = torch.load(self.processed_dir / f"{self.pdb_codes[real_idx]}.pt")
                features_dict = self.featuriser(pyg_object, pdb_code=self.pdb_codes[real_idx])

                if features_dict is None:
                    logger.warning(f"Featurizer returned None for {self.pdb_codes[real_idx]}")
                    return None
                    
                if self.return_pdb_codes:
                    return features_dict, self.pdb_codes[real_idx]
                return features_dict
                
        except Exception as e:
            logger.error(f"Failed to load {self.pdb_codes[real_idx]}: {e}")
            return None
