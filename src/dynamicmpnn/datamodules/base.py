from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import pytorch_lightning as L
import torch
from beartype import beartype as typechecker
from graphein import verbose
from graphein.protein.utils import get_obsolete_mapping
from loguru import logger
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from torch_geometric import transforms as T
from torch_geometric.data import Dataset

verbose(False)


class ProteinDataModule(L.LightningDataModule, ABC):
    """
    Source: https://github.com/a-r-j/ProteinWorkshop
    License: MIT Licence

    Base class for Protein datamodules.

    .. seealso::
        L.LightningDataModule
    """

    prepare_data_per_node = True  # class default for lighting 2.0 compatability

    def setup(self, stage: Optional[str] = None):
        # self.download()

        if stage == "fit" or stage is None:
            logger.info("Preprocessing training data")
            self.train_ds = self.train_dataset()
            logger.info("Preprocessing validation data")
            self.val_ds = self.val_dataset()
        elif stage == "test":
            logger.info("Preprocessing test data")
            if hasattr(self, "test_dataset_names"):
                for split in self.test_dataset_names:
                    setattr(self, f"{split}_ds", self.test_dataset(split))
            else:
                self.test_ds = self.test_dataset()
        elif stage == "lazy_init":
            logger.info("Preprocessing validation data")
            self.val_ds = self.val_dataset()

        # self.class_weights = self.get_class_weights()

    @property
    @lru_cache
    def obsolete_pdbs(self) -> Dict[str, str]:
        """Returns a mapping of obsolete PDB codes to their updated replacement.

        :return: Mapping of obsolete PDB codes to their updated replacements.
        :rtype: Dict[str, str]
        """
        return get_obsolete_mapping()

    @typechecker
    def compose_transforms(self, transforms: Iterable[Callable]) -> T.Compose:
        """Compose an iterable of Transforms into a single transform.

        :param transforms: An iterable of transforms.
        :type transforms: Iterable[Callable]
        :raises ValueError: If ``transforms`` is not a list or dict.
        :return: A single transform.
        :rtype: T.Compose
        """
        if isinstance(transforms, list):
            return T.Compose(transforms)
        elif isinstance(transforms, dict):
            return T.Compose(list(transforms.values()))
        else:
            raise ValueError("Transforms must be a list or dict")

    @abstractmethod
    def parse_dataset(self, split: str) -> pd.DataFrame:
        """
        Implement the parsing of the raw dataset to a dataframe.

        Override this method to implement custom parsing of raw data.

        :param split: The split to parse (e.g. train/val/test)
        :type split: str
        :return: The parsed dataset as a dataframe.
        :rtype: pd.DataFrame
        """
        ...

    @abstractmethod
    def parse_labels(self) -> Any:
        """Optional method to parse labels from the dataset.

        Labels may or may not be present in the dataframe returned by
        ``parse_dataset``.

        :return: The parsed labels in any format. We'd recommend:
            ``Dict[id, Tensor]``.
        :rtype: Any
        """
        ...

    @abstractmethod
    def exclude_pdbs(self):
        """Return a list of PDBs/IDs to exclude from the dataset."""
        ...

    @abstractmethod
    def train_dataset(self) -> Dataset:
        """
        Implement the construction of the training dataset.

        :return: The training dataset.
        :rtype: Dataset
        """
        ...

    @abstractmethod
    def val_dataset(self) -> Dataset:
        """
        Implement the construction of the validation dataset.

        :return: The validation dataset.
        :rtype: Dataset
        """
        ...

    @abstractmethod
    def test_dataset(self) -> Dataset:
        """
        Implement the construction of the test dataset.

        :return: The test dataset.
        :rtype: Dataset
        """
        ...

    @abstractmethod
    def train_dataloader(self) -> DataLoader:
        """
        Implement the construction of the training dataloader.

        :return: The training dataloader.
        :rtype: ProteinDataLoader
        """
        ...

    @abstractmethod
    def val_dataloader(self) -> DataLoader:
        """Implement the construction of the validation dataloader.

        :return: The validation dataloader.
        :rtype: ProteinDataLoader
        """
        ...

    @abstractmethod
    def test_dataloader(self) -> DataLoader:
        """Implement the construction of the test dataloader.

        :return: The test dataloader.
        :rtype: ProteinDataLoader
        """
        ...

    def get_class_weights(self) -> torch.Tensor:
        """Return tensor of class weights."""
        labels: Dict[str, torch.Tensor] = self.parse_labels()
        labels = list(labels.values())  # type: ignore
        labels = np.array(labels)  # type: ignore
        weights = compute_class_weight(class_weight="balanced", classes=np.unique(labels), y=labels)
        return torch.tensor(weights)
