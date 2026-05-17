from typing import Callable, List, Optional

import torch.nn as nn
import torch.nn.functional as F

from dynamicmpnn.types import LossType


def get_loss(
    name: LossType,
    smoothing: float = 0.0,
    ignore_index: int = -100,
    class_weights: Optional["torch.Tensor"] = None,
) -> Callable:
    if name == "cross_entropy":
        return nn.CrossEntropyLoss(
            label_smoothing=smoothing, weight=class_weights, ignore_index=ignore_index
        )
    if name == "bce":
        return nn.BCEWithLogitsLoss(weight=class_weights)
    elif name == "nll_loss":
        return F.nll_loss
    elif name == "mse_loss":
        return F.mse_loss
    elif name == "l1_loss":
        return F.l1_loss
    elif name == "dihedral_loss":
        raise NotImplementedError("Dihedral loss not implemented yet")
    else:
        raise ValueError(f"Incorrect Loss provided: {name}")


def flatten_list(l: List[List]) -> List:
    return [item for sublist in l for item in sublist]
