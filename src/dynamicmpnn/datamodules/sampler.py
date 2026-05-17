from loguru import logger
from torch_geometric.data import Batch


def safe_collate(data_list):
    """Collate function that fails fast on None values."""
    none_indices = [i for i, data in enumerate(data_list) if data is None]
    if none_indices:
        raise ValueError(
            f"Got {len(none_indices)} None sample(s) in batch at indices {none_indices}. "
            "Check for corrupted .pt files."
        )

    try:
        return Batch.from_data_list(data_list)
    except Exception as e:
        logger.error(f"Error in collating batch: {e}")
        raise
