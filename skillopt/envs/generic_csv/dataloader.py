"""Generic CSV task dataloader."""
from __future__ import annotations

from skillopt.datasets.base import SplitDataLoader

class GenericCSVDataLoader(SplitDataLoader):
    """Generic CSV dataloader that loads custom dataset splits."""
