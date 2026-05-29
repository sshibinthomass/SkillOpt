"""Ceramic Capacitors task dataloader located in train/."""
from __future__ import annotations

import json
from skillopt.datasets.base import SplitDataLoader

class CeramicCapacitorsDataLoader(SplitDataLoader):
    """Ceramic Capacitors dataloader."""

    def load_raw_items(self, data_path: str) -> list[dict]:
        with open(data_path, encoding="utf-8") as f:
            return json.load(f)
