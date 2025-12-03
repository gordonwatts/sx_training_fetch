from dataclasses import dataclass
from typing import Optional, List
from pathlib import Path
import awkward as ak


# Data class for combine configuration options
@dataclass
class CombineConfig:
    output_path: str = "main_training_file.parquet"
    event_filter: Optional[str] = None


def combine_training_data(input_files: List[Path], config: CombineConfig):
    arrays = [ak.from_parquet(f) for f in input_files]
    combined = ak.concatenate(arrays, axis=0)
    ak.to_parquet(combined, config.output_path)
    return combined
