from dataclasses import dataclass
from typing import Optional, List
from pathlib import Path


# Data class for combine configuration options
@dataclass
class CombineConfig:
    output_path: str = "main_training_file.parquet"
    event_filter: Optional[str] = None


def combine_training_data(input_file: List[Path], config: CombineConfig):
    print("hello!")
