from dataclasses import dataclass
from typing import Optional, List
from pathlib import Path
import awkward as ak
import re


# Data class for combine configuration options
@dataclass
class CombineConfig:
    output_path: str = "main_training_file.parquet"
    event_filter: Optional[str] = None


def combine_training_data(input_files: List[Path], config: CombineConfig):

    # Creating a filter for the events
    if config.event_filter is not None:
        # Apply the filter to the events
        arrays = []

        # using regex to get the mask
        m = re.search(r"% (\d+)\s*==\s*(\d+)", config.event_filter)

        # looping through the input files and applying the filter to them
        for f in input_files:
            arr = ak.from_parquet(f)
            mask = arr["eventNumber"] % int(m.group(1)) == int(m.group(2))
            filtered = arr[mask]
            arrays.append(filtered)
    else:
        arrays = [ak.from_parquet(f) for f in input_files]
    combined = ak.concatenate(arrays, axis=0)
    ak.to_parquet(combined, config.output_path)
    return combined
