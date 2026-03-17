from dataclasses import dataclass
from typing import Optional, List
from pathlib import Path
import awkward as ak
import re
import yaml
from glob import glob
import numpy as np


@dataclass
class InputSpec:
    pattern: str
    num_jets: Optional[int] = None


def parse_input_spec(text: str) -> InputSpec:
    """
    Parse CLI input of form:
        path
        path:num_jets
    """

    if ":" in text:
        pattern, jets = text.rsplit(":", 1)
        return InputSpec(pattern=pattern, num_jets=int(jets))

    return InputSpec(pattern=text)


# Data class for combine configuration options
@dataclass
class CombineConfig:
    inputs: List[InputSpec]
    output_path: str = "main_training_file.parquet"
    event_filter: Optional[str] = None


def load_yaml_config(path: Path) -> CombineConfig:

    with open(path) as f:
        data = yaml.safe_load(f)

    inputs = [
        InputSpec(
            pattern=item["path"],
            num_jets=item.get("num-jets"),
        )
        for item in data["input-files"]
    ]

    return CombineConfig(
        inputs=inputs,
        event_filter=data.get("event-filter"),
        output_path=Path(data.get("output", "main_training_file.parquet")),
    )


def expand_inputs(inputs):

    expanded = []

    for spec in inputs:
        files = glob(spec.pattern)

        if not files:
            raise RuntimeError(f"No files match pattern: {spec.pattern}")

        for f in files:
            expanded.append((Path(f), spec.num_jets))

    return expanded


def combine_training_data(config: CombineConfig):

    arrays = []

    expanded = expand_inputs(config.inputs)

    for file_path, num_jets in expanded:

        arr = ak.from_parquet(file_path)

        if config.event_filter:
            m = re.search(r"% (\d+)\s*==\s*(\d+)", config.event_filter)
            mask = arr["eventNumber"] % int(m.group(1)) == int(m.group(2))
            arr = arr[mask]

        if num_jets is not None:
            if num_jets > len(arr):
                print(
                    "Input num-jets is greater than number of jets in file, "
                    "instead including all jets"
                )
            else:
                # Randomly selecting jets (rows)
                random_indices = np.random.choice(len(arr), num_jets, replace=False)
                arr = arr[random_indices]

        arrays.append(arr)

    combined = ak.concatenate(arrays)

    ak.to_parquet(combined, config.output_path)

    return combined
