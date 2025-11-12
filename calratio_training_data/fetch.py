import logging
import logging.handlers
from enum import Enum
from typing import Optional, List
from pathlib import Path

import typer


app = typer.Typer()


def set_logging(verbosity: int) -> None:
    """
    Set logging level based on verbosity.

    Args:
        verbosity (int): Verbosity level (0: WARNING, 1: INFO, 2: DEBUG).
    """
    level = (
        logging.WARNING
        if verbosity == 0
        else logging.INFO if verbosity == 1 else logging.DEBUG
    )
    logging.basicConfig(level=level, force=True)

    # This isn't normally set. However, some of our functions need to grab everything, so
    # they may mess with the root logger's level. This keeps the "user" protected.
    for h in logging.getLogger().handlers:
        h.setLevel(level)


class DataType(str, Enum):
    """Allowed data types for the fetch command."""

    SIGNAL = "signal"
    QCD = "qcd"
    DATA = "data"
    BIB = "bib"


@app.command("fetch")
def fetch_command(
    data_type: DataType = typer.Argument(
        ..., help="Type of data to fetch (signal, qcd, data, bib)"
    ),
    dataset: str = typer.Argument(..., help="The data source"),
    verbosity: int = typer.Option(
        0,
        "--verbose",
        "-v",
        count=True,
        help="Increase verbosity level (use -v for INFO, -vv for DEBUG)",
    ),
    ignore_cache: bool = typer.Option(
        False,
        "--ignore-cache",
        help="Ignore cache and fetch fresh data",
    ),
    local: bool = typer.Option(
        False,
        "--local",
        help="Run ServiceX locally (requires docker)",
    ),
    output: str = typer.Option(
        "training.parquet",
        "--output",
        "-o",
        help="Output file path",
    ),
    rotation: bool = typer.Option(
        True,
        "--rotation/--no-rotation",
        help="Applies/does not apply rotations on cluster, track, mseg eta and phi variables. "
        "Rotations applied by default.",
    ),
    sx_backend: Optional[str] = typer.Option(
        None,
        "--sx-backend",
        help="ServiceX backend Name. Default is to use what is in your `servicex.yaml` file.",
    ),
    n_files: Optional[int] = typer.Option(
        None,
        "--n-files",
        "-n",
        help="Number of files to process in the dataset. Default is to process all files.",
    ),
):
    """
    Fetch training data for cal ratio.
    """
    set_logging(int(verbosity))
    from calratio_training_data.training_query import (
        fetch_training_data_to_file,
        RunConfig,
    )

    run_config = RunConfig(
        ignore_cache=ignore_cache,
        run_locally=local,
        output_path=output,
        rotation=rotation,
        sx_backend=sx_backend,
        n_files=n_files,
        datatype=data_type,
    )
    fetch_training_data_to_file(dataset, run_config)


@app.command("training-file")
def training_file_command(
    input_files: List[str] = typer.Argument(
        ..., help="List of input datasets to be combined."
    ),
    event_filter: str = typer.Option(
        None,
        "--event_filter",
        help="Expression that can be used to filter events, default is to run over all events.",
    ),
    output_path: Path = typer.Option(
        "main_training_file.parquet",
        "--ouput",
        "-o",
        help="Output path for combined training dataset.",
    ),
):
    """
    Combines individual processed datasets into large dataset used for training
    """
    from calratio_training_data.combining import combine_training_data, CombineConfig

    combining_config = CombineConfig(
        output_path=output_path,
        event_filter=event_filter,
    )

    combine_training_data(input_files, combining_config)


def run_from_command() -> None:
    "Run from the command line - this is the default"
    app()


if __name__ == "__main__":
    run_from_command()
