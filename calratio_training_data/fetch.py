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
    desc_label: str = typer.Argument(
        ...,
        help='Descriptive label used for labeling datasets. Ex. "HSS, JZ2, data24"',
    ),
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
        desc_label=desc_label,
    )
    fetch_training_data_to_file(dataset, run_config)


@app.command("training-file")
def training_file_command(
    input_files: Optional[List[str]] = typer.Argument(
        None,
        help="Input files or globs optionally with :num_jets",
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="YAML configuration file",
    ),
    event_filter: Optional[str] = typer.Option(
        None,
        "--event-filter",
        help="Expression used to filter events.",
    ),
    output_path: Path = typer.Option(
        None,
        "--output",
        "-o",
        help="Output path for combined dataset.",
    ),
):
    """
    Combines processed datasets into large dataset to be used for training
    """
    from calratio_training_data.combining import (
        combine_training_data,
        CombineConfig,
        parse_input_spec,
        load_yaml_config,
    )

    def merge_config(
        yaml_config: Optional[CombineConfig],
        cli_inputs: Optional[List[str]],
        event_filter: Optional[str],
        output_path: Optional[Path],
    ) -> CombineConfig:

        if yaml_config:
            config = yaml_config
        else:
            config = CombineConfig(inputs=[])

        # CLI inputs override YAML inputs
        if cli_inputs:
            config.inputs = [parse_input_spec(x) for x in cli_inputs]

        # CLI event filter overrides YAML
        if event_filter is not None:
            config.event_filter = event_filter

        # CLI output path overrides YAML
        if output_path is not None:
            config.output_path = output_path

        if not config.inputs:
            raise typer.BadParameter("No input files provided")

        return config

    yaml_config = load_yaml_config(config) if config else None

    final_config = merge_config(
        yaml_config,
        input_files,
        event_filter,
        output_path,
    )

    combine_training_data(final_config)


def run_from_command() -> None:
    "Run from the command line - this is the default"
    app()


if __name__ == "__main__":
    run_from_command()
