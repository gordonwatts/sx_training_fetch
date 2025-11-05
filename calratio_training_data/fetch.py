import logging
import logging.handlers
from typing import Optional

import typer


def set_logging(verbosity: int):
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


def main(
    dataset: str = typer.Argument(..., help="The data source"),
    verbosity=typer.Option(
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
    mc: bool = typer.Option(
        False,
        "--mc",
        help="Include LLP MC truth info (will crash if run on file without MC "
        "info!)",
    ),
    do_rotation: bool = typer.Option(
        True,
        "--no_rotation",
        help="Skips rotation step on cluster, track, mseg eta and phi variables. Without the flag "
        "the rotations are applied.",
    ),
    sx_backend: Optional[str] = typer.Option(
        None,
        "--sx-backend",
        help="ServiceX backend Name. Default is to use what is in your `servicex.yaml` file.",
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
        mc=mc,
        do_rotation=do_rotation,
        sx_backend=sx_backend,
    )
    fetch_training_data_to_file(dataset, run_config)


def run_from_command():
    "Run from the command line - this is the default"
    typer.run(main)


if __name__ == "__main__":
    run_from_command()
