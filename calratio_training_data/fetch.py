import logging
import typer

from servicex_local import Platform


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
    logging.basicConfig(level=level)

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
    sx_backend: str = typer.Option(
        "servicex",
        "--sx-backend",
        help="ServiceX backend name",
    ),
    platform: str = typer.Option(
        "docker",
        "--platform",
        help="Container platform to use with local ServiceX (docker, singularity, or wsl2)",
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

    if platform.lower() == "docker":
        platform_to_use = Platform.docker
    elif platform.lower() == "singularity":
        platform_to_use = Platform.singularity
    elif platform.lower() == "wsl2":
        platform_to_use = Platform.wsl2
    else:
        raise ValueError(
            f"Unknown platform {platform}, valid options are docker, singularity, or wsl2"
        )

    run_config = RunConfig(
        ignore_cache=ignore_cache,
        run_locally=local,
        output_path=output,
        mc=mc,
        sx_backend=sx_backend,
        platform=platform_to_use,
    )
    fetch_training_data_to_file(dataset, run_config)


def run_from_command():
    "Run from the command line - this is the default"
    typer.run(main)


if __name__ == "__main__":
    run_from_command()
