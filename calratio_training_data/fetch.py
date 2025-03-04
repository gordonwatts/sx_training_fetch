import logging
import typer

def set_logging(verbosity: int):
    """
    Set logging level based on verbosity.

    Args:
        verbosity (int): Verbosity level (0: WARNING, 1: INFO, 2: DEBUG).
    """
    if verbosity == 0:
        logging.basicConfig(level=logging.WARNING)
    elif verbosity == 1:
        logging.basicConfig(level=logging.INFO)
    elif verbosity >= 2:
        logging.basicConfig(level=logging.DEBUG)


def main(
    dataset: str = typer.Argument(..., help="The data source"),
    verbosity=typer.Option(
        0,
        "--verbose",
        "-v",
        count=True,
        help="Increase verbosity level (use -v for INFO, -vv for DEBUG)",
    ),
):
    """
    Fetch training data for cal ratio.
    """
    set_logging(int(verbosity))
    from .training_query import fetch_training_data
    fetch_training_data(dataset)


def run_from_command():
    "Run from the command line - this is the default"
    typer.run(main)
