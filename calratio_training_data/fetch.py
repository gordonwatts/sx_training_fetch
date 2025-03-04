import logging
import typer


def install_sx_local():
    """
    Set up and register a local ServiceX endpoint for data transformation.

    This function initializes the necessary components for a local ServiceX
    endpoint, including the code generator, science runner, and adaptor.
    It then registers this endpoint with the ServiceX configuration.

    Returns:
        tuple: A tuple containing the names of the codegen and backend.
    """
    from servicex_local import (
        LocalXAODCodegen,
        SXLocalAdaptor,
        DockerScienceImage,
    )
    from servicex.configuration import Configuration, Endpoint
    from servicex_local.adaptor import MinioLocalAdaptor

    codegen_name = "atlasr22-local"
    backend_name = "local-backend"

    codegen = LocalXAODCodegen()
    # science_runner = WSL2ScienceImage("atlas_al9", "25.2.12")
    science_runner = DockerScienceImage(
        "sslhep/servicex_func_adl_xaod_transformer:22.2.107"
    )
    adaptor = SXLocalAdaptor(
        codegen, science_runner, codegen_name, "http://localhost:5001"
    )

    Configuration.register_endpoint(
        Endpoint(
            name=backend_name,
            adapter=adaptor,
            minio=MinioLocalAdaptor.for_transform,  # type: ignore
            endpoint="bogus-endpoint-name",
        )
    )

    return codegen_name, backend_name


def find_dataset(ds_name: str):
    """Use heuristics to determine what it is we are after here.
    This function will return a dataset object that can be used to fetch the data.
    It will try to figure out if the input is a URL, a local file, or a Rucio dataset.

    Args:
        ds_name (str): The name of the dataset to be fetched.

    Returns:
        _type_: The dataset for ServiceX to use.
    """
    from pathlib import Path
    from servicex import dataset
    import re

    if re.match(r"^https?://", ds_name):
        # If ds_name is an HTTP URL
        logging.debug(f"Interpreting dataset as URL: {ds_name}")
        return dataset.FileList([ds_name])
    else:
        file = Path(ds_name).absolute()
        if file.exists():
            # If ds_name is a local file
            logging.debug(f"Interpreting dataset as local file: {file}")
            return dataset.FileList([str(file)])
        else:
            # Otherwise, assume ds_name is a Rucio dataset
            logging.debug(f"Interpreting dataset as Rucio dataset: {ds_name}")
            return dataset.Rucio(ds_name)


def fetch_dataset(ds_name: str):
    """
    Fetch the specified dataset.

    Args:
        ds_name (str): The name or identifier of the dataset to fetch.
    """
    # Register the back end
    codegen_name, backend_name = install_sx_local()
    logging.info(f"Using codegen: {codegen_name}, backend: {backend_name}")

    # Convert our dataset argument
    dataset = find_dataset(ds_name)

    # Replace the following line with the actual fetch command
    print(f"Fetching dataset: {ds_name}")


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
    fetch_dataset(dataset)


def run_from_command():
    "Run from the command line - this is the default"
    typer.run(main)
