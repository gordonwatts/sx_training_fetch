import logging
import re
from enum import Enum
from pathlib import Path
from typing import Tuple

from servicex import Sample, ServiceXSpec, dataset


class SXLocationOptions(Enum):
    """Options for which backend we can use"""

    mustUseLocal = "mustUseLocal"
    mustUseRemote = "mustUseRemote"
    anyLocation = "anyLocation"


def build_sx_spec(query, ds_name: str):
    """Build a ServiceX spec from the given query and dataset."""

    # Convert our dataset argument
    dataset, location_options = find_dataset(ds_name)

    # Determine the backend and codegen we will use, defaulting to
    # running remotely if possible.
    adaptor = None
    if location_options != SXLocationOptions.mustUseLocal:
        backend_name = "af.uchicago"
        codegen_name = "atlasr22"
    else:
        codegen_name, backend_name, adaptor = install_sx_local()

    # Build the ServiceX spec
    spec = ServiceXSpec(
        Sample=[
            Sample(
                Name="MySample",
                Dataset=dataset,
                Query=query,
                Codegen=codegen_name,
            ),
        ],
    )

    return spec, backend_name, adaptor


def find_dataset(
    ds_name: str,
) -> Tuple[dataset.FileList | dataset.Rucio, SXLocationOptions]:
    """Use heuristics to determine what it is we are after here.
    This function will return a dataset object that can be used to fetch the data.
    It will try to figure out if the input is a URL, a local file, or a Rucio dataset.

    Args:
        ds_name (str): The name of the dataset to be fetched.

    Returns:
        _type_: The dataset for ServiceX to use.
    """
    if re.match(r"^https?://", ds_name):
        # If ds_name is an HTTP URL
        logging.debug(f"Interpreting dataset as URL: {ds_name}")
        return dataset.FileList([ds_name]), SXLocationOptions.anyLocation
    else:
        file = Path(ds_name).absolute()
        if file.exists():
            # If ds_name is a local file
            logging.debug(f"Interpreting dataset as local file: {file}")
            return dataset.FileList([str(file)]), SXLocationOptions.mustUseLocal
        else:
            # Otherwise, assume ds_name is a Rucio dataset
            logging.debug(f"Interpreting dataset as Rucio dataset: {ds_name}")
            return dataset.Rucio(ds_name), SXLocationOptions.mustUseRemote


def install_sx_local():
    """
    Set up and register a local ServiceX endpoint for data transformation.

    This function initializes the necessary components for a local ServiceX
    endpoint, including the code generator, science runner, and adaptor.
    It then registers this endpoint with the ServiceX configuration.

    Returns:
        tuple: A tuple containing the names of the codegen and backend.
    """
    from servicex_local import DockerScienceImage, LocalXAODCodegen, SXLocalAdaptor

    codegen_name = "atlasr22-local"

    codegen = LocalXAODCodegen()
    # science_runner = WSL2ScienceImage("atlas_al9", "25.2.12")
    science_runner = DockerScienceImage(
        "sslhep/servicex_func_adl_xaod_transformer:25.2.41"
    )
    adaptor = SXLocalAdaptor(
        codegen, science_runner, codegen_name, "http://localhost:5001"
    )

    logging.info(f"Using local ServiceX endpoint: codegen {codegen_name}")

    return codegen_name, "local-backend", adaptor
