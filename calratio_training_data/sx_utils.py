import logging
import os
import re
from enum import Enum
from pathlib import Path
from typing import Tuple
from urllib.parse import unquote, urlparse

from servicex import Sample, ServiceXSpec, dataset


class SXLocationOptions(Enum):
    """Options for which backend we can use"""

    mustUseLocal = "mustUseLocal"
    mustUseRemote = "mustUseRemote"
    anyLocation = "anyLocation"


def build_sx_spec(
    query,
    ds_name: str,
    prefer_local: bool = False,
    backend_name: str = "af.uchicago",
):
    """Build a ServiceX spec from the given query and dataset."""

    # Pass our local preference to find_dataset.
    dataset, location_options = find_dataset(ds_name, prefer_local=prefer_local)

    # Determine whether to use the local endpoint.
    if location_options == SXLocationOptions.mustUseRemote:
        use_local = False
    elif prefer_local or location_options == SXLocationOptions.mustUseLocal:
        use_local = True
    else:
        use_local = False

    adaptor = None
    # Second branch: decide on the backend.
    if use_local:
        codegen_name, backend_name_local, adaptor = install_sx_local()
        backend = backend_name_local
    else:
        backend = backend_name
        codegen_name = "atlasr25"

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

    return spec, backend, adaptor


def find_dataset(
    ds_name: str, prefer_local: bool = False
) -> Tuple[dataset.FileList | dataset.Rucio | dataset.XRootD, SXLocationOptions]:
    """Use heuristics to determine what it is we are after here.
    This function will return a dataset object that can be used to fetch the data.
    It will try to figure out if the input is a URL, a local file, or a Rucio dataset.

    Args:
        ds_name (str): The name of the dataset to be fetched.
        prefer_local (bool): If we can construct the url in a way that will let us
            run locally, then do it. The returned location options must still
            be checked, however. And even if this can't run locally (e.g. rucio
            dataset), no error will be produced.

    Returns:
        _type_: The dataset for ServiceX to use.
    """
    # first, determine what we are looking at.
    what_is_it = None
    if re.match(r"^https?://", ds_name):
        what_is_it = "url"
        url = ds_name

        # Check for the special case of cernbox - which we might be able to convert to
        # a xrootd path.
        if not prefer_local:
            parsed_url = urlparse(url)
            if "cernbox.cern.ch" in parsed_url.netloc and parsed_url.path.startswith(
                "/files/spaces"
            ):
                remote_file = f"root://eospublic.cern.ch{parsed_url.path[13:]}"
                what_is_it = "remote_file"

    elif re.match(r"^file://", ds_name):
        # Convert file URI to a path in a cross-platform way

        parsed_uri = urlparse(ds_name)
        file_path = unquote(parsed_uri.path)
        if os.name == "nt" and file_path.startswith("/"):
            file_path = file_path[1:]

        file = Path(file_path).absolute()
        what_is_it = "file"
    elif re.match(r"^rucio://", ds_name):
        what_is_it = "rucio"
        did = ds_name[8:]
    else:
        # Now we need to use heuristics to decide what this is. If you are running
        # on a file that does not exist you'll get a DID error here. Ugh.
        file = Path(ds_name).absolute()
        if file.exists():
            what_is_it = "file"
        else:
            if os.path.sep in ds_name:
                raise ValueError(
                    f"{ds_name} looks like a file path, but the file does not exist"
                )
            did = ds_name
            what_is_it = "rucio"

    if what_is_it == "url":
        logging.debug(f"Interpreting {ds_name} as a url")
        return dataset.FileList([url]), SXLocationOptions.anyLocation
    elif what_is_it == "file":
        logging.debug(f"Interpreting {ds_name} as a local file ({file})")
        if file.exists():
            # If ds_name is a local file
            logging.debug(f"Interpreting dataset as local file: {file}")
            return dataset.FileList([str(file)]), SXLocationOptions.mustUseLocal
        else:
            raise ValueError(f"This local file {file} does not exist.")
    elif what_is_it == "remote_file":
        logging.debug(f"Interpreting {ds_name} as a remote file ({remote_file})")
        return dataset.FileList([remote_file]), SXLocationOptions.mustUseRemote
    elif what_is_it == "rucio":
        logging.debug(f"Interpreting {ds_name} as a rucio dataset ({did})")
        return dataset.Rucio(did), SXLocationOptions.mustUseRemote
    else:
        raise RuntimeError(f"Unknown type of input {what_is_it}")


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
