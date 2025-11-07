from typing import Optional
from servicex import Sample, ServiceXSpec
from servicex_local import install_sx_local, find_dataset, Platform


def extract_run_number_and_name(ds_name: str) -> Tuple[Optional[str], str]:
    """Extract the run number and dataset name from an ATLAS DID string.

    Args:
        ds_name (str): The dataset identifier to parse.

    Returns:
        Tuple[Optional[str], str]: A tuple containing the run number (if found)
        and the dataset name. When parsing fails, the dataset name is the first
        30 characters of ``ds_name`` and the run number is ``None``.
    """

    # Strip whitespace so we can reliably parse the DID string.
    did = ds_name.strip()

    # Match an optional scope (``scope:``), followed by the project and the run
    # number. The dataset name is the string immediately following the run.
    did_pattern = re.compile(r"(?:[^:]+:)?[^.]+\.(?P<run>\d+)\.(?P<name>[^.]+)")
    match = did_pattern.search(did)
    if not match:
        truncated_name = did[:30]
        return None, truncated_name

    run_number = match.group("run")
    dataset_name = match.group("name")

    return run_number, dataset_name


def build_sx_spec(
    query,
    ds_name: str,
    prefer_local: bool = False,
    backend_name: Optional[str] = None,
    n_files: Optional[int] = None,
    platform: Platform = Platform.docker,
):
    """Build a ServiceX spec from the given query and dataset."""

    # Pass our local preference to find_dataset.
    dataset, use_local = find_dataset(ds_name, prefer_local=prefer_local)
    adaptor = None

    if use_local is not prefer_local:
        raise ValueError(
            f"Unable to run dataset {ds_name} in preferred configuration. Did you forget "
            "the `--local` flag?"
        )

    # Second branch: decide on the backend.

    if platform == Platform.singularity:
        image = "docker://sslhep/servicex_func_adl_xaod_transformer:25.2.41"
    elif platform == Platform.docker:
        image = "sslhep/servicex_func_adl_xaod_transformer:25.2.41"
    else:
        raise RuntimeError(f"No image is present for the platform: {platform}")

    if use_local:
        codegen_name, adaptor = install_sx_local(image, platform)
        backend = "sx_local"
    else:
        backend = backend_name
        codegen_name = "atlasr25"

    # Build the ServiceX spec
    run_number, dataset_name = extract_run_number_and_name(ds_name)
    if run_number:
        sample_name = f"calratio_{run_number}_{dataset_name}"
    else:
        sample_name = f"calratio_{dataset_name}"

    spec = ServiceXSpec(
        Sample=[  # type: ignore
            Sample(
                Name=sample_name,
                Dataset=dataset,
                Query=query,
                Codegen=codegen_name,
                NFiles=n_files,
            ),
        ],
    )

    return spec, use_local, backend, adaptor
