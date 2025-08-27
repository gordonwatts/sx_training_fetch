from servicex import Sample, ServiceXSpec
from servicex_local import install_sx_local, find_dataset, Platform

def build_sx_spec(
    query,
    ds_name: str,
    prefer_local: bool = False,
    backend_name: str = "servicex",
    platform: Platform = Platform.docker,
):
    """Build a ServiceX spec from the given query and dataset."""

    # Pass our local preference to find_dataset.
    dataset, use_local = find_dataset(ds_name, prefer_local=prefer_local)
    adaptor = None

    if use_local is not prefer_local:
        raise ValueError(f"Unable to run dataset {ds_name} in prefered configuration.")

    # Second branch: decide on the backend.
    if use_local:
        codegen_name, adaptor = install_sx_local("docker://sslhep/servicex_func_adl_xaod_transformer:25.2.41", platform)
        backend = "sx_local"
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

    return spec, use_local, backend, adaptor