import logging

import awkward as ak
import servicex as sx
from servicex import ServiceXSpec
from servicex.dataset_identifier import FileListDataset, RucioDatasetIdentifier
from servicex_analysis_utils import to_awk

from .sx_utils import build_sx_spec


def fetch_training_data(ds_name: str):
    """
    Fetch the specified dataset.

    Args:
        ds_name (str): The name or identifier of the dataset to fetch.
    """
    # Start the query
    from func_adl_servicex_xaodr22 import FuncADLQueryPHYSLITE

    query_base = FuncADLQueryPHYSLITE()

    # Query the run number, etc.
    query = query_base.Select(
        lambda e: {
            "runNumber": e.EventInfo("EventInfo").runNumber(),
            "eventNumber": e.EventInfo("EventInfo").eventNumber(),
        }
    )

    # Build the ServiceX spec and run it.
    spec, backend_name = build_sx_spec(query, ds_name)
    result_list = to_awk(sx.deliver(spec, servicex_name=backend_name))["MySample"]

    logging.info(f"Received {len(result_list)} entries.")

    # Finally, write it out into a training file.
    ak.to_parquet(result_list, "training.parquet")
