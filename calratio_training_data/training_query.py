from dataclasses import dataclass
import logging
from tabnanny import verbose

import awkward as ak
import servicex as sx
from servicex import ServiceXSpec
from servicex.dataset_identifier import FileListDataset, RucioDatasetIdentifier
from servicex_analysis_utils import to_awk

from .sx_utils import build_sx_spec

from func_adl_servicex_xaodr22.xAOD.eventinfo_v1 import EventInfo_v1
from func_adl_servicex_xaodr22.xAOD.vertex_v1 import Vertex_v1
from func_adl_servicex_xaodr22 import FADLStream


@dataclass
class TopLevelEvent:
    event_info: EventInfo_v1
    verbose: FADLStream[Vertex_v1]


def fetch_training_data(ds_name: str):
    """
    Fetch the specified dataset.

    Args:
        ds_name (str): The name or identifier of the dataset to fetch.
    """
    # Start the query
    from func_adl_servicex_xaodr22 import FuncADLQueryPHYSLITE

    query_base = FuncADLQueryPHYSLITE()

    # Establish all the various types of objects we need.
    query_base_objects = query_base.Select(
        lambda e: {
            "event_info": e.EventInfo("EventInfo"),
            "vertices": e.Vertices("PrimaryVertices"),
            "pv_tracks": e.Vertices("PrimaryVertices").First().trackParticleLinks(),
        }
    )

    # Preselection
    query_preselection = query_base_objects.Where(
        lambda e: e["vertices"].First().nTrackParticles() > 0
    )

    # Query the run number, etc.
    query = query_preselection.Select(
        lambda e: {
            "runNumber": e["event_info"].runNumber(),
            "eventNumber": e["event_info"].eventNumber(),
            "track_pT": e["pv_tracks"].Select(lambda t: t.pt()),
        }
    )

    # Build the ServiceX spec and run it.
    spec, backend_name = build_sx_spec(query, ds_name)
    result_list = to_awk(sx.deliver(spec, servicex_name=backend_name))["MySample"]

    logging.info(f"Received {len(result_list)} entries.")

    # Finally, write it out into a training file.
    ak.to_parquet(result_list, "training.parquet")
