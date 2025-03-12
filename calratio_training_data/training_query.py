import logging
from dataclasses import dataclass

import awkward as ak
from func_adl import ObjectStream
import servicex as sx
from func_adl_servicex_xaodr25 import FADLStream
from func_adl_servicex_xaodr25.xAOD.eventinfo_v1 import EventInfo_v1
from func_adl_servicex_xaodr25.xAOD.trackparticle_v1 import TrackParticle_v1
from func_adl_servicex_xaodr25.xAOD.vertex_v1 import Vertex_v1
from func_adl_servicex_xaodr25.xAOD.vxtype import VxType
from func_adl_servicex_xaodr25 import FuncADLQueryPHYSLITE
from servicex_analysis_utils import to_awk

from .sx_utils import build_sx_spec


@dataclass
class TopLevelEvent:
    """Make it easy to type-safe carry everything around.

    Note: SX will only evaluate the terms that are actually asked for in the final query!
    """

    event_info: EventInfo_v1
    vertices: FADLStream[Vertex_v1]
    pv_tracks: FADLStream[TrackParticle_v1]


def build_preselection():
    # Start the query
    query_base = FuncADLQueryPHYSLITE()

    # Establish all the various types of objects we need.
    pv_type = VxType.VertexType.PriVtx.value
    query_base_objects = query_base.Select(
        lambda e: TopLevelEvent(
            event_info=e.EventInfo("EventInfo"),
            vertices=e.Vertices("PrimaryVertices").Where(
                lambda v: v.vertexType() == pv_type
            ),
            pv_tracks=(
                e.Vertices("PrimaryVertices")
                .Where(lambda v: v.vertexType() == pv_type)
                .First()
                .trackParticleLinks()
                .Where(lambda t: t.isValid())  # type: ignore
            ),
        )
    )

    # Preselection
    query_preselection = query_base_objects.Where(
        lambda e: len(e.vertices) > 0 and e.vertices.First().nTrackParticles() > 0  # type: ignore
    )

    return query_preselection


def run_query(ds_name: str, query: ObjectStream):
    # Build the ServiceX spec and run it.
    spec, backend_name = build_sx_spec(query, ds_name)
    result_list = to_awk(
        sx.deliver(
            spec, servicex_name=backend_name, progress_bar=sx.ProgressBarFormat.none
        )
    )["MySample"]

    logging.info(f"Received {len(result_list)} entries.")

    return result_list


def fetch_training_data(ds_name: str):
    """
    Fetch the specified dataset.

    Args:
        ds_name (str): The name or identifier of the dataset to fetch.
    """
    query_preselection = build_preselection()

    # Query the run number, etc.
    query = query_preselection.Select(
        lambda e: {
            "runNumber": e.event_info.runNumber(),
            "eventNumber": e.event_info.eventNumber(),
            "track_pT": [t.pt() / 1000.0 for t in e.pv_tracks],
            "track_eta": [t.eta() for t in e.pv_tracks],
        }
    )

    return run_query(ds_name, query)


def fetch_training_data_to_file(ds_name: str):
    result_list = fetch_training_data(ds_name)

    # Finally, write it out into a training file.
    ak.to_parquet(result_list, "training.parquet")
