import ast
import logging
from dataclasses import dataclass
from typing import Tuple, TypeVar

import awkward as ak
from func_adl import ObjectStream, func_adl_callable
import servicex as sx
import servicex_local as sx_local
from func_adl_servicex_xaodr25 import FADLStream
from func_adl_servicex_xaodr25.xAOD.eventinfo_v1 import EventInfo_v1
from func_adl_servicex_xaodr25.xAOD.trackparticle_v1 import TrackParticle_v1
from func_adl_servicex_xaodr25.xAOD.vertex_v1 import Vertex_v1
from func_adl_servicex_xaodr25.xAOD.muonsegment_v1 import MuonSegment_v1
from func_adl_servicex_xaodr25.xAOD.jet_v1 import Jet_v1
from func_adl_servicex_xaodr25.xAOD.vxtype import VxType
from func_adl_servicex_xaodr25.xaod import xAOD
from func_adl_servicex_xaodr25 import FuncADLQueryPHYS
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
    muon_segments: FADLStream[MuonSegment_v1]
    jets: FADLStream[Jet_v1]


T = TypeVar("T")


def track_summary_value_callback(
    s: ObjectStream[T], a: ast.Call
) -> Tuple[ObjectStream[T], ast.Call]:
    """The trackSummary method returns true/false if the value is there,
    and alter an argument passed by reference. In short, this isn't functional,
    so it won't work in `func_adl`. This wraps it to make it "work".

    Args:
        s (ObjectStream[T]): The stream we are operating against
        a (ast.Call): The actual call

    Returns:
        Tuple[ObjectStream[T], ast.Call]: Return the updated stream with the metdata code.
    """
    new_s = s.MetaData(
        {
            "metadata_type": "add_cpp_function",
            "name": "trackSummaryValue",
            "code": [
                "uint8_t result;\n"
                "xAOD::SummaryType st (static_cast<xAOD::SummaryType>(value_selector));\n"
                "if (!(*trk)->summaryValue(result, st)) {\n"
                "  result = -1;\n"
                "}\n"
            ],
            "result": "result",
            "include_files": [],
            "arguments": ["trk", "value_selector"],
            "return_type": "float",
        }
    )
    return new_s, a


# Declare the typing and name of the function to func_adl
@func_adl_callable(track_summary_value_callback)
def trackSummaryValue(trk: TrackParticle_v1, value_selector: int) -> int:
    """Call the `trackSummary` method on a track.

    * Return the value of the value_selector for the track
    * If it isn't present, return -1.

    Args:
        trk (TrackParticle_v1): The track we are operating against
        value_selector (int): Which value (pixel holes, etc.)

    NOTE: This is a dummy function that injects C++ into the object stream to do the
    actual work.

    Returns:
        int: Value requested or -1 if not available.
    """
    ...


def build_preselection():
    # Start the query
    query_base = FuncADLQueryPHYS()

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
            muon_segments=e.MuonSegments("MuonSegments"),
            jets=[
                j
                for j in e.Jets(collection="AntiKt4EMTopoJets", calibrate=False)
                if j.pt() / 1000.0 > 40.0
            ],  # type: ignore
        )
    )

    # Preselection
    query_preselection = query_base_objects.Where(
        lambda e: len(e.vertices) > 0 and e.vertices.First().nTrackParticles() > 0  # type: ignore
    )

    return query_preselection


def fetch_training_data(ds_name: str, ignore_cache: bool):
    """
    Fetch the specified dataset.

    Args:
        ds_name (str): The name or identifier of the dataset to fetch.
    """
    # Get the base query
    query_preselection = build_preselection()

    v_PixelShared = xAOD.SummaryType.numberOfPixelSharedHits.value
    v_SCTShared = xAOD.SummaryType.numberOfSCTSharedHits.value
    v_SCTHoles = xAOD.SummaryType.numberOfSCTHoles.value
    v_PixelHits = xAOD.SummaryType.numberOfPixelHits.value
    v_PixelHoles = xAOD.SummaryType.numberOfPixelHoles.value
    v_SCTHits = xAOD.SummaryType.numberOfSCTHits.value

    # Query the run number, etc.
    query = query_preselection.Select(
        lambda e: {
            "runNumber": e.event_info.runNumber(),
            "eventNumber": e.event_info.eventNumber(),
            #
            # Track Info
            #
            "track_pT": [t.pt() / 1000.0 for t in e.pv_tracks],
            "track_eta": [t.eta() for t in e.pv_tracks],
            "track_phi": [t.phi() for t in e.pv_tracks],
            # TODO: If we are limiting tracks to the PV, is there any point in this
            # input variable?
            # See bug func_adl/issues/181
            "track_vertex_nParticles": [len(e.pv_tracks) for t in e.pv_tracks],  # type: ignore
            "track_d0": [t.d0() for t in e.pv_tracks],
            "track_z0": [t.z0() for t in e.pv_tracks],
            "track_chiSquared": [t.chiSquared() for t in e.pv_tracks],
            "track_PixelShared": [
                trackSummaryValue(t, v_PixelShared) for t in e.pv_tracks
            ],
            "track_SCTShared": [trackSummaryValue(t, v_SCTShared) for t in e.pv_tracks],
            "track_PixelHoles": [
                trackSummaryValue(t, v_PixelHoles) for t in e.pv_tracks
            ],
            "track_SCTHoles": [trackSummaryValue(t, v_SCTHoles) for t in e.pv_tracks],
            "track_PixelHits": [trackSummaryValue(t, v_PixelHits) for t in e.pv_tracks],
            "track_SCTHits": [trackSummaryValue(t, v_SCTHits) for t in e.pv_tracks],
            #
            # Muon Segments. We will convert to eta and phi after we load these guys.
            #
            "MSeg_x": [s.x() for s in e.muon_segments],
            "MSeg_y": [s.y() for s in e.muon_segments],
            "MSeg_z": [s.z() for s in e.muon_segments],
            "MSeg_px": [s.px() for s in e.muon_segments],
            "MSeg_py": [s.py() for s in e.muon_segments],
            "MSeg_pz": [s.pz() for s in e.muon_segments],
            "MSeg_t0": [s.t0() for s in e.muon_segments],
            "MSeg_chiSquared": [s.chiSquared() for s in e.muon_segments],
            #
            # Jets
            #
            "jet_pt": [j.pt() / 1000.0 for j in e.jets],
            "jet_eta": [j.eta() for j in e.jets],
            "jet_phi": [j.phi() for j in e.jets],
        }
    )

    return run_query(ds_name, query, ignore_cache)


def fetch_training_data_to_file(ds_name: str, ignore_cache: bool):
    result_list = fetch_training_data(ds_name, ignore_cache)

    # Finally, write it out into a training file.
    ak.to_parquet(result_list, "training.parquet")


def run_query(ds_name: str, query: ObjectStream, ignore_cache: bool):
    # Build the ServiceX spec and run it.
    spec, backend_name, adaptor = build_sx_spec(query, ds_name)
    result_list = to_awk(
        # sx.deliver(
        #     spec, servicex_name=backend_name, progress_bar=sx.ProgressBarFormat.none
        # )
        sx_local.deliver(spec, adaptor=adaptor, ignore_local_cache=ignore_cache)
    )["MySample"]

    logging.info(f"Received {len(result_list)} entries.")

    return result_list
