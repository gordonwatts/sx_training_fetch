import logging
from dataclasses import dataclass

import awkward as ak
from func_adl import ObjectStream
from servicex import deliver
import servicex_local as sx_local
from func_adl_servicex_xaodr25 import FADLStream, FuncADLQueryPHYS
from func_adl_servicex_xaodr25.xaod import xAOD
from func_adl_servicex_xaodr25.xAOD.calocluster_v1 import CaloCluster_v1
from func_adl_servicex_xaodr25.xAOD.eventinfo_v1 import EventInfo_v1
from func_adl_servicex_xaodr25.xAOD.jet_v1 import Jet_v1
from func_adl_servicex_xaodr25.xAOD.muonsegment_v1 import MuonSegment_v1
from func_adl_servicex_xaodr25.xAOD.trackparticle_v1 import TrackParticle_v1
from func_adl_servicex_xaodr25.xAOD.vertex_v1 import Vertex_v1
from func_adl_servicex_xaodr25.xAOD.vxtype import VxType
from servicex_analysis_utils import to_awk

from .sx_utils import build_sx_spec
from .cpp_xaod_utils import track_summary_value, cvt_to_calo_cluster


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
    clusters: FADLStream[FADLStream[CaloCluster_v1]]


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
            clusters=[
                [cvt_to_calo_cluster(cl) for cl in j.getConstituents()]
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
            # "track_vertex_nParticles": [len(e.pv_tracks) for t in e.pv_tracks],  # type: ignore
            "track_d0": [t.d0() for t in e.pv_tracks],
            "track_z0": [t.z0() for t in e.pv_tracks],
            "track_chiSquared": [t.chiSquared() for t in e.pv_tracks],
            "track_PixelShared": [
                track_summary_value(t, v_PixelShared) for t in e.pv_tracks
            ],
            "track_SCTShared": [
                track_summary_value(t, v_SCTShared) for t in e.pv_tracks
            ],
            "track_PixelHoles": [
                track_summary_value(t, v_PixelHoles) for t in e.pv_tracks
            ],
            "track_SCTHoles": [track_summary_value(t, v_SCTHoles) for t in e.pv_tracks],
            "track_PixelHits": [
                track_summary_value(t, v_PixelHits) for t in e.pv_tracks
            ],
            "track_SCTHits": [track_summary_value(t, v_SCTHits) for t in e.pv_tracks],
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
            #
            # Clusters
            #   These are written out per-jet.
            #
            # "clus_eta": [[c.eta() for c in c_list] for c_list in e.clusters],
            # "clus_phi": [[c.phi() for c in c_list] for c_list in e.clusters],
            # TODO: Why is missing ::calM mean we can't load this!?
            # "clus_pt": [[c.pt() for c in c_list] for c_list in e.clusters],
            # "clus_l1hcal": [
            #     [e_sample(c, CaloSampling.CaloSample.PreSamplerB) for c in c_list]
            #     for c_list in e.clusters
            # ],
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

    sx_result = (
        deliver(spec, servicex_name=backend_name, ignore_local_cache=ignore_cache)
        if backend_name != "local-backend"
        else sx_local.deliver(spec, adaptor=adaptor, ignore_local_cache=ignore_cache)
    )

    result_list = to_awk(sx_result)["MySample"]

    logging.info(f"Received {len(result_list)} entries.")

    return result_list
