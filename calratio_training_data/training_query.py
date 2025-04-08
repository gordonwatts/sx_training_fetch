import logging
from dataclasses import dataclass

import awkward as ak
import servicex_local as sx_local
from func_adl import ObjectStream
from func_adl_servicex_xaodr25 import FADLStream, FuncADLQueryPHYS

from func_adl_servicex_xaodr25.calosampling import CaloSampling
from func_adl_servicex_xaodr25.xaod import xAOD
from func_adl_servicex_xaodr25.xAOD.calocluster_v1 import CaloCluster_v1
from func_adl_servicex_xaodr25.xAOD.eventinfo_v1 import EventInfo_v1
from func_adl_servicex_xaodr25.xAOD.jet_v1 import Jet_v1
from func_adl_servicex_xaodr25.xAOD.muonsegment_v1 import MuonSegment_v1
from func_adl_servicex_xaodr25.xAOD.trackparticle_v1 import TrackParticle_v1
from func_adl_servicex_xaodr25.xAOD.vertex_v1 import Vertex_v1
from func_adl_servicex_xaodr25.xAOD.vxtype import VxType
from servicex import deliver
from servicex_analysis_utils import to_awk

from .cpp_xaod_utils import cvt_to_raw_calocluster, track_summary_value
from .sx_utils import build_sx_spec


# New data class for run configuration options
@dataclass
class RunConfig:
    ignore_cache: bool
    run_locally: bool


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
    jet_clusters: FADLStream[FADLStream[CaloCluster_v1]]
    topo_clusters: FADLStream[CaloCluster_v1]

    # All tracks with no selection at all. From Inner Detector container
    all_tracks: FADLStream[TrackParticle_v1]


def build_preselection():
    # Start the query
    query_base = FuncADLQueryPHYS()

    # Establish all the various types of objects we need.
    query_base_objects = query_base.Select(
        lambda e: TopLevelEvent(
            event_info=e.EventInfo("EventInfo"),
            vertices=e.Vertices("PrimaryVertices").Where(
                lambda v: v.vertexType() == VxType.VertexType.PriVtx
            ),
            pv_tracks=(
                e.Vertices("PrimaryVertices")
                .Where(lambda v: v.vertexType() == VxType.VertexType.PriVtx)
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
            jet_clusters=[
                [
                    cvt_to_raw_calocluster(cl)
                    for cl in j.constituentLinks()
                    if cl.isValid()
                ]
                for j in e.Jets(collection="AntiKt4EMTopoJets", calibrate=False)
                if j.pt() / 1000.0 > 40.0
            ],  # type: ignore
            all_tracks=e.TrackParticles("InDetTrackParticles"),
            topo_clusters=e.CaloClusters("CaloCalTopoClusters"),
        )
    )

    # Preselection
    query_preselection = query_base_objects.Where(
        lambda e: len(e.vertices) > 0  # type: ignore
        and e.vertices.First().nTrackParticles() > 0
        and len(e.jets) > 0  # type: ignore
    )

    return query_preselection


def fetch_training_data(
    ds_name: str, config: RunConfig = RunConfig(ignore_cache=False, run_locally=False)
):
    """
    Fetch the specified dataset.

    Args:
        ds_name (str): The dataset identifier.
        config (RunConfig): Run configuration options.
    """
    # Get the base query
    query_preselection = build_preselection()

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
            "track_vertex_nParticles": [len(e.pv_tracks) for t in e.pv_tracks],  # type: ignore
            "track_d0": [t.d0() for t in e.pv_tracks],
            "track_z0": [t.z0() for t in e.pv_tracks],
            "track_chiSquared": [t.chiSquared() for t in e.pv_tracks],
            "track_PixelShared": [
                track_summary_value(t, xAOD.SummaryType.numberOfPixelSharedHits)
                for t in e.pv_tracks
            ],
            "track_SCTShared": [
                track_summary_value(t, xAOD.SummaryType.numberOfSCTSharedHits)
                for t in e.pv_tracks
            ],
            "track_PixelHoles": [
                track_summary_value(t, xAOD.SummaryType.numberOfPixelHoles)
                for t in e.pv_tracks
            ],
            "track_SCTHoles": [
                track_summary_value(t, xAOD.SummaryType.numberOfSCTHoles)
                for t in e.pv_tracks
            ],
            "track_PixelHits": [
                track_summary_value(t, xAOD.SummaryType.numberOfPixelHits)
                for t in e.pv_tracks
            ],
            "track_SCTHits": [
                track_summary_value(t, xAOD.SummaryType.numberOfSCTHits)
                for t in e.pv_tracks
            ],
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
            #   Write out all clusters
            #   Layer definitions come from https://gitlab.cern.ch/atlas-phys-exotics-llp-mscrid
            #       /fullrun2analysis/DiVertAnalysisR21/-/blob/master/DiVertAnalysis/Root
            #       /RegionVarCalculator_calRatio.cxx?ref_type=heads#L381
            # These are a double-nested list since the jet association is implicit in the xAOD.
            "clus_eta": [
                c.eta() for jet_clusters in e.jet_clusters for c in jet_clusters
            ],
            "clus_phi": [
                c.phi() for jet_clusters in e.jet_clusters for c in jet_clusters
            ],
            "clus_pt": [
                c.pt() / 1000.0 for jet_clusters in e.jet_clusters for c in jet_clusters
            ],
            "clus_l1hcal": [
                c.eSample(CaloSampling.CaloSample.HEC0)
                for jet_clusters in e.jet_clusters
                for c in jet_clusters
            ],
            "clus_l2hcal": [
                c.eSample(CaloSampling.CaloSample.HEC1)
                + c.eSample(CaloSampling.CaloSample.TileBar0)
                + c.eSample(CaloSampling.CaloSample.TileGap1)
                + c.eSample(CaloSampling.CaloSample.TileExt0)
                for jet_clusters in e.jet_clusters
                for c in jet_clusters
            ],
            "clus_l3hcal": [
                c.eSample(CaloSampling.CaloSample.HEC2)
                + c.eSample(CaloSampling.CaloSample.TileBar1)
                + c.eSample(CaloSampling.CaloSample.TileGap2)
                + c.eSample(CaloSampling.CaloSample.TileExt1)
                for jet_clusters in e.jet_clusters
                for c in jet_clusters
            ],
            "clus_l4hcal": [
                c.eSample(CaloSampling.CaloSample.HEC3)
                + c.eSample(CaloSampling.CaloSample.TileBar2)
                + c.eSample(CaloSampling.CaloSample.TileGap3)
                + c.eSample(CaloSampling.CaloSample.TileExt2)
                for jet_clusters in e.jet_clusters
                for c in jet_clusters
            ],
            "clus_l1ecal": [
                c.eSample(CaloSampling.CaloSample.PreSamplerB)
                + c.eSample(CaloSampling.CaloSample.PreSamplerE)
                for jet_clusters in e.jet_clusters
                for c in jet_clusters
            ],
            "clus_l2ecal": [
                c.eSample(CaloSampling.CaloSample.EMB1)
                + c.eSample(CaloSampling.CaloSample.EME1)
                + c.eSample(CaloSampling.CaloSample.FCAL0)
                for jet_clusters in e.jet_clusters
                for c in jet_clusters
            ],
            "clus_l3ecal": [
                c.eSample(CaloSampling.CaloSample.EMB2)
                + c.eSample(CaloSampling.CaloSample.EME2)
                + c.eSample(CaloSampling.CaloSample.FCAL1)
                for jet_clusters in e.jet_clusters
                for c in jet_clusters
            ],
            "clus_l4ecal": [
                c.eSample(CaloSampling.CaloSample.EMB3)
                + c.eSample(CaloSampling.CaloSample.EME3)
                + c.eSample(CaloSampling.CaloSample.FCAL2)
                for jet_clusters in e.jet_clusters
                for c in jet_clusters
            ],
            "clus_time": [
                c.time() for jet_clusters in e.jet_clusters for c in jet_clusters
            ],
        }
    )

    return run_query(ds_name, query, config)


def fetch_training_data_to_file(ds_name: str, config: RunConfig):
    result_list = fetch_training_data(ds_name, config)

    # Finally, write it out into a training file.
    ak.to_parquet(
        result_list, "training.parquet", compression="GZIP", compression_level=9
    )


def run_query(
    ds_name: str,
    query: ObjectStream,
    config: RunConfig = RunConfig(ignore_cache=False, run_locally=False),
):
    # Build the ServiceX spec and run it.
    spec, backend_name, adaptor = build_sx_spec(query, ds_name, config.run_locally)
    if config.run_locally or backend_name == "local-backend":
        sx_result = sx_local.deliver(
            spec, adaptor=adaptor, ignore_local_cache=config.ignore_cache
        )
    else:
        if config.run_locally:
            raise ValueError(f"Unable to run dataset {ds_name} locally.")
        sx_result = deliver(
            spec, servicex_name=backend_name, ignore_local_cache=config.ignore_cache
        )
    result_list = to_awk(sx_result)["MySample"]
    logging.info(f"Received {len(result_list)} entries.")
    return result_list
