import logging
from dataclasses import dataclass
from math import sqrt
from typing import Dict

import awkward as ak
import numpy as np
import servicex_local as sx_local
import vector
from func_adl import ObjectStream
from func_adl_servicex_xaodr25 import FADLStream, FuncADLQueryPHYS
from func_adl_servicex_xaodr25.calosampling import CaloSampling
from func_adl_servicex_xaodr25.xaod import xAOD
from func_adl_servicex_xaodr25.xAOD.calocluster_v1 import CaloCluster_v1
from func_adl_servicex_xaodr25.xAOD.eventinfo_v1 import EventInfo_v1
from func_adl_servicex_xaodr25.xAOD.jet_v1 import Jet_v1
from func_adl_servicex_xaodr25.xAOD.muonsegment_v1 import MuonSegment_v1
from func_adl_servicex_xaodr25.xAOD.trackparticle_v1 import TrackParticle_v1
from func_adl_servicex_xaodr25.xAOD.truthparticle_v1 import TruthParticle_v1
from func_adl_servicex_xaodr25.xAOD.vertex_v1 import Vertex_v1
from func_adl_servicex_xaodr25.xAOD.vxtype import VxType
from servicex import deliver
from servicex_analysis_utils import to_awk

from calratio_training_data.processing import processing_func

from calratio_training_data.constants import (
    JET_MSEG_DELTA_PHI,
    JET_TRACK_DELTA_R,
    LLP_JET_DELTA_R,
    LLP_central_eta_cut,
    LLP_Lxy_max,
    LLP_Lxy_min,
    LLP_Lz_max,
    LLP_Lz_min,
    min_jet_pt,
    max_jet_pt,
)

from .cpp_xaod_utils import (
    add_jet_selection_tool,
    cvt_to_raw_calocluster,
    jet_clean_llp,
    track_summary_value,
)


vector.register_awkward()


# New data class for run configuration options
@dataclass
class RunConfig:
    ignore_cache: bool
    run_locally: bool
    output_path: str = "training.parquet"
    mc: bool = False
    sx_backend: str = "servicex"


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

    # Truth particles
    bsm_particles: FADLStream[TruthParticle_v1]


def good_training_jet(jet: Jet_v1) -> bool:
    """Check that the jet is suitable for training"""
    return jet.pt() / 1000.0 > 40.0 and abs(jet.eta()) < 2.5 and jet_clean_llp(jet)


def build_preselection():
    # Start the query
    query_base = add_jet_selection_tool(
        FuncADLQueryPHYS(), "m_jetCleaning_llp", "LooseBadLLP"
    )

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
                if good_training_jet(j)
            ],  # type: ignore
            jet_clusters=[
                [
                    cvt_to_raw_calocluster(cl)
                    for cl in j.constituentLinks()
                    if cl.isValid()
                ]
                for j in e.Jets(collection="AntiKt4EMTopoJets", calibrate=False)
                if good_training_jet(j)
            ],  # type: ignore
            all_tracks=e.TrackParticles("InDetTrackParticles"),
            topo_clusters=e.CaloClusters("CaloCalTopoClusters"),
            bsm_particles=e.TruthParticles("TruthBSMWithDecayParticles").Where(
                lambda truth_p: truth_p.absPdgId() == 35 or truth_p.absPdgId() == 51
            ),
        )
    )

    # Preselection
    query_preselection = query_base_objects.Where(
        lambda e: len(e.vertices) > 0  # type: ignore
        and e.vertices.First().nTrackParticles() > 0
        and len(e.jets) > 0  # type: ignore
    )

    return query_preselection


def fetch_raw_training_data(
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
            "mcEventWeight": e.event_info.mcEventWeight(0),
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
            **(
                {
                    "LLP_eta": [p.eta() for p in e.bsm_particles],
                    "LLP_phi": [p.phi() for p in e.bsm_particles],
                    "LLP_pt": [p.pt() / 1000.0 for p in e.bsm_particles],
                    "LLP_pdgid": [p.absPdgId() for p in e.bsm_particles],
                    "LLP_Lz": [
                        p.decayVtx().z() if p.hasDecayVtx() else 0.0
                        for p in e.bsm_particles
                    ],
                    "LLP_Lxy": [
                        (
                            sqrt(p.decayVtx().x() ** 2 + p.decayVtx().y() ** 2)
                            if p.hasDecayVtx()
                            else 0.0
                        )
                        for p in e.bsm_particles
                    ],
                }
                if config.mc
                else {}
            ),
        }
    )

    return run_query(ds_name, query, config)


def convert_to_training_data(data: Dict[str, ak.Array], mc: bool = False) -> ak.Array:
    """
    Convert raw data dictionary to training data format.

    Args:
        raw_data (Dict[str, ak.Array]): The raw data as returned by run_query.
        mc (bool): If True, include LLP info.

    Returns:
        ak.Record: The processed training data, suitable for writing to parquet.
    """
    # Build the constructs we can use to do matching (associated them with 3D vectors!).
    jets = ak.values_astype(
        ak.zip(
            {
                "pt": data["jet_pt"],
                "eta": data["jet_eta"],
                "phi": data["jet_phi"],
            },
            with_name="Momentum3D",
        ),
        np.float32,
    )

    tracks = ak.values_astype(
        ak.zip(
            {
                "eta": data.track_eta,  # type: ignore
                "phi": data.track_phi,  # type: ignore
                "pt": data.track_pT,  # type: ignore
                "vertex_nParticles": data.track_vertex_nParticles,  # type: ignore
                "d0": data.track_d0,  # type: ignore
                "z0": data.track_z0,  # type: ignore
                "chiSquared": data.track_chiSquared,  # type: ignore
                "PixelShared": data.track_PixelShared,  # type: ignore
                "SCTShared": data.track_SCTShared,  # type: ignore
                "PixelHoles": data.track_PixelHoles,  # type: ignore
                "SCTHoles": data.track_SCTHoles,  # type: ignore
                "PixelHits": data.track_PixelHits,  # type: ignore
                "SCTHits": data.track_SCTHits,  # type: ignore
            },
            with_name="Momentum3D",
        ),
        np.float32,
    )

    msegs = ak.values_astype(
        ak.zip(
            {
                "x": data.MSeg_x,  # type: ignore
                "y": data.MSeg_y,  # type: ignore
                "z": data.MSeg_z,  # type: ignore
                "t0": data.MSeg_t0,  # type: ignore
                "chiSquared": data.MSeg_chiSquared,  # type: ignore
            },
            with_name="Vector3D",
        ),
        np.float32,
    )

    msegs_p = ak.values_astype(
        ak.zip(
            {
                "px": data.MSeg_px,  # type: ignore
                "py": data.MSeg_py,  # type: ignore
                "pz": data.MSeg_pz,  # type: ignore
            },
            with_name="Momentum3D",
        ),
        np.float32,
    )

    clusters = ak.values_astype(
        ak.zip(
            {
                "eta": data.clus_eta,  # type: ignore
                "phi": data.clus_phi,  # type: ignore
                "pt": data.clus_pt,  # type: ignore
                "l1hcal": data.clus_l1hcal,  # type: ignore
                "l2hcal": data.clus_l2hcal,  # type: ignore
                "l3hcal": data.clus_l3hcal,  # type: ignore
                "l4hcal": data.clus_l4hcal,  # type: ignore
                "l1ecal": data.clus_l1ecal,  # type: ignore
                "l2ecal": data.clus_l2ecal,  # type: ignore
                "l3ecal": data.clus_l3ecal,  # type: ignore
                "l4ecal": data.clus_l4ecal,  # type: ignore
                "time": data.clus_time,  # type: ignore
            },
            with_name="Momentum3D",
        ),
        np.float32,
    )

    # If we are doing signal, then we only want LLP's that are close to jets.
    if mc:
        llps = ak.values_astype(
            ak.zip(
                {
                    "eta": data["LLP_eta"],
                    "phi": data["LLP_phi"],
                    "pt": data["LLP_pt"],
                    "Lz": data["LLP_Lz"],
                    "Lxy": data["LLP_Lxy"],
                },
                with_name="Momentum3D",
            ),
            np.float32,
        )

        # Next make sure the LLP's decay in the calorimeter region.
        # if they are in the central region, then Lxy must be between LLP_Lxy_min and LLP_Lxy_max
        # if they are in the end-cap region, then Lz must be between LLP_Lz_min and LLP_Lz_max
        llps = llps[  # type: ignore
            (abs(llps.eta) < LLP_central_eta_cut)
            & (llps.Lxy > LLP_Lxy_min)
            & (llps.Lxy < LLP_Lxy_max)
            | (abs(llps.eta) >= LLP_central_eta_cut)
            & (abs(llps.Lz) > LLP_Lz_min)
            & (abs(llps.Lz) < LLP_Lz_max)
        ]

        llp_jet_pairs = ak.cartesian(
            {
                "jet": jets,
                "llp": llps,
            },
            axis=1,
            nested=True,
        )
        delta_r_jet_llp = llp_jet_pairs.jet.deltaR(llp_jet_pairs.llp)
        jets_near_llps_mask = ak.any(delta_r_jet_llp < LLP_JET_DELTA_R, axis=-1)

        # Window the jets (and clusters, which come pre-associated with the jets) to
        # only those near LLPs.
        jets = jets[jets_near_llps_mask]
        clusters = clusters[jets_near_llps_mask]
        if ak.count(jets) == 0:
            raise ValueError("No jets found near LLPs.")

        # And for those jets, get a match LLP. Easiest is to re-run the matching.
        llp_jet_pairs = ak.cartesian(
            {
                "jet": jets,
                "llp": llps,
            },
            axis=1,
            nested=True,
        )
        llp_match_jet_index = ak.argmin(
            llp_jet_pairs.jet.deltaR(llp_jet_pairs.llp), axis=-1
        )
        llp_match_jet = llps[llp_match_jet_index]

    # Compute DeltaR between each jet and all tracks in the same event
    jet_track_pairs = ak.cartesian({"jet": jets, "track": tracks}, axis=1, nested=True)
    delta_r = jet_track_pairs.jet.deltaR(jet_track_pairs.track)
    nearby_tracks = jet_track_pairs.track[delta_r < JET_TRACK_DELTA_R]

    # delta-phi matching for muon segments.
    jet_mseg_pairs = ak.cartesian(
        {
            "jet": jets,
            "mseg": ak.zip({"x": msegs, "p": msegs_p}),
        },
        axis=1,
        nested=True,
    )
    delta_phi = jet_mseg_pairs.jet.deltaphi(jet_mseg_pairs.mseg.x)
    mseg_mask = delta_phi < JET_MSEG_DELTA_PHI
    nearby_msegs = jet_mseg_pairs.mseg[mseg_mask]

    # Fill this dict with the leaves we want in the training data.
    per_jet_training_data_dict = {}

    # Build the final per-jet training data. This requires reshaping and broadcasting
    # a number of arrays we have.
    per_jet_training_data_dict["runNumber"] = ak.flatten(
        ak.broadcast_arrays(data["runNumber"], jets.pt)[0], axis=1
    )
    per_jet_training_data_dict["eventNumber"] = ak.flatten(
        ak.broadcast_arrays(data["eventNumber"], jets.pt)[0], axis=1
    )
    if mc:
        per_jet_training_data_dict["mcEventWeight"] = ak.flatten(
            ak.broadcast_arrays(data["mcEventWeight"], jets.pt)[0], axis=1
        )

    # Processing clusters, tracks, and msegs
    processed_clusters = processing_func(clusters, "cluster", ak.flatten(jets, axis=1))
    processed_tracks = processing_func(nearby_tracks, "track", ak.flatten(jets, axis=1))
    processed_msegs = processing_func(nearby_msegs, "mseg", ak.flatten(jets, axis=1))

    # The top level jet information.
    per_jet_training_data_dict["pt"] = processing_func(jets, "jet")["pt"]
    per_jet_training_data_dict["eta"] = ak.flatten(jets.eta, axis=1)
    per_jet_training_data_dict["phi"] = ak.flatten(jets.phi, axis=1)

    # Tracks, clusters, and muon segments.
    # per_jet_training_data_dict["tracks"] = ak.flatten(nearby_tracks, axis=1)
    per_jet_training_data_dict["tracks"] = ak.zip(
        {
            "eta": processed_tracks["eta"],
            "phi": processed_tracks["phi"],
            "pt": processed_tracks["pt"],
            "vertex_nParticles": ak.flatten(nearby_tracks["vertex_nParticles"], axis=1),
            "d0": ak.flatten(nearby_tracks["d0"], axis=1),
            "z0": processed_tracks["z0"],
            "chiSquared": ak.flatten(nearby_tracks["chiSquared"], axis=1),
            "PixelShared": ak.flatten(nearby_tracks["PixelShared"], axis=1),
            "SCTShared": ak.flatten(nearby_tracks["SCTShared"], axis=1),
            "PixelHits": ak.flatten(nearby_tracks["PixelHits"], axis=1),
            "SCTHits": ak.flatten(nearby_tracks["SCTHits"], axis=1),
        }
    )
    # per_jet_training_data_dict["clusters"] = ak.flatten(clusters, axis=1)
    per_jet_training_data_dict["clusters"] = ak.zip(
        {
            "eta": processed_clusters["eta"],
            "phi": ak.flatten(clusters["phi"], axis=1),
            "pt": processed_clusters["pt"],
            "l1hcal": processed_clusters["l1hcal"],
            "l2hcal": processed_clusters["l2hcal"],
            "l3hcal": processed_clusters["l3hcal"],
            "l4hcal": processed_clusters["l4hcal"],
            "l1ecal": processed_clusters["l1ecal"],
            "l2ecal": processed_clusters["l2ecal"],
            "l3ecal": processed_clusters["l3ecal"],
            "l4ecal": processed_clusters["l4ecal"],
            "time": ak.flatten(clusters["time"], axis=1),
        }
    )
    # per_jet_training_data_dict["msegs"] = ak.flatten(
    per_jet_training_data_dict["msegs"] = ak.zip(
        {
            "etaPos": processed_msegs["etaPos"],
            "phiPos": processed_msegs["phiPos"],
            "etaDir": ak.flatten(nearby_msegs.p.eta, axis=1),
            "phiDir": processed_msegs["phiDir"],
            "t0": ak.flatten(nearby_msegs.x.t0, axis=1),
            "chiSquared": ak.flatten(nearby_msegs.x.chiSquared, axis=1),
        }
    )
    # )

    # And LLP's if we are doing MC.
    if mc:
        per_jet_training_data_dict["llp"] = ak.flatten(llp_match_jet, axis=1)

    # Finally, build the data we will write out!
    training_data = ak.zip(
        per_jet_training_data_dict, with_name="Momentum3D", depth_limit=1
    )

    return training_data  # type: ignore


def fetch_training_data_to_file(ds_name: str, config: RunConfig):
    result_list = fetch_training_data(ds_name, config)

    # Finally, write it out into a training file.
    ak.to_parquet(
        result_list, config.output_path, compression="ZSTD", compression_level=-7
    )


def fetch_training_data(ds_name, config: RunConfig):
    raw_data = fetch_raw_training_data(ds_name, config)
    result_list = convert_to_training_data(raw_data, mc=config.mc)
    return result_list


def run_query(
    ds_name: str,
    query: ObjectStream,
    config: RunConfig = RunConfig(ignore_cache=False, run_locally=False),
) -> Dict[str, ak.Array]:
    # Build the ServiceX spec and run it.
    from .sx_utils import build_sx_spec

    spec, backend_name, adaptor = build_sx_spec(
        query, ds_name, config.run_locally, config.sx_backend
    )
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
