import ast
from typing import Tuple, TypeVar

from func_adl import ObjectStream, func_adl_callable
from func_adl_servicex_xaodr25.elementlink_datavector_xaod_iparticle__ import (
    ElementLink_DataVector_xAOD_IParticle__,
)
from func_adl_servicex_xaodr25.xAOD.calocluster_v1 import CaloCluster_v1
from func_adl_servicex_xaodr25.xAOD.trackparticle_v1 import TrackParticle_v1

# from func_adl_servicex_xaodr25.xaod import xAOD


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
        Tuple[ObjectStream[T], ast.Call]: Return the updated stream with the metadata code.
    """
    new_s = s.MetaData(
        {
            "metadata_type": "add_cpp_function",
            "name": "track_summary_value",
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


@func_adl_callable(track_summary_value_callback)
def track_summary_value(trk: TrackParticle_v1, value_selector: int) -> int:
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


def cvt_to_calo_cluster_callback(
    s: ObjectStream[T], a: ast.Call
) -> Tuple[ObjectStream[T], ast.Call]:
    """Use dynamic cast to convert

    Args:
        s (ObjectStream[T]): The stream we are operating against
        a (ast.Call): The actual call

    Returns:
        Tuple[ObjectStream[T], ast.Call]: Return the updated stream with the metadata code.
    """
    new_s = s.MetaData(
        {
            "metadata_type": "add_cpp_function",
            "name": "cvt_to_calo_cluster",
            "code": [
                "auto rawObj = cluster->rawConstituent();\n",
                "const xAOD::CaloCluster *r = dynamic_cast<const xAOD::CaloCluster*>(rawObj);\n",
                "const xAOD::CaloCluster result(*r);\n",
            ],
            "result": "result",
            "include_files": [],
            "arguments": ["cluster"],
            "return_type": "xAOD::CaloCluster_v1",
        }
    )
    return new_s, a


@func_adl_callable(cvt_to_calo_cluster_callback)
def cvt_to_calo_cluster(
    cluster: ElementLink_DataVector_xAOD_IParticle__,
) -> CaloCluster_v1:
    """
    Converts a given cluster to a CaloCluster_v1 object.

    Args:
        cluster (ElementLink_DataVector_xAOD_IParticle__): The input cluster to be converted.

    Returns:
        CaloCluster_v1: The converted CaloCluster_v1 object.
    """
    ...


def e_sample_callback(
    s: ObjectStream[T], a: ast.Call
) -> Tuple[ObjectStream[T], ast.Call]:
    """Because `func_adl` is doing a little too much, do the conversion needed.
    Args:
        s (ObjectStream[T]): The stream we are operating against
        a (ast.Call): The actual call

    Returns:
        Tuple[ObjectStream[T], ast.Call]: Return the updated stream with the metadata code.
    """
    new_s = s.MetaData(
        {
            "metadata_type": "add_cpp_function",
            "name": "e_sample",
            "code": [
                "xAOD::CaloCluster_v1::CaloSample st "
                "(static_cast<xAOD::CaloCluster_v1::CaloSample>(value_selector));\n"
                "double result = clus.eSample(st);\n"
            ],
            "result": "result",
            "include_files": ["xAODCaloEvent/versions/CaloCluster_v1.h"],
            "arguments": ["clus", "value_selector"],
            "return_type": "double",
        }
    )
    return new_s, a


@func_adl_callable(e_sample_callback)
def e_sample(clus: CaloCluster_v1, value_selector: int) -> float:
    """Call the `eSummary` method on a CaloCluster

    * Converts the enum integer into a C++ enum and makes the call.

    Args:
        clus (CaloCluster_v1): The cluster
        value_selector (int): Which value

    NOTE: This is a dummy function that injects C++ into the object stream to do the
    actual work.

    Returns:
        int: Value requested.
    """
    ...
