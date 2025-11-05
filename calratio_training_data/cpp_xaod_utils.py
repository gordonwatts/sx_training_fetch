import ast
from typing import Tuple, TypeVar

from func_adl import ObjectStream, func_adl_callable
from func_adl_servicex_xaodr25.elementlink_datavector_xaod_iparticle__ import (
    ElementLink_DataVector_xAOD_IParticle__,
)
from func_adl_servicex_xaodr25.xaod import add_enum_info, xAOD
from func_adl_servicex_xaodr25.xAOD.calocluster_v1 import CaloCluster_v1
from func_adl_servicex_xaodr25.xAOD.jet_v1 import Jet_v1
from func_adl_servicex_xaodr25.xAOD.trackparticle_v1 import TrackParticle_v1
from func_adl_servicex_xaodr25.xAOD.truthparticle_v1 import TruthParticle_v1

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
            "return_type": "int",
        }
    )
    new_s = add_enum_info(new_s, "SummaryType")
    return new_s, a


@func_adl_callable(track_summary_value_callback)
def track_summary_value(trk: TrackParticle_v1, value_selector: xAOD.SummaryType) -> int:
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


def cvt_to_raw_calocluster_callback(
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
            "name": "cvt_to_raw_calocluster",
            "code": [
                "// Very ugly",
                "const xAOD::CaloCluster* clus = dynamic_cast<const xAOD::CaloCluster*>(*link)",
                "const SG::AuxElement::ConstAccessor< ElementLink<xAOD::IParticleContainer> > "
                'originalObject("originalObjectLink")',
                "const xAOD::CaloCluster* result = dynamic_cast<const xAOD::CaloCluster*> "
                "(*originalObject(*clus))",
            ],
            "result": "result",
            "include_files": ["xAODCaloEvent/CaloCluster.h"],
            "arguments": ["link"],
            "return_type": "const xAOD::CaloCluster_v1*",
        }
    )

    return new_s, a


@func_adl_callable(cvt_to_raw_calocluster_callback)
def cvt_to_raw_calocluster(
    link: ElementLink_DataVector_xAOD_IParticle__,
) -> CaloCluster_v1:
    """
    Converts a given link to a cluster to a CaloCluster_v1 object.

    Args:
        link (ElementLink_DataVector_xAOD_IParticle__): The input cluster link to be converted.

    Returns:
        CaloCluster_v1: The converted CaloCluster_v1 object.
    """
    ...


def add_jet_selection_tool(
    stream: ObjectStream[T], tool_name: str, cut_name: str
) -> ObjectStream[T]:
    """
    Adds a JetCleaningTool to the given ObjectStream with specified properties.
    This function modifies the metadata of the provided ObjectStream to include
    a JetCleaningTool instance. The tool is initialized with the given name and
    configured with the specified cut level. Declared at a global level.

    Note:
        To access use the following code:

        {tool_name}->keep(*jet);

    Args:
        stream (ObjectStream[T]): The object stream to which the JetCleaningTool
            will be added.
        tool_name (str): The name of the JetCleaningTool instance.
        cut_name (str): The cut level to be set for the JetCleaningTool.
    Returns:
        ObjectStream[T]: The modified object stream with the added JetCleaningTool.
    """
    return stream.MetaData(
        {
            "metadata_type": "inject_code",
            "name": "jet_tool_{tool_name}",
            "header_includes": ["JetSelectorTools/JetCleaningTool.h"],
            "private_members": [f"IJetSelector *{tool_name};"],
            "instance_initialization": [
                f'{tool_name}(new JetCleaningTool("{tool_name}"))'
            ],
            # TODO: These should be in the initialize command, with an ANA_CHECK.
            "initialize_lines": [
                f'ANA_CHECK(asg::setProperty({tool_name}, "CutLevel", "{cut_name}"));',
                f"ANA_CHECK({tool_name}->initialize());",
            ],
            "link_libraries": ["JetSelectorToolsLib"],
        }
    )


def jet_clean_llp_callback(
    s: ObjectStream[T], a: ast.Call
) -> Tuple[ObjectStream[T], ast.Call]:
    new_s = s.MetaData(
        {
            "metadata_type": "add_cpp_function",
            "name": "jet_clean_llp",
            "code": ["bool result = m_jetCleaning_llp->keep(*jet)"],
            "result": "result",
            "include_files": [],
            "arguments": ["jet"],
            "return_type": "bool",
        }
    )
    return new_s, a


@func_adl_callable(jet_clean_llp_callback)
def jet_clean_llp(jet: Jet_v1) -> bool:
    """Call the jet selection on the jet.

    * return true or false if the jet passes the selection cut.

    Args:
        jet (Jet_v1): The jet we are operating against
        value_selector (int): Which value (pixel holes, etc.)

    NOTE: This is a dummy function that injects C++ into the object stream to do the
    actual work.

    Returns:
        bool: Did the jet pass?
    """
    ...


def follow_radiation_callback(
    s: ObjectStream[T], a: ast.Call
) -> Tuple[ObjectStream[T], ast.Call]:
    new_s = s.MetaData(
        {
            "metadata_type": "add_cpp_function",
            "name": "follow_radiation",
            "code": [
                "auto result = bsm;",
                "while(result->nChildren() > 0) {",
                "  for (int i=0; i < result->nChildren(); ++i) {",
                "    auto child = result->child(i);",
                "    if (child->pdgId() == pdgid) {",
                "      result = child;",
                "      break;",
                "    }",
                "  }",
                "}",
            ],
            "result": "result",
            "include_files": [],
            "arguments": ["bsm", "pdgid"],
            "return_type": "TruthParticle_v1",
        }
    )
    return new_s, a


@func_adl_callable(follow_radiation_callback)
def follow_radiation(bsm: TruthParticle_v1, pdgid: int) -> TruthParticle_v1:
    """Follow the radiation chain of a truth particle to find the final state
    particle with the given pdgid.

    Works by looking at the daughters of the particle, and if one matches the pdgid,
    it continues down that chain until no more matches are found.

    Args:
        bsm (TruthParticle_v1): The initial BSM particle.
        pdgid (int): The PDG ID of the final state particle to follow.

    Returns:
        TruthParticle_v1: The final state particle with the specified PDG ID.
    """
    ...
