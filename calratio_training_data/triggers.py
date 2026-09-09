from func_adl import ObjectStream
from func_adl_servicex_xaodr25 import tdt_chain_fired, tmt_match_object
from func_adl_servicex_xaodr25.event_collection import Event

from calratio_training_data.constants import BIB_TRIGGERS
from func_adl_servicex_xaodr25.xAOD.jet_v1 import Jet_v1


def trigger_bib_filter(
    query: ObjectStream[Event],
) -> ObjectStream[Event]:
    """Look for events that will have a BIB jet that we can use for training

    The `BIB_TRIGGERS` constant defines a list of trigger pairs.
        * The first trigger in each pair is an inclusive LLP trigger that will fire if
          there is anything signal like in the calorimeter.
        * The second trigger in each pair is a BIB removal trigger that will not fire
          if there is BIB activity in the event.

    The result is that we only keep events where we are likely to have some sort of BIB
    activity in the calorimeter.

    Args:
        query (ObjectStream[Event]): The event-level query

    Returns:
        ObjectStream[Event]: The event-level query with the proper set of triggers
        checked.
    """
    query = query.Where(
        lambda e: any(
            tdt_chain_fired(incl_trig) and not tdt_chain_fired(bib_trig)
            for incl_trig, bib_trig in BIB_TRIGGERS
        )
    )
    return query


def is_trigger_jet(jet: Jet_v1) -> bool:
    """For use in a query - true if the jet matched one of the triggers.
    Matches with a delta R of 0.2 or less.

    Args:
        jet (Jet_v1): Jet to check

    Returns:
        bool: Inside a query, evaluates to true if the jet matches one of the triggers
              in `BIB_TRIGGERS`
    """
    return any(tmt_match_object(trig, jet, 0.2) for trig, _ in BIB_TRIGGERS)
