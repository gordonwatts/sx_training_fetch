from func_adl import ObjectStream
from func_adl_servicex_xaodr25 import tdt_chain_fired
from func_adl_servicex_xaodr25.event_collection import Event

from calratio_training_data.constants import BIB_TRIGGERS


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
        query (ObjectStream[FuncADLQueryPHYS]): The event-level query

    Returns:
        ObjectStream[FuncADLQueryPHYS]: The event-level query with the proper set of triggers
        checked.
    """
    for incl_trig, bib_trig in BIB_TRIGGERS:
        query = query.Where(
            lambda e: tdt_chain_fired(incl_trig) and not tdt_chain_fired(bib_trig)
        )
    return query
