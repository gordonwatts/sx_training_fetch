import ast

from func_adl_servicex_xaodr25 import FuncADLQueryPHYS

from calratio_training_data.constants import (
    BIB_TRIGGERS,
    CR_DIJET_TRIGGER,
    CR_TTBAR_TRIGGER,
)
from calratio_training_data.triggers import (
    trigger_bib_filter,
    trigger_cr_dijet_filter,
    trigger_cr_ttbar_filter,
)


def _query_source(filter_fn) -> str:
    """Apply a trigger filter to an empty query and return the generated source."""
    return ast.unparse(filter_fn(FuncADLQueryPHYS()).query_ast)


def test_trigger_cr_dijet_filter_requires_dijet_trigger():
    """The dijet CR filter should check every trigger in CR_DIJET_TRIGGER."""
    src = _query_source(trigger_cr_dijet_filter)

    for trig in CR_DIJET_TRIGGER:
        assert trig in src


def test_trigger_cr_ttbar_filter_requires_ttbar_triggers():
    """The ttbar CR filter should check every trigger in CR_TTBAR_TRIGGER."""
    src = _query_source(trigger_cr_ttbar_filter)

    for pair in CR_TTBAR_TRIGGER:
        for trig in pair:
            assert trig in src


def test_trigger_cr_filters_do_not_overlap():
    """The two control regions must stay on their own triggers.

    They were merged from branches that both used the name `cr`, so this guards
    against one silently picking up the other's trigger list again.
    """
    dijet_src = _query_source(trigger_cr_dijet_filter)
    ttbar_src = _query_source(trigger_cr_ttbar_filter)

    for pair in CR_TTBAR_TRIGGER:
        for trig in pair:
            assert trig not in dijet_src

    for trig in CR_DIJET_TRIGGER:
        assert trig not in ttbar_src


def test_trigger_bib_filter_uses_bib_triggers():
    """BIB filtering should be unaffected by the control region work."""
    src = _query_source(trigger_bib_filter)

    for incl_trig, bib_trig in BIB_TRIGGERS:
        assert incl_trig in src
        assert bib_trig in src

    for trig in CR_DIJET_TRIGGER:
        assert trig not in src
