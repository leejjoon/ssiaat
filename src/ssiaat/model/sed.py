"""Backward-compatibility facade for the line/continuum model library.

The profile-shape primitives now live in :mod:`ssiaat.model.shapes` and the line
inventory in :mod:`ssiaat.model.lines`. This module re-exports the shapes
unchanged and keeps the old hand-coded line getters (``get_br_a``, ``get_pa_b``,
``get_pah``, ``get_h2o_ice``) as thin, **deprecated** wrappers over the registry
so existing imports (``from ssiaat.model.sed import get_br_a``) keep working.

Prefer, in new code::

    from ssiaat.model.shapes import hflatline, get_cont_model   # shapes
    from ssiaat.model.lines import get_line, get_line_model      # lines
"""
import warnings

# Shape primitives + continuum helpers (re-exported; not deprecated).
from .shapes import (  # noqa: F401
    tanh_step,
    const,
    cont_left,
    cont_right,
    hflattop,
    hflatline,
    get_conts,
    get_cont_model,
)


def _deprecated(old, new):
    warnings.warn(
        f"ssiaat.model.sed.{old} is deprecated; use {new} instead.",
        DeprecationWarning, stacklevel=3)


def get_h2o_ice():
    """Deprecated. Use ``ssiaat.model.lines.get_line('h2o_ice')``."""
    _deprecated("get_h2o_ice()", "ssiaat.model.lines.get_line('h2o_ice')")
    from .lines import get_line
    return get_line("h2o_ice")


def get_pah():
    """Deprecated. Returns ``(pah_narrow, pah_broad)`` from the registry."""
    _deprecated("get_pah()",
                "ssiaat.model.lines.get_line('pah_narrow'/'pah_broad')")
    from .lines import get_line
    return get_line("pah_narrow"), get_line("pah_broad")


def get_br_a():
    "band = 5. Deprecated. Use ``ssiaat.model.lines.get_line('br_a')``."
    _deprecated("get_br_a()", "ssiaat.model.lines.get_line('br_a')")
    from .lines import get_line
    return get_line("br_a")


def get_pa_b():
    "band = 2. Deprecated. Use ``ssiaat.model.lines.get_line('pa_b')``."
    _deprecated("get_pa_b()", "ssiaat.model.lines.get_line('pa_b')")
    from .lines import get_line
    return get_line("pa_b")
