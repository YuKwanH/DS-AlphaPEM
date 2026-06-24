"""Backwards-compatibility shim.

``PEMFC_dyn`` now lives in :mod:`model.dualscale` alongside the other two
dual-scale variants (``PEMFC`` and ``PEMFC_0D``). Notebooks and scripts
that still do ``from model.dynamic import PEMFC_dyn`` keep working via
this re-export.
"""

from model.dualscale import PEMFC_dyn  # noqa: F401

__all__ = ["PEMFC_dyn"]
