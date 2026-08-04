# -*- coding: utf-8 -*-
"""
Quant-Ultra Flow - Phase_8 Submodule Initializer
"""
def execute(*args, **kwargs):
    """Load the heavy statistical stack only when Phase 8 is executed."""
    from .step8_audit_stress_test import execute as _execute
    return _execute(*args, **kwargs)

__all__ = ["execute"]
