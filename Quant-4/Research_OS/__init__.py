"""Evidence-governed research plane for the existing Quant-Ultra kernel.

Research OS is intentionally additive: importing it never starts the kernel and
never mutates production configuration.
"""

from .contracts.common import AdmissionAction, LifecycleStatus, LockState

__all__ = ["AdmissionAction", "LifecycleStatus", "LockState"]
__version__ = "1.0.0"
