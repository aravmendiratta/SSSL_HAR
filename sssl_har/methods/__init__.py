"""
Unified method training wrappers for multi-view and conventional SSL frameworks.
"""

from .ssl_methods import get_ssl_method_trainer, CroSSLMethod, COCOAMethod, SimCLRMethod, CPCMethod

__all__ = [
    "get_ssl_method_trainer",
    "CroSSLMethod",
    "COCOAMethod",
    "SimCLRMethod",
    "CPCMethod",
]
