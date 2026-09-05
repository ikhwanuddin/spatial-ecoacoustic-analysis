#!/usr/bin/env python3
"""
Backward-compatible wrapper for spatial_clustering.py.

Legacy scripts referencing `cluster_poc.py` will import functions from
`spatial_clustering.py` seamlessly.
"""

from __future__ import annotations

import warnings
from spatial_clustering import *

# Emit informative note when imported directly
# warnings.warn(
#     "cluster_poc is deprecated; please import from spatial_clustering instead.",
#     DeprecationWarning,
#     stacklevel=2,
# )

if __name__ == "__main__":
    import spatial_clustering
    # Re-route to main if needed
    print("spatial_clustering wrapper active.")
