#!/usr/bin/env python3
"""
Backward-compatibility stub: visualization logic lives in the visualization package.

This file re-exports from visualization so existing imports still work:
  from combined_overhang_visualization import create_combined_visualization, parse_fasta
Prefer importing from the package:
  from visualization import create_combined_visualization, parse_fasta
"""

import sys
import os

_root = os.path.dirname(os.path.abspath(__file__))
if _root not in sys.path:
    sys.path.insert(0, _root)

from visualization.combined import *  # noqa: F401, F403
from visualization.multi_aa import (
    parse_fasta,
    parse_multi_aa_overhangs,
    parse_single_aa_overhangs,
    normalize_fragment_pair,
    get_filtered_peptide_indices,
)
