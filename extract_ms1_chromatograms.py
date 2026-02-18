#!/usr/bin/env python3
"""
Backward-compatibility stub: peak-window (chromatogram) production lives in the visualization package.

Run this file as a script to run chromatogram extraction (same as before):
  python extract_ms1_chromatograms.py <comet_csv> <raw_file> [output_png] [options]
Prefer calling the module:
  python -m visualization.chromatograms <comet_csv> <raw_file> [output_png] [options]
"""

import sys
import os

_root = os.path.dirname(os.path.abspath(__file__))
if _root not in sys.path:
    sys.path.insert(0, _root)

from visualization.chromatograms import run_chromatograms, _parse_chromatogram_args

if __name__ == '__main__':
    run_chromatograms(_parse_chromatogram_args())
