#!/usr/bin/env python3
"""
Step 7: Filter by isotopic envelope (M0 > M1, M1 > M2).

Runs after chromatogram extraction (Step 6). Keeps only rows where the isotopic
envelope passes: M+0 > M+1 OR M+1 > M+2 (on summed MS1 in the peak window).

Requires envelope_ok or m0_gt_m1_gt_m2 column in the input CSV (written by Step 6).

Output: comet_frags_perc_openMS_confidence_accuracy_extraction_envelope.csv

Usage:
  python filter_envelope.py
  python filter_envelope.py --input extraction.csv --output-dir results
"""

import argparse
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

DEFAULT_INPUT = os.path.join(_SCRIPT_DIR, 'data', 'comet_frags_perc_openMS_confidence_accuracy_extraction.csv')
ENVELOPE_SUFFIX = '_envelope'


def _to_bool(v):
    """Parse bool from various representations."""
    if v is None or (isinstance(v, float) and (v != v)):  # NaN
        return False
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    return s in ('true', '1', 'yes')


def main():
    ap = argparse.ArgumentParser(
        description='Step 7: Filter by isotopic envelope (M0>M1, M1>M2).'
    )
    ap.add_argument('--input', '-i', default=DEFAULT_INPUT,
                    help=f'Input CSV from Step 6 (default: {os.path.basename(DEFAULT_INPUT)})')
    ap.add_argument('--output', '-o', default=None, help='Output CSV path')
    ap.add_argument('--output-dir', default=None, help='Output directory')
    args = ap.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input CSV not found: {args.input}")
        sys.exit(1)

    try:
        import pandas as pd
        import numpy as np
    except ImportError:
        print("Error: pandas and numpy required.")
        sys.exit(1)

    input_dir = os.path.dirname(os.path.abspath(args.input))
    base = os.path.splitext(os.path.basename(args.input))[0]
    out_dir = args.output_dir or input_dir
    os.makedirs(out_dir, exist_ok=True)

    if args.output:
        out_csv = args.output
    else:
        if base.endswith('_extraction'):
            out_csv = os.path.join(out_dir, f'{base}{ENVELOPE_SUFFIX}.csv')
        else:
            out_csv = os.path.join(out_dir, f'{base}{ENVELOPE_SUFFIX}.csv')

    try:
        with open(args.input, 'r') as f:
            first = f.readline()
        skip = 1 if 'CometVersion' in first else 0
    except Exception:
        skip = 0
    df = pd.read_csv(args.input, sep=',', skiprows=skip, engine='python', quotechar='"', on_bad_lines='warn')

    # Check for envelope columns (written by Step 6 extraction)
    has_m0 = 'm0_gt_m1_gt_m2' in df.columns
    has_env_ok = 'envelope_ok' in df.columns

    if not has_m0 and not has_env_ok:
        print("Error: Input CSV has no envelope_ok or m0_gt_m1_gt_m2 columns.")
        print("Run chromatogram extraction (Step 6) first to produce the extraction CSV with envelope data.")
        sys.exit(1)

    # Filter: keep rows where m0_gt_m1_gt_m2 is True (or envelope_ok is True if m0 missing)
    def _passes_envelope(row):
        m0 = row.get('m0_gt_m1_gt_m2')
        env = row.get('envelope_ok')
        if has_m0 and pd.notna(m0):
            return _to_bool(m0)
        if has_env_ok and pd.notna(env):
            return _to_bool(env)
        return False

    mask = df.apply(_passes_envelope, axis=1)
    df_out = df[mask].reset_index(drop=True)

    n_before = len(df)
    n_after = len(df_out)
    n_dropped = n_before - n_after

    df_out.to_csv(out_csv, index=False)
    print(f"[Step 7] Envelope filter: {n_before} -> {n_after} rows (dropped {n_dropped})")
    print(f"[Step 7] Output: {os.path.abspath(out_csv)}")


if __name__ == '__main__':
    main()
