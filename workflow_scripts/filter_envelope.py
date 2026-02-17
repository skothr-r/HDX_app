#!/usr/bin/env python3
"""
Step 7: Filter by isotopic envelope (M0, M+1, M+2 present; M0 > M1 or M1 > M2).

Runs after chromatogram extraction (Step 6). Keeps only rows where:
  (1) M0, M+1, and M+2 are present in summed MS1 (has_required_isotopes)
  (2) Envelope passes: M+0 > M+1 OR M+1 > M+2 (envelope_ok / m0_gt_m1_gt_m2)

Requires envelope_ok or m0_gt_m1_gt_m2 column in the input CSV (written by Step 6).

Output: comet_frags_perc_openMS_prefilter_extraction_envelope.csv

When chromatogram_metrics_all.csv and chromatogram_traces.npz exist, generates
MS1 chromatogram plots for envelope-passed peptides (same style as extraction).

Usage:
  python filter_envelope.py
  python filter_envelope.py --input extraction.csv --output-dir results
"""

import argparse
import os
import subprocess
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

DEFAULT_INPUT = os.path.join(_PROJECT_ROOT, 'data', 'comet_frags_perc_openMS_prefilter_extraction.csv')
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
        description='Step 7: Filter by isotopic envelope (M0, M+1, M+2 present; M0>M+1 OR M+1>M+2).'
    )
    ap.add_argument('--input', '-i', default=DEFAULT_INPUT,
                    help=f'Input CSV from Step 6 (default: {os.path.basename(DEFAULT_INPUT)})')
    ap.add_argument('--output', '-o', default=None, help='Output CSV path')
    ap.add_argument('--output-dir', default=None, help='Output directory')
    ap.add_argument('--no-plots', action='store_true', help='Skip MS1 chromatogram plot generation')
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

    # Filter: keep rows where m0_gt_m1_gt_m2 (or envelope_ok) is True AND has_required_isotopes (M0, M+1, M+2) is True
    has_req_iso = 'has_required_isotopes' in df.columns

    def _passes_envelope(row):
        # Require M0, M+1, M+2 present when column exists
        if has_req_iso:
            req = row.get('has_required_isotopes')
            if pd.isna(req) or not _to_bool(req):
                return False
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

    # Generate MS1 chromatogram plots for envelope-passed peptides (same style as extraction)
    if not args.no_plots and n_after > 0:
        dataframes_dir = os.path.join(out_dir, 'dataframes')
        metrics_csv = os.path.join(dataframes_dir, 'chromatogram_metrics_all.csv')
        traces_path = os.path.join(dataframes_dir, 'chromatogram_traces.npz')
        if os.path.exists(metrics_csv) and os.path.exists(traces_path):
            envelope_plots_dir = os.path.join(out_dir, 'envelope')
            envelope_rejected_dir = os.path.join(out_dir, 'envelope_rejected')
            os.makedirs(envelope_plots_dir, exist_ok=True)
            os.makedirs(envelope_rejected_dir, exist_ok=True)
            plot_script = os.path.join(_SCRIPT_DIR, 'plot_chromatograms_from_extraction.py')
            cmd = [sys.executable, plot_script,
                   '--chromatogram-metrics-csv', metrics_csv,
                   '--chromatogram-traces', traces_path,
                   '--output-dir', out_dir,
                   '--filter-csv', out_csv,
                   '--output-accepted-dir', envelope_plots_dir,
                   '--output-rejected-dir', envelope_rejected_dir]
            try:
                subprocess.run(cmd, check=True, cwd=_PROJECT_ROOT)
                print(f"[Step 7] MS1 chromatogram plots saved to: {os.path.abspath(envelope_plots_dir)}")
            except subprocess.CalledProcessError as e:
                print(f"[Step 7] Warning: Plot generation failed: {e}", file=sys.stderr)
        else:
            print(f"[Step 7] Skipping plots (chromatogram_metrics_all.csv or chromatogram_traces.npz not found)")


if __name__ == '__main__':
    main()
