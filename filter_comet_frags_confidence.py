#!/usr/bin/env python3
"""
Step 4: Filter Comet fragments by confidence (Q-value and PEP).

Keeps rows where Q-value ≤ 0.05 OR PEP ≤ 0.05.
Creates diagnostics/ directory and a scatter plot of PEP vs Q-value with thresholds.
Output: comet_frags_perc_openMS_confidence.csv (all columns; rows not passing are removed).

Usage:
    python filter_comet_frags_confidence.py
    python filter_comet_frags_confidence.py --input data/comet_frags_perc_openMS.csv --output-dir data
"""

import argparse
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(_SCRIPT_DIR, 'data', 'comet_frags_perc_openMS.csv')

QVALUE_COLUMNS = ['percolator_qvalue', 'q-value', 'qvalue', 'percolator_q-value', 'FDR', 'fdr']
PEP_COLUMNS = ['percolator_PEP', 'PEP', 'pep']
THRESHOLD = 0.05


def main():
    ap = argparse.ArgumentParser(description='Filter by Q-value and PEP confidence')
    ap.add_argument('--input', '-i', default=DEFAULT_CSV,
                    help=f'Input CSV from Step 3 (default: {os.path.basename(DEFAULT_CSV)})')
    ap.add_argument('--output', '-o', default=None,
                    help='Output CSV path (default: same dir as input, suffix _confidence)')
    ap.add_argument('--output-dir', default=None,
                    help='Output directory for CSV and diagnostics (default: same as input dir)')
    ap.add_argument('--threshold', type=float, default=THRESHOLD,
                    help=f'PEP and Q-value threshold (default: {THRESHOLD})')
    args = ap.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    try:
        import pandas as pd
        import numpy as np
    except ImportError:
        print("Error: pandas and numpy required. pip install pandas numpy")
        sys.exit(1)

    # Read CSV
    try:
        with open(args.input, 'r') as f:
            first = f.readline()
        skip = 1 if 'CometVersion' in first else 0
    except Exception:
        skip = 0

    df = pd.read_csv(args.input, sep=',', skiprows=skip, engine='python', quotechar='"', on_bad_lines='warn')

    # Resolve output paths
    input_dir = os.path.dirname(os.path.abspath(args.input))
    base = os.path.splitext(os.path.basename(args.input))[0]
    out_dir = args.output_dir or input_dir
    os.makedirs(out_dir, exist_ok=True)
    diagnostics_dir = os.path.join(out_dir, 'diagnostics')
    os.makedirs(diagnostics_dir, exist_ok=True)

    if args.output:
        out_csv = args.output
    else:
        # Append _confidence if not already present
        if base.endswith('_confidence'):
            out_csv = os.path.join(out_dir, base + '.csv')
        else:
            out_csv = os.path.join(out_dir, base + '_confidence.csv')

    # Find Q-value and PEP columns
    qcol = None
    for c in QVALUE_COLUMNS:
        if c in df.columns:
            qcol = c
            break
    pepcol = None
    for c in PEP_COLUMNS:
        if c in df.columns:
            pepcol = c
            break

    def _pass(row):
        q_ok = True
        if qcol and qcol in df.columns and pd.notna(row.get(qcol)):
            try:
                q_ok = float(row[qcol]) <= args.threshold
            except (TypeError, ValueError):
                q_ok = True
        else:
            q_ok = True
        pep_ok = True
        if pepcol and pepcol in df.columns and pd.notna(row.get(pepcol)):
            try:
                pep_ok = float(row[pepcol]) <= args.threshold
            except (TypeError, ValueError):
                pep_ok = True
        else:
            pep_ok = True
        return q_ok or pep_ok

    if qcol is None and pepcol is None:
        print("Warning: No Q-value or PEP column found. Keeping all rows.")
        df_filtered = df.copy()
    else:
        mask = df.apply(_pass, axis=1)
        df_filtered = df.loc[mask].copy()

    n_total = len(df)
    n_pass = len(df_filtered)

    # Diagnostic scatter: PEP vs Q-value
    if qcol and pepcol and qcol in df.columns and pepcol in df.columns and n_total > 0:
        try:
            import matplotlib.pyplot as plt
            q_vals = pd.to_numeric(df[qcol], errors='coerce')
            pep_vals = pd.to_numeric(df[pepcol], errors='coerce')
            valid = q_vals.notna() & pep_vals.notna()
            q_vals = q_vals[valid]
            pep_vals = pep_vals[valid]
            passed = df.loc[valid].apply(_pass, axis=1).values if qcol and pepcol else np.ones(valid.sum(), dtype=bool)

            fig, ax = plt.subplots(figsize=(8, 8))
            fig.patch.set_facecolor('black')
            ax.set_facecolor('black')
            # Clamp for visualization (avoid log 0)
            q_plot = np.clip(q_vals, 1e-10, 1.0)
            pep_plot = np.clip(pep_vals, 1e-10, 1.0)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.scatter(q_plot[~passed], pep_plot[~passed], c='#CCCCCC', s=12, alpha=0.7, edgecolors='0.6', linewidths=0.3, label=f'Rejected ({n_total - n_pass})')
            ax.scatter(q_plot[passed], pep_plot[passed], c='#2E86AB', s=12, alpha=0.6, edgecolors='0.6', linewidths=0.3, label=f'Passing ({n_pass})')
            ax.axhline(args.threshold, color='red', linestyle='--', linewidth=2, label=f'PEP = {args.threshold}')
            ax.axvline(args.threshold, color='orange', linestyle='--', linewidth=2, label=f'Q-value = {args.threshold}')
            ax.set_xlabel(f'Q-value ({qcol})', fontsize=12, color='0.85')
            ax.set_ylabel(f'PEP ({pepcol})', fontsize=12, color='0.85')
            ax.set_title(f'Confidence filter: {n_pass} / {n_total} rows pass (Q or PEP ≤ {args.threshold})', fontsize=14, color='0.9')
            ax.tick_params(colors='0.85')
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_color('0.85')
            for spine in ax.spines.values():
                spine.set_color('0.6')
            ax.legend(loc='upper right', fontsize=10, facecolor='#1a1a1a', edgecolor='0.6', labelcolor='0.9')
            ax.grid(True, alpha=0.25)
            diag_path = os.path.join(diagnostics_dir, 'confidence_filter_PEP_vs_Qvalue_scatter.png')
            fig.savefig(diag_path, dpi=150, bbox_inches='tight', facecolor='black')
            plt.close(fig)
            print(f"Diagnostic scatter saved: {diag_path}")
        except Exception as e:
            print(f"Warning: Could not create diagnostic scatter: {e}")

    df_filtered.to_csv(out_csv, index=False)
    print(f"Output: {out_csv} ({n_pass} rows, {n_total - n_pass} removed)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
