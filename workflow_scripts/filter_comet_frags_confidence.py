#!/usr/bin/env python3
"""
Step 4: Filter Comet fragments by confidence (Q-value and PEP).

Keeps rows where Q-value ≤ 0.05 OR PEP ≤ 0.05.
Removes peptides where prev_aa or next_aa is proline (P).
Output column order: sequence_positions, plain_peptide, charge, theoretical_mz, MS1_RT_minutes, qvalue, pep, sp, then fragment columns.
Creates diagnostics/ directory and a scatter plot of PEP vs Q-value with thresholds.
Output: comet_frags_perc_openMS_prefilter.csv

Usage:
    python filter_comet_frags_confidence.py
    python filter_comet_frags_confidence.py --input data/comet_frags_perc_openMS.csv --output-dir data
"""

import argparse
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
DEFAULT_CSV = os.path.join(_PROJECT_ROOT, 'data', 'comet_frags_perc_openMS.csv')

QVALUE_COLUMNS = ['percolator_qvalue', 'q-value', 'qvalue', 'percolator_q-value', 'FDR', 'fdr']
PEP_COLUMNS = ['percolator_PEP', 'PEP', 'pep']
SP_COLUMNS = ['sp_score', 'sp', 'Sp']
THRESHOLD = 0.05

# Output column order: lead columns, then fragment columns, then rest
LEAD_COLS = ['sequence_positions', 'plain_peptide', 'charge', 'theoretical_mz', 'MS1_RT_minutes', 'qvalue', 'pep', 'sp']
FRAGMENT_COLS = ['comet_matched_frags', 'comet_matched_frags_mz', 'matched fragment ion intensities',
                 'single_aa_overhang_fragment_pairs', 'single_aa_overhangs_protein_positions', 'matched fragment ion quality scores']


def main():
    ap = argparse.ArgumentParser(description='Filter by Q-value and PEP confidence')
    ap.add_argument('--input', '-i', default=DEFAULT_CSV,
                    help=f'Input CSV from Step 3 (default: {os.path.basename(DEFAULT_CSV)})')
    ap.add_argument('--output', '-o', default=None,
                    help='Output CSV path (default: same dir as input, suffix _prefilter)')
    ap.add_argument('--output-dir', default=None,
                    help='Output directory for CSV and diagnostics (default: same as input dir)')
    ap.add_argument('--threshold', type=float, default=None,
                    help=f'[Deprecated] Use --q-threshold and --pep-threshold. If set, applies to both.')
    ap.add_argument('--q-threshold', type=float, default=THRESHOLD,
                    help=f'Q-value threshold (default: {THRESHOLD})')
    ap.add_argument('--pep-threshold', type=float, default=THRESHOLD,
                    help=f'PEP threshold (default: {THRESHOLD})')
    ap.add_argument('--allow-carbamidomethyl', action='store_true',
                    help='Allow peptides with only Carbamidomethyl (C, 57.0215). Default: reject all modifications.')
    args = ap.parse_args()
    if args.threshold is not None:
        args.q_threshold = args.threshold
        args.pep_threshold = args.threshold

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
        # Append _prefilter if not already present
        if base.endswith('_prefilter'):
            out_csv = os.path.join(out_dir, base + '.csv')
        else:
            out_csv = os.path.join(out_dir, base + '_prefilter.csv')

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
                q_ok = float(row[qcol]) <= args.q_threshold
            except (TypeError, ValueError):
                q_ok = True
        else:
            q_ok = True
        pep_ok = True
        if pepcol and pepcol in df.columns and pd.notna(row.get(pepcol)):
            try:
                pep_ok = float(row[pepcol]) <= args.pep_threshold
            except (TypeError, ValueError):
                pep_ok = True
        else:
            pep_ok = True
        return q_ok or pep_ok

    n_before_qpep = len(df)
    if qcol is None and pepcol is None:
        print("Warning: No Q-value or PEP column found. Keeping all rows.")
        df_filtered = df.copy()
    else:
        mask = df.apply(_pass, axis=1)
        df_filtered = df.loc[mask].copy()
    n_qpep_removed = n_before_qpep - len(df_filtered)
    if n_qpep_removed > 0:
        print(f"Removed {n_qpep_removed} rows (kept rows with Q-value ≤ {args.q_threshold} or PEP ≤ {args.pep_threshold})")

    # Remove peptides with modifications. Default: reject all (unmodified only).
    # With --allow-carbamidomethyl: allow only Carbamidomethyl (C, 57.0215).
    n_before_mods = len(df_filtered)
    ALLOWED_MOD_MASSES = frozenset(['57.0215', '57.021500', '57.02146', '57.021464'])

    def _has_modifications_to_reject(m):
        if m is None or (isinstance(m, float) and pd.isna(m)):
            return False
        s = str(m).strip()
        if s.lower() in ('', '-', 'nan', 'none'):
            return False
        if args.allow_carbamidomethyl:
            # Reject only if any mass is not Carbamidomethyl
            masses = []
            for part in s.split(','):
                part = part.strip()
                if '_' in part:
                    mass = part.rsplit('_', 1)[-1].strip()
                    if mass and mass.replace('.', '').replace('-', '').isdigit():
                        masses.append(mass.rstrip('0').rstrip('.') if '.' in mass else mass)
            if not masses:
                return True  # Unknown format, reject
            for mass in masses:
                norm = mass.rstrip('0').rstrip('.') if '.' in mass else mass
                if norm not in ALLOWED_MOD_MASSES:
                    return True
            return False
        else:
            # Reject all modifications (unmodified peptides only)
            return True

    if 'modifications' in df_filtered.columns:
        mask_mods = ~df_filtered['modifications'].apply(_has_modifications_to_reject)
        df_filtered = df_filtered.loc[mask_mods].copy()
        n_mods_removed = n_before_mods - len(df_filtered)
        if n_mods_removed > 0:
            mod_desc = 'modifications (unmodified only)' if not args.allow_carbamidomethyl else 'variable modifications'
            print(f"Removed {n_mods_removed} rows ({mod_desc})")

    # Remove peptides where prev_aa or next_aa is proline (P), or peptide starts/ends with proline
    n_before_pro = len(df_filtered)
    def _is_proline_reject(row):
        # Flanking prolines (prev_aa or next_aa = P)
        if 'prev_aa' in df_filtered.columns and 'next_aa' in df_filtered.columns:
            prev = str(row.get('prev_aa', '')).strip().upper()
            next_ = str(row.get('next_aa', '')).strip().upper()
            if prev == 'P' or next_ == 'P':
                return True
        # Peptide termini (starts or ends with P)
        seq = str(row.get('plain_peptide', '')).strip() if 'plain_peptide' in df_filtered.columns else ''
        if seq and (seq[0].upper() == 'P' or seq[-1].upper() == 'P'):
            return True
        return False
    mask_pro = ~df_filtered.apply(_is_proline_reject, axis=1)
    df_filtered = df_filtered.loc[mask_pro].copy()
    n_pro_removed = n_before_pro - len(df_filtered)
    if n_pro_removed > 0:
        print(f"Removed {n_pro_removed} rows (proline: flanking or peptide termini)")

    n_total = len(df)
    n_pass = len(df_filtered)

    # Prepare output with standardized column names and order
    out = df_filtered.copy()
    # Rename for output
    if 'mz' in out.columns and 'theoretical_mz' not in out.columns:
        out = out.rename(columns={'mz': 'theoretical_mz'})
    if 'MS1_retention_time_min' in out.columns and 'MS1_RT_minutes' not in out.columns:
        out = out.rename(columns={'MS1_retention_time_min': 'MS1_RT_minutes'})
    qcol_out = qcol if qcol and qcol in out.columns else None
    if qcol_out and qcol_out != 'qvalue':
        out = out.rename(columns={qcol_out: 'qvalue'})
    pepcol_out = pepcol if pepcol and pepcol in out.columns else None
    if pepcol_out and pepcol_out != 'pep':
        out = out.rename(columns={pepcol_out: 'pep'})
    spcol = next((c for c in SP_COLUMNS if c in out.columns), None)
    if spcol and spcol != 'sp':
        out = out.rename(columns={spcol: 'sp'})
    if 'matched fragment ions' in out.columns and 'comet_matched_frags' not in out.columns:
        out = out.rename(columns={'matched fragment ions': 'comet_matched_frags'})
    if 'matched fragment ion mz' in out.columns and 'comet_matched_frags_mz' not in out.columns:
        out = out.rename(columns={'matched fragment ion mz': 'comet_matched_frags_mz'})

    # Reorder columns: lead, fragment (that exist), then rest
    lead = [c for c in LEAD_COLS if c in out.columns]
    frag = [c for c in FRAGMENT_COLS if c in out.columns]
    rest = [c for c in out.columns if c not in lead and c not in frag]
    col_order = lead + frag + rest
    out = out[[c for c in col_order if c in out.columns]]

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
            ax.axhline(args.pep_threshold, color='red', linestyle='--', linewidth=2, label=f'PEP = {args.pep_threshold}')
            ax.axvline(args.q_threshold, color='orange', linestyle='--', linewidth=2, label=f'Q-value = {args.q_threshold}')
            ax.set_xlabel(f'Q-value ({qcol})', fontsize=12, color='0.85')
            ax.set_ylabel(f'PEP ({pepcol})', fontsize=12, color='0.85')
            ax.set_title(f'Confidence filter: {n_pass} / {n_total} rows pass (Q≤{args.q_threshold} or PEP≤{args.pep_threshold})', fontsize=14, color='0.9')
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

    out.to_csv(out_csv, index=False)
    print(f"Output: {out_csv} ({n_pass} rows, {n_total - n_pass} removed)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
