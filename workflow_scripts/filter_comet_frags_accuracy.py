#!/usr/bin/env python3
"""
Step 5: Significant fragmentation — filter Comet fragments by PPM (5 ppm).

Each row = one scan. Fragments that fail 5 ppm are removed.
Scans with 0 fragments after filtering are discarded entirely → reduces scan count per peptide.
Keeps only fragments that pass ppm. Rejects fragments that match b/y ions (CID/HCD mis-assigned as ETD c/z).
All mass constants and formulas aligned with Comet (CometDataInternal.h, CometMassSpecUtils).
Uses charge-agnostic matching (z=1..5), picks best match (min |ppm|).
Overwrites matched fragment ions, pairs, single-AA overhangs with significant-only.
Output: comet_frags_perc_openMS_confidence_accuracy.csv

Diagnostics (in diagnostics/):
  1) significant_fragmentation_observed_vs_expected_mz.png — scatter: theoretical m/z (x) vs observed/theoretical (y)
  2) significant_fragmentation_delta_histogram.png — histogram of implied Δ (m/z) = obs − theo for excluded/rejected

Usage:
    python filter_comet_frags_accuracy.py
    python filter_comet_frags_accuracy.py --input data/comet_frags_perc_openMS_confidence.csv --output-dir data
    python filter_comet_frags_accuracy.py --ppm 20  # loosen if needed
"""

import argparse
import os
import re
import sys

PPM_THRESHOLD = 5.0  # Orbitrap ~5 ppm when assignment correct; all calculations aligned with Comet

# Add project root for imports
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
DEFAULT_CSV = os.path.join(_PROJECT_ROOT, 'data', 'comet_frags_perc_openMS_confidence.csv')
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def _significant_pairs_and_overhangs_from_ions(sig_ions, peptide):
    """From significant fragment ion names, compute pairs and overhang positions."""
    clean = re.sub(r'\[.*?\]', '', peptide) if peptide else ''
    L = len(clean) if clean else 0
    if L == 0 or not sig_ions:
        return '', ''
    c_nums, z_nums, z1_nums = [], [], []
    for name in sig_ions:
        if not name or not isinstance(name, str):
            continue
        name = name.strip()
        if name.startswith('c') and name[1:].isdigit():
            n = int(name[1:])
            if 1 <= n <= L:
                c_nums.append(n)
        elif name.startswith('z1_') and name.split('_')[-1].isdigit():
            n = int(name.split('_')[-1])
            if 1 <= n <= L:
                z1_nums.append(n)
        elif name.startswith('z') and name[1:].isdigit():
            n = int(name[1:])
            if 1 <= n <= L:
                z_nums.append(n)
    c_set, z_set, z1_set = set(c_nums), set(z_nums), set(z1_nums)
    pairs = []
    overhang_pos = set()
    for n in sorted(c_set):
        if (n + 1) in c_set:
            pairs.append(f'c{n}|c{n+1}')
            overhang_pos.add(n + 1)
    for n in sorted(z_set):
        if (n + 1) in z_set:
            pairs.append(f'z{n}|z{n+1}')
            overhang_pos.add(L - n + 1)
    for n in sorted(z1_set):
        if (n + 1) in z1_set:
            pairs.append(f'z{n}+1|z{n+1}+1')
            overhang_pos.add(L - n + 1)
    pairs_str = ','.join(pairs) if pairs else ''
    overhang_parts = [f'{pos}{clean[pos - 1]}' for pos in sorted(overhang_pos) if 1 <= pos <= L]
    overhang_str = ','.join(overhang_parts) if overhang_parts else ''
    return pairs_str, overhang_str


def main():
    ap = argparse.ArgumentParser(description='Step 5: Significant fragmentation — filter fragments by 5 ppm')
    ap.add_argument('--input', '-i', default=DEFAULT_CSV,
                    help=f'Input CSV from Step 4 (default: {os.path.basename(DEFAULT_CSV)})')
    ap.add_argument('--output', '-o', default=None, help='Output CSV path')
    ap.add_argument('--output-dir', default=None, help='Output directory')
    ap.add_argument('--ppm', type=float, default=PPM_THRESHOLD,
                    help=f'PPM threshold (default: {PPM_THRESHOLD}). If all peptides are rejected, try --ppm 2000 for testing (mass-constant mismatch).')
    args = ap.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    try:
        import pandas as pd
        import numpy as np
    except ImportError:
        print("Error: pandas and numpy required.")
        sys.exit(1)

    try:
        from visualization.chromatograms import get_best_theoretical_match, matches_b_or_y_ion
    except ImportError as e:
        print("Error: Could not import visualization.chromatograms (run from project root).", file=sys.stderr)
        print(f"  {e}", file=sys.stderr)
        sys.exit(1)

    # Read CSV
    try:
        with open(args.input, 'r') as f:
            first = f.readline()
        skip = 1 if 'CometVersion' in first else 0
    except Exception:
        skip = 0
    df = pd.read_csv(args.input, sep=',', skiprows=skip, engine='python', quotechar='"', on_bad_lines='warn')

    # Normalize column names: comet_matched_frags (was "matched fragment ions")
    if 'matched fragment ions' in df.columns and 'comet_matched_frags' not in df.columns:
        df = df.rename(columns={'matched fragment ions': 'comet_matched_frags'})
    if 'matched fragment ion mz' in df.columns and 'comet_matched_frags_mz' not in df.columns:
        df = df.rename(columns={'matched fragment ion mz': 'comet_matched_frags_mz'})
    if 'matched fragment ion intensities' in df.columns and 'comet_matched_frags_intensities' not in df.columns:
        df = df.rename(columns={'matched fragment ion intensities': 'comet_matched_frags_intensities'})
    if 'comet_amtched_frags_intensities' in df.columns and 'comet_matched_frags_intensities' not in df.columns:
        df = df.rename(columns={'comet_amtched_frags_intensities': 'comet_matched_frags_intensities'})
    if 'matched fragment ion quality scores' in df.columns and 'comet_matched_frags_quality_scores' not in df.columns:
        df = df.rename(columns={'matched fragment ion quality scores': 'comet_matched_frags_quality_scores'})

    col_ions = 'comet_matched_frags'
    col_mz = 'comet_matched_frags_mz'
    col_int = 'comet_matched_frags_intensities'
    col_pairs = 'single_aa_overhang_fragment_pairs'
    col_positions = 'single_aa_overhangs_protein_positions'
    if 'significant_frags' not in df.columns:
        df['significant_frags'] = ''
    if 'significant_frags_mz' not in df.columns:
        df['significant_frags_mz'] = ''

    if col_ions not in df.columns or col_mz not in df.columns:
        print("Error: CSV must have 'comet_matched_frags' and 'comet_matched_frags_mz' (or legacy 'matched fragment ions'/'matched fragment ion mz').")
        sys.exit(1)

    # Collect all fragment data for diagnostic scatter (before filtering)
    scatter_data = []  # list of (obs_mz, exp_mz, ppm_diff, row_idx, kept, peptide_rejected)
    peptides_rejected = set()  # row indices of peptides that will have 0 sig fragments

    for idx in df.index:
        row = df.loc[idx]
        peptide = row.get('peptide_sequence') or row.get('plain_peptide') or row.get('sequence') or ''
        if not peptide or not isinstance(peptide, str):
            continue
        ions_val = row.get(col_ions, '')
        mz_val = row.get(col_mz, '')
        if pd.isna(ions_val) or pd.isna(mz_val) or not str(ions_val).strip() or not str(mz_val).strip():
            continue
        ions_list = [p.strip() for p in str(ions_val).strip().strip('"').split(',') if p.strip()]
        mz_list = [p.strip() for p in str(mz_val).strip().strip('"').split(',') if p.strip()]
        if len(ions_list) != len(mz_list):
            continue
        sig_count = 0
        for name, mz_s in zip(ions_list, mz_list):
            if not name or not mz_s:
                continue
            if not (name.startswith('c') and name[1:].isdigit()) and not (name.startswith('z1_') and name.split('_')[-1].isdigit()) and not (name.startswith('z') and name[1:].isdigit()):
                continue
            try:
                obs_mz = float(mz_s)
            except ValueError:
                continue
            if obs_mz <= 0:
                continue
            match = get_best_theoretical_match(name, peptide, obs_mz)
            if match is None:
                continue
            theo_mz, best_charge, ppm_diff = match
            passed_ppm = ppm_diff <= args.ppm
            kept = passed_ppm and not matches_b_or_y_ion(name, peptide, obs_mz, args.ppm)
            if kept:
                sig_count += 1
            scatter_data.append((obs_mz, theo_mz, ppm_diff, idx, kept, False))
        if sig_count == 0:
            peptides_rejected.add(idx)
            for i in range(len(scatter_data) - len(ions_list), len(scatter_data)):
                scatter_data[i] = (scatter_data[i][0], scatter_data[i][1], scatter_data[i][2], scatter_data[i][3], scatter_data[i][4], True)

    # Filter: keep only fragments that pass ppm (and are not b/y mis-assignments)
    has_int = col_int in df.columns
    for idx in df.index:
        row = df.loc[idx]
        peptide = row.get('peptide_sequence') or row.get('plain_peptide') or row.get('sequence') or ''
        if not peptide or not isinstance(peptide, str):
            continue
        ions_val = row.get(col_ions, '')
        mz_val = row.get(col_mz, '')
        if pd.isna(ions_val) or pd.isna(mz_val) or not str(ions_val).strip() or not str(mz_val).strip():
            continue
        ions_list = [p.strip() for p in str(ions_val).strip().strip('"').split(',') if p.strip()]
        mz_list = [p.strip() for p in str(mz_val).strip().strip('"').split(',') if p.strip()]
        if len(ions_list) != len(mz_list):
            continue
        int_list = []
        if has_int and pd.notna(row.get(col_int)) and str(row.get(col_int)).strip():
            int_list = [p.strip() for p in str(row[col_int]).strip().strip('"').split(',') if p.strip()]
            if len(int_list) != len(ions_list):
                int_list = []
        sig_ions, sig_mz, sig_int = [], [], []
        for i, (name, mz_s) in enumerate(zip(ions_list, mz_list)):
            if not name or not mz_s:
                continue
            if not (name.startswith('c') and name[1:].isdigit()) and not (name.startswith('z1_') and name.split('_')[-1].isdigit()) and not (name.startswith('z') and name[1:].isdigit()):
                continue
            try:
                obs_mz = float(mz_s)
            except ValueError:
                continue
            if obs_mz <= 0:
                continue
            match = get_best_theoretical_match(name, peptide, obs_mz)
            if match is None:
                continue
            theo_mz, best_charge, ppm_diff = match
            passed_ppm = ppm_diff <= args.ppm
            kept = passed_ppm and not matches_b_or_y_ion(name, peptide, obs_mz, args.ppm)
            if kept:
                sig_ions.append(name)
                sig_mz.append(mz_s)
                if int_list and i < len(int_list):
                    sig_int.append(int_list[i])
        df.at[idx, col_ions] = ','.join(sig_ions) if sig_ions else ''
        df.at[idx, col_mz] = ','.join(sig_mz) if sig_mz else ''
        # Also write significant_frags (5ppm-filtered stage)
        df.at[idx, 'significant_frags'] = ','.join(sig_ions) if sig_ions else ''
        df.at[idx, 'significant_frags_mz'] = ','.join(sig_mz) if sig_mz else ''
        if has_int and col_int in df.columns:
            df.at[idx, col_int] = ','.join(sig_int) if sig_int else ''
        pairs_str, overhang_str = _significant_pairs_and_overhangs_from_ions(sig_ions, peptide)
        if col_pairs not in df.columns:
            df[col_pairs] = ''
        if col_positions not in df.columns:
            df[col_positions] = ''
        df.at[idx, col_pairs] = pairs_str
        seq_start = row.get('sequence_start_pos') or row.get('protein_position') or row.get('sequence_positions')
        start = None
        if pd.notna(seq_start) and str(seq_start).strip():
            try:
                start = int(float(str(seq_start).strip().split('-')[0].split()[0]))
            except (ValueError, TypeError):
                pass
        if start is not None and start > 0 and overhang_str:
            clean = re.sub(r'\[.*?\]', '', peptide) if peptide else ''
            parts = []
            for pos_res in overhang_str.split(','):
                pos_res = pos_res.strip()
                if pos_res and len(pos_res) >= 2 and pos_res[-1].isalpha() and pos_res[:-1].isdigit():
                    pos, res = int(pos_res[:-1]), pos_res[-1]
                    parts.append(f'{start + pos - 1}{res}')
                else:
                    parts.append(pos_res)
            overhang_str = ','.join(parts) if parts else overhang_str
        df.at[idx, col_positions] = overhang_str

    # Drop scans with 0 significant fragments
    mask = df[col_ions].apply(lambda x: bool(pd.notna(x) and str(x).strip()))
    df_filtered = df.loc[mask].copy()
    # Remove loss types column (no longer computed)
    if 'matched fragment ion loss types' in df_filtered.columns:
        df_filtered = df_filtered.drop(columns=['matched fragment ion loss types'])

    n_total = len(df)
    n_pass = len(df_filtered)
    n_frag_excluded = sum(1 for _ in scatter_data if not _[4])  # not kept
    n_frag_total = len(scatter_data)

    # Output paths
    input_dir = os.path.dirname(os.path.abspath(args.input))
    base = os.path.splitext(os.path.basename(args.input))[0]
    if base.endswith('_confidence'):
        base = base[:-len('_confidence')]
    out_dir = args.output_dir or input_dir
    os.makedirs(out_dir, exist_ok=True)
    diagnostics_dir = os.path.join(out_dir, 'diagnostics')
    os.makedirs(diagnostics_dir, exist_ok=True)
    if args.output:
        out_csv = args.output
    else:
        out_csv = os.path.join(out_dir, base + '_confidence_accuracy.csv')

    # Diagnostic scatter
    if scatter_data:
        try:
            import matplotlib.pyplot as plt
            from matplotlib.colors import Normalize

            obs_arr = np.array([d[0] for d in scatter_data])
            exp_arr = np.array([d[1] for d in scatter_data])
            ppm_arr = np.array([d[2] for d in scatter_data])
            rejected_pep = np.array([d[5] for d in scatter_data])
            kept_arr = np.array([d[4] for d in scatter_data])  # kept = ppm pass (no b/y)

            # Y = observed / theoretical; X = theoretical
            ratio_arr = np.where(exp_arr > 0, obs_arr / exp_arr, np.nan)
            valid = np.isfinite(ratio_arr)

            fig, ax = plt.subplots(figsize=(10, 10))
            fig.patch.set_facecolor('black')
            ax.set_facecolor('black')
            # Normalize ppm for colorbar (clip for display)
            ppm_clip = np.clip(ppm_arr, 0, 50)
            norm = Normalize(vmin=0, vmax=50)
            try:
                cmap = plt.colormaps.get_cmap('viridis_r')
            except AttributeError:
                cmap = plt.cm.get_cmap('viridis_r')
            # Plot: kept fragments (ppm pass, not b/y)
            pass_mask = kept_arr & ~rejected_pep & valid
            if np.any(pass_mask):
                ax.scatter(exp_arr[pass_mask], ratio_arr[pass_mask], c=ppm_clip[pass_mask], cmap=cmap, norm=norm, s=20, alpha=0.8, edgecolors='0.6', linewidths=0.5)
            # Plot: excluded fragments (fail ppm or b/y match)
            excl_mask = ~kept_arr & ~rejected_pep & valid
            if np.any(excl_mask):
                ax.scatter(exp_arr[excl_mask], ratio_arr[excl_mask], c='orange', s=20, alpha=0.6, edgecolors='0.6', linewidths=0.5)
            # Plot: rejected peptide fragments (0 sig fragments)
            rej_mask = rejected_pep & valid
            if np.any(rej_mask):
                ax.scatter(exp_arr[rej_mask], ratio_arr[rej_mask], c='#888888', s=20, alpha=0.7)
            ax.axhline(1.0, color='0.6', linestyle='--', alpha=0.5, linewidth=1)
            ax.set_xlabel('Theoretical m/z', fontsize=12, color='0.85')
            ax.set_ylabel('Observed m/z / Theoretical m/z', fontsize=12, color='0.85')
            ax.set_title(f'Significant fragmentation: {n_frag_excluded} / {n_frag_total} fragments excluded (> {args.ppm} ppm); {len(peptides_rejected)} peptides rejected (0 sig fragments)', fontsize=11, color='0.9')
            ax.set_xlim(exp_arr[valid].min() * 0.98, exp_arr[valid].max() * 1.02)
            r_min, r_max = ratio_arr[valid].min(), ratio_arr[valid].max()
            margin = max(0.01, (r_max - r_min) * 0.1)
            ax.set_ylim(r_min - margin, r_max + margin)
            ax.tick_params(colors='0.85')
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_color('0.85')
            for spine in ax.spines.values():
                spine.set_color('0.6')
            ax.legend([
                f'Kept: ppm pass ({np.sum(pass_mask)})',
                f'Excluded: fail ppm or b/y ({np.sum(excl_mask)})',
                f'Rejected peptide fragments ({np.sum(rej_mask)})'
            ], loc='upper left', fontsize=9, facecolor='#1a1a1a', edgecolor='0.6', labelcolor='0.9')
            ax.grid(True, alpha=0.25)
            diag_path = os.path.join(diagnostics_dir, 'significant_fragmentation_observed_vs_expected_mz.png')
            fig.savefig(diag_path, dpi=150, bbox_inches='tight', facecolor='black')
            plt.close(fig)
            print(f"Significant fragmentation: diagnostic scatter saved: {diag_path}")

            # Delta histogram: Δ (m/z) = obs − theo for excluded points
            bad_mask = ~kept_arr & valid
            if np.any(bad_mask):
                delta_arr = (ratio_arr - 1.0) * exp_arr  # obs − theo in m/z
                delta_bad = delta_arr[bad_mask]
                delta_bad = delta_bad[np.isfinite(delta_bad)]
                if len(delta_bad) > 0:
                    try:
                        fig2, ax2 = plt.subplots(figsize=(10, 6))
                        fig2.patch.set_facecolor('black')
                        ax2.set_facecolor('black')
                        ax2.hist(delta_bad, bins=80, color='steelblue', alpha=0.7, edgecolor='0.6')
                        ax2.axvline(0, color='green', linestyle='-', alpha=0.5, linewidth=1)
                        ax2.set_xlabel('Δ (m/z) = observed − theoretical', fontsize=11, color='0.85')
                        ax2.set_ylabel('Count', fontsize=11, color='0.85')
                        ax2.set_title('Fragment offset histogram (excluded + rejected)', fontsize=10, color='0.9')
                        ax2.tick_params(colors='0.85')
                        for label in ax2.get_xticklabels() + ax2.get_yticklabels():
                            label.set_color('0.85')
                        for spine in ax2.spines.values():
                            spine.set_color('0.6')
                        ax2.grid(True, alpha=0.25)
                        delta_path = os.path.join(diagnostics_dir, 'significant_fragmentation_delta_histogram.png')
                        fig2.savefig(delta_path, dpi=150, bbox_inches='tight', facecolor='black')
                        plt.close(fig2)
                        print(f"Significant fragmentation: delta histogram saved: {delta_path}")
                    except Exception as e2:
                        print(f"Warning: Could not create delta histogram: {e2}")
        except Exception as e:
            print(f"Warning: Could not create diagnostic scatter: {e}")

    for old, new in [
        ('MS1_retention_time_sec', 'MS1_RT_sec'),
        ('MS1_retention_time_min', 'MS1_RT_minutes'),
        ('MS1_retention_time_intensity', 'MS1_RT_intensity'),
        ('MS2_retention_time_sec', 'MS2_RT_sec'),
        ('MS2_retention_time_min', 'MS2_RT_minutes'),
        ('retention_time_sec', 'RT_sec'),
        ('retention_time_min', 'RT_minutes'),
    ]:
        if old in df_filtered.columns:
            if new in df_filtered.columns:
                df_filtered[new] = df_filtered[old].where(pd.notna(df_filtered[old]), df_filtered[new])
                df_filtered = df_filtered.drop(columns=[old])
            else:
                df_filtered = df_filtered.rename(columns={old: new})

    # Final ordering rule: RT columns explicitly in seconds always go last.
    rt_sec_cols = [
        c for c in df_filtered.columns
        if (('RT' in c or 'retention_time' in c) and c.endswith('_sec'))
    ]
    if rt_sec_cols:
        non_rt_sec_cols = [c for c in df_filtered.columns if c not in rt_sec_cols]
        df_filtered = df_filtered[non_rt_sec_cols + rt_sec_cols]

    df_filtered.to_csv(out_csv, index=False)
    n_scans_dropped = n_total - n_pass
    print(f"Significant fragmentation: Output: {out_csv} ({n_pass} scans retained, {n_scans_dropped} scans discarded (0 fragments); {n_frag_excluded} fragments excluded by {args.ppm} ppm)")
    if n_pass == 0 and n_total > 0:
        print("Warning: 0 rows passed. If this is unexpected, try --ppm 2000 (mass-constant mismatch between theoretical and Comet-observed m/z).")
    return 0


if __name__ == '__main__':
    sys.exit(main())
