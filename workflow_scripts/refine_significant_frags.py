#!/usr/bin/env python3
"""
Step 8: Refine significant fragments by 0.5% max MS2 signal.

Keeps only fragments that have signal >= 1% of max MS2 signal in the integration window.

Replaces significant_fragment_* columns with refined data. Removes rows with 0
significant fragments after refinement.

Output: comet_frags_perc_openMS_confidence_accuracy_extraction_significance.csv

Diagnostic plots (to diagnostics/):
  1) MS2 max signal vs fragment signal (denote fragments that do not pass)
  2) Comparative total peptide overlay before vs after (if chromatogram traces available)
  3) All peptides overlay showing rejected peaks only

Usage:
  python refine_significant_frags.py --mzml data/file.mzML
  python refine_significant_frags.py --mzml file.mzML --input extraction.csv --output-dir results
"""

import argparse
import os
import re
import sys
import time

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

MIN_INTENSITY_FRAC = 0.01  # 1% of max MS2
PPM_MS2 = 5.0
DEFAULT_INPUT = os.path.join(_PROJECT_ROOT, 'data', 'comet_frags_perc_openMS_confidence_accuracy_extraction.csv')
DEFAULT_MZML = os.path.join(_PROJECT_ROOT, 'data', 'WT_nep2_0MUrea_08.mzML')


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
    ap = argparse.ArgumentParser(description='Step 8: Refine significant fragments by 1% max MS2 signal.')
    ap.add_argument('--mzml', '-m', default=DEFAULT_MZML, help=f'Path to mzML file (default: {os.path.basename(DEFAULT_MZML)})')
    ap.add_argument('--input', '-i', default=DEFAULT_INPUT, help=f'Input CSV from Step 6 (default: {os.path.basename(DEFAULT_INPUT)})')
    ap.add_argument('--output', '-o', default=None, help='Output CSV path')
    ap.add_argument('--output-dir', default=None, help='Output directory')
    ap.add_argument('--min-frac', type=float, default=MIN_INTENSITY_FRAC,
                    help=f'Min fragment signal as fraction of max MS2 (default: {MIN_INTENSITY_FRAC})')
    args = ap.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input CSV not found: {args.input}")
        sys.exit(1)
    if not os.path.exists(args.mzml):
        print(f"Error: mzML file not found: {args.mzml}")
        sys.exit(1)

    try:
        import pandas as pd
        import numpy as np
    except ImportError:
        print("Error: pandas and numpy required.")
        sys.exit(1)

    try:
        from visualization.chromatograms import get_averaged_ms2_spectrum_in_window
    except ImportError as e:
        print("Error: Could not import visualization.chromatograms (run from project root).", file=sys.stderr)
        print(f"  {e}", file=sys.stderr)
        sys.exit(1)

    input_dir = os.path.dirname(os.path.abspath(args.input))
    base = os.path.splitext(os.path.basename(args.input))[0]
    out_dir = args.output_dir or input_dir
    os.makedirs(out_dir, exist_ok=True)
    diag_dir = os.path.join(out_dir, 'diagnostics')
    os.makedirs(diag_dir, exist_ok=True)

    if args.output:
        out_csv = args.output
    else:
        out_csv = os.path.join(out_dir, f'{base}_significance.csv')

    # Input: significant_frags (Step 5) or legacy significant_fragment_ions/comet_matched_frags
    # Output: refined_significant_frags (1% noise filter)
    cols_0 = pd.read_csv(args.input, nrows=0).columns
    has_sig = 'significant_frags' in cols_0
    has_sig_mz = 'significant_frags_mz' in cols_0
    has_legacy_sig = 'significant_fragment_ions' in cols_0
    has_legacy_mz = 'significant_fragment_mz' in cols_0
    has_comet = 'comet_matched_frags' in cols_0 or 'matched fragment ions' in cols_0
    col_ions_in = 'significant_frags' if has_sig else ('significant_fragment_ions' if has_legacy_sig else ('comet_matched_frags' if has_comet else 'comet_matched_frags'))
    col_mz_in = 'significant_frags_mz' if has_sig_mz else ('significant_fragment_mz' if has_legacy_mz else ('comet_matched_frags_mz' if has_comet else 'comet_matched_frags_mz'))
    col_ions = 'refined_significant_frags'
    col_mz = 'refined_significant_frags_mz'
    col_matched_ions = 'comet_matched_frags'
    col_matched_mz = 'comet_matched_frags_mz'
    col_int = 'matched fragment ion intensities'
    col_pairs = 'refined_significant_fragment_pairs'
    col_overhangs = 'refined_significant_single_aa_overhangs'
    col_overhangs_prot = 'refined_significant_single_aa_overhangs_protein_positions'

    print("[Step 8] Loading input CSV...", flush=True)
    try:
        with open(args.input, 'r') as f:
            first = f.readline()
        skip = 1 if 'CometVersion' in first else 0
    except Exception:
        skip = 0
    df = pd.read_csv(args.input, sep=',', skiprows=skip, engine='python', quotechar='"', on_bad_lines='warn')
    print(f"[Step 8] Loaded {len(df)} rows", flush=True)

    # Normalize column names: comet_matched_frags (was "matched fragment ions")
    if 'matched fragment ions' in df.columns and 'comet_matched_frags' not in df.columns:
        df = df.rename(columns={'matched fragment ions': 'comet_matched_frags'})
    if 'matched fragment ion mz' in df.columns and 'comet_matched_frags_mz' not in df.columns:
        df = df.rename(columns={'matched fragment ion mz': 'comet_matched_frags_mz'})

    # Resolve RT window columns
    min_rt_col = 'detected_peak_min_rt' if 'detected_peak_min_rt' in df.columns else 'min_rt'
    max_rt_col = 'detected_peak_max_rt' if 'detected_peak_max_rt' in df.columns else 'max_rt'
    if min_rt_col not in df.columns:
        min_rt_col = 'collection_min_rt'
    if max_rt_col not in df.columns:
        max_rt_col = 'collection_max_rt'
    if min_rt_col not in df.columns:
        min_rt_col = 'anchor_rt'
    if max_rt_col not in df.columns:
        max_rt_col = 'anchor_rt'

    if col_ions_in not in df.columns or col_mz_in not in df.columns:
        print("Error: CSV must have fragment ion columns (significant_frags or comet_matched_frags).")
        sys.exit(1)
    if col_ions not in df.columns:
        df[col_ions] = ''
    if col_mz not in df.columns:
        df[col_mz] = ''
    if col_pairs not in df.columns:
        df[col_pairs] = ''
    if col_overhangs not in df.columns:
        df[col_overhangs] = ''
    if col_overhangs_prot not in df.columns:
        df[col_overhangs_prot] = ''

    n_before = len(df)
    scatter_data = []  # (max_ms2, frag_signal, passed, row_idx, ion_name)
    rows_to_drop = []
    skip_no_rt = 0
    skip_no_ions = 0
    skip_ions_mismatch = 0
    skip_no_ms2 = 0
    first_ms2_call = True

    def _log(msg: str) -> None:
        print(msg, flush=True)

    _log("[Step 8] Refine significant fragments (1% max MS2 signal filter)")
    _log(f"[Step 8] Input: {args.input}")
    _log(f"[Step 8] mzML: {args.mzml}")
    _log(f"[Step 8] Output: {out_csv}")
    _log(f"[Step 8] Threshold: {args.min_frac*100:.2f}% of max MS2 signal in RT window")
    _log(f"[Step 8] Input column: {col_ions_in} -> Output: {col_ions}")
    _log(f"[Step 8] Processing {n_before} rows (extracting MS2 spectra per row)...")
    progress_interval = max(1, min(50, n_before // 10))  # every 50 rows or ~10%
    t_start = time.perf_counter()

    for i, idx in enumerate(df.index):
        if i == 1 and not first_ms2_call:
            t_first = time.perf_counter() - t_start
            _log(f"[Step 8] First row done ({t_first:.0f}s). mzML cached. Processing...")
        elif (i + 1) % progress_interval == 0 or i == n_before - 1:
            elapsed = time.perf_counter() - t_start
            rate = (i + 1) / elapsed if elapsed > 0.5 else 0
            eta = (n_before - i - 1) / rate if rate > 0 else 0
            pct = 100 * (i + 1) / n_before
            _log(f"[Step 8]   Progress: {i + 1}/{n_before} ({pct:.1f}%) | {elapsed:.0f}s elapsed | ~{eta:.0f}s left")
        row = df.loc[idx]
        peptide = row.get('plain_peptide') or row.get('sequence') or ''
        if not peptide or not isinstance(peptide, str):
            continue
        min_rt = row.get(min_rt_col)
        max_rt = row.get(max_rt_col)
        if pd.isna(min_rt) or pd.isna(max_rt) or min_rt is None or max_rt is None:
            min_rt = row.get('anchor_rt') or row.get('MS1_retention_time_sec')
            max_rt = min_rt
        min_rt = float(min_rt) if min_rt is not None and not pd.isna(min_rt) else None
        max_rt = float(max_rt) if max_rt is not None and not pd.isna(max_rt) else None
        if min_rt is None or max_rt is None or max_rt <= min_rt:
            skip_no_rt += 1
            rows_to_drop.append(idx)
            continue

        ions_val = row.get(col_ions_in, '')
        mz_val = row.get(col_mz_in, '')
        int_val = row.get(col_int, '')
        if pd.isna(ions_val) or pd.isna(mz_val) or not str(ions_val).strip() or not str(mz_val).strip():
            skip_no_ions += 1
            rows_to_drop.append(idx)
            continue

        ions_list = [p.strip() for p in str(ions_val).strip().strip('"').split(',') if p.strip()]
        mz_list = [p.strip() for p in str(mz_val).strip().strip('"').split(',') if p.strip()]
        int_list = [p.strip() for p in str(int_val).strip().strip('"').split(',') if p.strip()] if pd.notna(int_val) and str(int_val).strip() else []
        loss_val = row.get('matched fragment ion loss types', '')
        loss_list = [p.strip() for p in str(loss_val).strip().strip('"').split(',') if p.strip()] if pd.notna(loss_val) and str(loss_val).strip() else []
        # When significant_fragment_ions exists but significant_fragment_mz does not, ions (24) and mz (12) length mismatch causes drops.
        # Fallback: use matched fragment ions + mz (always 1:1) so we can refine.
        if len(ions_list) != len(mz_list) and col_matched_ions in df.columns and col_matched_mz in df.columns and (col_matched_ions != col_ions_in or col_matched_mz != col_mz_in):
            matched_ions_val = row.get(col_matched_ions, '')
            matched_mz_val = row.get(col_matched_mz, '')
            if pd.notna(matched_ions_val) and pd.notna(matched_mz_val) and str(matched_ions_val).strip() and str(matched_mz_val).strip():
                ions_list = [p.strip() for p in str(matched_ions_val).strip().strip('"').split(',') if p.strip()]
                mz_list = [p.strip() for p in str(matched_mz_val).strip().strip('"').split(',') if p.strip()]
        if len(ions_list) != len(mz_list):
            skip_ions_mismatch += 1
            rows_to_drop.append(idx)
            continue

        if first_ms2_call:
            _log(f"[Step 8] Row {i+1}: first MS2 extraction - loading mzML (1-2 min for large files)...")
            first_ms2_call = False
        ms2_mzs, ms2_ints = get_averaged_ms2_spectrum_in_window(args.mzml, min_rt, max_rt, ppm_tolerance=20.0)
        if ms2_mzs is None or ms2_ints is None or len(ms2_mzs) == 0 or len(ms2_ints) == 0:
            skip_no_ms2 += 1
            rows_to_drop.append(idx)
            continue

        max_ms2 = float(np.max(ms2_ints))
        noise_floor = max_ms2 * args.min_frac if max_ms2 > 0 else 0.0
        ms2_mzs_arr = np.asarray(ms2_mzs)
        ms2_ints_arr = np.asarray(ms2_ints)

        sig_ions, sig_mz, sig_int, sig_loss = [], [], [], []
        for j, (name, mz_s) in enumerate(zip(ions_list, mz_list)):
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
            # Use observed m/z from CSV (Comet-reported) to find peak in MS2; charge-agnostic
            idx_mz = np.argmin(np.abs(ms2_mzs_arr - obs_mz))
            tol = max(0.02, obs_mz * PPM_MS2 * 1e-6)
            if np.abs(ms2_mzs_arr[idx_mz] - obs_mz) <= tol:
                frag_signal = float(ms2_ints_arr[idx_mz])
                peak_mz = float(ms2_mzs_arr[idx_mz])
            else:
                frag_signal = 0.0
                peak_mz = obs_mz
            passed = frag_signal >= noise_floor
            scatter_data.append((max_ms2, frag_signal, passed, idx, name))
            if passed:
                sig_ions.append(name)
                sig_mz.append(f'{peak_mz:.6f}')
                if j < len(int_list):
                    sig_int.append(int_list[j])
                sig_loss.append(loss_list[j] if j < len(loss_list) else 'none')

        if len(sig_ions) == 0:
            rows_to_drop.append(idx)
        else:
            df.at[idx, col_ions] = ','.join(sig_ions)
            if col_mz and col_mz in df.columns:
                df.at[idx, col_mz] = ','.join(sig_mz)
            if col_int in df.columns and sig_int:
                df.at[idx, col_int] = ','.join(sig_int)
            if 'matched fragment ion loss types' in df.columns:
                df.at[idx, 'matched fragment ion loss types'] = ','.join(sig_loss) if sig_loss else ''
            pairs_str, overhang_str = _significant_pairs_and_overhangs_from_ions(sig_ions, peptide)
            clean = re.sub(r'\[.*?\]', '', peptide) if peptide else ''
            if col_pairs in df.columns:
                df.at[idx, col_pairs] = pairs_str
            if col_overhangs in df.columns:
                df.at[idx, col_overhangs] = overhang_str
            if col_overhangs_prot in df.columns and overhang_str:
                try:
                    overhang_positions = set()
                    for part in overhang_str.split(','):
                        m = re.match(r'^(\d+)([A-Z])$', part.strip())
                        if m:
                            overhang_positions.add((int(m.group(1)), m.group(2)))
                    seq_start = row.get('sequence_start_pos') or row.get('sequence_positions')
                    if pd.notna(seq_start) and str(seq_start).strip():
                        start = int(str(seq_start).split('-')[0].split(',')[0].strip())
                        prot_parts = [f'{start + p - 1}{aa}' for p, aa in sorted(overhang_positions) if 1 <= p <= len(clean)]
                        df.at[idx, col_overhangs_prot] = ','.join(prot_parts)
                except Exception:
                    pass

    df = df.drop(index=rows_to_drop).reset_index(drop=True)
    n_after = len(df)

    n_passed = len([x for x in scatter_data if x[2]])
    n_failed = len([x for x in scatter_data if not x[2]])
    t_total = time.perf_counter() - t_start
    _log(f"[Step 8] Done processing in {t_total:.0f}s ({n_before / t_total:.1f} rows/s)")
    _log(f"[Step 8] Fragments: {n_passed} passed, {n_failed} below {args.min_frac*100:.2f}% threshold")
    _log(f"[Step 8] Rows: {n_before} -> {n_after} (removed {n_before - n_after} with 0 refined fragments)")
    if skip_no_rt or skip_no_ions or skip_ions_mismatch or skip_no_ms2:
        parts = []
        if skip_no_rt:
            parts.append(f"no RT window ({skip_no_rt})")
        if skip_no_ions:
            parts.append(f"no ions ({skip_no_ions})")
        if skip_ions_mismatch:
            parts.append(f"ions/mz mismatch ({skip_ions_mismatch})")
        if skip_no_ms2:
            parts.append(f"no MS2 in window ({skip_no_ms2})")
        _log("[Step 8] Skipped: " + ", ".join(parts))
    _log("[Step 8] Writing diagnostic plots...")

    # Diagnostic 1: MS2 max vs fragment signal scatter
    if scatter_data:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            passed_vals = [(m, f) for m, f, p, _, _ in scatter_data if p]
            fail_vals = [(m, f) for m, f, p, _, _ in scatter_data if not p]
            fig, ax = plt.subplots(figsize=(10, 8))
            fig.patch.set_facecolor('black')
            ax.set_facecolor('black')
            if passed_vals:
                ax.scatter([m for m, f in passed_vals], [f for m, f in passed_vals], c='green', alpha=0.6, s=20, label=f'Pass (n={len(passed_vals)})')
            if fail_vals:
                ax.scatter([m for m, f in fail_vals], [f for m, f in fail_vals], c='red', alpha=0.6, s=20, label=f'Fail (n={len(fail_vals)})')
            max_all = max(m for m, _ in scatter_data) if scatter_data else 1
            ax.plot([0, max_all], [0, max_all * args.min_frac], color='0.6', linestyle='--', linewidth=2, label=f'{args.min_frac*100:.1f}% threshold')
            ax.set_xlabel('MS2 max signal (in window)', fontsize=12, color='0.85')
            ax.set_ylabel('Fragment signal', fontsize=12, color='0.85')
            ax.set_title('Step 8: MS2 max vs fragment signal (1% filter)', fontsize=14, color='0.9')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.tick_params(colors='0.85')
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_color('0.85')
            for spine in ax.spines.values():
                spine.set_color('0.6')
            ax.legend(facecolor='#1a1a1a', edgecolor='0.6', labelcolor='0.9')
            ax.grid(True, alpha=0.25)
            plt.tight_layout()
            diag_path = os.path.join(diag_dir, 'refine_significant_frags_ms2_scatter.png')
            fig.savefig(diag_path, dpi=150, bbox_inches='tight', facecolor='black')
            plt.close(fig)
            _log(f"[Step 8] Diagnostic: MS2 scatter -> {os.path.basename(diag_path)}")
        except Exception as e:
            _log(f"[Step 8] Warning: Could not create scatter plot: {e}")

    _log("[Step 8] Writing output CSV...")
    df.to_csv(out_csv, index=False)
    _log(f"[Step 8] Output saved: {os.path.abspath(out_csv)}")
    _log(f"[Step 8] Summary: {n_after} rows retained, {n_before - n_after} rows removed (0 refined fragments)")


if __name__ == '__main__':
    main()
