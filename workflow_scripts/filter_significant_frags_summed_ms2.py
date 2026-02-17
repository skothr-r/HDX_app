#!/usr/bin/env python3
"""
Combined significant fragmentation: 5 ppm + min sig frags + 1% max MS2 (all from summed extracted MS2).

Uses summed MS2 spectrum in the integration window for all three filters:
  1) 5 ppm: keep fragments with peak within 5 ppm of theoretical c/z/z+1 m/z
  2) Min sig frags: drop rows with fewer than min_sig_frag_count
  3) 1% max: keep only fragments with intensity >= 1% of max MS2 in window

Replaces Step 5 (filter_comet_frags_accuracy) + Step 8 (refine_significant_frags).
Input: extraction output (has RT windows: detected_peak_min_rt/max_rt or collection_min_rt/max_rt).
Output: significant_frags columns (pass all 3 criteria: 5 ppm, min sig frags, 1% max). Writes _significance.csv.

Usage:
  python filter_significant_frags_summed_ms2.py --mzml file.mzML --input extraction.csv --output-dir results
  python filter_significant_frags_summed_ms2.py --mzml file.mzML --input extraction.csv --min-frac 0.01 --ppm 5 --min-sig-frags 3
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
PPM_THRESHOLD = 5.0
MIN_SIG_FRAGS = 3
DEFAULT_INPUT = os.path.join(_PROJECT_ROOT, 'data', 'comet_frags_perc_openMS_prefilter_extraction_envelope.csv')
DEFAULT_MZML = os.path.join(_PROJECT_ROOT, 'data', 'WT_nep2_0MUrea_08.mzML')


def _ion_name_to_comet_format(label):
    """Convert theoretical label (e.g. 'z7+1') to Comet CSV format ('z1_7')."""
    if label.endswith('+1') and label.startswith('z') and label[1:-2].isdigit():
        k = int(label[1:-2])
        return f'z1_{k}'
    return label


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
    ap = argparse.ArgumentParser(
        description='Combined significant fragmentation: 5 ppm + min sig frags + 1%% max MS2 (all from summed MS2).'
    )
    ap.add_argument('--mzml', '-m', default=DEFAULT_MZML, help=f'Path to mzML file')
    ap.add_argument('--input', '-i', default=DEFAULT_INPUT, help='Input CSV from extraction (has RT windows)')
    ap.add_argument('--output', '-o', default=None, help='Output CSV path')
    ap.add_argument('--output-dir', default=None, help='Output directory')
    ap.add_argument('--ppm', type=float, default=PPM_THRESHOLD, help=f'PPM tolerance (default: {PPM_THRESHOLD})')
    ap.add_argument('--min-frac', type=float, default=MIN_INTENSITY_FRAC,
                    help=f'Min fragment signal as fraction of max MS2 (default: {MIN_INTENSITY_FRAC})')
    ap.add_argument('--min-sig-frags', type=int, default=MIN_SIG_FRAGS,
                    help=f'Min significant fragments per row (default: {MIN_SIG_FRAGS})')
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
        from visualization.chromatograms import (
            get_averaged_ms2_spectrum_in_window,
            theoretical_c_z_z1_ion_mz,
            matches_b_or_y_ion,
        )
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
        out_csv = os.path.join(out_dir, base + '_significance.csv')

    # Resolve RT window columns
    try:
        with open(args.input, 'r') as f:
            first = f.readline()
        skip = 1 if 'CometVersion' in first else 0
    except Exception:
        skip = 0
    df = pd.read_csv(args.input, sep=',', skiprows=skip, engine='python', quotechar='"', on_bad_lines='warn')

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

    if min_rt_col not in df.columns or max_rt_col not in df.columns:
        print("Error: CSV must have RT window columns (detected_peak_min_rt/max_rt, collection_min_rt/max_rt, or anchor_rt).")
        sys.exit(1)

    # Ensure output columns exist (significant_frags = pass all 3 criteria)
    for col in ['significant_frags', 'significant_frags_mz', 'single_aa_overhang_fragment_pairs',
                'single_aa_overhangs_protein_positions']:
        if col not in df.columns:
            df[col] = ''

    n_before = len(df)
    scatter_data = []
    rows_to_drop = []
    skip_no_rt = 0
    skip_no_peptide = 0
    skip_no_ms2 = 0
    first_ms2_call = True

    def _log(msg):
        print(msg, flush=True)

    _log("[Significant fragmentation] Combined: 5 ppm + min sig frags + 1% max (all from summed MS2)")
    _log(f"[Significant fragmentation] Input: {args.input}")
    _log(f"[Significant fragmentation] mzML: {args.mzml}")
    _log(f"[Significant fragmentation] Output: {out_csv}")
    _log(f"[Significant fragmentation] PPM: {args.ppm}, min frac: {args.min_frac*100:.2f}%, min sig frags: {args.min_sig_frags}")
    _log(f"[Significant fragmentation] Processing {n_before} rows...")
    # Report every N rows (finer for small datasets) and at least every 30s
    progress_interval = max(10, min(25, n_before // 20))  # e.g. 1500 -> 25, 500 -> 25
    t_start = time.perf_counter()
    last_progress_log = t_start

    for i, idx in enumerate(df.index):
        now = time.perf_counter()
        elapsed = now - t_start
        # First row: mzML load message already printed; after row 0, confirm cache
        if i == 1 and not first_ms2_call:
            t_first = now - t_start
            rate_est = 1.0 / t_first if t_first > 0 else 0
            eta = (n_before - 1) / rate_est if rate_est > 0 else 0
            _log(f"[Significant fragmentation] First row done ({t_first:.0f}s). mzML cached. ~{eta:.0f}s remaining for {n_before - 1} rows.")
        # Progress: every N rows, or every 30s, or on last row
        elif (i + 1) % progress_interval == 0 or i == n_before - 1 or (now - last_progress_log) >= 30:
            last_progress_log = now
            rate = (i + 1) / elapsed if elapsed > 0.5 else 0
            eta = (n_before - i - 1) / rate if rate > 0 else 0
            pct = 100 * (i + 1) / n_before
            _log(f"[Significant fragmentation]   Progress: {i + 1}/{n_before} ({pct:.1f}%) | {elapsed:.0f}s elapsed | ~{eta:.0f}s left | {rate:.1f} rows/s")

        row = df.loc[idx]
        peptide = row.get('plain_peptide') or row.get('sequence') or ''
        if not peptide or not isinstance(peptide, str):
            skip_no_peptide += 1
            rows_to_drop.append(idx)
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

        if first_ms2_call:
            _log(f"[Significant fragmentation] Row {i+1}: first MS2 extraction - loading mzML (1-2 min for large files)...")
            first_ms2_call = False

        # Get theoretical c/z/z+1 ions (charge 1)
        try:
            c_list, z_list, z1_list = theoretical_c_z_z1_ion_mz(peptide, fragment_charge=1)
        except Exception:
            rows_to_drop.append(idx)
            continue

        # Restrict m/z range to theoretical ions + margin (faster spectrum extraction)
        all_mz = [m for m, _ in c_list + z_list + z1_list if m > 0]
        mz_min = max(50, min(all_mz) - 50) if all_mz else None
        mz_max = min(5000, max(all_mz) + 50) if all_mz else None
        ms2_mzs, ms2_ints = get_averaged_ms2_spectrum_in_window(
            args.mzml, min_rt, max_rt, mz_min=mz_min, mz_max=mz_max, ppm_tolerance=20.0)
        if ms2_mzs is None or ms2_ints is None or len(ms2_mzs) == 0 or len(ms2_ints) == 0:
            skip_no_ms2 += 1
            rows_to_drop.append(idx)
            continue

        max_ms2 = float(np.max(ms2_ints))
        noise_floor = max_ms2 * args.min_frac if max_ms2 > 0 else 0.0
        ms2_mzs_arr = np.asarray(ms2_mzs)
        ms2_ints_arr = np.asarray(ms2_ints)

        all_theo = []
        for mz, label in c_list:
            all_theo.append((mz, label, _ion_name_to_comet_format(label)))
        for mz, label in z_list:
            all_theo.append((mz, label, _ion_name_to_comet_format(label)))
        for mz, label in z1_list:
            all_theo.append((mz, label, _ion_name_to_comet_format(label)))

        sig_ions, sig_mz = [], []
        for theo_mz, display_label, ion_name in all_theo:
            if theo_mz <= 0:
                continue
            tol = max(0.02, theo_mz * args.ppm * 1e-6)
            idx_mz = np.argmin(np.abs(ms2_mzs_arr - theo_mz))
            peak_mz = float(ms2_mzs_arr[idx_mz])
            peak_int = float(ms2_ints_arr[idx_mz])
            if np.abs(peak_mz - theo_mz) > tol:
                continue
            if peak_int < noise_floor:
                scatter_data.append((max_ms2, peak_int, False, idx, ion_name))
                continue
            if matches_b_or_y_ion(ion_name, peptide, peak_mz, args.ppm):
                continue
            sig_ions.append(ion_name)
            sig_mz.append(f'{peak_mz:.6f}')
            scatter_data.append((max_ms2, peak_int, True, idx, ion_name))

        if len(sig_ions) < args.min_sig_frags:
            rows_to_drop.append(idx)
        else:
            df.at[idx, 'significant_frags'] = ','.join(sig_ions)
            df.at[idx, 'significant_frags_mz'] = ','.join(sig_mz)
            pairs_str, overhang_str = _significant_pairs_and_overhangs_from_ions(sig_ions, peptide)
            df.at[idx, 'single_aa_overhang_fragment_pairs'] = pairs_str
            # Convert peptide positions to protein positions when sequence_start_pos available
            seq_start = row.get('sequence_start_pos') or row.get('sequence_positions')
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
            df.at[idx, 'single_aa_overhangs_protein_positions'] = overhang_str

    df = df.drop(index=rows_to_drop).reset_index(drop=True)
    n_after = len(df)

    n_passed = len([x for x in scatter_data if x[2]])
    n_failed = len([x for x in scatter_data if not x[2]])
    t_total = time.perf_counter() - t_start
    _log(f"[Significant fragmentation] Done in {t_total:.0f}s ({n_before / t_total:.1f} rows/s)")
    _log(f"[Significant fragmentation] Fragments: {n_passed} passed, {n_failed} below {args.min_frac*100:.2f}% threshold")
    _log(f"[Significant fragmentation] Rows: {n_before} -> {n_after} (removed {n_before - n_after})")
    if skip_no_rt or skip_no_peptide or skip_no_ms2:
        parts = []
        if skip_no_rt:
            parts.append(f"no RT window ({skip_no_rt})")
        if skip_no_peptide:
            parts.append(f"no peptide ({skip_no_peptide})")
        if skip_no_ms2:
            parts.append(f"no MS2 in window ({skip_no_ms2})")
        _log("[Significant fragmentation] Skipped: " + ", ".join(parts))

    # Diagnostic scatter
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
            max_all = max(m for m, f, p, _, _ in scatter_data) if scatter_data else 1
            ax.plot([0, max_all], [0, max_all * args.min_frac], color='0.6', linestyle='--', linewidth=2, label=f'{args.min_frac*100:.1f}% threshold')
            ax.set_xlabel('MS2 max signal (in window)', fontsize=12, color='0.85')
            ax.set_ylabel('Fragment signal', fontsize=12, color='0.85')
            ax.set_title('Significant fragmentation: 5 ppm + 1% max (summed MS2)', fontsize=14, color='0.9')
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
            diag_path = os.path.join(diag_dir, 'significant_frags_summed_ms2_scatter.png')
            fig.savefig(diag_path, dpi=150, bbox_inches='tight', facecolor='black')
            plt.close(fig)
            _log(f"[Significant fragmentation] Diagnostic: {os.path.basename(diag_path)}")
        except Exception as e:
            _log(f"[Significant fragmentation] Warning: Could not create scatter: {e}")

    _log(f"[Significant fragmentation] Writing: {os.path.abspath(out_csv)}")
    df.to_csv(out_csv, index=False)
    return 0


if __name__ == '__main__':
    sys.exit(main())
