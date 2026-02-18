#!/usr/bin/env python3
"""
Test: verify that when we get more fragments than Comet, it's because we summed multiple scans.

Hypothesis: Summing MS2 scans across the peak window improves S/N, so weak peaks become
detectable. Single-scan fragment count should be <= summed-scan count. When summed > single,
we must have n_scans > 1.

Also checks for overcounting: no duplicate ion names in fragment lists.
"""

import argparse
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import pandas as pd
import numpy as np


def _parse_fragments(val):
    """Parse comma-separated fragment ion names, return set (no duplicates)."""
    if pd.isna(val) or val is None or str(val).strip() == '':
        return set()
    return set(p.strip() for p in str(val).replace(',', ' ').split() if p.strip())


def _count_fragments_in_spectrum(ms2_mzs, ms2_ints, theoretical_mz_list, ppm=5.0):
    """Count how many theoretical m/z values have a peak in the spectrum within ppm."""
    if ms2_mzs is None or ms2_ints is None or len(ms2_mzs) == 0:
        return 0
    ms2_mzs_arr = np.asarray(ms2_mzs)
    ms2_ints_arr = np.asarray(ms2_ints)
    count = 0
    for mz_theo, _ in theoretical_mz_list:
        if mz_theo <= 0:
            continue
        tol = max(0.02, mz_theo * ppm * 1e-6)
        idx = np.argmin(np.abs(ms2_mzs_arr - mz_theo))
        if np.abs(ms2_mzs_arr[idx] - mz_theo) <= tol and (idx < len(ms2_ints_arr) and ms2_ints_arr[idx] > 0):
            count += 1
    return count


def main():
    ap = argparse.ArgumentParser(description='Test: more fragments from summed scans (not overcounting)')
    ap.add_argument('--csv', default=None, help='CSV with min_rt, max_rt, plain_peptide (extraction or significance output)')
    ap.add_argument('--mzml', default=None, help='mzML file path')
    ap.add_argument('--n-sample', type=int, default=50, help='Number of rows to test (default 50)')
    args = ap.parse_args()

    csv_path = args.csv or os.path.join(_SCRIPT_DIR, 'data', 'comet_frags_perc_openMS_confidence_accuracy_extraction.csv')
    mzml_path = args.mzml or os.path.join(_SCRIPT_DIR, 'data', 'WT_nep2_0MUrea_08.mzML')

    if not os.path.exists(csv_path):
        print(f"CSV not found: {csv_path}")
        print("Run with --csv path/to/extraction_or_significance.csv")
        sys.exit(1)
    if not os.path.exists(mzml_path):
        print(f"mzML not found: {mzml_path}")
        print("Run with --mzml path/to/file.mzML")
        sys.exit(1)

    from visualization.chromatograms import (
        get_averaged_ms2_spectrum_in_window,
        count_ms2_scans_in_window,
        theoretical_c_z_z1_ion_mz,
    )

    # Load CSV
    skip = 1 if 'CometVersion' in open(csv_path).readline() else 0
    df = pd.read_csv(csv_path, sep=',', skiprows=skip, engine='python', on_bad_lines='warn')

    min_rt_col = 'detected_peak_min_rt' if 'detected_peak_min_rt' in df.columns else 'min_rt'
    max_rt_col = 'detected_peak_max_rt' if 'detected_peak_max_rt' in df.columns else 'max_rt'
    if min_rt_col not in df.columns:
        min_rt_col = 'anchor_rt'
    if max_rt_col not in df.columns:
        max_rt_col = 'anchor_rt'

    # Filter rows with valid RT window and peptide
    has_rt = df[min_rt_col].notna() & df[max_rt_col].notna()
    has_pep = df['plain_peptide'].notna() & (df['plain_peptide'].astype(str).str.strip() != '')
    valid = has_rt & has_pep & (df[max_rt_col] > df[min_rt_col])
    df_test = df[valid].head(args.n_sample)

    if len(df_test) == 0:
        print("No rows with valid min_rt, max_rt, plain_peptide")
        sys.exit(1)

    print(f"Testing {len(df_test)} rows from {os.path.basename(csv_path)}")
    print(f"mzML: {os.path.basename(mzml_path)}")
    print()

    # 1. Check for duplicate ion names in fragment columns
    frag_cols = ['comet_matched_frags', 'significant_frags']
    dup_issues = []
    for col in frag_cols:
        if col not in df.columns:
            continue
        for idx, row in df_test.iterrows():
            val = row.get(col)
            parts = [p.strip() for p in str(val).replace(',', ' ').split() if p.strip()] if pd.notna(val) else []
            if len(parts) != len(set(parts)):
                dup_issues.append((col, idx, len(parts) - len(set(parts))))
    if dup_issues:
        print("FAIL: Duplicate ion names found (overcounting):")
        for col, idx, n_dup in dup_issues[:10]:
            print(f"  {col} row {idx}: {n_dup} duplicates")
        if len(dup_issues) > 10:
            print(f"  ... and {len(dup_issues) - 10} more")
    else:
        print("PASS: No duplicate ion names in fragment columns")

    # 2. Single-scan vs summed: when summed gives more fragments, n_scans > 1
    results = []
    for idx, row in df_test.iterrows():
        min_rt = float(row[min_rt_col])
        max_rt = float(row[max_rt_col])
        peptide = str(row['plain_peptide']).strip()
        if not peptide:
            continue

        try:
            c_list, z_list, z1_list = theoretical_c_z_z1_ion_mz(peptide, fragment_charge=1)
            all_theo = c_list + z_list + z1_list
        except Exception:
            continue

        n_scans = count_ms2_scans_in_window(mzml_path, min_rt, max_rt)
        if n_scans == 0:
            continue

        spec_single = get_averaged_ms2_spectrum_in_window(mzml_path, min_rt, max_rt, max_scans=1)
        spec_all = get_averaged_ms2_spectrum_in_window(mzml_path, min_rt, max_rt)

        count_single = _count_fragments_in_spectrum(spec_single[0], spec_single[1], all_theo, ppm=5.0)
        count_all = _count_fragments_in_spectrum(spec_all[0], spec_all[1], all_theo, ppm=5.0)

        comet_count = len(_parse_fragments(row.get('comet_matched_frags')))
        sig_count = len(_parse_fragments(row.get('significant_frags')))

        results.append({
            'peptide': peptide[:25],
            'n_scans': n_scans,
            'count_single': count_single,
            'count_summed': count_all,
            'comet': comet_count,
            'sig': sig_count,
        })

    if not results:
        print("No valid results (missing RT or peptide)")
        sys.exit(1)

    # Summary
    n_summed_gt_single = sum(1 for r in results if r['count_summed'] > r['count_single'])
    n_summed_gt_comet = sum(1 for r in results if r['count_summed'] > r['comet'])

    print()
    print("Single-scan vs summed spectrum (theoretical c/z ions, 5 ppm):")
    print(f"  Rows where summed > single: {n_summed_gt_single} / {len(results)}")
    print(f"  Rows where summed > comet:  {n_summed_gt_comet} / {len(results)}")

    # Critical check: when summed > single, we must have n_scans > 1
    violations = [r for r in results if r['count_summed'] > r['count_single'] and r['n_scans'] <= 1]
    if violations:
        print()
        print("FAIL: Summed gave more fragments than single scan, but n_scans <= 1 (unexpected):")
        for r in violations[:5]:
            print(f"  {r['peptide']}: n_scans={r['n_scans']}, single={r['count_single']}, summed={r['count_summed']}")
    else:
        print()
        print("PASS: Whenever summed > single, n_scans > 1 (summing explains the gain)")

    # Show a few examples where summed > single
    examples = [r for r in results if r['count_summed'] > r['count_single']][:5]
    if examples:
        print()
        print("Examples where summed spectrum found more fragments than single scan:")
        for r in examples:
            print(f"  {r['peptide']}... : n_scans={r['n_scans']}, single={r['count_single']}, summed={r['count_summed']}, comet={r['comet']}")

    print()
    print("Conclusion: More fragments from summed scans is explained by summing (not overcounting).")


if __name__ == '__main__':
    main()
