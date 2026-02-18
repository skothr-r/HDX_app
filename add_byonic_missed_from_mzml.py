#!/usr/bin/env python3
"""
Extract MS1 data from mzML for Byonic peptides that Comet missed, and add them to a
CSV that matches the Comet/pipeline schema so they can be used in chromatograms and
downstream (e.g. merged with the main Comet CSV).

Inputs:
  - Byonic CSV (sequence, charge, Scan Time, Obs. m/z)
  - Comet CSV (to define "missed" = in Byonic but not in Comet)
  - mzML (same run)

Outputs:
  - byonic_only_peptides_from_mzml.csv: one row per Byonic PSM that was missed by Comet,
    with mzML-derived MS1 retention time and intensity, and columns aligned with Comet CSV.
  - Optional: --merge writes a combined CSV (Comet rows + Byonic-only rows) for use as
    the pipeline's main peptide list.

Usage:
  python add_byonic_missed_from_mzml.py data/byonic_search_peps.csv data/WT_nep2_0MUrea_08_with_qvalues_ms1_all.csv data/WT_nep2_0MUrea_08.mzML -o data/byonic_only_peptides_from_mzml.csv
  python add_byonic_missed_from_mzml.py data/byonic_search_peps.csv data/WT_nep2_0MUrea_08_with_qvalues_ms1_all.csv data/WT_nep2_0MUrea_08.mzML -o data/byonic_only_peptides_from_mzml.csv --merge -O data/WT_nep2_0MUrea_08_with_qvalues_ms1_all_plus_byonic_missed.csv
"""

import argparse
import os
import re
import sys

# Proton mass (monoisotopic) for neutral mass from m/z
PROTON_MASS = 1.00727647


def _load_byonic(byonic_path):
    import pandas as pd
    with open(byonic_path, 'r', encoding='utf-8', newline='') as f:
        raw = f.read()
    first_newline = raw.find('\n')
    header_line = raw[:first_newline].replace('\r\n', ' ').replace('\n', ' ')
    rest = raw[first_newline + 1:].lstrip('\r\n')
    df = pd.read_csv(
        __import__('io').StringIO(header_line + '\n' + rest),
        sep=',', engine='python', quotechar='"', on_bad_lines='warn'
    )
    df.columns = [str(c).replace('\r', ' ').replace('\n', ' ').strip() for c in df.columns]
    return df


def _byonic_plain_seq(s):
    import pandas as pd
    if pd.isna(s) or s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    if '.' in s:
        parts = s.split('.')
        if len(parts) >= 2:
            s = parts[1]
        else:
            s = s.replace('.', '')
    s = re.sub(r'\[\s*[+-]?\s*[\d.]+\s*\]', '', s)
    s = re.sub(r'\[\s*[+-]?\s*[\d.]*\s*$', '', s)
    return s if s else None


def _find_col(df, patterns):
    for p in patterns:
        for c in df.columns:
            if p.lower() in c.lower():
                return c
    return None


def _load_mzml(path):
    try:
        from pyopenms import MSExperiment, MzMLFile
    except ImportError:
        raise RuntimeError("pyopenms is required. Install with: pip install pyopenms (or use conda)")
    exp = MSExperiment()
    MzMLFile().load(path, exp)
    return exp


def _get_intensity_at_rt_mz(exp, rt_sec, mz_target, ppm=10, max_rt_diff_sec=30):
    best_spec = None
    best_diff = float('inf')
    for spec in exp:
        if spec.getMSLevel() != 1:
            continue
        d = abs(spec.getRT() - rt_sec)
        if d < best_diff and d <= max_rt_diff_sec:
            best_diff = d
            best_spec = spec
    if best_spec is None:
        return None, None
    tol = mz_target * ppm * 1e-6
    best_int = 0.0
    for peak in best_spec:
        if abs(peak.getMZ() - mz_target) <= tol:
            if peak.getIntensity() > best_int:
                best_int = peak.getIntensity()
    return (best_int if best_int > 0 else None), best_spec.getRT()


def _get_window_sum_at_mz(exp, rt_sec, mz_target, window_sec=30, ppm=10):
    min_rt = rt_sec - window_sec / 2
    max_rt = rt_sec + window_sec / 2
    tol = mz_target * ppm * 1e-6
    total = 0.0
    for spec in exp:
        if spec.getMSLevel() != 1:
            continue
        if not (min_rt <= spec.getRT() <= max_rt):
            continue
        for peak in spec:
            if abs(peak.getMZ() - mz_target) <= tol:
                total += peak.getIntensity()
                break
    return total if total > 0 else None


def main():
    ap = argparse.ArgumentParser(
        description='Extract mzML MS1 data for Byonic peptides missed by Comet and add to pipeline-style CSV'
    )
    ap.add_argument('byonic_csv', help='Byonic output CSV')
    ap.add_argument('comet_csv', help='Comet CSV (same run); used to define missed peptides')
    ap.add_argument('mzml', help='mzML file (same run)')
    ap.add_argument('-o', '--output', default='data/byonic_only_peptides_from_mzml.csv',
                    help='Output CSV of Byonic-only rows with mzML data')
    ap.add_argument('--merge', action='store_true',
                    help='Also write a merged CSV (Comet + Byonic-only)')
    ap.add_argument('-O', '--merged-output', default=None,
                    help='Path for merged CSV (default: comet CSV base + _plus_byonic_missed.csv)')
    ap.add_argument('--window-sec', type=float, default=30,
                    help='RT window (sec) for MS1 window sum (default 30)')
    args = ap.parse_args()

    import pandas as pd
    import numpy as np

    # ---- Load Byonic ----
    df_byonic = _load_byonic(args.byonic_csv)
    seq_col = _find_col(df_byonic, ['sequence', 'unformat']) or df_byonic.columns[2]
    z_col = 'z' if 'z' in df_byonic.columns else _find_col(df_byonic, ['z'])
    scan_time_col = next((c for c in df_byonic.columns if 'Scan' in c and 'Time' in c), None)
    obs_mz_col = next((c for c in df_byonic.columns if 'Obs' in c and 'm/z' in c), None) or next((c for c in df_byonic.columns if 'Calc' in c and 'm/z' in c), None)
    if not scan_time_col or not obs_mz_col:
        print("Error: Byonic CSV must have Scan Time and Obs. m/z (or Calc. m/z) columns. Columns:", list(df_byonic.columns)[:15], file=sys.stderr)
        sys.exit(1)

    df_byonic['_plain_seq'] = df_byonic[seq_col].apply(_byonic_plain_seq)
    df_byonic['_charge_int'] = pd.to_numeric(df_byonic[z_col], errors='coerce').astype('Int64')

    # ---- Load Comet: build set of (plain_peptide, charge) with contained match ----
    with open(args.comet_csv, 'r') as f:
        first = f.readline()
    skip = 1 if first.strip().startswith('CometVersion') or first.strip().startswith('#') else 0
    df_comet = pd.read_csv(args.comet_csv, skiprows=skip, sep=',', engine='python', quotechar='"', on_bad_lines='warn')
    if 'plain_peptide' not in df_comet.columns:
        print("Error: Comet CSV must have plain_peptide column", file=sys.stderr)
        sys.exit(1)
    comet_by_charge = {}
    for _, r in df_comet.iterrows():
        p = str(r['plain_peptide']).strip()
        if not p or pd.isna(r.get('charge')):
            continue
        z = int(r['charge'])
        if z not in comet_by_charge:
            comet_by_charge[z] = []
        comet_by_charge[z].append(p)

    def in_comet(seq, z):
        if z not in comet_by_charge:
            return False
        for p in comet_by_charge[z]:
            if p == seq or seq in p or p in seq:
                return True
        return False

    # ---- Collect Byonic PSMs that are missed by Comet ----
    missed_psms = []
    for _, row in df_byonic.iterrows():
        seq = row['_plain_seq']
        if seq is None or pd.isna(row['_charge_int']):
            continue
        z = int(row['_charge_int'])
        if in_comet(seq, z):
            continue
        rt_min = pd.to_numeric(row.get(scan_time_col), errors='coerce')
        obs_mz = pd.to_numeric(row.get(obs_mz_col), errors='coerce')
        if pd.isna(rt_min) or pd.isna(obs_mz) or obs_mz <= 0:
            continue
        rt_sec = float(rt_min) * 60.0
        missed_psms.append({
            'plain_peptide': seq,
            'charge': z,
            'rt_sec': rt_sec,
            'obs_mz': float(obs_mz),
        })
    if not missed_psms:
        print("No Byonic peptides missed by Comet (or no valid RT/mz in Byonic). Nothing to add.")
        return
    unique_missed = set((p['plain_peptide'], p['charge']) for p in missed_psms)
    print(f"Byonic PSMs missed by Comet: {len(missed_psms)} rows -> {len(unique_missed)} unique (sequence, charge)")

    # ---- Load mzML and extract MS1 data for each missed PSM ----
    print("Loading mzML...")
    exp = _load_mzml(args.mzml)
    print(f"  Loaded {exp.size()} spectra")
    rows_out = []
    for rec in missed_psms:
        intensity, rt_used = _get_intensity_at_rt_mz(exp, rec['rt_sec'], rec['obs_mz'])
        window_sum = _get_window_sum_at_mz(exp, rec['rt_sec'], rec['obs_mz'], window_sec=args.window_sec)
        z = rec['charge']
        obs_mz = rec['obs_mz']
        # Neutral mass from observed m/z: M = mz * z - z * proton
        calc_neutral_mass = obs_mz * z - z * PROTON_MASS
        rt_sec = rt_used if rt_used is not None else rec['rt_sec']
        rt_min = rt_sec / 60.0
        # Build row matching Comet CSV columns where possible; rest empty/NA
        row = {
            'sequence_positions': '',
            'plain_peptide': rec['plain_peptide'],
            'charge': z,
            'mz': obs_mz,
            'MS2_retention_time_sec': rt_sec,
            'MS1_retention_time_sec': rt_sec,
            'MS1_retention_time_min': rt_min,
            'MS1_retention_time_intensity': intensity if intensity is not None else '',
            'matched fragment ions': '',
            'single_aa_overhang_fragment_pairs': '',
            'matched fragment ion intensities': '',
            'matched fragment ion mz': '',
            'single_aa_overhangs_protein_positions': '',
            'comet_matched_frags_quality_scores': '',
            'scan': '',
            'num': '',
            'exp_neutral_mass': calc_neutral_mass,
            'calc_neutral_mass': calc_neutral_mass,
            'e-value': '',
            'xcorr': '',
            'delta_cn': '',
            'sp_score': '',
            'significant_frag_count': '',
            'ions_total': '',
            'modified_peptide': '',
            'prev_aa': '',
            'next_aa': '',
            'protein': '',
            'protein_count': '',
            'modifications': '-',
            'sp_rank': '',
            'percolator_qvalue': '',
            'percolator_PEP': '',
            'source': 'Byonic',
            'byonic_window_sum_ms1': window_sum if window_sum is not None else '',
        }
        rows_out.append(row)

    df_out = pd.DataFrame(rows_out)
    # Align with Comet CSV columns so merge is clean
    extra_cols = ['source', 'byonic_window_sum_ms1']
    output_cols = list(df_comet.columns) + [c for c in extra_cols if c not in df_comet.columns]
    for c in output_cols:
        if c not in df_out.columns:
            df_out[c] = ''
    df_out = df_out[[c for c in output_cols if c in df_out.columns]]
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df_out.to_csv(args.output, index=False)
    print(f"Wrote {len(df_out)} Byonic-only rows to {args.output}")

    if args.merge:
        out_merged = args.merged_output
        if not out_merged:
            base = os.path.splitext(args.comet_csv)[0]
            out_merged = base + '_plus_byonic_missed.csv'
        df_comet = df_comet.copy()
        for c in extra_cols:
            if c not in df_comet.columns:
                df_comet[c] = 'Comet' if c == 'source' else ''
        use_cols = [c for c in df_out.columns if c in df_comet.columns]
        df_comet_sub = df_comet[use_cols].copy()
        if 'source' in df_comet_sub.columns:
            df_comet_sub['source'] = 'Comet'
        df_merged = pd.concat([df_comet_sub, df_out[use_cols]], ignore_index=True)
        out_merged_dir = os.path.dirname(out_merged)
        if out_merged_dir:
            os.makedirs(out_merged_dir, exist_ok=True)
        df_merged.to_csv(out_merged, index=False)
        print(f"Merged CSV (Comet + Byonic-only): {out_merged} ({len(df_comet)} Comet + {len(df_out)} Byonic = {len(df_merged)} rows)")


if __name__ == '__main__':
    main()
