#!/usr/bin/env python3
"""
Compare Byonic vs Comet peptides with metrics: signal, fragmentation, scores.
Reports inconsistencies (e.g. Byonic peptides missed by Comet and their Byonic metrics;
Comet metrics for peptides found in both).

Usage:
  python compare_byonic_comet_metrics.py data/byonic_search_peps.csv data/WT_nep2_0MUrea_08_with_qvalues_ms1_all.csv
  python compare_byonic_comet_metrics.py data/byonic_search_peps.csv data/WT_nep2_0MUrea_08_with_qvalues_ms1_all.csv --chrom-csv plots_v82_chromatograms_rt-windows_combined/peptide_chromatograms_primary/peak_windows_all.csv
"""

import argparse
import os
import sys

def _load_byonic(byonic_path):
    import pandas as pd
    with open(byonic_path, 'r', encoding='utf-8', newline='') as f:
        raw = f.read()
    first_newline = raw.find('\n')
    header_line = raw[:first_newline].replace('\r\n', ' ').replace('\n', ' ')
    rest = raw[first_newline + 1:].lstrip('\r\n')
    df = pd.read_csv(pd.io.common.StringIO(header_line + '\n' + rest), sep=',', engine='python', quotechar='"', on_bad_lines='warn')
    df.columns = [str(c).replace('\r', ' ').replace('\n', ' ').strip() for c in df.columns]
    return df

def _byonic_plain_seq(s):
    """Extract plain peptide from Byonic: remove flanking X., strip modification brackets [+mass] so comparison matches Comet plain_peptide + mods."""
    import re
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
    # Strip Byonic modification notation: full [+15.99492] or truncated [+15 (column width in export)
    s = re.sub(r'\[\s*[+-]?\s*[\d.]+\s*\]', '', s)
    s = re.sub(r'\[\s*[+-]?\s*[\d.]*\s*$', '', s)  # trailing incomplete e.g. [+15
    return s if s else None

def _find_col(df, patterns):
    for p in patterns:
        for c in df.columns:
            if p.lower() in c.lower():
                return c
    return None

def main():
    ap = argparse.ArgumentParser(description='Compare Byonic vs Comet metrics')
    ap.add_argument('byonic_csv', help='Byonic output CSV')
    ap.add_argument('comet_csv', help='Comet CSV (with plain_peptide, charge, ions_matched, xcorr, etc.)')
    ap.add_argument('--chrom-csv', help='Optional: chromatogram CSV with total_area, apex_intensity (e.g. peak_windows_all.csv)')
    ap.add_argument('-o', '--output', help='Output CSV path for full comparison table')
    args = ap.parse_args()

    import pandas as pd
    import numpy as np

    # ---- Load Byonic ----
    df_byonic = _load_byonic(args.byonic_csv)
    seq_col = _find_col(df_byonic, ['sequence', 'unformat']) or df_byonic.columns[2]
    z_col = 'z' if 'z' in df_byonic.columns else _find_col(df_byonic, ['z'])
    score_col = _find_col(df_byonic, ['Score', 'score'])
    pep_col = _find_col(df_byonic, ['PEP 2D', 'PEP 1D', 'pep'])
    qval_col = _find_col(df_byonic, ['q-value 2D', 'q-value 1D', 'q-value'])
    scan_time_col = _find_col(df_byonic, ['Scan Time', 'Scan Time'])
    frag_type_col = _find_col(df_byonic, ['Fragment Type', 'Frag'])

    df_byonic['_plain_seq'] = df_byonic[seq_col].apply(_byonic_plain_seq)
    df_byonic['_charge_int'] = pd.to_numeric(df_byonic[z_col], errors='coerce').astype('Int64')
    byonic_agg = []
    for (seq, z), g in df_byonic.groupby(['_plain_seq', '_charge_int'], dropna=True):
        if seq is None or pd.isna(z) or z is None:
            continue
        z = int(z)
        row = {
            'sequence': seq,
            'charge': z,
            'seq_len': len(seq),
            'n_psms_byonic': len(g),
        }
        if score_col and score_col in g.columns:
            row['byonic_score_max'] = g[score_col].astype(float, errors='ignore').max()
            row['byonic_score_median'] = g[score_col].astype(float, errors='ignore').median()
        if pep_col and pep_col in g.columns:
            vals = pd.to_numeric(g[pep_col], errors='coerce').dropna()
            if len(vals) > 0:
                row['byonic_pep_min'] = vals.min()
                row['byonic_pep_median'] = vals.median()
        if qval_col and qval_col in g.columns:
            vals = pd.to_numeric(g[qval_col], errors='coerce').dropna()
            if len(vals) > 0:
                row['byonic_qvalue_min'] = vals.min()
        if scan_time_col and scan_time_col in g.columns:
            vals = pd.to_numeric(g[scan_time_col], errors='coerce').dropna()
            if len(vals) > 0:
                row['byonic_scan_time_min'] = vals.min()
        if frag_type_col and frag_type_col in g.columns:
            row['byonic_frag_types'] = ','.join(sorted(g[frag_type_col].dropna().astype(str).unique()))
        byonic_agg.append(row)
    df_byonic_agg = pd.DataFrame(byonic_agg)
    print(f"Byonic: {len(df_byonic_agg)} unique (sequence, charge) peptides from {len(df_byonic)} rows")

    # ---- Load Comet ----
    with open(args.comet_csv, 'r') as f:
        first = f.readline()
    skip = 1 if first.strip().startswith('CometVersion') or first.strip().startswith('#') else 0
    df_comet = pd.read_csv(args.comet_csv, sep=',', skiprows=skip, engine='python', quotechar='"', on_bad_lines='warn')
    if 'plain_peptide' not in df_comet.columns:
        print("Error: Comet CSV must have plain_peptide column", file=sys.stderr)
        sys.exit(1)

    comet_agg = []
    df_comet['_charge_int'] = df_comet['charge'].astype(int)
    for (p, z), g in df_comet.groupby(['plain_peptide', '_charge_int']):
        row = {
            'plain_peptide': p,
            'charge': int(z),
            'n_psms_comet': len(g),
        }
        if 'ions_matched' in g.columns:
            row['comet_ions_matched_max'] = pd.to_numeric(g['ions_matched'], errors='coerce').max()
            row['comet_ions_matched_median'] = pd.to_numeric(g['ions_matched'], errors='coerce').median()
        if 'ions_total' in g.columns:
            row['comet_ions_total_max'] = pd.to_numeric(g['ions_total'], errors='coerce').max()
        if 'xcorr' in g.columns:
            row['comet_xcorr_max'] = pd.to_numeric(g['xcorr'], errors='coerce').max()
        if 'e-value' in g.columns:
            row['comet_evalue_min'] = pd.to_numeric(g['e-value'], errors='coerce').min()
        if 'MS1_retention_time_intensity' in g.columns:
            row['comet_ms1_intensity_max'] = pd.to_numeric(g['MS1_retention_time_intensity'], errors='coerce').max()
            row['comet_ms1_intensity_sum'] = pd.to_numeric(g['MS1_retention_time_intensity'], errors='coerce').sum()
        if 'percolator_qvalue' in g.columns:
            row['comet_qvalue_min'] = pd.to_numeric(g['percolator_qvalue'], errors='coerce').min()
        if 'percolator_PEP' in g.columns:
            row['comet_pep_min'] = pd.to_numeric(g['percolator_PEP'], errors='coerce').min()
        comet_agg.append(row)
    df_comet_agg = pd.DataFrame(comet_agg)
    comet_by_charge = {}
    for _, r in df_comet_agg.iterrows():
        z = int(r['charge'])
        if z not in comet_by_charge:
            comet_by_charge[z] = []
        comet_by_charge[z].append(r)

    # ---- Match Byonic to Comet (contained) ----
    def in_comet(seq, z):
        if z not in comet_by_charge:
            return None
        for r in comet_by_charge[z]:
            p = r['plain_peptide']
            if p == seq or seq in p or p in seq:
                return r
        return None

    rows_out = []
    for _, b in df_byonic_agg.iterrows():
        seq, z = b['sequence'], int(b['charge'])
        match = in_comet(seq, z)
        in_comet_flag = match is not None
        row = {
            'sequence': seq,
            'charge': z,
            'seq_len': int(b['seq_len']),
            'in_comet': in_comet_flag,
            'n_psms_byonic': int(b['n_psms_byonic']),
            'byonic_score_max': b.get('byonic_score_max'),
            'byonic_pep_min': b.get('byonic_pep_min'),
            'byonic_qvalue_min': b.get('byonic_qvalue_min'),
            'byonic_scan_time_min': b.get('byonic_scan_time_min'),
            'byonic_frag_types': b.get('byonic_frag_types'),
        }
        if match is not None:
            row['comet_plain_peptide_matched'] = match['plain_peptide']
            row['n_psms_comet'] = int(match['n_psms_comet'])
            for k in ['comet_ions_matched_max', 'comet_xcorr_max', 'comet_ms1_intensity_max', 'comet_ms1_intensity_sum',
                      'comet_qvalue_min', 'comet_pep_min', 'comet_evalue_min']:
                if k in match and pd.notna(match[k]):
                    row[k] = match[k]
        rows_out.append(row)

    # ---- Optional: chromatogram metrics (total_area, apex_intensity) ----
    chrom_joined = False
    if args.chrom_csv and os.path.exists(args.chrom_csv):
        df_chrom = pd.read_csv(args.chrom_csv, sep=',', engine='python', quotechar='"', on_bad_lines='warn')
        if 'total_area' in df_chrom.columns and 'apex_intensity' in df_chrom.columns and 'charge' in df_chrom.columns:
            pep_col_chrom = 'peptide' if 'peptide' in df_chrom.columns else ('plain_peptide' if 'plain_peptide' in df_chrom.columns else None)
            if pep_col_chrom:
                chrom_list = []
                for _, r in df_chrom.iterrows():
                    p = str(r[pep_col_chrom]).strip()
                    z = int(r['charge'])
                    chrom_list.append((p, z, r['total_area'], r['apex_intensity']))
                for r in rows_out:
                    seq, z = r['sequence'], r['charge']
                    for (pk, pz, ta, ap) in chrom_list:
                        if pz != z:
                            continue
                        if pk == seq or (seq in pk) or (pk in seq):
                            r['comet_total_area'] = ta
                            r['comet_apex_intensity'] = ap
                            chrom_joined = True
                            break
        if chrom_joined:
            print(f"Joined chromatogram metrics from {args.chrom_csv}")
        else:
            print(f"Warning: Could not join chrom CSV (missing columns or key); total_area/apex_intensity may be absent")
    else:
        if args.chrom_csv:
            print(f"Warning: Chrom CSV not found: {args.chrom_csv}")

    df_out = pd.DataFrame(rows_out)

    # ---- Summary: in Comet vs missed ----
    found = df_out[df_out['in_comet'] == True]
    missed = df_out[df_out['in_comet'] == False]
    print(f"\n--- Byonic peptides: {len(found)} in Comet, {len(missed)} MISSED by Comet ---\n")

    print("BYONIC metrics (Byonic-side only):")
    print("  In Comet:")
    if len(found) > 0:
        print(f"    Score (max):     median = {found['byonic_score_max'].median():.1f}, mean = {found['byonic_score_max'].mean():.1f}")
        if found['byonic_pep_min'].notna().any():
            print(f"    PEP (min):       median = {found['byonic_pep_min'].median():.2e}, mean = {found['byonic_pep_min'].mean():.2e}")
        print(f"    Charge:          {found['charge'].value_counts().sort_index().to_dict()}")
        print(f"    Sequence length: median = {found['seq_len'].median():.0f}, mean = {found['seq_len'].mean():.1f}")
    print("  MISSED by Comet:")
    if len(missed) > 0:
        print(f"    Score (max):     median = {missed['byonic_score_max'].median():.1f}, mean = {missed['byonic_score_max'].mean():.1f}")
        if missed['byonic_pep_min'].notna().any():
            print(f"    PEP (min):       median = {missed['byonic_pep_min'].median():.2e}, mean = {missed['byonic_pep_min'].mean():.2e}")
        print(f"    Charge:          {missed['charge'].value_counts().sort_index().to_dict()}")
        print(f"    Sequence length: median = {missed['seq_len'].median():.0f}, mean = {missed['seq_len'].mean():.1f}")

    print("\nCOMET metrics (only for peptides that ARE in Comet):")
    if len(found) > 0:
        if 'comet_ions_matched_max' in found.columns and found['comet_ions_matched_max'].notna().any():
            print(f"    Ions matched (max):  median = {found['comet_ions_matched_max'].median():.0f}, mean = {found['comet_ions_matched_max'].mean():.1f}")
        if 'comet_xcorr_max' in found.columns and found['comet_xcorr_max'].notna().any():
            print(f"    XCorr (max):         median = {found['comet_xcorr_max'].median():.3f}, mean = {found['comet_xcorr_max'].mean():.3f}")
        if 'comet_ms1_intensity_max' in found.columns and found['comet_ms1_intensity_max'].notna().any():
            print(f"    MS1 intensity (max): median = {found['comet_ms1_intensity_max'].median():.2e}, mean = {found['comet_ms1_intensity_max'].mean():.2e}")
        if 'comet_total_area' in found.columns and found['comet_total_area'].notna().any():
            print(f"    Total area (max):    median = {found['comet_total_area'].median():.2e}, mean = {found['comet_total_area'].mean():.2e}")
        if 'comet_apex_intensity' in found.columns and found['comet_apex_intensity'].notna().any():
            print(f"    Apex intensity:     median = {found['comet_apex_intensity'].median():.2e}, mean = {found['comet_apex_intensity'].mean():.2e}")

    print("\nMISSED peptides (no Comet metrics by definition):")
    if len(missed) > 0:
        print("  Sequence, charge, seq_len, byonic_score_max, byonic_pep_min, byonic_frag_types")
        for _, r in missed.sort_values(['sequence', 'charge']).iterrows():
            pep_str = f"  {r['byonic_pep_min']:.2e}" if pd.notna(r.get('byonic_pep_min')) else "  —"
            print(f"  {r['sequence'][:45]:45} z={r['charge']} len={r['seq_len']} score={r.get('byonic_score_max')} pep={pep_str} frag={str(r.get('byonic_frag_types',''))[:20]}")

    if args.output:
        df_out.to_csv(args.output, index=False)
        print(f"\nWrote full comparison table to {args.output}")

    return df_out

if __name__ == '__main__':
    main()
