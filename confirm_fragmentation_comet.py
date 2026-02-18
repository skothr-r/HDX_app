#!/usr/bin/env python3
"""
Confirm fragmentation pattern for filtered peptides by re-running Comet and comparing results.

This script:
1. (Optional) Re-runs Comet on the same mzML/FASTA with the same params.
2. Loads the filtered peptides CSV (e.g. final_filtered_peptides.csv) and the Comet rerun CSV.
3. For each PSM (scan, plain_peptide, charge), finds the same in the rerun CSV and compares:
   - significant fragments
   - xcorr, significant fragment count (optional)
4. Writes a report CSV with status: match, mismatch (fragmentation differs), or missing_in_rerun.

Usage:
  # Run Comet then compare (recommended)
  python confirm_fragmentation_comet.py \\
    --filtered-csv results_v50/final_filtered_peptides.csv \\
    --mzml data/WT_nep2_0MUrea_08.mzML \\
    --fasta data/Ube2D3.fasta \\
    --params comet.params.new \\
    --run-comet \\
    --output results_v50/fragmentation_confirm_report.csv

  # Compare against an existing rerun CSV (no Comet run)
  python confirm_fragmentation_comet.py \\
    --filtered-csv results_v50/final_filtered_peptides.csv \\
    --rerun-csv data/WT_nep2_0MUrea_08.csv \\
    --output results_v50/fragmentation_confirm_report.csv
"""

import argparse
import os
import sys
import pandas as pd


def _norm_mods(m):
    if m is None or (isinstance(m, float) and pd.isna(m)):
        return '-'
    s = str(m).strip()
    return s if s and s.lower() != 'nan' else '-'


def _read_csv(path, skip_comet_header=True):
    with open(path, 'r') as f:
        first = f.readline()
    skip = 1 if skip_comet_header and 'CometVersion' in first else 0
    return pd.read_csv(path, sep=',', skiprows=skip, engine='python', quotechar='"', on_bad_lines='warn')


def run_comet(mzml_file, fasta_file, params_file, comet_exe=None):
    """Run Comet search; return path to output CSV or None."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        from run_comet_with_percolator import run_comet as _run
        out = _run(mzml_file, fasta_file, params_file, comet_exe)
        return out['csv'] if out else None
    except Exception as e:
        print(f"Error running Comet: {e}", file=sys.stderr)
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Confirm fragmentation for filtered peptides by re-running Comet and comparing.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--filtered-csv', required=True,
                        help='Filtered peptides CSV (e.g. final_filtered_peptides.csv)')
    parser.add_argument('--rerun-csv',
                        help='Comet rerun output CSV. Required unless --run-comet is set.')
    parser.add_argument('--run-comet', action='store_true',
                        help='Run Comet first; use --mzml, --fasta, --params. Rerun CSV path is then inferred.')
    parser.add_argument('--mzml', help='mzML input (required if --run-comet)')
    parser.add_argument('--fasta', help='FASTA database (required if --run-comet)')
    parser.add_argument('--params', help='Comet params file (required if --run-comet)')
    parser.add_argument('--comet-exe', help='Path to Comet executable')
    parser.add_argument('--output', default=None,
                        help='Report CSV path (default: fragmentation_confirm_report.csv in same dir as filtered-csv)')
    args = parser.parse_args()

    if not os.path.exists(args.filtered_csv):
        print(f"Error: Filtered CSV not found: {args.filtered_csv}", file=sys.stderr)
        sys.exit(1)

    rerun_csv = args.rerun_csv
    if args.run_comet:
        if not all([args.mzml, args.fasta, args.params]):
            print("Error: --run-comet requires --mzml, --fasta, --params", file=sys.stderr)
            sys.exit(1)
        if not os.path.exists(args.mzml):
            print(f"Error: mzML not found: {args.mzml}", file=sys.stderr)
            sys.exit(1)
        if not os.path.exists(args.fasta):
            print(f"Error: FASTA not found: {args.fasta}", file=sys.stderr)
            sys.exit(1)
        if not os.path.exists(args.params):
            print(f"Error: Params file not found: {args.params}", file=sys.stderr)
            sys.exit(1)
        print("Running Comet...")
        rerun_csv = run_comet(args.mzml, args.fasta, args.params, args.comet_exe)
        if not rerun_csv or not os.path.exists(rerun_csv):
            print("Error: Comet did not produce a CSV.", file=sys.stderr)
            sys.exit(1)
        print(f"Comet output: {rerun_csv}")
    else:
        if not rerun_csv or not os.path.exists(rerun_csv):
            print("Error: --rerun-csv is required when not using --run-comet.", file=sys.stderr)
            sys.exit(1)

    # Output report path
    if args.output is None:
        base_dir = os.path.dirname(os.path.abspath(args.filtered_csv))
        args.output = os.path.join(base_dir, 'fragmentation_confirm_report.csv')
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or '.', exist_ok=True)

    df_filtered = _read_csv(args.filtered_csv)
    df_rerun = _read_csv(rerun_csv)

    for col in ['plain_peptide', 'charge', 'scan']:
        if col not in df_filtered.columns:
            print(f"Error: filtered-csv missing column '{col}'", file=sys.stderr)
            sys.exit(1)
        if col not in df_rerun.columns:
            print(f"Error: rerun-csv missing column '{col}'", file=sys.stderr)
            sys.exit(1)

    sig_col = None
    for c in ['significant_frags', 'refined_significant_frags', 'significant_fragment_ions']:
        if c in df_filtered.columns and c in df_rerun.columns:
            sig_col = c
            break
    if sig_col is None:
        print("Warning: no shared significant-fragment column found between filtered and rerun CSVs", file=sys.stderr)

    def _frag_count(v):
        if pd.isna(v) or v is None:
            return ''
        s = str(v).strip()
        if not s:
            return ''
        return len([x.strip() for x in s.split(',') if x.strip()])

    # Index rerun by (scan, plain_peptide, charge)
    df_rerun['_scan'] = df_rerun['scan'].astype(int)
    df_rerun['_seq'] = df_rerun['plain_peptide'].astype(str).str.strip()
    df_rerun['_ch'] = df_rerun['charge'].fillna(0).astype(int)
    rerun_key_to_row = {}
    for idx, row in df_rerun.iterrows():
        key = (int(row['_scan']), row['_seq'], int(row['_ch']))
        rerun_key_to_row[key] = row

    # Compare each filtered row
    report_rows = []
    for idx, row in df_filtered.iterrows():
        scan = int(row['scan']) if pd.notna(row['scan']) else None
        seq = str(row['plain_peptide']).strip()
        ch = int(row['charge']) if pd.notna(row['charge']) else 0
        if scan is None:
            report_rows.append({
                'scan': row['scan'],
                'plain_peptide': seq,
                'charge': ch,
                'status': 'no_scan',
                'significant_frags_original': row.get(sig_col, '') if sig_col else '',
                'significant_frags_rerun': '',
                'xcorr_original': row.get('xcorr', ''),
                'xcorr_rerun': '',
                'significant_frag_count_original': _frag_count(row.get(sig_col, '')) if sig_col else '',
                'significant_frag_count_rerun': '',
            })
            continue
        key = (scan, seq, ch)
        rerun_row = rerun_key_to_row.get(key)
        if rerun_row is None:
            report_rows.append({
                'scan': scan,
                'plain_peptide': seq,
                'charge': ch,
                'status': 'missing_in_rerun',
                'significant_frags_original': row.get(sig_col, '') if sig_col else '',
                'significant_frags_rerun': '',
                'xcorr_original': row.get('xcorr', ''),
                'xcorr_rerun': '',
                'significant_frag_count_original': _frag_count(row.get(sig_col, '')) if sig_col else '',
                'significant_frag_count_rerun': '',
            })
            continue
        orig_ions = str(row.get(sig_col, '')).strip() if sig_col and pd.notna(row.get(sig_col)) else ''
        rerun_ions = str(rerun_row.get(sig_col, '')).strip() if sig_col and pd.notna(rerun_row.get(sig_col)) else ''
        orig_set = set(x.strip() for x in orig_ions.split(',') if x.strip())
        rerun_set = set(x.strip() for x in rerun_ions.split(',') if x.strip())
        if orig_set == rerun_set:
            status = 'match'
        else:
            status = 'mismatch'
        report_rows.append({
            'scan': scan,
            'plain_peptide': seq,
            'charge': ch,
            'status': status,
            'significant_frags_original': orig_ions,
            'significant_frags_rerun': rerun_ions,
            'xcorr_original': row.get('xcorr', ''),
            'xcorr_rerun': rerun_row.get('xcorr', ''),
            'significant_frag_count_original': _frag_count(orig_ions),
            'significant_frag_count_rerun': _frag_count(rerun_ions),
        })

    df_report = pd.DataFrame(report_rows)
    df_report.to_csv(args.output, index=False)
    n_match = (df_report['status'] == 'match').sum()
    n_mismatch = (df_report['status'] == 'mismatch').sum()
    n_missing = (df_report['status'] == 'missing_in_rerun').sum()
    n_no_scan = (df_report['status'] == 'no_scan').sum()
    print(f"Report: {os.path.abspath(args.output)}")
    print(f"  Total PSMs: {len(df_report)}")
    print(f"  match: {n_match}")
    print(f"  mismatch (fragmentation differs): {n_mismatch}")
    print(f"  missing_in_rerun: {n_missing}")
    print(f"  no_scan: {n_no_scan}")
    if n_mismatch or n_missing:
        sys.exit(1)
    sys.exit(0)


if __name__ == '__main__':
    main()
