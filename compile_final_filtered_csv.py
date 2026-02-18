#!/usr/bin/env python3
"""
Compile the final CSV of filtered peptides with EVERY column, including integration and collection windows.

Workflow:
  1. Start with the full Comet CSV (e.g. data/WT_nep2_0MUrea_08_with_ms1_perc_qvalues.csv).
  2. Add integration and collection window columns by passing a CSV that has them (e.g. filtered_peptides_with_windows.csv).
  3. Keep only rows for peptides present in that filtered list.
  4. Output = filtered rows with all columns from the full CSV plus integration/collection window columns.

So: full CSV + (filter list and window columns from filtered_peptides_with_windows.csv) -> final CSV with every column including detected_peak_min_rt, detected_peak_max_rt, collection_min_rt, collection_max_rt.

Usage:
  # Add integration/collection windows from filtered_peptides_with_windows, keep only those peptides:
  python compile_final_filtered_csv.py \\
    --full-csv data/WT_nep2_0MUrea_08_with_ms1_perc_qvalues.csv \\
    --filtered-csv results_v50/filtered_peptides_with_windows.csv \\
    --output results_v50/final_filtered_peptides.csv

  # No filter, just copy full CSV (e.g. if it already has all columns):
  python compile_final_filtered_csv.py --full-csv results_v50/filtered_peptides.csv --output results_v50/final_filtered_peptides.csv

If --filtered-csv is omitted, --full-csv is written to --output as-is.
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


def main():
    parser = argparse.ArgumentParser(
        description='Compile final filtered peptides CSV with every column from the full source.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--full-csv', required=True,
                        help='Full Comet CSV (with all columns: comet, percolator, peak windows, etc.)')
    parser.add_argument('--filtered-csv',
                        help='Filtered peptides CSV (plain_peptide, charge, modifications). If omitted, full-csv is copied to output.')
    parser.add_argument('--output', default=None,
                        help='Output CSV path (default: same dir as full-csv, name final_filtered_peptides.csv)')
    args = parser.parse_args()

    if not os.path.exists(args.full_csv):
        print(f"Error: Full CSV not found: {args.full_csv}", file=sys.stderr)
        sys.exit(1)

    # Output path
    if args.output is None:
        base_dir = os.path.dirname(os.path.abspath(args.full_csv))
        args.output = os.path.join(base_dir, 'final_filtered_peptides.csv')
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or '.', exist_ok=True)

    # Read full CSV (skip Comet version line if present)
    with open(args.full_csv, 'r') as f:
        first_line = f.readline()
    skip_rows = 1 if 'CometVersion' in first_line else 0
    df_full = pd.read_csv(args.full_csv, sep=',', skiprows=skip_rows, engine='python', quotechar='"', on_bad_lines='warn')

    if 'plain_peptide' not in df_full.columns:
        print("Error: full-csv must contain column 'plain_peptide'", file=sys.stderr)
        sys.exit(1)

    if args.filtered_csv is None:
        # No filter: use full CSV as-is (all rows, all columns)
        df_out = df_full
        print(f"Using full CSV as-is ({len(df_out)} rows, {len(df_out.columns)} columns)")
    elif os.path.abspath(args.filtered_csv) == os.path.abspath(args.full_csv):
        df_out = df_full
        print(f"Full and filtered CSV are the same; writing all rows ({len(df_out)} rows, {len(df_out.columns)} columns)")
    else:
        if not os.path.exists(args.filtered_csv):
            print(f"Error: Filtered CSV not found: {args.filtered_csv}", file=sys.stderr)
            sys.exit(1)
        df_filtered = pd.read_csv(args.filtered_csv, sep=',', skiprows=1 if 'CometVersion' in open(args.filtered_csv).readline() else 0,
                                  engine='python', quotechar='"', on_bad_lines='warn')
        if 'plain_peptide' not in df_filtered.columns:
            print("Error: filtered-csv must contain column 'plain_peptide'", file=sys.stderr)
            sys.exit(1)
        filtered_keys = set()
        for _, row in df_filtered.iterrows():
            seq = str(row.get('plain_peptide', '')).strip()
            try:
                c = int(row.get('charge', 0)) if pd.notna(row.get('charge')) else 0
            except (TypeError, ValueError):
                c = 0
            mods = _norm_mods(row.get('modifications', '-'))
            filtered_keys.add((seq, c, mods))
        df_full['_mods_norm'] = df_full.apply(lambda row: _norm_mods(row.get('modifications', '-')), axis=1)
        seq_strip = df_full['plain_peptide'].astype(str).str.strip()
        charge_int = df_full['charge'].fillna(0).astype(int)
        mask = [k in filtered_keys for k in zip(seq_strip, charge_int, df_full['_mods_norm'])]
        df_out = df_full.drop(columns=['_mods_norm']).loc[mask].copy()
        # Merge in any columns that exist in filtered-csv but not in full-csv (e.g. peak windows from a merge)
        extra_cols = [c for c in df_filtered.columns if c not in df_out.columns]
        if extra_cols and 'scan' in df_filtered.columns and 'scan' in df_out.columns:
            merge_df = df_filtered[['scan', 'plain_peptide', 'charge'] + extra_cols].drop_duplicates(subset=['scan', 'plain_peptide', 'charge'])
            df_out = df_out.merge(merge_df, on=['scan', 'plain_peptide', 'charge'], how='left')
        elif extra_cols:
            # No scan column: merge by (plain_peptide, charge, mods) and take first match per row (may duplicate if many PSMs per peptide)
            merge_on = ['plain_peptide', 'charge']
            if 'modifications' in df_filtered.columns and 'modifications' in df_out.columns:
                merge_on.append('modifications')
            merge_df = df_filtered[[c for c in merge_on if c in df_filtered.columns] + extra_cols]
            merge_df = merge_df.drop_duplicates(subset=merge_on)
            df_out = df_out.merge(merge_df, on=merge_on, how='left')
        print(f"Filtered to {len(df_out)} rows (from {len(filtered_keys)} unique peptides) with {len(df_out.columns)} columns")

    # Preserve MS1 column name if present
    if 'MS1_retention_time_sec' not in df_out.columns and 'retention_time_sec' in df_out.columns:
        df_out = df_out.rename(columns={'retention_time_sec': 'MS1_retention_time_sec'})

    df_out.to_csv(args.output, index=False)
    print(f"Wrote: {os.path.abspath(args.output)}")
    print(f"Columns ({len(df_out.columns)}): {', '.join(df_out.columns[:8])}{'...' if len(df_out.columns) > 8 else ''}")


if __name__ == '__main__':
    main()
