#!/usr/bin/env python3
"""
Compare Byonic peptide output to Comet CSVs to find peptides that Byonic found
but Comet did not report (at any charge or as substring).

Usage:
  python compare_byonic_comet.py data/byonic_search_peps.csv data/WT_nep2_0MUrea_08_with_qvalues_ms1_all.csv
  python compare_byonic_comet.py data/byonic_search_peps.csv data/WT_nep2_0MUrea_08.csv data/WT_nep2_0MUrea_08_with_qvalues_ms1_all.csv
"""

import argparse
import os
import re
import sys

def main():
    ap = argparse.ArgumentParser(description='Compare Byonic vs Comet peptides')
    ap.add_argument('byonic_csv', help='Byonic output CSV (e.g. byonic_search_peps.csv)')
    ap.add_argument('comet_csvs', nargs='+', help='One or more Comet CSV files (e.g. WT_nep2_0MUrea_08_with_qvalues_ms1_all.csv)')
    ap.add_argument('--exact', action='store_true', help='Require exact (sequence, charge) match; default is sequence contained in any Comet peptide at same charge')
    args = ap.parse_args()

    import pandas as pd

    # ---- Load Byonic ----
    byonic_path = args.byonic_csv
    if not os.path.exists(byonic_path):
        print(f"Error: Byonic file not found: {byonic_path}", file=sys.stderr)
        sys.exit(1)
    with open(byonic_path, 'r', encoding='utf-8', newline='') as f:
        raw = f.read()
    # Normalize header: replace \r\n with space inside quoted fields so one header line
    first_newline = raw.find('\n')
    header_line = raw[:first_newline].replace('\r\n', ' ').replace('\n', ' ')
    rest = raw[first_newline + 1:].lstrip('\r\n')
    # Find column indices by parsing header
    try:
        df_byonic = pd.read_csv(pd.io.common.StringIO(header_line + '\n' + rest), sep=',', engine='python', quotechar='"', on_bad_lines='warn')
    except Exception as e:
        # Fallback: read with default (might have multi-line header)
        df_byonic = pd.read_csv(byonic_path, sep=',', engine='python', quotechar='"', on_bad_lines='warn')
    # Detect sequence column (may be "Sequence (unformatted)" or similar after newline normalization)
    seq_col = None
    for c in df_byonic.columns:
        if 'sequence' in c.lower() and 'unformat' in c.lower():
            seq_col = c
            break
    if seq_col is None:
        for c in df_byonic.columns:
            if 'sequence' in c.lower():
                seq_col = c
                break
    if seq_col is None:
        seq_col = df_byonic.columns[2]  # often 3rd column
    # Charge column
    z_col = None
    for c in df_byonic.columns:
        if c.strip().lower() == 'z':
            z_col = c
            break
    if z_col is None and 'z' in df_byonic.columns:
        z_col = 'z'
    if z_col is None:
        print("Error: Could not find charge column in Byonic CSV. Columns:", list(df_byonic.columns), file=sys.stderr)
        sys.exit(1)

    def byonic_sequence_to_plain(s):
        """Extract plain peptide from Byonic: remove flanking X., strip modification brackets [+mass] so comparison matches Comet plain_peptide."""
        if pd.isna(s) or s is None:
            return None
        s = str(s).strip()
        if not s:
            return None
        # Remove flanking X. and .X (cleavage notation)
        if '.' in s:
            parts = s.split('.')
            if len(parts) >= 2:
                s = parts[1]
            else:
                s = s.replace('.', '')
        # Strip Byonic mod notation e.g. [+15.99492] so plain sequence matches Comet
        s = re.sub(r'\[\s*[+-]?\s*[\d.]+\s*\]', '', s)
        return s if s else None

    byonic_peptides = set()
    for _, row in df_byonic.iterrows():
        seq = byonic_sequence_to_plain(row.get(seq_col))
        if not seq:
            continue
        try:
            z = int(float(row[z_col]))
        except (TypeError, ValueError):
            continue
        byonic_peptides.add((seq, z))
    print(f"Byonic: {len(df_byonic)} rows -> {len(byonic_peptides)} unique (sequence, charge) peptides")

    # ---- Load Comet ----
    comet_plain_charge = set()
    comet_sequences_by_charge = {}  # charge -> set of plain_peptide
    for comet_path in args.comet_csvs:
        if not os.path.exists(comet_path):
            print(f"Warning: Comet file not found: {comet_path}", file=sys.stderr)
            continue
        try:
            with open(comet_path, 'r') as f:
                first = f.readline()
            # Only skip first line if it's a comment (e.g. CometVersion), not the header
            skip = 1 if first.strip().startswith('CometVersion') or first.strip().startswith('#') else 0
            df_comet = pd.read_csv(comet_path, sep=',', skiprows=skip, engine='python', quotechar='"', on_bad_lines='warn')
        except Exception as e:
            print(f"Warning: Could not read {comet_path}: {e}", file=sys.stderr)
            continue
        if 'plain_peptide' not in df_comet.columns:
            print(f"Warning: No plain_peptide column in {comet_path}", file=sys.stderr)
            continue
        for _, row in df_comet.iterrows():
            p = row.get('plain_peptide')
            if pd.isna(p) or p is None:
                continue
            p = str(p).strip()
            if not p:
                continue
            try:
                z = int(float(row['charge']))
            except (TypeError, ValueError):
                continue
            comet_plain_charge.add((p, z))
            if z not in comet_sequences_by_charge:
                comet_sequences_by_charge[z] = set()
            comet_sequences_by_charge[z].add(p)
        print(f"Comet {os.path.basename(comet_path)}: {len(df_comet)} rows -> {len(comet_plain_charge)} unique (plain_peptide, charge) across all files so far")
    print(f"Comet combined: {len(comet_plain_charge)} unique (sequence, charge)")

    # ---- Compare ----
    if args.exact:
        missed = [(seq, z) for (seq, z) in byonic_peptides if (seq, z) not in comet_plain_charge]
        print(f"\nExact (sequence, charge) match: Byonic peptides NOT in Comet: {len(missed)}")
    else:
        # Byonic (seq, z) is "found" in Comet if any Comet peptide at charge z equals seq or contains seq (or seq contains it)
        missed = []
        for (seq, z) in byonic_peptides:
            if z not in comet_sequences_by_charge:
                missed.append((seq, z))
                continue
            found = False
            for comet_seq in comet_sequences_by_charge[z]:
                if comet_seq == seq or seq in comet_seq or comet_seq in seq:
                    found = True
                    break
            if not found:
                missed.append((seq, z))
        print(f"\nContained match (Byonic seq in Comet peptide at same charge): Byonic peptides NOT in Comet: {len(missed)}")

    if missed:
        print("\nByonic peptides not found in Comet (sequence, charge):")
        for (seq, z) in sorted(missed, key=lambda x: (x[0], x[1])):
            print(f"  {seq}  z={z}")
        # Optionally write to file
        out_path = os.path.join(os.path.dirname(byonic_path), 'byonic_missed_in_comet.txt')
        with open(out_path, 'w') as f:
            f.write("sequence\tcharge\n")
            for (seq, z) in sorted(missed, key=lambda x: (x[0], x[1])):
                f.write(f"{seq}\t{z}\n")
        print(f"\nWrote {len(missed)} missed peptides to {out_path}")
    else:
        print("\nAll Byonic peptides were found in Comet (at same charge, exact or contained).")

    # Summary: Comet-only and overlap
    if not args.exact:
        comet_only_peptides = set()
        for (p, z) in comet_plain_charge:
            matched_byonic = False
            for (b_seq, b_z) in byonic_peptides:
                if b_z != z:
                    continue
                if p == b_seq or b_seq in p or p in b_seq:
                    matched_byonic = True
                    break
            if not matched_byonic:
                comet_only_peptides.add((p, z))
        print(f"\nComet peptides not in Byonic (sequence, charge): {len(comet_only_peptides)} (sample up to 20)")
        for (p, z) in sorted(comet_only_peptides, key=lambda x: (x[0], x[1]))[:20]:
            print(f"  {p}  z={z}")
        if len(comet_only_peptides) > 20:
            print(f"  ... and {len(comet_only_peptides) - 20} more")

if __name__ == '__main__':
    main()
