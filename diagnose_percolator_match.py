#!/usr/bin/env python3
"""
Re-run Percolator merge with --diagnose-unmatched to see why some CSV rows
don't get a q-value/PEP from Percolator (matched vs unmatched, and whether
unmatched (scan, charge) appear in Percolator output).

Use the same CSV and Percolator output (.psms or re-run from .pin) that your
pipeline uses. Example:

  python3 diagnose_percolator_match.py
  python3 diagnose_percolator_match.py --csv data/WT_nep2_0MUrea_08_with_ms1.csv --percolator-output data/WT_nep2_0MUrea_08_PP.psms

If you see " (scan,charge) in Percolator lookup: False" for all unmatched rows,
then those (scan, charge) pairs are not in the Percolator output at all — usually
because the CSV and the .pin file are from different runs (e.g. different raw file
or different Comet output). Fix by running Percolator on the .pin that matches
your CSV (same Comet run), or use the E-value FDR fallback (decoy/protein column).
"""

import os
import sys
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--csv', default=None,
                        help=f'CSV that was (or will be) merged with Percolator output (default: {DATA_DIR}/WT_nep2_0MUrea_08_with_ms1.csv)')
    parser.add_argument('--pin', default=None,
                        help='Path to .pin file (optional; used only if --percolator-output not given)')
    parser.add_argument('--percolator-output', default=None,
                        help='Path to existing Percolator .psms output (skip re-running Percolator)')
    parser.add_argument('--output', default=None,
                        help='Output CSV path (default: same dir as CSV with _perc_qvalues_diagnose.csv)')
    args = parser.parse_args()

    csv_path = args.csv or os.path.join(DATA_DIR, 'WT_nep2_0MUrea_08_with_ms1.csv')
    if not os.path.isfile(csv_path):
        print(f"CSV not found: {csv_path}")
        print("Use --csv to point to the CSV that has scan/charge columns (e.g. Comet CSV with MS1 added).")
        return 1

    if args.output is None:
        base = os.path.splitext(os.path.basename(csv_path))[0]
        args.output = os.path.join(os.path.dirname(csv_path), f"{base}_perc_qvalues_diagnose.csv")

    # Import and run add_percolator_qvalues with diagnose
    sys.path.insert(0, SCRIPT_DIR)
    from add_percolator_qvalues import (
        check_percolator, find_pin_file, run_percolator,
        parse_percolator_output, merge_qvalues_into_csv
    )

    percolator_output = args.percolator_output
    if not percolator_output and args.pin:
        percolator_output = run_percolator(args.pin)
    if not percolator_output:
        # Try to find .psms next to CSV or in data dir
        for name in [os.path.splitext(os.path.basename(csv_path))[0].replace('_with_ms1', '') + '_PP.psms',
                     'WT_nep2_0MUrea_08_PP.psms']:
            cand = os.path.join(os.path.dirname(csv_path), name)
            if os.path.isfile(cand):
                percolator_output = cand
                print(f"Using existing Percolator output: {percolator_output}")
                break
    if not percolator_output:
        print("No Percolator output. Use --percolator-output path/to/file.psms or --pin path/to/file.pin")
        return 1

    qvalues = parse_percolator_output(percolator_output)
    if not qvalues:
        print("Failed to parse Percolator output.")
        return 1

    ok = merge_qvalues_into_csv(csv_path, qvalues, args.output, diagnose_unmatched=True)
    if ok:
        print(f"\nDiagnostic output written to: {args.output}")
        return 0
    return 1

if __name__ == '__main__':
    sys.exit(main())
