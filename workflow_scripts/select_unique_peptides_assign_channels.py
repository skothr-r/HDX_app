#!/usr/bin/env python3
"""
Step 9: Assign channels to selected target peptides.

Uses the channel assignment algorithm from regenerate_rt_windows_from_csv.
Peptides with protect_peptide=True (or valuable_sequence=1) are protected from intensity-based filters
and prioritized for the first 3 channels.

Creates combined and individual channel plots with all peptides overlays.

Usage:
  python select_unique_peptides_assign_channels.py --fasta data/protein.fasta
  python select_unique_peptides_assign_channels.py --fasta protein.fasta --input valuable_sequences.csv --output-dir results --mzml file.mzML
"""

import argparse
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

DEFAULT_INPUT = os.path.join(_PROJECT_ROOT, 'data', 'comet_frags_perc_openMS_confidence_accuracy_extraction_significance_sequence.csv')
DEFAULT_FASTA = os.path.join(_PROJECT_ROOT, 'data', 'Ube2D3.fasta')
DEFAULT_MZML = os.path.join(_PROJECT_ROOT, 'data', 'WT_nep2_0MUrea_08.mzML')


def main():
    ap = argparse.ArgumentParser(
        description='Step 9: Assign channels to selected target peptides (calls regenerate_rt_windows_from_csv).'
    )
    ap.add_argument('--input', '-i', default=DEFAULT_INPUT,
                    help=f'Input CSV from Step 8 (default: {os.path.basename(DEFAULT_INPUT)})')
    ap.add_argument('--output-dir', '-o', default=None,
                    help='Output directory (default: same as input CSV directory)')
    ap.add_argument('--fasta', '-f', default=DEFAULT_FASTA,
                    help=f'FASTA file for unique peptides and combined plots (default: {os.path.basename(DEFAULT_FASTA)})')
    ap.add_argument('--mzml', '-m', default=DEFAULT_MZML,
                    help=f'mzML file for chromatogram overlays (default: {os.path.basename(DEFAULT_MZML)})')
    ap.add_argument('--protein', '-p', default=None,
                    help='Protein ID in FASTA')
    args = ap.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input CSV not found: {args.input}")
        sys.exit(1)

    from regenerate_rt_windows_from_csv import run_rt_windows

    input_dir = os.path.dirname(os.path.abspath(args.input))
    out_dir = args.output_dir or input_dir
    os.makedirs(out_dir, exist_ok=True)

    ns = argparse.Namespace(
        csv_path=os.path.abspath(args.input),
        output_dir=out_dir,
        exclude_mods=False,
        mzml=args.mzml,
        raw_file=args.mzml,
        chromatogram_traces=None,
        chromatogram_trace_csv=None,
        run_max_rt_sec=None,
        fasta=args.fasta,
        protein=args.protein,
        no_re_extract_windows=True,
        re_extract_chromatograms=False,
        comet_csv=None,
    )

    print(f"[Step 9] Running channel assignment on: {args.input}")
    print(f"[Step 9] Output directory: {out_dir}")
    run_rt_windows(ns)
    print(f"[Step 9] Done. Channel plots and CSVs saved to: {os.path.abspath(out_dir)}")


if __name__ == '__main__':
    main()
