#!/usr/bin/env python3
"""
Plot chromatograms from extraction output (chromatogram_metrics_all.csv + chromatogram_traces.npz).

Use after Step 6 (Extraction) with --extract-only. Generates individual chromatogram plots
in extraction/ and rejected_extraction/ subdirs.

Usage:
  python plot_chromatograms_from_extraction.py --output-dir results
  python plot_chromatograms_from_extraction.py --chromatogram-metrics-csv path/to/chromatogram_metrics_all.csv --chromatogram-traces path/to/chromatogram_traces.npz --output-dir results
"""

import argparse
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def main():
    ap = argparse.ArgumentParser(
        description='Plot chromatograms from extraction output (chromatogram_metrics_all.csv + chromatogram_traces.npz).'
    )
    ap.add_argument('--output-dir', '-o', default=None,
                    help='Output directory (default: dir containing chromatogram_metrics_all.csv)')
    ap.add_argument('--chromatogram-metrics-csv', default=None,
                    help='Path to chromatogram_metrics_all.csv (default: output-dir/dataframes/chromatogram_metrics_all.csv)')
    ap.add_argument('--chromatogram-traces', default=None,
                    help='Path to chromatogram_traces.npz (default: same dir as metrics CSV)')
    ap.add_argument('--filter-csv', default=None,
                    help='Optional: current step input/output CSV used as peptide include-list for plotting.')
    ap.add_argument('--output-accepted-dir', default=None,
                    help='Override accepted plots directory (default: output-dir/extraction)')
    ap.add_argument('--output-rejected-dir', default=None,
                    help='Override rejected plots directory (default: output-dir/rejected_extraction)')
    args = ap.parse_args()

    if args.chromatogram_metrics_csv:
        metrics_csv = os.path.abspath(args.chromatogram_metrics_csv)
    elif args.output_dir:
        metrics_csv = os.path.join(os.path.abspath(args.output_dir), 'dataframes', 'chromatogram_metrics_all.csv')
    else:
        print('Error: provide --output-dir or --chromatogram-metrics-csv')
        sys.exit(1)

    if args.chromatogram_traces:
        traces_path = os.path.abspath(args.chromatogram_traces)
    else:
        traces_path = os.path.join(os.path.dirname(metrics_csv), 'chromatogram_traces.npz')

    out_dir = args.output_dir or os.path.dirname(os.path.dirname(metrics_csv))
    out_dir = os.path.abspath(out_dir)
    accepted_dir = args.output_accepted_dir or os.path.join(out_dir, 'extraction')
    rejected_dir = args.output_rejected_dir or os.path.join(out_dir, 'rejected_extraction')
    dataframes_dir = os.path.dirname(metrics_csv)

    if not os.path.exists(metrics_csv):
        print(f'Error: chromatogram_metrics_all.csv not found: {metrics_csv}')
        sys.exit(1)
    if not os.path.exists(traces_path):
        print(f'Error: chromatogram_traces.npz not found: {traces_path}')
        sys.exit(1)

    try:
        from visualization.chromatograms import run_chromatograms
    except ImportError as e:
        print(f'Error: Could not import visualization.chromatograms: {e}', file=sys.stderr)
        sys.exit(1)

    ns = argparse.Namespace(
        plot_only=True,
        chromatogram_metrics_csv=metrics_csv,
        chromatogram_traces=traces_path,
        filter_csv=args.filter_csv,
        exclude_mods=False,
        output_accepted_dir=accepted_dir,
        output_rejected_dir=rejected_dir,
        dataframes_dir=dataframes_dir,
        no_psm_filter=False,
    )
    print(f'[Plot Chromatograms] Metrics: {metrics_csv}')
    if args.filter_csv:
        print(f'[Plot Chromatograms] Filter CSV: {args.filter_csv} (current step peptide list used for plotting)')
    print(f'[Plot Chromatograms] Traces: {traces_path}')
    print(f'[Plot Chromatograms] Output: {accepted_dir} / {rejected_dir}')
    run_chromatograms(ns)


if __name__ == '__main__':
    main()
