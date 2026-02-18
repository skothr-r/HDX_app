#!/usr/bin/env python3
"""
Step 6: Extract chromatograms from mzML and add metrics to CSV.

Reads comet_frags_perc_openMS_prefilter.csv and mzML; performs MS1
chromatogram extraction; adds total_area, apex_peak, integration windows (min_rt,
max_rt), collection window, peak drift buffer; records fragment signals and max
MS2 signal for further filtering.

Output:
  - comet_frags_perc_openMS_prefilter_extraction.csv (all columns + extraction metrics)
  - Individual chromatogram plots in accepted/chromatograms_peptides/
  - Zoom overlays (*_zoom_peptides_all_*.png)

Usage:
  python extract_chromatograms_from_comet_frags_csv_mzml.py --mzml data/WT_nep2_0MUrea_08.mzML
  python extract_chromatograms_from_comet_frags_csv_mzml.py --mzml file.mzML --csv my_prefilter.csv --output-dir results
"""

import argparse
import os
import shutil
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

DEFAULT_CSV = os.path.join(_PROJECT_ROOT, 'data', 'comet_frags_perc_openMS_prefilter.csv')
DEFAULT_MZML = os.path.join(_PROJECT_ROOT, 'data', 'WT_nep2_0MUrea_08.mzML')
EXTRACTION_SUFFIX = '_extraction'
EXTRACTION_TEST_SUFFIX = '_extraction_test'


def main():
    ap = argparse.ArgumentParser(
        description='Step 6: Extract MS1 chromatograms from mzML and add metrics to CSV.'
    )
    ap.add_argument('--mzml', '-m', default=DEFAULT_MZML,
                    help=f'Path to mzML file (default: {os.path.basename(DEFAULT_MZML)})')
    ap.add_argument('--csv', '-c', default=DEFAULT_CSV,
                    help=f'Input CSV from Step 5 (default: {os.path.basename(DEFAULT_CSV)})')
    ap.add_argument('--output-dir', '-o', default=None,
                    help='Output directory (default: same as input CSV directory)')
    ap.add_argument('--use-nested-dirs', action='store_true',
                    help='Use extraction/ and rejected_extraction/ subdirs (for app output layout)')
    ap.add_argument('--test', action='store_true',
                    help='Test mode: random sample of 30 peptides (faster run)')
    ap.add_argument('--extract-only', action='store_true',
                    help='Extract metrics only (no individual chromatogram plots). CSV + traces saved for plotting later.')
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        print(f"Error: Input CSV not found: {args.csv}")
        sys.exit(1)
    if not os.path.exists(args.mzml):
        print(f"Error: mzML file not found: {args.mzml}")
        sys.exit(1)

    input_dir = os.path.dirname(os.path.abspath(args.csv))
    base = os.path.splitext(os.path.basename(args.csv))[0]
    out_dir = args.output_dir or input_dir
    os.makedirs(out_dir, exist_ok=True)

    # Use input base for output naming (e.g. x_prefilter -> x_prefilter_extraction)
    base_clean = base

    output_png = os.path.join(out_dir, f'{base_clean}_chromatograms.png')
    dataframes_dir = os.path.join(out_dir, 'dataframes')
    if getattr(args, 'use_nested_dirs', False):
        accepted_plots = os.path.join(out_dir, 'extraction')
        rejected_plots = os.path.join(out_dir, 'rejected_extraction')
        output_accepted_dir = accepted_plots
        output_rejected_dir = rejected_plots
    else:
        accepted_plots = os.path.join(out_dir, 'accepted')
        rejected_plots = os.path.join(out_dir, 'rejected')
        output_accepted_dir = os.path.join(accepted_plots, 'chromatograms_peptides')
        output_rejected_dir = os.path.join(rejected_plots, 'chromatograms_peptides_rejected')
    accepted_df_dir = accepted_plots
    rejected_df_dir = rejected_plots

    for d in [dataframes_dir, accepted_plots, rejected_plots, output_accepted_dir, output_rejected_dir]:
        os.makedirs(d, exist_ok=True)
    os.makedirs(output_accepted_dir, exist_ok=True)
    os.makedirs(output_rejected_dir, exist_ok=True)

    try:
        from visualization.chromatograms import run_chromatograms
    except ImportError as e:
        print("Error: Could not import visualization.chromatograms (run from project root).", file=sys.stderr)
        print(f"  {e}", file=sys.stderr)
        sys.exit(1)

    ns = argparse.Namespace(
        comet_csv=os.path.abspath(args.csv),
        raw_file=os.path.abspath(args.mzml),
        output_png=output_png,
        filter_csv=None,
        exclude_mods=False,
        test=args.test,
        extract_only=getattr(args, 'extract_only', False),
        extract_output_dir=dataframes_dir if getattr(args, 'extract_only', False) else None,
        dataframes_dir=dataframes_dir,
        output_accepted_dir=output_accepted_dir,
        output_rejected_dir=output_rejected_dir,
        dataframes_accepted_dir=accepted_df_dir,
        dataframes_rejected_dir=rejected_df_dir,
        no_psm_filter=False,
        skip_significance_intensity_filter=True,  # 5 ppm from Step 5; 0.5% applied in Step 7 (refine_significant_frags)
    )

    def _log(msg: str) -> None:
        print(msg, flush=True)

    _log(f"[Step 6] Extracting chromatograms from {args.mzml}")
    _log(f"[Step 6] Input CSV: {args.csv}")
    _log(f"[Step 6] Output directory: {out_dir}")
    _log("[Step 6] Starting chromatogram extraction (this may take several minutes)...")
    run_chromatograms(ns)

    # Copy merged CSV to workflow naming: *_extraction.csv or *_extraction_test.csv when --test
    metrics_name = f'{base_clean}_with_chromatogram_metrics.csv'
    metrics_path = os.path.join(dataframes_dir, metrics_name)
    extraction_suffix = EXTRACTION_TEST_SUFFIX if args.test else EXTRACTION_SUFFIX
    # When test and input ends with _extraction, output x_prefilter_extraction_test (not x_prefilter_extraction_extraction_test)
    out_base = base_clean[:-len('_extraction')] if args.test and base_clean.endswith('_extraction') else base_clean
    extraction_name = f'{out_base}{extraction_suffix}.csv'
    extraction_path = os.path.join(out_dir, extraction_name)

    if os.path.exists(metrics_path):
        shutil.copy2(metrics_path, extraction_path)
        _log(f"[Step 6] Output CSV saved to: {os.path.abspath(extraction_path)}")
    else:
        _log(f"[Step 6] Warning: Merged metrics not found at {metrics_path}")
        _log(f"[Step 6] Check dataframes_dir contents: {os.listdir(dataframes_dir) if os.path.isdir(dataframes_dir) else 'N/A'}")


if __name__ == '__main__':
    main()
