#!/usr/bin/env python3
"""
Plot channel assignment from a pre-assigned CSV (no re-extraction, no re-assignment).

Reads a CSV with channel, collection_min_rt, collection_max_rt, drift_min_rt, drift_max_rt,
total_area, plain_peptide, charge, modifications. Creates the RT windows by channel plot.

Usage:
  python plot_channels_from_csv.py <channels_csv> [--output-dir <dir>] [--protein <id>]
"""

import argparse
import os
import sys

import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)


def _norm_mods(m):
    if m is None or (isinstance(m, float) and pd.isna(m)):
        return '-'
    s = str(m).strip()
    return s if s and s.lower() != 'nan' else '-'


def main():
    ap = argparse.ArgumentParser(description='Plot channel assignment from CSV (no re-extraction)')
    ap.add_argument('csv', help='Channels CSV (with channel, collection_min_rt, drift_min_rt, etc.)')
    ap.add_argument('--output-dir', '-o', default=None, help='Output directory (default: same as CSV)')
    ap.add_argument('--protein', default='', help='Protein ID for plot title')
    args = ap.parse_args()

    if not os.path.isfile(args.csv):
        print(f"Error: not found: {args.csv}")
        sys.exit(1)

    skip = 0
    with open(args.csv) as f:
        if 'CometVersion' in f.readline():
            skip = 1
    df = pd.read_csv(args.csv, skiprows=skip)

    required = ['plain_peptide', 'charge', 'channel', 'collection_min_rt', 'collection_max_rt']
    for c in required:
        if c not in df.columns:
            print(f"Error: required column '{c}' not in CSV")
            sys.exit(1)

    # Build peptide dicts for create_rt_integration_windows_plot
    peptides = []
    for _, row in df.iterrows():
        seq = str(row.get('plain_peptide', '')).strip()
        ch = row.get('charge', 0)
        try:
            ch = int(ch) if pd.notna(ch) else 0
        except (TypeError, ValueError):
            ch = 0
        mods = _norm_mods(row.get('modifications'))
        p = {
            'plain_peptide': seq,
            'peptide_seq': seq,
            'charge': ch,
            'modifications': mods,
            'channel': int(row['channel']) if pd.notna(row.get('channel')) and row.get('channel') >= 0 else 0,
            'collection_min_rt': row.get('collection_min_rt'),
            'collection_max_rt': row.get('collection_max_rt'),
            'detected_peak_min_rt': row.get('detected_peak_min_rt'),
            'detected_peak_max_rt': row.get('detected_peak_max_rt'),
            'spectrum_window_min_rt': row.get('spectrum_window_min_rt'),
            'spectrum_window_max_rt': row.get('spectrum_window_max_rt'),
            'drift_min_rt': row.get('drift_min_rt') or row.get('buffer_min_rt'),
            'drift_max_rt': row.get('drift_max_rt') or row.get('buffer_max_rt'),
            'total_area': row.get('total_area'),
            'relative_area_within_selection': None,
            'mz': row.get('mz') or row.get('theoretical_mz'),
            'is_backfill': False,
            'is_repeat_in_channel': False,
        }
        # Use buffer as drift if drift not present
        if (p['drift_min_rt'] is None or pd.isna(p['drift_min_rt'])) and 'buffer_min_rt' in df.columns:
            p['drift_min_rt'] = row.get('buffer_min_rt')
        if (p['drift_max_rt'] is None or pd.isna(p['drift_max_rt'])) and 'buffer_max_rt' in df.columns:
            p['drift_max_rt'] = row.get('buffer_max_rt')
        peptides.append(p)

    # Compute relative_area for color scale
    total_sum = sum(float(p.get('total_area') or 0) for p in peptides)
    for p in peptides:
        ta = p.get('total_area')
        if ta is not None and ta == ta and float(ta) > 0 and total_sum > 0:
            p['relative_area_within_selection'] = float(ta) / total_sum
        else:
            p['relative_area_within_selection'] = 0.0

    out_dir = args.output_dir or os.path.dirname(os.path.abspath(args.csv))
    os.makedirs(out_dir, exist_ok=True)
    rt_windows_dir = os.path.join(out_dir, 'rt_windows_channels')
    os.makedirs(rt_windows_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(args.csv))[0]
    # Remove _3channels suffix for output base if present
    output_base = base.replace('_3channels', '') if '_3channels' in base else base
    protein_id = args.protein or 'protein'
    num_channels = max(1, max((p.get('channel') or 0) for p in peptides) + 1)

    from visualization import create_rt_integration_windows_plot

    output_file = os.path.join(rt_windows_dir, f'{output_base}_rt_windows_by_channel.png')
    create_rt_integration_windows_plot(
        peptides, output_file, protein_id,
        dedupe_by_peptide=False, rt_max_sec=None, sort_by_collection_start=True,
        max_zoom_segments=0, sort_by_channel=True, peptide_traces=None,
        num_channels=num_channels, stage_label=f'{len(peptides)} peptides, 3 channels'
    )
    print(f"Channel assignment plot: {output_file}")


if __name__ == '__main__':
    main()
