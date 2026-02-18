#!/usr/bin/env python3
"""
Assign peptides from a CSV to exactly 3 channels (no layers, no backfill).

Reads a CSV with collection_min_rt, collection_max_rt (or detected_peak_*), total_area.
Uses assign_peptides_to_channels with num_channels=3.

Usage:
  python assign_channels_3_from_csv.py <input.csv> [--output <output.csv>]
"""

import argparse
import csv
import os
import sys

import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from regenerate_rt_windows_from_csv import (
    assign_peptides_to_channels,
    reorder_channels_by_relative_area,
    _get_collection_bounds,
    _get_drift_bounds,
    CHANNEL_COLLECTION_GAP_SEC,
)


def _norm_mods(m):
    if m is None or (isinstance(m, float) and pd.isna(m)):
        return '-'
    s = str(m).strip()
    return s if s and s.lower() != 'nan' else '-'


def main():
    ap = argparse.ArgumentParser(description='Assign peptides to exactly 3 channels')
    ap.add_argument('input', help='Input CSV (e.g. rel07pct_envelope)')
    ap.add_argument('--output', '-o', default=None, help='Output CSV path')
    args = ap.parse_args()

    if not os.path.isfile(args.input):
        print(f"Error: not found: {args.input}")
        sys.exit(1)

    skip = 0
    with open(args.input) as f:
        if 'CometVersion' in f.readline():
            skip = 1
    df = pd.read_csv(args.input, skiprows=skip)

    required = ['plain_peptide', 'charge', 'collection_min_rt', 'collection_max_rt']
    for c in required:
        if c not in df.columns:
            print(f"Error: required column '{c}' not in CSV")
            sys.exit(1)

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
            'collection_min_rt': float(row['collection_min_rt']) if pd.notna(row.get('collection_min_rt')) else None,
            'collection_max_rt': float(row['collection_max_rt']) if pd.notna(row.get('collection_max_rt')) else None,
            'detected_peak_min_rt': row.get('detected_peak_min_rt'),
            'detected_peak_max_rt': row.get('detected_peak_max_rt'),
            'total_area': row.get('total_area'),
        }
        for col in df.columns:
            if col not in p:
                p[col] = row[col] if col in row.index else None
        peptides.append(p)

    # Compute relative_area_within_selection
    total_sum = sum(float(p.get('total_area') or 0) for p in peptides)
    for p in peptides:
        ta = p.get('total_area')
        if ta is not None and ta == ta and float(ta) > 0 and total_sum > 0:
            p['relative_area_within_selection'] = float(ta) / total_sum
        else:
            p['relative_area_within_selection'] = 0.0

    # Keep all peptides (no rel area filter - user wants full CSV preserved)

    # Deduplicate: keep one row per (plain_peptide, charge, modifications), prefer highest total_area
    def _key(p):
        return (p.get('plain_peptide', ''), p.get('charge', 0), p.get('modifications', '-'))
    seen = {}
    for p in peptides:
        k = _key(p)
        ta = float(p.get('total_area') or 0)
        if k not in seen or float(seen[k].get('total_area') or 0) < ta:
            seen[k] = p
    n_before_dedup = len(peptides)
    peptides = list(seen.values())
    if len(peptides) < n_before_dedup:
        print(f"Deduplicated: {n_before_dedup} -> {len(peptides)} unique peptides (1 per seq+charge+mods)")

    # Recompute relative area for priority
    total_sum = sum(float(p.get('total_area') or 0) for p in peptides)
    for p in peptides:
        ta = p.get('total_area')
        if ta is not None and ta == ta and float(ta) > 0 and total_sum > 0:
            p['relative_area_within_selection'] = float(ta) / total_sum
        else:
            p['relative_area_within_selection'] = 0.0

    # Prioritize >1% relative area first, then backfill with <1% (each tier sorted by total_area desc)
    REL_PRIORITY = 0.01  # 1%
    high_priority = [p for p in peptides if (p.get('relative_area_within_selection') or 0) >= REL_PRIORITY]
    low_priority = [p for p in peptides if (p.get('relative_area_within_selection') or 0) < REL_PRIORITY]
    high_priority.sort(key=lambda p: (-(float(p.get('total_area') or 0)), p.get('collection_min_rt') or 0))
    low_priority.sort(key=lambda p: (-(float(p.get('total_area') or 0)), p.get('collection_min_rt') or 0))
    peptides = high_priority + low_priority
    print(f"Prioritized: {len(high_priority)} peptides >=1% rel area, {len(low_priority)} <1% for backfill")

    # Assign to 3 channels: use collection bounds only (no drift - drift is instrument, not for channel assignment).
    # Enforce gap between collection windows so they never overlap (1 s min gap).
    # Assign to the channel that becomes available first (earliest end); tie-break by intensity (already sorted by total_area desc).
    num_channels = 3
    assign_peptides_to_channels(peptides, num_channels, gap_sec=CHANNEL_COLLECTION_GAP_SEC, preserve_order=True, use_collection_bounds=True)
    assigned = [p for p in peptides if p.get('channel', -1) >= 0]
    unassigned = [p for p in peptides if p.get('channel', -1) < 0]
    if unassigned:
        print(f"{len(unassigned)} peptides unassigned (overlap in all 3 channels) - kept in CSV with channel=-1")
    reorder_channels_by_relative_area(assigned, num_channels)
    # Keep ALL peptides in output (assigned + unassigned); plot will only show assigned

    # Order: assigned first (by collection start), then unassigned (by collection start)
    assigned.sort(key=lambda p: (float(p.get('collection_min_rt') or 0), float(p.get('collection_max_rt') or 0)))
    unassigned.sort(key=lambda p: (float(p.get('collection_min_rt') or 0), float(p.get('collection_max_rt') or 0)))
    peptides = assigned + unassigned

    # Output path
    if args.output:
        out_path = args.output
    else:
        base = os.path.splitext(os.path.basename(args.input))[0]
        out_dir = os.path.dirname(os.path.abspath(args.input))
        out_path = os.path.join(out_dir, f'{base}_3channels.csv')

    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f)
        header = list(df.columns)
        if 'channel' not in header:
            header.append('channel')
        if 'drift_min_rt' not in header:
            header.extend(['drift_min_rt', 'drift_max_rt'])
        # First 6 columns: buffer (min, sec), then collection (min, sec)
        BUFFER_SEC = 30.0
        lead_cols = ['buffer_min_rt_min', 'buffer_max_rt_min', 'buffer_min_rt', 'buffer_max_rt', 'collection_min_rt_min', 'collection_max_rt_min', 'collection_min_rt', 'collection_max_rt']
        header = lead_cols + [c for c in header if c not in lead_cols]
        w.writerow(header)
        for p in peptides:
            cmin, cmax = _get_collection_bounds(p)
            drift_lo, drift_hi = _get_drift_bounds(p)
            buf_min = max(0.0, float(cmin) - BUFFER_SEC) if cmin is not None and cmin == cmin else ''
            buf_max = float(cmax) + BUFFER_SEC if cmax is not None and cmax == cmax else ''
            buf_min_min = (float(buf_min) / 60.0) if buf_min != '' else ''
            buf_max_min = (float(buf_max) / 60.0) if buf_max != '' else ''
            cmin_min = (float(cmin) / 60.0) if cmin is not None and cmin == cmin else ''
            cmax_min = (float(cmax) / 60.0) if cmax is not None and cmax == cmax else ''
            row = [buf_min_min, buf_max_min, buf_min, buf_max, cmin_min, cmax_min, cmin, cmax]
            for col in header:
                if col in lead_cols:
                    continue
                if col == 'channel':
                    row.append(p.get('channel', -1))
                elif col == 'drift_min_rt':
                    row.append(drift_lo)
                elif col == 'drift_max_rt':
                    row.append(drift_hi)
                else:
                    row.append(p.get(col, ''))
            w.writerow(row)

    n_ch = [sum(1 for p in assigned if p.get('channel') == ch) for ch in range(num_channels)]
    sum_rel = [sum((p.get('relative_area_within_selection') or 0) for p in assigned if p.get('channel') == ch) for ch in range(num_channels)]
    print(f"Assigned {len(assigned)} peptides to 3 channels: Ch1={n_ch[0]} (relΣ={sum_rel[0]:.3f}), Ch2={n_ch[1]} (relΣ={sum_rel[1]:.3f}), Ch3={n_ch[2]} (relΣ={sum_rel[2]:.3f}); {len(unassigned)} unassigned (channel=-1)")
    print(f"Wrote: {out_path}")


if __name__ == '__main__':
    main()
