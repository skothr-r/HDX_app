#!/usr/bin/env python3
"""
Create a fixed version of the accepted windows integration CSV where the collection
window equals the integration window (3% of max peak intensity boundary) with no
30 sec buffer extension.

Original: collection = integration + extension (max(30s, 50% integration width))
Fixed:    collection = integration (no extension beyond 3% boundary)

Drift buffer (collection ± 30s) is recomputed for instrument calibration.
"""

import argparse
import os
import pandas as pd

PEAK_DRIFT_BUFFER_SEC = 30.0


def main():
    ap = argparse.ArgumentParser(description='Fix collection window to use 3% boundary only (no 30s extension)')
    ap.add_argument('--input', '-i', required=True, help='Input CSV (accepted_windows_integration.csv)')
    ap.add_argument('--output', '-o', default=None, help='Output CSV (default: input with _fixed_3pct suffix)')
    args = ap.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input not found: {args.input}")
        return 1

    # Read CSV
    try:
        with open(args.input, 'r') as f:
            first = f.readline()
        skip = 1 if 'CometVersion' in first else 0
    except Exception:
        skip = 0

    df = pd.read_csv(args.input, sep=',', skiprows=skip, engine='python', quotechar='"', on_bad_lines='warn')

    min_col = 'detected_peak_min_rt'
    max_col = 'detected_peak_max_rt'

    if min_col not in df.columns or max_col not in df.columns:
        print(f"Error: CSV must have {min_col} and {max_col} columns")
        return 1

    # Where we have detected_peak_min_rt and detected_peak_max_rt, set collection = integration (no extension)
    for idx in df.index:
        dmin = df.loc[idx, min_col]
        dmax = df.loc[idx, max_col]
        if pd.notna(dmin) and pd.notna(dmax) and dmin != '' and dmax != '':
            try:
                dmin_f = float(dmin)
                dmax_f = float(dmax)
                if dmax_f > dmin_f:
                    df.at[idx, 'collection_min_rt'] = dmin_f
                    df.at[idx, 'collection_max_rt'] = dmax_f
                    df.at[idx, 'collection_window_size'] = dmax_f - dmin_f
                    df.at[idx, 'drift_min_rt'] = max(0.0, dmin_f - PEAK_DRIFT_BUFFER_SEC)
                    df.at[idx, 'drift_max_rt'] = dmax_f + PEAK_DRIFT_BUFFER_SEC
            except (TypeError, ValueError):
                pass

    # Output path
    if args.output:
        out_path = args.output
    else:
        base, ext = os.path.splitext(args.input)
        out_path = f'{base}_fixed_3pct{ext}'

    df.to_csv(out_path, index=False)
    print(f"[fix_collection_window_3pct] Wrote: {os.path.abspath(out_path)}")
    print(f"  Collection window = integration window (3% boundary), no 30s extension")
    print(f"  Drift = collection ± {PEAK_DRIFT_BUFFER_SEC}s (for instrument calibration)")

    return 0


if __name__ == '__main__':
    exit(main())
