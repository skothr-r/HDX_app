#!/usr/bin/env python3
"""
Generate per-peptide summed MS1 spectrum plots for Envelope step.

Classification criteria (from plotted summed MS1 spectrum):
  1) M0, M+1, M+2, M+3 present (highest-intensity match within +/-5 ppm for each)
  2) M0 and M+1 are both > M+2 and M+3

Usage:
  python generate_envelope_summed_ms1_plots.py \
    --input extraction.csv --mzml sample.mzML --mode accepted --output-dir envelope/
"""

import argparse
import os
import re
import sys

import matplotlib.pyplot as plt
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def _safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_") or "peptide"


def _mods_norm(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "-"
    s = str(v).strip()
    return "-" if not s or s.lower() == "nan" else s


def _to_float(v):
    try:
        return float(v)
    except Exception:
        return None


def _resolve_rt_window_seconds(row):
    """
    Resolve RT window in seconds across legacy/new extraction column names.
    Returns (min_rt_sec, max_rt_sec) or (None, None).
    """
    # Preferred direct seconds columns
    min_rt = _to_float(row.get("detected_peak_min_rt"))
    max_rt = _to_float(row.get("detected_peak_max_rt"))
    if min_rt is None or max_rt is None:
        min_rt = _to_float(row.get("collection_min_rt"))
        max_rt = _to_float(row.get("collection_max_rt"))
    if min_rt is None or max_rt is None:
        min_rt = _to_float(row.get("MS1_RT_integration_start_sec"))
        max_rt = _to_float(row.get("MS1_RT_integration_stop_sec"))
    if min_rt is None or max_rt is None:
        min_rt = _to_float(row.get("MS1_RT_collection_start_sec"))
        max_rt = _to_float(row.get("MS1_RT_collection_stop_sec"))
    # Minutes fallbacks
    if min_rt is None or max_rt is None:
        min_min = _to_float(row.get("MS1_RT_integration_start_minutes"))
        max_min = _to_float(row.get("MS1_RT_integration_stop_minutes"))
        if min_min is not None and max_min is not None:
            min_rt, max_rt = min_min * 60.0, max_min * 60.0
    if min_rt is None or max_rt is None:
        min_min = _to_float(row.get("MS1_RT_collection_start_minutes"))
        max_min = _to_float(row.get("MS1_RT_collection_stop_minutes"))
        if min_min is not None and max_min is not None:
            min_rt, max_rt = min_min * 60.0, max_min * 60.0

    if min_rt is None or max_rt is None:
        return (None, None)
    if max_rt < min_rt:
        min_rt, max_rt = max_rt, min_rt
    return (min_rt, max_rt)


def main():
    ap = argparse.ArgumentParser(description="Generate Envelope summed MS1 plots.")
    ap.add_argument("--input", "-i", required=True, help="Envelope-step input CSV (typically latest _extraction.csv)")
    ap.add_argument("--mzml", required=True, help="mzML path used for MS1 spectrum extraction")
    ap.add_argument("--mode", choices=["accepted", "rejected"], required=True, help="Which subset to plot")
    ap.add_argument("--output-dir", "-o", required=True, help="Directory to write PNG files")
    ap.add_argument("--clear-output", action="store_true", help="Delete existing PNGs in output directory before writing")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: input CSV not found: {args.input}")
        sys.exit(1)
    if not os.path.exists(args.mzml):
        print(f"Error: mzML not found: {args.mzml}")
        sys.exit(1)

    try:
        with open(args.input, "r") as f:
            first = f.readline()
        skip = 1 if "CometVersion" in first else 0
    except Exception:
        skip = 0

    df = pd.read_csv(args.input, sep=",", skiprows=skip, engine="python", quotechar='"', on_bad_lines="warn")
    if df.empty:
        print("Input CSV is empty. Nothing to plot.")
        return

    # Plot one spectrum per unique peptide key.
    pep_col = (
        "plain_peptide" if "plain_peptide" in df.columns else
        ("peptide" if "peptide" in df.columns else
         ("peptide_sequence" if "peptide_sequence" in df.columns else "sequence"))
    )
    if pep_col not in df.columns or "charge" not in df.columns:
        print("Error: input CSV must contain one of plain_peptide/peptide/peptide_sequence/sequence and charge columns.")
        sys.exit(1)
    df["_mods"] = df.apply(lambda r: _mods_norm(r.get("modifications", "-")), axis=1)
    df["_key"] = df.apply(lambda r: f"{str(r.get(pep_col, '')).strip()}_{int(r.get('charge', 0) or 0)}_{r['_mods']}", axis=1)
    target_df = df.drop_duplicates(subset=["_key"], keep="first").reset_index(drop=True)

    os.makedirs(args.output_dir, exist_ok=True)
    if args.clear_output:
        for name in os.listdir(args.output_dir):
            if name.lower().endswith(".png"):
                try:
                    os.remove(os.path.join(args.output_dir, name))
                except OSError:
                    pass

    from visualization.chromatograms import create_summed_ms1_spectrum_figure

    saved = 0
    failed = 0
    accepted_ct = 0
    rejected_ct = 0
    total = len(target_df)
    print(f"[Envelope Summed MS1] Mode={args.mode}; candidate peptides: {total}")
    for i, (_, row) in enumerate(target_df.iterrows(), start=1):
        key = row["_key"]
        out_name = _safe_name(key) + ".png"
        out_path = os.path.join(args.output_dir, out_name)
        min_rt, max_rt = _resolve_rt_window_seconds(row)
        result = create_summed_ms1_spectrum_figure(
            row, args.mzml, min_rt=min_rt, max_rt=max_rt, peptide_key=key, return_metrics=True
        )
        if result is None:
            failed += 1
            continue
        fig, metrics = result
        passes = bool(metrics.get("passes_envelope", False))
        if passes:
            accepted_ct += 1
        else:
            rejected_ct += 1
        keep = (passes and args.mode == "accepted") or ((not passes) and args.mode == "rejected")
        if not keep:
            plt.close(fig)
            continue
        try:
            fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="black", pad_inches=0.2)
            saved += 1
        finally:
            plt.close(fig)
        if i % 25 == 0:
            print(f"[Envelope Summed MS1] Progress: {i}/{total}")

    print(f"[Envelope Summed MS1] Classification from plotted spectra: accepted={accepted_ct}, rejected={rejected_ct}, failed={failed}")
    print(f"[Envelope Summed MS1] Saved {saved} '{args.mode}' plots to: {os.path.abspath(args.output_dir)}")
    if failed:
        print(f"[Envelope Summed MS1] Skipped {failed} peptides (missing RT window/mz data)")


if __name__ == "__main__":
    main()
