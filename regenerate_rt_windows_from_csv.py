#!/usr/bin/env python3
"""
Regenerate RT integration windows plots from a filtered peptides CSV.

Reads a CSV that has detected_peak_min_rt, detected_peak_max_rt, collection_min_rt,
collection_max_rt (and optionally apex_intensity, total_area for color-coding), and calls create_rt_integration_windows_plot
to produce:
  - <output_base>_rt_windows.png
  - <output_base>_rt_windows_by_peptide.png

Also copies individual peptide chromatogram PNGs for the selected set into
  <output_dir>/<output_base>_chromatograms/
for easy inspection. Channel count is determined by the number of peptides that passed filters (no thinning to 3).

Integration window = colored bar (quantification). Collection window = dark gray on either side (overlaps visible).

Use --exclude-mods when you run the chromatogram script with --exclude-mods, so RT windows
use the same peptide set (e.g. 82 unmodified) and zoom panels match.

Usage:
  python regenerate_rt_windows_from_csv.py <filtered_peptides.csv> <output_dir> [--exclude-mods]

Example:
  python regenerate_rt_windows_from_csv.py path/to/filtered_peptides.csv path/to/output_dir
  python regenerate_rt_windows_from_csv.py path/to/filtered_peptides.csv path/to/output_dir --exclude-mods

Quick refresh (replot RT windows only, no re-extraction):
  Use the accepted CSV from your last run and --no-re-extract-windows. The script will find
  peak_windows.csv or peak_windows_all.csv in output_dir (or the CSV's directory) and merge
  window data, then regenerate all RT window PNGs.
  Example:
    python regenerate_rt_windows_from_csv.py plots_v102/peak_windows.csv plots_v102 --no-re-extract-windows --fasta data/Ube2D3.fasta
  Or point csv_path at your combined output dir so it finds peak_windows there:
    python regenerate_rt_windows_from_csv.py plots_v102_chromatograms_rt-windows_combined/peak_windows.csv plots_v102_chromatograms_rt-windows_combined --no-re-extract-windows
"""

import os
import re
import sys
import argparse
import shutil
import csv
import math
from collections import defaultdict
import numpy as np
import pandas as pd

# Use non-interactive backend before any matplotlib import to avoid exit code 5 / display crashes when running headless
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _has_modifications(m):
    """True if peptide has any modification (not empty, -, or nan)."""
    if m is None or (isinstance(m, float) and pd.isna(m)):
        return False
    s = str(m).strip().lower()
    return s not in ('', '-', 'nan', 'none')


def _dedupe_peptides_by_mz(peptides):
    """
    Among peptides with the same (charge, rounded m/z), keep at most one per m/z
    (the one with highest total_area). Peptides without valid mz_theoretical are left as-is.
    """
    MZ_DECIMALS = 3
    def _mz_key(p):
        mz = p.get('mz_theoretical') or p.get('mz')
        if mz is None or (isinstance(mz, float) and (pd.isna(mz) or mz <= 0)):
            return None
        try:
            return round(float(mz), MZ_DECIMALS)
        except (TypeError, ValueError):
            return None
    def _total_area_val(p):
        v = p.get('total_area')
        if v is None or (isinstance(v, float) and (pd.isna(v) or v < 0)):
            return 0.0
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0
    groups = {}
    for p in peptides:
        mz_k = _mz_key(p)
        ch = p.get('charge', 0)
        try:
            ch = int(ch) if ch not in (None, '') else 0
        except (TypeError, ValueError):
            ch = 0
        if mz_k is not None:
            key = (ch, mz_k)
            if key not in groups:
                groups[key] = []
            groups[key].append(p)
        else:
            # No m/z: keep as singleton group
            key = ('no_mz', id(p))
            groups[key] = [p]
    out = []
    for key, group in groups.items():
        if key[0] == 'no_mz':
            out.extend(group)
        else:
            # Keep the one with highest total_area
            best = max(group, key=lambda p: (_total_area_val(p), p.get('apex_intensity') or 0))
            out.append(best)
    return out


def thin_peptides_by_collection_overlap(peptides, max_overlap=3):
    """
    Reduce the peptide list so that at every retention time, no more than max_overlap
    collection windows overlap. Peptides without collection_min_rt/collection_max_rt
    are left in the list and not counted for overlap.
    When a region has > max_overlap overlapping windows, remove the lowest-intensity
    peptide first (so primary keeps the "best" peaks): by lowest total_area, then
    lowest apex_intensity. Use n_candidates as tie-breaker only (when tied, prefer
    to remove the one with more candidate peaks so we keep the cleaner peak).
    Returns (kept_list, removed_list) so callers can use removed for backfill.
    """
    def _has_collection(p):
        cmin = p.get('collection_min_rt')
        cmax = p.get('collection_max_rt')
        if cmin is None or cmax is None:
            return False
        try:
            cmin, cmax = float(cmin), float(cmax)
            return not (cmin != cmin or cmax != cmax or cmin >= cmax)
        except (TypeError, ValueError):
            return False

    def _total_area_val(p):
        v = p.get('total_area')
        if v is None or (isinstance(v, float) and (pd.isna(v) or v < 0)):
            return 0.0
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0

    def _apex_val(p):
        v = p.get('apex_intensity')
        if v is None or (isinstance(v, float) and (pd.isna(v) or v < 0)):
            return 0.0
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0

    def _n_candidates_val(p):
        v = p.get('n_candidates')
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return 0
        try:
            return int(v)
        except (TypeError, ValueError):
            return 0

    # Remove the lowest-intensity peptide first so the kept set retains the best peaks.
    # Key: (total_area, apex) ascending so min = remove lowest; use -n_candidates so when tied we remove the one with more candidates (keep cleaner peak).
    def _remove_key(p):
        return (_total_area_val(p), _apex_val(p), -_n_candidates_val(p))

    remaining = list(peptides)
    removed = []
    with_collection = [p for p in remaining if _has_collection(p)]
    if not with_collection:
        return (remaining, removed)

    critical_rts = set()
    for p in with_collection:
        try:
            critical_rts.add(float(p['collection_min_rt']))
            critical_rts.add(float(p['collection_max_rt']))
        except (TypeError, ValueError):
            pass
    critical_rts = sorted(critical_rts)

    def count_spanning_at(rt, pool):
        return [p for p in pool if _has_collection(p)
                and float(p['collection_min_rt']) <= rt <= float(p['collection_max_rt'])]

    while True:
        spanning_at_some = []
        for rt in critical_rts:
            span = count_spanning_at(rt, remaining)
            if len(span) > max_overlap:
                spanning_at_some.append((rt, span))
        if not spanning_at_some:
            break
        rt, span = max(spanning_at_some, key=lambda x: len(x[1]))
        # Remove the lowest-intensity peptide (min total_area, then apex); n_candidates only as tie-breaker
        to_remove = min(span, key=_remove_key)
        remaining.remove(to_remove)
        removed.append(to_remove)

    return (remaining, removed)


def _m0_envelope_ok(p):
    """True if peptide has M+0 > M+1 or M+1 > M+2 (or no data so we don't exclude). Used for backfill pool."""
    v = p.get('m0_gt_m1_gt_m2')
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return True  # no data: allow for backward compatibility
    if isinstance(v, bool):
        return v
    if isinstance(v, str) and v.strip().lower() in ('false', '0', 'no'):
        return False
    return True


def _m0_envelope_ok_strict(p):
    """True only when M+0 > M+1 or M+1 > M+2 is explicitly True. Every peptide gets True or False from the chromatogram step (missing is set to False when building the grid)."""
    v = p.get('m0_gt_m1_gt_m2')
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return False  # missing/unknown => exclude
    if isinstance(v, bool):
        return v is True
    if isinstance(v, str) and v.strip().lower() in ('true', '1', 'yes'):
        return True
    if isinstance(v, (int, float)) and v == 1:
        return True
    return False


def _peptide_key(p):
    return (p.get('peptide_seq', ''), p.get('charge', 0), str(p.get('modifications', '-') or '-').strip() or '-')


# Isotope spacing in m/z: 13C-12C mass diff / charge (used for charge-state consistency)
C13_C12_DIFF_MZ = 1.0033548378


def _compute_charge_state_consistency(peptides, rt_tolerance_sec=45.0, isotope_span_tolerance=0.4):
    """
    Set charge-state consistency fields on each peptide (in place).
    - isotope_spacing_ok: True if observed m/z span matches expected spacing for charge (M0..M+2 ~ 2*C13_C12/charge); False if inconsistent; None if no m/z data.
    - Same peptide (seq, mods) across charges: require RT within rt_tolerance_sec.
    - n_charge_states: number of charge states observed for this (seq, mods).
    - charge_state_rt_consistent: True if single charge or multiple charges with RT within tolerance.
    - charge_state_consistent: True when (isotope_spacing_ok is not False) and charge_state_rt_consistent.
    - multi_charge_agreement: True when n_charge_states >= 2 and RT consistent (strong identity certainty).
    """
    # Group by (peptide_seq, modifications) to get multi-charge RT consistency
    by_identity = defaultdict(list)
    for p in peptides:
        seq = (p.get('peptide_seq') or '').strip()
        mods = str(p.get('modifications', '-') or '-').strip() or '-'
        by_identity[(seq, mods)].append(p)
    for (seq, mods), group in by_identity.items():
        n_charge = len(group)
        # Apex RT: center of detected peak or collection window
        rts = []
        for p in group:
            lo = p.get('detected_peak_min_rt') or p.get('collection_min_rt')
            hi = p.get('detected_peak_max_rt') or p.get('collection_max_rt')
            if lo is not None and hi is not None:
                try:
                    rts.append((float(lo) + float(hi)) / 2.0)
                except (TypeError, ValueError):
                    pass
        rt_consistent = True
        if n_charge >= 2 and rts:
            rt_span = max(rts) - min(rts)
            rt_consistent = rt_span <= rt_tolerance_sec
        for p in group:
            p['n_charge_states'] = n_charge
            p['charge_state_rt_consistent'] = rt_consistent
            p['multi_charge_agreement'] = (n_charge >= 2 and rt_consistent)
    # Per-peptide: isotope spacing vs expected for charge
    for p in peptides:
        charge = p.get('charge')
        try:
            z = int(charge) if charge not in (None, '') and not (isinstance(charge, float) and (charge != charge)) else 0
        except (TypeError, ValueError):
            z = 0
        mz_lo = p.get('mz_observed_min')
        mz_hi = p.get('mz_observed_max')
        if mz_lo is None or mz_hi is None or z <= 0:
            p['isotope_spacing_ok'] = None
        else:
            try:
                mz_lo = float(mz_lo)
                mz_hi = float(mz_hi)
            except (TypeError, ValueError):
                p['isotope_spacing_ok'] = None
                continue
            observed_span = mz_hi - mz_lo
            expected_spacing = C13_C12_DIFF_MZ / z
            # M0, M+1, M+2 -> 2 gaps -> expected span ~ 2 * spacing
            expected_span = 2.0 * expected_spacing
            if expected_span <= 0:
                p['isotope_spacing_ok'] = None
            else:
                ratio = observed_span / expected_span
                # Observed span often from PSM precursor range or isolation window, not isotope spread.
                # Only treat as False when ratio is in a "plausible but wrong" range; else None (don't fail peptide).
                if ratio < 0.2 or ratio > 2.0:
                    p['isotope_spacing_ok'] = None  # unclear data (e.g. zero span or isolation window)
                else:
                    p['isotope_spacing_ok'] = (1.0 - isotope_span_tolerance) <= ratio <= (1.0 + isotope_span_tolerance)
        # Combined: consistent only if spacing not False and RT consistent
        p['charge_state_consistent'] = (p.get('isotope_spacing_ok') is not False) and p.get('charge_state_rt_consistent', True)


def _get_collection_bounds(p):
    """Return (cmin, cmax) in seconds for overlap checks; fallback to integration window if collection missing."""
    def _valid_interval(lo, hi):
        if lo is None or hi is None:
            return False
        try:
            lo, hi = float(lo), float(hi)
            if lo != lo or hi != hi or lo >= hi:  # NaN or invalid
                return False
            return True
        except (TypeError, ValueError):
            return False

    cmin = p.get('collection_min_rt')
    cmax = p.get('collection_max_rt')
    if _valid_interval(cmin, cmax):
        return (float(cmin), float(cmax))
    min_rt = p.get('detected_peak_min_rt')
    max_rt = p.get('detected_peak_max_rt')
    if _valid_interval(min_rt, max_rt):
        return (float(min_rt), float(max_rt))
    rt = p.get('rt', 0.0)
    try:
        rt = float(rt)
        return (rt - 10.0, rt + 10.0)
    except (TypeError, ValueError):
        return (0.0, 20.0)


def _compute_drift_limits_per_channel(peptides_for_rt_grid, num_channels=3):
    """
    For each peptide, compute drift_min and drift_max (full peak drift window).
    Full window = collection ± buffer; no trimming. Channel assignment already ensures
    these full windows do not overlap within a channel.
    """
    rows = []
    for ch in range(num_channels):
        peptides_ch = [p for p in peptides_for_rt_grid if p.get('channel') == ch]
        peptides_ch_rt = sorted(peptides_ch, key=lambda p: (float(p.get('collection_min_rt') or p.get('detected_peak_min_rt') or 0)))
        for p in peptides_ch_rt:
            cmin, cmax = _get_collection_bounds(p)
            if cmin is None or cmax is None:
                continue
            buf_sec = _peak_drift_buffer_sec(cmin, cmax)
            drift_min = cmin - buf_sec
            drift_max = cmax + buf_sec
            if drift_max <= drift_min:
                drift_max = drift_min + max(1.0, (cmax - cmin) * 0.1)
            rows.append((p, drift_min, drift_max))
    return rows


def _total_area_for_channel_sort(p):
    """Value for sorting: higher = earlier in list. Missing/invalid -> 0 so they sort last."""
    v = p.get('total_area')
    if v is None or (isinstance(v, float) and (v != v or v <= 0)):
        return 0.0
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _windows_overlap(p, q):
    """True if collection windows of p and q overlap (interval overlap)."""
    a_start, a_end = _get_collection_bounds(p)
    b_start, b_end = _get_collection_bounds(q)
    if a_start is None or a_end is None or b_start is None or b_end is None:
        return False
    return not (a_end <= b_start or b_end <= a_start)


# Peaks that only "touch" on log scale (tails overlap at noise floor) are bad scheduling candidates:
# treat them as conflicting so they don't share a channel (deconvolution, window packing, channel conflicts).
LOG_SCALE_TOUCH_BUFFER_SEC = 20.0
# Required gap (seconds) between collection windows within a channel so peaks never touch or overlap.
CHANNEL_COLLECTION_GAP_SEC = 1.0


def _windows_overlap_or_touch(p, q, buffer_sec=None):
    """
    True if collection windows overlap OR are within buffer_sec of each other (log-scale touch).
    Peaks that look separate on linear scale but blend on log scale have tails at the noise floor;
    treat them as conflicting for channel assignment.
    """
    if buffer_sec is None:
        buffer_sec = LOG_SCALE_TOUCH_BUFFER_SEC
    a_start, a_end = _get_collection_bounds(p)
    b_start, b_end = _get_collection_bounds(q)
    if a_start is None or a_end is None or b_start is None or b_end is None:
        return False
    a_start, a_end = float(a_start), float(a_end)
    b_start, b_end = float(b_start), float(b_end)
    if _windows_overlap(p, q):
        return True
    # Separation when disjoint: one of (b_start - a_end), (a_start - b_end) is positive
    gap_sep = max(b_start - a_end, a_start - b_end)
    return gap_sep < buffer_sec


def max_overlap_depth(peptides):
    """
    Sweep-line: maximum number of collection windows overlapping at any RT.
    Returns (max_depth,) for interval graph; layers_needed >= ceil(max_depth / 3).
    """
    events = []
    for p in peptides:
        cmin, cmax = _get_collection_bounds(p)
        if cmin is None or cmax is None:
            continue
        try:
            a, b = float(cmin), float(cmax)
            if a >= b:
                continue
            events.append((a, 1))
            events.append((b, -1))
        except (TypeError, ValueError):
            continue
    if not events:
        return 0
    # Sort by time; at same time process ends before starts so we don't double-count
    events.sort(key=lambda x: (x[0], x[1]))
    depth = 0
    max_depth = 0
    for _, delta in events:
        depth += delta
        max_depth = max(max_depth, depth)
    return max_depth


def _priority_signal(p):
    """Priority score for packing: higher = place first. Uses relative_area or total_area."""
    v = p.get('relative_area_within_selection')
    if v is None or (isinstance(v, float) and (v != v or v < 0)):
        v = p.get('total_area')
    if v is None or (isinstance(v, float) and (v != v or v <= 0)):
        return 0.0
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def _rt_center(p):
    """Center of collection/integration window (seconds) for tie-breaking; earlier RT preferred."""
    cmin, cmax = _get_collection_bounds(p)
    if cmin is not None and cmax is not None:
        return (float(cmin) + float(cmax)) / 2.0
    return 0.0


def _n_significant_fragments(p):
    """
    Number of significant fragments for tie-breaking (higher = prefer when signal ties).
    Uses significant_frag_count if available, else number of single-AA overhang
    positions (fragment-supported).
    """
    v = p.get('significant_frag_count')
    if v is not None:
        try:
            n = int(float(v))
            if n >= 0:
                return n
        except (TypeError, ValueError):
            pass
    pos = p.get('single_aa_positions')
    if isinstance(pos, dict):
        return len(pos)
    if isinstance(pos, str) and pos.strip():
        return len([x.strip() for x in pos.strip().split(',') if x.strip()])
    return 0


def assign_peptides_to_channels_by_layers(peptides, channels_per_layer=3):
    """
    Assign high-signal (≥1%) peptides into 3-channel sets. Full drift window per peptide; no overlap within channel.

    - First 3-channel set: fill as many ≥1% as fit (each peptide at most once in that set).
    - Next 3-channel set: (1) place any remaining ≥1% that didn't fit in the first set;
      (2) then fill gaps with repeats from the first set (same peptide can appear in another set).
    - Continue adding sets until all ≥1% are assigned in at least one channel.
    - Uses full drift window (collection ± buffer) for fit; no overlap means drift_min >= last drift_max in channel.

    Mutates p['channel'] for each p in peptides. Returns (num_channels, repeat_copies) where repeat_copies
    are dict copies placed as repeats (is_repeat_in_channel=True); caller should use kept + repeat_copies for downstream.
    """
    if not peptides:
        return (1, [])
    CH = channels_per_layer
    # Drift bounds for each peptide; sort by drift_lo for placement
    with_drift = []
    for p in peptides:
        d_lo, d_hi = _get_drift_bounds(p)
        if d_lo is None or d_hi is None:
            d_lo, d_hi = 0.0, 1.0
        if d_lo >= d_hi:
            d_hi = d_lo + 1.0
        with_drift.append((p, d_lo, d_hi))
    with_drift.sort(key=lambda x: (x[1], x[2]))

    def _channel_end(ch_list):
        """Max drift_max in channel (list of (p, d_lo, d_hi) or list of p with _drift_* stored)."""
        if not ch_list:
            return -float('inf')
        ends = []
        for it in ch_list:
            if isinstance(it, tuple):
                _, _, d_hi = it
                ends.append(d_hi)
            else:
                ends.append(it.get('_drift_max', -float('inf')))
        return max(ends) if ends else -float('inf')

    def _slack_if_fits_drift(ch_tuples, d_lo, d_hi):
        """If (d_lo, d_hi) fits in channel (no overlap: d_lo >= last drift_max), return slack; else inf."""
        end = _channel_end(ch_tuples)
        if d_lo >= end:
            return d_lo - end
        return float('inf')

    def _gaps_in_channel(ch_tuples, max_rt_sec=None):
        """Return list of (gap_start, gap_end) in channel. Uses drift bounds."""
        if not ch_tuples:
            return [(0.0, float(max_rt_sec or 7200))]
        sorted_ch = sorted(ch_tuples, key=lambda t: t[1])
        gaps = []
        last_end = 0.0
        for (p, d_lo, d_hi) in sorted_ch:
            if d_lo > last_end + 1e-6:
                gaps.append((last_end, d_lo))
            last_end = max(last_end, d_hi)
        if max_rt_sec is not None and last_end < max_rt_sec - 1e-6:
            gaps.append((last_end, float(max_rt_sec)))
        return gaps

    # Layer 0: place as many as fit; track (p, d_lo, d_hi) per channel so we have drift for gap/repeat
    layers = []  # each layer: list of CH lists of (p, d_lo, d_hi)
    unplaced = []  # (p, d_lo, d_hi) not yet in any layer
    max_rt_sec = None
    for p, d_lo, d_hi in with_drift:
        pk = _peptide_key(p)
        p['_drift_min'], p['_drift_max'] = d_lo, d_hi
        placed = False
        if not layers:
            layers.append([[] for _ in range(CH)])
        for li, layer in enumerate(layers):
            layer_keys = {_peptide_key(t[0]) for ch in range(CH) for t in layer[ch]}
            if pk in layer_keys:
                continue
            ch_slacks = [(_slack_if_fits_drift(layer[ch], d_lo, d_hi), ch) for ch in range(CH)]
            valid = [(slack, ch) for slack, ch in ch_slacks if slack != float('inf')]
            if not valid:
                continue
            best_ch = min(valid, key=lambda x: (x[0], x[1]))[1]
            layer[best_ch].append((p, d_lo, d_hi))
            placed = True
            break
        if not placed:
            unplaced.append((p, d_lo, d_hi))

    # Add layers: place unplaced first, then fill gaps with repeats from previous layers
    repeat_copies = []
    max_rt_sec = max((t[2] for t in with_drift), default=7200)
    MAX_LAYERS = 20  # cap to prevent infinite loop when many peptides overlap
    while unplaced and len(layers) < MAX_LAYERS:
        new_layer = [[] for _ in range(CH)]
        # (1) Place as many unplaced as fit in this layer
        still_unplaced = []
        for (p, d_lo, d_hi) in unplaced:
            pk = _peptide_key(p)
            placed = False
            layer_keys = {_peptide_key(t[0]) for ch in range(CH) for t in new_layer[ch]}
            if pk in layer_keys:
                still_unplaced.append((p, d_lo, d_hi))
                continue
            ch_slacks = [(_slack_if_fits_drift(new_layer[ch], d_lo, d_hi), ch) for ch in range(CH)]
            valid = [(slack, ch) for slack, ch in ch_slacks if slack != float('inf')]
            if not valid:
                still_unplaced.append((p, d_lo, d_hi))
                continue
            best_ch = min(valid, key=lambda x: (x[0], x[1]))[1]
            new_layer[best_ch].append((p, d_lo, d_hi))
            placed = True
        prev_unplaced_len = len(unplaced)
        unplaced = still_unplaced
        if len(unplaced) >= prev_unplaced_len and len(unplaced) > 0:
            # No progress: remaining peptides overlap each other; stop adding layers
            break
        # (2) Fill gaps in new_layer with repeats from all previous layers
        prev_peptides = []
        for li, layer in enumerate(layers):
            for ch in range(CH):
                for t in layer[ch]:
                    prev_peptides.append(t)
        for ch in range(CH):
            gaps = _gaps_in_channel(new_layer[ch], max_rt_sec)
            layer_keys_ch = {_peptide_key(t[0]) for t in new_layer[ch]}
            for (gap_start, gap_end) in gaps:
                if gap_end - gap_start < 5.0:
                    continue
                best_repeat = None
                best_width = -1.0
                for (prev_p, pd_lo, pd_hi) in prev_peptides:
                    if _peptide_key(prev_p) in layer_keys_ch:
                        continue
                    if pd_lo >= gap_start and pd_hi <= gap_end and (pd_hi - pd_lo) > best_width:
                        best_width = pd_hi - pd_lo
                        best_repeat = (prev_p, pd_lo, pd_hi)
                if best_repeat is None:
                    continue
                prev_p, pd_lo, pd_hi = best_repeat
                cp = dict(prev_p)
                cp['is_repeat_in_channel'] = True
                cp['_drift_min'], cp['_drift_max'] = pd_lo, pd_hi
                repeat_copies.append(cp)
                new_layer[ch].append((cp, pd_lo, pd_hi))
                layer_keys_ch.add(_peptide_key(cp))
        layers.append(new_layer)
        if not unplaced:
            break

    # Flatten: assign global channel index; drop _drift_* from peptides
    num_channels = CH * len(layers)
    for li, layer in enumerate(layers):
        for ch, peak_list in enumerate(layer):
            for t in peak_list:
                p = t[0]
                p['channel'] = li * CH + ch
                if '_drift_min' in p:
                    del p['_drift_min']
                if '_drift_max' in p:
                    del p['_drift_max']
    for cp in repeat_copies:
        if '_drift_min' in cp:
            del cp['_drift_min']
        if '_drift_max' in cp:
            del cp['_drift_max']

    # Reorder within each layer by priority (optional)
    layer_lists = [[[] for _ in range(CH)] for _ in range(len(layers))]
    for p in peptides + repeat_copies:
        ch = p.get('channel', 0)
        if 0 <= ch < num_channels:
            li, cj = ch // CH, ch % CH
            layer_lists[li][cj].append(p)
    for layer in layer_lists:
        _reshuffle_layer_channels_drift(layer, CH)
    for li, layer in enumerate(layer_lists):
        for ch, peak_list in enumerate(layer):
            for p in peak_list:
                p['channel'] = li * CH + ch
    return (num_channels, repeat_copies)


def _peak_score_for_reshuffle(p, alpha=0.3, tau_sec=600.0, coverage_weight=0.01):
    """
    Weighted score for reshuffle: earlier channels should be dense and useful.
    score = signal * (1 + α * exp(-rt_center/τ)) + coverage_weight * duration
    Rewards high signal, early RT, and duration (coverage) so one giant peak doesn't dominate.
    """
    sig = _priority_signal(p)
    cmin, cmax = _get_collection_bounds(p)
    rt_center = (float(cmin) + float(cmax)) / 2.0 if (cmin is not None and cmax is not None) else 0.0
    duration = (float(cmax) - float(cmin)) if (cmin is not None and cmax is not None and cmax > cmin) else 0.0
    early_bonus = 1.0 + alpha * math.exp(-rt_center / tau_sec) if tau_sec > 0 else 1.0
    return sig * early_bonus + coverage_weight * duration


def _drift_windows_overlap(p, q):
    """True if full drift windows of p and q overlap (used for reshuffle within layer)."""
    d_lo_p, d_hi_p = _get_drift_bounds(p)
    d_lo_q, d_hi_q = _get_drift_bounds(q)
    if d_lo_p is None or d_hi_p is None or d_lo_q is None or d_hi_q is None:
        return _windows_overlap_or_touch(p, q)
    return not (d_hi_p <= d_lo_q or d_hi_q <= d_lo_p)


def _reshuffle_layer_channels_drift(layer, CH):
    """Like _reshuffle_layer_channels but use drift bounds for overlap (no drift overlap within channel)."""
    def _channel_score(ch_list):
        return sum(_peak_score_for_reshuffle(p) for p in ch_list)
    def _scores_vec():
        return tuple(_channel_score(layer[ch]) for ch in range(CH))
    def _fits_in(ch_list, peak):
        return not any(_drift_windows_overlap(peak, q) for q in ch_list)
    improved = True
    while improved:
        improved = False
        old_vec = _scores_vec()
        for ch_lo in range(CH):
            for ch_hi in range(ch_lo + 1, CH):
                for p in list(layer[ch_hi]):
                    if not _fits_in(layer[ch_lo], p):
                        continue
                    layer[ch_hi].remove(p)
                    layer[ch_lo].append(p)
                    new_vec = _scores_vec()
                    if new_vec > old_vec:
                        old_vec = new_vec
                        improved = True
                        break
                    else:
                        layer[ch_lo].remove(p)
                        layer[ch_hi].append(p)
                if improved:
                    break
            if improved:
                break


def _reshuffle_layer_channels(layer, CH):
    """
    Within one layer, try to improve so channel 0 has highest weighted score, then ch1, ch2.
    Score = signal * (1 + α*exp(-rt_center/τ)) + coverage_weight*duration (dense, useful, not one giant peak).
    Accept a move only if (score_ch0, score_ch1, score_ch2) improves lexicographically.
    """
    def _channel_score(ch_list):
        return sum(_peak_score_for_reshuffle(p) for p in ch_list)

    def _scores_vec():
        return tuple(_channel_score(layer[ch]) for ch in range(CH))

    def _fits_in(ch_list, peak):
        return not any(_windows_overlap_or_touch(peak, q) for q in ch_list)

    improved = True
    while improved:
        improved = False
        old_vec = _scores_vec()
        for ch_lo in range(CH):
            for ch_hi in range(ch_lo + 1, CH):
                for p in list(layer[ch_hi]):
                    if not _fits_in(layer[ch_lo], p):
                        continue
                    layer[ch_hi].remove(p)
                    layer[ch_lo].append(p)
                    new_vec = _scores_vec()
                    # Accept only if lexicographically better (earlier channels gain more)
                    if new_vec > old_vec:
                        old_vec = new_vec
                        improved = True
                        break
                    else:
                        layer[ch_lo].remove(p)
                        layer[ch_hi].append(p)
                if improved:
                    break
            if improved:
                break
    return


def assign_peptides_to_channels(peptides, num_channels, gap_sec=None, preserve_order=False, use_collection_bounds=False):
    """
    Assign peptides to channels. No overlap within a channel.
    When use_collection_bounds=True: use collection window (cmin, cmax) for overlap - next peptide's
    collection_min >= previous collection_max + gap_sec in channel. Collection windows never overlap.
    When False: use drift window (collection ± 30s) - next drift_min >= previous drift_max. More spacing.
    Among channels where the peptide fits, prefer the channel with the earliest end time.
    When no channel fits without overlap, set channel=-1 (caller should filter these out).
    When preserve_order=True, process in input order (e.g. total_area desc); otherwise sort by start (earliest first).
    """
    min_gap = max(0.0, float(gap_sec)) if gap_sec is not None else 0.0
    with_bounds = []
    for p in peptides:
        if use_collection_bounds:
            lo, hi = _get_collection_bounds(p)
            with_bounds.append((p, lo, hi))
        else:
            drift_lo, drift_hi = _get_drift_bounds(p)
            with_bounds.append((p, drift_lo, drift_hi))
    if not preserve_order:
        with_bounds.sort(key=lambda x: (x[1] if x[1] is not None else 0.0, x[2] if x[2] is not None else 0.0))
    channel_end_times = [-float('inf')] * num_channels
    for p, lo, hi in with_bounds:
        if lo is None or hi is None:
            lo, hi = 0.0, 1.0
        if lo >= hi:
            hi = lo + 1.0
        # No overlap: next peptide's start must be >= previous end + gap
        candidates = [ch for ch in range(num_channels) if lo >= channel_end_times[ch] + min_gap]
        if candidates:
            ch = min(candidates, key=lambda k: channel_end_times[k])
            p['channel'] = ch
            channel_end_times[ch] = hi
        else:
            p['channel'] = -1
    return peptides


def assign_peptides_to_three_channels(peptides, num_channels=3):
    """Convenience wrapper: assign to 3 channels (or num_channels)."""
    return assign_peptides_to_channels(peptides, num_channels)


def reorder_channels_by_relative_area(peptides, num_channels):
    """
    After channels are maximally filled (backfill + repeat), remap channel indices so
    sum of relative_area_within_selection per channel is descending: Ch1 has the highest
    collected signal, Ch2 the next, etc. Mutates p['channel'].
    """
    if not peptides or num_channels <= 0:
        return
    sum_rel = [0.0] * num_channels
    for p in peptides:
        ch = p.get('channel')
        if ch is not None and 0 <= ch < num_channels:
            rel = p.get('relative_area_within_selection')
            if rel is not None:
                try:
                    r = float(rel)
                    if r >= 0 and r == r:
                        sum_rel[ch] += r
                except (TypeError, ValueError):
                    pass
    # sorted_by_sum: indices of channels in descending order of sum (highest first)
    sorted_by_sum = sorted(range(num_channels), key=lambda c: -sum_rel[c])
    # old_ch -> new_ch: new_ch 0 = channel with highest sum, etc.
    old_to_new = {old: new for new, old in enumerate(sorted_by_sum)}
    for p in peptides:
        ch = p.get('channel')
        if ch is not None and ch in old_to_new:
            p['channel'] = old_to_new[ch]


def reorder_channels_by_relative_area_within_layers(peptides, num_channels, channels_per_layer=3):
    """
    Reorder channel indices so within each layer the channel with highest total
    relative_area is first, etc. Does not mix layers (preserves layer packing).
    Mutates p['channel'].
    """
    if not peptides or num_channels <= 0:
        return
    CH = channels_per_layer
    sum_rel = [0.0] * num_channels
    for p in peptides:
        ch = p.get('channel')
        if ch is not None and 0 <= ch < num_channels:
            rel = p.get('relative_area_within_selection')
            if rel is not None:
                try:
                    r = float(rel)
                    if r >= 0 and r == r:
                        sum_rel[ch] += r
                except (TypeError, ValueError):
                    pass
    old_to_new = {}
    num_layers = (num_channels + CH - 1) // CH
    for li in range(num_layers):
        ch_start = li * CH
        ch_end = min(ch_start + CH, num_channels)
        indices = list(range(ch_start, ch_end))
        # Sort this layer's channels by descending sum (highest first within layer)
        indices_sorted = sorted(indices, key=lambda c: -sum_rel[c])
        for new_off, old_ch in enumerate(indices_sorted):
            old_to_new[old_ch] = ch_start + new_off
    for p in peptides:
        ch = p.get('channel')
        if ch is not None and ch in old_to_new:
            p['channel'] = old_to_new[ch]


def _get_channel_gaps(peptides_in_channel, min_overlap=1, max_rt=None, min_gap_width=10.0):
    """Return list of (a, b) RT intervals (seconds) where a new window could fit (fewer than min_overlap spanning)."""
    def _has_collection(p):
        cmin = p.get('collection_min_rt')
        cmax = p.get('collection_max_rt')
        if cmin is None or cmax is None:
            return False
        try:
            cmin, cmax = float(cmin), float(cmax)
            return not (cmin != cmin or cmax != cmax or cmin >= cmax)
        except (TypeError, ValueError):
            return False

    with_coll = [p for p in peptides_in_channel if _has_collection(p)]
    if not with_coll:
        return []
    critical_rts = set()
    for p in with_coll:
        try:
            critical_rts.add(float(p['collection_min_rt']))
            critical_rts.add(float(p['collection_max_rt']))
        except (TypeError, ValueError):
            pass
    critical_rts = sorted(critical_rts)
    if len(critical_rts) < 2:
        return []

    def count_spanning_at(rt, pool):
        return [p for p in pool if _has_collection(p)
                and float(p['collection_min_rt']) <= rt <= float(p['collection_max_rt'])]

    gaps = []
    first_rt = min(critical_rts)
    if first_rt >= min_gap_width:
        gaps.append((0.0, first_rt))
    for i in range(len(critical_rts) - 1):
        a, b = critical_rts[i], critical_rts[i + 1]
        mid = (a + b) / 2.0
        if len(count_spanning_at(mid, peptides_in_channel)) < min_overlap:
            gaps.append((a, b))
    if max_rt is not None:
        try:
            max_rt_f = float(max_rt)
            last_rt = max(critical_rts)
            if max_rt_f > last_rt and (max_rt_f - last_rt) >= min_gap_width:
                gaps.append((last_rt, max_rt_f))
        except (TypeError, ValueError):
            pass
    return gaps


def _window_fits_in_gap(p, gap_a, gap_b):
    """True if p's collection window [cmin, cmax] is contained in (gap_a, gap_b)."""
    try:
        cmin = float(p.get('collection_min_rt'))
        cmax = float(p.get('collection_max_rt'))
    except (TypeError, ValueError):
        return False
    return cmin >= gap_a and cmax <= gap_b and cmin < cmax


def _window_fits_in_gap_with_drift_overlap(p, gap_a, gap_b, max_overlap_frac=0.5):
    """
    True if at least (1 - max_overlap_frac) of p's peak drift window lies inside (gap_a, gap_b).
    E.g. max_overlap_frac=0.5 allows 50% of the drift window to overlap outside the gap (more flexibility for shuffling).
    """
    try:
        cmin = float(p.get('collection_min_rt'))
        cmax = float(p.get('collection_max_rt'))
    except (TypeError, ValueError):
        return False
    if cmin >= cmax or gap_a >= gap_b:
        return False
    buf = _peak_drift_buffer_sec(cmin, cmax)
    drift_min = cmin - buf
    drift_max = cmax + buf
    drift_width = drift_max - drift_min
    if drift_width <= 0:
        return _window_fits_in_gap(p, gap_a, gap_b)
    overlap_start = max(drift_min, gap_a)
    overlap_end = min(drift_max, gap_b)
    inside_width = max(0.0, overlap_end - overlap_start)
    return inside_width >= (1.0 - max_overlap_frac) * drift_width


def shuffle_higher_signal_into_earlier_channels(peptides, num_channels, max_rt=None, channels_per_layer=3):
    """
    Try to move higher-signal windows from later channels into earlier channels when they fit in a gap.
    Restricted to same layer only (ch_from // channels_per_layer == ch_to // channels_per_layer) so
    layer packing is preserved. Allows up to 50% overlap of the peak drift window for flexibility.
    Greedy until no moves possible. Mutates p['channel'].
    """
    if not peptides or num_channels <= 1:
        return
    CH = channels_per_layer
    moved = True
    while moved:
        moved = False
        for ch_to in range(num_channels):
            in_to = [p for p in peptides if p.get('channel') == ch_to]
            gaps = _get_channel_gaps(in_to, min_overlap=1, max_rt=max_rt, min_gap_width=10.0)
            if not gaps:
                continue
            layer_to = ch_to // CH
            # Only consider peptides from later channels *in the same layer*
            best_p = None
            best_rel = -1.0
            for ch_from in range(ch_to + 1, num_channels):
                if (ch_from // CH) != layer_to:
                    continue
                for p in peptides:
                    if p.get('channel') != ch_from:
                        continue
                    rel = p.get('relative_area_within_selection') or 0
                    try:
                        rel = float(rel)
                    except (TypeError, ValueError):
                        rel = 0.0
                    if rel <= best_rel:
                        continue
                    for (a, b) in gaps:
                        if _window_fits_in_gap_with_drift_overlap(p, a, b, max_overlap_frac=0.5):
                            best_p, best_rel = p, rel
                            break
            if best_p is not None:
                best_p['channel'] = ch_to
                moved = True
                break
        # if moved, re-run from top so ch_to gets updated gaps next iteration
    return


def set_repeat_outline_by_earliest_channel(peptides, num_channels):
    """
    For each peptide identity (seq, charge, mods): earliest channel = filled (shaded);
    all later occurrences = outline only (is_repeat_in_channel=True).
    Mutates p['is_repeat_in_channel'].
    """
    def _key(p):
        return (p.get('peptide_seq'), p.get('charge'), str(p.get('modifications', '-') or '-').strip() or '-')

    from collections import defaultdict
    by_key = defaultdict(list)
    for p in peptides:
        ch = p.get('channel')
        if ch is not None and 0 <= ch < num_channels:
            by_key[_key(p)].append((ch, p))
    for key, ch_plist in by_key.items():
        if len(ch_plist) <= 1:
            for _, p in ch_plist:
                p['is_repeat_in_channel'] = False
            continue
        # Sort by channel; earliest channel gets filled, rest outlined
        ch_plist.sort(key=lambda x: x[0])
        for idx, (ch, p) in enumerate(ch_plist):
            p['is_repeat_in_channel'] = (idx > 0)


def _draw_filtration_funnel(funnel_counts, output_path, protein_id=''):
    """
    Draw a funnel schematic: how many peptides passed each filtration step (initial, stages 1–6, backfill, final).
    funnel_counts: dict with keys input, after_total_area, stage1..stage6, backfill_added, final.
    """
    steps = [
        ('Input (with valid windows)', 'input'),
        ('After total_area > 0', 'after_total_area'),
        ('Stage 1: ≥2 single-AA overhangs', 'stage1'),
        ('Stage 2: ≥5 fragments matched', 'stage2'),
        ('Stage 3: q or PEP threshold', 'stage3'),
        ('Stage 4: Envelope (M+0>M+1 or M+1>M+2)', 'stage4'),
        ('Stage 5: Relative area ≥0.01%', 'stage5'),
        ('Stage 6: Charge-state consistency', 'stage6'),
        ('Backfill added', 'backfill_added'),
        ('Final (channels + backfill)', 'final'),
    ]
    labels = [s[0] for s in steps]
    keys = [s[1] for s in steps]
    counts = [funnel_counts.get(k, 0) for k in keys]
    # Backfill is additive; show cumulative at "Final" and as delta at "Backfill added"
    n = len(steps)
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    y_pos = np.arange(n)[::-1]  # top to bottom: input at top, final at bottom
    colors = ['#e0e0e0', '#d0d0d0', '#b8d4e3', '#9ec9e3', '#84bee3', '#6ab3e3', '#50a8e3', '#3d9dd9', '#ffcc80', '#2e7d32']
    if len(colors) < n:
        colors = colors + ['#2e7d32'] * (n - len(colors))
    bars = ax.barh(y_pos, counts, height=0.6, color=colors[:n], edgecolor='gray', linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9, fontfamily='serif')
    ax.set_xlabel('Number of peptides', fontsize=11, fontweight='bold', fontfamily='serif')
    ax.set_title(f'{protein_id} – Filtration funnel' if protein_id else 'Filtration funnel', fontsize=12, fontweight='bold', fontfamily='serif')
    for i, (bar, c) in enumerate(zip(bars, counts)):
        if c > 0:
            ax.text(c + (max(counts) * 0.01), bar.get_y() + bar.get_height() / 2, str(int(c)),
                    fontsize=10, fontweight='bold', va='center', ha='left', fontfamily='serif')
    ax.set_xlim(0, max(counts) * 1.25 if counts else 1)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white', pad_inches=0.25)
    plt.close(fig)


def find_min_channels(peptides, gap_sec=None):
    """
    Return the minimum number of channels needed to fit all peptides so that after
    assignment no two full windows (peak drift + collection + integration) overlap within any channel.
    Tries k=1,2,3,... until assign_peptides_to_channels(peptides, k) yields no overlap of full drift windows.
    Peptides must have collection_min_rt/collection_max_rt.
    """
    if not peptides:
        return 0
    for k in range(1, 501):  # cap at 500 channels
        copies = [dict(p) for p in peptides]
        assign_peptides_to_channels(copies, k, gap_sec=gap_sec)
        has_overlap = False
        for ch in range(k):
            ch_peps = [p for p in copies if p.get('channel') == ch]
            ch_peps = sorted(ch_peps, key=lambda p: (_get_drift_bounds(p)[0] if _get_drift_bounds(p)[0] is not None else 0.0))
            for i in range(len(ch_peps) - 1):
                _, prev_drift_max = _get_drift_bounds(ch_peps[i])
                next_drift_min, _ = _get_drift_bounds(ch_peps[i + 1])
                if prev_drift_max is not None and next_drift_min is not None:
                    try:
                        if float(prev_drift_max) > float(next_drift_min):
                            has_overlap = True
                            break
                    except (TypeError, ValueError):
                        pass
            if has_overlap:
                break
        if not has_overlap:
            return k
    return max(1, len(peptides))


def enforce_channel_no_overlap(peptides_for_rt_grid, num_channels=3, min_window_sec=5.0, gap_sec=None):
    """
    Within each channel, ensure no two collection windows overlap (require gap_sec between).
    Sort by collection start. Never crop collection windows. When two collection windows overlap: drop the weaker signal
    (or keep both with full windows if one is a repeat). Drift (black buffer) does not overlap; collection (gray) must not.
    Mutates peptide dicts. Returns the filtered list (may be shorter).
    """
    if gap_sec is None:
        gap_sec = CHANNEL_COLLECTION_GAP_SEC
    def _area(p):
        v = p.get('total_area')
        if v is None or (isinstance(v, float) and (pd.isna(v) or v < 0)):
            return 0.0
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0

    def _coll_bounds(p):
        cmin = p.get('collection_min_rt')
        cmax = p.get('collection_max_rt')
        try:
            a = float(cmin) if cmin is not None else None
            b = float(cmax) if cmax is not None else None
            if a is not None and b is not None and not (a != a or b != b) and a < b:
                return (a, b)
        except (TypeError, ValueError):
            pass
        return (None, None)

    def _int_bounds(p):
        """Integration window: detected_peak or spectrum_window."""
        imin = p.get('detected_peak_min_rt') or p.get('spectrum_window_min_rt')
        imax = p.get('detected_peak_max_rt') or p.get('spectrum_window_max_rt')
        try:
            a = float(imin) if imin is not None else None
            b = float(imax) if imax is not None else None
            if a is not None and b is not None and not (a != a or b != b) and a < b:
                return (a, b)
        except (TypeError, ValueError):
            pass
        return (None, None)

    def _integration_overlap(prev, p):
        """True if prev and p have overlapping integration windows."""
        pa, pb = _int_bounds(prev)
        qa, qb = _int_bounds(p)
        if pa is None or pb is None or qa is None or qb is None:
            return False
        return not (pb < qa or qb < pa)

    result = []
    for ch in range(num_channels):
        peptides_ch = [p for p in peptides_for_rt_grid if p.get('channel') == ch]
        peptides_ch = sorted(peptides_ch, key=lambda p: (float(p.get('collection_min_rt') or 0)))
        channel_result = []
        for p in peptides_ch:
            cmin, cmax = _coll_bounds(p)
            if cmin is None or cmax is None:
                channel_result.append(p)
                continue
            if not channel_result:
                channel_result.append(p)
                continue
            prev = channel_result[-1]
            prev_min, prev_max = _coll_bounds(prev)
            if prev_min is None or prev_max is None:
                channel_result.append(p)
                continue
            # No collection overlap: keep as is
            if cmin >= prev_max + gap_sec:
                channel_result.append(p)
                continue
            # Collection overlap: if integration overlaps, drop the weaker unless one is a repeat — then keep both with full windows (no cropping)
            weaker = p if _area(p) <= _area(prev) else prev
            prev_is_repeat = prev.get('is_repeat_in_channel')
            p_is_repeat = p.get('is_repeat_in_channel')
            if _integration_overlap(prev, p):
                if prev_is_repeat or p_is_repeat:
                    # Keep both with full collection windows (do not crop)
                    channel_result.append(p)
                else:
                    # Neither is repeat: drop the weaker
                    if weaker is prev:
                        channel_result[-1] = p
                    else:
                        pass  # drop p
            else:
                # No integration overlap but collection overlaps: do not crop; drop the weaker
                if weaker is prev:
                    channel_result[-1] = p
                else:
                    pass  # drop p
        result.extend(channel_result)
    return result


def backfill_peptides_from_gaps(kept_peptides, removed_peptides, min_overlap=3, require_passed_psm=False, exclude_keys=None, min_rt=None, max_rt=None, min_gap_width=10.0, min_start_end_gap=1.0, layer_peptides=None):
    """
    Find RT intervals where fewer than min_overlap collection windows overlap (from kept only).
    For each such gap, pick a removed peptide whose collection window fits entirely in the gap;
    add it back with is_backfill=True. Each removed peptide is used at most once.
    layer_peptides: when set (list of peptides in the same 3-channel layer), enforce no repeat
    (candidate key not in layer) and no overlap with any peptide in the layer (so the 3-channel set stays valid).
    exclude_keys: optional set of (peptide_seq, charge, mods) to skip (e.g. already used in a previous pass).
    min_rt, max_rt, min_gap_width, min_start_end_gap: as before.
    """
    def _has_at_least_one_passed(p):
        v = p.get('_scan_count_passed')
        if v is None or (isinstance(v, float) and (pd.isna(v) or v < 0)):
            return False
        return int(v) >= 1

    def _has_collection(p):
        cmin = p.get('collection_min_rt')
        cmax = p.get('collection_max_rt')
        if cmin is None or cmax is None:
            return False
        try:
            cmin, cmax = float(cmin), float(cmax)
            return not (cmin != cmin or cmax != cmax or cmin >= cmax)
        except (TypeError, ValueError):
            return False
    def _total_area_val(p):
        v = p.get('total_area')
        if v is None or (isinstance(v, float) and (pd.isna(v) or v < 0)):
            return 0.0
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0

    def _ms1_trace_ok(p):
        """True if MS1 trace was within 5 ppm of theoretical m/z (from peak_windows). Missing => True for backward compat."""
        v = p.get('ms1_trace_ok')
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return True
        return v is True or (isinstance(v, str) and v.strip().lower() in ('true', '1', 'yes'))

    exclude_keys = exclude_keys or set()
    layer_keys = {_peptide_key(q) for q in (layer_peptides or [])}
    # Pool: has collection window, MS1 within 5 ppm, M+0>M+1 or M+1>M+2, optional PSM requirement, not in exclude_keys, not already in layer (no repeat within 3-channel set)
    removed_pool = [
        p for p in removed_peptides
        if _has_collection(p) and _ms1_trace_ok(p) and _m0_envelope_ok(p)
        and (not require_passed_psm or _has_at_least_one_passed(p))
        and _peptide_key(p) not in exclude_keys
        and _peptide_key(p) not in layer_keys
    ]

    with_coll = [p for p in kept_peptides if _has_collection(p)]
    start_lo = float(min_rt) if min_rt is not None else 0.0
    try:
        _max_rt_f = float(max_rt) if max_rt is not None else start_lo + 7200
    except (TypeError, ValueError):
        _max_rt_f = start_lo + 7200

    if not with_coll:
        # Empty channel: entire run is one gap — fill with backfill from pool
        gaps = [(start_lo, _max_rt_f)] if _max_rt_f > start_lo else []
    else:
        critical_rts = set()
        for p in with_coll:
            try:
                critical_rts.add(float(p['collection_min_rt']))
                critical_rts.add(float(p['collection_max_rt']))
            except (TypeError, ValueError):
                pass
        critical_rts = sorted(critical_rts)
        if len(critical_rts) < 2:
            gaps = [(start_lo, _max_rt_f)] if _max_rt_f > start_lo else []
        else:
            def count_spanning_at(rt, pool):
                return [p for p in pool if _has_collection(p)
                        and float(p['collection_min_rt']) <= rt <= float(p['collection_max_rt'])]

            gaps = []
            first_rt = min(critical_rts)
            last_rt = max(critical_rts)
            if first_rt > start_lo and (first_rt - start_lo) >= min_start_end_gap:
                gaps.append((start_lo, first_rt))
            for i in range(len(critical_rts) - 1):
                a, b = critical_rts[i], critical_rts[i + 1]
                mid = (a + b) / 2.0
                n = len(count_spanning_at(mid, kept_peptides))
                if n < min_overlap:
                    gaps.append((a, b))
            if max_rt is not None:
                try:
                    max_rt_f = float(max_rt)
                    if max_rt_f > last_rt and (max_rt_f - last_rt) >= min_start_end_gap:
                        gaps.append((last_rt, max_rt_f))
                except (TypeError, ValueError):
                    pass

    backfill = []
    used_keys = set()
    for a, b in gaps:
        candidates = []
        for p in removed_pool:
            key = _peptide_key(p)
            if key in used_keys:
                continue
            try:
                cmin = float(p['collection_min_rt'])
                cmax = float(p['collection_max_rt'])
            except (TypeError, ValueError):
                continue
            if cmin >= a and cmax <= b:
                # No overlap with any peptide already in this layer (enforce "at most 3 overlapping at any RT")
                if layer_peptides and any(_windows_overlap_or_touch(p, q) for q in layer_peptides):
                    continue
                candidates.append(p)
        if not candidates:
            continue
        best = max(candidates, key=lambda p: (_total_area_val(p), p.get('apex_intensity') or 0))
        used_keys.add(_peptide_key(best))
        bf = dict(best)
        bf['is_backfill'] = True
        backfill.append(bf)
        # So next gap in this call sees the new peptide for overlap check (no repeat within layer)
        if layer_peptides is not None:
            layer_peptides.append(bf)
            layer_keys.add(_peptide_key(bf))
    return backfill


def backfill_repeat_into_gaps(peptides_in_channel, candidate_pool, min_overlap=1, min_rt=None, max_rt=None, min_gap_width=10.0, min_start_end_gap=1.0):
    """
    Fill remaining RT gaps by reusing peptides already assigned (to this or other channels).
    Same peptide can be repeated any number of times (in different gaps/channels) to maximize coverage.
    Per gap: pack as many non-overlapping candidates as fit (greedy left-to-right) so beginning to end is covered as much as possible.
    Includes start gap (min_rt or 0 to first window) and end gap (last window to max_rt) when applicable.
    Returns list of copies with is_repeat_in_channel=True for outline-only drawing.
    """
    def _has_collection(p):
        cmin = p.get('collection_min_rt')
        cmax = p.get('collection_max_rt')
        if cmin is None or cmax is None:
            return False
        try:
            cmin, cmax = float(cmin), float(cmax)
            return not (cmin != cmin or cmax != cmax or cmin >= cmax)
        except (TypeError, ValueError):
            return False

    def _total_area_val(p):
        v = p.get('total_area')
        if v is None or (isinstance(v, float) and (pd.isna(v) or v < 0)):
            return 0.0
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0

    with_coll = [p for p in peptides_in_channel if _has_collection(p)]
    start_lo = float(min_rt) if min_rt is not None else 0.0
    try:
        _max_rt_f = float(max_rt) if max_rt is not None else start_lo + 7200
    except (TypeError, ValueError):
        _max_rt_f = start_lo + 7200

    if not with_coll:
        # Empty channel: entire run is one gap — fill with repeats from other channels
        gaps = [(start_lo, _max_rt_f)] if _max_rt_f > start_lo else []
    else:
        critical_rts = set()
        for p in with_coll:
            try:
                critical_rts.add(float(p['collection_min_rt']))
                critical_rts.add(float(p['collection_max_rt']))
            except (TypeError, ValueError):
                pass
        critical_rts = sorted(critical_rts)
        if len(critical_rts) < 2:
            gaps = [(start_lo, _max_rt_f)] if _max_rt_f > start_lo else []
        else:
            def count_spanning_at(rt, pool):
                return [p for p in pool if _has_collection(p)
                        and float(p['collection_min_rt']) <= rt <= float(p['collection_max_rt'])]

            gaps = []
            first_rt = min(critical_rts)
            last_rt = max(critical_rts)
            if first_rt > start_lo and (first_rt - start_lo) >= min_start_end_gap:
                gaps.append((start_lo, first_rt))
            for i in range(len(critical_rts) - 1):
                a, b = critical_rts[i], critical_rts[i + 1]
                mid = (a + b) / 2.0
                n = len(count_spanning_at(mid, peptides_in_channel))
                if n < min_overlap:
                    gaps.append((a, b))
            if max_rt is not None:
                try:
                    max_rt_f = float(max_rt)
                    if max_rt_f > last_rt and (max_rt_f - last_rt) >= min_start_end_gap:
                        gaps.append((last_rt, max_rt_f))
                except (TypeError, ValueError):
                    pass

    pool = [p for p in candidate_pool if _has_collection(p)]
    repeats = []
    for gap_idx, (a, b) in enumerate(gaps):
        candidates = []
        is_start_gap = (a <= start_lo + 1.0)  # gap at run start: allow repeat that starts in gap even if it extends past b
        for p in pool:
            try:
                cmin = float(p['collection_min_rt'])
                cmax = float(p['collection_max_rt'])
            except (TypeError, ValueError):
                continue
            if cmin >= a and cmax <= b:
                candidates.append((cmin, cmax, p))
            elif is_start_gap and cmin >= a and cmin < b:
                # Start gap: allow candidate that starts in gap but extends past gap end (so early peptide can appear in every channel)
                candidates.append((cmin, cmax, p))
        if not candidates:
            continue
        # Pack non-overlapping; leave CHANNEL_COLLECTION_GAP_SEC between consecutive repeats (no overlap within channel)
        candidates.sort(key=lambda x: (x[0], -_total_area_val(x[2])))
        last_end = a
        gap_sec = CHANNEL_COLLECTION_GAP_SEC
        for cmin, cmax, p in candidates:
            if cmin >= last_end:
                cp = dict(p)
                cp['is_repeat_in_channel'] = True
                repeats.append(cp)
                last_end = cmax + gap_sec
    return repeats


def _chromatogram_filename_mods_match(filename, mods_norm):
    """Return True if filename's mods part matches mods_norm (-/empty -> nomod)."""
    if not mods_norm or str(mods_norm).strip() in ('-', ''):
        return 'nomod' in filename
    # Filename has safe_mods (e.g. nomod or sanitized); exact match is fragile, so allow nomod only for unmodified
    return True  # accept any if we already matched sequence+charge


def copy_filtered_chromatograms(peptides_for_rt_grid, output_dir, csv_dir, output_base='filtered'):
    """
    Copy individual peptide chromatogram PNGs for the selected peptide set into one directory
    (output_dir/filtered_chromatograms or output_dir/<output_base>_chromatograms) for easy inspection.
    Chromatogram filenames are: {pos:05d}_{idx:04d}_{safe_peptide}_{mz}_nomod_z{charge}_rt..._ta....png
    Match by sequence (safe form), charge, and mods (nomod vs other).
    """
    def _norm_mods(m):
        if m is None or (isinstance(m, float) and pd.isna(m)):
            return '-'
        s = str(m).strip()
        return s if s and s.lower() != 'nan' else '-'

    def _safe_seq(seq):
        return "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in (seq or ''))

    # Find chromatograms directory: must contain individual peptide chromatogram PNGs
    # (format 00002_0003_SEQUENCE_mz_nomod_z... not RT windows / overlay PNGs)
    search_dirs = [
        os.path.join(csv_dir, 'accepted', 'chromatograms_peptides'),
        os.path.join(output_dir, 'accepted', 'chromatograms_peptides'),
        os.path.join(csv_dir, 'chromatograms_peptides'),
        os.path.join(output_dir, 'chromatograms_peptides'),
        output_dir,
        csv_dir,
    ]
    chrom_dir = None
    peptide_png_pattern = re.compile(r'^\d{5}_\d{4}_')  # position_index at start of filename
    for d in search_dirs:
        if os.path.isdir(d):
            try:
                listing = os.listdir(d)
            except OSError:
                continue
            pngs = [f for f in listing if f.endswith('.png') and not f.startswith('all_') and peptide_png_pattern.match(f)]
            if pngs:
                chrom_dir = d
                break
    if not chrom_dir:
        print("  No chromatograms directory with peptide PNGs found; skipping copy of filtered chromatograms")
        return

    dest_dir = os.path.join(output_dir, f'{output_base}_chromatograms')
    os.makedirs(dest_dir, exist_ok=True)

    # Build list of (safe_seq, charge, mods_norm, mz, peptide) for matching
    peptide_keys = []
    for p in peptides_for_rt_grid:
        seq = (p.get('peptide_seq') or '').strip()
        if not seq:
            continue
        try:
            ch = int(p.get('charge', 0)) if p.get('charge') not in (None, '') else 0
        except (TypeError, ValueError):
            ch = 0
        mods_norm = _norm_mods(p.get('modifications'))
        mz = p.get('mz_theoretical') or p.get('mz')
        if mz is not None and isinstance(mz, float) and (pd.isna(mz) or mz <= 0):
            mz = None
        peptide_keys.append((_safe_seq(seq), ch, mods_norm, mz, p))

    # List only individual peptide chromatogram PNGs (format 00002_0003_SEQ_...)
    all_pngs = [f for f in os.listdir(chrom_dir) if f.endswith('.png') and not f.startswith('all_') and peptide_png_pattern.match(f)]
    if not all_pngs:
        print("  No peptide chromatogram PNGs in directory; skipping copy")
        return

    copied = 0
    used_filenames = set()
    for safe_seq, charge, mods_norm, mz, peptide in peptide_keys:
        # Match: filename must contain safe_seq and _z{charge}_ or end with _z{charge}.png
        charge_suffix = f'_z{charge}_'
        charge_suffix_end = f'_z{charge}.png'
        candidates = []
        for f in all_pngs:
            if safe_seq not in f:
                continue
            if charge_suffix not in f and not f.endswith(charge_suffix_end):
                continue
            if not _chromatogram_filename_mods_match(f, mods_norm):
                continue
            # Parse m/z from filename: ..._SEQ_XXXpXX_nomod_z... (mz has 'p' for decimal, e.g. 429p59)
            file_mz = None
            try:
                mo = re.search(r'_(\d+p\d+(?:p\d+)?)_(?:nomod|[^_]+)_z\d+_', f)
                if mo:
                    file_mz = float(mo.group(1).replace('p', '.'))
                if file_mz is not None and not (100 < file_mz < 5000):
                    file_mz = None
            except (ValueError, IndexError):
                pass
            candidates.append((f, file_mz))
        if not candidates:
            continue
        # Prefer candidate with mz closest to peptide mz
        if mz is not None and len(candidates) > 1:
            def mz_dist(c):
                fmz = c[1]
                if fmz is None:
                    return 1e9
                return abs(float(fmz) - float(mz))
            candidates.sort(key=mz_dist)
        best_file = candidates[0][0]
        src = os.path.join(chrom_dir, best_file)
        if not os.path.isfile(src):
            continue
        # Copy with original filename (avoid overwrite by keeping unique names)
        dest_name = best_file
        if dest_name in used_filenames:
            base, ext = os.path.splitext(dest_name)
            dest_name = f"{base}_dup{ext}"
        used_filenames.add(dest_name)
        dest_path = os.path.join(dest_dir, dest_name)
        try:
            shutil.copy2(src, dest_path)
            copied += 1
        except Exception as e:
            print(f"  Warning: could not copy {best_file}: {e}")
    if copied:
        print(f"  Copied {copied} chromatograms to {os.path.abspath(dest_dir)}")
    else:
        print("  No matching chromatograms copied (filename matching may differ)")


def _primary_peptide_keys_from_three_channel_csv(three_channel_csv_path, min_single_aa_overhangs=1):
    """
    Return list of (safe_seq, charge, mods_norm) for peptides in the primary set
    (same as _write_filter_csv_from_three_channel_csv: n_overhangs >= min_single_aa_overhangs).
    Used to check if we already have chromatogram PNGs for the primary set.
    """
    def _safe_seq(seq):
        return "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in (seq or ''))
    def _norm_mods(m):
        if m is None or (isinstance(m, float) and pd.isna(m)):
            return '-'
        s = str(m).strip()
        return s if s and s.lower() != 'nan' else '-'
    if not os.path.isfile(three_channel_csv_path):
        return []
    try:
        df = pd.read_csv(three_channel_csv_path)
    except Exception:
        return []
    seq_col = 'peptide_seq' if 'peptide_seq' in df.columns else 'plain_peptide'
    if seq_col not in df.columns:
        return []
    if 'single_aa_positions' in df.columns and min_single_aa_overhangs > 0:
        def _n_overhangs(val):
            if pd.isna(val) or not str(val).strip():
                return 0
            return len([x.strip() for x in str(val).strip().split(',') if x.strip()])
        n_overhangs = df['single_aa_positions'].apply(_n_overhangs)
        df = df.loc[n_overhangs >= min_single_aa_overhangs].copy()
    out = []
    for _, r in df.iterrows():
        seq = str(r.get(seq_col, '')).strip()
        if not seq:
            continue
        try:
            ch = int(r.get('charge', 0)) if pd.notna(r.get('charge')) else 0
        except (TypeError, ValueError):
            ch = 0
        mods = _norm_mods(r.get('modifications', '-'))
        out.append((_safe_seq(seq), ch, mods))
    return out


def _chromatogram_dir_has_pngs_for_peptide_keys(chrom_dir, peptide_keys):
    """
    Return True if chrom_dir contains at least one matching PNG for every (safe_seq, charge, mods_norm) in peptide_keys.
    Uses same matching logic as copy_filtered_chromatograms.
    """
    if not chrom_dir or not os.path.isdir(chrom_dir) or not peptide_keys:
        return False
    peptide_png_pattern = re.compile(r'^\d{5}_\d{4}_')
    all_pngs = [f for f in os.listdir(chrom_dir) if f.endswith('.png') and not f.startswith('all_') and peptide_png_pattern.match(f)]
    if not all_pngs:
        return False
    for safe_seq, charge, mods_norm in peptide_keys:
        charge_suffix = f'_z{charge}_'
        charge_suffix_end = f'_z{charge}.png'
        found = False
        for f in all_pngs:
            if safe_seq not in f:
                continue
            if charge_suffix not in f and not f.endswith(charge_suffix_end):
                continue
            if not _chromatogram_filename_mods_match(f, mods_norm):
                continue
            found = True
            break
        if not found:
            return False
    return True


def _find_chromatograms_dir(output_dir, csv_dir):
    """Return first directory that contains full peptide chromatogram PNGs (format 00002_0004_...)."""
    search_dirs = [
        os.path.join(csv_dir, 'accepted', 'chromatograms_peptides') if csv_dir else None,
        os.path.join(output_dir, 'accepted', 'chromatograms_peptides'),
        os.path.join(csv_dir, 'chromatograms_peptides') if csv_dir else None,
        os.path.join(output_dir, 'chromatograms_peptides'),
        output_dir,
        csv_dir if csv_dir else None,
    ]
    peptide_png_pattern = re.compile(r'^\d{5}_\d{4}_')
    for d in search_dirs:
        if not d or not os.path.isdir(d):
            continue
        try:
            pngs = [f for f in os.listdir(d) if f.endswith('.png') and not f.startswith('all_') and peptide_png_pattern.match(f)]
            if pngs:
                return d
        except OSError:
            continue
    return None


def _draw_drift_lines_on_chromatogram_png(filepath, drift_min_rt, drift_max_rt, rts_sec, panel_frac_height=0.35, line_width=6):
    """
    Draw thick vertical black lines at drift_min_rt and drift_max_rt (seconds) on a saved chromatogram PNG.
    rts_sec: array of RT values in seconds (used to infer x-axis range in min). Panel assumed top panel_frac_height of image.
    """
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return
    try:
        img = Image.open(filepath).convert('RGBA')
    except Exception:
        return
    w, h = img.size
    rts_sec = np.asarray(rts_sec, dtype=float)
    if len(rts_sec) < 2:
        return
    x_min = float(np.min(rts_sec)) / 60.0
    x_max = float(np.max(rts_sec)) / 60.0
    _draw_drift_lines_on_chromatogram_png_using_range(filepath, drift_min_rt, drift_max_rt, x_min, x_max, img=img, w=w, h=h, panel_frac_height=panel_frac_height, line_width=line_width)


def _draw_drift_lines_on_chromatogram_png_using_range(filepath, drift_min_rt, drift_max_rt, x_min_min, x_max_min, img=None, w=None, h=None, panel_frac_height=0.35, line_width=6):
    """
    Draw thick vertical black lines at drift_min_rt and drift_max_rt (seconds) using explicit x-axis range in minutes.
    x_min_min, x_max_min: chromatogram x-axis range (minutes). If None or x_max_min <= x_min_min, uses drift window with padding.
    """
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return
    if img is None:
        try:
            img = Image.open(filepath).convert('RGBA')
        except Exception:
            return
    if w is None or h is None:
        w, h = img.size
    if x_min_min is None or x_max_min is None or x_max_min <= x_min_min:
        x_min_min = float(drift_min_rt) / 60.0 - 1.0
        x_max_min = float(drift_max_rt) / 60.0 + 1.0
    if x_max_min <= x_min_min:
        x_max_min = x_min_min + 0.01
    panel_x_left = 0.08 * w
    panel_x_right = 0.92 * w
    panel_w = panel_x_right - panel_x_left
    y_bottom = int(panel_frac_height * h)

    def rt_to_px(rt_sec):
        x = rt_sec / 60.0
        frac = (x - x_min_min) / (x_max_min - x_min_min)
        frac = max(0.0, min(1.0, frac))
        return int(panel_x_left + frac * panel_w)

    px_min = rt_to_px(float(drift_min_rt))
    px_max = rt_to_px(float(drift_max_rt))
    draw = ImageDraw.Draw(img)
    for px in (px_min, px_max):
        draw.line((px, 0, px, y_bottom), fill=(0, 0, 0, 255), width=line_width)
    img.convert('RGB').save(filepath, 'PNG', dpi=(200, 200))


def _run_full_chromatograms_for_filter(comet_csv_path, raw_file_path, filter_csv_path, output_accepted_dir, output_rejected_dir, exclude_mods=False):
    """
    Run the full chromatogram visualization (run_chromatograms) restricted to peptides in filter_csv_path.
    filter_csv_path must be a CSV with columns plain_peptide, charge, modifications.
    Writes full peptide chromatogram PNGs to output_accepted_dir (and rejected to output_rejected_dir).
    """
    if not os.path.isfile(comet_csv_path):
        print(f"  Skipping full chromatogram run: comet CSV not found: {comet_csv_path}")
        return False
    if not os.path.isfile(raw_file_path):
        print(f"  Skipping full chromatogram run: raw/mzML file not found: {raw_file_path}")
        return False
    if not os.path.isfile(filter_csv_path):
        print(f"  Skipping full chromatogram run: filter CSV not found: {filter_csv_path}")
        return False
    from visualization.chromatograms import run_chromatograms
    import argparse
    ns = argparse.Namespace(
        comet_csv=comet_csv_path,
        raw_file=raw_file_path,
        output_png=os.path.join(output_accepted_dir, 'chromatograms.png'),
        filter_csv=filter_csv_path,
        exclude_mods=exclude_mods,
        test=False,
        extract_only=False,
        plot_only=False,
        output_accepted_dir=output_accepted_dir,
        output_rejected_dir=output_rejected_dir,
        skip_zoom_segments=True,  # Regeneration: use new layout only; no segmented zoom figures
    )
    run_chromatograms(ns)
    return True


def _write_combined_visualization_for_peptide_set(df, peptides, _norm_mods, output_dir, output_base_name, fasta_path, protein_id, peak_windows_csv=None, filter_criteria_override=None):
    """
    Write a CSV containing only the given peptides (from df), then run create_combined_visualization
    so that unique_peptides, _filtered, and related plots are saved in output_dir.
    filter_criteria_override: if set (e.g. {'min_single_aa_overhangs': 0}), passed as filter_criteria
    so the plot shows all peptides in the set (e.g. stage4 envelope) instead of applying default overhang filter.
    Returns True if plots were written.
    """
    if not peptides or not os.path.isfile(fasta_path):
        return False
    keys = set()
    for p in peptides:
        seq = (p.get('peptide_seq') or p.get('plain_peptide') or '').strip()
        ch = p.get('charge', 0)
        try:
            ch = int(ch) if ch is not None else 0
        except (TypeError, ValueError):
            ch = 0
        mods = _norm_mods(p.get('modifications'))
        keys.add((seq, ch, mods))
    if not keys:
        return False
    # Filter df to rows matching (plain_peptide, charge, mods)
    def row_key(r):
        seq = str(r.get('plain_peptide', '')).strip()
        ch = r.get('charge', 0)
        try:
            ch = int(ch) if pd.notna(ch) else 0
        except (TypeError, ValueError):
            ch = 0
        mods = _norm_mods(r.get('modifications'))
        return (seq, ch, mods)
    mask = df.apply(lambda r: row_key(r) in keys, axis=1)
    df_sub = df.loc[mask].copy()
    if len(df_sub) == 0:
        return False
    # Build lookup (seq, charge, mods) -> peptide dict so we can add RT window columns.
    # Combined's unique_peptides plot only includes peptides with _has_rt_window_data(p); without
    # these columns only 9 appear instead of the full set.
    peptide_lookup = {}
    for p in peptides:
        seq = (p.get('peptide_seq') or p.get('plain_peptide') or '').strip()
        ch = p.get('charge', 0)
        try:
            ch = int(ch) if ch is not None else 0
        except (TypeError, ValueError):
            ch = 0
        mods = _norm_mods(p.get('modifications'))
        peptide_lookup[(seq, ch, mods)] = p
    rt_cols = ['detected_peak_min_rt', 'detected_peak_max_rt', 'spectrum_window_min_rt', 'spectrum_window_max_rt', 'collection_min_rt', 'collection_max_rt']
    for col in rt_cols:
        if col not in df_sub.columns:
            df_sub[col] = None
    for idx, row in df_sub.iterrows():
        k = row_key(row)
        if k not in peptide_lookup:
            continue
        p = peptide_lookup[k]
        for col in rt_cols:
            v = p.get(col)
            if v is not None and not (isinstance(v, float) and pd.isna(v)):
                df_sub.at[idx, col] = v
    csv_path = os.path.join(output_dir, f'{output_base_name}_peptides_for_combined.csv')
    df_sub.to_csv(csv_path, index=False)
    output_file = os.path.join(output_dir, f'{output_base_name}.png')
    try:
        from visualization.combined import create_combined_visualization
        create_combined_visualization._results_dir = output_dir
        create_combined_visualization(
            csv_path, protein_id, fasta_path, output_file,
            ion_type_filter=None, no_filtering=False, use_two_pass=True, no_progressive=False, no_two_pass=False,
            mzml_file=None, peak_windows_csv=peak_windows_csv, filter_by_peak_window_status=None, test=False, filter_criteria=filter_criteria_override
        )
        return True
    except Exception as e:
        print(f"  Warning: could not create combined/unique_peptides for {output_base_name}: {e}", file=sys.stderr)
        return False


def _write_filter_csv_from_three_channel_csv(three_channel_csv_path, out_filter_path, min_single_aa_overhangs=2):
    """
    Write a filter CSV with columns plain_peptide, charge, modifications, and optionally
    drift_min_rt, drift_max_rt, m0_gt_m1_gt_m2 from the 3-channel CSV (for chromatogram x-axis,
    flank shading, and skipping envelope re-check on re-extraction).
    All peptides in the channel list already satisfy M+0 > M+1 or M+1 > M+2, so we set
    m0_gt_m1_gt_m2=True so the chromatogram re-extraction does not re-reject them.
    Exclude peptides with fewer than min_single_aa_overhangs significant single-AA overhangs
    (so we do not generate individual chromatogram PNGs for them; 0-overhang peptides are
    backfill-only for RT gaps). Drop duplicates. Returns True if written.
    """
    if not os.path.isfile(three_channel_csv_path):
        return False
    try:
        df = pd.read_csv(three_channel_csv_path)
    except Exception:
        return False
    seq_col = 'peptide_seq' if 'peptide_seq' in df.columns else 'plain_peptide'
    if seq_col not in df.columns:
        return False
    # Exclude any peptide with fewer than min_single_aa_overhangs (0-overhang peptides get no individual PNGs)
    if 'single_aa_positions' in df.columns and min_single_aa_overhangs > 0:
        def _n_overhangs(val):
            if pd.isna(val) or not str(val).strip():
                return 0
            return len([x.strip() for x in str(val).strip().split(',') if x.strip()])
        n_overhangs = df['single_aa_positions'].apply(_n_overhangs)
        keep = n_overhangs >= min_single_aa_overhangs
        dropped = (~keep).sum()
        if dropped:
            print(f"  Primary chromatogram filter: excluding {int(dropped)} peptide(s) with <{min_single_aa_overhangs} single-AA overhang(s) (no individual chromatogram PNGs)")
        df = df.loc[keep].copy()
    sub = df[['charge', 'modifications']].copy()
    sub['plain_peptide'] = df[seq_col].astype(str).str.strip()
    if 'drift_min_rt' in df.columns and 'drift_max_rt' in df.columns:
        sub['drift_min_rt'] = df['drift_min_rt'].values
        sub['drift_max_rt'] = df['drift_max_rt'].values
    # Channel list only contains envelope-OK peptides; set m0_gt_m1_gt_m2 so re-extraction skips envelope re-check
    if 'm0_gt_m1_gt_m2' in df.columns:
        sub['m0_gt_m1_gt_m2'] = df['m0_gt_m1_gt_m2'].fillna(True).astype(bool)
    else:
        sub['m0_gt_m1_gt_m2'] = True
    sub = sub.drop_duplicates(subset=['plain_peptide', 'charge', 'modifications'], keep='first')
    sub.to_csv(out_filter_path, index=False)
    return True


def _add_drift_lines_to_chromatograms_in_dir(csv_path, chrom_dir):
    """
    For each row in the 3-channel CSV, find a matching chromatogram PNG in chrom_dir and draw
    thick vertical black lines at drift_min_rt, drift_max_rt. Uses spectrum_window or collection
    from CSV for x-axis range (minutes).
    """
    if not os.path.isfile(csv_path):
        return
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return
    for col in ('drift_min_rt', 'drift_max_rt', 'peptide_seq', 'charge'):
        if col not in df.columns:
            return
    peptide_png_pattern = re.compile(r'^\d{5}_\d{4}_')
    all_pngs = [f for f in os.listdir(chrom_dir) if f.endswith('.png') and not f.startswith('all_') and peptide_png_pattern.match(f)]
    if not all_pngs:
        return

    def _norm_mods(m):
        if m is None or (isinstance(m, float) and pd.isna(m)):
            return '-'
        s = str(m).strip()
        return s if s and s.lower() != 'nan' else '-'

    def _safe_seq(seq):
        return "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in (str(seq or '')))

    drawn = 0
    for _, row in df.iterrows():
        seq = str(row.get('peptide_seq', '')).strip()
        try:
            ch = int(row.get('charge', 0)) if row.get('charge') not in (None, '') and pd.notna(row.get('charge')) else 0
        except (TypeError, ValueError):
            ch = 0
        mods = _norm_mods(row.get('modifications', '-'))
        drift_min = row.get('drift_min_rt')
        drift_max = row.get('drift_max_rt')
        if pd.isna(drift_min) or pd.isna(drift_max):
            continue
        try:
            drift_min = float(drift_min)
            drift_max = float(drift_max)
        except (TypeError, ValueError):
            continue
        x_min_min = row.get('spectrum_window_min_rt') or row.get('collection_min_rt') or row.get('detected_peak_min_rt')
        x_max_min = row.get('spectrum_window_max_rt') or row.get('collection_max_rt') or row.get('detected_peak_max_rt')
        if x_min_min is not None and x_max_min is not None and pd.notna(x_min_min) and pd.notna(x_max_min):
            try:
                x_min_min = float(x_min_min) / 60.0
                x_max_min = float(x_max_min) / 60.0
            except (TypeError, ValueError):
                x_min_min = x_max_min = None
        else:
            x_min_min = x_max_min = None
        safe_seq = _safe_seq(seq)
        charge_suffix = f'_z{ch}_'
        charge_suffix_end = f'_z{ch}.png'
        candidates = [f for f in all_pngs if safe_seq in f and (charge_suffix in f or f.endswith(charge_suffix_end)) and _chromatogram_filename_mods_match(f, mods)]
        if not candidates:
            continue
        dest_path = os.path.join(chrom_dir, candidates[0])
        _draw_drift_lines_on_chromatogram_png_using_range(dest_path, drift_min, drift_max, x_min_min, x_max_min, line_width=6)
        drawn += 1
    if drawn:
        print(f"  Drew peak drift lines on {drawn} chromatograms in {os.path.abspath(chrom_dir)}")


def write_peptide_chromatograms_with_drift_lines(peptides_for_rt_grid, drift_rows, peptide_traces, output_dir, csv_dir=None):
    """
    Copy the full peptide chromatogram PNGs (same style as filtered_chromatograms) from the
    chromatograms directory into output_dir, then draw thick vertical black lines at the peak
    drift window min and max on each. If no pre-existing chromatograms are found, skips.
    csv_dir: used with output_dir to search for chromatograms_peptides (same as copy_filtered_chromatograms).
    """
    def _norm_mods(m):
        if m is None or (isinstance(m, float) and pd.isna(m)):
            return '-'
        s = str(m).strip()
        return s if s and s.lower() != 'nan' else '-'

    def _safe_seq(seq):
        return "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in (seq or ''))

    chrom_dir = _find_chromatograms_dir(output_dir, csv_dir or output_dir)
    if not chrom_dir:
        print("  No full chromatogram PNGs found; skipping peptide_chromatograms (run chromatogram extraction first)")
        return

    os.makedirs(output_dir, exist_ok=True)
    drift_by_id = {id(p): (dmin, dmax) for p, dmin, dmax in drift_rows}
    n = len(peptides_for_rt_grid)
    peptide_traces_list = peptide_traces if (peptide_traces and len(peptide_traces) == n) else [None] * n

    peptide_keys = []
    for i, p in enumerate(peptides_for_rt_grid):
        seq = (p.get('peptide_seq') or p.get('plain_peptide') or '').strip()
        if not seq:
            continue
        try:
            ch = int(p.get('charge', 0)) if p.get('charge') not in (None, '') else 0
        except (TypeError, ValueError):
            ch = 0
        mods_norm = _norm_mods(p.get('modifications'))
        mz = p.get('mz_theoretical') or p.get('mz')
        if mz is not None and isinstance(mz, float) and (pd.isna(mz) or mz <= 0):
            mz = None
        peptide_keys.append((_safe_seq(seq), ch, mods_norm, mz, p, i))

    all_pngs = [f for f in os.listdir(chrom_dir) if f.endswith('.png') and not f.startswith('all_') and re.compile(r'^\d{5}_\d{4}_').match(f)]
    if not all_pngs:
        print("  No peptide chromatogram PNGs in directory; skipping")
        return

    copied = 0
    used_filenames = set()
    for safe_seq, charge, mods_norm, mz, peptide, idx in peptide_keys:
        charge_suffix = f'_z{charge}_'
        charge_suffix_end = f'_z{charge}.png'
        candidates = []
        for f in all_pngs:
            if safe_seq not in f:
                continue
            if charge_suffix not in f and not f.endswith(charge_suffix_end):
                continue
            if not _chromatogram_filename_mods_match(f, mods_norm):
                continue
            file_mz = None
            try:
                mo = re.search(r'_(\d+p\d+(?:p\d+)?)_(?:nomod|[^_]+)_z\d+_', f)
                if mo:
                    file_mz = float(mo.group(1).replace('p', '.'))
                if file_mz is not None and not (100 < file_mz < 5000):
                    file_mz = None
            except (ValueError, IndexError):
                pass
            candidates.append((f, file_mz))
        if not candidates:
            continue
        if mz is not None and len(candidates) > 1:
            def mz_dist(c):
                fmz = c[1]
                if fmz is None:
                    return 1e9
                return abs(float(fmz) - float(mz))
            candidates.sort(key=mz_dist)
        best_file = candidates[0][0]
        src = os.path.join(chrom_dir, best_file)
        if not os.path.isfile(src):
            continue
        dest_name = best_file
        if dest_name in used_filenames:
            base, ext = os.path.splitext(dest_name)
            dest_name = f"{base}_dup{ext}"
        used_filenames.add(dest_name)
        dest_path = os.path.join(output_dir, dest_name)
        try:
            shutil.copy2(src, dest_path)
            copied += 1
        except Exception as e:
            print(f"  Warning: could not copy {best_file}: {e}", file=sys.stderr)
            continue
        drift_min, drift_max = drift_by_id.get(id(peptide), (None, None))
        trace = peptide_traces_list[idx] if idx < len(peptide_traces_list) else None
        if drift_min is not None and drift_max is not None and trace is not None:
            rts_sec = np.asarray(trace[0], dtype=float)
            if len(rts_sec) >= 2:
                _draw_drift_lines_on_chromatogram_png(dest_path, float(drift_min), float(drift_max), rts_sec)
    if copied:
        print(f"  Wrote {copied} full peptide chromatograms (with peak drift lines) to {os.path.abspath(output_dir)}")


def _load_chromatogram_traces(npz_path, trace_csv_path, peptides_for_rt_grid):
    """
    Load chromatogram traces from an npz file, matching by peptide (seq, charge, mods) using the CSV that was used to build the npz.
    Returns list of (rts, total_intensity) in same order as peptides_for_rt_grid, or None if matching fails.
    """
    if not os.path.isfile(npz_path):
        return None
    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as e:
        print(f"  Warning: could not load chromatogram traces from {npz_path}: {e}")
        return None
    keys = [k for k in data.files if k.startswith('trace_') and k.endswith('_rts')]
    n_traces = len(keys)
    if n_traces == 0:
        return None

    def _norm_mods(m):
        if m is None or (isinstance(m, float) and pd.isna(m)):
            return '-'
        s = str(m).strip()
        return s if s and s.lower() != 'nan' else '-'

    if trace_csv_path and os.path.isfile(trace_csv_path):
        try:
            df = pd.read_csv(trace_csv_path, nrows=0)
            if 'peptide' not in df.columns and 'plain_peptide' not in df.columns and 'peptide_seq' not in df.columns:
                trace_csv_path = None
        except Exception:
            trace_csv_path = None
    else:
        trace_csv_path = None

    if trace_csv_path:
        try:
            df_trace = pd.read_csv(trace_csv_path)
            seq_col = 'peptide_seq' if 'peptide_seq' in df_trace.columns else ('plain_peptide' if 'plain_peptide' in df_trace.columns else 'peptide')
            if seq_col not in df_trace.columns:
                trace_csv_path = None
        except Exception:
            trace_csv_path = None

    if trace_csv_path and len(df_trace) != n_traces:
        print(f"  Warning: trace CSV has {len(df_trace)} rows but npz has {n_traces} traces; skipping load")
        return None

    out = []
    for p in peptides_for_rt_grid:
        seq = p.get('peptide_seq') or p.get('plain_peptide') or ''
        ch = p.get('charge')
        try:
            ch = int(ch) if ch not in (None, '') else 0
        except (TypeError, ValueError):
            ch = 0
        mods = _norm_mods(p.get('modifications'))

        if trace_csv_path:
            row_idx = None
            for i, (_, row) in enumerate(df_trace.iterrows()):
                r_seq = str(row.get(seq_col, '')).strip()
                r_ch = row.get('charge')
                try:
                    r_ch = int(r_ch) if r_ch not in (None, '') and not pd.isna(r_ch) else 0
                except (TypeError, ValueError):
                    r_ch = 0
                r_mods = _norm_mods(row.get('modifications', '-'))
                if r_seq == seq and r_ch == ch and r_mods == mods:
                    row_idx = i
                    break
            if row_idx is None:
                return None
        else:
            if len(out) >= n_traces:
                return None
            row_idx = len(out)

        rts_key = f'trace_{row_idx}_rts'
        int_key = f'trace_{row_idx}_int'
        if rts_key not in data or int_key not in data:
            return None
        rts = np.asarray(data[rts_key], dtype=float)
        int_mat = np.asarray(data[int_key], dtype=float)
        if int_mat.ndim == 2:
            total_intensity = np.sum(int_mat, axis=1)
        else:
            total_intensity = np.asarray(int_mat, dtype=float)
        if len(rts) != len(total_intensity):
            return None
        out.append((rts, total_intensity))

    if trace_csv_path and len(out) != len(peptides_for_rt_grid):
        return None
    return out


# Match visualization/combined.py so overlay uses same drift window
PEAK_DRIFT_BUFFER_SEC = 30.0       # fallback when relative buffer cannot be computed
PEAK_DRIFT_BUFFER_FRAC = 0.5       # buffer = this fraction of collection window width (each side); scales with peak width
PEAK_DRIFT_BUFFER_MIN_SEC = 15.0   # minimum buffer each side (narrow peaks)
PEAK_DRIFT_BUFFER_MAX_SEC = 90.0   # maximum buffer each side (wide peaks)
PEAK_DRIFT_GAP_SEC = 2.0   # minimal gap between drift windows so we maximize RT time collecting on each channel
PEAK_DRIFT_TRIM_FRAC = 0.2  # allow trimming up to 20% of buffer on either side when arranging to avoid overlap
PEAK_DRIFT_OVERLAP_FRAC = 0.0  # no overlap: drift windows are separated by at least PEAK_DRIFT_GAP_SEC for easy day-of calibration when peaks shift


def _peak_drift_buffer_sec(cmin, cmax):
    """
    Compute peak drift buffer (seconds) on each side of the collection window.
    Fixed +30 seconds on either side.
    Returns buffer in seconds to add on left and right (same value).
    """
    return PEAK_DRIFT_BUFFER_SEC


def _get_drift_bounds(p):
    """Return (drift_min, drift_max) = full reserved window (collection ± peak drift buffer). Used for channel assignment so no two peptides in a channel have overlapping full windows."""
    cmin, cmax = _get_collection_bounds(p)
    if cmin is None or cmax is None:
        return (None, None)
    try:
        cmin, cmax = float(cmin), float(cmax)
        if cmin >= cmax or cmin != cmin or cmax != cmax:
            return (None, None)
    except (TypeError, ValueError):
        return (None, None)
    buf = _peak_drift_buffer_sec(cmin, cmax)
    return (cmin - buf, cmax + buf)


def _get_collection_bounds_for_drift(p):
    """Return (coll_min, coll_max) for peak drift window; fallback to integration window + pad."""
    coll_min = p.get('collection_min_rt')
    coll_max = p.get('collection_max_rt')
    if coll_min is not None and coll_max is not None:
        try:
            a, b = float(coll_min), float(coll_max)
            if a == a and b == b and a < b:
                return (a, b)
        except (TypeError, ValueError):
            pass
    min_rt = p.get('detected_peak_min_rt') or p.get('spectrum_window_min_rt')
    max_rt = p.get('detected_peak_max_rt') or p.get('spectrum_window_max_rt')
    rt = p.get('rt', 0.0)
    if min_rt is None or max_rt is None:
        try:
            min_rt = float(rt) - 5.0
            max_rt = float(rt) + 5.0
        except (TypeError, ValueError):
            min_rt, max_rt = 0.0, 10.0
    try:
        min_rt, max_rt = float(min_rt), float(max_rt)
    except (TypeError, ValueError):
        min_rt, max_rt = 0.0, 10.0
    pad = max(10.0, (max_rt - min_rt) * 0.15)
    return (min_rt - pad, max_rt + pad)


def _reextract_chromatogram_traces(raw_file, peptides_for_rt_grid):
    """
    Re-extract MS1 chromatogram traces for each peptide from the mzML/raw file,
    sliced to the full peak drift window (collection ± PEAK_DRIFT_BUFFER_SEC) for the overlay.
    Returns list of (rts, total_intensity) in same order as peptides_for_rt_grid, or None if extraction fails entirely.
    On per-peptide failure, appends (empty array, empty array) so overlay length matches and other peptides still plot.
    """
    if not raw_file or not os.path.isfile(raw_file):
        print("  Warning: mzML/raw file missing or not a file; cannot re-extract traces")
        return None
    try:
        from visualization.chromatograms import isotope_mz_list, extract_aligned_isotope_chromatograms
    except ImportError:
        try:
            from chromatograms import isotope_mz_list, extract_aligned_isotope_chromatograms
        except ImportError:
            print("  Warning: visualization.chromatograms not found; cannot re-extract traces")
            return None
    out = []
    n_ok = 0
    for idx, p in enumerate(peptides_for_rt_grid):
        mz = p.get('mz_theoretical') or p.get('mz')
        if mz is None or (isinstance(mz, float) and (np.isnan(mz) or mz <= 0)):
            out.append((np.array([]), np.array([])))
            continue
        charge = p.get('charge')
        try:
            charge = int(charge) if charge not in (None, '') else 0
        except (TypeError, ValueError):
            charge = 0
        if charge <= 0:
            out.append((np.array([]), np.array([])))
            continue
        isotope_mzs = isotope_mz_list(float(mz), charge, n_isos=4)
        if not isotope_mzs:
            out.append((np.array([]), np.array([])))
            continue
        # Extract over full peak drift window only (for overlay); buffer is relative to collection width
        coll_min, coll_max = _get_collection_bounds_for_drift(p)
        buf_sec = _peak_drift_buffer_sec(coll_min, coll_max)
        drift_lo = max(0.0, coll_min - buf_sec)
        drift_hi = coll_max + buf_sec
        try:
            rts, intensity_matrix = extract_aligned_isotope_chromatograms(
                raw_file, isotope_mzs, ppm_tolerance=20.0,
                rt_min_sec=drift_lo, rt_max_sec=drift_hi)
        except Exception as e:
            print(f"  Warning: extract failed for peptide {idx+1}: {e}")
            out.append((np.array([]), np.array([])))
            continue
        if len(rts) == 0:
            out.append((np.array([]), np.array([])))
            continue
        total_intensity = np.sum(intensity_matrix, axis=1) if intensity_matrix.ndim == 2 else np.asarray(intensity_matrix, dtype=float)
        rts = np.asarray(rts, dtype=float)
        total_intensity = np.asarray(total_intensity, dtype=float)
        out.append((rts, total_intensity))
        n_ok += 1
        # Debug first successful trace: confirm RT range matches drift window (e.g. seconds vs minutes)
        if n_ok == 1:
            r0, r1 = float(np.min(rts)), float(np.max(rts))
            print(f"  [re-extract] first trace: drift_lo={drift_lo:.1f} drift_hi={drift_hi:.1f} s, rts=[{r0:.1f},{r1:.1f}] s (n_pts={len(rts)})")
    if n_ok == 0:
        print("  Warning: no peptide traces extracted; overlay will be empty")
        return None
    return out


def run_rt_windows(args):
    """Regenerate RT integration windows plots from a CSV. args: csv_path (or filter_csv), output_dir, exclude_mods, output_base (optional)."""
    csv_path = getattr(args, 'csv_path', None) or getattr(args, 'filter_csv', None)
    output_dir = getattr(args, 'output_dir', None)
    exclude_mods = getattr(args, 'exclude_mods', False)
    output_base = getattr(args, 'output_base', 'filtered')

    if not csv_path:
        print("Error: csv_path required. Usage: python regenerate_rt_windows_from_csv.py <filtered_peptides.csv> <output_dir> [options]")
        sys.exit(1)
    if not output_dir:
        print("Error: output_dir required. Usage: python regenerate_rt_windows_from_csv.py <filtered_peptides.csv> <output_dir> [options]")
        sys.exit(1)
    if not os.path.exists(csv_path):
        print(f"Error: CSV not found: {csv_path}")
        sys.exit(1)
    os.makedirs(output_dir, exist_ok=True)
    rt_windows_dir = os.path.join(output_dir, 'rt_windows_channels')
    os.makedirs(rt_windows_dir, exist_ok=True)

    # Read CSV (skip Comet header if present)
    with open(csv_path, 'r') as f:
        first = f.readline()
    skip = 1 if 'CometVersion' in first else 0
    df = pd.read_csv(csv_path, sep=',', skiprows=skip, engine='python', quotechar='"', on_bad_lines='warn')

    required = ['plain_peptide', 'charge']
    for c in required:
        if c not in df.columns:
            print(f"Error: required column '{c}' not in CSV")
            sys.exit(1)
    try:
        from visualization.csv_validation import validate_and_log
        validate_and_log(
            df,
            source_name=os.path.basename(csv_path),
            required_columns=('plain_peptide', 'charge'),
            validate_identifiers=True,
            validate_numeric=True,
            validate_windows=True,
        )
    except Exception as e:
        print(f"  Validation warning: {e}")

    csv_dir = os.path.dirname(os.path.abspath(csv_path))
    dataframes_dir = getattr(args, 'dataframes_dir', None) or csv_dir

    def _norm_mods(m):
        if m is None or (isinstance(m, float) and pd.isna(m)):
            return '-'
        s = str(m).strip()
        return s if s and s.lower() != 'nan' else '-'

    # Optionally re-extract integration/collection/peak-drift windows from mzML before filtering
    re_extract_windows = not getattr(args, 'no_re_extract_windows', False)
    raw_file = getattr(args, 'mzml', None) or getattr(args, 'raw_file', None)
    pw_path = None
    if re_extract_windows and raw_file and os.path.isfile(raw_file):
        comet_csv = getattr(args, 'comet_csv', None) or csv_path
        # When using input CSV as comet CSV it must have RT and m/z for extraction
        use_input_as_comet = (comet_csv == csv_path)
        has_rt = 'MS1_retention_time_sec' in df.columns or ('retention_time_sec' in df.columns and use_input_as_comet)
        has_mz = 'mz' in df.columns or 'calc_neutral_mass' in df.columns
        if use_input_as_comet and (not has_rt or not has_mz):
            if getattr(args, 'comet_csv', None) and os.path.isfile(args.comet_csv):
                comet_csv = args.comet_csv
                use_input_as_comet = False
            else:
                print("  Re-extract skipped: input CSV lacks MS1_retention_time_sec (or retention_time_sec) and/or mz (or calc_neutral_mass). Pass --comet-csv <full Comet CSV> to re-extract.")
        if (use_input_as_comet and has_rt and has_mz) or (not use_input_as_comet and os.path.isfile(comet_csv)):
            extract_output_dir = os.path.join(rt_windows_dir, 're_extracted_windows')
            os.makedirs(extract_output_dir, exist_ok=True)
            print("  Re-extracting integration/collection/peak-drift windows from mzML before filtering...")
            try:
                from visualization.chromatograms import run_chromatograms
                chrom_ns = argparse.Namespace(
                    comet_csv=comet_csv,
                    raw_file=raw_file,
                    output_png=os.path.join(extract_output_dir, 'chromatograms.png'),
                    filter_csv=csv_path,
                    exclude_mods=exclude_mods,
                    test=False,
                    extract_only=True,
                    plot_only=False,
                    extract_output_dir=extract_output_dir,
                    dataframes_dir=dataframes_dir,
                )
                run_chromatograms(chrom_ns)
                re_extracted_file = os.path.join(extract_output_dir, 'peak_windows_all.csv')
                if os.path.isfile(re_extracted_file):
                    pw_path = re_extracted_file
                    print(f"  Peak/window data written to: {extract_output_dir}")
            except Exception as e:
                print(f"  Re-extract failed: {e}")
                import traceback
                traceback.print_exc()

    # If we didn't re-extract, look for existing peak_windows
    if pw_path is None:
        for pw_name in ['peak_windows_all.csv', 'peak_windows.csv']:
            for search_dir in [output_dir, dataframes_dir, csv_dir,
                               os.path.join(dataframes_dir, 'accepted'), os.path.join(csv_dir, 'accepted'),
                               os.path.join(dataframes_dir, 'accepted', 'chromatograms_peptides'), os.path.join(csv_dir, 'accepted', 'chromatograms_peptides'), os.path.join(output_dir, 'accepted', 'chromatograms_peptides'),
                               os.path.join(csv_dir, 'chromatograms_peptides'), os.path.join(output_dir, 'chromatograms_peptides')]:
                p = os.path.join(search_dir, pw_name)
                if os.path.isfile(p):
                    pw_path = p
                    break
            if pw_path is not None:
                break

    # Merge integration/collection windows and total_area from peak_windows (re-extracted or found)
    if pw_path is not None:
        try:
            df_pw = pd.read_csv(pw_path)
            if 'peptide' in df_pw.columns and 'detected_peak_min_rt' in df_pw.columns:
                pw_lookup = {}
                for _, r in df_pw.iterrows():
                    key = (str(r.get('peptide', '')).strip(),
                           int(r.get('charge', 0)) if pd.notna(r.get('charge')) else 0,
                           _norm_mods(r.get('modifications', '-')))
                    entry = {}
                    for col, conv in [('detected_peak_min_rt', float), ('detected_peak_max_rt', float),
                                      ('collection_min_rt', float), ('collection_max_rt', float),
                                      ('spectrum_window_min_rt', float), ('spectrum_window_max_rt', float),
                                      ('apex_intensity', float), ('total_area', float), ('n_candidates', int)]:
                        if col in df_pw.columns and pd.notna(r.get(col)) and r.get(col) is not None:
                            try:
                                entry[col] = conv(r[col])
                            except (TypeError, ValueError):
                                pass
                    # Every peptide must have M+0 > M+1 or M+1 > M+2 designation (True/False) from chromatogram step
                    if 'm0_gt_m1_gt_m2' in df_pw.columns:
                        v = r.get('m0_gt_m1_gt_m2')
                        if isinstance(v, bool):
                            entry['m0_gt_m1_gt_m2'] = v
                        elif isinstance(v, str) and v.strip().lower() in ('true', '1', 'yes'):
                            entry['m0_gt_m1_gt_m2'] = True
                        elif isinstance(v, str) and v.strip().lower() in ('false', '0', 'no'):
                            entry['m0_gt_m1_gt_m2'] = False
                        elif pd.notna(v) and v is not None:
                            try:
                                entry['m0_gt_m1_gt_m2'] = bool(float(v))
                            except (TypeError, ValueError):
                                entry['m0_gt_m1_gt_m2'] = False
                        else:
                            entry['m0_gt_m1_gt_m2'] = False  # missing/NaN => not envelope OK
                    if 'ms1_trace_ok' in df_pw.columns and pd.notna(r.get('ms1_trace_ok')):
                        v = r.get('ms1_trace_ok')
                        if isinstance(v, bool):
                            entry['ms1_trace_ok'] = v
                        elif isinstance(v, str) and v.strip().lower() in ('true', '1', 'yes'):
                            entry['ms1_trace_ok'] = True
                        elif isinstance(v, str) and v.strip().lower() in ('false', '0', 'no'):
                            entry['ms1_trace_ok'] = False
                        else:
                            try:
                                entry['ms1_trace_ok'] = bool(float(v))
                            except (TypeError, ValueError):
                                entry['ms1_trace_ok'] = True  # default when column present
                    elif 'ms1_trace_ok' not in entry:
                        entry['ms1_trace_ok'] = True  # backward compat: no column -> allow backfill
                    if entry:
                        pw_lookup[key] = entry
                if pw_lookup:
                    for col in ['detected_peak_min_rt', 'detected_peak_max_rt', 'collection_min_rt', 'collection_max_rt', 'spectrum_window_min_rt', 'spectrum_window_max_rt', 'apex_intensity', 'total_area', 'm0_gt_m1_gt_m2', 'ms1_trace_ok', 'n_candidates']:
                        if col not in df.columns:
                            df[col] = None
                        if not any(col in pw_lookup[k] for k in pw_lookup):
                            continue
                        def _fill(row):
                            key = (str(row.get('plain_peptide', '')).strip(),
                                   int(row.get('charge', 0)) if pd.notna(row.get('charge')) else 0,
                                   _norm_mods(row.get('modifications', '-')))
                            if key not in pw_lookup or col not in pw_lookup[key]:
                                return row.get(col)
                            return pw_lookup[key][col]
                        df[col] = df.apply(lambda row: row[col] if pd.notna(row.get(col)) and row.get(col) is not None else _fill(row), axis=1)
                    print(f"  Merged peak windows and total_area from peak_windows into RT data for by-peptide zooms")
            else:
                pw_path = None  # so downstream total_area fallback can still search
        except Exception as e:
            print(f"  Note: could not merge from {pw_path}: {e}")
            pw_path = None

    if 'detected_peak_min_rt' not in df.columns or 'detected_peak_max_rt' not in df.columns:
        print("Error: detected_peak_min_rt and detected_peak_max_rt required (merge from peak_windows failed or CSV has no such columns)")
        sys.exit(1)
    if df['detected_peak_min_rt'].notna().sum() == 0 or df['detected_peak_max_rt'].notna().sum() == 0:
        print("Error: no rows with valid detected_peak_min_rt/detected_peak_max_rt (run chromatogram step first to produce peak_windows)")
        sys.exit(1)

    # If total_area is still missing or all NaN, try to merge from peak_windows_all.csv (for heatmap/colorbar)
    has_total_area = 'total_area' in df.columns and df['total_area'].notna().any()
    if not has_total_area:
        for search_dir in [output_dir, csv_dir, os.path.join(csv_dir, 'chromatograms_peptides'), os.path.join(csv_dir, 'accepted', 'chromatograms_peptides'), os.path.join(output_dir, 'accepted', 'chromatograms_peptides')]:
            pw_path = os.path.join(search_dir, 'peak_windows_all.csv')
            if os.path.isfile(pw_path):
                try:
                    df_pw = pd.read_csv(pw_path)
                    if 'total_area' in df_pw.columns and 'peptide' in df_pw.columns:
                        def _norm_mods(m):
                            if m is None or (isinstance(m, float) and pd.isna(m)):
                                return '-'
                            s = str(m).strip()
                            return s if s and s.lower() != 'nan' else '-'
                        pw_lookup = {}
                        for _, r in df_pw.iterrows():
                            key = (str(r.get('peptide', '')).strip(),
                                   int(r.get('charge', 0)) if pd.notna(r.get('charge')) else 0,
                                   _norm_mods(r.get('modifications', '-')))
                            if pd.notna(r.get('total_area')) and r.get('total_area') is not None:
                                pw_lookup[key] = float(r['total_area'])
                        if pw_lookup:
                            df['total_area'] = df.apply(
                                lambda row: pw_lookup.get((
                                    str(row.get('plain_peptide', '')).strip(),
                                    int(row.get('charge', 0)) if pd.notna(row.get('charge')) else 0,
                                    _norm_mods(row.get('modifications', '-'))
                                ), None), axis=1)
                            print(f"  Merged total_area from {pw_path} for heatmap/colorbar")
                except Exception as e:
                    print(f"  Note: could not merge total_area from {pw_path}: {e}")
                break

    def _parse_single_aa_fragment_pairs_from_csv(single_aa_positions_dict, pairs_str):
        """Parse CSV column single_aa_overhang_fragment_pairs into {pos: set of pair strings} for the unique-peptides grid.
        Format: comma-separated blocks (one per position in sorted order); within each block pairs are separated by |.
        e.g. 'c3|c4, z2|z3' for one position, or 'c3|c4,c5|c6' for two positions.
        """
        out = {}
        if not single_aa_positions_dict or pairs_str is None or (isinstance(pairs_str, float) and pd.isna(pairs_str)) or not str(pairs_str).strip():
            return out
        positions_sorted = sorted(single_aa_positions_dict.keys())
        blocks = [b.strip() for b in str(pairs_str).strip().strip('"').split(',') if b.strip()]
        for i, pos in enumerate(positions_sorted):
            if i >= len(blocks):
                break
            normalized = set()
            parts = [p.strip() for p in blocks[i].replace('|', '-').split('-') if p.strip()]
            # Consecutive pairs: (parts[0], parts[1]), (parts[2], parts[3]), ...
            for j in range(0, len(parts) - 1, 2):
                pair_str = f'{parts[j]}-{parts[j+1]}'
                # Normalize z1_N -> zN like visualization.multi_aa.normalize_fragment_pair
                pair_str = re.sub(r'z1_(\d+)', r'z\1', pair_str)
                normalized.add(pair_str)
            if normalized:
                out[pos] = normalized
        return out

    # Build peptide dicts for create_rt_integration_windows_plot
    peptides_for_rt_grid = []
    for _, row in df.iterrows():
        seq = str(row.get('plain_peptide', '')).strip()
        charge = row.get('charge')
        try:
            charge = int(charge) if pd.notna(charge) else 0
        except (TypeError, ValueError):
            charge = 0
        mods = row.get('modifications', '-')
        if pd.isna(mods) or mods == '':
            mods = '-'
        rt_sec = row.get('MS1_retention_time_sec')
        if pd.isna(rt_sec):
            rt_sec = 0.0
        else:
            rt_sec = float(rt_sec)
        min_rt = row.get('detected_peak_min_rt')
        max_rt = row.get('detected_peak_max_rt')
        coll_min = row.get('collection_min_rt')
        coll_max = row.get('collection_max_rt')
        mz = row.get('mz')
        if pd.notna(mz):
            mz = float(mz)
        else:
            mz = None
        # Theoretical m/z for by-peptide grouping (one row per peptide); from calc_neutral_mass if present
        calc_mass = row.get('calc_neutral_mass')
        if pd.notna(calc_mass) and charge:
            try:
                mz_theoretical = (float(calc_mass) + charge * 1.007276) / charge
            except (TypeError, ValueError):
                mz_theoretical = mz
        else:
            mz_theoretical = mz
        total_area = row.get('total_area')
        if pd.notna(total_area):
            try:
                total_area = float(total_area)
            except (TypeError, ValueError):
                total_area = None
        else:
            total_area = None
        apex_intensity = row.get('apex_intensity')
        if pd.notna(apex_intensity):
            try:
                apex_intensity = float(apex_intensity)
            except (TypeError, ValueError):
                apex_intensity = None
        else:
            apex_intensity = None
        n_candidates = row.get('n_candidates')
        if pd.notna(n_candidates) and n_candidates is not None:
            try:
                n_candidates = int(float(n_candidates))
            except (TypeError, ValueError):
                n_candidates = None
        else:
            n_candidates = None
        m0_gt_m1_gt_m2 = row.get('m0_gt_m1_gt_m2')
        if pd.notna(m0_gt_m1_gt_m2) and m0_gt_m1_gt_m2 is not None:
            if isinstance(m0_gt_m1_gt_m2, bool):
                pass
            elif isinstance(m0_gt_m1_gt_m2, str) and m0_gt_m1_gt_m2.strip().lower() in ('true', '1', 'yes'):
                m0_gt_m1_gt_m2 = True
            elif isinstance(m0_gt_m1_gt_m2, str) and m0_gt_m1_gt_m2.strip().lower() in ('false', '0', 'no'):
                m0_gt_m1_gt_m2 = False
            else:
                try:
                    m0_gt_m1_gt_m2 = bool(float(m0_gt_m1_gt_m2))
                except (TypeError, ValueError):
                    m0_gt_m1_gt_m2 = None  # unparseable => fail required check below
        if m0_gt_m1_gt_m2 is None:
            # Every peptide must have m0_gt_m1_gt_m2 (True/False) from the chromatogram step; run chromatograms first and merge from peak_windows
            print(f"Error: peptide {seq} charge {charge} has no m0_gt_m1_gt_m2 designation. Run the chromatogram step first so peak_windows has every peptide, then run RT windows with the same filter.", file=sys.stderr)
            sys.exit(1)
        ms1_trace_ok = row.get('ms1_trace_ok')
        if pd.notna(ms1_trace_ok) and ms1_trace_ok is not None:
            if isinstance(ms1_trace_ok, bool):
                pass
            elif isinstance(ms1_trace_ok, str) and ms1_trace_ok.strip().lower() in ('true', '1', 'yes'):
                ms1_trace_ok = True
            elif isinstance(ms1_trace_ok, str) and ms1_trace_ok.strip().lower() in ('false', '0', 'no'):
                ms1_trace_ok = False
            else:
                try:
                    ms1_trace_ok = bool(float(ms1_trace_ok))
                except (TypeError, ValueError):
                    ms1_trace_ok = None
        else:
            ms1_trace_ok = None  # missing => backward compat, _ms1_trace_ok treats as True

        # PSM scores for total vs passed counts (same thresholds as chromatograms)
        percolator_qvalue = None
        for col in ['percolator_qvalue', 'q-value', 'qvalue']:
            if col in df.columns and pd.notna(row.get(col)):
                try:
                    percolator_qvalue = float(row[col])
                    break
                except (TypeError, ValueError):
                    pass
        percolator_PEP = None
        for col in ['percolator_PEP', 'pep', 'PEP']:
            if col in df.columns and pd.notna(row.get(col)):
                try:
                    percolator_PEP = float(row[col])
                    break
                except (TypeError, ValueError):
                    pass
        evalue = None
        for col in ['e-value', 'e_value', 'E-value']:
            if col in df.columns and pd.notna(row.get(col)):
                try:
                    evalue = float(row[col])
                    break
                except (TypeError, ValueError):
                    pass

        # Significant fragment count for MS2 (significant-only workflow).
        significant_frag_count = None
        for col in ['significant_frag_count']:
            if col in df.columns and pd.notna(row.get(col)):
                try:
                    significant_frag_count = int(float(row[col]))
                    break
                except (TypeError, ValueError):
                    pass
        if significant_frag_count is None:
            for col in ['significant_frags', 'refined_significant_frags', 'significant_fragment_ions']:
                if col in df.columns:
                    val = row.get(col)
                    if pd.notna(val) and str(val).strip():
                        parts = [x.strip() for x in str(val).split(',') if x.strip()]
                        if parts:
                            significant_frag_count = len(parts)
                            break

        # Prefer significant_single_aa_overhangs_protein_positions (5 ppm + >0.5% intensity) when present,
        # so Stage 1 drops peptides that only had overhangs by Comet's criteria but not our filters.
        single_aa_positions = {}
        significant_single_aa_used = False
        for col in ['significant_single_aa_overhangs_protein_positions', 'single_aa_overhangs_protein_positions']:
            if col not in df.columns:
                continue
            val = row.get(col)
            if pd.isna(val) or not str(val).strip():
                continue
            single_aa_positions = {}
            significant_single_aa_used = (col == 'significant_single_aa_overhangs_protein_positions')
            for part in str(val).strip().split(','):
                part = part.strip()
                if part and part[-1].isalpha() and part[:-1].isdigit():
                    try:
                        pos = int(part[:-1])
                        single_aa_positions[pos] = part[-1]
                    except ValueError:
                        pass
            if single_aa_positions:
                break

        # Significant fragment pairs/ions (our criteria: 5 ppm + >0.5% intensity) — used for backfill pool when 0 significant overhang pairs
        has_significant_fragment_pairs = False
        for col in ['significant_fragment_pairs', 'significant_fragment_ions']:
            if col not in df.columns:
                continue
            val = row.get(col)
            if pd.notna(val) and str(val).strip():
                has_significant_fragment_pairs = True
                break

        # For unique-peptides grid: overhang positions string and fragment pairs dict (so per-channel grids show red/yellow/blue-black)
        significant_single_aa_overhangs_protein_positions = None
        if single_aa_positions:
            significant_single_aa_overhangs_protein_positions = ','.join(f'{p}{single_aa_positions[p]}' for p in sorted(single_aa_positions))
        significant_fragment_pairs_raw = None
        for col in ['significant_fragment_pairs', 'significant_fragment_ions']:
            if col in df.columns and pd.notna(row.get(col)) and str(row.get(col)).strip():
                significant_fragment_pairs_raw = str(row[col]).strip()
                break
        single_aa_fragment_pairs_dict = _parse_single_aa_fragment_pairs_from_csv(single_aa_positions, row.get('single_aa_overhang_fragment_pairs') if 'single_aa_overhang_fragment_pairs' in df.columns else None)

        spectrum_window_min_rt = row.get('spectrum_window_min_rt')
        spectrum_window_max_rt = row.get('spectrum_window_max_rt')
        # Precursor (protein) for grouping; total_area used as confidence weight (w_area ∝ sqrt(area)), not a cutoff
        protein = None
        for col in ['protein', 'protein_id', 'accession', 'gene']:
            if col in df.columns and pd.notna(row.get(col)) and str(row.get(col)).strip():
                protein = str(row[col]).strip()
                break
        # Scan count for plot label: from CSV (e.g. num_psms from peak_windows) so plots show real count when CSV is 1 row per peptide
        scan_count_from_row = 1
        for col in ['num_psms', 'n_scans', 'scan_count', 'n_spectra', '_scan_count_total', '_scan_count']:
            if col in df.columns and pd.notna(row.get(col)):
                try:
                    v = int(float(row[col]))
                    if v > 0:
                        scan_count_from_row = v
                        break
                except (TypeError, ValueError):
                    pass
        peptides_for_rt_grid.append({
            'peptide_seq': seq,
            'charge': charge,
            'modifications': mods,
            'protein': protein,
            'rt': rt_sec,
            'detected_peak_min_rt': min_rt if pd.notna(min_rt) else None,
            'detected_peak_max_rt': max_rt if pd.notna(max_rt) else None,
            'collection_min_rt': coll_min if pd.notna(coll_min) else None,
            'collection_max_rt': coll_max if pd.notna(coll_max) else None,
            'spectrum_window_min_rt': spectrum_window_min_rt if pd.notna(spectrum_window_min_rt) else None,
            'spectrum_window_max_rt': spectrum_window_max_rt if pd.notna(spectrum_window_max_rt) else None,
            'mz': mz,
            'mz_theoretical': mz_theoretical,
            'apex_intensity': apex_intensity,
            'total_area': total_area,
            'm0_gt_m1_gt_m2': m0_gt_m1_gt_m2,
            'ms1_trace_ok': ms1_trace_ok,
            'single_aa_positions': single_aa_positions,
            '_significant_single_aa_used': significant_single_aa_used,
            '_has_significant_fragment_pairs': has_significant_fragment_pairs,
            '_scan_count': scan_count_from_row,
            'percolator_qvalue': percolator_qvalue,
            'percolator_PEP': percolator_PEP,
            'evalue': evalue,
            'n_candidates': n_candidates,
            'significant_frag_count': significant_frag_count,
            'matched fragment ions': row.get('matched fragment ions') if 'matched fragment ions' in df.columns else None,
            'matched fragment ion intensities': row.get('matched fragment ion intensities') if 'matched fragment ion intensities' in df.columns else None,
            'significant_single_aa_overhangs_protein_positions': significant_single_aa_overhangs_protein_positions,
            'significant_fragment_pairs': significant_fragment_pairs_raw,
            'single_aa_fragment_pairs': single_aa_fragment_pairs_dict,
        })

    def _parse_ion_intensities(ions_str, intensities_str):
        """Parse Comet 'matched fragment ions' and 'matched fragment ion intensities' into list of (ion_type, n, intensity). ion_type is 'c' or 'z'. z1_6 -> z, 6."""
        out = []
        if not ions_str or not intensities_str:
            return out
        ions_list = [x.strip() for x in str(ions_str).strip().strip('"').split(',') if x.strip()]
        try:
            int_list = [float(x.strip()) for x in str(intensities_str).strip().strip('"').split(',') if x.strip()]
        except (TypeError, ValueError):
            return out
        n_use = min(len(ions_list), len(int_list))
        for i in range(n_use):
            part = ions_list[i]
            intensity = int_list[i] if i < len(int_list) else 0.0
            if not part or intensity <= 0:
                continue
            if part.startswith('c') and part[1:].isdigit():
                out.append(('c', int(part[1:]), intensity))
            elif part.startswith('z1_'):
                n_str = part.split('_')[-1]
                if n_str.isdigit():
                    out.append(('z', int(n_str), intensity))
            elif part.startswith('z') and part[1:].isdigit():
                out.append(('z', int(part[1:]), intensity))
        return out

    def _aggregate_fragment_intensities(group):
        """Sum fragment intensities across PSMs by (type, n). Returns (fragment_intensity_c, fragment_intensity_z) dicts."""
        c_sum = {}
        z_sum = {}
        for p in group:
            ions_str = p.get('matched fragment ions')
            int_str = p.get('matched fragment ion intensities')
            for ion_type, n, intensity in _parse_ion_intensities(ions_str, int_str):
                if ion_type == 'c':
                    c_sum[n] = c_sum.get(n, 0.0) + intensity
                else:
                    z_sum[n] = z_sum.get(n, 0.0) + intensity
        return (c_sum, z_sum)

    # Collapse to one row per unique peptide (sequence + charge + mods) so RT windows are per unique peptide, not per PSM
    from collections import defaultdict
    def _key(p):
        return (p['peptide_seq'], p['charge'], str(p.get('modifications', '-') or '-').strip() or '-')
    # PSM pass/fail (same thresholds as chromatograms: Q<=0.01, PEP<=0.05, E<=0.01)
    FILTER_PSM_QVALUE_MAX = 0.01
    FILTER_PSM_PEP_MAX = 0.05
    FILTER_PSM_EVALUE_MAX = 0.01
    def _psm_passes(p):
        q = p.get('percolator_qvalue')
        if q is not None and not (isinstance(q, float) and pd.isna(q)) and q > FILTER_PSM_QVALUE_MAX:
            return False
        pep = p.get('percolator_PEP')
        if pep is not None and not (isinstance(pep, float) and pd.isna(pep)) and pep > FILTER_PSM_PEP_MAX:
            return False
        e = p.get('evalue')
        if e is not None and not (isinstance(e, float) and pd.isna(e)) and e > FILTER_PSM_EVALUE_MAX:
            return False
        return True
    groups = defaultdict(list)
    for p in peptides_for_rt_grid:
        groups[_key(p)].append(p)
    peptides_for_rt_grid = []
    for key, group in groups.items():
        seq, charge, mods = key
        min_rts = [p['detected_peak_min_rt'] for p in group if p.get('detected_peak_min_rt') is not None]
        max_rts = [p['detected_peak_max_rt'] for p in group if p.get('detected_peak_max_rt') is not None]
        coll_mins = [p['collection_min_rt'] for p in group if p.get('collection_min_rt') is not None]
        coll_maxes = [p['collection_max_rt'] for p in group if p.get('collection_max_rt') is not None]
        areas = [p['total_area'] for p in group if p.get('total_area') is not None and not (isinstance(p['total_area'], float) and (pd.isna(p['total_area']) or p['total_area'] < 0))]
        apex_ints = [p.get('apex_intensity') for p in group if p.get('apex_intensity') is not None and not (isinstance(p.get('apex_intensity'), float) and (pd.isna(p.get('apex_intensity')) or p.get('apex_intensity') < 0))]
        first = group[0]
        merged_positions = {}
        any_significant = any(p.get('_significant_single_aa_used') for p in group)
        for p in group:
            if any_significant and not p.get('_significant_single_aa_used'):
                continue  # When any PSM has significant data, only merge positions from those PSMs
            merged_positions.update(p.get('single_aa_positions') or {})
        total_psms = len(group)
        passed_psms = sum(1 for p in group if _psm_passes(p))
        # When CSV has one row per peptide, use that row's scan count for _scan_count_total so plot shows real scan count (e.g. num_psms from peak_windows)
        scan_count_merged = sum(p.get('_scan_count', 1) for p in group)
        scan_count_total_merged = total_psms if total_psms > 1 else (scan_count_merged or 1)
        m0_ok = any(p.get('m0_gt_m1_gt_m2') is True for p in group)
        ms1_ok = any(p.get('ms1_trace_ok') is True for p in group)  # True if any PSM had MS1 within 5 ppm
        # Observed m/z range and std across PSMs in this group
        mz_vals = []
        for p in group:
            v = p.get('mz')
            if v is not None and not (isinstance(v, float) and pd.isna(v)):
                try:
                    mz_vals.append(float(v))
                except (TypeError, ValueError):
                    pass
        mz_observed_min = min(mz_vals) if mz_vals else None
        mz_observed_max = max(mz_vals) if mz_vals else None
        mz_observed_std = float(np.std(mz_vals)) if len(mz_vals) > 1 else None
        # Significant fragment count: max across PSMs (peptide passes ">= N fragments"
        # if any PSM had >= N)
        sig_frag_count_vals = []
        for p in group:
            v = p.get('significant_frag_count')
            if v is not None and not (isinstance(v, float) and pd.isna(v)):
                try:
                    sig_frag_count_vals.append(int(float(v)))
                except (TypeError, ValueError):
                    pass
        sig_frag_count_max = max(sig_frag_count_vals) if sig_frag_count_vals else None
        has_significant_fragment_pairs_merged = any(p.get('_has_significant_fragment_pairs') for p in group)
        fragment_intensity_c, fragment_intensity_z = _aggregate_fragment_intensities(group)
        # Merge overhang/fragment keys for unique-peptides grid (per-channel and combined)
        merged_sig_overhangs = []
        for p in group:
            v = p.get('significant_single_aa_overhangs_protein_positions')
            if v and str(v).strip() and not (isinstance(v, float) and pd.isna(v)):
                for part in str(v).strip().split(','):
                    if part.strip():
                        merged_sig_overhangs.append(part.strip())
        def _overhang_pos_key(s):
            if not s or not str(s).strip() or len(str(s)) < 2:
                return 0
            s = str(s).strip()
            if s[-1].isalpha() and s[:-1].replace('.', '').replace('-', '').isdigit():
                try:
                    return int(float(s[:-1]))
                except (TypeError, ValueError):
                    pass
            return 0
        significant_single_aa_overhangs_protein_positions_merged = ','.join(sorted(set(merged_sig_overhangs), key=_overhang_pos_key)) if merged_sig_overhangs else None
        sig_pairs_set = set()
        for p in group:
            v = p.get('significant_fragment_pairs')
            if v and str(v).strip() and not (isinstance(v, float) and pd.isna(v)):
                for part in str(v).strip().replace('|', '-').split(','):
                    if '-' in part.strip():
                        sig_pairs_set.add(part.strip())
        significant_fragment_pairs_merged = ','.join(sorted(sig_pairs_set)) if sig_pairs_set else None
        single_aa_fragment_pairs_merged = {}
        for p in group:
            d = p.get('single_aa_fragment_pairs') or {}
            if isinstance(d, dict):
                for pos, pairs in d.items():
                    pos_int = int(pos) if isinstance(pos, (int, float)) else pos
                    if pos_int not in single_aa_fragment_pairs_merged:
                        single_aa_fragment_pairs_merged[pos_int] = set()
                    for x in (pairs if isinstance(pairs, (set, list)) else [pairs]):
                        if x is not None and str(x).strip():
                            single_aa_fragment_pairs_merged[pos_int].add(str(x).strip())
        # Min q and min PEP across PSMs (for filter: pass if q<0.05 OR PEP<0.05)
        q_vals = [p.get('percolator_qvalue') for p in group if p.get('percolator_qvalue') is not None and not (isinstance(p.get('percolator_qvalue'), float) and pd.isna(p.get('percolator_qvalue')))]
        pep_vals = [p.get('percolator_PEP') for p in group if p.get('percolator_PEP') is not None and not (isinstance(p.get('percolator_PEP'), float) and pd.isna(p.get('percolator_PEP')))]
        percolator_qvalue_min = min(q_vals) if q_vals else None
        percolator_PEP_min = min(pep_vals) if pep_vals else None
        # Spectrum window = RT range of MS1 spectra used; same as integration window (colored bar) when present
        sw_mins = [p.get('spectrum_window_min_rt') for p in group if p.get('spectrum_window_min_rt') is not None and not (isinstance(p.get('spectrum_window_min_rt'), float) and pd.isna(p.get('spectrum_window_min_rt')))]
        sw_maxes = [p.get('spectrum_window_max_rt') for p in group if p.get('spectrum_window_max_rt') is not None and not (isinstance(p.get('spectrum_window_max_rt'), float) and pd.isna(p.get('spectrum_window_max_rt')))]
        try:
            spectrum_window_min_rt = min(sw_mins) if sw_mins else first.get('spectrum_window_min_rt')
            spectrum_window_max_rt = max(sw_maxes) if sw_maxes else first.get('spectrum_window_max_rt')
        except (TypeError, ValueError):
            spectrum_window_min_rt = first.get('spectrum_window_min_rt')
            spectrum_window_max_rt = first.get('spectrum_window_max_rt')
        peptides_for_rt_grid.append({
            'peptide_seq': seq,
            'charge': charge,
            'modifications': mods,
            'protein': first.get('protein'),
            'rt': first.get('rt', 0.0),
            'detected_peak_min_rt': min(min_rts) if min_rts else first.get('detected_peak_min_rt'),
            'detected_peak_max_rt': max(max_rts) if max_rts else first.get('detected_peak_max_rt'),
            'collection_min_rt': min(coll_mins) if coll_mins else first.get('collection_min_rt'),
            'collection_max_rt': max(coll_maxes) if coll_maxes else first.get('collection_max_rt'),
            'spectrum_window_min_rt': spectrum_window_min_rt,
            'spectrum_window_max_rt': spectrum_window_max_rt,
            'm0_gt_m1_gt_m2': m0_ok,
            'ms1_trace_ok': ms1_ok,
            'mz': first.get('mz'),
            'mz_theoretical': first.get('mz_theoretical'),
            'mz_observed_min': mz_observed_min,
            'mz_observed_max': mz_observed_max,
            'mz_observed_std': mz_observed_std,
            'apex_intensity': max(apex_ints) if apex_ints else first.get('apex_intensity'),
            'total_area': max(areas) if areas else first.get('total_area'),
            'single_aa_positions': merged_positions,
            '_scan_count': scan_count_merged,
            '_scan_count_total': scan_count_total_merged,
            '_scan_count_passed': passed_psms,
            'n_candidates': max([p.get('n_candidates') for p in group if p.get('n_candidates') is not None]) if any(p.get('n_candidates') is not None for p in group) else first.get('n_candidates'),
            'significant_frag_count': sig_frag_count_max,
            '_has_significant_fragment_pairs': has_significant_fragment_pairs_merged,
            'fragment_intensity_c': fragment_intensity_c,
            'fragment_intensity_z': fragment_intensity_z,
            'percolator_qvalue_min': percolator_qvalue_min,
            'percolator_PEP_min': percolator_PEP_min,
            'significant_single_aa_overhangs_protein_positions': significant_single_aa_overhangs_protein_positions_merged,
            'significant_fragment_pairs': significant_fragment_pairs_merged,
            'single_aa_fragment_pairs': single_aa_fragment_pairs_merged,
        })
    print(f"Collapsed to {len(peptides_for_rt_grid)} unique peptides (sequence + charge + mods) for RT windows")

    # Main set: all peptides (unmodified + modified) unless --exclude-mods.
    modified_pool = []
    if exclude_mods:
        n_before = len(peptides_for_rt_grid)
        peptides_for_rt_grid = [p for p in peptides_for_rt_grid if not _has_modifications(p.get('modifications', '-'))]
        n_after = len(peptides_for_rt_grid)
        if n_before > n_after:
            print(f"Excluded {n_before - n_after} modified peptides (--exclude-mods); {n_after} peptides for RT windows")
    else:
        # Include modified in main set; no separate modified-only backfill pool
        n_mod = sum(1 for p in peptides_for_rt_grid if _has_modifications(p.get('modifications', '-')))
        if n_mod:
            print(f"Main set: all peptides including modified ({len(peptides_for_rt_grid)} total, {n_mod} modified)")

    if not peptides_for_rt_grid:
        print("No rows with valid window data, exiting.")
        sys.exit(0)

    # Drop peptides with no/zero total_area (gray bars with total area 0)
    def _has_positive_total_area(p):
        v = p.get('total_area')
        if v is None or (isinstance(v, float) and (pd.isna(v) or v <= 0)):
            return False
        try:
            return float(v) > 0
        except (TypeError, ValueError):
            return False
    n_before_area = len(peptides_for_rt_grid)
    funnel_n_input = n_before_area  # for filtration funnel schematic
    peptides_for_rt_grid = [p for p in peptides_for_rt_grid if _has_positive_total_area(p)]
    if len(peptides_for_rt_grid) < n_before_area:
        print(f"Excluded {n_before_area - len(peptides_for_rt_grid)} peptides with total_area 0 or missing; {len(peptides_for_rt_grid)} peptides for RT windows")

    if not peptides_for_rt_grid:
        print("No rows with valid window data after filtering, exiting.")
        sys.exit(0)

    # Full set of accepted peptides with valid windows (for "before thinning" plot — show all, not just main subset)
    all_accepted_for_rt_windows = list(peptides_for_rt_grid)

    # Max RT for gap backfill (used after each stage and after thinning)
    run_max_rt_sec = getattr(args, 'run_max_rt_sec', None)
    max_rt_global = None
    for p in all_accepted_for_rt_windows:
        for key in ('collection_max_rt', 'detected_peak_max_rt'):
            v = p.get(key)
            if v is not None:
                try:
                    v = float(v)
                    if not (v != v or v <= 0):
                        max_rt_global = v if max_rt_global is None else max(max_rt_global, v)
                except (TypeError, ValueError):
                    pass
    if run_max_rt_sec is not None:
        try:
            run_max = float(run_max_rt_sec)
            if run_max > 0:
                max_rt_global = run_max if max_rt_global is None else max(max_rt_global, run_max)
                print(f"Using run end RT for backfill: {run_max:.0f} s ({run_max/60:.1f} min)")
        except (TypeError, ValueError):
            pass

    all_excluded_by_key = {}  # key -> peptide; accumulate across stages for per-channel backup pool

    def _n_single_aa_overhangs(p):
        return len(p.get('single_aa_positions') or {})

    def _backfill_after_stage(kept_list, previous_list, stage_name):
        """Track excluded peptides for later backfill; do not add backfill here. Backfill runs once after channel assignment."""
        keys_kept = {_peptide_key(p) for p in kept_list}
        removed = [p for p in previous_list if _peptide_key(p) not in keys_kept]
        for p in removed:
            p['_exclusion_count'] = p.get('_exclusion_count', 0) + 1
            all_excluded_by_key[_peptide_key(p)] = p
        return list(kept_list)

    # Filtering stages (all applied in regenerate step; backfill after each stage to fill RT gaps)
    # Thresholds used for filtering and for plot/filename labels (change here to update script behavior and labels)
    MIN_SINGLE_AA_OVERHANGS = 1
    MIN_FRAGMENTS_MATCHED = 5
    Q_OR_PEP_THRESH = 0.05
    RELATIVE_AREA_MIN = 0.0001  # 0.01%: remove peptides with relative total area below this (Stage 5)
    BACKFILL_RELATIVE_AREA_MIN = 0.0005  # 0.05%: backfill candidates must have relative area >= this (weaker signal fills gaps)
    CHANNEL_ASSIGNMENT_RELATIVE_AREA_MIN = 0.01  # 1%: only peptides with relative area >= this get initial channel assignment and determine channel count; 0.05–1% are backfill only
    RELATIVE_AREA_PROTECTED = 0.07  # 7%: peptides with relative total area >= this bypass stages 2–6 (still need ≥2 single-AA overhangs)

    def _set_relative_area_for_plot(peptides):
        """Set relative_area_within_selection for each peptide in list (total_area / sum) so plots show % on every bar."""
        if not peptides:
            return
        _sum = sum((p.get('total_area') or 0) for p in peptides if p.get('total_area') is not None and not (isinstance(p.get('total_area'), float) and (pd.isna(p.get('total_area')) or p.get('total_area') <= 0)))
        try:
            _sum = float(_sum)
        except (TypeError, ValueError):
            _sum = 0.0
        for p in peptides:
            ta = p.get('total_area')
            if ta is not None and not (isinstance(ta, float) and (pd.isna(ta) or ta <= 0)):
                p['relative_area_within_selection'] = float(ta) / _sum if _sum > 0 else 0.0
            else:
                p['relative_area_within_selection'] = 0.0

    # Pipeline overview (stages 0-6)
    print("\n--- RT windows filtering pipeline (Stages 0-6) ---")
    print("  Stage 0: Reference (before any filter); all accepted peptides.")
    print("  Stage 1: Require ≥{} single-AA overhangs.".format(MIN_SINGLE_AA_OVERHANGS))
    print("  Stage 2: Require ≥{} fragments matched (MS2).".format(MIN_FRAGMENTS_MATCHED))
    print("  Stage 3: Require q<{} or PEP<{} (per PSM).".format(Q_OR_PEP_THRESH, Q_OR_PEP_THRESH))
    print("  Stage 4: Require envelope (M+0>M+1 or M+1>M+2).")
    print("  Stage 5: Require relative total area ≥{:.2f}%.".format(RELATIVE_AREA_MIN * 100))
    print("  Stage 6: Require charge-state consistency.")
    print("  (Backfill runs once after channel assignment, filling gaps in all channels.)")
    print("----------------------------------------\n")

    # Envelope at pipeline entry: only envelope-OK or protected (≥7% rel area) peptides enter stages 1–6.
    # So Stage 4 no longer rejects anyone — envelope is applied here instead.
    def _envelope_ok(p):
        v = p.get('m0_gt_m1_gt_m2')
        if v is True or (isinstance(v, str) and str(v).strip().lower() in ('true', '1', 'yes')):
            return True
        if isinstance(v, (int, float)) and not (isinstance(v, float) and (v != v or v == 0)):
            return bool(v)
        return False
    _sum_initial = sum((p.get('total_area') or 0) for p in all_accepted_for_rt_windows if p.get('total_area') is not None and not (isinstance(p.get('total_area'), float) and (pd.isna(p.get('total_area')) or p.get('total_area') <= 0)))
    try:
        _sum_initial = float(_sum_initial)
    except (TypeError, ValueError):
        _sum_initial = 0.0
    def _is_valuable(p):
        # protect_peptide (True/False) or valuable_sequence (1) — prioritize for channel assignment, avoid dropping
        if p.get('protect_peptide') in (True, 1, 'True', 'true', '1'):
            return True
        v = p.get('valuable_sequence')
        if v == 1 or (isinstance(v, (int, float)) and v == 1):
            return True
        if isinstance(v, str) and str(v).strip() == '1':
            return True
        return False
    for p in all_accepted_for_rt_windows:
        rel = (float(p.get('total_area') or 0) / _sum_initial) if _sum_initial > 0 else 0.0
        p['_protected_high_relative_area'] = rel >= RELATIVE_AREA_PROTECTED or _is_valuable(p)
    n_before_env = len(all_accepted_for_rt_windows)
    all_accepted_for_rt_windows = [p for p in all_accepted_for_rt_windows if p.get('_protected_high_relative_area') or _envelope_ok(p)]
    n_env_excluded = n_before_env - len(all_accepted_for_rt_windows)
    if n_env_excluded:
        print(f"Envelope at pipeline entry: excluded {n_env_excluded} peptides (envelope not OK and relative area < {RELATIVE_AREA_PROTECTED*100:.0f}%); {len(all_accepted_for_rt_windows)} peptides enter stages 1–6.")

    # Stage 1: require at least MIN_SINGLE_AA_OVERHANGS single-amino-acid overhangs
    print("[Stage 1/6] Running: require ≥{} single-AA overhangs. Next: Stage 2 (fragments matched).".format(MIN_SINGLE_AA_OVERHANGS))
    peptides_stage1 = [p for p in all_accepted_for_rt_windows if len(p.get('single_aa_positions') or {}) >= MIN_SINGLE_AA_OVERHANGS]
    rejected_stage1 = [p for p in all_accepted_for_rt_windows if p not in peptides_stage1]
    for p in rejected_stage1:
        p.setdefault('_rejection_reason', f'Fewer than {MIN_SINGLE_AA_OVERHANGS} single-AA overhangs')
    n_drop_single_aa = len(all_accepted_for_rt_windows) - len(peptides_stage1)
    if n_drop_single_aa:
        print(f"Stage 1 (>={MIN_SINGLE_AA_OVERHANGS} single-AA overhangs): {len(all_accepted_for_rt_windows)} -> {len(peptides_stage1)} peptides (dropped {n_drop_single_aa})")
    peptides_stage1 = _backfill_after_stage(peptides_stage1, all_accepted_for_rt_windows, "Stage 1")
    # Relative area within stage-1 set: used to protect high-abundance peaks (≥7%) from later filters
    _sum_area_stage1 = sum((p.get('total_area') or 0) for p in peptides_stage1 if p.get('total_area') is not None and not (isinstance(p.get('total_area'), float) and (pd.isna(p.get('total_area')) or p.get('total_area') <= 0)))
    try:
        _sum_area_stage1 = float(_sum_area_stage1)
    except (TypeError, ValueError):
        _sum_area_stage1 = 0.0
    for p in peptides_stage1:
        ta = p.get('total_area')
        if ta is not None and not (isinstance(ta, float) and (pd.isna(ta) or ta <= 0)) and _sum_area_stage1 > 0:
            rel = float(ta) / _sum_area_stage1
            p['relative_area_stage1'] = rel
            p['_protected_high_relative_area'] = rel >= RELATIVE_AREA_PROTECTED
        else:
            p['relative_area_stage1'] = 0.0
            p['_protected_high_relative_area'] = False
    n_protected = sum(1 for p in peptides_stage1 if p.get('_protected_high_relative_area'))
    if n_protected:
        print(f"  Protected (≥{RELATIVE_AREA_PROTECTED*100:.0f}% relative total area, exempt from stages 2–6): {n_protected} peptides")
    # Stage 2: require at least MIN_FRAGMENTS_MATCHED significant fragments (skip if no significant fragment counts in data)
    print("[Stage 2/6] Running: require ≥{} fragments matched. Next: Stage 3 (q-value/PEP).".format(MIN_FRAGMENTS_MATCHED))
    has_sig_frag_count = any(p.get('significant_frag_count') is not None for p in peptides_stage1)
    if has_sig_frag_count:
        stage2_passed = [p for p in peptides_stage1 if p.get('_protected_high_relative_area') or (p.get('significant_frag_count') or 0) >= MIN_FRAGMENTS_MATCHED]
        rejected_stage2 = [p for p in peptides_stage1 if p not in stage2_passed]
        for p in rejected_stage2:
            p.setdefault('_rejection_reason', f'Fewer than {MIN_FRAGMENTS_MATCHED} significant fragments')
        n_drop_frag = len(peptides_stage1) - len(stage2_passed)
        if n_drop_frag:
            print(f"Stage 2 (>={MIN_FRAGMENTS_MATCHED} significant fragments): {len(peptides_stage1)} -> {len(stage2_passed)} peptides (dropped {n_drop_frag})")
        peptides_stage2 = _backfill_after_stage(stage2_passed, peptides_stage1, "Stage 2")
    else:
        rejected_stage2 = []
        peptides_stage2 = list(peptides_stage1)
        print(f"Stage 2 (>={MIN_FRAGMENTS_MATCHED} significant fragments): skipped (no significant fragment columns in CSV)")
    # Stage 3: require EITHER q < Q_OR_PEP_THRESH OR PEP < Q_OR_PEP_THRESH (at least one PSM per peptide)
    print("[Stage 3/6] Running: require q<{} or PEP<{}. Next: Stage 4 (envelope).".format(Q_OR_PEP_THRESH, Q_OR_PEP_THRESH))
    def _passes_q_or_pep(p):
        """Keep peptide if q < 0.05 or PEP < 0.05; if either q or PEP is missing, do not filter (keep)."""
        q = p.get('percolator_qvalue_min') if p.get('percolator_qvalue_min') is not None else p.get('percolator_qvalue')
        pep = p.get('percolator_PEP_min') if p.get('percolator_PEP_min') is not None else p.get('percolator_PEP')
        q_ok = q is not None and not (isinstance(q, float) and pd.isna(q))
        pep_ok = pep is not None and not (isinstance(pep, float) and pd.isna(pep))
        if not q_ok or not pep_ok:
            return True  # either missing: do not filter by this criterion
        if q < Q_OR_PEP_THRESH or pep < Q_OR_PEP_THRESH:
            return True
        return False
    stage3_passed = [p for p in peptides_stage2 if p.get('_protected_high_relative_area') or _passes_q_or_pep(p)]
    rejected_stage3 = [p for p in peptides_stage2 if p not in stage3_passed]
    for p in rejected_stage3:
        p.setdefault('_rejection_reason', f'q≥{Q_OR_PEP_THRESH} and PEP≥{Q_OR_PEP_THRESH}')
    n_drop_qpep = len(peptides_stage2) - len(stage3_passed)
    if n_drop_qpep:
        print(f"Stage 3 (q<{Q_OR_PEP_THRESH} or PEP<{Q_OR_PEP_THRESH}): {len(peptides_stage2)} -> {len(stage3_passed)} peptides (dropped {n_drop_qpep})")
    peptides_stage3 = _backfill_after_stage(stage3_passed, peptides_stage2, "Stage 3")
    # Stage 4: require M+0 > M+1 OR M+1 > M+2 (from CSV/peak_windows m0_gt_m1_gt_m2)
    print("[Stage 4/6] Running: require envelope (M+0>M+1 or M+1>M+2). Next: Stage 5 (relative area).")
    def _envelope_ok(p):
        v = p.get('m0_gt_m1_gt_m2')
        if v is True or (isinstance(v, str) and str(v).strip().lower() in ('true', '1', 'yes')):
            return True
        if isinstance(v, (int, float)) and not (isinstance(v, float) and (v != v or v == 0)):
            return bool(v)
        return False
    stage4_passed = [p for p in peptides_stage3 if p.get('_protected_high_relative_area') or _envelope_ok(p)]
    rejected_stage4 = [p for p in peptides_stage3 if p not in stage4_passed]
    for p in rejected_stage4:
        p.setdefault('_rejection_reason', 'Envelope (M+0>M+1 or M+1>M+2) not met')
    n_drop_env = len(peptides_stage3) - len(stage4_passed)
    if n_drop_env:
        print(f"Stage 4 (M+0>M+1 or M+1>M+2): {len(peptides_stage3)} -> {len(stage4_passed)} peptides (dropped {n_drop_env})")
    peptides_stage4 = _backfill_after_stage(stage4_passed, peptides_stage3, "Stage 4")
    # After envelope filter: relative area within selection (total_area / sum(all total_areas))
    # Normalizes sampling differences; useful for weighting fragment reliability and HDX workflows.
    _sum_area_stage4 = sum((p.get('total_area') or 0) for p in peptides_stage4 if p.get('total_area') is not None and not (isinstance(p.get('total_area'), float) and (pd.isna(p.get('total_area')) or p.get('total_area') <= 0)))
    try:
        _sum_area_stage4 = float(_sum_area_stage4)
    except (TypeError, ValueError):
        _sum_area_stage4 = 0.0
    for p in peptides_stage4:
        ta = p.get('total_area')
        if ta is not None and not (isinstance(ta, float) and (pd.isna(ta) or ta <= 0)):
            p['relative_area_within_selection'] = float(ta) / _sum_area_stage4 if _sum_area_stage4 > 0 else 0.0
        else:
            p['relative_area_within_selection'] = 0.0
    if _sum_area_stage4 > 0 and peptides_stage4:
        print(f"  Relative area (after envelope): total_area / sum(total_area) computed for {len(peptides_stage4)} peptides (sum={_sum_area_stage4:.2e})")
    # Stage 4: no thinning — use fewest channels to accommodate all; save channels plot + unique sequences
    peptides_stage4_for_plot = list(peptides_stage4)
    min_channels_stage4 = find_min_channels(peptides_stage4_for_plot) if peptides_stage4_for_plot else 0
    if peptides_stage4_for_plot and min_channels_stage4 > 0:
        assign_peptides_to_channels(peptides_stage4_for_plot, min_channels_stage4)
        n_before_enforce_s4 = len(peptides_stage4_for_plot)
        peptides_stage4_for_plot = enforce_channel_no_overlap(peptides_stage4_for_plot, num_channels=min_channels_stage4, min_window_sec=5.0, gap_sec=1.0)
        if len(peptides_stage4_for_plot) < n_before_enforce_s4:
            print(f"Stage 4: channel no-overlap: {n_before_enforce_s4} -> {len(peptides_stage4_for_plot)} peptides (cropped/dropped overlaps)")
        print(f"Stage 4: {len(peptides_stage4_for_plot)} peptides in {min_channels_stage4} channel(s) (fewest channels to fit all, no overlap)")

    # Stage 5: Remove peptides with relative total area (relative intensity) below 0.01%, after Stage 4 envelope, before channel assignment
    print("[Stage 5/6] Running: require relative total area ≥{:.2f}%. Next: Stage 6 (charge-state consistency).".format(RELATIVE_AREA_MIN * 100))
    stage5_passed = [p for p in peptides_stage4 if p.get('_protected_high_relative_area') or (p.get('relative_area_within_selection') or 0) >= RELATIVE_AREA_MIN]
    rejected_stage5 = [p for p in peptides_stage4 if p not in stage5_passed]
    for p in rejected_stage5:
        p.setdefault('_rejection_reason', f'Relative area <{RELATIVE_AREA_MIN*100:.2f}%')
    n_drop_relarea = len(peptides_stage4) - len(stage5_passed)
    if n_drop_relarea:
        print(f"Stage 5 (relative total area ≥{RELATIVE_AREA_MIN*100:.2f}%): {len(peptides_stage4)} -> {len(stage5_passed)} peptides (dropped {n_drop_relarea})")
    peptides_stage5 = _backfill_after_stage(stage5_passed, peptides_stage4, "Stage 5")

    # Stage 6: No filter (charge-state consistency removed). All Stage 5 peptides pass through.
    print("[Stage 6/6] No charge-state filter. Next: channel assignment and plots.")
    stage6_passed = list(peptides_stage5)
    rejected_stage6 = []
    peptides_stage6 = _backfill_after_stage(stage6_passed, peptides_stage5, "Stage 6")
    # total_area is a confidence weight only (no strict cutoff). fragment_score ∝ sqrt(fragment_area); see w_area below.
    # Main set: use all peptides that passed filters (no thinning). Assign to as many channels as needed so no overlap within channel.
    kept_7 = peptides_stage6  # keep name for total_area_range list
    peptides_main = list(peptides_stage6)
    if not peptides_main:
        print("No peptides for RT windows after filtering stages. Exiting.")
        sys.exit(0)

    # Initial channel assignment: only consider peaks with > 0.1% relative area; 0.01–0.1% go to backfill only
    kept = [p for p in peptides_main if (p.get('relative_area_within_selection') or 0) >= CHANNEL_ASSIGNMENT_RELATIVE_AREA_MIN]
    not_in_kept = [p for p in peptides_main if p not in kept]  # 0.01% <= rel < 1%; eligible for backfill if >= 0.05%
    if not_in_kept:
        print(f"Channel assignment: {len(kept)} peptides with ≥{CHANNEL_ASSIGNMENT_RELATIVE_AREA_MIN*100:.0f}% relative area get initial slots (channel count from these); {len(not_in_kept)} peptides (<1% rel area) available for backfill if ≥{BACKFILL_RELATIVE_AREA_MIN*100:.2f}%")
    removed = []  # no thinning
    # Diagnostic: min window start across kept peaks before channel assignment (if plot starts later → plotting bug)
    _starts_kept = [(_get_collection_bounds(p)[0]) for p in kept]
    _starts_kept = [float(x) for x in _starts_kept if x is not None]
    min_rt_start_kept = min(_starts_kept) if _starts_kept else None
    if min_rt_start_kept is not None:
        print(f"[diagnostic] Before channel assignment: min_rt_start across {len(kept)} kept peaks = {min_rt_start_kept:.1f} s ({min_rt_start_kept/60:.2f} min)")
    else:
        print(f"[diagnostic] Before channel assignment: no valid window start in {len(kept)} kept peaks")
    # Distinct sets of 3 channels; within each set same peak cannot repeat; within each channel no overlap.
    # Goal: maximally fill each channel; repeating peptides across different sets of 3 is encouraged later.
    # Prioritize protect_peptide=True or valuable_sequence=1 and high relative area for first channels (workflow Step 9)
    def _valuable_sort_key(p):
        is_prot = p.get('protect_peptide') in (True, 1, 'True', 'true', '1')
        v = p.get('valuable_sequence')
        is_val = is_prot or (v == 1 or (isinstance(v, (int, float)) and v == 1) or (isinstance(v, str) and str(v).strip() == '1'))
        rel = p.get('relative_area_within_selection') or 0
        return (0 if is_val else 1, -rel)
    kept = sorted(kept, key=_valuable_sort_key)
    num_channels = assign_peptides_to_channels_by_layers(kept, channels_per_layer=3)
    num_channels = max(1, num_channels)
    n_layers = (num_channels + 2) // 3
    max_overlap = max_overlap_depth(kept)
    layers_lb = math.ceil(max_overlap / 3.0) if max_overlap else 0
    print(f"Channel assignment: {num_channels} channel(s) ({n_layers} layer(s) × 3), two-phase (min layers then priority within layer, no overlap within channel)")
    print(f"[diagnostic] Max overlap depth = {max_overlap} → theoretical min layers = ceil({max_overlap}/3) = {layers_lb}; actual layers = {n_layers}")
    n_ch_kept = [sum(1 for p in kept if p.get('channel') == ch) for ch in range(num_channels)]
    print(f"Channels (all passed): " + ", ".join(f"Ch{ch}={n_ch_kept[ch]}" for ch in range(num_channels)))

    # Backfill after channel assignment: fill gaps in all channels that have RT windows assigned.
    # Backfill only with peptides that (1) pass envelope, (2) have relative area >= backfill min, (3) are "repeated peaks" (same key in >= 2 layers in kept).
    print(f"Backfill (after channel assignment): filling gaps in all {num_channels} channel(s) (candidates: envelope OK, rel area ≥{BACKFILL_RELATIVE_AREA_MIN*100:.2f}%, repeated in another layer)...")
    all_excluded = list(all_excluded_by_key.values())
    # Repeated peaks = (seq, charge, mods) appears in at least 2 different 3-channel layers in kept
    CH_PER_LAYER = 3
    layers_by_key = defaultdict(set)
    for p in kept:
        layers_by_key[_peptide_key(p)].add((p.get('channel') or 0) // CH_PER_LAYER)
    repeated_peptide_keys = {k for k, layers in layers_by_key.items() if len(layers) >= 2}
    all_excluded_backfill = [
        p for p in all_excluded
        if (p.get('relative_area_within_selection') or 0) >= BACKFILL_RELATIVE_AREA_MIN
        and _envelope_ok(p)
        and _peptide_key(p) in repeated_peptide_keys
    ]
    primary_list = [p for p in all_excluded_backfill if p.get('_exclusion_count') == 1]
    backup_list = [p for p in all_excluded_backfill if p.get('_exclusion_count') > 1]
    primary_2plus = [p for p in primary_list if _n_single_aa_overhangs(p) >= 2]
    primary_1 = [p for p in primary_list if _n_single_aa_overhangs(p) == 1]
    primary_0 = [p for p in primary_list if _n_single_aa_overhangs(p) == 0]
    backup_2plus = [p for p in backup_list if _n_single_aa_overhangs(p) >= 2]
    backup_1 = [p for p in backup_list if _n_single_aa_overhangs(p) == 1]
    backup_0 = [p for p in backup_list if _n_single_aa_overhangs(p) == 0]
    # Peptides with significant fragment pairs/ions but 0 significant single-AA overhang positions
    fragmentation_only_pool = [
        p for p in all_excluded_backfill
        if _n_single_aa_overhangs(p) == 0 and p.get('_has_significant_fragment_pairs')
    ]
    # Peptides that passed all stages but have 0.01–0.1% rel area: no initial channel slot, but eligible for backfill
    not_in_kept_backfill = [
        p for p in not_in_kept
        if _envelope_ok(p) and (p.get('relative_area_within_selection') or 0) >= BACKFILL_RELATIVE_AREA_MIN
    ]
    if not_in_kept_backfill:
        print(f"Backfill: {len(not_in_kept_backfill)} peptides (0.05–1% rel area, no initial slot) added to backfill pool")
    backfill_pool_low_rel = [p for p in all_excluded_backfill if (p.get('relative_area_within_selection') or 0) < RELATIVE_AREA_MIN]
    used_keys = set()
    all_backfill = []
    for ch in range(num_channels):
        ch0 = (ch // CH_PER_LAYER) * CH_PER_LAYER
        layer_peptides = [p for p in kept if ch0 <= p.get('channel') < ch0 + CH_PER_LAYER] + [p for p in all_backfill if ch0 <= p.get('channel') < ch0 + CH_PER_LAYER]
        peptides_ch = [p for p in kept if p.get('channel') == ch]
        bf_pools = [
            primary_2plus + [p for p in modified_pool if _n_single_aa_overhangs(p) >= 2],
            primary_1 + [p for p in modified_pool if _n_single_aa_overhangs(p) == 1],
            primary_0 + [p for p in modified_pool if _n_single_aa_overhangs(p) == 0],
            backup_2plus, backup_1, backup_0,
            fragmentation_only_pool,  # after overhang-based: backfill with significant fragmentation even if 0 overhang pairs
            not_in_kept_backfill,  # 0.01–0.1% rel area: passed all stages but no initial slot
        ]
        ch_backfill = []
        n_frag_only = 0
        for pool in bf_pools:
            bf = backfill_peptides_from_gaps(
                peptides_ch, pool, min_overlap=1,
                require_passed_psm=False, exclude_keys=used_keys,
                max_rt=max_rt_global, min_gap_width=10.0,
                layer_peptides=layer_peptides
            )
            if pool is fragmentation_only_pool and bf:
                n_frag_only = len(bf)
            for p in bf:
                p['channel'] = ch
                p['is_backfill'] = True
                p['backfill_rejection_reason'] = p.get('_rejection_reason', '')
            ch_backfill.extend(bf)
            peptides_ch = peptides_ch + bf  # so next pool sees updated gaps
            used_keys |= {_peptide_key(p) for p in bf}
        backfill_ch3 = backfill_peptides_from_gaps(
            peptides_ch, backfill_pool_low_rel, min_overlap=4,
            require_passed_psm=False, exclude_keys=used_keys,
            max_rt=max_rt_global, min_gap_width=10.0,
            layer_peptides=layer_peptides
        )
        used_keys |= {_peptide_key(p) for p in backfill_ch3}
        for p in backfill_ch3:
            p['channel'] = ch
            p['is_backfill'] = True
            p['backfill_rejection_reason'] = p.get('_rejection_reason', '')
        ch_backfill.extend(backfill_ch3)
        all_backfill.extend(ch_backfill)
        if ch_backfill:
            frag_msg = f"; significant fragment pairs (0 overhang): {n_frag_only}" if n_frag_only else ""
            print(f"Backfill Ch{ch}: added {len(ch_backfill)} peptides into channel gaps (prefer ≥2 then 1 overhang, then 0{frag_msg}; low rel area where <4 overlap: {len(backfill_ch3)})")

    # Repeat pass: fill remaining gaps by reusing peptides from other layers (outline-only in plot).
    # Within one set of 3 channels: same peak cannot repeat (candidate_pool = not in this layer).
    # Repeating the same peptide across different sets of 3 is encouraged to maximally use channel time.
    n_repeat_total = 0
    for ch in range(num_channels):
        peptides_ch = [p for p in kept if p.get('channel') == ch] + [p for p in all_backfill if p.get('channel') == ch]
        added_this_ch = 0
        while True:
            layer_ch = ch // CH_PER_LAYER
            layer_keys = {_peptide_key(p) for p in kept + all_backfill if (p.get('channel') or 0) // CH_PER_LAYER == layer_ch}
            candidate_pool = [p for p in kept + all_backfill if _peptide_key(p) not in layer_keys]
            repeat_list = backfill_repeat_into_gaps(
                peptides_ch, candidate_pool, min_overlap=1,
                max_rt=max_rt_global, min_gap_width=10.0, min_start_end_gap=1.0
            )
            if not repeat_list:
                break
            for p in repeat_list:
                p['channel'] = ch
                p['is_backfill'] = True
                all_backfill.append(p)
                peptides_ch.append(p)
                n_repeat_total += 1
                added_this_ch += 1
        if added_this_ch > 0:
            print(f"Backfill Ch{ch+1}: repeated {added_this_ch} peak(s) into remaining gaps (outline-only; iterative fill)")
    if n_repeat_total:
        print(f"Backfill (repeat): {n_repeat_total} total repeat(s) across channels (outline-only in plot)")

    peptides_for_rt_grid = kept + all_backfill
    n_before_enforce = len(peptides_for_rt_grid)
    peptides_for_rt_grid = enforce_channel_no_overlap(peptides_for_rt_grid, num_channels=num_channels, min_window_sec=5.0, gap_sec=1.0)
    if len(peptides_for_rt_grid) < n_before_enforce:
        print(f"Channel no-overlap: dropped or cropped {n_before_enforce - len(peptides_for_rt_grid)} peptides to remove overlaps")

    # Relative area within primary selection (total_area / sum(all total_areas)) for weighting and HDX interpretation
    _sum_area_primary = sum((p.get('total_area') or 0) for p in peptides_for_rt_grid if p.get('total_area') is not None and not (isinstance(p.get('total_area'), float) and (pd.isna(p.get('total_area')) or p.get('total_area') <= 0)))
    try:
        _sum_area_primary = float(_sum_area_primary)
    except (TypeError, ValueError):
        _sum_area_primary = 0.0
    _areas_primary = [float(p.get('total_area')) for p in peptides_for_rt_grid if p.get('total_area') is not None and not (isinstance(p.get('total_area'), float) and (pd.isna(p.get('total_area')) or p.get('total_area') <= 0))]
    _max_area_primary = max(_areas_primary) if _areas_primary else 0.0
    for p in peptides_for_rt_grid:
        ta = p.get('total_area')
        if ta is not None and not (isinstance(ta, float) and (pd.isna(ta) or ta <= 0)):
            ta_f = float(ta)
            p['relative_area_within_selection'] = ta_f / _sum_area_primary if _sum_area_primary > 0 else 0.0
            # fragment_score ∝ sqrt(fragment_area); w_area = sqrt(area/max_area) as confidence weight (1 at max, <1 for smaller)
            p['w_area'] = math.sqrt(ta_f / _max_area_primary) if _max_area_primary > 0 else 0.0
        else:
            p['relative_area_within_selection'] = 0.0
            p['w_area'] = 0.0

    # Reorder within each layer only (Ch1 = highest signal in layer, etc.) so layer packing is preserved
    reorder_channels_by_relative_area_within_layers(peptides_for_rt_grid, num_channels, channels_per_layer=3)

    # No shuffle between channels: assignment (≥1% in 3-channel sets, then backfill) is final

    # Repeated peptide: earliest channel = shaded (filled); later channels = outline only
    set_repeat_outline_by_earliest_channel(peptides_for_rt_grid, num_channels)

    # Sort within each channel by RT (retention time) so peaks appear in time order
    new_order = []
    for ch in range(num_channels):
        peptides_ch = [p for p in peptides_for_rt_grid if p.get('channel') == ch]
        peptides_ch.sort(key=lambda p: float(p.get('collection_min_rt') or p.get('detected_peak_min_rt') or 0))
        new_order.extend(peptides_ch)
    peptides_for_rt_grid = new_order

    n_ch = [sum(1 for p in peptides_for_rt_grid if p.get('channel') == ch) for ch in range(num_channels)]
    sum_rel_ch = [sum((p.get('relative_area_within_selection') or 0) for p in peptides_for_rt_grid if p.get('channel') == ch) for ch in range(num_channels)]
    print(f"Channels (after backfill + repeat, reordered by collected signal descending): " + ", ".join(f"Ch{ch+1}={n_ch[ch]} (relΣ={sum_rel_ch[ch]:.3f})" for ch in range(num_channels)) + " (no overlap within channel)")
    # Diagnostic: min_rt_start per channel after backfill
    for ch in range(num_channels):
        peptides_ch = [p for p in peptides_for_rt_grid if p.get('channel') == ch]
        starts_ch = [(_get_collection_bounds(p)[0]) for p in peptides_ch]
        starts_ch = [float(x) for x in starts_ch if x is not None]
        min_ch = min(starts_ch) if starts_ch else None
        if min_ch is not None:
            print(f"[diagnostic] Ch{ch+1}: min_rt_start = {min_ch:.1f} s ({min_ch/60:.2f} min) over {len(peptides_ch)} peptides")
        else:
            print(f"[diagnostic] Ch{ch+1}: no valid window start (n={len(peptides_ch)})")

    # Write channel assignment to CSV for downstream use
    channels_csv = os.path.join(rt_windows_dir, f'{output_base}_rt_windows_channels.csv')
    with open(channels_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['peptide_seq', 'charge', 'modifications', 'channel', 'collection_min_rt', 'collection_max_rt',
                    'detected_peak_min_rt', 'detected_peak_max_rt', 'total_area', 'relative_area_within_selection', 'w_area',
                    'n_charge_states', 'charge_state_consistent', 'multi_charge_agreement', 'is_backfill', 'is_repeat_in_channel'])
        for p in peptides_for_rt_grid:
            cmin, cmax = _get_collection_bounds(p)
            w.writerow([
                p.get('peptide_seq', ''),
                p.get('charge', ''),
                p.get('modifications', '-'),
                p.get('channel', -1),
                cmin, cmax,
                p.get('detected_peak_min_rt', ''),
                p.get('detected_peak_max_rt', ''),
                p.get('total_area', ''),
                p.get('relative_area_within_selection', ''),
                p.get('w_area', ''),
                p.get('n_charge_states', ''),
                1 if p.get('charge_state_consistent') else 0,
                1 if p.get('multi_charge_agreement') else 0,
                1 if p.get('is_backfill') else 0,
                1 if p.get('is_repeat_in_channel') else 0,
            ])
    print(f"  Wrote channel assignment to {channels_csv} (includes relative_area_within_selection, w_area, charge-state consistency)")

    # N-channel plot dataframe: all pertinent info, ordered by channel then peak drift window min
    drift_rows = _compute_drift_limits_per_channel(peptides_for_rt_grid, num_channels=num_channels)
    df_rows = []
    for p, drift_min, drift_max in drift_rows:
        cmin, cmax = _get_collection_bounds(p)
        pos_dict = p.get('single_aa_positions') or {}
        def _pos_key(item):
            k = item[0]
            try:
                return (int(k), k)
            except (TypeError, ValueError):
                return (0, str(k))
        pos_parts = [f"{k}{v}" for k, v in sorted(pos_dict.items(), key=_pos_key)[:20]]
        pos_str = ','.join(pos_parts)
        if len(pos_dict) > 20:
            pos_str += f',+{len(pos_dict)-20}'
        # Integration window = RT range used for quantification (colored bar). Same as spectrum_window when present; else detected_peak.
        int_min = p.get('detected_peak_min_rt') or p.get('spectrum_window_min_rt')
        int_max = p.get('detected_peak_max_rt') or p.get('spectrum_window_max_rt')
        df_rows.append({
            'channel': p.get('channel', -1),
            'drift_min_rt': drift_min,
            'drift_max_rt': drift_max,
            'collection_min_rt': cmin,
            'collection_max_rt': cmax,
            'integration_min_rt': int_min,
            'integration_max_rt': int_max,
            'peptide_seq': p.get('peptide_seq', ''),
            'charge': p.get('charge', ''),
            'modifications': p.get('modifications', '-'),
            'detected_peak_min_rt': p.get('detected_peak_min_rt'),
            'detected_peak_max_rt': p.get('detected_peak_max_rt'),
            'spectrum_window_min_rt': p.get('spectrum_window_min_rt'),
            'spectrum_window_max_rt': p.get('spectrum_window_max_rt'),
            'total_area': p.get('total_area'),
            'relative_area_within_selection': p.get('relative_area_within_selection'),
            'w_area': p.get('w_area'),
            'apex_intensity': p.get('apex_intensity'),
            'is_backfill': 1 if p.get('is_backfill') else 0,
            'is_repeat_in_channel': 1 if p.get('is_repeat_in_channel') else 0,
            'mz': p.get('mz'),
            'mz_theoretical': p.get('mz_theoretical'),
            'mz_observed_min': p.get('mz_observed_min'),
            'mz_observed_max': p.get('mz_observed_max'),
            'mz_observed_std': p.get('mz_observed_std'),
            'm0_gt_m1_gt_m2': p.get('m0_gt_m1_gt_m2'),
            'ms1_trace_ok': p.get('ms1_trace_ok'),
            'scan_count_passed': p.get('_scan_count_passed'),
            'n_charge_states': p.get('n_charge_states'),
            'charge_state_consistent': p.get('charge_state_consistent'),
            'multi_charge_agreement': p.get('multi_charge_agreement'),
            'isotope_spacing_ok': p.get('isotope_spacing_ok'),
            'single_aa_positions': pos_str if pos_str else '',
        })
    df_channel = pd.DataFrame(df_rows)
    channel_dataframe_csv = os.path.join(rt_windows_dir, f'{output_base}_rt_windows_by_channel_dataframe.csv')
    df_channel.to_csv(channel_dataframe_csv, index=False)
    print(f"  Wrote N-channel selected peptides (by channel, drift_min) to {channel_dataframe_csv}")

    # Copy individual chromatograms for the thinned peptide set into one directory for inspection
    copy_filtered_chromatograms(peptides_for_rt_grid, output_dir, csv_dir, output_base)

    # Build fallback list from peak_windows_all.csv (accepted + rejected) for empty zoom segments
    main_keys = {(p['peptide_seq'], p['charge'], str(p.get('modifications', '-') or '-').strip() or '-') for p in peptides_for_rt_grid}
    peptides_fallback = []
    for search_dir in [output_dir, csv_dir, os.path.join(csv_dir, 'chromatograms_peptides'), os.path.join(output_dir, 'chromatograms_peptides'), os.path.join(csv_dir, 'accepted', 'chromatograms_peptides'), os.path.join(output_dir, 'accepted', 'chromatograms_peptides')]:
        pw_path = os.path.join(search_dir, 'peak_windows_all.csv')
        if os.path.isfile(pw_path):
            try:
                df_pw = pd.read_csv(pw_path)
                if 'peptide' not in df_pw.columns or 'detected_peak_min_rt' not in df_pw.columns:
                    break
                def _norm_mods(m):
                    if m is None or (isinstance(m, float) and pd.isna(m)):
                        return '-'
                    s = str(m).strip()
                    return s if s and s.lower() != 'nan' else '-'
                for _, r in df_pw.iterrows():
                    seq = str(r.get('peptide', '')).strip()
                    ch = int(r.get('charge', 0)) if pd.notna(r.get('charge')) else 0
                    mods = _norm_mods(r.get('modifications', '-'))
                    if (seq, ch, mods) in main_keys:
                        continue
                    min_rt = r.get('detected_peak_min_rt')
                    max_rt = r.get('detected_peak_max_rt')
                    if pd.isna(min_rt) or pd.isna(max_rt):
                        continue
                    try:
                        min_rt = float(min_rt)
                        max_rt = float(max_rt)
                    except (TypeError, ValueError):
                        continue
                    rt_mid = (min_rt + max_rt) / 2.0
                    coll_min = r.get('collection_min_rt')
                    coll_max = r.get('collection_max_rt')
                    if pd.notna(coll_min):
                        coll_min = float(coll_min)
                    else:
                        coll_min = None
                    if pd.notna(coll_max):
                        coll_max = float(coll_max)
                    else:
                        coll_max = None
                    mz = r.get('precursor_mz')
                    if pd.notna(mz):
                        mz = float(mz)
                    else:
                        mz = None
                    total_area = r.get('total_area')
                    if pd.notna(total_area):
                        try:
                            total_area = float(total_area)
                        except (TypeError, ValueError):
                            total_area = None
                    else:
                        total_area = None
                    apex_intensity = r.get('apex_intensity')
                    if pd.notna(apex_intensity):
                        try:
                            apex_intensity = float(apex_intensity)
                        except (TypeError, ValueError):
                            apex_intensity = None
                    else:
                        apex_intensity = None
                    peptides_fallback.append({
                        'peptide_seq': seq,
                        'charge': ch,
                        'modifications': mods,
                        'rt': rt_mid,
                        'detected_peak_min_rt': min_rt,
                        'detected_peak_max_rt': max_rt,
                        'collection_min_rt': coll_min,
                        'collection_max_rt': coll_max,
                        'mz': mz,
                        'mz_theoretical': mz,
                        'apex_intensity': apex_intensity,
                        'total_area': total_area,
                        'single_aa_positions': {},
                        '_scan_count': 1,
                    })
                if peptides_fallback:
                    if exclude_mods:
                        peptides_fallback = [p for p in peptides_fallback if not _has_modifications(p.get('modifications', '-'))]
                    print(f"  Loaded {len(peptides_fallback)} fallback peptides from {pw_path} for empty zoom segments")
                break
            except Exception as e:
                print(f"  Note: could not load fallback from {pw_path}: {e}")
            break

    # Import and call create_rt_integration_windows_plot and create_rt_overlay_only_plot (same as combined_overhang_visualization)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from visualization import create_rt_integration_windows_plot, create_rt_overlay_only_plot

    protein_id = 'Filtered peptides'
    protein_sequence_plot = None
    fasta_path = getattr(args, 'fasta', None)
    if fasta_path and os.path.isfile(fasta_path):
        try:
            from visualization.multi_aa import parse_fasta
            sequences = parse_fasta(fasta_path)
            protein_sequence_plot = sequences.get(protein_id)
            if protein_sequence_plot is None and protein_id != 'Filtered peptides':
                for fid, seq in sequences.items():
                    if protein_id in fid or fid in protein_id:
                        protein_sequence_plot = seq
                        break
        except Exception as e:
            print(f"  Note: could not load protein sequence from FASTA for RT plot labels: {e}")
    if 'protein' in df.columns and pd.notna(df['protein'].iloc[0]):
        protein_id = str(df['protein'].iloc[0]).strip() or protein_id

    # Three versions of complete RT window plot: before thinning, after thinning, after backfill
    peptides_before_thinning = list(all_accepted_for_rt_windows)  # all accepted with windows (before passed/envelope/mz/thin)
    peptides_after_thinning = list(kept)
    peptides_after_backfill = list(peptides_for_rt_grid)  # kept + per-channel backfill

    # Write _ms1_ms2_qvalues.csv: accepted windows integration + all columns from original CSV + Selected (1=in final set, 0=not)
    selected_keys = {
        (str(p.get('peptide_seq', '')).strip(), int(p.get('charge', 0)), _norm_mods(p.get('modifications', '-')))
        for p in peptides_after_backfill
    }
    df_out = df.copy()
    df_out['Selected'] = df_out.apply(
        lambda row: 1
        if (str(row.get('plain_peptide', '')).strip(), int(row.get('charge', 0)), _norm_mods(row.get('modifications', '-')))
        in selected_keys
        else 0,
        axis=1,
    )
    original_csv = getattr(args, 'original_csv', None)
    if original_csv and os.path.isfile(original_csv):
        try:
            with open(original_csv, 'r') as f:
                first_orig = f.readline()
            skip_orig = 1 if 'CometVersion' in first_orig else 0
            orig = pd.read_csv(original_csv, sep=',', skiprows=skip_orig, engine='python', quotechar='"', on_bad_lines='warn')
            if 'plain_peptide' in orig.columns and 'charge' in orig.columns:
                merge_on = [c for c in ['plain_peptide', 'charge', 'modifications'] if c in df_out.columns and c in orig.columns]
                orig_dedup = orig.drop_duplicates(subset=merge_on, keep='first')
                extra_cols = [c for c in orig_dedup.columns if c not in df_out.columns]
                if extra_cols:
                    df_out = df_out.merge(orig_dedup[merge_on + extra_cols], on=merge_on, how='left')
        except Exception as e:
            print(f"  Note: could not merge original CSV for _ms1_ms2_qvalues: {e}")
    try:
        from visualization.csv_validation import validate_and_log
        validate_and_log(
            df_out,
            source_name=output_base + '_ms1_ms2_qvalues',
            required_columns=('plain_peptide', 'charge', 'Selected'),
            validate_identifiers=True,
            validate_numeric=True,
        )
    except Exception as e:
        print(f"  Validation warning: {e}")
    out_ms1_ms2 = os.path.join(output_dir, f'{output_base}_ms1_ms2_qvalues.csv')
    df_out.to_csv(out_ms1_ms2, index=False)
    print(f"  Wrote {os.path.abspath(out_ms1_ms2)} (Selected=1 for {len(selected_keys)} peptides in final set)")

    rt_max_sec_plot = getattr(args, 'run_max_rt_sec', None)  # extend x-axis to run end when set (e.g. 1140 for 19 min)
    # Global total_area range so all plots share the same color scale
    def _total_area_val(p):
        v = p.get('total_area')
        if v is None or (isinstance(v, float) and (pd.isna(v) or v <= 0)):
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None
    _all_peptide_lists = [peptides_before_thinning, peptides_after_thinning, peptides_after_backfill, peptides_stage1, peptides_stage2, peptides_stage3, kept_7]
    if peptides_fallback:
        _all_peptide_lists.append(peptides_fallback)
    _all_areas = []
    for _pl in _all_peptide_lists:
        for _p in _pl:
            _a = _total_area_val(_p)
            if _a is not None:
                _all_areas.append(_a)
    total_area_range = None
    if _all_areas:
        total_area_range = (min(_all_areas), max(_all_areas))
        if total_area_range[1] <= total_area_range[0]:
            total_area_range = (total_area_range[0], total_area_range[0] + 1.0)
        print(f"  Global total_area range for color scale: [{total_area_range[0]:.2e}, {total_area_range[1]:.2e}]")
    overlay_files_for_progress = []  # collect (accepted_path, rejected_path) per stage for copying to filtering_progress

    # Load chromatogram traces for overlay plots (RT vs intensity peaks). Use full set so all stages can show real peaks.
    trace_by_key = {}
    _npz_path = getattr(args, 'chromatogram_traces', None)
    _trace_csv_path = getattr(args, 'chromatogram_trace_csv', None)
    _raw_file = getattr(args, 'mzml', None) or getattr(args, 'raw_file', None)
    if _npz_path and peptides_before_thinning:
        _traces = _load_chromatogram_traces(_npz_path, _trace_csv_path, peptides_before_thinning)
        if _traces is not None and len(_traces) == len(peptides_before_thinning):
            trace_by_key = {_peptide_key(p): t for p, t in zip(peptides_before_thinning, _traces)}
            print(f"  Loaded chromatogram traces for {len(trace_by_key)} peptides (stage overlay plots will show RT vs intensity peaks)")
    if not trace_by_key and _raw_file and peptides_before_thinning:
        _traces = _reextract_chromatogram_traces(_raw_file, peptides_before_thinning)
        if _traces is not None and len(_traces) == len(peptides_before_thinning):
            trace_by_key = {_peptide_key(p): t for p, t in zip(peptides_before_thinning, _traces)}
            print(f"  Re-extracted chromatogram traces for {len(trace_by_key)} peptides (stage overlay plots will show RT vs intensity peaks)")
    if not trace_by_key and peptides_before_thinning:
        print("  Stage overlay plots will use synthetic peaks (no traces). For real RT vs intensity: pass --mzml <file.mzML> or --chromatogram-traces <traces.npz>)")

    def _write_stage_overlays(rejected_list, accepted_list, stage_num, stage_short_label, out_accepted_path, out_rejected_path, trace_by_key=None):
        """Write overlay plots for accepted and rejected at this stage. Paths appended to overlay_files_for_progress.
        trace_by_key: optional dict peptide_key -> (rts_sec, intensity) so overlay shows real chromatogram peaks (RT on x, intensity on y)."""
        def _traces_for_list(pl):
            if not trace_by_key or not pl:
                return None
            out = [trace_by_key.get(_peptide_key(p), None) for p in pl]
            return out if any(t is not None for t in out) else None
        def _log_path(p):
            if p and p.endswith('.png'):
                return p[:-4] + '_log.png'
            return (p + '_log.png') if p else None
        if accepted_list:
            accepted_traces = _traces_for_list(accepted_list)
            create_rt_overlay_only_plot(
                accepted_list, out_accepted_path, protein_id,
                total_area_range=total_area_range, stage_label=f'Stage {stage_num} (accepted): {stage_short_label}',
                peptide_traces=accepted_traces, rt_max_sec=rt_max_sec_plot
            )
            overlay_files_for_progress.append(out_accepted_path)
            log_accepted = _log_path(out_accepted_path)
            if log_accepted:
                overlay_files_for_progress.append(log_accepted)
        if rejected_list:
            rejected_traces = _traces_for_list(rejected_list)
            create_rt_overlay_only_plot(
                rejected_list, out_rejected_path, protein_id,
                total_area_range=total_area_range, stage_label=f'Stage {stage_num} (rejected): {stage_short_label}',
                peptide_traces=rejected_traces, rt_max_sec=rt_max_sec_plot
            )
            overlay_files_for_progress.append(out_rejected_path)
            log_rejected = _log_path(out_rejected_path)
            if log_rejected:
                overlay_files_for_progress.append(log_rejected)

    # Stage 0: before any filter (reference) — all accepted, none rejected
    print("\n--- Writing stage plots ---")
    print("[Plots Stage 0/6] Writing reference (before any filter). Next: Stage 1 plots.")
    _set_relative_area_for_plot(peptides_before_thinning)
    n_stage0 = len(peptides_before_thinning)
    rt_windows_before = os.path.join(rt_windows_dir, f'{output_base}_stage0_rt_windows_before_thinning.png')
    create_rt_integration_windows_plot(
        peptides_before_thinning, rt_windows_before, protein_id,
        dedupe_by_peptide=False, protein_sequence=protein_sequence_plot, rt_max_sec=rt_max_sec_plot, peptides_fallback=peptides_fallback or None,
        sort_by_collection_start=True, max_zoom_segments=0, total_area_range=total_area_range,
        stage_label='Stage 0', n_peptides_previous=None
    )
    overlay_stage0_accepted = os.path.join(rt_windows_dir, f'{output_base}_stage0_overlay_accepted.png')
    overlay_stage0_rejected = os.path.join(rt_windows_dir, f'{output_base}_stage0_overlay_rejected.png')
    _write_stage_overlays([], peptides_before_thinning, 0, 'before thinning', overlay_stage0_accepted, overlay_stage0_rejected, trace_by_key=trace_by_key)
    # Stage 1: after requiring >= MIN_SINGLE_AA_OVERHANGS single-AA overhangs
    print("[Plots Stage 1/6] Writing ≥{} single-AA overhangs. Next: Stage 2 plots.".format(MIN_SINGLE_AA_OVERHANGS))
    _set_relative_area_for_plot(peptides_stage1)
    n_stage1 = len(peptides_stage1)
    rt_windows_2_single_aa = os.path.join(rt_windows_dir, f'{output_base}_stage1_rt_windows_after_{MIN_SINGLE_AA_OVERHANGS}_single_aa_overhangs.png')
    create_rt_integration_windows_plot(
        peptides_stage1, rt_windows_2_single_aa, protein_id,
        dedupe_by_peptide=False, protein_sequence=protein_sequence_plot, rt_max_sec=rt_max_sec_plot, peptides_fallback=peptides_fallback or None,
        sort_by_collection_start=True, max_zoom_segments=0, total_area_range=total_area_range,
        stage_label=f'Stage 1 (≥{MIN_SINGLE_AA_OVERHANGS} single-AA overhangs)', n_peptides_previous=n_stage0
    )
    overlay_stage1_accepted = os.path.join(rt_windows_dir, f'{output_base}_stage1_overlay_accepted.png')
    overlay_stage1_rejected = os.path.join(rt_windows_dir, f'{output_base}_stage1_overlay_rejected.png')
    _write_stage_overlays(rejected_stage1, peptides_stage1, 1, f'≥{MIN_SINGLE_AA_OVERHANGS} single-AA overhangs', overlay_stage1_accepted, overlay_stage1_rejected, trace_by_key=trace_by_key)
    # Stage 2: after requiring >= MIN_FRAGMENTS_MATCHED fragments matched in MS2
    print("[Plots Stage 2/6] Writing ≥{} fragments matched. Next: Stage 3 plots.".format(MIN_FRAGMENTS_MATCHED))
    _set_relative_area_for_plot(peptides_stage2)
    n_stage2 = len(peptides_stage2)
    rt_windows_5_fragments = os.path.join(rt_windows_dir, f'{output_base}_stage2_rt_windows_after_{MIN_FRAGMENTS_MATCHED}_fragments_matched.png')
    create_rt_integration_windows_plot(
        peptides_stage2, rt_windows_5_fragments, protein_id,
        dedupe_by_peptide=False, protein_sequence=protein_sequence_plot, rt_max_sec=rt_max_sec_plot, peptides_fallback=peptides_fallback or None,
        sort_by_collection_start=True, max_zoom_segments=0, total_area_range=total_area_range,
        stage_label=f'Stage 2 (≥{MIN_FRAGMENTS_MATCHED} fragments matched)', n_peptides_previous=n_stage1
    )
    overlay_stage2_accepted = os.path.join(rt_windows_dir, f'{output_base}_stage2_overlay_accepted.png')
    overlay_stage2_rejected = os.path.join(rt_windows_dir, f'{output_base}_stage2_overlay_rejected.png')
    _write_stage_overlays(rejected_stage2, peptides_stage2, 2, f'≥{MIN_FRAGMENTS_MATCHED} fragments matched', overlay_stage2_accepted, overlay_stage2_rejected, trace_by_key=trace_by_key)
    # Stage 3: after requiring q < Q_OR_PEP_THRESH OR PEP < Q_OR_PEP_THRESH
    print("[Plots Stage 3/6] Writing q/PEP filter. Next: Stage 4 plots.")
    _set_relative_area_for_plot(peptides_stage3)
    n_stage3 = len(peptides_stage3)
    q_pep_thresh_str = str(Q_OR_PEP_THRESH).replace('.', '')
    rt_windows_q_pep = os.path.join(rt_windows_dir, f'{output_base}_stage3_rt_windows_after_q_or_pep_{q_pep_thresh_str}.png')
    create_rt_integration_windows_plot(
        peptides_stage3, rt_windows_q_pep, protein_id,
        dedupe_by_peptide=False, protein_sequence=protein_sequence_plot, rt_max_sec=rt_max_sec_plot, peptides_fallback=peptides_fallback or None,
        sort_by_collection_start=True, max_zoom_segments=0, total_area_range=total_area_range,
        stage_label=f'Stage 3 (q<{Q_OR_PEP_THRESH} or PEP<{Q_OR_PEP_THRESH})', n_peptides_previous=n_stage2
    )
    overlay_stage3_accepted = os.path.join(rt_windows_dir, f'{output_base}_stage3_overlay_accepted.png')
    overlay_stage3_rejected = os.path.join(rt_windows_dir, f'{output_base}_stage3_overlay_rejected.png')
    _write_stage_overlays(rejected_stage3, peptides_stage3, 3, f'q<{Q_OR_PEP_THRESH} or PEP<{Q_OR_PEP_THRESH}', overlay_stage3_accepted, overlay_stage3_rejected, trace_by_key=trace_by_key)
    rt_windows_stage4 = None  # set below when stage4 envelope channels plot is created
    # Stage 4: after envelope filter
    print("[Plots Stage 4/6] Writing envelope filter. Next: Stage 5 plots.")
    n_stage4 = len(peptides_stage4) if peptides_stage4 else 0
    if peptides_stage4:
        rt_windows_after_envelope = os.path.join(rt_windows_dir, f'{output_base}_stage4_rt_windows_after_envelope_filter.png')
        create_rt_integration_windows_plot(
            peptides_stage4, rt_windows_after_envelope, protein_id,
            dedupe_by_peptide=False, protein_sequence=protein_sequence_plot, rt_max_sec=rt_max_sec_plot, peptides_fallback=None,
            sort_by_collection_start=True, max_zoom_segments=0, total_area_range=total_area_range,
            stage_label='Stage 4', n_peptides_previous=n_stage3
        )
        overlay_stage4_accepted = os.path.join(rt_windows_dir, f'{output_base}_stage4_overlay_accepted.png')
        overlay_stage4_rejected = os.path.join(rt_windows_dir, f'{output_base}_stage4_overlay_rejected.png')
        _write_stage_overlays(rejected_stage4, peptides_stage4, 4, 'envelope (M+0>M+1 or M+1>M+2)', overlay_stage4_accepted, overlay_stage4_rejected, trace_by_key=trace_by_key)
    # Stage 4 envelope: N-channel plot (no thinning) + unique sequences (when we have channels)
    if peptides_stage4_for_plot and min_channels_stage4 > 0:
        rt_windows_stage4 = os.path.join(rt_windows_dir, f'{output_base}_stage4_channels_rt_windows_envelope.png')
        create_rt_integration_windows_plot(
            peptides_stage4_for_plot, rt_windows_stage4, protein_id,
            dedupe_by_peptide=False, protein_sequence=protein_sequence_plot, rt_max_sec=rt_max_sec_plot, peptides_fallback=None,
            sort_by_collection_start=True, max_zoom_segments=0, sort_by_channel=True, peptide_traces=None,
            total_area_range=total_area_range, num_channels=min_channels_stage4,
            stage_label='Stage 4 (channels)', n_peptides_previous=None
        )
        stage4_dir = os.path.join(rt_windows_dir, 'stage4_envelope')
        os.makedirs(stage4_dir, exist_ok=True)
        fasta_path = getattr(args, 'fasta', None)
        if fasta_path and os.path.isfile(fasta_path):
            # Show all stage4 peptides in unique_peptides (min_single_aa_overhangs=0); they have RT window data.
            if _write_combined_visualization_for_peptide_set(df, peptides_stage4_for_plot, _norm_mods, stage4_dir, 'stage4_envelope_combined', fasta_path, protein_id, filter_criteria_override={'min_single_aa_overhangs': 0}):
                print(f"  Stage 4 envelope: channels plot and unique sequences saved in {os.path.abspath(stage4_dir)}")
    # Stage 5: after relative area ≥0.01% filter
    print("[Plots Stage 5/6] Writing relative area filter. Next: Stage 6 plots.")
    rt_windows_stage5 = None
    if peptides_stage5:
        _set_relative_area_for_plot(peptides_stage5)
        rt_windows_stage5 = os.path.join(rt_windows_dir, f'{output_base}_stage5_rt_windows_after_relative_area_0p05.png')
        create_rt_integration_windows_plot(
            peptides_stage5, rt_windows_stage5, protein_id,
            dedupe_by_peptide=False, protein_sequence=protein_sequence_plot, rt_max_sec=rt_max_sec_plot, peptides_fallback=None,
            sort_by_collection_start=True, max_zoom_segments=0, total_area_range=total_area_range,
            stage_label=f'Stage 5 (relative area ≥{RELATIVE_AREA_MIN*100:.2f}%)', n_peptides_previous=n_stage4
        )
        overlay_stage5_accepted = os.path.join(rt_windows_dir, f'{output_base}_stage5_overlay_accepted.png')
        overlay_stage5_rejected = os.path.join(rt_windows_dir, f'{output_base}_stage5_overlay_rejected.png')
        _write_stage_overlays(rejected_stage5, peptides_stage5, 5, f'relative area ≥{RELATIVE_AREA_MIN*100:.2f}%', overlay_stage5_accepted, overlay_stage5_rejected, trace_by_key=trace_by_key)
    n_stage5_count = len(peptides_after_thinning)
    # Stage 6: after backfill (kept + backfill)
    print("[Plots Stage 6/6] Writing charge-state/backfill. Next: filtering progress and done.")
    rt_windows_after_backfill_file = os.path.join(rt_windows_dir, f'{output_base}_stage6_rt_windows_after_backfill.png')
    create_rt_integration_windows_plot(
        peptides_after_backfill, rt_windows_after_backfill_file, protein_id,
        dedupe_by_peptide=False, protein_sequence=protein_sequence_plot, rt_max_sec=rt_max_sec_plot, peptides_fallback=peptides_fallback or None,
        sort_by_collection_start=True, max_zoom_segments=0, total_area_range=total_area_range,
        stage_label='Stage 6', n_peptides_previous=n_stage5_count
    )
    overlay_stage6_accepted = os.path.join(rt_windows_dir, f'{output_base}_stage6_overlay_accepted.png')
    overlay_stage6_rejected = os.path.join(rt_windows_dir, f'{output_base}_stage6_overlay_rejected.png')
    _write_stage_overlays(rejected_stage6, peptides_stage6, 6, 'charge-state consistency', overlay_stage6_accepted, overlay_stage6_rejected, trace_by_key=trace_by_key)
    # Also write/copy main RT windows name for backward compatibility (same as after backfill)
    rt_windows_file = os.path.join(rt_windows_dir, f'{output_base}_rt_windows.png')
    if rt_windows_after_backfill_file != rt_windows_file:
        try:
            shutil.copy2(rt_windows_after_backfill_file, rt_windows_file)
        except OSError:
            pass
    # Thinning and progress filtering plots in one folder for easy review
    filtering_progress_dir = os.path.join(rt_windows_dir, 'filtering_progress')
    os.makedirs(filtering_progress_dir, exist_ok=True)
    for src in [rt_windows_before, rt_windows_2_single_aa, rt_windows_5_fragments, rt_windows_q_pep,
                rt_windows_after_backfill_file]:
        if src and os.path.isfile(src):
            shutil.copy2(src, os.path.join(filtering_progress_dir, os.path.basename(src)))
    if peptides_stage4:
        if os.path.isfile(rt_windows_after_envelope):
            shutil.copy2(rt_windows_after_envelope, os.path.join(filtering_progress_dir, os.path.basename(rt_windows_after_envelope)))
    if rt_windows_stage5 and os.path.isfile(rt_windows_stage5):
        shutil.copy2(rt_windows_stage5, os.path.join(filtering_progress_dir, os.path.basename(rt_windows_stage5)))
    if rt_windows_stage4 and os.path.isfile(rt_windows_stage4):
        shutil.copy2(rt_windows_stage4, os.path.join(filtering_progress_dir, os.path.basename(rt_windows_stage4)))
    for ov_path in overlay_files_for_progress:
        if ov_path and os.path.isfile(ov_path):
            shutil.copy2(ov_path, os.path.join(filtering_progress_dir, os.path.basename(ov_path)))
    print(f"  Filtering and thinning plots also in {os.path.abspath(filtering_progress_dir)}")
    by_peptide_file = os.path.join(rt_windows_dir, f'{output_base}_rt_windows_by_peptide.png')
    create_rt_integration_windows_plot(
        peptides_after_backfill, by_peptide_file, protein_id,
        dedupe_by_peptide=False, protein_sequence=protein_sequence_plot, rt_max_sec=rt_max_sec_plot, peptides_fallback=peptides_fallback or None,
        sort_by_collection_start=False, max_zoom_segments=0, total_area_range=total_area_range
    )
    # By channel: rows grouped by channel (0, 1, 2) so within each block no overlaps; overlay uses real chromatogram peaks if traces available
    peptide_traces = None
    npz_path = getattr(args, 'chromatogram_traces', None)
    trace_csv_path = getattr(args, 'chromatogram_trace_csv', None)
    raw_file = getattr(args, 'mzml', None) or getattr(args, 'raw_file', None)
    if npz_path:
        peptide_traces = _load_chromatogram_traces(npz_path, trace_csv_path, peptides_after_backfill)
        if peptide_traces is not None:
            print(f"  Loaded chromatogram traces for {len(peptide_traces)} peptides from {npz_path}")
        else:
            print("  Could not load chromatogram traces (npz/CSV match failed); overlay will be empty")
    if peptide_traces is None and raw_file:
        peptide_traces = _reextract_chromatogram_traces(raw_file, peptides_after_backfill)
        if peptide_traces is not None:
            print(f"  Re-extracted chromatogram traces for {len(peptide_traces)} peptides from {raw_file}")
        else:
            print("  Could not re-extract chromatogram traces; overlay will be empty")
    if peptide_traces is None:
        print("  Overlay will be empty (no chromatogram traces). To show peaks: pass --mzml <path/to/file.mzML> or --chromatogram-traces <path/to/traces.npz>)")
    rt_windows_by_channel_file = os.path.join(rt_windows_dir, f'{output_base}_rt_windows_by_channel.png')
    create_rt_integration_windows_plot(
        peptides_after_backfill, rt_windows_by_channel_file, protein_id,
        dedupe_by_peptide=False, protein_sequence=protein_sequence_plot, rt_max_sec=rt_max_sec_plot, peptides_fallback=peptides_fallback or None,
        sort_by_collection_start=True, max_zoom_segments=0, sort_by_channel=True, peptide_traces=peptide_traces, total_area_range=total_area_range,
        num_channels=num_channels
    )
    # Primary: run full chromatogram visualization (as a check) using primary CSV as filter, then add drift lines
    primary_chrom_dir = os.path.join(output_dir, 'peptide_chromatograms_primary')
    os.makedirs(primary_chrom_dir, exist_ok=True)
    rt_windows_by_channel_log_file = (rt_windows_by_channel_file[:-4] + '_log.png') if rt_windows_by_channel_file.endswith('.png') else (rt_windows_by_channel_file + '_log.png')
    for src in [rt_windows_by_channel_file, rt_windows_by_channel_log_file, channel_dataframe_csv, channels_csv]:
        if src and os.path.isfile(src):
            shutil.copy2(src, os.path.join(primary_chrom_dir, os.path.basename(src)))
    raw_file = getattr(args, 'mzml', None) or getattr(args, 'raw_file', None)
    filtered_chrom_dir = os.path.join(output_dir, f'{output_base}_chromatograms')
    primary_keys = _primary_peptide_keys_from_three_channel_csv(three_channel_csv, min_single_aa_overhangs=1)
    re_extract_chromatograms = getattr(args, 're_extract_chromatograms', False)
    # Skip full chromatogram re-extraction when we already have PNGs for the primary set (e.g. from copy_filtered_chromatograms)
    if raw_file and os.path.isfile(raw_file):
        primary_filter_csv = os.path.join(primary_chrom_dir, 'primary_peptide_filter_for_chromatograms.csv')
        _write_filter_csv_from_three_channel_csv(three_channel_csv, primary_filter_csv, min_single_aa_overhangs=1)
        if not re_extract_chromatograms and primary_keys and _chromatogram_dir_has_pngs_for_peptide_keys(filtered_chrom_dir, primary_keys):
            # Copy existing PNGs from filtered_chromatograms to primary dir so layout is consistent; no mzML re-extraction
            peptide_png_pattern = re.compile(r'^\d{5}_\d{4}_')
            all_pngs = [f for f in os.listdir(filtered_chrom_dir) if f.endswith('.png') and not f.startswith('all_') and peptide_png_pattern.match(f)] if os.path.isdir(filtered_chrom_dir) else []
            copied = 0
            for safe_seq, charge, mods_norm in primary_keys:
                charge_suffix = f'_z{charge}_'
                charge_suffix_end = f'_z{charge}.png'
                for f in all_pngs:
                    if safe_seq not in f or (charge_suffix not in f and not f.endswith(charge_suffix_end)) or not _chromatogram_filename_mods_match(f, mods_norm):
                        continue
                    src = os.path.join(filtered_chrom_dir, f)
                    dst = os.path.join(primary_chrom_dir, f)
                    if os.path.isfile(src):
                        try:
                            shutil.copy2(src, dst)
                            copied += 1
                        except OSError:
                            pass
                    break
            print(f"  Skipping chromatogram re-extraction (using existing PNGs for primary set; copied {copied} to {os.path.basename(primary_chrom_dir)})")
        elif primary_filter_csv and os.path.isfile(primary_filter_csv):
            primary_rejected_dir = os.path.join(primary_chrom_dir, 'rejected')
            print("  Running full chromatogram visualization for primary peptide set (check)...")
            _run_full_chromatograms_for_filter(
                csv_path, raw_file, primary_filter_csv,
                primary_chrom_dir, primary_rejected_dir,
                exclude_mods=exclude_mods
            )
            # Update filtered_chromatograms with newly generated (new layout) PNGs so layout change is visible there too
            if os.path.isdir(primary_chrom_dir) and os.path.isdir(filtered_chrom_dir):
                primary_pngs = [f for f in os.listdir(primary_chrom_dir) if f.endswith('.png') and re.match(r'^\d{5}_\d{4}_', f)]
                for f in primary_pngs:
                    src = os.path.join(primary_chrom_dir, f)
                    dst = os.path.join(filtered_chrom_dir, f)
                    if os.path.isfile(src):
                        try:
                            shutil.copy2(src, dst)
                        except OSError:
                            pass
                if primary_pngs:
                    print(f"  Updated {filtered_chrom_dir} with {len(primary_pngs)} chromatograms (new layout)")
        else:
            print("  Could not create primary filter CSV; skipping full chromatogram run")
    else:
        # Fallback: copy existing full chromatogram PNGs and add drift lines (no mzML to re-run)
        write_peptide_chromatograms_with_drift_lines(
            peptides_after_backfill, drift_rows, peptide_traces, primary_chrom_dir, csv_dir=csv_dir
        )
    if os.path.isfile(rt_windows_by_channel_file):
        print(f"  {num_channels}-channel plot and CSVs also saved in {os.path.abspath(primary_chrom_dir)}")
    # Combined/unique_peptides (saved in primary dir)
    fasta_path = getattr(args, 'fasta', None)
    protein_id = getattr(args, 'protein', None) or ''
    if not fasta_path or not os.path.isfile(fasta_path):
        print("  Skipping combined and unique_peptides grid plots (no FASTA file). Pass --fasta <path> to generate these.")
    elif fasta_path and os.path.isfile(fasta_path):
        if _write_combined_visualization_for_peptide_set(df, peptides_after_backfill, _norm_mods, primary_chrom_dir, 'combined', fasta_path, protein_id):
            print(f"  Combined/unique_peptides plots saved in {os.path.abspath(primary_chrom_dir)}")
    # Explicit unique peptides grid + segmented/family plots (one row per sequence+charge+mods; when fasta and peptides exist)
    if fasta_path and os.path.isfile(fasta_path) and peptides_after_backfill:
        try:
            from visualization.combined import create_accepted_peptides_protein_grid
            from visualization.multi_aa import parse_fasta
            sequences = parse_fasta(fasta_path)
            if not sequences:
                pass  # no sequences in FASTA
            else:
                pid = protein_id.strip() if protein_id else None
                if not pid:
                    pid = list(sequences.keys())[0]
                protein_sequence = sequences.get(pid)
                if protein_sequence is None and pid:
                    clean_pid = re.sub(r'(\d+)$', '', pid)
                    pid_alt = re.sub(r'^sp\|', '', clean_pid)
                    protein_sequence = sequences.get(clean_pid) or sequences.get(pid_alt)
                if protein_sequence is None:
                    protein_sequence = list(sequences.values())[0]
                    pid = list(sequences.keys())[0]
                if protein_sequence:
                    protein_length = len(protein_sequence)
                    unique_pep_output = os.path.join(primary_chrom_dir, f'{output_base}_unique_peptides.png')
                    create_accepted_peptides_protein_grid(
                        peptides_after_backfill, protein_sequence, pid or 'protein', protein_length,
                        unique_pep_output, primary_chrom_dir
                    )
                    print(f"  Unique peptides plots (full, by_total_area, lowres, segments, families) saved in {os.path.abspath(primary_chrom_dir)}")
                    # Per-channel unique peptides (one row per sequence+charge+mods for that channel)
                    for ch in range(num_channels):
                        peptides_ch = [p for p in peptides_after_backfill if p.get('channel') == ch]
                        if not peptides_ch:
                            continue
                        ch_unique_output = os.path.join(rt_windows_dir, f'{output_base}_rt_windows_by_channel_ch{ch + 1}_unique_peptides.png')
                        try:
                            create_accepted_peptides_protein_grid(
                                peptides_ch, protein_sequence, pid or 'protein', protein_length,
                                ch_unique_output, rt_windows_dir, channel_only=True
                            )
                            print(f"  Channel {ch + 1} unique peptides (single plot, no families) saved to {os.path.basename(ch_unique_output)}")
                        except Exception as ec:
                            print(f"  Warning: could not create unique_peptides for channel {ch + 1}: {ec}", file=sys.stderr)
        except Exception as e:
            print(f"  Warning: could not create unique_peptides/segmented plots: {e}", file=sys.stderr)

    # Binned histogram of total_area for primary peptides
    primary_areas = []
    for p in peptides_after_backfill:
        v = _total_area_val(p)
        if v is not None:
            primary_areas.append(v)
    if primary_areas:
        fig_hist, ax_hist = plt.subplots(1, 1, figsize=(8, 5))
        ax_hist.hist(primary_areas, bins=min(25, max(10, len(primary_areas) // 3)), color='steelblue', edgecolor='navy', alpha=0.8)
        ax_hist.set_xlabel('Total area (MS1 peak)', fontsize=11)
        ax_hist.set_ylabel('Number of peptides', fontsize=11)
        ax_hist.set_title(f'Peptides total_area distribution (n={len(primary_areas)})', fontsize=12)
        ax_hist.grid(True, alpha=0.3, linestyle='--')
        if max(primary_areas) / min(primary_areas) > 100:
            ax_hist.set_xscale('log')
        hist_path = os.path.join(primary_chrom_dir, 'peptides_total_area_histogram.png')
        fig_hist.tight_layout()
        fig_hist.savefig(hist_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig_hist)
        print(f"  Peptides total_area histogram saved to {os.path.abspath(hist_path)}")

    # Filtration funnel schematic: how many passed each step (initial, stages 1–6, backfill, final)
    funnel_counts = {
        'input': funnel_n_input,
        'after_total_area': len(all_accepted_for_rt_windows),
        'stage1': len(peptides_stage1),
        'stage2': len(peptides_stage2),
        'stage3': len(peptides_stage3),
        'stage4': len(peptides_stage4),
        'stage5': len(peptides_stage5),
        'stage6': len(kept),
        'backfill_added': len(all_backfill),
        'final': len(peptides_for_rt_grid),
    }
    funnel_path = os.path.join(rt_windows_dir, f'{output_base}_filtering_funnel.png')
    try:
        _draw_filtration_funnel(funnel_counts, funnel_path, protein_id or '')
        print(f"  Filtration funnel schematic saved to {os.path.abspath(funnel_path)}")
    except Exception as e:
        print(f"  Could not save filtration funnel: {e}", file=sys.stderr)

    print(f"RT windows plots and channel CSVs saved in {os.path.abspath(rt_windows_dir)}")
    print(f"  - {rt_windows_before} (before thinning)")
    if peptides_stage4:
        print(f"  - {rt_windows_after_envelope} (after envelope filter)")
    print(f"  - {rt_windows_after_backfill_file} (after backfill)")
    print(f"  - {rt_windows_file} (copy of after backfill)")
    print(f"  - (all zoom segments for each)")
    print(f"  - {by_peptide_file}")
    print(f"  - {rt_windows_by_channel_file} (by channel, no overlap within channel)")
    print(f"  - {channels_csv}")
    print(f"  - {channel_dataframe_csv} (N-channel dataframe: by channel, drift_min)")
    print(f"  - {three_channel_csv} (three_channel_selected_peptide_rt_windows)")
    print(f"  - {os.path.join(rt_windows_dir, 'filtering_progress')} (thinning and stage filtering plots)")
    print(f"  - {os.path.join(output_dir, 'peptide_chromatograms_primary')} (peptide chromatograms, total_area histogram, peak drift lines, unique_peptides + segments)")
    chrom_copy_dir = os.path.join(output_dir, f'{output_base}_chromatograms')
    if os.path.isdir(chrom_copy_dir):
        print(f"  - Filtered chromatograms: {os.path.abspath(chrom_copy_dir)}")


# Default mzML/raw and fasta (same as run_visualization.py) when run standalone without --mzml
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MZML = os.path.join(_SCRIPT_DIR, 'data', 'WT_nep2_0MUrea_08.mzML')
DEFAULT_FASTA = os.path.join(_SCRIPT_DIR, 'data', 'Ube2D3.fasta')


def main():
    parser = argparse.ArgumentParser(description='Regenerate RT integration windows plots from a CSV')
    parser.add_argument('csv_path', nargs='?', default=None, help='Input CSV with peak windows (e.g. path/to/filtered_peptides.csv)')
    parser.add_argument('output_dir', nargs='?', default=None, help='Output directory for plots and CSVs')
    parser.add_argument('--exclude-mods', action='store_true', default=False,
                        help='Exclude modified peptides entirely (default: False = main set unmodified only, modified allowed in backfill only)')
    parser.add_argument('--mzml', metavar='PATH', default=None,
                        help=f'mzML/raw file for overlay and full chromatogram run (default: {os.path.basename(DEFAULT_MZML)} if present)')
    parser.add_argument('--chromatogram-traces', metavar='NPZ', default=None,
                        help='chromatogram_traces.npz from extract step; overlay will show real peaks if peptide order matches or --chromatogram-trace-csv given')
    parser.add_argument('--chromatogram-trace-csv', metavar='CSV', default=None,
                        help='CSV used to build the npz (for matching peptides to trace_0, trace_1, ...)')
    parser.add_argument('--run-max-rt-sec', metavar='SEC', type=float, default=None,
                        help='End of run RT in seconds (e.g. 1200 for 20 min). Used for end-gap backfill so peptides after the last window can fill to run end. If omitted, end gap uses max RT among peptides only.')
    parser.add_argument('--fasta', metavar='PATH', default=None,
                        help=f'FASTA file for combined/unique_peptides plots (default: {os.path.basename(DEFAULT_FASTA)} if present)')
    parser.add_argument('--protein', metavar='ID', default=None,
                        help='Protein ID for combined/unique_peptides plots')
    parser.add_argument('--no-re-extract-windows', action='store_true', dest='no_re_extract_windows',
                        help='Do not re-extract integration/collection/peak-drift windows from mzML; use existing peak_windows if present (default: re-extract before filtering)')
    parser.add_argument('--re-extract-chromatograms', action='store_true', dest='re_extract_chromatograms',
                        help='Force full chromatogram extraction for primary set even when PNGs already exist (default: skip re-extraction and use existing PNGs from filtered_chromatograms)')
    parser.add_argument('--comet-csv', metavar='PATH', default=None,
                        help='Full Comet search CSV for re-extraction when input CSV lacks MS1_retention_time_sec/mz (peptides to process still come from csv_path)')
    args = parser.parse_args()
    # Use default mzML/raw when not provided (same as run_visualization.py defaults)
    if getattr(args, 'mzml', None) is None and getattr(args, 'raw_file', None) is None and os.path.isfile(DEFAULT_MZML):
        args.mzml = DEFAULT_MZML
    if getattr(args, 'raw_file', None) is None and getattr(args, 'mzml', None) is not None:
        args.raw_file = args.mzml
    if getattr(args, 'fasta', None) is None and os.path.isfile(DEFAULT_FASTA):
        args.fasta = DEFAULT_FASTA
    run_rt_windows(args)


if __name__ == '__main__':
    main()
