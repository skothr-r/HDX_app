#!/usr/bin/env python3
"""
Unified visualization module: run any combination of plots from one entry point.

Plot types (select one or more with flags):
  --chromatograms   MS1 chromatograms per peptide + zoom overlays (extract_ms1_chromatograms)
  --rt-windows      RT integration windows plots from filtered CSV (regenerate_rt_windows_from_csv)
  --combined        Combined overhang: coverage heatmap, single-AA histogram, RT grid, unique sequences,
                    RT windows, total_area vs position; plus peptide statistic scatter plots (good/ok/poor)

Output: versioned directories per run:
  - dataframes_vN_<suffix>/  all output CSVs (e.g. input CSV + chromatogram metrics)
  - plots_vN_<suffix>/       all output plots (chromatograms, RT windows, combined)

Default input files (override with --csv, --mzml, --fasta):
  CSV:   data/WT_nep2_0MUrea_08_with_ms1_perc_qvalues.csv  (full PSM set with q-values; use a CSV with "matched fragment ion mz" for Comet fragment labels if available)
  mzML:  data/WT_nep2_0MUrea_08.mzML
  FASTA: data/Ube2D3.fasta
"""

import argparse
import os
import re
import sys
import textwrap

# Allow importing from same directory
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# Default input files (project-relative; override with --csv, --mzml, --fasta)
# Default CSV keeps all rows (from add_ms1_data_openms.py without --filter). Use --csv to point at a filtered CSV if desired.
DEFAULT_CSV = os.path.join(_SCRIPT_DIR, 'data', 'WT_nep2_0MUrea_08_with_qvalues_ms1_all.csv')
DEFAULT_MZML = os.path.join(_SCRIPT_DIR, 'data', 'WT_nep2_0MUrea_08.mzML')
DEFAULT_FASTA = os.path.join(_SCRIPT_DIR, 'data', 'Ube2D3.fasta')

# =============================================================================
# FILTERING CRITERIA — edit these values; they are printed when you run the script
# =============================================================================
# Single filtered peptide CSV (q-value, PEP, mods, min single-AA overhangs); used as filter for chromatograms and RT windows
FILTERED_PEPTIDES_CSV_BASENAME = 'comet_frags_filtered_by_qvalue_pepscore_mods_minaa.csv'
# Temporarily disable filters: set to True to keep all rows (no q/PEP/overhang cut). Alternative: run with --no-default-filter.
DISABLE_FILTERED_PEPTIDES_FILTERS = False

# CSV / default filter (when --filter-csv is not provided)
FILTER_QVALUE_MAX = 0.05             # percolator q-value: keep rows with q-value < this
FILTER_PEP_MAX = 0.05                # percolator PEP: keep rows with PEP < this (when column present)
FILTER_MIN_SCANS_CSV = 1             # minimum scans per peptide for CSV filter (no extra filter if 1)

# What we filter on: min scans, observed m/z vs theoretical, reject mods, M0>M+1 or M+1>M+2, total_area and apex peak intensity
FILTER_MIN_SCANS = 1                  # keep only peptides with >= this many scans (observed m/z within PPM of theoretical); 1 scan passing PSM + isotopic requirements is enough
FILTER_PPM_TOLERANCE = 5.00          # observed m/z within this many ppm of theoretical m/z (ppm = parts per million; 5 ppm at m/z 500 = ±0.0025 Da)
FILTER_REJECT_MODS = False           # keep modified peptides for backfill; do not reject at start
FILTER_MIN_SINGLE_AA_OVERHANGS = 1   # require >= this many single-AA overhangs per row (0 = disable; matches regenerate_rt_windows_from_csv Stage 1 when >= 1)
# Two-stage "significant" fragments (avoids mismatch between Comet-predicted and what we deem significant):
#   Stage 1 (initial CSV): 5 ppm only — overwrite matched fragment ions/pairs/overhangs to fragments within FILTER_SIGNIFICANT_FRAGMENT_PPM of theoretical (no intensity filter; no MS2 at this step).
#   Stage 2 (chromatogram extraction): 5 ppm + % max MS2 intensity — significant = within 5 ppm AND above SIGNIFICANT_FRAGMENT_MIN_INTENSITY_FRAC of max MS2 in window (see visualization/chromatograms.py). Red squares, rejection "no significant overhang", and output significant_fragment_* columns use Stage 2 when MS2 is available.
FILTER_SIGNIFICANT_FRAGMENT_PPM = 5.0
FILTER_REQUIRE_M0_GT_M1_GT_M2 = True  # require M0 > M+1 or M+1 > M+2 on summed MS1 (peak window data from chromatogram step)
FILTER_MIN_APEX_INTENSITY = 1e5       # reject peaks with apex below this (noise); pretty much never select below 10^5 (chromatogram step)
FILTER_TOTAL_AREA_APEX_PERCENTILE = 0    # discard below this percentile (0 = do not discard by total_area/apex)
FILTER_PROTECT_TOTAL_AREA_PERCENTILE = 100  # protect above this percentile (100 = do not protect any)

# Test mode (--test)
FILTER_TEST_SAMPLE_SIZE = 30          # when --test: randomly sample this many peptides for all plots

# How RT integration windows are determined (informational; algorithm in visualization/chromatograms.py)
FILTER_WINDOWS_HOW_MESSAGE = (
    "RT integration windows: For each peptide, the MS1 chromatogram is extracted around the precursor RT. "
    "The peak apex (max intensity) is found in a search window; boundaries are set by walking left/right from "
    "the apex until intensity falls below max(1% of apex, 3× noise estimate) or a local minimum (valley). "
    "That (min_rt, max_rt) is the integration window used for total_area, apex_intensity, and the RT grid."
)

# Legacy names used elsewhere in this script (do not edit; they mirror the values above)
DEFAULT_QVALUE_MAX = FILTER_QVALUE_MAX
DEFAULT_MIN_SCANS = FILTER_MIN_SCANS_CSV
QVALUE_COLUMNS = ['percolator_qvalue', 'q-value', 'qvalue', 'percolator_q-value', 'FDR', 'fdr']
PEP_COLUMNS = ['percolator_PEP', 'PEP', 'pep']


def print_filtering_criteria():
    """Print all filtering criteria in use (so edits above are visible when you run the script)."""
    print("\n" + "=" * 70)
    print(" FILTERING CRITERIA (current values — edit the constants at top of run_visualization.py)")
    print("=" * 70)
    print("  CSV / default filter:")
    print(f"    FILTER_QVALUE_MAX              = {FILTER_QVALUE_MAX}")
    print(f"    FILTER_PEP_MAX                 = {FILTER_PEP_MAX}")
    print(f"    FILTER_MIN_SCANS_CSV           = {FILTER_MIN_SCANS_CSV}")
    print("  Min scans and observed m/z vs theoretical:")
    print(f"    FILTER_MIN_SCANS               = {FILTER_MIN_SCANS} (keep peptides with >= this many scans)")
    print(f"    FILTER_PPM_TOLERANCE           = {FILTER_PPM_TOLERANCE:.2f} (observed m/z within this ppm of theoretical)")
    print("  Modifications:")
    print(f"    FILTER_REJECT_MODS             = {FILTER_REJECT_MODS} (reject modified peptides; keep only unmodified)")
    print("  Single-AA overhangs (from input CSV):")
    print(f"    FILTER_MIN_SINGLE_AA_OVERHANGS = {FILTER_MIN_SINGLE_AA_OVERHANGS} (keep only rows with >= this many single-AA overhangs)")
    print("  Significant fragments (when CSV has 'matched fragment ion mz'):")
    print(f"    FILTER_SIGNIFICANT_FRAGMENT_PPM = {FILTER_SIGNIFICANT_FRAGMENT_PPM} (keep only frags within this ppm of expected m/z)")
    print("  Apex intensity (noise cutoff; applied in chromatogram step when apex is assigned):")
    print(f"    FILTER_MIN_APEX_INTENSITY      = {FILTER_MIN_APEX_INTENSITY:.0f} (10^5; reject apex < this)")
    print("  Isotope envelope (uses peak window data from chromatogram step):")
    print(f"    FILTER_REQUIRE_M0_GT_M1_GT_M2  = {FILTER_REQUIRE_M0_GT_M1_GT_M2} (require M0 > M+1 or M+1 > M+2 on summed MS1)")
    print("  Total area and apex peak intensity (uses peak window data from chromatogram step):")
    print(f"    FILTER_TOTAL_AREA_APEX_PERCENTILE  = {FILTER_TOTAL_AREA_APEX_PERCENTILE} (discard below in either total_area or apex intensity)")
    print(f"    FILTER_PROTECT_TOTAL_AREA_PERCENTILE = {FILTER_PROTECT_TOTAL_AREA_PERCENTILE} (protect above in total_area)")
    print("  How RT integration windows are determined:")
    for line in textwrap.wrap(FILTER_WINDOWS_HOW_MESSAGE, width=66):
        print(f"    {line}")
    print("  (Peak window data is generated by the chromatogram step; run --chromatograms or --all so combined uses it.)")
    print("  Test mode (--test):")
    print(f"    FILTER_TEST_SAMPLE_SIZE        = {FILTER_TEST_SAMPLE_SIZE}")
    print("=" * 70 + "\n")


def _next_versioned_dirs(do_chromatograms, do_rt_windows, do_combined, base_dir=None):
    """Return (dataframes_dir, plots_dir) with next version and suffix for this run."""
    parts = []
    if do_chromatograms:
        parts.append('chromatograms')
    if do_rt_windows:
        parts.append('rt-windows')
    if do_combined:
        parts.append('combined')
    suffix = '_'.join(parts) if parts else 'none'

    search_dir = base_dir if base_dir else _SCRIPT_DIR
    max_version = 0
    for name in ('dataframes', 'plots'):
        pattern = re.compile(rf'^{name}_v(\d+)(?:_|$)')
        try:
            for entry in os.listdir(search_dir):
                m = pattern.match(entry)
                if m and os.path.isdir(os.path.join(search_dir, entry)):
                    v = int(m.group(1))
                    if v > max_version:
                        max_version = v
        except OSError:
            pass
    version = max_version + 1

    parent = base_dir if base_dir else _SCRIPT_DIR
    dataframes_dir = os.path.join(parent, f'dataframes_v{version}_{suffix}')
    plots_dir = os.path.join(parent, f'plots_v{version}_{suffix}')
    return dataframes_dir, plots_dir


def _significant_pairs_and_overhangs_from_ions(sig_ions, peptide):
    """
    From a list of significant fragment ion names (c/z/z+1), compute consecutive same-series pairs
    and single-AA overhang positions. Returns (pairs_str, overhang_positions_str).
    peptide: plain sequence (no mods) for length and residue letters.
    """
    import re
    clean = re.sub(r'\[.*?\]', '', peptide) if peptide else ''
    L = len(clean) if clean else 0
    if L == 0 or not sig_ions:
        return '', ''
    c_nums, z_nums, z1_nums = [], [], []
    for name in sig_ions:
        if not name or not isinstance(name, str):
            continue
        name = name.strip()
        if name.startswith('c') and name[1:].isdigit():
            n = int(name[1:])
            if 1 <= n <= L:
                c_nums.append(n)
        elif name.startswith('z1_') and name.split('_')[-1].isdigit():
            n = int(name.split('_')[-1])
            if 1 <= n <= L:
                z1_nums.append(n)
        elif name.startswith('z') and name[1:].isdigit():
            n = int(name[1:])
            if 1 <= n <= L:
                z_nums.append(n)
    c_set, z_set, z1_set = set(c_nums), set(z_nums), set(z1_nums)
    pairs = []
    overhang_pos = set()
    for n in sorted(c_set):
        if (n + 1) in c_set:
            pairs.append(f'c{n}|c{n+1}')
            overhang_pos.add(n + 1)
    for n in sorted(z_set):
        if (n + 1) in z_set:
            pairs.append(f'z{n}|z{n+1}')
            overhang_pos.add(L - n + 1)
    for n in sorted(z1_set):
        if (n + 1) in z1_set:
            pairs.append(f'z{n}+1|z{n+1}+1')
            overhang_pos.add(L - n + 1)
    pairs_str = ','.join(pairs) if pairs else ''
    overhang_parts = []
    for pos in sorted(overhang_pos):
        if 1 <= pos <= L:
            overhang_parts.append(f'{pos}{clean[pos - 1]}')
    overhang_str = ','.join(overhang_parts) if overhang_parts else ''
    return pairs_str, overhang_str


def _filter_fragment_columns_to_significant_ppm(df, ppm=5.0):
    """
    Stage 1 significant-fragment filter (initial CSV only). When CSV has 'matched fragment ions'
    and 'matched fragment ion mz', keep only fragments whose observed m/z is within ppm of
    theoretical (c/z/z+1 from plain_peptide). Overwrites those columns (and 'matched fragment
    ion intensities' if present) in place. Also derives fragment pairs and single-AA overhang
    positions from this 5-ppm-only list. Stage 2 (chromatogram extraction) further refines
    with % max MS2 intensity (5 ppm + intensity) for red squares and output significant_fragment_*.
    """
    import re
    import pandas as pd
    col_ions = 'matched fragment ions'
    col_mz = 'matched fragment ion mz'
    col_int = 'matched fragment ion intensities'
    col_pairs = 'single_aa_overhang_fragment_pairs'
    col_positions = 'single_aa_overhangs_protein_positions'
    if col_ions not in df.columns or col_mz not in df.columns:
        return 0
    try:
        from visualization.chromatograms import comet_matched_ions_to_theoretical_mz
    except Exception:
        return 0
    n_changed = 0
    for idx in df.index:
        row = df.loc[idx]
        peptide = row.get('plain_peptide') or row.get('sequence') or ''
        if not peptide or not isinstance(peptide, str):
            continue
        ions_val = row.get(col_ions, '')
        mz_val = row.get(col_mz, '')
        if pd.isna(ions_val) or pd.isna(mz_val) or not str(ions_val).strip() or not str(mz_val).strip():
            continue
        ions_list = [p.strip() for p in str(ions_val).strip().strip('"').split(',') if p.strip()]
        mz_list = [p.strip() for p in str(mz_val).strip().strip('"').split(',') if p.strip()]
        if len(ions_list) != len(mz_list):
            continue
        has_int = col_int in df.columns and pd.notna(row.get(col_int)) and str(row.get(col_int)).strip()
        int_list = []
        if has_int:
            int_list = [p.strip() for p in str(row[col_int]).strip().strip('"').split(',') if p.strip()]
            if len(int_list) != len(ions_list):
                has_int = False
        sig_ions, sig_mz, sig_int = [], [], []
        for i, (name, mz_s) in enumerate(zip(ions_list, mz_list)):
            if not name or not mz_s:
                continue
            if not (name.startswith('c') and name[1:].isdigit()) and not (name.startswith('z1_') and name.split('_')[-1].isdigit()) and not (name.startswith('z') and name[1:].isdigit()):
                continue
            try:
                obs_mz = float(mz_s)
            except ValueError:
                continue
            if obs_mz <= 0:
                continue
            theo_list = comet_matched_ions_to_theoretical_mz([name], peptide, fragment_charge=1)
            if not theo_list:
                continue
            theo_mz = theo_list[0][0]
            if theo_mz <= 0:
                continue
            ppm_diff = abs(obs_mz - theo_mz) / theo_mz * 1e6
            if ppm_diff <= ppm:
                sig_ions.append(name)
                sig_mz.append(mz_s)
                if has_int and i < len(int_list):
                    sig_int.append(int_list[i])
        # Always overwrite with significant-only (5 ppm): fragment list and derived pairs/overhangs. Never use original Comet values for display—only significant fragments.
        df.at[idx, col_ions] = ','.join(sig_ions) if sig_ions else ''
        df.at[idx, col_mz] = ','.join(sig_mz) if sig_mz else ''
        if has_int and col_int in df.columns:
            df.at[idx, col_int] = ','.join(sig_int) if sig_int else ''
        pairs_str, overhang_str = _significant_pairs_and_overhangs_from_ions(sig_ions, peptide)
        if col_pairs not in df.columns:
            df[col_pairs] = ''
        if col_positions not in df.columns:
            df[col_positions] = ''
        df.at[idx, col_pairs] = pairs_str
        seq_start = row.get('sequence_start_pos') or row.get('sequence_positions')
        if pd.notna(seq_start) and str(seq_start).strip():
            try:
                start = int(float(str(seq_start).strip().split('-')[0].split()[0]))
            except (ValueError, TypeError):
                start = None
        else:
            start = None
        if start is not None and start > 0 and overhang_str:
            clean = re.sub(r'\[.*?\]', '', peptide) if peptide else ''
            parts = []
            for pos_res in overhang_str.split(','):
                pos_res = pos_res.strip()
                if pos_res and len(pos_res) >= 2 and pos_res[-1].isalpha() and pos_res[:-1].isdigit():
                    pos = int(pos_res[:-1])
                    res = pos_res[-1]
                    parts.append(f'{start + pos - 1}{res}')
                else:
                    parts.append(pos_res)
            overhang_str = ','.join(parts) if parts else overhang_str
        df.at[idx, col_positions] = overhang_str
        n_changed += 1
    return n_changed


def _default_filtered_csv(csv_path, dataframes_dir, reject_mods=None):
    """Build the initial filtered CSV: peptides passing Q/PEP, min overhangs; optionally reject mods. Fragment columns (matched fragment ions, pairs, single-AA overhangs) are overwritten with significant-only (5 ppm) when CSV has 'matched fragment ion mz', so the output contains no non-significant fragments or overhangs. Envelope (M0>M+1 or M+1>M+2) is applied after chromatogram extraction. Saves to dataframes_dir. Returns path."""

    import pandas as pd
    if not csv_path or not os.path.exists(csv_path):
        return None
    if reject_mods is None:
        reject_mods = FILTER_REJECT_MODS
    try:
        with open(csv_path, 'r') as f:
            first = f.readline()
        skip = 1 if 'CometVersion' in first else 0
    except Exception:
        skip = 0
    df = pd.read_csv(csv_path, sep=',', skiprows=skip, engine='python', quotechar='"', on_bad_lines='warn')
    # Validate input CSV so we don't build filters from bad data
    try:
        from visualization.csv_validation import validate_and_log
        validate_and_log(
            df,
            source_name=os.path.basename(csv_path),
            required_columns=('plain_peptide', 'charge'),
            validate_identifiers=True,
            validate_numeric=True,
        )
    except Exception as e:
        print(f"[run_visualization] Validation warning: {e}")
        try:
            from visualization.csv_validation import record_run_warning
            record_run_warning(f"Input CSV validation: {e}")
        except Exception:
            pass
    qcol = None
    for c in QVALUE_COLUMNS:
        if c in df.columns:
            qcol = c
            break
    if qcol is None:
        print("[run_visualization] No q-value column found; using full CSV as filter (no default filter applied).")
        out_path = os.path.join(dataframes_dir, FILTERED_PEPTIDES_CSV_BASENAME)
        df.to_csv(out_path, index=False)
        return out_path
    # Keep rows where q < FILTER_QVALUE_MAX OR PEP < FILTER_PEP_MAX (match regenerate_rt_windows_from_csv Stage 3)
    # If DISABLE_FILTERED_PEPTIDES_FILTERS: keep all rows (no q/PEP filter).
    def _q_ok(ser, col, thresh):
        try:
            return ser[col].notna() & (ser[col].astype(float) < thresh)
        except (TypeError, ValueError):
            return ser[col].notna()
    if DISABLE_FILTERED_PEPTIDES_FILTERS:
        mask = pd.Series(True, index=df.index)
        df_filtered = df.loc[mask].copy()
    else:
        q_pass = _q_ok(df, qcol, FILTER_QVALUE_MAX)
        pepcol = None
        for c in PEP_COLUMNS:
            if c in df.columns:
                pepcol = c
                break
        if pepcol is not None:
            pep_pass = _q_ok(df, pepcol, FILTER_PEP_MAX)
            # Keep if: (q passes) OR (pep passes) OR (q missing) OR (pep missing)
            mask = q_pass | pep_pass | df[qcol].isna() | df[pepcol].isna()
        else:
            # No PEP column: match regenerate (missing => do not filter), so keep all
            mask = pd.Series(True, index=df.index)
        df_filtered = df.loc[mask].copy()
    if reject_mods and 'modifications' in df_filtered.columns:
        def _has_mod(m):
            if pd.isna(m):
                return False
            s = str(m).strip().lower()
            return s not in ('', '-', 'nan', 'none')
        before = len(df_filtered)
        df_filtered = df_filtered.loc[~df_filtered['modifications'].apply(_has_mod)].copy()
        if before > len(df_filtered):
            print(f"[run_visualization] Reject mods: {before} -> {len(df_filtered)} rows (unmodified only)")
    # Overwrite fragment columns with significant-only (5 ppm). Output CSV will not contain non-significant fragments or overhangs when mz is present.
    if FILTER_SIGNIFICANT_FRAGMENT_PPM > 0 and 'matched fragment ions' in df_filtered.columns and 'matched fragment ion mz' in df_filtered.columns:
        n_rows_updated = _filter_fragment_columns_to_significant_ppm(df_filtered, ppm=FILTER_SIGNIFICANT_FRAGMENT_PPM)
        if n_rows_updated:
            print(f"[run_visualization] Significant fragments only (within {FILTER_SIGNIFICANT_FRAGMENT_PPM} ppm): fragment columns and derived pairs/overhangs set to significant-only in {n_rows_updated} rows")
    elif FILTER_SIGNIFICANT_FRAGMENT_PPM > 0 and 'matched fragment ions' in df_filtered.columns:
        print("[run_visualization] Significant-only fragments skipped (no 'matched fragment ion mz' column); output may contain all Comet fragments. Add mz column to restrict to 5 ppm.")
    # Require minimum single-AA overhangs per row (from CSV; now significant-only when mz was present).
    # Use max of counts from positions and fragment_pairs so we don't drop all rows when 5 ppm leaves positions empty but pairs present.
    has_overhang_cols = 'single_aa_overhangs_protein_positions' in df_filtered.columns or 'single_aa_overhang_fragment_pairs' in df_filtered.columns
    effective_min_overhangs = 0 if DISABLE_FILTERED_PEPTIDES_FILTERS else FILTER_MIN_SINGLE_AA_OVERHANGS
    if effective_min_overhangs > 0 and has_overhang_cols:
        def _count_overhangs(s, col_name):
            if pd.isna(s) or s is None:
                return 0
            s = str(s).strip()
            if not s:
                return 0
            if col_name == 'single_aa_overhangs_protein_positions':
                parts = [x.strip() for x in s.replace(';', ',').split(',') if x.strip()]
                return len(parts)
            if col_name == 'single_aa_overhang_fragment_pairs':
                parts = [x.strip() for x in s.replace(';', ',').split(',') if x.strip() and '|' in x]
                return len(parts)
            return len([x for x in s.split(',') if str(x).strip()])
        count_positions = df_filtered['single_aa_overhangs_protein_positions'].apply(lambda r: _count_overhangs(r, 'single_aa_overhangs_protein_positions')) if 'single_aa_overhangs_protein_positions' in df_filtered.columns else pd.Series(0, index=df_filtered.index)
        count_pairs = df_filtered['single_aa_overhang_fragment_pairs'].apply(lambda r: _count_overhangs(r, 'single_aa_overhang_fragment_pairs')) if 'single_aa_overhang_fragment_pairs' in df_filtered.columns else pd.Series(0, index=df_filtered.index)
        n_overhangs = count_positions.combine(count_pairs, max)
        before = len(df_filtered)
        would_keep = (n_overhangs >= effective_min_overhangs).sum()
        if would_keep == 0:
            print(f"[run_visualization] Min single-AA overhangs >= {effective_min_overhangs} would leave 0 rows; skipping overhang filter so chromatogram step can run (it will enforce >= 1 significant overhang from MS2).")
        else:
            df_filtered = df_filtered.loc[n_overhangs >= effective_min_overhangs].copy()
            if before > len(df_filtered):
                print(f"[run_visualization] Min single-AA overhangs >= {effective_min_overhangs}: {before} -> {len(df_filtered)} rows (from positions and/or fragment_pairs)")
    out_path = os.path.join(dataframes_dir, FILTERED_PEPTIDES_CSV_BASENAME)
    os.makedirs(dataframes_dir, exist_ok=True)
    # Validate before writing so we don't propagate bad data
    try:
        from visualization.csv_validation import validate_and_log
        validate_and_log(
            df_filtered,
            source_name=os.path.basename(out_path),
            required_columns=('plain_peptide', 'charge'),
            validate_identifiers=True,
            validate_numeric=True,
        )
    except Exception as e:
        print(f"[run_visualization] Validation warning: {e}")
        try:
            from visualization.csv_validation import record_run_warning
            record_run_warning(f"Filtered CSV validation: {e}")
        except Exception:
            pass
    df_filtered.to_csv(out_path, index=False)
    # Unique peptides = unique (plain_peptide, charge, mods) — this is the count used by chromatograms
    if 'plain_peptide' in df_filtered.columns and 'charge' in df_filtered.columns:
        mods_col = 'modifications' if 'modifications' in df_filtered.columns else None
        if mods_col:
            n_unique = df_filtered.groupby(['plain_peptide', 'charge', mods_col]).ngroups
        else:
            n_unique = df_filtered.groupby(['plain_peptide', 'charge']).ngroups
        if DISABLE_FILTERED_PEPTIDES_FILTERS:
            print(f"[run_visualization] Filters disabled (DISABLE_FILTERED_PEPTIDES_FILTERS=True): {len(df_filtered)} rows, {n_unique} unique peptides -> {out_path}")
        else:
            print(f"[run_visualization] Default filter: q < {FILTER_QVALUE_MAX} or PEP < {FILTER_PEP_MAX}, min overhangs >= {FILTER_MIN_SINGLE_AA_OVERHANGS} -> {len(df_filtered)} rows, {n_unique} unique peptides -> {out_path}")
    else:
        if DISABLE_FILTERED_PEPTIDES_FILTERS:
            print(f"[run_visualization] Filters disabled (DISABLE_FILTERED_PEPTIDES_FILTERS=True): {len(df_filtered)} rows -> {out_path}")
        else:
            print(f"[run_visualization] Default filter: q < {FILTER_QVALUE_MAX} or PEP < {FILTER_PEP_MAX} -> {len(df_filtered)} rows -> {out_path}")
    return out_path


def _parse_args():
    p = argparse.ArgumentParser(
        description='Run visualization plots: chromatograms, RT windows, and/or combined overhang visualization'
    )
    # Plot selection
    p.add_argument('--chromatograms', action='store_true',
                   help='Run MS1 chromatogram extraction and overlay plots')
    p.add_argument('--rt-windows', action='store_true',
                   help='Regenerate RT integration windows from filtered CSV')
    p.add_argument('--combined', action='store_true',
                   help='Run combined overhang visualization (heatmap, scatter, histogram, RT grid)')
    p.add_argument('--all', action='store_true',
                   help='Run all three plot types (equivalent to --chromatograms --rt-windows --combined)')

    # Common (defaults from module constants)
    p.add_argument('--csv', default=DEFAULT_CSV,
                   help=f'Comet CSV (default: {os.path.basename(DEFAULT_CSV)})')
    p.add_argument('--mzml', default=DEFAULT_MZML,
                   help=f'mzML file (default: {os.path.basename(DEFAULT_MZML)})')
    p.add_argument('--fasta', default=DEFAULT_FASTA,
                   help=f'FASTA file (default: {os.path.basename(DEFAULT_FASTA)})')
    p.add_argument('--output-dir', default=None,
                   help='Override versioned output base (default: auto dataframes_vN_<suffix> and plots_vN_<suffix>)')
    p.add_argument('--filter-csv',
                   help='Filtered peptides CSV (optional; if not set, default filter q-value < %.2f, PEP < %.2f is applied)' % (FILTER_QVALUE_MAX, FILTER_PEP_MAX))
    p.add_argument('--no-default-filter', action='store_true',
                   help='Do not apply default q-value/PEP/overhang filter; use full CSV as peptide list')
    p.add_argument('--no-filter', action='store_true',
                   help='Same as --no-default-filter. Use with --csv <full_row_csv> (e.g. ..._with_qvalues_ms1_all.csv) to include all peptides.')
    p.add_argument('--exclude-mods', action='store_true',
                   help='Exclude modified peptides (overrides FILTER_REJECT_MODS if False)')
    p.add_argument('--no-exclude-mods', action='store_true',
                   help='Do not exclude modified peptides (overrides FILTER_REJECT_MODS when True)')
    p.add_argument('--protein', help='Protein ID (optional, for --combined)')

    # Chromatogram-specific
    p.add_argument('--test', action='store_true', help='Test mode: use only 30 random peptides for all plots (combined, chromatograms, RT windows)')
    p.add_argument('--extract-only', action='store_true',
                   help='Chromatograms: only extract and save chromatogram_metrics_all.csv + chromatogram_traces.npz (no plots)')
    p.add_argument('--plot-only', action='store_true',
                   help='Chromatograms: only create plots from existing chromatogram_metrics_all.csv and chromatogram_traces.npz')
    p.add_argument('--chromatogram-metrics-csv', default=None,
                   help='Path to chromatogram_metrics_all.csv (for --plot-only; default: dataframes_dir/chromatogram_metrics_all.csv)')
    p.add_argument('--chromatogram-traces', default=None,
                   help='Path to chromatogram_traces.npz (for --plot-only; default: dataframes_dir/chromatogram_traces.npz)')

    # Combined-specific
    p.add_argument('--c-ions-only', action='store_true', help='Combined: c ions only')
    p.add_argument('--z-ions-only', action='store_true', help='Combined: z ions only')
    p.add_argument('--no-filtering', action='store_true', help='Combined: skip filtering')
    p.add_argument('--two-pass', action='store_true', help='Combined: use two-pass selection')
    p.add_argument('--no-progressive', action='store_true', help='Combined: disable progressive selection')
    p.add_argument('--no-two-pass', action='store_true', help='Combined: disable two-pass')
    return p.parse_args()


def _run_chromatograms(args, plots_dir, dataframes_dir, accepted_plots_dir, rejected_plots_dir, accepted_df_dir, rejected_df_dir, force_full_run=False, chrom_filter_csv=None):
    """Run chromatogram extraction and individual peptide plots. If force_full_run=True (e.g. when running for RT windows/combined), always generate individual PNGs, not extract-only.
    chrom_filter_csv: when provided, use this filter for chromatograms (full CSV q-value filter) so one plot per peptide; RT/combined still use args.filter_csv."""
    print("[run_visualization] Starting chromatogram extraction (loading CSV and mzML may take a moment)...")
    sys.stdout.flush()
    from visualization import run_chromatograms
    extract_only = False if force_full_run else getattr(args, 'extract_only', False)
    plot_only = getattr(args, 'plot_only', False)

    if plot_only:
        metrics_csv = getattr(args, 'chromatogram_metrics_csv', None) or (os.path.join(dataframes_dir, 'chromatogram_metrics_all.csv') if dataframes_dir else None)
        traces_path = getattr(args, 'chromatogram_traces', None) or (os.path.join(dataframes_dir, 'chromatogram_traces.npz') if dataframes_dir else None)
        if not metrics_csv or not os.path.exists(metrics_csv):
            print(f"Error: --plot-only requires chromatogram_metrics_all.csv (got: {metrics_csv})")
            return False
        if not traces_path or not os.path.exists(traces_path):
            print(f"Error: --plot-only requires chromatogram_traces.npz (got: {traces_path})")
            return False
        exclude_mods = (FILTER_REJECT_MODS or args.exclude_mods) and not getattr(args, 'no_exclude_mods', False)
        no_psm_filter = getattr(args, 'no_default_filter', False)
        ns = argparse.Namespace(
            chromatogram_metrics_csv=metrics_csv,
            chromatogram_traces=traces_path,
            exclude_mods=exclude_mods,
            output_accepted_dir=os.path.join(accepted_plots_dir, 'chromatograms_peptides'),
            output_rejected_dir=os.path.join(rejected_plots_dir, 'chromatograms_peptides_rejected'),
            dataframes_dir=dataframes_dir,
            no_psm_filter=no_psm_filter,
        )
        run_chromatograms(ns)
        return True

    csv_path = args.csv
    if not csv_path or not os.path.exists(csv_path):
        print("Error: --chromatograms requires --csv with an existing file")
        return False
    mzml = args.mzml
    if not mzml or not os.path.exists(mzml):
        print("Error: --chromatograms requires --mzml with an existing file")
        return False
    base = os.path.splitext(os.path.basename(csv_path))[0]
    output_png = os.path.join(plots_dir, f'{base}_chromatograms.png')
    exclude_mods = (FILTER_REJECT_MODS or args.exclude_mods) and not getattr(args, 'no_exclude_mods', False)
    extract_output_dir = dataframes_dir  # extract-only writes metrics + traces here
    # Use chrom_filter_csv (default filter from full CSV) so we get one plot per peptide; avoids only-one-plot when args.filter_csv is small
    filter_for_chrom = chrom_filter_csv if chrom_filter_csv and os.path.exists(chrom_filter_csv) else (args.filter_csv or None)
    no_psm_filter = getattr(args, 'no_default_filter', False)
    ns = argparse.Namespace(
        comet_csv=csv_path,
        raw_file=mzml,
        output_png=output_png,
        filter_csv=filter_for_chrom,
        exclude_mods=exclude_mods,
        test=args.test,
        extract_only=extract_only,
        extract_output_dir=extract_output_dir if extract_only else None,
        dataframes_dir=dataframes_dir,
        output_accepted_dir=os.path.join(accepted_plots_dir, 'chromatograms_peptides'),
        output_rejected_dir=os.path.join(rejected_plots_dir, 'chromatograms_peptides_rejected'),
        dataframes_accepted_dir=accepted_df_dir,
        dataframes_rejected_dir=rejected_df_dir,
        no_psm_filter=no_psm_filter,
    )
    run_chromatograms(ns)
    # Peptide statistic scatter plots (good/ok/poor) from chromatograms accepted CSV when available
    if accepted_df_dir and os.path.isdir(accepted_df_dir):
        csv_for_stats = None
        for f in os.listdir(accepted_df_dir):
            if '_chromatograms_accepted.csv' in f or f == FILTERED_PEPTIDES_CSV_BASENAME:
                csv_for_stats = os.path.join(accepted_df_dir, f)
                break
        if csv_for_stats and os.path.isfile(csv_for_stats):
            from combined_overhang_visualization import create_peptide_statistic_scatter_plots
            stats_plot_dir = os.path.join(accepted_plots_dir, 'peptide_statistics_scatter')
            os.makedirs(stats_plot_dir, exist_ok=True)
            print(f"[run_visualization] Peptide statistic scatter plots (good/ok/poor) -> {stats_plot_dir}")
            create_peptide_statistic_scatter_plots(csv_for_stats, stats_plot_dir, protein_id=getattr(args, 'protein', '') or '')
    return True


def _run_rt_windows(args, plots_dir, dataframes_dir, accepted_df_dir):
    from regenerate_rt_windows_from_csv import run_rt_windows
    filter_csv = getattr(args, 'filter_csv', None)
    if not filter_csv:
        filter_csv = os.path.join(plots_dir, FILTERED_PEPTIDES_CSV_BASENAME)
    if not os.path.exists(filter_csv):
        # Try dataframes dir for a chromatogram-metrics accepted CSV to use as filter
        candidate = None
        if dataframes_dir and os.path.isdir(dataframes_dir):
            for d in [accepted_df_dir, dataframes_dir]:
                if not d or not os.path.isdir(d):
                    continue
                try:
                    for f in os.listdir(d):
                        if 'chromatograms_accepted' in f or f == FILTERED_PEPTIDES_CSV_BASENAME or f.endswith('_with_chromatogram_metrics_accepted.csv'):
                            p = os.path.join(d, f)
                            if os.path.isfile(p):
                                candidate = p
                                break
                except OSError:
                    pass
                if candidate:
                    break
        if candidate:
            filter_csv = candidate
        if not os.path.exists(filter_csv):
            print(f"Error: --rt-windows requires --filter-csv or filtered/chromatograms_accepted CSV; not found")
            return False
    exclude_mods = (FILTER_REJECT_MODS or args.exclude_mods) and not getattr(args, 'no_exclude_mods', False)
    ns = argparse.Namespace(
        csv_path=filter_csv,
        output_dir=plots_dir,
        exclude_mods=exclude_mods,
        output_base='filtered',
        mzml=getattr(args, 'mzml', None),
        raw_file=getattr(args, 'mzml', None),
        chromatogram_traces=getattr(args, 'chromatogram_traces', None) or (os.path.join(dataframes_dir, 'chromatogram_traces.npz') if dataframes_dir and os.path.isdir(dataframes_dir) else None),
        chromatogram_trace_csv=None,
        dataframes_dir=dataframes_dir,
        fasta=getattr(args, 'fasta', None),
        protein=getattr(args, 'protein', None),
        original_csv=getattr(args, 'csv', None),
    )
    if ns.chromatogram_trace_csv is None and dataframes_dir and os.path.isdir(dataframes_dir):
        for name in ['chromatogram_metrics_all.csv', 'chromatogram_metrics_accepted.csv']:
            p = os.path.join(dataframes_dir, name)
            if os.path.isfile(p):
                ns.chromatogram_trace_csv = p
                break
    run_rt_windows(ns)
    # CSV output for this step: filtered dataset used for RT windows (accepted only) -> dataframes/accepted/
    if accepted_df_dir and os.path.isdir(accepted_df_dir) and filter_csv and os.path.exists(filter_csv):
        import shutil
        out_rt = os.path.join(accepted_df_dir, 'filtered_peptides_rt_windows.csv')
        shutil.copy2(filter_csv, out_rt)
        print(f"[run_visualization] RT-windows step CSV output: {os.path.abspath(out_rt)}")
    return True


def _run_combined(args, plots_dir, dataframes_dir, accepted_df_dir):
    from visualization import create_combined_visualization, parse_fasta
    csv_path = args.csv
    fasta_path = args.fasta
    if not csv_path or not os.path.exists(csv_path):
        print("Error: --combined requires --csv with an existing file")
        return False
    if not fasta_path or not os.path.exists(fasta_path):
        print("Error: --combined requires --fasta with an existing file")
        return False
    sequences = parse_fasta(fasta_path)
    if not sequences:
        print("Error: No sequences found in FASTA")
        return False
    protein_id = args.protein or list(sequences.keys())[0]
    if args.protein is None:
        print(f"No protein specified, using: {protein_id}")
    output_file = os.path.join(plots_dir, 'combined_overhang_visualization.png')
    ion_type_filter = None
    if args.c_ions_only:
        ion_type_filter = 'c'
    elif args.z_ions_only:
        ion_type_filter = 'z'
    if args.c_ions_only and args.z_ions_only:
        print("Error: --c-ions-only and --z-ions-only are mutually exclusive")
        return False
    # Build filter_criteria from FILTERING CRITERIA constants (single source of truth)
    filter_criteria = {
        'min_scans_combined': FILTER_MIN_SCANS,
        'min_single_aa_overhangs': FILTER_MIN_SINGLE_AA_OVERHANGS,
        'ppm_tolerance': FILTER_PPM_TOLERANCE,
        'reject_mods': FILTER_REJECT_MODS,
        'require_m0_gt_m1_gt_m2': FILTER_REQUIRE_M0_GT_M1_GT_M2,
        'total_area_apex_percentile': FILTER_TOTAL_AREA_APEX_PERCENTILE,
        'protect_total_area_percentile': FILTER_PROTECT_TOTAL_AREA_PERCENTILE,
        'test_sample_size': FILTER_TEST_SAMPLE_SIZE,
    }
    # Use peak window data generated by the chromatogram step in this pipeline (run --chromatograms or --all first)
    peak_windows_csv = None
    for name in ['peak_windows.csv', 'peak_windows_all.csv']:
        if dataframes_dir and os.path.isdir(dataframes_dir):
            p = os.path.join(dataframes_dir, name)
            if os.path.isfile(p):
                peak_windows_csv = p
                break
    if peak_windows_csv:
        print(f"[run_visualization] Using peak window data from chromatogram step: {os.path.basename(peak_windows_csv)}")
    # Force combined visualization to write into plots_dir
    import combined_overhang_visualization as cov
    cov.create_combined_visualization._results_dir = plots_dir
    test_mode = getattr(args, 'test', False)
    create_combined_visualization(
        csv_path, protein_id, fasta_path, output_file,
        ion_type_filter=ion_type_filter,
        no_filtering=args.no_filtering,
        use_two_pass=args.two_pass,
        no_progressive=args.no_progressive,
        no_two_pass=args.no_two_pass,
        mzml_file=args.mzml,
        peak_windows_csv=peak_windows_csv,
        test=test_mode,
        filter_criteria=filter_criteria,
    )
    if ion_type_filter is None:
        base = output_file[:-4] if output_file.endswith('.png') else output_file
        for label, filt in [('_c_ions', 'c'), ('_z_ions', 'z')]:
            out_ion = base + label + '.png'
            create_combined_visualization(
                csv_path, protein_id, fasta_path, out_ion,
                ion_type_filter=filt,
                no_filtering=args.no_filtering,
                use_two_pass=args.two_pass,
                no_progressive=args.no_progressive,
                no_two_pass=args.no_two_pass,
                mzml_file=args.mzml,
                peak_windows_csv=peak_windows_csv,
                test=test_mode,
                filter_criteria=filter_criteria,
            )
    # CSV output for this step: use same canonical filter CSV as RT-windows so all output CSVs are consistent
    import shutil
    canonical_filter = getattr(args, 'filter_csv', None)
    combined_filtered_plots = os.path.join(plots_dir, FILTERED_PEPTIDES_CSV_BASENAME)
    if accepted_df_dir and os.path.isdir(accepted_df_dir):
        if canonical_filter and os.path.isfile(canonical_filter):
            out_combined = os.path.join(accepted_df_dir, 'filtered_peptides_combined.csv')
            shutil.copy2(canonical_filter, out_combined)
            print(f"[run_visualization] Combined step CSV output (same as RT-windows source): {os.path.abspath(out_combined)}")
        elif os.path.isfile(combined_filtered_plots):
            out_combined = os.path.join(accepted_df_dir, 'filtered_peptides_combined.csv')
            shutil.copy2(combined_filtered_plots, out_combined)
            print(f"[run_visualization] Combined step CSV output: {os.path.abspath(out_combined)}")
    # Peptide statistic scatter plots (RT and sequence position vs q-value, total_area, etc.) with good/ok/poor
    csv_for_stats = None
    if accepted_df_dir and os.path.isdir(accepted_df_dir):
        for name in ['filtered_peptides_combined.csv', 'filtered_peptides_rt_windows.csv']:
            p = os.path.join(accepted_df_dir, name)
            if os.path.isfile(p):
                csv_for_stats = p
                break
    if not csv_for_stats and os.path.isfile(combined_filtered_plots):
        csv_for_stats = combined_filtered_plots
    if not csv_for_stats and accepted_df_dir and os.path.isdir(accepted_df_dir):
        for f in os.listdir(accepted_df_dir):
            if 'chromatograms_accepted' in f or f == FILTERED_PEPTIDES_CSV_BASENAME:
                csv_for_stats = os.path.join(accepted_df_dir, f)
                break
    if csv_for_stats and os.path.isfile(csv_for_stats):
        from combined_overhang_visualization import create_peptide_statistic_scatter_plots
        stats_plot_dir = os.path.join(plots_dir, 'accepted', 'peptide_statistics_scatter')
        os.makedirs(stats_plot_dir, exist_ok=True)
        print(f"[run_visualization] Peptide statistic scatter plots (good/ok/poor) -> {stats_plot_dir}")
        create_peptide_statistic_scatter_plots(csv_for_stats, stats_plot_dir, protein_id=args.protein or '')
    return True


def _print_run_summary():
    """Print final summary of warnings and errors from this run."""
    try:
        from visualization.csv_validation import get_run_summary
        summary = get_run_summary()
        if not summary:
            return
        errors = [m for k, m in summary if k == "error"]
        warnings = [m for k, m in summary if k == "warning"]
        print("\n" + "=" * 70)
        print(" RUN SUMMARY — Warnings and errors during this run")
        print("=" * 70)
        if errors:
            print(f"\n  Errors ({len(errors)}):")
            for msg in errors[:20]:
                print(f"    - {msg}")
            if len(errors) > 20:
                print(f"    ... and {len(errors) - 20} more")
        if warnings:
            print(f"\n  Warnings ({len(warnings)}):")
            for msg in warnings[:20]:
                print(f"    - {msg}")
            if len(warnings) > 20:
                print(f"    ... and {len(warnings) - 20} more")
        print("=" * 70 + "\n")
    except Exception as e:
        print(f"\n[run_visualization] Could not print run summary: {e}\n")


def main():
    args = _parse_args()
    try:
        from visualization.csv_validation import debug_breakpoint
        debug_breakpoint("run_visualization main start")
    except Exception:
        pass
    # Print all filtering criteria first so edits are visible when you run the script
    print_filtering_criteria()

    run_all = getattr(args, 'all', False)
    do_chrom = args.chromatograms or run_all
    do_rt = args.rt_windows or run_all
    do_combined = args.combined or run_all

    if not (do_chrom or do_rt or do_combined):
        print("No plot type selected. Use --chromatograms, --rt-windows, --combined, or --all.")
        sys.exit(1)

    base_dir = args.output_dir if getattr(args, 'output_dir', None) else None
    dataframes_dir, plots_dir = _next_versioned_dirs(do_chrom, do_rt, do_combined, base_dir=base_dir)
    os.makedirs(dataframes_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    accepted_df = os.path.join(dataframes_dir, 'accepted')
    rejected_df = os.path.join(dataframes_dir, 'rejected')
    accepted_plots = os.path.join(plots_dir, 'accepted')
    rejected_plots = os.path.join(plots_dir, 'rejected')
    os.makedirs(accepted_df, exist_ok=True)
    os.makedirs(rejected_df, exist_ok=True)
    os.makedirs(accepted_plots, exist_ok=True)
    os.makedirs(rejected_plots, exist_ok=True)
    if do_chrom:
        os.makedirs(os.path.join(accepted_plots, 'chromatograms_peptides'), exist_ok=True)
        os.makedirs(os.path.join(rejected_plots, 'chromatograms_peptides_rejected'), exist_ok=True)
    print(f"Output: dataframes -> {dataframes_dir} (accepted/ rejected/)")
    print(f"Output: plots -> {plots_dir} (accepted/ rejected/)")
    sys.stdout.flush()

    # --no-filter is alias for --no-default-filter
    if getattr(args, 'no_filter', False):
        args.no_default_filter = True

    # When --no-default-filter and no --filter-csv: use full CSV as the peptide list (no rows dropped by q-value/PEP/overhangs)
    if not getattr(args, 'filter_csv', None) and getattr(args, 'no_default_filter', False):
        if args.csv and os.path.exists(args.csv):
            import shutil
            filter_path = os.path.join(dataframes_dir, FILTERED_PEPTIDES_CSV_BASENAME)
            shutil.copy2(args.csv, filter_path)
            plots_filter = os.path.join(plots_dir, FILTERED_PEPTIDES_CSV_BASENAME)
            shutil.copy2(args.csv, plots_filter)
            args.filter_csv = filter_path
            try:
                with open(args.csv, 'r') as f:
                    lines = f.readlines()
                skip = 1 if (lines and 'CometVersion' in lines[0]) else 0
                n_rows = len(lines) - 1 - skip  # one header line; optional CometVersion line
            except Exception:
                n_rows = '?'
            print(f"[run_visualization] No filter: using full CSV ({os.path.basename(args.csv)}) as peptide list ({n_rows} rows) -> {filter_path}")
    # Default filter: when no --filter-csv and not --no-default-filter, build filtered peptide list from CSV (q-value, PEP, mods, min overhangs)
    elif not getattr(args, 'filter_csv', None) and not getattr(args, 'no_default_filter', False):
        print("[run_visualization] Building filtered peptide list from CSV...")
        sys.stdout.flush()
        reject_mods = FILTER_REJECT_MODS and not getattr(args, 'no_exclude_mods', False)
        filter_path = _default_filtered_csv(args.csv, dataframes_dir, reject_mods=reject_mods)
        if filter_path:
            args.filter_csv = filter_path
            import shutil
            plots_filter = os.path.join(plots_dir, FILTERED_PEPTIDES_CSV_BASENAME)
            os.makedirs(plots_dir, exist_ok=True)
            shutil.copy2(filter_path, plots_filter)

    test_mode = getattr(args, 'test', False)
    # With --test: create a larger peptide pool so chromatograms can cycle until 30 accepted in accepted_chromatograms.
    # Chromatogram extraction must always run first so peak_windows/total_area exist for RT windows and combined.
    TEST_POOL_SIZE = 150  # Pool size for test; chromatogram step will stop once 30 accepted are saved
    if test_mode and (do_chrom or do_rt or do_combined) and args.filter_csv and os.path.isfile(args.filter_csv):
        print("[run_visualization] Loading peptide list for test sample...")
        sys.stdout.flush()
        import pandas as pd
        try:
            with open(args.filter_csv, 'r') as f:
                first = f.readline()
            skip = 1 if 'CometVersion' in first else 0
            df = pd.read_csv(args.filter_csv, sep=',', skiprows=skip, engine='python', quotechar='"', on_bad_lines='warn')
            # Validate filter CSV before test sampling
            try:
                from visualization.csv_validation import validate_and_log
                validate_and_log(
                    df,
                    source_name=os.path.basename(args.filter_csv) + ' (test pool)',
                    required_columns=('plain_peptide', 'charge'),
                    validate_identifiers=True,
                    validate_numeric=True,
                )
            except Exception as e:
                print(f"[run_visualization] Validation warning: {e}")
                try:
                    from visualization.csv_validation import record_run_warning
                    record_run_warning(f"Test pool CSV validation: {e}")
                except Exception:
                    pass
            n_orig = len(df)
            size = min(TEST_POOL_SIZE, n_orig)
            if size < n_orig:
                # No fixed random_state so each run gets a different random sample of peptides
                df = df.sample(n=size).copy()
            test_filtered = os.path.join(plots_dir, FILTERED_PEPTIDES_CSV_BASENAME)
            os.makedirs(plots_dir, exist_ok=True)
            df.to_csv(test_filtered, index=False)
            args.filter_csv = test_filtered
            print(f"[TEST] Using {size} peptides as pool (chromatograms will cycle until {FILTER_TEST_SAMPLE_SIZE} accepted in accepted/chromatograms_peptides)")
        except Exception as e:
            print(f"[run_visualization] WARNING: Could not create test sample: {e}; using full filter.")
            try:
                from visualization.csv_validation import record_run_warning
                record_run_warning(f"Could not create test sample: {e}")
            except Exception:
                pass

    # Chromatogram extraction MUST run first. It produces peak_windows.csv / peak_windows_all.csv and total_area
    # used by RT windows (filtered_rt_windows_zoom_*, etc.) and combined (unique_peptides, etc.).
    need_chrom_for_downstream = do_rt or do_combined
    if do_chrom or need_chrom_for_downstream:
        if need_chrom_for_downstream and not do_chrom:
            print("\n" + "=" * 60 + "\n[ run_visualization ] Chromatograms (required for RT windows / combined metrics)\n" + "=" * 60)
        else:
            print("\n" + "=" * 60 + "\n[ run_visualization ] Chromatograms\n" + "=" * 60)
        sys.stdout.flush()
        try:
            from visualization.csv_validation import debug_breakpoint
            debug_breakpoint("before chromatogram step")
        except Exception:
            pass
        # Chromatogram step: use same filter as RT/combined (full default or test 30-peptide list).
        chrom_filter = args.filter_csv if (args.filter_csv and os.path.isfile(args.filter_csv)) else None
        if chrom_filter is None and args.csv and os.path.exists(args.csv) and not getattr(args, 'no_default_filter', False):
            reject_mods = FILTER_REJECT_MODS and not getattr(args, 'no_exclude_mods', False)
            chrom_filter = _default_filtered_csv(args.csv, dataframes_dir, reject_mods=reject_mods)
        _run_chromatograms(args, plots_dir, dataframes_dir, accepted_plots, rejected_plots, accepted_df, rejected_df, force_full_run=need_chrom_for_downstream, chrom_filter_csv=chrom_filter)
        # Use accepted_windows_integration.csv as filter for RT windows and combined (and 3-channel plots)
        if (do_rt or do_combined) and chrom_filter and dataframes_dir and os.path.isdir(dataframes_dir):
            base_name = os.path.splitext(os.path.basename(chrom_filter))[0]
            accepted_wi = os.path.join(dataframes_dir, f'{base_name}_accepted_windows_integration.csv')
            if os.path.isfile(accepted_wi):
                args.filter_csv = accepted_wi
                print(f"[run_visualization] Using accepted windows integration as filter for RT windows and combined: {os.path.basename(accepted_wi)}")
        chrom_accepted = os.path.join(accepted_plots, 'chromatograms_peptides')
        chrom_rejected = os.path.join(rejected_plots, 'chromatograms_peptides_rejected')
        os.makedirs(chrom_accepted, exist_ok=True)
        os.makedirs(chrom_rejected, exist_ok=True)
        n_accepted = len([f for f in os.listdir(chrom_accepted) if f.endswith('.png') and not f.startswith('all_')]) if os.path.isdir(chrom_accepted) else 0
        n_rejected = len([f for f in os.listdir(chrom_rejected) if f.endswith('.png')]) if os.path.isdir(chrom_rejected) else 0
        print(f"\n[ run_visualization ] Chromatogram peptide plots:")
        print(f"  Accepted -> {os.path.abspath(chrom_accepted)}  ({n_accepted} peptide PNGs)")
        print(f"  Rejected -> {os.path.abspath(chrom_rejected)}  ({n_rejected} peptide PNGs)")
        if n_accepted == 0 and n_rejected == 0:
            print("\n  *** WARNING: No chromatogram peptide plots were saved! ***")
            print("  Run with: python run_visualization.py --chromatograms  (or --all)")
            print("  Possible causes: step failed, all peptides filtered out, or --extract-only was used.")
            print("  Check the chromatogram step output above for errors or early-reject reasons.")
            try:
                from visualization.csv_validation import record_run_warning
                record_run_warning("No chromatogram peptide plots were saved (0 accepted, 0 rejected)")
            except Exception:
                pass
    if do_rt:
        try:
            from visualization.csv_validation import debug_breakpoint
            debug_breakpoint("before RT windows step")
        except Exception:
            pass
        print("\n" + "=" * 60 + "\n[ run_visualization ] RT windows\n" + "=" * 60)
        _run_rt_windows(args, plots_dir, dataframes_dir, accepted_df)
    if do_combined:
        try:
            from visualization.csv_validation import debug_breakpoint
            debug_breakpoint("before combined step")
        except Exception:
            pass
        print("\n" + "=" * 60 + "\n[ run_visualization ] Combined overhang\n" + "=" * 60)
        _run_combined(args, plots_dir, dataframes_dir, accepted_df)
    try:
        from visualization.csv_validation import debug_breakpoint
        debug_breakpoint("run_visualization main end")
    except Exception:
        pass
    _print_run_summary()


if __name__ == '__main__':
    main()
