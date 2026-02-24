#!/usr/bin/env python3
"""
Step 7: Filter by isotopic envelope (M0, M+1, M+2, M+3 present; M0/M+1 > M+2 > M+3).

Runs after chromatogram extraction (Step 6). Keeps only rows where:
  (1) M0, M+1, M+2, and M+3 are present in summed MS1 (has_required_isotopes)
  (2) Envelope passes: both M0 and M+1 are greater than M+2, and M+2 > M+3
      (computed directly from m0/m1/m2/m3_intensity when available)

Requires extraction envelope columns from Step 6. Prefers direct intensity columns:
  m0_intensity, m1_intensity, m2_intensity, m3_intensity.
Falls back to envelope_ok/m0_m1_gt_m2_m3/m0_gt_m1_gt_m2 for older CSVs.

Output: comet_frags_perc_openMS_prefilter_extraction_envelope.csv

When chromatogram_metrics_all.csv and chromatogram_traces.npz exist, generates
MS1 chromatogram plots for all peptides in the current Envelope step input
(accepted + rejected split into separate output folders).

Usage:
  python filter_envelope.py
  python filter_envelope.py --input extraction.csv --output-dir results
"""

import argparse
import os
import subprocess
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

DEFAULT_INPUT = os.path.join(_PROJECT_ROOT, 'data', 'comet_frags_perc_openMS_prefilter_extraction.csv')
ENVELOPE_SUFFIX = '_envelope'


def _to_bool(v):
    """Parse bool from various representations."""
    if v is None or (isinstance(v, float) and (v != v)):  # NaN
        return False
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    return s in ('true', '1', 'yes')


def main():
    ap = argparse.ArgumentParser(
        description='Step 7: Filter by isotopic envelope (M0, M+1, M+2, M+3 present; M0/M+1 > M+2 > M+3).'
    )
    ap.add_argument('--input', '-i', default=DEFAULT_INPUT,
                    help=f'Input CSV from Step 6 (default: {os.path.basename(DEFAULT_INPUT)})')
    ap.add_argument('--output', '-o', default=None, help='Output CSV path')
    ap.add_argument('--output-dir', default=None, help='Output directory')
    ap.add_argument(
        '--envelope-mode',
        choices=['strict_m3_required', 'm3_optional'],
        default='strict_m3_required',
        help=(
            "strict_m3_required: require M0/M+1/M+2/M+3 and M0/M+1 > M+2 > M+3; "
            "m3_optional: require M0/M+1/M+2 and M0/M+1 > M+2, then enforce M0/M+1/M+2 > M+3 only if M+3 has signal."
        ),
    )
    ap.add_argument('--no-plots', action='store_true', help='Skip MS1 chromatogram plot generation')
    args = ap.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input CSV not found: {args.input}")
        sys.exit(1)

    try:
        import pandas as pd
        import numpy as np
    except ImportError:
        print("Error: pandas and numpy required.")
        sys.exit(1)

    input_dir = os.path.dirname(os.path.abspath(args.input))
    base = os.path.splitext(os.path.basename(args.input))[0]
    out_dir = args.output_dir or input_dir
    os.makedirs(out_dir, exist_ok=True)

    if args.output:
        out_csv = args.output
    else:
        if base.endswith('_extraction'):
            out_csv = os.path.join(out_dir, f'{base}{ENVELOPE_SUFFIX}.csv')
        else:
            out_csv = os.path.join(out_dir, f'{base}{ENVELOPE_SUFFIX}.csv')

    try:
        with open(args.input, 'r') as f:
            first = f.readline()
        skip = 1 if 'CometVersion' in first else 0
    except Exception:
        skip = 0
    df = pd.read_csv(args.input, sep=',', skiprows=skip, engine='python', quotechar='"', on_bad_lines='warn')

    # Check for required envelope columns (written by Step 6 extraction)
    has_i0 = 'm0_intensity' in df.columns
    has_i1 = 'm1_intensity' in df.columns
    has_i2 = 'm2_intensity' in df.columns
    has_i3 = 'm3_intensity' in df.columns
    has_intensity_quartet = has_i0 and has_i1 and has_i2 and has_i3
    has_new_env = 'm0_m1_gt_m2_m3' in df.columns
    has_m0 = 'm0_gt_m1_gt_m2' in df.columns
    has_env_ok = 'envelope_ok' in df.columns
    has_req_iso = 'has_required_isotopes' in df.columns
    has_matched_isotopes = 'matched_isotopes' in df.columns

    if not has_intensity_quartet and not has_new_env and not has_m0 and not has_env_ok:
        print("Error: Input CSV is missing envelope metrics.")
        print("Need either m0/m1/m2/m3 intensity columns, or m0_m1_gt_m2_m3/envelope_ok/m0_gt_m1_gt_m2.")
        print("Run chromatogram extraction (Step 6) first to produce the extraction CSV with envelope data.")
        sys.exit(1)
    if not has_req_iso and not has_matched_isotopes:
        print("Error: Input CSV has no has_required_isotopes or matched_isotopes column.")
        print("Run chromatogram extraction (Step 6) first to produce summed-MS1 isotope presence metrics.")
        sys.exit(1)

    # Filter criteria from summed MS1 (integration window):
    # strict_m3_required:
    #   (1) M0, M+1, M+2, M+3 present
    #   (2) M0 and M+1 both > M+2, and M+2 > M+3
    # m3_optional:
    #   (1) M0, M+1, M+2 present
    #   (2) M0 and M+1 both > M+2
    #   (3) if M+3 has signal, require M0/M+1/M+2 > M+3

    def _num(v):
        x = pd.to_numeric(pd.Series([v]), errors='coerce').iloc[0]
        return float(x) if pd.notna(x) else None

    def _iso_tokens(row):
        mi = row.get('matched_isotopes')
        if pd.isna(mi):
            return set()
        s = str(mi).replace(' ', '')
        out = set()
        for tok in ('M+0', 'M+1', 'M+2', 'M+3'):
            if tok in s:
                out.add(tok)
        return out

    def _has_isotope_triplet(row):
        if has_matched_isotopes:
            toks = _iso_tokens(row)
            return ('M+0' in toks) and ('M+1' in toks) and ('M+2' in toks)
        if has_i0 and has_i1 and has_i2:
            i0 = _num(row.get('m0_intensity'))
            i1 = _num(row.get('m1_intensity'))
            i2 = _num(row.get('m2_intensity'))
            return (i0 is not None and i0 > 0) and (i1 is not None and i1 > 0) and (i2 is not None and i2 > 0)
        if has_req_iso:
            req = row.get('has_required_isotopes')
            return (not pd.isna(req)) and _to_bool(req)
        return False

    def _has_isotope_quartet(row):
        if has_matched_isotopes:
            toks = _iso_tokens(row)
            return ('M+0' in toks) and ('M+1' in toks) and ('M+2' in toks) and ('M+3' in toks)
        if has_req_iso:
            req = row.get('has_required_isotopes')
            return (not pd.isna(req)) and _to_bool(req)
        if has_intensity_quartet:
            i0 = _num(row.get('m0_intensity'))
            i1 = _num(row.get('m1_intensity'))
            i2 = _num(row.get('m2_intensity'))
            i3 = _num(row.get('m3_intensity'))
            return (
                (i0 is not None and i0 > 0)
                and (i1 is not None and i1 > 0)
                and (i2 is not None and i2 > 0)
                and (i3 is not None and i3 > 0)
            )
        return False

    def _m3_has_signal(row):
        if has_matched_isotopes:
            return 'M+3' in _iso_tokens(row)
        if has_i3:
            i3 = _num(row.get('m3_intensity'))
            return i3 is not None and i3 > 0
        return False

    def _passes_envelope(row):
        if args.envelope_mode == 'strict_m3_required':
            if not _has_isotope_quartet(row):
                return False
        else:
            if not _has_isotope_triplet(row):
                return False
        m_new = row.get('m0_m1_gt_m2_m3')
        m0 = row.get('m0_gt_m1_gt_m2')
        env = row.get('envelope_ok')
        # Primary path: compute directly from extraction intensity columns.
        if has_intensity_quartet:
            i0 = _num(row.get('m0_intensity'))
            i1 = _num(row.get('m1_intensity'))
            i2 = _num(row.get('m2_intensity'))
            i3 = _num(row.get('m3_intensity'))
            if i0 is not None and i1 is not None and i2 is not None:
                if not ((i0 > i2) and (i1 > i2)):
                    return False
                if args.envelope_mode == 'strict_m3_required':
                    if i3 is None:
                        return False
                    return i2 > i3
                # m3_optional mode
                if _m3_has_signal(row):
                    if i3 is None:
                        return False
                    return (i0 > i3) and (i1 > i3) and (i2 > i3)
                return True
        # Fallback: explicit boolean metric from newer extraction output.
        if has_new_env and pd.notna(m_new):
            if args.envelope_mode == 'strict_m3_required':
                return _to_bool(m_new)
            # In m3_optional mode, m0_m1_gt_m2_m3 is stricter than needed and still valid if True.
            if _to_bool(m_new):
                return True
        # Legacy fallbacks.
        if has_m0 and pd.notna(m0):
            if args.envelope_mode == 'strict_m3_required':
                return _to_bool(m0)
            # m0_gt_m1_gt_m2 does not include M+3 relation; apply conditional M+3 check if we can.
            if not _to_bool(m0):
                return False
            if _m3_has_signal(row) and has_i3 and has_i0 and has_i1 and has_i2:
                i0 = _num(row.get('m0_intensity'))
                i1 = _num(row.get('m1_intensity'))
                i2 = _num(row.get('m2_intensity'))
                i3 = _num(row.get('m3_intensity'))
                if None not in (i0, i1, i2, i3):
                    return (i0 > i3) and (i1 > i3) and (i2 > i3)
                return False
            return True
        if has_env_ok and pd.notna(env):
            return _to_bool(env)
        return False

    mask = df.apply(_passes_envelope, axis=1)
    df_out = df[mask].reset_index(drop=True)

    def _rows_counts(df_rows):
        if df_rows is None or len(df_rows) == 0:
            return (0, 0, 0)
        seq_col = next((c for c in ['plain_peptide', 'peptide_sequence', 'peptide', 'sequence'] if c in df_rows.columns), None)
        if seq_col is None:
            return (len(df_rows), 0, 0)
        seq_series = df_rows[seq_col].astype(str).str.strip()
        valid_seq_mask = seq_series.ne('') & seq_series.str.lower().ne('nan')
        seq_series = seq_series[valid_seq_mask]
        if len(seq_series) == 0:
            return (len(df_rows), 0, 0)
        unique_sequences = int(seq_series.nunique())
        if 'charge' in df_rows.columns:
            charge_series = pd.to_numeric(df_rows.loc[valid_seq_mask, 'charge'], errors='coerce')
            unique_peptides = int(pd.DataFrame({'seq': seq_series.values, 'charge': charge_series.values}).drop_duplicates().shape[0])
        else:
            unique_peptides = unique_sequences
        scan_col = next((c for c in ['scan', 'scan_num', 'scan_number', 'spectrum_scan', 'scan_id'] if c in df_rows.columns), None)
        if scan_col and 'charge' in df_rows.columns:
            scan_series = pd.to_numeric(df_rows.loc[valid_seq_mask, scan_col], errors='coerce')
            charge_series = pd.to_numeric(df_rows.loc[valid_seq_mask, 'charge'], errors='coerce')
            psms = int(pd.DataFrame({'seq': seq_series.values, 'charge': charge_series.values, 'scan': scan_series.values}).dropna(subset=['scan']).drop_duplicates().shape[0])
            if psms == 0:
                psms = int(len(df_rows))
        else:
            psms = int(len(df_rows))
        return (psms, unique_sequences, unique_peptides)
    # Standardize legacy retention_time headers to RT headers.
    for old, new in [
        ('MS1_retention_time_sec', 'MS1_RT_sec'),
        ('MS1_retention_time_min', 'MS1_RT_minutes'),
        ('MS1_retention_time_intensity', 'MS1_RT_intensity'),
        ('MS2_retention_time_sec', 'MS2_RT_sec'),
        ('MS2_retention_time_min', 'MS2_RT_minutes'),
        ('retention_time_sec', 'RT_sec'),
        ('retention_time_min', 'RT_minutes'),
    ]:
        if old in df_out.columns:
            if new in df_out.columns:
                df_out[new] = df_out[old].where(pd.notna(df_out[old]), df_out[new])
                df_out = df_out.drop(columns=[old])
            else:
                df_out = df_out.rename(columns={old: new})

    n_before = len(df)
    n_after = len(df_out)
    n_dropped = n_before - n_after

    # Preserve shared workflow row organization across downstream tabs.
    pos_col = 'protein_position' if 'protein_position' in df_out.columns else ('sequence_positions' if 'sequence_positions' in df_out.columns else None)
    seq_col = 'peptide_sequence' if 'peptide_sequence' in df_out.columns else ('plain_peptide' if 'plain_peptide' in df_out.columns else None)
    rt_col = 'MS1_RT_sec' if 'MS1_RT_sec' in df_out.columns else ('MS1_retention_time_sec' if 'MS1_retention_time_sec' in df_out.columns else None)
    if pos_col or seq_col:
        def _start_len(row):
            if pos_col:
                s = str(row.get(pos_col, '')).strip()
                if s and '-' in s:
                    try:
                        a, b = s.split('-', 1)
                        start = int(str(a).strip())
                        end = int(str(b).strip().split(',')[0])
                        return start, max(0, end - start + 1)
                    except Exception:
                        pass
            seq = str(row.get(seq_col, '')).strip() if seq_col else ''
            return 999999, (len(seq) if seq else 999999)
        sl = df_out.apply(_start_len, axis=1, result_type='expand')
        df_out['_sort_start'] = sl[0]
        df_out['_sort_len'] = sl[1]
        df_out['_sort_charge'] = pd.to_numeric(df_out.get('charge'), errors='coerce').fillna(999999)
        df_out['_sort_rt'] = pd.to_numeric(df_out.get(rt_col), errors='coerce').fillna(999999.0) if rt_col else 999999.0
        df_out = df_out.sort_values(by=['_sort_start', '_sort_len', '_sort_charge', '_sort_rt'], ascending=[True, True, True, True], kind='mergesort')
        df_out = df_out.drop(columns=['_sort_start', '_sort_len', '_sort_charge', '_sort_rt'])

    if 'MS1_mz_error' in df_out.columns and 'MS1_mz_error_ppm' not in df_out.columns:
        df_out = df_out.rename(columns={'MS1_mz_error': 'MS1_mz_error_ppm'})
    front_cols = [c for c in ['protein_position', 'peptide_sequence', 'charge', 'observed_mz', 'theoretical_mz', 'MS1_mz_error_ppm', 'MS1_RT_minutes'] if c in df_out.columns]
    ms1_cols = [c for c in ['MS1_RT_intensity'] if c in df_out.columns]
    frag_tail = [c for c in ['comet_matched_frags', 'comet_matched_frags_mz', 'comet_matched_frags_intensities', 'comet_matched_frags_quality_scores'] if c in df_out.columns]
    ms2_cols = [c for c in ['MS1_RT_sec', 'MS2_RT_sec', 'MS2_RT_minutes'] if c in df_out.columns]
    if front_cols or ms1_cols or frag_tail or ms2_cols:
        lead_cols = [c for c in df_out.columns if c not in front_cols and c not in ms1_cols and c not in frag_tail and c not in ms2_cols]
        df_out = df_out[front_cols + ms1_cols + lead_cols + frag_tail + ms2_cols]
    # Final ordering rule: RT columns explicitly in seconds always go last.
    rt_sec_cols = [
        c for c in df_out.columns
        if (('RT' in c or 'retention_time' in c) and c.endswith('_sec'))
    ]
    if rt_sec_cols:
        non_rt_sec_cols = [c for c in df_out.columns if c not in rt_sec_cols]
        df_out = df_out[non_rt_sec_cols + rt_sec_cols]

    df_out.to_csv(out_csv, index=False)
    print(f"[Step 7] Envelope filter: {n_before} -> {n_after} rows (dropped {n_dropped})")
    total_psms, total_unique_sequences, total_unique_peptides = _rows_counts(df)
    passed_psms, passed_unique_sequences, passed_unique_peptides = _rows_counts(df_out)
    print(f"Total PSMs: {total_psms}")
    print(f"Total unique sequences: {total_unique_sequences}")
    print(f"Total unique peptides (sequence + charge): {total_unique_peptides}")
    print(f"Passed PSMs: {passed_psms}")
    print(f"Passed unique sequences: {passed_unique_sequences}")
    print(f"Passed unique peptides (sequence + charge): {passed_unique_peptides}")
    print(f"[Step 7] Output: {os.path.abspath(out_csv)}")

    # Generate MS1 chromatogram plots for all peptides in this step input
    # (accepted and rejected are separated by envelope criteria downstream).
    if not args.no_plots and n_before > 0:
        dataframes_dir = os.path.join(out_dir, 'dataframes')
        metrics_csv = os.path.join(dataframes_dir, 'chromatogram_metrics_all.csv')
        traces_path = os.path.join(dataframes_dir, 'chromatogram_traces.npz')
        if os.path.exists(metrics_csv) and os.path.exists(traces_path):
            envelope_plots_dir = os.path.join(out_dir, 'envelope')
            envelope_rejected_dir = os.path.join(out_dir, 'envelope_rejected')
            os.makedirs(envelope_plots_dir, exist_ok=True)
            os.makedirs(envelope_rejected_dir, exist_ok=True)
            plot_script = os.path.join(_SCRIPT_DIR, 'plot_chromatograms_from_extraction.py')
            cmd = [sys.executable, plot_script,
                   '--chromatogram-metrics-csv', metrics_csv,
                   '--chromatogram-traces', traces_path,
                   '--output-dir', out_dir,
                   '--filter-csv', args.input,
                   '--output-accepted-dir', envelope_plots_dir,
                   '--output-rejected-dir', envelope_rejected_dir]
            try:
                subprocess.run(cmd, check=True, cwd=_PROJECT_ROOT)
                print(f"[Step 7] MS1 chromatogram plots saved to: {os.path.abspath(envelope_plots_dir)}")
            except subprocess.CalledProcessError as e:
                print(f"[Step 7] Warning: Plot generation failed: {e}", file=sys.stderr)
        else:
            print(f"[Step 7] Skipping plots (chromatogram_metrics_all.csv or chromatogram_traces.npz not found)")


if __name__ == '__main__':
    main()
