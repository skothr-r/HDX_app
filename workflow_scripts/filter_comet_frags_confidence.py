#!/usr/bin/env python3
"""
Step 4: Filter Comet fragments by confidence (Q-value and PEP).

Q/PEP thresholds are optional (disabled by default). When enabled,
rows must satisfy Q-value ≤ threshold AND PEP ≤ threshold.
Removes peptides where prev_aa or next_aa is proline (P).
Output column order: protein_position, peptide_sequence, charge, theoretical_mz, MS1_RT_minutes, qvalue, pep, sp, then fragment columns.
Creates diagnostics/ directory and a scatter plot of PEP vs Q-value with thresholds.
Output: comet_frags_perc_openMS_prefilter.csv

Usage:
    python filter_comet_frags_confidence.py
    python filter_comet_frags_confidence.py --input data/comet_frags_perc_openMS.csv --output-dir data
"""

import argparse
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
DEFAULT_CSV = os.path.join(_PROJECT_ROOT, 'data', 'comet_frags_perc_openMS.csv')

QVALUE_COLUMNS = ['perc_qvalue', 'percolator_qvalue', 'q-value', 'qvalue', 'percolator_q-value', 'FDR', 'fdr']
PEP_COLUMNS = ['perc_PEP', 'percolator_PEP', 'PEP', 'pep']
SP_COLUMNS = ['sp_score', 'sp', 'Sp']
XCORR_COLUMNS = ['xcorr', 'XCorr', 'xCorr']
Q_THRESHOLD_DEFAULT = 0.05
PEP_THRESHOLD_DEFAULT = 0.05
MS1_MZ_ERROR_PPM_THRESHOLD = 6.0

# Output column order: lead columns, then fragment columns, then rest
LEAD_COLS = ['protein_position', 'peptide_sequence', 'charge', 'observed_mz', 'theoretical_mz', 'MS1_mz_error_ppm', 'MS1_RT_minutes']
FRAGMENT_COLS = ['comet_matched_frags', 'comet_matched_frags_mz', 'comet_matched_frags_intensities', 'comet_matched_frags_quality_scores',
                 'single_aa_overhang_fragment_pairs', 'single_aa_overhangs_protein_positions', 'matched fragment ion quality scores']


def main():
    ap = argparse.ArgumentParser(description='Filter by Q-value/PEP with optional XCorr/Sp thresholds')
    ap.add_argument('--input', '-i', default=DEFAULT_CSV,
                    help=f'Input CSV from Step 3 (default: {os.path.basename(DEFAULT_CSV)})')
    ap.add_argument('--output', '-o', default=None,
                    help='Output CSV path (default: same dir as input, suffix _prefilter)')
    ap.add_argument('--output-dir', default=None,
                    help='Output directory for CSV and diagnostics (default: same as input dir)')
    ap.add_argument('--threshold', type=float, default=None,
                    help=f'[Deprecated] Use --q-threshold and --pep-threshold. If set, applies to both.')
    ap.add_argument('--use-qpep', action='store_true',
                    help='Enable Q-value/PEP thresholds. Off by default.')
    ap.add_argument('--q-threshold', type=float, default=Q_THRESHOLD_DEFAULT,
                    help=f'Q-value threshold (default: {Q_THRESHOLD_DEFAULT})')
    ap.add_argument('--pep-threshold', type=float, default=PEP_THRESHOLD_DEFAULT,
                    help=f'PEP threshold (default: {PEP_THRESHOLD_DEFAULT})')
    ap.add_argument('--use-xcorr', action='store_true',
                    help='Enable XCorr threshold filter')
    ap.add_argument('--xcorr-min', type=float, default=0.0,
                    help='Minimum XCorr when --use-xcorr is enabled (default: 0.0)')
    ap.add_argument('--use-sp', action='store_true',
                    help='Enable Sp threshold filter')
    ap.add_argument('--sp-min', type=float, default=0.0,
                    help='Minimum Sp when --use-sp is enabled (default: 0.0)')
    ap.add_argument('--allow-carbamidomethyl', action='store_true',
                    help='Allow peptides with only Carbamidomethyl (C, 57.0215). Default: reject all modifications.')
    args = ap.parse_args()
    if args.threshold is not None:
        args.use_qpep = True
        args.q_threshold = args.threshold
        args.pep_threshold = args.threshold

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    try:
        import pandas as pd
        import numpy as np
    except ImportError:
        print("Error: pandas and numpy required. pip install pandas numpy")
        sys.exit(1)

    # Read CSV
    try:
        with open(args.input, 'r') as f:
            first = f.readline()
        skip = 1 if 'CometVersion' in first else 0
    except Exception:
        skip = 0

    df = pd.read_csv(args.input, sep=',', skiprows=skip, engine='python', quotechar='"', on_bad_lines='warn')
    # Compatibility: accept canonical renamed identifiers and legacy names.
    if 'peptide_sequence' in df.columns and 'plain_peptide' not in df.columns:
        df['plain_peptide'] = df['peptide_sequence']
    if 'protein_position' in df.columns and 'sequence_positions' not in df.columns:
        df['sequence_positions'] = df['protein_position']
    if 'MS2_RT_sec' in df.columns and 'MS2_retention_time_sec' not in df.columns:
        df['MS2_retention_time_sec'] = df['MS2_RT_sec']
    if 'MS2_RT_minutes' in df.columns and 'MS2_retention_time_min' not in df.columns:
        df['MS2_retention_time_min'] = df['MS2_RT_minutes']
    if 'MS1_mz_error' in df.columns and 'MS1_mz_error_ppm' not in df.columns:
        df['MS1_mz_error_ppm'] = df['MS1_mz_error']

    # Resolve output paths
    input_dir = os.path.dirname(os.path.abspath(args.input))
    base = os.path.splitext(os.path.basename(args.input))[0]
    out_dir = args.output_dir or input_dir
    os.makedirs(out_dir, exist_ok=True)
    diagnostics_dir = os.path.join(out_dir, 'diagnostics')
    os.makedirs(diagnostics_dir, exist_ok=True)

    if args.output:
        out_csv = args.output
    else:
        # Append _prefilter if not already present
        if base.endswith('_prefilter'):
            out_csv = os.path.join(out_dir, base + '.csv')
        else:
            out_csv = os.path.join(out_dir, base + '_prefilter.csv')

    # Find Q-value and PEP columns
    qcol = None
    for c in QVALUE_COLUMNS:
        if c in df.columns:
            qcol = c
            break
    pepcol = None
    for c in PEP_COLUMNS:
        if c in df.columns:
            pepcol = c
            break
    xcorrcol = None
    for c in XCORR_COLUMNS:
        if c in df.columns:
            xcorrcol = c
            break
    spcol = None
    for c in SP_COLUMNS:
        if c in df.columns:
            spcol = c
            break
    mzerrcol = 'MS1_mz_error_ppm' if 'MS1_mz_error_ppm' in df.columns else None

    active_criteria = ['prolines', 'modifications']
    if mzerrcol:
        active_criteria.append(f'|MS1_mz_error_ppm| ≤ {MS1_MZ_ERROR_PPM_THRESHOLD:g}')
    if args.use_qpep:
        active_criteria.append(f'Q-value ≤ {args.q_threshold}')
        active_criteria.append(f'PEP ≤ {args.pep_threshold}')
    if args.use_xcorr:
        active_criteria.append(f'XCorr ≥ {args.xcorr_min}')
    if args.use_sp:
        active_criteria.append(f'Sp ≥ {args.sp_min}')
    print("Prefilter active exclusion criteria: " + "; ".join(active_criteria))

    def _pass(row):
        q_ok = True
        if qcol and qcol in df.columns and pd.notna(row.get(qcol)):
            try:
                q_ok = float(row[qcol]) <= args.q_threshold
            except (TypeError, ValueError):
                q_ok = True
        else:
            q_ok = True
        pep_ok = True
        if pepcol and pepcol in df.columns and pd.notna(row.get(pepcol)):
            try:
                pep_ok = float(row[pepcol]) <= args.pep_threshold
            except (TypeError, ValueError):
                pep_ok = True
        else:
            pep_ok = True
        conf_ok = (q_ok and pep_ok) if args.use_qpep else True
        xcorr_ok = True
        if args.use_xcorr and xcorrcol and xcorrcol in df.columns and pd.notna(row.get(xcorrcol)):
            try:
                xcorr_ok = float(row[xcorrcol]) >= args.xcorr_min
            except (TypeError, ValueError):
                xcorr_ok = True
        sp_ok = True
        if args.use_sp and spcol and spcol in df.columns and pd.notna(row.get(spcol)):
            try:
                sp_ok = float(row[spcol]) >= args.sp_min
            except (TypeError, ValueError):
                sp_ok = True
        mzerr_ok = True
        if mzerrcol and mzerrcol in df.columns and pd.notna(row.get(mzerrcol)):
            try:
                mzerr_ok = abs(float(row[mzerrcol])) <= MS1_MZ_ERROR_PPM_THRESHOLD
            except (TypeError, ValueError):
                mzerr_ok = True
        return conf_ok and xcorr_ok and sp_ok and mzerr_ok

    n_before_qpep = len(df)
    if args.use_qpep:
        if qcol is None and pepcol is None:
            print("Warning: Q/PEP filter enabled but no Q-value or PEP column found. Keeping all rows.")
            df_filtered = df.copy()
        else:
            mask = df.apply(_pass, axis=1)
            df_filtered = df.loc[mask].copy()
        n_qpep_removed = n_before_qpep - len(df_filtered)
        if n_qpep_removed > 0:
            extra = []
            if args.use_xcorr:
                extra.append(f'XCorr ≥ {args.xcorr_min}')
            if args.use_sp:
                extra.append(f'Sp ≥ {args.sp_min}')
            extra_txt = (' and ' + ' and '.join(extra)) if extra else ''
            print(f"Removed {n_qpep_removed} rows (kept rows with Q-value ≤ {args.q_threshold} and PEP ≤ {args.pep_threshold}{extra_txt})")
    else:
        df_filtered = df.copy()
        print("Q/PEP thresholds disabled (columns kept for diagnostics).")
    if mzerrcol:
        print(f"MS1 m/z error filter: |MS1_mz_error_ppm| ≤ {MS1_MZ_ERROR_PPM_THRESHOLD:g}")

    # Remove peptides with modifications. Default: reject all (unmodified only).
    # With --allow-carbamidomethyl: allow only Carbamidomethyl (C, 57.0215).
    n_before_mods = len(df_filtered)
    ALLOWED_MOD_MASSES = frozenset(['57.0215', '57.021500', '57.02146', '57.021464'])

    def _has_modifications_to_reject(m):
        if m is None or (isinstance(m, float) and pd.isna(m)):
            return False
        s = str(m).strip()
        if s.lower() in ('', '-', 'nan', 'none'):
            return False
        if args.allow_carbamidomethyl:
            # Reject only if any mass is not Carbamidomethyl
            masses = []
            for part in s.split(','):
                part = part.strip()
                if '_' in part:
                    mass = part.rsplit('_', 1)[-1].strip()
                    if mass and mass.replace('.', '').replace('-', '').isdigit():
                        masses.append(mass.rstrip('0').rstrip('.') if '.' in mass else mass)
            if not masses:
                return True  # Unknown format, reject
            for mass in masses:
                norm = mass.rstrip('0').rstrip('.') if '.' in mass else mass
                if norm not in ALLOWED_MOD_MASSES:
                    return True
            return False
        else:
            # Reject all modifications (unmodified peptides only)
            return True

    if 'modifications' in df_filtered.columns:
        mask_mods = ~df_filtered['modifications'].apply(_has_modifications_to_reject)
        df_filtered = df_filtered.loc[mask_mods].copy()
        n_mods_removed = n_before_mods - len(df_filtered)
        if n_mods_removed > 0:
            mod_desc = 'modifications (unmodified only)' if not args.allow_carbamidomethyl else 'variable modifications'
            print(f"Removed {n_mods_removed} rows ({mod_desc})")

    # Remove peptides where prev_aa or next_aa is proline (P), or peptide starts/ends with proline
    n_before_pro = len(df_filtered)
    def _is_proline_reject(row):
        # Flanking prolines (prev_aa or next_aa = P)
        if 'prev_aa' in df.columns and 'next_aa' in df.columns:
            prev = str(row.get('prev_aa', '')).strip().upper()
            next_ = str(row.get('next_aa', '')).strip().upper()
            if prev == 'P' or next_ == 'P':
                return True
        # Peptide termini (starts or ends with P)
        seq = str(row.get('plain_peptide', '')).strip() if 'plain_peptide' in df.columns else ''
        if seq and (seq[0].upper() == 'P' or seq[-1].upper() == 'P'):
            return True
        return False
    mask_pro = ~df_filtered.apply(_is_proline_reject, axis=1)
    df_filtered = df_filtered.loc[mask_pro].copy()
    n_pro_removed = n_before_pro - len(df_filtered)
    if n_pro_removed > 0:
        print(f"Removed {n_pro_removed} rows (proline: flanking or peptide termini)")

    # Build per-row rejection reason tags against original input rows.
    # These tags are used for rejected CSV + rejection-reason diagnostics.
    idx_all = df.index
    q_fail = pd.Series(False, index=idx_all)
    pep_fail = pd.Series(False, index=idx_all)
    conf_fail = pd.Series(False, index=idx_all)
    mods_fail = pd.Series(False, index=idx_all)
    pro_fail = pd.Series(False, index=idx_all)
    xcorr_fail = pd.Series(False, index=idx_all)
    sp_fail = pd.Series(False, index=idx_all)
    mzppm_fail = pd.Series(False, index=idx_all)

    # Confidence failure matches current logic when enabled: q_ok AND pep_ok must pass.
    q_ok_series = pd.Series(True, index=idx_all)
    pep_ok_series = pd.Series(True, index=idx_all)
    if qcol and qcol in df.columns:
        q_num = pd.to_numeric(df[qcol], errors='coerce')
        q_fail = q_num.notna() & (q_num > args.q_threshold)
        q_ok_series = ~q_fail
    if pepcol and pepcol in df.columns:
        pep_num = pd.to_numeric(df[pepcol], errors='coerce')
        pep_fail = pep_num.notna() & (pep_num > args.pep_threshold)
        pep_ok_series = ~pep_fail
    conf_fail = ~(q_ok_series & pep_ok_series) if args.use_qpep else pd.Series(False, index=idx_all)

    if 'modifications' in df.columns:
        mods_fail = df['modifications'].apply(_has_modifications_to_reject)
    pro_fail = df.apply(_is_proline_reject, axis=1)
    if args.use_xcorr and xcorrcol and xcorrcol in df.columns:
        xv = pd.to_numeric(df[xcorrcol], errors='coerce')
        xcorr_fail = xv.notna() & (xv < args.xcorr_min)
    if args.use_sp and spcol and spcol in df.columns:
        sv = pd.to_numeric(df[spcol], errors='coerce')
        sp_fail = sv.notna() & (sv < args.sp_min)
    if mzerrcol and mzerrcol in df.columns:
        mv = pd.to_numeric(df[mzerrcol], errors='coerce')
        mzppm_fail = mv.notna() & (mv.abs() > MS1_MZ_ERROR_PPM_THRESHOLD)

    def _reason_tags(i):
        tags = []
        if bool(mods_fail.get(i, False)):
            tags.append('mods')
        if bool(pro_fail.get(i, False)):
            tags.append('prolines')
        if bool(conf_fail.get(i, False)):
            if bool(q_fail.get(i, False)):
                tags.append('Q')
            if bool(pep_fail.get(i, False)):
                tags.append('PEP')
            if not bool(q_fail.get(i, False)) and not bool(pep_fail.get(i, False)):
                tags.append('confidence')
        if bool(xcorr_fail.get(i, False)):
            tags.append('XCorr')
        if bool(sp_fail.get(i, False)):
            tags.append('Sp')
        if bool(mzppm_fail.get(i, False)):
            tags.append('MS1_ppm')
        return '; '.join(tags) if tags else 'other'

    n_total = len(df)
    n_pass = len(df_filtered)

    def _format_prefilter_output(df_in):
        out = df_in.copy()
        # Rename for output
        if 'mz' in out.columns and 'theoretical_mz' not in out.columns:
            out = out.rename(columns={'mz': 'theoretical_mz'})
        if 'MS1_retention_time_min' in out.columns and 'MS1_RT_minutes' not in out.columns:
            out = out.rename(columns={'MS1_retention_time_min': 'MS1_RT_minutes'})
        if 'MS1_retention_time_sec' in out.columns:
            if 'MS1_RT_sec' not in out.columns:
                out = out.rename(columns={'MS1_retention_time_sec': 'MS1_RT_sec'})
            else:
                out = out.drop(columns=['MS1_retention_time_sec'])
        if 'MS1_retention_time_intensity' in out.columns:
            if 'MS1_RT_intensity' not in out.columns:
                out = out.rename(columns={'MS1_retention_time_intensity': 'MS1_RT_intensity'})
            else:
                out = out.drop(columns=['MS1_retention_time_intensity'])
        # Keep original confidence metric columns from previous step;
        # do not add duplicate aliases like qvalue/pep/sp when perc_qvalue/perc_PEP/sp_score already exist.
        if 'matched fragment ions' in out.columns and 'comet_matched_frags' not in out.columns:
            out = out.rename(columns={'matched fragment ions': 'comet_matched_frags'})
        if 'matched fragment ion mz' in out.columns and 'comet_matched_frags_mz' not in out.columns:
            out = out.rename(columns={'matched fragment ion mz': 'comet_matched_frags_mz'})
        if 'matched fragment ion intensities' in out.columns and 'comet_matched_frags_intensities' not in out.columns:
            out = out.rename(columns={'matched fragment ion intensities': 'comet_matched_frags_intensities'})
        if 'comet_amtched_frags_intensities' in out.columns and 'comet_matched_frags_intensities' not in out.columns:
            out = out.rename(columns={'comet_amtched_frags_intensities': 'comet_matched_frags_intensities'})
        if 'matched fragment ion quality scores' in out.columns and 'comet_matched_frags_quality_scores' not in out.columns:
            out = out.rename(columns={'matched fragment ion quality scores': 'comet_matched_frags_quality_scores'})
        # Canonical workflow names; drop temporary legacy aliases if canonical already present.
        if 'plain_peptide' in out.columns:
            if 'peptide_sequence' not in out.columns:
                out = out.rename(columns={'plain_peptide': 'peptide_sequence'})
            else:
                out = out.drop(columns=['plain_peptide'])
        if 'sequence_positions' in out.columns:
            if 'protein_position' not in out.columns:
                out = out.rename(columns={'sequence_positions': 'protein_position'})
            else:
                out = out.drop(columns=['sequence_positions'])
        if 'MS2_retention_time_sec' in out.columns:
            if 'MS2_RT_sec' not in out.columns:
                out = out.rename(columns={'MS2_retention_time_sec': 'MS2_RT_sec'})
            else:
                out = out.drop(columns=['MS2_retention_time_sec'])
        if 'MS2_RT_sec' in out.columns and 'MS2_RT_minutes' not in out.columns:
            out.insert(
                out.columns.get_loc('MS2_RT_sec') + 1,
                'MS2_RT_minutes',
                pd.to_numeric(out['MS2_RT_sec'], errors='coerce') / 60.0
            )
        if 'MS1_mz_error' in out.columns and 'MS1_mz_error_ppm' not in out.columns:
            out = out.rename(columns={'MS1_mz_error': 'MS1_mz_error_ppm'})

        # Reorder columns:
        # - keep MS1_RT_minutes/intensity early (after observed_mz/theoretical_mz)
        # - keep comet_matched_frags* near end
        # - push MS2_RT* to very end
        lead = [c for c in LEAD_COLS if c in out.columns]
        ms1_extra_cols = [c for c in ['MS1_RT_intensity'] if c in out.columns]
        frag_tail = [c for c in ['comet_matched_frags', 'comet_matched_frags_mz', 'comet_matched_frags_intensities', 'comet_matched_frags_quality_scores'] if c in out.columns]
        ms1_cols = ms1_extra_cols
        ms2_cols = [c for c in ['MS1_RT_sec', 'MS2_RT_sec', 'MS2_RT_minutes'] if c in out.columns]
        rest = [c for c in out.columns if c not in lead and c not in frag_tail and c not in ms1_cols and c not in ms2_cols]
        col_order = lead + ms1_cols + rest + frag_tail + ms2_cols
        out = out[[c for c in col_order if c in out.columns]]
        if out.empty:
            return out

        # Preserve shared workflow row organization:
        # sequence start -> peptide length (short->long) -> charge -> MS1 RT (if present).
        pos_col = 'protein_position' if 'protein_position' in out.columns else ('sequence_positions' if 'sequence_positions' in out.columns else None)
        seq_col = 'peptide_sequence' if 'peptide_sequence' in out.columns else ('plain_peptide' if 'plain_peptide' in out.columns else None)
        rt_col = 'MS1_RT_sec' if 'MS1_RT_sec' in out.columns else ('MS1_retention_time_sec' if 'MS1_retention_time_sec' in out.columns else None)
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
            sl = out.apply(_start_len, axis=1, result_type='expand')
            out['_sort_start'] = sl[0]
            out['_sort_len'] = sl[1]
            out['_sort_charge'] = pd.to_numeric(out.get('charge'), errors='coerce').fillna(999999)
            out['_sort_rt'] = pd.to_numeric(out.get(rt_col), errors='coerce').fillna(999999.0) if rt_col else 999999.0
            out = out.sort_values(by=['_sort_start', '_sort_len', '_sort_charge', '_sort_rt'], ascending=[True, True, True, True], kind='mergesort')
            out = out.drop(columns=['_sort_start', '_sort_len', '_sort_charge', '_sort_rt'])
        return out

    # Diagnostic scatter: PEP vs Q-value
    if qcol and pepcol and qcol in df.columns and pepcol in df.columns and n_total > 0:
        try:
            import matplotlib.pyplot as plt
            q_vals = pd.to_numeric(df[qcol], errors='coerce')
            pep_vals = pd.to_numeric(df[pepcol], errors='coerce')
            valid = q_vals.notna() & pep_vals.notna()
            q_vals = q_vals[valid]
            pep_vals = pep_vals[valid]
            passed = df.loc[valid].apply(_pass, axis=1).values if qcol and pepcol else np.ones(valid.sum(), dtype=bool)

            fig, ax = plt.subplots(figsize=(8, 8))
            fig.patch.set_facecolor('black')
            ax.set_facecolor('black')
            # Clamp for visualization (avoid log 0)
            q_plot = np.clip(q_vals, 1e-10, 1.0)
            pep_plot = np.clip(pep_vals, 1e-10, 1.0)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.scatter(q_plot[~passed], pep_plot[~passed], c='#CCCCCC', s=12, alpha=0.7, edgecolors='0.6', linewidths=0.3, label=f'Rejected ({n_total - n_pass})')
            ax.scatter(q_plot[passed], pep_plot[passed], c='#2E86AB', s=12, alpha=0.6, edgecolors='0.6', linewidths=0.3, label=f'Passing ({n_pass})')
            ax.axhline(args.pep_threshold, color='red', linestyle='--', linewidth=2, label=f'PEP = {args.pep_threshold}')
            ax.axvline(args.q_threshold, color='orange', linestyle='--', linewidth=2, label=f'Q-value = {args.q_threshold}')
            ax.set_xlabel(f'Q-value ({qcol})', fontsize=12, color='0.85')
            ax.set_ylabel(f'PEP ({pepcol})', fontsize=12, color='0.85')
            if args.use_qpep:
                ttl = f'Confidence filter: {n_pass} / {n_total} rows pass (Q≤{args.q_threshold} and PEP≤{args.pep_threshold})'
            else:
                ttl = f'Confidence filter (disabled): {n_pass} / {n_total} rows pass'
            ax.set_title(ttl, fontsize=14, color='0.9')
            ax.tick_params(colors='0.85')
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_color('0.85')
            for spine in ax.spines.values():
                spine.set_color('0.6')
            ax.legend(loc='upper right', fontsize=10, facecolor='#1a1a1a', edgecolor='0.6', labelcolor='0.9')
            ax.grid(True, alpha=0.25)
            diag_path = os.path.join(diagnostics_dir, 'confidence_filter_PEP_vs_Qvalue_scatter.png')
            fig.savefig(diag_path, dpi=150, bbox_inches='tight', facecolor='black')
            plt.close(fig)
            print(f"Diagnostic scatter saved: {diag_path}")
        except Exception as e:
            print(f"Warning: Could not create diagnostic scatter: {e}")

    out = _format_prefilter_output(df_filtered)
    rejected_raw = df.loc[~df.index.isin(df_filtered.index)].copy()
    if len(rejected_raw) > 0:
        rejected_raw['rejection_reason'] = [ _reason_tags(i) for i in rejected_raw.index ]
    else:
        rejected_raw['rejection_reason'] = pd.Series(dtype=str)
    rejected = _format_prefilter_output(rejected_raw)
    if 'rejection_reason' in rejected.columns:
        cols = ['rejection_reason'] + [c for c in rejected.columns if c != 'rejection_reason']
        rejected = rejected[cols]
    rejected_csv = out_csv.replace('_prefilter.csv', '_prefilter_rejected.csv') if out_csv.endswith('_prefilter.csv') else out_csv.replace('.csv', '_rejected.csv')
    out.to_csv(out_csv, index=False)
    rejected.to_csv(rejected_csv, index=False)
    print(f"Output: {out_csv} ({n_pass} rows, {n_total - n_pass} removed)")
    print(f"Rejected rows output: {rejected_csv} ({len(rejected)} rows)")

    # Diagnostic Venn diagram for rejected peptide reasons (top 2-3 reasons by peptide count).
    try:
        if len(rejected_raw) > 0 and 'rejection_reason' in rejected_raw.columns:
            pep_col_diag = 'plain_peptide' if 'plain_peptide' in rejected_raw.columns else ('peptide_sequence' if 'peptide_sequence' in rejected_raw.columns else None)
            if pep_col_diag:
                def _pep_key(row):
                    pep = str(row.get(pep_col_diag, '')).strip()
                    try:
                        ch = int(float(row.get('charge', 0)))
                    except (TypeError, ValueError):
                        ch = 0
                    m = str(row.get('modifications', '-')).strip()
                    if not m or m.lower() == 'nan':
                        m = '-'
                    return f'{pep}|{ch}|{m}'

                rejected_raw['_pep_key'] = rejected_raw.apply(_pep_key, axis=1)
                reason_sets = {}
                for _, row in rejected_raw.iterrows():
                    key = row['_pep_key']
                    reasons = [r.strip() for r in str(row.get('rejection_reason', '')).split(';') if r.strip()]
                    for r in reasons:
                        reason_sets.setdefault(r, set()).add(key)

                reason_sets = {k: v for k, v in reason_sets.items() if len(v) > 0}
                if len(reason_sets) >= 2:
                    import matplotlib.pyplot as plt
                    top = sorted(reason_sets.items(), key=lambda kv: len(kv[1]), reverse=True)[:3]
                    labels = [k for k, _ in top]
                    sets = [v for _, v in top]
                    fig, ax = plt.subplots(figsize=(8, 8))
                    fig.patch.set_facecolor('black')
                    ax.set_facecolor('black')
                    venn_ok = False
                    try:
                        from matplotlib_venn import venn2, venn3
                        if len(sets) == 2:
                            venn2(subsets=sets, set_labels=labels, ax=ax)
                            venn_ok = True
                        elif len(sets) == 3:
                            venn3(subsets=sets, set_labels=labels, ax=ax)
                            venn_ok = True
                    except Exception:
                        venn_ok = False
                    if not venn_ok:
                        # Fallback: overlap matrix heatmap when matplotlib_venn is unavailable.
                        import numpy as np
                        mat = np.zeros((len(sets), len(sets)), dtype=int)
                        for i in range(len(sets)):
                            for j in range(len(sets)):
                                mat[i, j] = len(sets[i] & sets[j]) if i != j else len(sets[i])
                        im = ax.imshow(mat, cmap='magma')
                        ax.set_xticks(range(len(labels)))
                        ax.set_yticks(range(len(labels)))
                        ax.set_xticklabels(labels, color='0.9')
                        ax.set_yticklabels(labels, color='0.9')
                        for i in range(len(sets)):
                            for j in range(len(sets)):
                                ax.text(j, i, str(mat[i, j]), ha='center', va='center', color='white', fontsize=10)
                        cbar = fig.colorbar(im, ax=ax)
                        cbar.ax.yaxis.set_tick_params(color='0.9')
                        plt.setp(cbar.ax.get_yticklabels(), color='0.9')
                        ax.set_title('Rejected peptide reason overlap (fallback matrix)', color='0.9')
                    else:
                        ax.set_title('Rejected peptide reason overlap (top reasons)', color='0.9')
                    for t in ax.texts:
                        try:
                            t.set_color('0.95')
                        except Exception:
                            pass
                    diag_venn = os.path.join(diagnostics_dir, 'prefilter_rejection_reasons_venn.png')
                    fig.savefig(diag_venn, dpi=150, bbox_inches='tight', facecolor='black')
                    plt.close(fig)
                    print(f"Diagnostic rejection-reason venn saved: {diag_venn}")
    except Exception as e:
        print(f"Warning: Could not create rejection-reason venn diagnostic: {e}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
