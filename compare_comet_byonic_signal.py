#!/usr/bin/env python3
"""
Compare total area and signal intensities for matched peptides: Comet vs Byonic.

Byonic CSV does not export total_area or apex intensity. We compute MS1 intensity
at Byonic's retention times (and optional small-window sum) from the same mzML
so we compare "our calculation at Comet's peak" vs "our calculation at Byonic's RT".

Output: scatter plots (Comet total_area vs intensity-at-Byonic-RT, Comet apex vs
intensity-at-Byonic-RT, Comet total_area vs apex) and correlation stats; CSV of
matched peptides with both metrics.

Usage:
  python compare_comet_byonic_signal.py data/byonic_search_peps.csv data/WT_nep2_0MUrea_08_with_qvalues_ms1_all.csv data/WT_nep2_0MUrea_08.mzML plots_v82_chromatograms_rt-windows_combined/peptide_chromatograms_primary/peak_windows_all.csv -o data/comet_byonic_signal_comparison
"""

import argparse
import os
import sys

def _load_mzml(raw_file):
    from pyopenms import MSExperiment, MzMLFile
    exp = MSExperiment()
    MzMLFile().load(raw_file, exp)
    return exp

def _get_intensity_at_rt_mz(exp, rt_sec, mz_target, ppm=10):
    """Get MS1 intensity at closest spectrum to rt_sec, at m/z within ppm. Returns (intensity, rt_used) or (None, None)."""
    best_spec = None
    best_diff = float('inf')
    for spec in exp:
        if spec.getMSLevel() != 1:
            continue
        d = abs(spec.getRT() - rt_sec)
        if d < best_diff and d <= 30:
            best_diff = d
            best_spec = spec
    if best_spec is None:
        return None, None
    tol = mz_target * ppm * 1e-6
    best_int = 0.0
    for peak in best_spec:
        if abs(peak.getMZ() - mz_target) <= tol:
            if peak.getIntensity() > best_int:
                best_int = peak.getIntensity()
    return (best_int if best_int > 0 else None), best_spec.getRT()

def _get_window_sum_at_mz(exp, rt_sec, mz_target, window_sec=30, ppm=10):
    """Sum MS1 intensities at m/z within ppm over RT window. Proxy for area."""
    min_rt = rt_sec - window_sec / 2
    max_rt = rt_sec + window_sec / 2
    tol = mz_target * ppm * 1e-6
    total = 0.0
    for spec in exp:
        if spec.getMSLevel() != 1:
            continue
        if not (min_rt <= spec.getRT() <= max_rt):
            continue
        for peak in spec:
            if abs(peak.getMZ() - mz_target) <= tol:
                total += peak.getIntensity()
                break
    return total if total > 0 else None

def main():
    ap = argparse.ArgumentParser(description='Compare Comet vs Byonic signal (total area, intensity) for matched peptides')
    ap.add_argument('--byonic-csv', dest='byonic_csv', required=True, help='Byonic CSV')
    ap.add_argument('--comet-csv', dest='comet_csv', required=True, help='Comet CSV with plain_peptide, charge')
    ap.add_argument('--mzml', dest='mzml', required=True, help='mzML file (same run)')
    ap.add_argument('--chrom-csv', dest='chrom_csv', required=True, help='Chromatogram CSV with total_area, apex_intensity, peptide, charge (e.g. peak_windows_all.csv)')
    ap.add_argument('-o', '--output', default='data/comet_byonic_signal_comparison', help='Output prefix for CSV and plots')
    ap.add_argument('--window-sec', type=float, default=30, help='RT window (sec) for Byonic window-sum (default 30)')
    args = ap.parse_args()

    import pandas as pd
    import numpy as np

    # Load Byonic (header row can contain newlines inside quoted fields)
    df_byonic = pd.read_csv(
        args.byonic_csv,
        encoding='utf-8',
        sep=',',
        engine='python',
        quotechar='"',
        on_bad_lines='warn',
    )
    df_byonic.columns = [str(c).replace('\r', ' ').replace('\n', ' ').strip() for c in df_byonic.columns]

    def plain_seq(s):
        if pd.isna(s):
            return None
        s = str(s).strip()
        if '.' in s:
            parts = s.split('.')
            if len(parts) >= 2:
                return parts[1]
            return s.replace('.', '')
        return s

    seq_col = next((c for c in df_byonic.columns if 'sequence' in c.lower() and 'unformat' in c.lower()), df_byonic.columns[2])
    z_col = 'z'
    scan_time_col = next((c for c in df_byonic.columns if 'Scan' in c and 'Time' in c), None)
    obs_mz_col = next((c for c in df_byonic.columns if 'Obs' in c and 'm/z' in c), None)
    if not scan_time_col or not obs_mz_col:
        print("Byonic CSV missing Scan Time or Obs. m/z column", file=sys.stderr)
        sys.exit(1)

    # Load Comet
    with open(args.comet_csv, 'r') as f:
        first = f.readline()
    skip = 1 if first.strip().startswith('CometVersion') or first.strip().startswith('#') else 0
    df_comet = pd.read_csv(args.comet_csv, skiprows=skip, sep=',', engine='python', quotechar='"', on_bad_lines='warn')
    comet_by_key = {}
    for (p, z), g in df_comet.groupby(['plain_peptide', df_comet['charge'].astype(int)]):
        comet_by_key[(str(p).strip(), int(z))] = g

    # Load chrom (total_area, apex_intensity per peptide)
    df_chrom = pd.read_csv(args.chrom_csv, sep=',', engine='python', quotechar='"', on_bad_lines='warn')
    pep_col = 'peptide' if 'peptide' in df_chrom.columns else 'plain_peptide'
    chrom_by_key = {}
    for _, r in df_chrom.iterrows():
        p = str(r[pep_col]).strip()
        z = int(r['charge'])
        chrom_by_key[(p, z)] = (r.get('total_area'), r.get('apex_intensity'))

    print(f"Comet: {len(comet_by_key)} unique (plain_peptide, charge)")
    print(f"Chrom: {len(chrom_by_key)} entries (peptide, charge); sample keys: {list(chrom_by_key.keys())[:5]}")

    def in_comet(seq, z):
        for (p, pz), g in comet_by_key.items():
            if pz != z:
                continue
            if p == seq or (seq in p) or (p in seq):
                return (p, g)
        return None

    def chrom_for_comet_peptide(seq, z):
        match = in_comet(seq, z)
        if not match:
            return None, None
        p, _ = match
        for (cp, cz), (ta, ap) in chrom_by_key.items():
            if cz != z:
                continue
            if cp == p or (seq in cp) or (cp in seq):
                return ta, ap
        return None, None

    # Build list of matched (seq, charge) from Byonic
    byonic_psms = []
    for _, row in df_byonic.iterrows():
        seq = plain_seq(row.get(seq_col))
        if not seq:
            continue
        try:
            z = int(float(row[z_col]))
        except (TypeError, ValueError):
            continue
        if in_comet(seq, z) is None:
            continue
        rt_min = pd.to_numeric(row.get(scan_time_col), errors='coerce')
        obs_mz = pd.to_numeric(row.get(obs_mz_col), errors='coerce')
        if pd.isna(rt_min) or pd.isna(obs_mz):
            continue
        byonic_psms.append({'sequence': seq, 'charge': z, 'rt_sec': float(rt_min) * 60, 'obs_mz': float(obs_mz)})

    byonic_unique_keys = set((r['sequence'], r['charge']) for r in byonic_psms)
    print(f"Byonic PSMs also in Comet: {len(byonic_psms)}; unique (sequence, charge): {len(byonic_unique_keys)}")
    if len(byonic_unique_keys) == 0:
        print("No Byonic peptides matched Comet. Check that both searches used the same run and compatible sequence format.")
        sys.exit(1)

    # Load mzML once
    print("Loading mzML...")
    exp = _load_mzml(args.mzml)
    print(f"  Loaded {exp.size()} spectra")

    # Aggregate Byonic: per (seq, z) max intensity at RT and sum over window
    from collections import defaultdict
    byonic_agg = defaultdict(lambda: {'intensity_at_rt': [], 'window_sum': []})
    for rec in byonic_psms:
        key = (rec['sequence'], rec['charge'])
        rt_sec, mz = rec['rt_sec'], rec['obs_mz']
        intensity, _ = _get_intensity_at_rt_mz(exp, rt_sec, mz)
        if intensity is not None:
            byonic_agg[key]['intensity_at_rt'].append(intensity)
        ws = _get_window_sum_at_mz(exp, rt_sec, mz, window_sec=args.window_sec)
        if ws is not None:
            byonic_agg[key]['window_sum'].append(ws)

    byonic_with_intensity = sum(1 for agg in byonic_agg.values() if agg['intensity_at_rt'])
    print(f"Byonic (seq, charge) with mzML intensity at RT: {byonic_with_intensity} / {len(byonic_agg)}")

    # Build comparison table: matched only
    rows = []
    chrom_match_count = 0
    for (seq, z), agg in byonic_agg.items():
        comet_ta, comet_apex = chrom_for_comet_peptide(seq, z)
        if comet_ta is None and comet_apex is None:
            continue
        chrom_match_count += 1
        byonic_max_int = max(agg['intensity_at_rt']) if agg['intensity_at_rt'] else None
        byonic_sum_int = sum(agg['intensity_at_rt']) if agg['intensity_at_rt'] else None
        byonic_window_sum = sum(agg['window_sum']) if agg['window_sum'] else None
        rows.append({
            'sequence': seq,
            'charge': z,
            'comet_total_area': comet_ta,
            'comet_apex_intensity': comet_apex,
            'byonic_intensity_max_at_rt': byonic_max_int,
            'byonic_intensity_sum_at_rt': byonic_sum_int,
            'byonic_window_sum': byonic_window_sum,
        })
    print(f"Byonic (seq, charge) with chrom match: {chrom_match_count} / {len(byonic_agg)}")

    col_names = ['sequence', 'charge', 'comet_total_area', 'comet_apex_intensity',
                 'byonic_intensity_max_at_rt', 'byonic_intensity_sum_at_rt', 'byonic_window_sum']
    df = pd.DataFrame(rows, columns=col_names) if rows else pd.DataFrame(columns=col_names)

    # Drop rows with no Byonic intensity
    df = df[df['byonic_intensity_max_at_rt'].notna()].copy()
    if len(df) == 0:
        print("No matched peptides with valid Byonic intensity from mzML.")
        print("Diagnostics:")
        print(f"  Chrom keys (first 8): {list(chrom_by_key.keys())[:8]}")
        print(f"  Byonic_agg keys (first 8): {list(byonic_agg.keys())[:8]}")
        if chrom_match_count == 0:
            print("  No Byonic key had a chrom match — check peptide/charge format in chrom CSV vs Comet/Byonic (e.g. plain_peptide vs modified).")
        else:
            print("  Some Byonic keys had chrom match but no mzML intensity at RT — check mzML is same run and RT/m/z columns.")
        sys.exit(1)

    out_csv = args.output + '.csv' if not args.output.endswith('.csv') else args.output
    df.to_csv(out_csv, index=False)
    print(f"Wrote {len(df)} matched peptides to {out_csv}")

    # Correlations (log scale: total_area and intensities span orders of magnitude)
    df_log = df.copy()
    for c in ['comet_total_area', 'comet_apex_intensity', 'byonic_intensity_max_at_rt', 'byonic_window_sum']:
        if c in df_log.columns and df_log[c].notna().any():
            df_log[c + '_log10'] = np.log10(df_log[c].clip(lower=1e-10))

    from scipy import stats
    def corr(x, y):
        mask = x.notna() & y.notna()
        if mask.sum() < 3:
            return None, None
        r, p = stats.pearsonr(x[mask], y[mask])
        return r, p

    print("\n--- Correlations (log10 scale) ---")
    if 'comet_total_area_log10' in df_log.columns and 'byonic_intensity_max_at_rt_log10' in df_log.columns:
        r, p = corr(df_log['comet_total_area_log10'], df_log['byonic_intensity_max_at_rt_log10'])
        if r is not None:
            print(f"  Comet total_area vs Byonic (intensity at Byonic RT): r = {r:.3f}, p = {p:.2e}")
    if 'comet_apex_intensity_log10' in df_log.columns and 'byonic_intensity_max_at_rt_log10' in df_log.columns:
        r, p = corr(df_log['comet_apex_intensity_log10'], df_log['byonic_intensity_max_at_rt_log10'])
        if r is not None:
            print(f"  Comet apex_intensity vs Byonic (intensity at Byonic RT): r = {r:.3f}, p = {p:.2e}")
    if 'comet_total_area_log10' in df_log.columns and 'comet_apex_intensity_log10' in df_log.columns:
        r, p = corr(df_log['comet_total_area_log10'], df_log['comet_apex_intensity_log10'])
        if r is not None:
            print(f"  Comet total_area vs Comet apex_intensity: r = {r:.3f}, p = {p:.2e}")
    if 'comet_total_area_log10' in df_log.columns and 'byonic_window_sum_log10' in df_log.columns:
        r, p = corr(df_log['comet_total_area_log10'], df_log['byonic_window_sum_log10'])
        if r is not None:
            print(f"  Comet total_area vs Byonic window sum ({args.window_sec}s): r = {r:.3f}, p = {p:.2e}")

    # Plots
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception:
        print("Matplotlib not available; skipping plots.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    xlabel_ta = 'Comet total_area (log10)'
    ylabel_bi = 'Byonic intensity at RT (log10)'
    x = df_log.get('comet_total_area_log10')
    y = df_log.get('byonic_intensity_max_at_rt_log10')
    if x is not None and y is not None and x.notna().any() and y.notna().any():
        ax = axes[0, 0]
        ax.scatter(x, y, alpha=0.6, s=20)
        ax.set_xlabel(xlabel_ta)
        ax.set_ylabel(ylabel_bi)
        ax.set_title('Comet total_area vs intensity at Byonic RT')
        r, p = corr(x, y)
        if r is not None:
            ax.text(0.05, 0.95, f'r = {r:.3f}', transform=ax.transAxes, fontsize=10, verticalalignment='top')

    x = df_log.get('comet_apex_intensity_log10')
    if x is not None and y is not None and x.notna().any() and y.notna().any():
        ax = axes[0, 1]
        ax.scatter(x, y, alpha=0.6, s=20)
        ax.set_xlabel('Comet apex_intensity (log10)')
        ax.set_ylabel(ylabel_bi)
        ax.set_title('Comet apex vs intensity at Byonic RT')
        r, p = corr(x, y)
        if r is not None:
            ax.text(0.05, 0.95, f'r = {r:.3f}', transform=ax.transAxes, fontsize=10, verticalalignment='top')

    x = df_log.get('comet_total_area_log10')
    y2 = df_log.get('comet_apex_intensity_log10')
    if x is not None and y2 is not None and x.notna().any() and y2.notna().any():
        ax = axes[1, 0]
        ax.scatter(x, y2, alpha=0.6, s=20)
        ax.set_xlabel(xlabel_ta)
        ax.set_ylabel('Comet apex_intensity (log10)')
        ax.set_title('Comet total_area vs apex (internal consistency)')
        r, p = corr(x, y2)
        if r is not None:
            ax.text(0.05, 0.95, f'r = {r:.3f}', transform=ax.transAxes, fontsize=10, verticalalignment='top')

    y3 = df_log.get('byonic_window_sum_log10')
    x = df_log.get('comet_total_area_log10')
    if x is not None and y3 is not None and x.notna().any() and y3.notna().any():
        ax = axes[1, 1]
        ax.scatter(x, y3, alpha=0.6, s=20)
        ax.set_xlabel(xlabel_ta)
        ax.set_ylabel(f'Byonic window sum {args.window_sec}s (log10)')
        ax.set_title('Comet total_area vs Byonic window-sum (area-like)')
        r, p = corr(x, y3)
        if r is not None:
            ax.text(0.05, 0.95, f'r = {r:.3f}', transform=ax.transAxes, fontsize=10, verticalalignment='top')

    plt.tight_layout()
    out_plot = args.output if not args.output.endswith('.csv') else args.output.replace('.csv', '')
    if out_plot.endswith('.csv'):
        out_plot = out_plot[:-4]
    plt.savefig(out_plot + '_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved scatter plot to {out_plot}_scatter.png")

if __name__ == '__main__':
    main()
