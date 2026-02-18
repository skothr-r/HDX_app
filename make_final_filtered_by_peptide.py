#!/usr/bin/env python3
"""
Create final_filtered_peptides_by_peptide.csv: one row per (plain_peptide, charge, modifications) from final_filtered_peptides.csv.

Each row is defined by sequence (plain_peptide), charge, mz, and modifications. For each group,
keeps the best PSM (min percolator_qvalue, then max xcorr) as the representative row and adds:
n_psms, total_area_sum, representative RT (median apex or MS1 RT), and amino acid coverage from
fragmentation (union of residue positions covered by matched fragment ions across all PSMs).

Usage:
  python make_final_filtered_by_peptide.py results_v50/final_filtered_peptides.csv
  python make_final_filtered_by_peptide.py results_v50/final_filtered_peptides.csv -o results_v50/final_filtered_peptides_by_peptide.csv
"""

import argparse
import os
import re
import pandas as pd

FRAGMENT_LABEL_RE = re.compile(r'^([cyz])(\d+)(?:_(\d+))?$', re.IGNORECASE)


def _parse_fragment_label(label):
    """Parse fragment ion label. Returns (type, start, end) or None. E.g. c4 -> (c,4,4), z1_6 -> (z,1,6)."""
    m = FRAGMENT_LABEL_RE.match(str(label).strip())
    if not m:
        return None
    typ = m.group(1).lower()
    n1 = int(m.group(2))
    n2 = int(m.group(3)) if m.group(3) else n1
    if n1 > n2:
        return None
    return (typ, n1, n2)


def _fragment_to_positions(parsed, L):
    """Return set of 1-based residue positions covered by this fragment. L = peptide length."""
    if not parsed or L <= 0:
        return set()
    typ, start, end = parsed
    if typ == 'c':
        return set(range(1, min(end, L) + 1))
    if typ == 'y':
        return set(range(max(1, L - end + 1), L + 1))
    if typ == 'z':
        return set(range(start, min(end, L) + 1))
    return set()


# Same naming convention as chromatogram extraction plots (format_title in extract_ms1_chromatograms.py)
_MOD_MAP = {
    '15.9949': 'Ox',
    '15.994900': 'Ox',
    '57.0215': 'Carbamidomethyl',
    '57.021500': 'Carbamidomethyl',
    '57.021464': 'Carbamidomethyl',
    '0.9840': 'Deamidated',
    '0.984000': 'Deamidated',
}


def _modifications_readable(mod_str):
    """Format modifications like chromatogram plots: 'none' or 'Ox@5', 'Carbamidomethyl@16'."""
    if pd.isna(mod_str) or not str(mod_str).strip() or str(mod_str).strip() in ('-', 'nan'):
        return 'none'
    s = str(mod_str).strip()
    mod_parts = []
    for mod_entry in s.split(','):
        mod_entry = mod_entry.strip()
        if '_' in mod_entry:
            parts = mod_entry.split('_')
            if len(parts) >= 3:
                pos = parts[0]
                mass_raw = parts[2].split(')')[0]
                mod_name = _MOD_MAP.get(mass_raw)
                if mod_name is None:
                    try:
                        mass_rounded = f"{float(mass_raw):.4f}"
                        mod_name = _MOD_MAP.get(mass_rounded, f"+{mass_raw}")
                    except (ValueError, TypeError):
                        mod_name = f"+{mass_raw}"
                mod_parts.append(f"{mod_name}@{pos}")
    if mod_parts:
        return ','.join(mod_parts)
    return s


def _positions_to_coverage_str(positions):
    """Format set of positions as merged intervals e.g. '1-4,6-9'. Empty -> ''."""
    if not positions:
        return ''
    sorted_pos = sorted(positions)
    intervals = []
    lo, hi = sorted_pos[0], sorted_pos[0]
    for p in sorted_pos[1:]:
        if p == hi + 1:
            hi = p
        else:
            intervals.append((lo, hi))
            lo, hi = p, p
    intervals.append((lo, hi))
    return ','.join(f'{a}-{b}' if a != b else str(a) for a, b in intervals)


def _row_coverage_positions(row, mfi_col='matched fragment ions', seq_col='plain_peptide'):
    """From one row, return set of 1-based residue positions covered by matched fragment ions."""
    seq = str(row.get(seq_col, '')).strip()
    L = len(seq)
    if L == 0:
        return set()
    mfi = row.get(mfi_col)
    if pd.isna(mfi) or not str(mfi).strip():
        return set()
    positions = set()
    for part in str(mfi).split(','):
        part = part.strip()
        if not part:
            continue
        parsed = _parse_fragment_label(part)
        if parsed:
            positions |= _fragment_to_positions(parsed, L)
    return positions


def _read_csv(path):
    with open(path, 'r') as f:
        first = f.readline()
    skip = 1 if 'CometVersion' in first else 0
    return pd.read_csv(path, sep=',', skiprows=skip, engine='python', quotechar='"', on_bad_lines='warn')


def main():
    parser = argparse.ArgumentParser(description='Collapse final_filtered_peptides.csv to one row per peptide (per charge).')
    parser.add_argument('csv', help='Path to final_filtered_peptides.csv')
    parser.add_argument('-o', '--output', default=None, help='Output path (default: same dir, final_filtered_peptides_by_peptide.csv)')
    parser.add_argument('--tsv', action='store_true', help='Also write a TSV version (same basename, .tsv)')
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        parser.error(f"CSV not found: {args.csv}")

    df = _read_csv(args.csv)
    if 'plain_peptide' not in df.columns or 'charge' not in df.columns or 'modifications' not in df.columns:
        parser.error("CSV must contain 'plain_peptide', 'charge', and 'modifications'")
    df['_mods'] = df['modifications'].fillna('-').astype(str).str.strip().replace('', '-')

    # Best PSM per (plain_peptide, charge, modifications): min percolator_qvalue, then max xcorr
    qval_col = 'percolator_qvalue'
    xcorr_col = 'xcorr'
    df['_qval'] = pd.to_numeric(df[qval_col], errors='coerce').fillna(1.0) if qval_col in df.columns else 1.0
    df['_xcorr'] = pd.to_numeric(df[xcorr_col], errors='coerce').fillna(0.0) if xcorr_col in df.columns else 0.0
    df_sorted = df.sort_values(['plain_peptide', 'charge', '_mods', '_qval', '_xcorr'], ascending=[True, True, True, True, False])
    df_best = df_sorted.drop_duplicates(subset=['plain_peptide', 'charge', '_mods'], keep='first').drop(columns=['_qval', '_xcorr'], errors='ignore')
    df_best = df_best.drop(columns=['modifications'], errors='ignore').rename(columns={'_mods': 'modifications'})

    # n_psms, total_area_sum, and representative RT per (plain_peptide, charge, modifications)
    agg = df.groupby(['plain_peptide', 'charge', '_mods']).agg(
        n_psms=('plain_peptide', 'count'),
    ).reset_index()
    if 'total_area' in df.columns:
        area = df.groupby(['plain_peptide', 'charge', '_mods'])['total_area'].apply(lambda x: pd.to_numeric(x, errors='coerce').sum()).reset_index()
        area = area.rename(columns={'total_area': 'total_area_sum'})
        agg = agg.merge(area, on=['plain_peptide', 'charge', '_mods'], how='left')
    else:
        agg['total_area_sum'] = None
    # Representative RT: median apex_rt if available, else median MS1_retention_time_sec
    rt_sec = pd.to_numeric(df['MS1_retention_time_sec'], errors='coerce')
    apex = pd.to_numeric(df['apex_rt'], errors='coerce') if 'apex_rt' in df.columns else pd.Series(dtype=float)
    use_apex = apex.notna() & (apex > 0)
    df['_repr_rt'] = apex.where(use_apex).fillna(rt_sec)
    repr_rt = df.groupby(['plain_peptide', 'charge', '_mods'])['_repr_rt'].median().reset_index()
    repr_rt = repr_rt.rename(columns={'_repr_rt': 'representative_rt_sec'})
    agg = agg.merge(repr_rt, on=['plain_peptide', 'charge', '_mods'], how='left')

    # Amino acid coverage from fragmentation: union of residue positions across all PSMs in group
    mfi_col = 'matched fragment ions'
    if mfi_col in df.columns:
        df['_cov_pos'] = df.apply(lambda r: _row_coverage_positions(r, mfi_col=mfi_col, seq_col='plain_peptide'), axis=1)
        union_pos = df.groupby(['plain_peptide', 'charge', '_mods'])['_cov_pos'].agg(lambda s: set().union(*s)).reset_index()
        union_pos['coverage_positions'] = union_pos['_cov_pos'].apply(_positions_to_coverage_str)
        union_pos['coverage_n_residues'] = union_pos['_cov_pos'].apply(len)
        union_pos = union_pos.drop(columns=['_cov_pos'])
        agg = agg.merge(union_pos, on=['plain_peptide', 'charge', '_mods'], how='left')
    else:
        agg['coverage_positions'] = ''
        agg['coverage_n_residues'] = 0
    agg = agg.rename(columns={'_mods': 'modifications'})

    # Merge representative row with aggregates (one merge only)
    by_pep = df_best.merge(agg, on=['plain_peptide', 'charge', 'modifications'], how='left')
    # Use representative RT as the observed RT (overwrite MS1 RT columns)
    if 'representative_rt_sec' in by_pep.columns:
        repr_sec = by_pep['representative_rt_sec']
        by_pep['MS1_retention_time_sec'] = repr_sec
        if 'MS1_retention_time_min' in by_pep.columns:
            by_pep['MS1_retention_time_min'] = repr_sec / 60.0

    # Human-readable modifications and column 4 = modifications (right after sequence, charge)
    if 'modifications' in by_pep.columns:
        by_pep['modifications'] = by_pep['modifications'].apply(_modifications_readable)
    # Order: sequence_positions, plain_peptide, charge, modifications (col 4), then n_psms, total_area_sum, then rest
    lead = ['sequence_positions', 'plain_peptide', 'charge', 'modifications']
    after_mods = ['n_psms', 'total_area_sum']
    rest = [c for c in by_pep.columns if c not in lead and c not in after_mods]
    by_pep = by_pep[[c for c in lead if c in by_pep.columns] + [c for c in after_mods if c in by_pep.columns] + rest]

    # representative_rt_sec and coverage after MS1 RT cols
    coverage_cols = ['coverage_positions', 'coverage_n_residues']
    cols = list(by_pep.columns)
    if 'MS1_retention_time_min' in cols:
        i = cols.index('MS1_retention_time_min') + 1
        if 'representative_rt_sec' in by_pep.columns and 'representative_rt_sec' not in cols[:i]:
            cols.insert(i, 'representative_rt_sec')
            i += 1
        for c in coverage_cols:
            if c in by_pep.columns and c not in cols[:i]:
                cols.insert(i, c)
                i += 1
    by_pep = by_pep[[c for c in cols if c in by_pep.columns]]

    out = args.output
    if out is None:
        out = os.path.join(os.path.dirname(os.path.abspath(args.csv)), 'final_filtered_peptides_by_peptide.csv')
    os.makedirs(os.path.dirname(os.path.abspath(out)) or '.', exist_ok=True)
    by_pep.to_csv(out, index=False)
    print(f"Wrote {len(by_pep)} rows (one per peptide: sequence + charge + modifications) to {os.path.abspath(out)}")
    if args.tsv:
        tsv_path = os.path.splitext(out)[0] + '.tsv'
        by_pep.to_csv(tsv_path, sep='\t', index=False)
        print(f"  Also wrote TSV: {os.path.abspath(tsv_path)}")
    print(f"  Input PSMs: {len(df)}  Unique peptides: {len(by_pep)}")


if __name__ == '__main__':
    main()
