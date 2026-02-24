#!/usr/bin/env python3
"""
Step 8: Assess peptide sequence coverage and add valuable_sequence + protect_peptide columns.

Identifies protein positions where a significant single-AA overhang is unique to one
peptide (i.e., no other peptide covers that protein position). Peptides that cover any
such unique position get valuable_sequence=1 and protect_peptide=True; others get 0/False.
protect_peptide=True prioritizes peptides for channel assignment and prevents them from
being dropped during sequence assignment.

Output: ..._prefilter_extraction_envelope_significance_sequence.csv

When FASTA is provided, creates unique peptides grid plots.

Usage:
  python compare_unique_peptides_sequence.py --fasta data/protein.fasta
  python compare_unique_peptides_sequence.py --fasta protein.fasta --input significance.csv --protein-id sp|P12345
"""

import argparse
import os
import re
import sys
from collections import defaultdict

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

DEFAULT_INPUT = os.path.join(_PROJECT_ROOT, 'data', 'comet_frags_perc_openMS_prefilter_extraction_envelope_significance.csv')
DEFAULT_FASTA = os.path.join(_PROJECT_ROOT, 'data', 'Ube2D3.fasta')


def _parse_protein_positions(val, seq_start=None):
    """Parse overhang string to set of protein positions (1-based)."""
    pos_set = set()
    if val is None or (isinstance(val, float) and str(val) == 'nan') or not str(val).strip():
        return pos_set
    for part in str(val).strip().split(','):
        part = part.strip()
        if not part or not part[-1].isalpha() or not part[:-1].replace('.', '').replace('-', '').replace(' ', '').isdigit():
            continue
        try:
            pos = int(float(part[:-1]))
        except (TypeError, ValueError):
            continue
        if seq_start is not None and seq_start > 0 and pos <= 1000:  # peptide-relative
            pos = seq_start + pos - 1
        pos_set.add(pos)
    return pos_set


def _first_nonempty_str(row, candidates):
    """Return first non-empty/non-NaN string from candidate columns."""
    for c in candidates:
        v = row.get(c)
        if v is None:
            continue
        try:
            import pandas as pd  # local import
            if pd.isna(v):
                continue
        except Exception:
            pass
        s = str(v).strip()
        if s and s.lower() != 'nan':
            return s
    return ''


def _load_fasta(fasta_path, protein_id=None):
    """Load protein sequence from FASTA. Returns (seq, id) or (None, None)."""
    if not os.path.exists(fasta_path):
        return None, None
    seq = []
    ident = None
    with open(fasta_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if seq and (protein_id is None or ident == protein_id):
                    return ''.join(seq), ident
                ident = line[1:].split()[0] if line[1:] else ''
                seq = []
                if protein_id and ident == protein_id:
                    pass
            elif line and not line.startswith('>'):
                seq.append(line)
    if seq and (protein_id is None or ident == protein_id):
        return ''.join(seq), ident
    return (''.join(seq), ident) if seq else (None, None)


def _sorted_unique_tokens(values, sep=','):
    out = set()
    for v in values:
        if v is None:
            continue
        s = str(v).strip()
        if not s or s.lower() == 'nan':
            continue
        for tok in s.split(sep):
            t = tok.strip()
            if t:
                out.add(t)
    return sorted(out)


def main():
    ap = argparse.ArgumentParser(description='Step 8: Assess sequence coverage and add valuable_sequence column.')
    ap.add_argument('--input', '-i', default=DEFAULT_INPUT, help=f'Input CSV from Step 7 (default: {os.path.basename(DEFAULT_INPUT)})')
    ap.add_argument('--output', '-o', default=None, help='Output CSV path')
    ap.add_argument('--output-dir', default=None, help='Output directory for CSV')
    ap.add_argument('--sequence-coverage-dir', default=None,
                    help='Directory for unique peptides plots (default: output-dir/sequence_coverage)')
    ap.add_argument('--fasta', '-f', default=DEFAULT_FASTA, help=f'FASTA file for unique peptides plots (default: {os.path.basename(DEFAULT_FASTA)})')
    ap.add_argument('--protein-id', default=None, help='Protein ID in FASTA (default: first sequence)')
    ap.add_argument('--debug', action='store_true', help='Print extra debug messages')
    args = ap.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input CSV not found: {args.input}")
        sys.exit(1)

    print("[Step 8] Loading input CSV...", flush=True)
    try:
        import pandas as pd
    except ImportError:
        print("Error: pandas required.")
        sys.exit(1)

    input_dir = os.path.dirname(os.path.abspath(args.input))
    base = os.path.splitext(os.path.basename(args.input))[0]
    out_dir = args.output_dir or input_dir
    os.makedirs(out_dir, exist_ok=True)
    seq_cov_dir = args.sequence_coverage_dir or (os.path.join(out_dir, 'sequence_coverage') if out_dir else out_dir)

    if args.output:
        out_csv = args.output
    else:
        out_csv = os.path.join(out_dir, f'{base}_sequence.csv')

    try:
        with open(args.input, 'r') as f:
            first = f.readline()
        skip = 1 if 'CometVersion' in first else 0
    except Exception:
        skip = 0
    df = pd.read_csv(args.input, sep=',', skiprows=skip, engine='python', quotechar='"', on_bad_lines='warn')
    print(f"[Step 8] Loaded {len(df)} rows from {os.path.basename(args.input)}", flush=True)

    # Normalize legacy column naming from earlier runs.
    if 'valuable_peptide' in df.columns:
        if 'valuable_sequence' in df.columns:
            legacy_vals = pd.to_numeric(df['valuable_peptide'], errors='coerce').fillna(0).astype(int)
            current_vals = pd.to_numeric(df['valuable_sequence'], errors='coerce').fillna(0).astype(int)
            df['valuable_sequence'] = (legacy_vals.gt(0) | current_vals.gt(0)).astype(int)
            df = df.drop(columns=['valuable_peptide'])
        else:
            df = df.rename(columns={'valuable_peptide': 'valuable_sequence'})

    col_overhangs_prot = 'significant_single_aa_overhangs_protein_positions' if 'significant_single_aa_overhangs_protein_positions' in df.columns else None
    if col_overhangs_prot is None and 'single_aa_overhangs_protein_positions' in df.columns:
        col_overhangs_prot = 'single_aa_overhangs_protein_positions'
    col_overhangs = 'significant_single_aa_overhangs' if 'significant_single_aa_overhangs' in df.columns else 'single_aa_overhangs'
    col_seq_start = 'sequence_start_pos' if 'sequence_start_pos' in df.columns else 'sequence_positions'
    if col_overhangs_prot is not None and col_overhangs_prot not in df.columns:
        col_overhangs_prot = None

    # Build per-position coverage: position -> count of peptides covering it
    position_coverage = defaultdict(int)
    peptide_to_positions = {}  # idx -> set of protein positions

    for idx in df.index:
        row = df.loc[idx]
        seq_start = row.get(col_seq_start)
        if pd.notna(seq_start) and str(seq_start).strip():
            try:
                s = str(seq_start).strip()
                if '-' in s:
                    seq_start = int(s.split('-')[0])
                elif ',' in s:
                    seq_start = int(s.split(',')[0].strip())
                else:
                    seq_start = int(s)
            except (ValueError, TypeError):
                seq_start = None
        else:
            seq_start = None

        positions = set()
        if col_overhangs_prot and col_overhangs_prot in df.columns:
            val = row.get(col_overhangs_prot)
            positions = _parse_protein_positions(val, seq_start=None)
        if not positions and col_overhangs in df.columns:
            val = row.get(col_overhangs)
            positions = _parse_protein_positions(val, seq_start=seq_start)
        peptide_to_positions[idx] = positions
        for p in positions:
            position_coverage[p] += 1

    unique_overhang_positions = {p for p, c in position_coverage.items() if c == 1}
    if args.debug:
        print(f"[Step 8] DEBUG: {len(position_coverage)} positions with overhang coverage; {len(unique_overhang_positions)} unique positions (covered by one peptide)", flush=True)
        if position_coverage:
            cov_vals = sorted(position_coverage.values(), reverse=True)
            print(f"[Step 8] DEBUG: Coverage range: min={min(cov_vals)}, max={max(cov_vals)}, median={cov_vals[len(cov_vals)//2]}", flush=True)

    # valuable_sequence = 1 if peptide covers any unique overhang position
    # protect_peptide = True/False for channel assignment priority and to avoid dropping during sequence assignment
    df['valuable_sequence'] = 0
    df['protect_peptide'] = False
    for idx in df.index:
        positions = peptide_to_positions.get(idx, set())
        if positions & unique_overhang_positions:
            df.at[idx, 'valuable_sequence'] = 1
            df.at[idx, 'protect_peptide'] = True

    n_valuable = (df['valuable_sequence'] == 1).sum()
    print(f"[Step 8] Unique overhang positions (covered by one peptide): {len(unique_overhang_positions)}", flush=True)
    print(f"[Step 8] Peptides with protect_peptide=True (valuable_sequence=1): {n_valuable} / {len(df)}", flush=True)
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
        if old in df.columns:
            if new in df.columns:
                df[new] = df[old].where(pd.notna(df[old]), df[new])
                df = df.drop(columns=[old])
            else:
                df = df.rename(columns={old: new})

    # Preserve shared workflow row organization across downstream tabs.
    pos_col = 'protein_position' if 'protein_position' in df.columns else ('sequence_positions' if 'sequence_positions' in df.columns else None)
    seq_col = 'peptide_sequence' if 'peptide_sequence' in df.columns else ('plain_peptide' if 'plain_peptide' in df.columns else None)
    rt_col = 'MS1_RT_sec' if 'MS1_RT_sec' in df.columns else ('MS1_retention_time_sec' if 'MS1_retention_time_sec' in df.columns else None)
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
        sl = df.apply(_start_len, axis=1, result_type='expand')
        df['_sort_start'] = sl[0]
        df['_sort_len'] = sl[1]
        df['_sort_charge'] = pd.to_numeric(df.get('charge'), errors='coerce').fillna(999999)
        df['_sort_rt'] = pd.to_numeric(df.get(rt_col), errors='coerce').fillna(999999.0) if rt_col else 999999.0
        df = df.sort_values(by=['_sort_start', '_sort_len', '_sort_charge', '_sort_rt'], ascending=[True, True, True, True], kind='mergesort')
        df = df.drop(columns=['_sort_start', '_sort_len', '_sort_charge', '_sort_rt'])

    # Final ordering rule: RT columns explicitly in seconds always go last.
    rt_sec_cols = [
        c for c in df.columns
        if (('RT' in c or 'retention_time' in c) and c.endswith('_sec'))
    ]
    if rt_sec_cols:
        non_rt_sec_cols = [c for c in df.columns if c not in rt_sec_cols]
        df = df[non_rt_sec_cols + rt_sec_cols]

    print("[Step 8] Writing output CSV...", flush=True)
    df.to_csv(out_csv, index=False)
    print(f"[Step 8] Output saved to: {os.path.abspath(out_csv)}", flush=True)

    # Write compact unique-peptide table (one row per sequence+charge+modifications),
    # explicitly collapsing scan/row-specific duplicates.
    pep_candidates = ['peptide_sequence', 'plain_peptide', 'peptide', 'sequence']
    df_u = df.copy()
    df_u['_pep_seq'] = df_u.apply(lambda r: _first_nonempty_str(r, pep_candidates), axis=1)
    df_u['_charge'] = pd.to_numeric(df_u.get('charge'), errors='coerce')
    df_u['_mods'] = df_u.get('modifications', pd.Series(['-'] * len(df_u))).astype(str).str.strip()
    df_u.loc[df_u['_mods'].isin(['', 'nan', 'None']), '_mods'] = '-'
    df_u = df_u[df_u['_pep_seq'].astype(str).str.strip() != ''].copy()

    overhang_col = (
        'significant_single_aa_overhangs_protein_positions'
        if 'significant_single_aa_overhangs_protein_positions' in df_u.columns
        else ('single_aa_overhangs_protein_positions' if 'single_aa_overhangs_protein_positions' in df_u.columns else None)
    )
    pairs_col = (
        'single_aa_overhang_fragment_pairs'
        if 'single_aa_overhang_fragment_pairs' in df_u.columns
        else ('significant_fragment_pairs' if 'significant_fragment_pairs' in df_u.columns else None)
    )

    # Keep all original columns: pick one representative row per sequence+charge+mods,
    # then add n_rows_collapsed and aggregated overhang fields.
    unique_rows = []
    q_candidates = ['percolator_qvalue', 'q_value', 'qvalue', 'q-val', 'qval']
    q_col = next((c for c in q_candidates if c in df_u.columns), None)
    for (pep, ch, mods_key), g in df_u.groupby(['_pep_seq', '_charge', '_mods'], dropna=False, sort=True):
        if q_col:
            qv = pd.to_numeric(g[q_col], errors='coerce')
            best_idx = qv.idxmin() if qv.notna().any() else g.index[0]
        else:
            best_idx = g.index[0]
        row_out = g.loc[best_idx].copy()
        overhang_positions = _sorted_unique_tokens(g[overhang_col]) if overhang_col else []
        overhang_pairs = _sorted_unique_tokens(g[pairs_col]) if pairs_col else []
        row_out['n_rows_collapsed'] = int(len(g))
        row_out['valuable_sequence'] = int(pd.to_numeric(g.get('valuable_sequence', 0), errors='coerce').fillna(0).astype(int).max() > 0)
        row_out['protect_peptide'] = bool(pd.Series(g.get('protect_peptide', False)).astype(bool).max())
        row_out['modifications'] = str(mods_key) if str(mods_key).strip() else '-'
        # Normalize key fields to the grouping key.
        if 'peptide_sequence' in row_out.index:
            row_out['peptide_sequence'] = str(pep)
        elif 'plain_peptide' in row_out.index:
            row_out['plain_peptide'] = str(pep)
        if 'charge' in row_out.index:
            row_out['charge'] = (int(ch) if pd.notna(ch) else '')
        if overhang_col:
            row_out[overhang_col] = ','.join(overhang_positions)
        if pairs_col:
            row_out[pairs_col] = ','.join(overhang_pairs)
        unique_rows.append(row_out)

    unique_out_csv = os.path.join(out_dir, f'{base}_sequence_unique_peptides.csv')
    df_unique = pd.DataFrame(unique_rows)
    helper_cols = [c for c in ['_pep_seq', '_charge', '_mods'] if c in df_unique.columns]
    if helper_cols:
        df_unique = df_unique.drop(columns=helper_cols)
    if 'n_rows_collapsed' in df_unique.columns:
        cols = [c for c in df_unique.columns if c != 'n_rows_collapsed'] + ['n_rows_collapsed']
        df_unique = df_unique[cols]
    df_unique.to_csv(unique_out_csv, index=False)
    print(f"[Step 8] Unique sequence+charge+mods CSV saved to: {os.path.abspath(unique_out_csv)} ({len(df_unique)} rows)", flush=True)

    # Unique peptides plots if FASTA provided
    if args.fasta and os.path.exists(args.fasta):
        print(f"[Step 8] Loading FASTA: {args.fasta}", flush=True)
        protein_sequence, pid = _load_fasta(args.fasta, args.protein_id)
        if protein_sequence:
            try:
                from visualization.combined import create_accepted_peptides_protein_grid
                print("[Step 8] Building peptides list from CSV (one entry per unique seq+charge+mods)...", flush=True)
                peptides = []
                seen = set()
                for idx in df.index:
                    row = df.loc[idx]
                    peptide = _first_nonempty_str(row, ['plain_peptide', 'peptide_sequence', 'peptide', 'sequence'])
                    if not peptide:
                        continue
                    clean = re.sub(r'\[.*?\]', '', peptide)
                    charge = int(row.get('charge', 0)) if pd.notna(row.get('charge')) else 0
                    mods = str(row.get('modifications', '-')).strip() or '-'
                    key = (clean, charge, mods)
                    if key in seen:
                        continue
                    seen.add(key)
                    seq_start = row.get(col_seq_start)
                    if pd.notna(seq_start):
                        try:
                            s = str(seq_start).strip()
                            start = int(s.split('-')[0].split(',')[0])
                        except (ValueError, TypeError):
                            start = 0
                    else:
                        start = 0
                    if start <= 0 and protein_sequence:
                        up = protein_sequence.upper()
                        i = up.find(clean.upper())
                        start = i + 1 if i >= 0 else 1
                    end = start + len(clean) - 1 if start > 0 else 0
                    p = {
                        'plain_peptide': peptide,
                        'peptide_seq': peptide,
                        'start': start,
                        'end': end,
                        'charge': charge,
                        'modifications': mods,
                        'total_area': row.get('total_area'),
                        'significant_single_aa_overhangs': row.get(col_overhangs),
                        'significant_single_aa_overhangs_protein_positions': ','.join(map(str, sorted(peptide_to_positions.get(idx, set())))) if peptide_to_positions.get(idx) else '',
                        'significant_fragment_pairs': row.get('significant_fragment_pairs', ''),
                    }
                    peptides.append(p)
                print(f"[Step 8] Built {len(peptides)} peptide entries for grid (deduped by seq+charge+mods)", flush=True)
                if args.debug and peptides:
                    p0 = peptides[0]
                    print(f"[Step 8] DEBUG: First peptide: {p0.get('plain_peptide', '')[:40]}... start={p0.get('start')} end={p0.get('end')}", flush=True)
                if peptides:
                    os.makedirs(seq_cov_dir, exist_ok=True)
                    unique_output = os.path.join(seq_cov_dir, f'{base}_unique_peptides.png')
                    print(f"[Step 8] Creating unique peptides grid ({len(peptides)} entries -> unique sequences)...", flush=True)
                    create_accepted_peptides_protein_grid(
                        peptides, protein_sequence, pid or 'protein', len(protein_sequence),
                        unique_output, seq_cov_dir
                    )
                    print(f"[Step 8] Unique peptides plot saved to: {unique_output}", flush=True)
                else:
                    print("[Step 8] No peptide rows available for plotting (check peptide columns).", flush=True)
            except Exception as e:
                print(f"[Step 8] Warning: Could not create unique peptides plot: {e}")


if __name__ == '__main__':
    main()
