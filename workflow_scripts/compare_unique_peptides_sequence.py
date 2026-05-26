#!/usr/bin/env python3
"""
Step 8: Assess peptide sequence coverage and add valuable_sequence + protect_peptide columns.

Identifies protein positions where a significant single-AA overhang is unique to one
peptide (i.e., no other peptide covers that protein position). Peptides that cover any
such unique position get valuable_sequence=1 and protect_peptide=True; others get 0/False.
protect_peptide=True prioritizes peptides for channel assignment and prevents them from
being dropped during sequence assignment.

Output (each table is written with identical columns, sorted four ways):
  - ..._sequence.csv (primary = by protein position; also _by_total_area, _by_collection_min_rt, _by_overhang_pair_count)
  - ..._sequence_unique_seq_charge.csv (+ sort variants)
  - ..._sequence_unique_peptides.csv (+ sort variants)

When FASTA is provided and --plots or --plots-only is set, creates unique peptides grid plots.

Usage:
  python compare_unique_peptides_sequence.py --fasta data/protein.fasta
  python compare_unique_peptides_sequence.py --fasta protein.fasta --input significance.csv --protein-id sp|P12345
"""

import argparse
import os
import re
import sys
from collections import defaultdict

import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None  # type: ignore

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

try:
    from filter_significant_frags_summed_ms2 import (
        _ion_name_to_comet_format,
        _significant_pairs_and_overhangs_from_ions,
    )
except Exception:
    def _ion_name_to_comet_format(label):  # type: ignore
        return str(label or '').strip()

    def _significant_pairs_and_overhangs_from_ions(sig_ions, peptide):  # type: ignore
        return '', ''

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


def _split_csv_field(val) -> list[str]:
    if val is None:
        return []
    try:
        import pandas as pd  # local import
        if pd.isna(val):
            return []
    except Exception:
        pass
    return [t.strip() for t in str(val).split(',') if str(t).strip()]


def _join_csv_field(parts: list[str]) -> str:
    return ','.join(parts) if parts else ''


def _parse_allowed_ion_types(raw: str) -> set[str]:
    return {str(x).strip().lower() for x in str(raw or '').split(',') if str(x).strip()}


def _parse_allowed_charges(raw: str) -> set[int] | None:
    allowed: set[int] = set()
    for tok in str(raw or '').split(','):
        tok = str(tok).strip()
        if not tok:
            continue
        try:
            z = int(float(tok))
        except (ValueError, TypeError):
            continue
        if z > 0:
            allowed.add(z)
    return allowed or None


def _parse_allowed_fragment_charges(raw: str) -> set[int] | None:
    return _parse_allowed_charges(raw)


def discover_fragment_charges(df: 'pd.DataFrame') -> list[int]:
    """Fragment ion charges (+) present in fragment label columns (implicit +1 when untagged)."""
    import pandas as pd

    if df is None or df.empty:
        return []
    found: set[int] = set()
    label_cols = [
        c for c in (
            'significant_frags', 'significant_frags_base', 'matched_frags', 'fragment_fit_key',
        )
        if c in df.columns
    ]
    for col in label_cols:
        for v in df[col].dropna().astype(str):
            for tok in v.split(','):
                tok = tok.strip().strip('"')
                if tok:
                    found.add(_frag_charge_from_label(tok))
    if 'fragment_charge' in df.columns:
        for v in pd.to_numeric(df['fragment_charge'], errors='coerce').dropna():
            z = int(v)
            if z > 0:
                found.add(z)
    return sorted(found)


_PRECURSOR_CHARGE_COLUMNS = (
    'precursor_charge',
    'charge',
    'charge_state',
    'precursor_charge_state',
    'z',
    'precursor_charge_from_name',
)


def discover_precursor_charges(df: 'pd.DataFrame') -> list[int]:
    """Union of precursor charges from all charge-related columns in the table."""
    import pandas as pd

    if df is None or df.empty:
        return []
    charges: set[int] = set()
    for col in _PRECURSOR_CHARGE_COLUMNS:
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors='coerce').dropna()
        charges.update(int(v) for v in vals if int(v) > 0)
    return sorted(charges)


def _frag_charge_from_label(label: str) -> int:
    """Extract fragment charge from labels like 'c5_2+' or 'x11 2+'. Default +1."""
    s = str(label or '').strip()
    if not s:
        return 1
    m = re.search(r'(\d+)\s*\+\s*$', s)
    if m:
        try:
            z = int(m.group(1))
            return z if z > 0 else 1
        except (ValueError, TypeError):
            return 1
    m2 = re.search(r'[\s_](\d+)\+$', s)
    if m2:
        try:
            z = int(m2.group(1))
            return z if z > 0 else 1
        except (ValueError, TypeError):
            return 1
    return 1


def _ion_family_from_base_name(base_name: str) -> str:
    b = str(base_name or '').strip().lower()
    if not b:
        return ''
    if b.startswith('z1_'):
        return 'z1'
    m = re.match(r'^([abcxyz])', b)
    return m.group(1) if m else ''


def _peptide_seq_from_row(row) -> str:
    for c in ('peptide_sequence', 'plain_peptide', 'peptide', 'sequence'):
        if c in row.index:
            v = row.get(c)
            if v is not None and str(v).strip() and str(v).strip().lower() != 'nan':
                return str(v).strip()
    return ''


def _sequence_start_from_row(row) -> int | None:
    for c in ('sequence_start_pos', 'sequence_positions'):
        if c not in row.index:
            continue
        val = row.get(c)
        if val is None:
            continue
        try:
            import pandas as pd  # local import
            if pd.isna(val):
                continue
        except Exception:
            pass
        s = str(val).strip()
        if not s:
            continue
        try:
            if '-' in s:
                return int(float(s.split('-')[0].strip()))
            if ',' in s:
                return int(float(s.split(',')[0].strip()))
            return int(float(s))
        except (ValueError, TypeError):
            continue
    return None


def _overhangs_to_protein_positions(overhang_str: str, peptide: str, seq_start: int | None) -> str:
    if not overhang_str:
        return ''
    if seq_start is None or seq_start <= 0:
        return overhang_str
    clean = re.sub(r'\[.*?\]', '', peptide) if peptide else ''
    parts = []
    for pos_res in overhang_str.split(','):
        pos_res = pos_res.strip()
        if not pos_res:
            continue
        if len(pos_res) >= 2 and pos_res[-1].isalpha() and pos_res[:-1].replace('.', '').isdigit():
            try:
                pos = int(float(pos_res[:-1]))
                res = pos_res[-1]
                parts.append(f'{seq_start + pos - 1}{res}')
                continue
            except (ValueError, TypeError):
                pass
        parts.append(pos_res)
    return ','.join(parts) if parts else overhang_str


def _filter_row_fragments(row, allowed_families: set[str] | None, allowed_charges: set[int] | None):
    """Keep selected ion families and fragment charges; recompute overhang fields."""
    if not allowed_families and not allowed_charges:
        return row
    peptide = _peptide_seq_from_row(row)
    seq_start = _sequence_start_from_row(row)

    bases = _split_csv_field(row.get('significant_frags_base'))
    fulls = _split_csv_field(row.get('significant_frags'))
    if not bases and fulls:
        bases = [_ion_name_to_comet_format(x) for x in fulls]
    if not bases:
        return row

    parallel_cols = [
        'significant_frags',
        'significant_frags_base',
        'significant_frags_mz',
        'significant_frags_intensities',
        'significant_frags_subclasses',
        'significant_frags_subtypes',
    ]
    lists = {c: _split_csv_field(row.get(c)) for c in parallel_cols if c in row.index}

    keep_idx = []
    for i, base in enumerate(bases):
        if allowed_families and _ion_family_from_base_name(base) not in allowed_families:
            continue
        if allowed_charges:
            full_label = fulls[i] if i < len(fulls) else base
            if _frag_charge_from_label(full_label) not in allowed_charges:
                continue
        keep_idx.append(i)

    def _pick(lst: list[str]) -> list[str]:
        if not lst:
            return []
        return [lst[i] for i in keep_idx if i < len(lst)]

    filtered_bases = _pick(bases)
    if 'significant_frags_base' in row.index:
        row['significant_frags_base'] = _join_csv_field(filtered_bases)
    if 'significant_frags' in row.index:
        row['significant_frags'] = _join_csv_field(_pick(lists.get('significant_frags', fulls)))
    for c in parallel_cols:
        if c in row.index and c not in ('significant_frags', 'significant_frags_base'):
            row[c] = _join_csv_field(_pick(lists.get(c, [])))

    pairs_str, overhang_pep = _significant_pairs_and_overhangs_from_ions(filtered_bases, peptide)
    overhang_prot = _overhangs_to_protein_positions(overhang_pep, peptide, seq_start)

    for c in (
        'single_aa_overhang_fragment_pairs',
        'significant_fragment_pairs',
    ):
        if c in row.index:
            row[c] = pairs_str
    for c in ('single_aa_overhangs', 'significant_single_aa_overhangs'):
        if c in row.index:
            row[c] = overhang_pep
    for c in (
        'single_aa_overhangs_protein_positions',
        'significant_single_aa_overhangs_protein_positions',
    ):
        if c in row.index:
            row[c] = overhang_prot
    return row


def _filter_row_fragments_by_ion_types(row, allowed_families: set[str]):
    return _filter_row_fragments(row, allowed_families, None)


def _apply_row_mask_respecting_protected(
    df: 'pd.DataFrame',
    keep_mask,
    log_label: str,
    *,
    protect_valuable: bool = False,
    protect_strong_psm: bool = False,
) -> 'pd.DataFrame':
    """Apply a boolean row mask; optionally keep protected annotation rows regardless."""
    import pandas as pd

    if df is None or df.empty:
        return df
    keep_mask = pd.Series(keep_mask, index=df.index).fillna(False)
    n_before = len(df)
    final_mask = keep_mask.copy()
    n_valuable = 0
    n_strong = 0
    if protect_valuable and 'valuable_sequence' in df.columns:
        valuable = pd.to_numeric(df['valuable_sequence'], errors='coerce').fillna(0).astype(int).eq(1)
        n_valuable = int((~final_mask & valuable).sum())
        final_mask = final_mask | valuable
    if protect_strong_psm and 'strong_psm_support' in df.columns:
        strong = pd.to_numeric(df['strong_psm_support'], errors='coerce').fillna(0).astype(int).eq(1)
        n_strong = int((~final_mask & strong).sum())
        final_mask = final_mask | strong
    out = df[final_mask].copy()
    msg = f'[Step 8] {log_label}: kept {len(out)}/{n_before} rows'
    parts = []
    if n_valuable:
        parts.append(f'{n_valuable} valuable')
    if n_strong:
        parts.append(f'{n_strong} strong_psm_support')
    if parts:
        msg += f' ({"; ".join(parts)} kept despite filter)'
    print(msg, flush=True)
    return out


def _apply_row_mask_respecting_valuable(
    df: 'pd.DataFrame',
    keep_mask,
    protect_valuable: bool,
    log_label: str,
    *,
    protect_strong_psm: bool = False,
) -> 'pd.DataFrame':
    return _apply_row_mask_respecting_protected(
        df,
        keep_mask,
        log_label,
        protect_valuable=protect_valuable,
        protect_strong_psm=protect_strong_psm,
    )


def _row_psm_count_series(df: 'pd.DataFrame') -> 'pd.Series':
    """PSM count per row: num_psms when present, else n_rows_collapsed after collapse."""
    import pandas as pd

    if df is None or df.empty:
        return pd.Series(dtype=float)
    if 'num_psms' in df.columns:
        return pd.to_numeric(df['num_psms'], errors='coerce').fillna(0)
    if 'n_rows_collapsed' in df.columns:
        return pd.to_numeric(df['n_rows_collapsed'], errors='coerce').fillna(0)
    return pd.Series(0.0, index=df.index)


def _assign_strong_psm_support_column(df: 'pd.DataFrame', min_psms: int) -> 'pd.DataFrame':
    """Set strong_psm_support=1 when row PSM count is >= min_psms (else 0)."""
    import pandas as pd

    if df is None or df.empty:
        return df
    out = df.copy()
    min_psms = max(0, int(min_psms or 0))
    counts = _row_psm_count_series(out)
    out['strong_psm_support'] = (counts >= min_psms).astype(int)
    n_strong = int(pd.to_numeric(out['strong_psm_support'], errors='coerce').fillna(0).astype(int).sum())
    print(
        f'[Step 8] strong_psm_support: {n_strong}/{len(out)} rows with >={min_psms} PSM(s)',
        flush=True,
    )
    return out


def _peptide_sequence_keys_series(df: 'pd.DataFrame') -> 'pd.Series':
    """One sequence key per row (for charge-redundancy grouping)."""
    pep_candidates = ['peptide_sequence', 'plain_peptide', 'peptide', 'sequence']
    return df.apply(lambda r: _first_nonempty_str(r, pep_candidates), axis=1)


def _filter_df_drop_z2_when_higher_charge_present(
    df: 'pd.DataFrame',
    *,
    reference_charge: int = 2,
) -> 'pd.DataFrame':
    """Drop Z=reference_charge rows when the same peptide sequence has a higher precursor charge.

    Does not honor protect_valuable: redundant lower-charge rows are always removed so
    valuable_sequence can be recomputed on the remaining set.
    """
    import pandas as pd

    if df is None or df.empty:
        return df
    if 'charge' not in df.columns:
        print(
            '[Step 8] Warning: no charge column; skipping drop-Z2-when-higher-charge filter.',
            flush=True,
        )
        return df
    ref_z = int(reference_charge)
    pep_keys = _peptide_sequence_keys_series(df)
    charges = pd.to_numeric(df['charge'], errors='coerce')
    valid = pep_keys.astype(str).str.strip().ne('') & charges.notna()
    if not valid.any():
        return df
    max_charge_by_pep = (
        pd.DataFrame({'pep': pep_keys[valid], 'z': charges[valid]})
        .groupby('pep', sort=False)['z']
        .max()
    )
    peps_with_higher = set(max_charge_by_pep.index[max_charge_by_pep > ref_z])
    if not peps_with_higher:
        print(
            f'[Step 8] Drop Z={ref_z} when higher charge present: no rows to remove.',
            flush=True,
        )
        return df
    drop_mask = pep_keys.isin(peps_with_higher) & (charges == ref_z)
    keep_mask = ~drop_mask
    return _apply_row_mask_respecting_valuable(
        df,
        keep_mask,
        False,
        f'Drop Z={ref_z} when same peptide has higher precursor charge',
    )


def _apply_precursor_charge_filter_df(
    df,
    precursor_charges_raw: str,
    protect_valuable: bool = False,
    protect_strong_psm: bool = False,
):
    allowed = _parse_allowed_charges(precursor_charges_raw)
    if not allowed:
        return df
    if 'charge' not in df.columns:
        print('[Step 8] Warning: no charge column; skipping precursor-charge row filter.', flush=True)
        return df
    import pandas as pd

    ch = pd.to_numeric(df['charge'], errors='coerce')
    keep_mask = ch.isin(list(allowed))
    label = (
        f"Precursor charges {', '.join(str(z) for z in sorted(allowed))}"
    )
    return _apply_row_mask_respecting_valuable(
        df, keep_mask, protect_valuable, label, protect_strong_psm=protect_strong_psm
    )


def _apply_fragment_filters_df(df, ion_types_raw: str, fragment_charges_raw: str):
    allowed_ions = _parse_allowed_ion_types(ion_types_raw)
    allowed_charges = _parse_allowed_fragment_charges(fragment_charges_raw)
    if not allowed_ions and not allowed_charges:
        return df
    if 'significant_frags' not in df.columns and 'significant_frags_base' not in df.columns:
        print('[Step 8] Warning: no significant_frags columns to filter fragments.', flush=True)
        return df
    if allowed_ions:
        print(f"[Step 8] Filtering fragments to ion types: {', '.join(sorted(allowed_ions))}", flush=True)
    if allowed_charges:
        print(
            f"[Step 8] Filtering fragments to charges: {', '.join(str(z) for z in sorted(allowed_charges))}",
            flush=True,
        )
    return df.apply(lambda r: _filter_row_fragments(r.copy(), allowed_ions, allowed_charges), axis=1)


def _apply_ion_type_filter_df(df, ion_types_raw: str):
    return _apply_fragment_filters_df(df, ion_types_raw, '')


def _is_decoy_protein_label(name: str) -> bool:
    s = str(name or '').strip().upper()
    if not s:
        return False
    return (
        s.startswith('DECOY_')
        or s.startswith('REV_')
        or s.endswith('_DECOY')
        or bool(re.search(r'(^|[^A-Z0-9])DECOY([^A-Z0-9]|$)', s))
    )


def _resolve_protein_column(df) -> str | None:
    for col in ('protein', 'Protein', 'protein_id', 'accession', 'protein_name'):
        if col in df.columns:
            return col
    return None


def _filter_df_by_include_proteins(
    df,
    include_proteins: str | None,
    protect_valuable: bool = False,
    protect_strong_psm: bool = False,
):
    allowed = {p.strip() for p in str(include_proteins or '').split(',') if str(p).strip()}
    if not allowed:
        return df
    col = _resolve_protein_column(df)
    if not col:
        print('[Step 8] Warning: --include-proteins set but no protein column found; ignoring filter.', flush=True)
        return df
    mask = df[col].astype(str).str.strip().isin(allowed)
    return _apply_row_mask_respecting_valuable(
        df,
        mask,
        protect_valuable,
        f'Protein filter ({col}, {len(allowed)} label(s))',
        protect_strong_psm=protect_strong_psm,
    )


def _filter_df_exclude_decoys(df, protect_valuable: bool = False, protect_strong_psm: bool = False):
    col = _resolve_protein_column(df)
    if not col:
        return df
    labels = df[col].astype(str).str.strip()
    mask = ~labels.map(_is_decoy_protein_label)
    return _apply_row_mask_respecting_valuable(
        df,
        mask,
        protect_valuable,
        f'Exclude decoys ({col})',
        protect_strong_psm=protect_strong_psm,
    )


def _has_valid_extraction_status(val) -> bool:
    if val is None:
        return False
    try:
        if pd.isna(val):
            return False
    except Exception:
        pass
    s = str(val).strip()
    return bool(s) and s.lower() not in {'nan', 'none', 'null'}


def _filter_rows_for_sequence_coverage(
    df,
    require_extraction_status: bool = True,
    accepted_only: bool = False,
    protect_valuable: bool = False,
    protect_strong_psm: bool = False,
) -> tuple:
    """Drop rows that never received extraction status / optional accepted-only subset."""
    if df is None or df.empty:
        return df, 0
    n0 = int(len(df))
    out = df
    if require_extraction_status and 'status' in out.columns:
        mask = out['status'].map(_has_valid_extraction_status)
        out = _apply_row_mask_respecting_valuable(
            out,
            mask,
            protect_valuable,
            'Require extraction status',
            protect_strong_psm=protect_strong_psm,
        )
    if accepted_only and 'status' in out.columns:
        mask = out['status'].astype(str).str.strip().str.lower() == 'accepted'
        out = _apply_row_mask_respecting_valuable(
            out,
            mask,
            protect_valuable,
            'Accepted extraction status only',
            protect_strong_psm=protect_strong_psm,
        )
    dropped = n0 - int(len(out))
    return out, dropped


def _resolve_overhang_columns(df):
    col_overhangs_prot = (
        'significant_single_aa_overhangs_protein_positions'
        if 'significant_single_aa_overhangs_protein_positions' in df.columns
        else None
    )
    if col_overhangs_prot is None and 'single_aa_overhangs_protein_positions' in df.columns:
        col_overhangs_prot = 'single_aa_overhangs_protein_positions'
    col_overhangs = (
        'significant_single_aa_overhangs'
        if 'significant_single_aa_overhangs' in df.columns
        else 'single_aa_overhangs'
    )
    col_seq_start = 'sequence_start_pos' if 'sequence_start_pos' in df.columns else 'sequence_positions'
    if col_overhangs_prot is not None and col_overhangs_prot not in df.columns:
        col_overhangs_prot = None
    return col_overhangs_prot, col_overhangs, col_seq_start


def _assign_valuable_sequence_columns(
    df: 'pd.DataFrame',
    col_overhangs_prot,
    col_overhangs,
    col_seq_start,
):
    """Set valuable_sequence / protect_peptide from unique overhang position coverage."""
    from collections import defaultdict

    peptide_to_positions = _build_peptide_position_map(df, col_overhangs_prot, col_overhangs, col_seq_start)
    position_coverage = defaultdict(int)
    for _idx, positions in peptide_to_positions.items():
        for p in positions:
            position_coverage[p] += 1
    unique_overhang_positions = {p for p, c in position_coverage.items() if c == 1}
    df = df.copy()
    df['valuable_sequence'] = 0
    df['protect_peptide'] = False
    for idx in df.index:
        positions = peptide_to_positions.get(idx, set())
        if positions & unique_overhang_positions:
            df.at[idx, 'valuable_sequence'] = 1
            df.at[idx, 'protect_peptide'] = True
    return df, peptide_to_positions, unique_overhang_positions


def _recompute_valuable_sequence_columns(
    df: 'pd.DataFrame',
    col_overhangs_prot,
    col_overhangs,
    col_seq_start,
    *,
    step_label: str = '',
    log_summary: bool = True,
) -> tuple:
    """Recompute valuable_sequence on the current row set (call after each filter step)."""
    df, peptide_to_positions, unique_overhang_positions = _assign_valuable_sequence_columns(
        df,
        col_overhangs_prot,
        col_overhangs,
        col_seq_start,
    )
    if log_summary:
        _log_valuable_sequence_summary(df, unique_overhang_positions, step_label=step_label)
    return df, peptide_to_positions, unique_overhang_positions


def _annotate_valuable_and_strong_psm_support(
    df: 'pd.DataFrame',
    col_overhangs_prot,
    col_overhangs,
    col_seq_start,
    min_strong_psm: int,
    *,
    step_label: str = 'post ion/charge filter',
) -> tuple:
    """After ion filtering: recompute valuable_sequence, then assign strong_psm_support."""
    df, peptide_to_positions, unique_overhang_positions = _recompute_valuable_sequence_columns(
        df,
        col_overhangs_prot,
        col_overhangs,
        col_seq_start,
        step_label=step_label,
    )
    df = _assign_strong_psm_support_column(df, min_strong_psm)
    return df, peptide_to_positions, unique_overhang_positions


def _log_valuable_sequence_summary(
    df: 'pd.DataFrame',
    unique_overhang_positions: set,
    *,
    step_label: str = '',
) -> None:
    import pandas as pd

    val_mask = pd.to_numeric(df['valuable_sequence'], errors='coerce').fillna(0).astype(int).eq(1)
    try:
        from workflow_scripts.extraction_status_utils import peptide_identity_counts
    except ImportError:
        from extraction_status_utils import peptide_identity_counts
    total_ids = peptide_identity_counts(df)
    valuable_ids = peptide_identity_counts(df, row_mask=val_mask)
    where = f' ({step_label})' if step_label else ''
    print(
        f'[Step 8] Unique overhang positions (one row each){where}: '
        f'{len(unique_overhang_positions)}',
        flush=True,
    )
    print(
        f"[Step 8] Rows{where}: {total_ids['rows']} | unique sequences: "
        f"{total_ids['unique_sequences']} | unique seq+charge: {total_ids['unique_seq_charge']}",
        flush=True,
    )
    print(
        f"[Step 8] Valuable{where} (valuable_sequence=1): rows "
        f"{valuable_ids['rows']}/{total_ids['rows']} | unique sequences "
        f"{valuable_ids['unique_sequences']}/{total_ids['unique_sequences']} | "
        f"unique seq+charge {valuable_ids['unique_seq_charge']}/{total_ids['unique_seq_charge']}",
        flush=True,
    )


def _build_peptide_position_map(df, col_overhangs_prot, col_overhangs, col_seq_start) -> dict:
    peptide_to_positions: dict = {}
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
    return peptide_to_positions


# First columns in sequence CSV outputs (first matching name per group is used).
_SEQUENCE_CSV_LEAD_COLUMN_GROUPS = (
    ('num_psms',),
    ('coelution_score',),
    ('theoretical_mz',),
    ('observed_mz',),
    ('protein_position', 'sequence_positions'),
    ('valuable_sequence',),
    ('strong_psm_support',),
    ('peptide_sequence', 'plain_peptide'),
    ('charge',),
    ('total_area',),
    ('collection_min_rt',),
    ('collection_max_rt',),
)

_SEQUENCE_CSV_TAIL_COLUMNS = (
    'n_mod_variants_collapsed',
    'n_rows_collapsed',
    'collapsed_scans',
    'collapsed_source_mzml',
)

# Far-right fragment list columns (after RT *_sec and collapse metadata).
_SEQUENCE_CSV_TRAILING_COLUMNS = (
    'significant_frags_mz',
    'significant_frags_intensities',
)

_SEQUENCE_CSV_OVERHANG_ANCHOR_COLUMNS = (
    'single_aa_overhangs_protein_positions',
    'significant_single_aa_overhangs_protein_positions',
)

# Immediately after the overhang protein-positions column.
_SEQUENCE_CSV_AFTER_OVERHANG_COLUMNS = (
    'perc_PEP',
    'e-value',
    'xcorr',
    'sp_score',
    'delta_cn',
)

# Absolute far-right score columns (after RT sec and fragment m/z list columns).
_SEQUENCE_CSV_FAR_RIGHT_COLUMNS = (
    'perc_qvalue',
)


def _sequence_csv_lead_columns(df: 'pd.DataFrame') -> list[str]:
    cols = list(df.columns) if df is not None else []
    lead: list[str] = []
    for group in _SEQUENCE_CSV_LEAD_COLUMN_GROUPS:
        for c in group:
            if c in cols and c not in lead:
                lead.append(c)
                break
    return lead


def _protein_position_sort_start(val) -> int:
    s = str(val or '').strip()
    if not s or s.lower() == 'nan':
        return 999999
    try:
        if '-' in s:
            return int(float(s.split('-')[0].strip()))
        if ',' in s:
            return int(float(s.split(',')[0].strip()))
        return int(float(s))
    except (ValueError, TypeError):
        return 999999


# Single-AA overhang pairs only (do not use significant_fragment_pairs — that lists all adjacent fragments).
_SEQUENCE_CSV_OVERHANG_PAIRS_COLUMNS = (
    'single_aa_overhang_fragment_pairs',
)

_SEQUENCE_CSV_OVERHANG_PROTEIN_POSITION_COLUMNS = (
    'single_aa_overhangs_protein_positions',
    'significant_single_aa_overhangs_protein_positions',
)


def _count_single_aa_overhang_pairs(val) -> int:
    s = str(val or '').strip()
    if not s or s.lower() in ('nan', 'none'):
        return 0
    return sum(1 for tok in s.split(',') if str(tok).strip())


def _overhang_pairs_count_column(df: 'pd.DataFrame') -> str | None:
    return next((c for c in _SEQUENCE_CSV_OVERHANG_PAIRS_COLUMNS if c in df.columns), None)


def _resolved_overhang_protein_position_count(row, overhang_prot_col: str | None = None) -> int:
    """Count resolved protein positions from single-AA overhang columns (unique positions)."""
    col = overhang_prot_col
    if not col:
        col = next((c for c in _SEQUENCE_CSV_OVERHANG_PROTEIN_POSITION_COLUMNS if c in row.index), None)
    seq_start = _sequence_start_from_row(row)
    if col:
        n = len(_parse_protein_positions(row.get(col), seq_start=seq_start))
        if n > 0:
            return n
    pairs_col = next((c for c in _SEQUENCE_CSV_OVERHANG_PAIRS_COLUMNS if c in row.index), None)
    if pairs_col:
        return _count_single_aa_overhang_pairs(row.get(pairs_col))
    return 0


def _filter_df_by_min_total_area(
    df: 'pd.DataFrame',
    min_total_area: float,
    protect_valuable: bool = False,
    protect_strong_psm: bool = False,
) -> 'pd.DataFrame':
    """Drop rows with total_area below the minimum threshold."""
    import pandas as pd

    min_total_area = float(min_total_area or 0.0)
    if min_total_area <= 0 or df is None or df.empty:
        return df
    if 'total_area' not in df.columns:
        print('[Step 8] Warning: no total_area column; skipping min total_area filter.', flush=True)
        return df
    ta = pd.to_numeric(df['total_area'], errors='coerce')
    keep_mask = ta >= min_total_area
    return _apply_row_mask_respecting_valuable(
        df,
        keep_mask,
        protect_valuable,
        f'Min total_area (>={min_total_area:.6g})',
        protect_strong_psm=protect_strong_psm,
    )


def _row_coelution_score_series(df: 'pd.DataFrame') -> 'pd.Series':
    """Numeric coelution_score per row; semicolon-separated values use the minimum."""
    import pandas as pd

    if df is None or df.empty or 'coelution_score' not in df.columns:
        return pd.Series(dtype=float)

    def _parse_one(val) -> float:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return float('nan')
        s = str(val).strip()
        if not s or s.lower() in ('nan', 'none'):
            return float('nan')
        if ';' in s:
            parts = []
            for tok in s.split(';'):
                tok = tok.strip()
                if not tok:
                    continue
                try:
                    parts.append(float(tok))
                except (TypeError, ValueError):
                    continue
            return min(parts) if parts else float('nan')
        try:
            return float(s)
        except (TypeError, ValueError):
            return float('nan')

    return df['coelution_score'].map(_parse_one)


def _filter_df_by_min_coelution_score(
    df: 'pd.DataFrame',
    min_score: float,
    protect_valuable: bool = False,
    protect_strong_psm: bool = False,
) -> 'pd.DataFrame':
    """Drop rows with coelution_score below min_score (NaN scores fail the filter)."""
    import pandas as pd

    min_score = float(min_score)
    if df is None or df.empty:
        return df
    if 'coelution_score' not in df.columns:
        print('[Step 8] Warning: no coelution_score column; skipping min coelution filter.', flush=True)
        return df
    scores = _row_coelution_score_series(df)
    keep_mask = scores >= min_score
    return _apply_row_mask_respecting_valuable(
        df,
        keep_mask,
        protect_valuable,
        f'Min coelution_score (>={min_score:g})',
        protect_strong_psm=protect_strong_psm,
    )


def _filter_df_by_min_overhang_protein_positions(
    df: 'pd.DataFrame',
    min_positions: int,
    overhang_prot_col: str | None = None,
    protect_valuable: bool = False,
    protect_strong_psm: bool = False,
) -> 'pd.DataFrame':
    """Drop rows with fewer than min_positions resolved single-AA overhang protein sites."""
    min_positions = int(min_positions or 0)
    if min_positions <= 0 or df is None or df.empty:
        return df
    counts = df.apply(lambda r: _resolved_overhang_protein_position_count(r, overhang_prot_col), axis=1)
    keep_mask = counts >= min_positions
    return _apply_row_mask_respecting_valuable(
        df,
        keep_mask,
        protect_valuable,
        f'Min resolved single-AA overhang protein positions (>={min_positions})',
        protect_strong_psm=protect_strong_psm,
    )


_SEQUENCE_CSV_SORT_VARIANTS = (
    ('by_total_area', False),
    ('by_protein_position', True),
    ('by_collection_min_rt', True),
    ('by_overhang_pair_count', False),
)


def _sort_sequence_dataframe(df: 'pd.DataFrame', sort_tag: str) -> 'pd.DataFrame':
    """Return a row-sorted copy (column order unchanged)."""
    import pandas as pd

    if df is None or df.empty:
        return df
    out = df.copy()
    if sort_tag == 'by_total_area':
        if 'total_area' not in out.columns:
            return out
        key = pd.to_numeric(out['total_area'], errors='coerce').fillna(-np.inf)
        order = key.argsort(kind='mergesort')[::-1]
    elif sort_tag == 'by_protein_position':
        pos_col = next((c for c in ('protein_position', 'sequence_positions') if c in out.columns), None)
        if not pos_col:
            return out
        key = out[pos_col].map(_protein_position_sort_start)
        order = key.argsort(kind='mergesort')
    elif sort_tag == 'by_collection_min_rt':
        if 'collection_min_rt' not in out.columns:
            return out
        key = pd.to_numeric(out['collection_min_rt'], errors='coerce').fillna(np.inf)
        order = key.argsort(kind='mergesort')
    elif sort_tag == 'by_overhang_pair_count':
        pairs_col = _overhang_pairs_count_column(out)
        if not pairs_col:
            return out
        key = out[pairs_col].map(_count_single_aa_overhang_pairs)
        order = key.argsort(kind='mergesort')[::-1]
    else:
        return out
    return out.iloc[order].reset_index(drop=True)


def _format_total_area_scientific_for_csv(df: 'pd.DataFrame') -> 'pd.DataFrame':
    """Format total_area column as scientific notation strings for CSV export."""
    import pandas as pd

    if df is None or df.empty or 'total_area' not in df.columns:
        return df
    out = df.copy()

    def _fmt_cell(val) -> str:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return ''
        s = str(val).strip()
        if not s or s.lower() in ('nan', 'none'):
            return ''
        if ';' in s:
            parts = []
            for tok in s.split(';'):
                tok = tok.strip()
                if not tok:
                    continue
                try:
                    parts.append(f'{float(tok):.3e}')
                except (TypeError, ValueError):
                    parts.append(tok)
            return ';'.join(parts)
        try:
            return f'{float(s):.3e}'
        except (TypeError, ValueError):
            return s

    out['total_area'] = out['total_area'].map(_fmt_cell)
    return out


def _collection_rt_columns_to_minutes(df: 'pd.DataFrame') -> 'pd.DataFrame':
    """Write collection window RT columns in minutes (input pipeline values are seconds), 1 decimal."""
    import pandas as pd

    if df is None or df.empty:
        return df
    out = df.copy()
    for col in ('collection_min_rt', 'collection_max_rt'):
        if col in out.columns:
            out[col] = (pd.to_numeric(out[col], errors='coerce') / 60.0).round(1)
    return out


def _write_sequence_csv_sort_variants(df: 'pd.DataFrame', stem: str) -> None:
    """Write the same table with one CSV per row-sort variant."""
    for sort_tag, _asc_hint in _SEQUENCE_CSV_SORT_VARIANTS:
        sorted_df = _sort_sequence_dataframe(df, sort_tag)
        sorted_df = _collection_rt_columns_to_minutes(sorted_df)
        sorted_df = _reorder_sequence_output_columns(sorted_df)
        sorted_df = _format_total_area_scientific_for_csv(sorted_df)
        if sort_tag == 'by_protein_position':
            out_path = f'{stem}.csv'
        else:
            out_path = f'{stem}_{sort_tag}.csv'
        sorted_df.to_csv(out_path, index=False)
        print(f'[Step 8] Saved ({sort_tag}, {len(sorted_df)} rows): {os.path.abspath(out_path)}', flush=True)


def _reorder_sequence_output_columns(df: 'pd.DataFrame') -> 'pd.DataFrame':
    """Lead: theoretical_mz then observed_mz; scores after overhang positions; perc_qvalue last."""
    if df is None or df.empty:
        return df
    out = df.copy()
    orig = list(out.columns)
    lead = _sequence_csv_lead_columns(out)
    tail = [c for c in _SEQUENCE_CSV_TAIL_COLUMNS if c in out.columns]
    trailing = [c for c in _SEQUENCE_CSV_TRAILING_COLUMNS if c in out.columns]
    after_overhang = [c for c in _SEQUENCE_CSV_AFTER_OVERHANG_COLUMNS if c in out.columns]
    far_right = [c for c in _SEQUENCE_CSV_FAR_RIGHT_COLUMNS if c in out.columns]
    anchor = next((c for c in _SEQUENCE_CSV_OVERHANG_ANCHOR_COLUMNS if c in out.columns), None)

    rt_sec_cols = [
        c for c in orig
        if (('RT' in c or 'retention_time' in c) and str(c).endswith('_sec'))
    ]
    reserved = set(lead) | set(tail) | set(trailing) | set(rt_sec_cols) | set(after_overhang) | set(far_right)
    if anchor:
        reserved.add(anchor)
    middle = [c for c in orig if c not in reserved]

    if anchor:
        anchor_idx = orig.index(anchor)
        before_anchor = [c for c in middle if orig.index(c) < anchor_idx]
        after_anchor = [c for c in middle if orig.index(c) > anchor_idx]
        body = before_anchor + [anchor] + after_overhang + after_anchor
    else:
        body = middle

    return out[lead + body + tail + rt_sec_cols + trailing + far_right]


_COMMA_LIST_COL_KEYWORDS = (
    'frag',
    'overhang',
    'matched_frags',
    'significant_frags',
    'fragment_matched',
    'single_aa_overhang',
    'significant_fragment',
)

_NUMERIC_SUM_COLS = frozenset({'total_area', 'ions_matched'})
_NUMERIC_MIN_COLS = frozenset({'perc_qvalue', 'perc_PEP', 'e-value', 'percolator_qvalue', 'q_value', 'qvalue'})


def _is_comma_list_column(col_name: str) -> bool:
    c = str(col_name or '').lower()
    return any(tok in c for tok in _COMMA_LIST_COL_KEYWORDS)


def _format_numeric_aggregate(series) -> object:
    import pandas as pd

    num = pd.to_numeric(series, errors='coerce').dropna()
    if num.empty:
        return np.nan
    uniq = sorted(set(float(v) for v in num.tolist()))
    if len(uniq) == 1:
        return uniq[0]
    return ';'.join(f'{v:.6g}' for v in uniq)


def _aggregate_group_column(series, col_name: str) -> object:
    """Merge one column across PSM rows in a peptide group (union / min / sum as appropriate)."""
    import pandas as pd

    if series is None or len(series) == 0:
        return np.nan
    s = series.dropna()
    if s.empty:
        return np.nan
    col = str(col_name)
    if _is_comma_list_column(col):
        return ','.join(_sorted_unique_tokens(s))
    if col in _NUMERIC_SUM_COLS:
        num = pd.to_numeric(s, errors='coerce').dropna()
        return float(num.sum()) if len(num) else np.nan
    if col == 'collection_min_rt':
        num = pd.to_numeric(s, errors='coerce').dropna()
        return float(num.min()) if len(num) else np.nan
    if col == 'collection_max_rt':
        num = pd.to_numeric(s, errors='coerce').dropna()
        return float(num.max()) if len(num) else np.nan
    if col in _NUMERIC_MIN_COLS or 'qvalue' in col.lower() or col == 'e-value':
        num = pd.to_numeric(s, errors='coerce').dropna()
        return float(num.min()) if len(num) else np.nan
    if pd.api.types.is_numeric_dtype(s):
        return _format_numeric_aggregate(s)
    parts = _sorted_unique_tokens([str(v).strip() for v in s.tolist() if str(v).strip().lower() not in ('', 'nan', 'none')])
    if not parts:
        return np.nan
    if len(parts) == 1:
        return parts[0]
    return ';'.join(parts)


def _collapse_peptide_identity_rows(
    df: 'pd.DataFrame',
    *,
    group_cols: list[str],
    overhang_col: str | None,
    pairs_col: str | None,
    q_col: str | None,
) -> 'pd.DataFrame':
    """Collapse scan/PSM rows to one row per grouped peptide identity, unioning multi-PSM fields."""
    import pandas as pd

    pep_candidates = ['peptide_sequence', 'plain_peptide', 'peptide', 'sequence']
    df_u = df.copy()
    df_u['_pep_seq'] = df_u.apply(lambda r: _first_nonempty_str(r, pep_candidates), axis=1)
    df_u['_charge'] = pd.to_numeric(df_u.get('charge'), errors='coerce')
    df_u['_mods'] = df_u.get('modifications', pd.Series(['-'] * len(df_u))).astype(str).str.strip()
    df_u.loc[df_u['_mods'].isin(['', 'nan', 'None']), '_mods'] = '-'
    df_u = df_u[df_u['_pep_seq'].astype(str).str.strip() != ''].copy()

    unique_rows = []
    for key_vals, g in df_u.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(key_vals, tuple):
            key_vals = (key_vals,)
        key_map = dict(zip(group_cols, key_vals))
        pep = str(key_map.get('_pep_seq', '')).strip()
        ch = key_map.get('_charge')
        mods_key = key_map.get('_mods', '-')

        if 'status' in g.columns:
            g_valid = g[g['status'].map(_has_valid_extraction_status)]
            if g_valid.empty:
                continue
            g = g_valid
            g_acc = g[g['status'].astype(str).str.strip().str.lower() == 'accepted']
            if not g_acc.empty:
                g = g_acc
        if q_col:
            qv = pd.to_numeric(g[q_col], errors='coerce')
            best_idx = qv.idxmin() if qv.notna().any() else g.index[0]
        else:
            best_idx = g.index[0]
        row_out = g.loc[best_idx].copy()
        overhang_positions = _sorted_unique_tokens(g[overhang_col]) if overhang_col else []
        overhang_pairs = _sorted_unique_tokens(g[pairs_col]) if pairs_col else []
        row_out['n_rows_collapsed'] = int(len(g))
        if '_mods' not in group_cols:
            row_out['n_mod_variants_collapsed'] = int(g['_mods'].nunique())
        row_out['valuable_sequence'] = int(
            pd.to_numeric(g.get('valuable_sequence', 0), errors='coerce').fillna(0).astype(int).max() > 0
        )
        row_out['num_psms'] = int(len(g))
        row_out['protect_peptide'] = bool(pd.Series(g.get('protect_peptide', False)).astype(bool).max())
        if 'peptide_sequence' in row_out.index:
            row_out['peptide_sequence'] = pep
        if 'plain_peptide' in row_out.index:
            row_out['plain_peptide'] = pep
        if 'charge' in row_out.index:
            row_out['charge'] = int(ch) if pd.notna(ch) else ''
        if 'modifications' in row_out.index:
            if '_mods' in group_cols:
                row_out['modifications'] = str(mods_key) if str(mods_key).strip() else '-'
            else:
                row_out['modifications'] = ','.join(_sorted_unique_tokens(g['_mods']))
        if overhang_col:
            row_out[overhang_col] = ','.join(overhang_positions)
        if pairs_col:
            row_out[pairs_col] = ','.join(overhang_pairs)
        skip_aggregate = set(group_cols) | {
            'peptide_sequence', 'plain_peptide', 'peptide', 'sequence', 'charge',
            'valuable_sequence', 'strong_psm_support', 'protect_peptide',
            'num_psms', 'n_rows_collapsed', 'n_mod_variants_collapsed',
        }
        if overhang_col:
            skip_aggregate.add(overhang_col)
        if pairs_col:
            skip_aggregate.add(pairs_col)
        for col in g.columns:
            if col in skip_aggregate or str(col).startswith('_'):
                continue
            try:
                row_out[col] = _aggregate_group_column(g[col], col)
            except Exception:
                pass
        if 'scan' in g.columns:
            row_out['collapsed_scans'] = ','.join(
                _sorted_unique_tokens(pd.to_numeric(g['scan'], errors='coerce').dropna().astype(int).astype(str))
            )
        if 'source_mzml' in g.columns:
            row_out['collapsed_source_mzml'] = ';'.join(
                _sorted_unique_tokens(g['source_mzml'].astype(str))
            )
        unique_rows.append(row_out)

    df_unique = pd.DataFrame(unique_rows)
    helper_cols = [c for c in ['_pep_seq', '_charge', '_mods'] if c in df_unique.columns]
    if helper_cols:
        df_unique = df_unique.drop(columns=helper_cols)
    tail_cols = [c for c in ('n_mod_variants_collapsed', 'n_rows_collapsed') if c in df_unique.columns]
    if tail_cols:
        front = [c for c in df_unique.columns if c not in tail_cols]
        df_unique = df_unique[front + tail_cols]
    return df_unique


def _generate_sequence_coverage_plots(
    df,
    fasta_path: str,
    protein_id: str | None,
    seq_cov_dir: str,
    output_base: str,
    col_overhangs: str,
    col_seq_start: str,
    peptide_to_positions: dict,
    debug: bool = False,
) -> bool:
    """Create unique-peptide protein grid PNGs under seq_cov_dir. Returns True if any plot written."""
    if not fasta_path or not os.path.exists(fasta_path):
        print('[Step 8] No FASTA provided; skipping sequence coverage plots.', flush=True)
        return False
    protein_sequence, pid = _load_fasta(fasta_path, protein_id)
    if not protein_sequence:
        print('[Step 8] Could not load protein sequence from FASTA.', flush=True)
        return False
    try:
        from visualization.combined import create_accepted_peptides_protein_grid
    except Exception as e:
        print(f'[Step 8] Warning: Could not import plotting module: {e}', flush=True)
        return False
    print('[Step 8] Building peptides list from CSV (one entry per unique seq+charge+mods)...', flush=True)
    peptides = []
    seen = set()
    for idx in df.index:
        row = df.loc[idx]
        peptide = _first_nonempty_str(row, ['peptide_sequence', 'plain_peptide', 'peptide', 'sequence'])
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
        peptides.append({
            'plain_peptide': peptide,
            'peptide_seq': peptide,
            'start': start,
            'end': end,
            'charge': charge,
            'modifications': mods,
            'total_area': row.get('total_area'),
            'significant_single_aa_overhangs': row.get(col_overhangs),
            'significant_single_aa_overhangs_protein_positions': (
                ','.join(map(str, sorted(peptide_to_positions.get(idx, set()))))
                if peptide_to_positions.get(idx) else ''
            ),
            'significant_fragment_pairs': row.get('significant_fragment_pairs', ''),
        })
    print(f'[Step 8] Built {len(peptides)} peptide entries for grid (deduped by seq+charge+mods)', flush=True)
    if debug and peptides:
        p0 = peptides[0]
        print(
            f"[Step 8] DEBUG: First peptide: {p0.get('plain_peptide', '')[:40]}... "
            f"start={p0.get('start')} end={p0.get('end')}",
            flush=True,
        )
    if not peptides:
        print('[Step 8] No peptide rows available for plotting (check peptide columns).', flush=True)
        return False
    os.makedirs(seq_cov_dir, exist_ok=True)
    unique_output = os.path.join(seq_cov_dir, f'{output_base}_unique_peptides.png')
    print(f'[Step 8] Creating unique peptides grid ({len(peptides)} entries -> unique sequences)...', flush=True)
    create_accepted_peptides_protein_grid(
        peptides, protein_sequence, pid or 'protein', len(protein_sequence),
        unique_output, seq_cov_dir,
    )
    print(f'[Step 8] Unique peptides plot saved to: {unique_output}', flush=True)
    return True


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
    ap.add_argument(
        '--mode',
        choices=['protein', 'peptide'],
        default='protein',
        help='Workflow mode hint for logging/compatibility (default: protein).',
    )
    ap.add_argument(
        '--ion-types',
        default='c,z',
        help=(
            'Comma-separated ion families to keep in significant_frags and overhang columns '
            '(choices: a,b,c,x,y,z,z1). Default: c,z.'
        ),
    )
    ap.add_argument(
        '--fragment-charges',
        default='',
        help=(
            'Comma-separated fragment ion charges (+) to keep in significant_frags '
            '(lowercase z in the UI). Labels without an explicit charge are treated as +1. '
            'Empty = no fragment-charge filtering.'
        ),
    )
    ap.add_argument(
        '--precursor-charges',
        default='',
        help=(
            'Comma-separated precursor (peptide) charges to keep (capital Z in the UI). '
            'Uses the charge column. Empty = keep all precursor charges.'
        ),
    )
    ap.add_argument(
        '--drop-z2-when-higher-charge-present',
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            'Drop precursor charge Z=2 rows when the same peptide sequence also appears '
            'at Z>2 (after other filters). valuable_sequence is recomputed afterward on '
            'the remaining rows (redundant Z=2 rows are always dropped).'
        ),
    )
    ap.add_argument(
        '--plots',
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            'After writing sequence CSVs, also generate unique-peptide protein grid PNGs. '
            'Default: off (use the app Generate sequence coverage plots button or --plots-only).'
        ),
    )
    ap.add_argument(
        '--plots-only',
        action='store_true',
        help='Only generate sequence coverage PNGs (no sequence CSV rewrite).',
    )
    ap.add_argument(
        '--include-proteins',
        default=None,
        help='Comma-separated protein labels to keep (matches the protein column in the input CSV).',
    )
    ap.add_argument(
        '--exclude-decoys',
        action='store_true',
        help='Drop rows whose protein label looks like a decoy (DECOY_*, REV_*, etc.).',
    )
    ap.add_argument(
        '--require-extraction-status',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Drop rows with missing extraction status/accept_class (default: true).',
    )
    ap.add_argument(
        '--accepted-only',
        action='store_true',
        help='Keep only rows with status=accepted (after --require-extraction-status).',
    )
    ap.add_argument(
        '--min-overhang-protein-positions',
        type=int,
        default=3,
        help=(
            'Minimum count of resolved protein positions in single_aa_overhangs_protein_positions '
            '(from single-AA overhang pairs). Applied immediately after --ion-types filtering '
            '(overhang columns are recomputed for the selected ion families first). Use 0 to disable.'
        ),
    )
    ap.add_argument(
        '--min-total-area',
        type=float,
        default=0.0,
        help='Minimum total_area to keep a row. Rows below this are excluded. Use 0 to disable.',
    )
    ap.add_argument(
        '--protect-valuable-sequences',
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            'Keep valuable_sequence=1 rows when applying filters after the post-ion annotation step '
            '(min overhang, protein, extraction status, min total_area, min coelution, precursor charge; '
            'default: true). valuable_sequence is recomputed after each filter; protection uses the '
            'flag from immediately before that filter. '
            'Does not apply to redundant Z=2 removal, which always drops Z=2 and then recomputes valuable_sequence.'
        ),
    )
    ap.add_argument(
        '--min-strong-psm-support',
        type=int,
        default=4,
        help=(
            'Minimum PSM count per row for strong_psm_support=1 (uses num_psms, or n_rows_collapsed '
            'after collapse). Rows with count >= this value get strong_psm_support=1.'
        ),
    )
    ap.add_argument(
        '--protect-strong-psm-support',
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            'Keep strong_psm_support=1 rows when applying filters after strong_psm_support is assigned '
            '(post ion/charge filter; same steps as --protect-valuable-sequences; not applied to Z=2 drop).'
        ),
    )
    ap.add_argument(
        '--filter-min-coelution-score',
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            'Drop rows with coelution_score below --min-coelution-score (default: on). '
            'Honors --protect-valuable-sequences and --protect-strong-psm-support.'
        ),
    )
    ap.add_argument(
        '--min-coelution-score',
        type=float,
        default=0.5,
        help='Minimum coelution_score to keep a row when --filter-min-coelution-score is enabled.',
    )
    args = ap.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input CSV not found: {args.input}")
        sys.exit(1)

    print(f"[Step 8] Loading input CSV (mode={args.mode})...", flush=True)
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

    protect_valuable = bool(args.protect_valuable_sequences)
    protect_strong_psm = bool(args.protect_strong_psm_support)
    min_strong_psm = max(0, int(args.min_strong_psm_support or 0))
    if protect_valuable:
        print(
            '[Step 8] Protect valuable sequences: ON '
            '(recompute valuable_sequence after each filter; valuable rows bypass later filters)',
            flush=True,
        )
    else:
        print('[Step 8] Protect valuable sequences: OFF', flush=True)
    if protect_strong_psm:
        print(
            f'[Step 8] Protect strong PSM support: ON (strong_psm_support=1 when PSM count >= {min_strong_psm})',
            flush=True,
        )
    else:
        print('[Step 8] Protect strong PSM support: OFF', flush=True)

    print(
        '[Step 8] Pipeline: ion/charge filter -> valuable_sequence + strong_psm_support -> '
        'min overhang (honors protection) -> recompute valuable -> each later filter -> '
        'recompute valuable',
        flush=True,
    )

    # Normalize legacy column naming from earlier runs.
    if 'valuable_peptide' in df.columns:
        if 'valuable_sequence' in df.columns:
            legacy_vals = pd.to_numeric(df['valuable_peptide'], errors='coerce').fillna(0).astype(int)
            current_vals = pd.to_numeric(df['valuable_sequence'], errors='coerce').fillna(0).astype(int)
            df['valuable_sequence'] = (legacy_vals.gt(0) | current_vals.gt(0)).astype(int)
            df = df.drop(columns=['valuable_peptide'])
        else:
            df = df.rename(columns={'valuable_peptide': 'valuable_sequence'})

    df = _apply_fragment_filters_df(
        df,
        str(args.ion_types or '').strip(),
        str(args.fragment_charges or '').strip(),
    )
    if df.empty:
        print('[Step 8] No rows remain after fragment ion/charge filtering.', flush=True)
        sys.exit(1)

    col_overhangs_prot, col_overhangs, col_seq_start = _resolve_overhang_columns(df)
    df, peptide_to_positions, unique_overhang_positions = _annotate_valuable_and_strong_psm_support(
        df,
        col_overhangs_prot,
        col_overhangs,
        col_seq_start,
        min_strong_psm,
        step_label='post ion/charge filter (valuable_sequence + strong_psm_support)',
    )

    min_oh = int(args.min_overhang_protein_positions or 0)
    if min_oh > 0:
        df = _filter_df_by_min_overhang_protein_positions(
            df,
            min_oh,
            col_overhangs_prot,
            protect_valuable=protect_valuable,
            protect_strong_psm=protect_strong_psm,
        )
        if df.empty:
            print(
                '[Step 8] No rows remain after min overhang protein-position filtering '
                '(post ion/charge filter).',
                flush=True,
            )
            sys.exit(1)
        df, peptide_to_positions, unique_overhang_positions = _recompute_valuable_sequence_columns(
            df,
            col_overhangs_prot,
            col_overhangs,
            col_seq_start,
            step_label='min overhang protein positions filter (post ion/charge)',
        )

    if args.include_proteins:
        df = _filter_df_by_include_proteins(
            df,
            args.include_proteins,
            protect_valuable=protect_valuable,
            protect_strong_psm=protect_strong_psm,
        )
    elif args.exclude_decoys:
        df = _filter_df_exclude_decoys(
            df,
            protect_valuable=protect_valuable,
            protect_strong_psm=protect_strong_psm,
        )
    if df.empty:
        print('[Step 8] No rows remain after protein filtering.', flush=True)
        sys.exit(1)
    df, peptide_to_positions, unique_overhang_positions = _recompute_valuable_sequence_columns(
        df,
        col_overhangs_prot,
        col_overhangs,
        col_seq_start,
        step_label='protein filter',
    )

    df, _ = _filter_rows_for_sequence_coverage(
        df,
        require_extraction_status=bool(args.require_extraction_status),
        accepted_only=bool(args.accepted_only),
        protect_valuable=protect_valuable,
        protect_strong_psm=protect_strong_psm,
    )
    if df.empty:
        print('[Step 8] No rows remain after extraction-status filtering.', flush=True)
        sys.exit(1)
    df, peptide_to_positions, unique_overhang_positions = _recompute_valuable_sequence_columns(
        df,
        col_overhangs_prot,
        col_overhangs,
        col_seq_start,
        step_label='extraction status filter',
    )

    df = _filter_df_by_min_total_area(
        df,
        float(args.min_total_area),
        protect_valuable=protect_valuable,
        protect_strong_psm=protect_strong_psm,
    )
    if df.empty:
        print('[Step 8] No rows remain after min total_area filtering.', flush=True)
        sys.exit(1)
    df, peptide_to_positions, unique_overhang_positions = _recompute_valuable_sequence_columns(
        df,
        col_overhangs_prot,
        col_overhangs,
        col_seq_start,
        step_label='min total_area filter',
    )

    if bool(args.filter_min_coelution_score):
        df = _filter_df_by_min_coelution_score(
            df,
            float(args.min_coelution_score),
            protect_valuable=protect_valuable,
            protect_strong_psm=protect_strong_psm,
        )
        if df.empty:
            print('[Step 8] No rows remain after min coelution_score filtering.', flush=True)
            sys.exit(1)
        df, peptide_to_positions, unique_overhang_positions = _recompute_valuable_sequence_columns(
            df,
            col_overhangs_prot,
            col_overhangs,
            col_seq_start,
            step_label='min coelution_score filter',
        )

    df = _apply_precursor_charge_filter_df(
        df,
        str(args.precursor_charges or '').strip(),
        protect_valuable=protect_valuable,
        protect_strong_psm=protect_strong_psm,
    )
    if df.empty:
        print('[Step 8] No rows remain after precursor-charge filtering.', flush=True)
        sys.exit(1)
    df, peptide_to_positions, unique_overhang_positions = _recompute_valuable_sequence_columns(
        df,
        col_overhangs_prot,
        col_overhangs,
        col_seq_start,
        step_label='precursor charge filter',
    )

    if bool(args.drop_z2_when_higher_charge_present):
        df = _filter_df_drop_z2_when_higher_charge_present(df, reference_charge=2)
        if df.empty:
            print(
                '[Step 8] No rows remain after drop-Z2-when-higher-charge filtering.',
                flush=True,
            )
            sys.exit(1)
        print(
            '[Step 8] Recomputing valuable_sequence / protect_peptide after dropping redundant Z=2 rows...',
            flush=True,
        )
        df, peptide_to_positions, unique_overhang_positions = _recompute_valuable_sequence_columns(
            df,
            col_overhangs_prot,
            col_overhangs,
            col_seq_start,
            step_label='drop Z=2 when higher charge present (valuable_sequence reset)',
        )

    if args.debug:
        print(
            f'[Step 8] DEBUG: final {len(unique_overhang_positions)} unique overhang positions '
            '(covered by one row)',
            flush=True,
        )

    peptide_to_positions = _build_peptide_position_map(df, col_overhangs_prot, col_overhangs, col_seq_start)

    if args.plots_only:
        ok = _generate_sequence_coverage_plots(
            df,
            str(args.fasta or ''),
            args.protein_id,
            seq_cov_dir,
            base,
            col_overhangs,
            col_seq_start,
            peptide_to_positions,
            debug=bool(args.debug),
        )
        if not ok:
            sys.exit(1)
        return 0

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

    df = _assign_strong_psm_support_column(df, min_strong_psm)

    print('[Step 8] Writing output CSVs (4 row sorts each)...', flush=True)
    _write_sequence_csv_sort_variants(df, os.path.join(out_dir, f'{base}_sequence'))

    overhang_col = (
        'significant_single_aa_overhangs_protein_positions'
        if 'significant_single_aa_overhangs_protein_positions' in df.columns
        else ('single_aa_overhangs_protein_positions' if 'single_aa_overhangs_protein_positions' in df.columns else None)
    )
    pairs_col = (
        'single_aa_overhang_fragment_pairs'
        if 'single_aa_overhang_fragment_pairs' in df.columns
        else ('significant_fragment_pairs' if 'significant_fragment_pairs' in df.columns else None)
    )
    q_candidates = ['percolator_qvalue', 'q_value', 'qvalue', 'q-val', 'qval']
    q_col = next((c for c in q_candidates if c in df.columns), None)

    df_unique_sc = _collapse_peptide_identity_rows(
        df,
        group_cols=['_pep_seq', '_charge'],
        overhang_col=overhang_col,
        pairs_col=pairs_col,
        q_col=q_col,
    )
    df_unique_sc, _, _ = _recompute_valuable_sequence_columns(
        df_unique_sc,
        col_overhangs_prot,
        col_overhangs,
        col_seq_start,
        step_label='collapse to unique seq+charge (valuable_sequence reset)',
        log_summary=bool(args.drop_z2_when_higher_charge_present),
    )
    df_unique_sc = _assign_strong_psm_support_column(df_unique_sc, min_strong_psm)
    _write_sequence_csv_sort_variants(
        df_unique_sc,
        os.path.join(out_dir, f'{base}_sequence_unique_seq_charge'),
    )
    print(
        f'[Step 8] Unique sequence+charge CSVs written ({len(df_unique_sc)} rows each; '
        "multi-PSM fields unioned, ';' separates differing values)",
        flush=True,
    )

    df_unique_mods = _collapse_peptide_identity_rows(
        df,
        group_cols=['_pep_seq', '_charge', '_mods'],
        overhang_col=overhang_col,
        pairs_col=pairs_col,
        q_col=q_col,
    )
    df_unique_mods, _, _ = _recompute_valuable_sequence_columns(
        df_unique_mods,
        col_overhangs_prot,
        col_overhangs,
        col_seq_start,
        step_label='collapse to unique seq+charge+mods (valuable_sequence reset)',
        log_summary=bool(args.drop_z2_when_higher_charge_present),
    )
    df_unique_mods = _assign_strong_psm_support_column(df_unique_mods, min_strong_psm)
    _write_sequence_csv_sort_variants(
        df_unique_mods,
        os.path.join(out_dir, f'{base}_sequence_unique_peptides'),
    )
    print(
        f'[Step 8] Unique sequence+charge+mods CSVs written ({len(df_unique_mods)} rows each)',
        flush=True,
    )

    if bool(args.plots):
        _generate_sequence_coverage_plots(
            df,
            str(args.fasta or ''),
            args.protein_id,
            seq_cov_dir,
            base,
            col_overhangs,
            col_seq_start,
            peptide_to_positions,
            debug=bool(args.debug),
        )
    else:
        print(
            '[Step 8] Skipping sequence coverage PNGs '
            '(use --plots, or run with --plots-only to plot without rewriting CSVs).',
            flush=True,
        )


if __name__ == '__main__':
    main()
