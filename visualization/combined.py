#!/usr/bin/env python3
"""
Combined Overhang Visualization Module

This module provides functions for creating various overhang visualizations:
- create_combined_visualization(): Main combined plot with heatmap, histogram, and RT grid
- create_fragment_contributors_plot(): Fragment contributors visualization
- generate_threshold_scatter_plot(): Threshold analysis scatter plot

For command-line usage, use the separate script files:
- generate_combined_plot.py: Main combined visualization
- generate_fragment_contributors.py: Fragment contributors plot

All functions use the same q-value extraction and filtering logic for consistency.
"""

import argparse
import re
import csv
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
import numpy as np
import sys
import os
import textwrap

# Import from same package (peak-window production is in visualization.chromatograms)
from .multi_aa import (
    parse_fasta, parse_multi_aa_overhangs, parse_single_aa_overhangs,
    normalize_fragment_pair, get_filtered_peptide_indices
)

# All overhang lengths for coverage/RT grid (1-AA through 20-AA)
OVERHANG_TYPE_KEYS = ['1aa'] + [f'{n}aa' for n in range(2, 21)]

# Total area colorbar: show powers of 10 from 10^4 to 10^11 on all total_area colorbars
TOTAL_AREA_CBAR_POWERS = [4, 5, 6, 7, 8, 9, 10, 11]
TOTAL_AREA_CBAR_TICKS = [10**k for k in TOTAL_AREA_CBAR_POWERS]
TOTAL_AREA_CBAR_LABELS = ['10⁴', '10⁵', '10⁶', '10⁷', '10⁸', '10⁹', '10¹⁰', '10¹¹']

# Peak drift window: buffer each side of collection window (relative to width, with min/max)
PEAK_DRIFT_BUFFER_DEFAULT_SEC = 30.0
PEAK_DRIFT_BUFFER_FRAC_DEFAULT = 0.5
PEAK_DRIFT_BUFFER_MIN_SEC_DEFAULT = 15.0
PEAK_DRIFT_BUFFER_MAX_SEC_DEFAULT = 90.0


def peak_drift_buffer_sec(cmin, cmax, frac=PEAK_DRIFT_BUFFER_FRAC_DEFAULT,
                          min_sec=PEAK_DRIFT_BUFFER_MIN_SEC_DEFAULT,
                          max_sec=PEAK_DRIFT_BUFFER_MAX_SEC_DEFAULT,
                          default=PEAK_DRIFT_BUFFER_DEFAULT_SEC):
    """Buffer (seconds) to add on each side of collection window. Fixed +30 seconds on either side."""
    return default


def _set_total_area_colorbar_ticks(cbar):
    """Set tick positions and labels on a total_area colorbar to 10^4 through 10^11."""
    cbar.set_ticks(TOTAL_AREA_CBAR_TICKS)
    cbar.set_ticklabels(TOTAL_AREA_CBAR_LABELS)


def filter_fragment_pair_by_ion_type(pair, ion_type_filter):
    """
    Check if a fragment pair matches the ion type filter.
    
    Arguments (Args):
        pair: Fragment pair string like "c3-c4", "z5-z6", "z1_5-z1_6"
        ion_type_filter: 'c' for c ions only, 'z' for z/z+1 ions only, None for all
   """
    if ion_type_filter is None:
        return True
    normalized_pair = normalize_fragment_pair(pair)
    #turns a fragment pair string into a single string
    if ion_type_filter == 'c':
        # Check if both fragments start with 'c'
        return normalized_pair.startswith('c') and '-' in normalized_pair and normalized_pair.split('-')[1].startswith('c')
    elif ion_type_filter == 'z':
        # Check if both fragments start with 'z' (includes z1_ which normalizes to z)
        return normalized_pair.startswith('z') and '-' in normalized_pair and normalized_pair.split('-')[1].startswith('z')
    return True


def _peptide_c_z_fragment_lists(single_aa_fragment_pairs, ion_type_filter):
    """
    From a peptide's single_aa_fragment_pairs dict (pos -> set of pair strings), return
    (c_pairs, z_pairs) as sorted lists of normalized pair strings for y-label and grid display.
    Respects ion_type_filter: when 'c' or 'z', only that type is returned.
    """
    c_pairs = set()
    z_pairs = set()
    for _pos, pairs_set in (single_aa_fragment_pairs or {}).items():
        for pair_str in pairs_set:
            if not filter_fragment_pair_by_ion_type(pair_str, ion_type_filter):
                continue
            n = normalize_fragment_pair(pair_str)
            if n.startswith('c') and '-' in n and n.split('-')[1].startswith('c'):
                c_pairs.add(n)
            elif n.startswith('z') and '-' in n and n.split('-')[1].startswith('z'):
                z_pairs.add(n)
    return (sorted(c_pairs), sorted(z_pairs))


def _c_z_numbers_from_fragment_pairs_union(pairs_union):
    """
    From single_aa_fragment_pairs_union (position -> set of pair strings like 'c3-c4', 'z2-z3'),
    return (c_list, z_list) as sorted lists of fragment numbers for drawing c/z rows with yellow overhangs.
    """
    c_set = set()
    z_set = set()
    for _pos, pairs in (pairs_union or {}).items():
        for p in pairs:
            part0, part1 = (p.split('-') + ['', ''])[:2]
            for part in (part0.strip(), part1.strip()):
                if not part:
                    continue
                if part.startswith('c') and part[1:].isdigit():
                    c_set.add(int(part[1:]))
                elif part.startswith('z'):
                    n = part[1:].replace('+1', '').strip()
                    if n.isdigit():
                        z_set.add(int(n))
    return (sorted(c_set), sorted(z_set))


def format_mods_for_display(mods):
    """
    Format modifications for y-axis/title display, matching chromatogram extraction plot titles exactly.
    Converts raw mod strings (e.g. '5_V_15.9949' or '5_V_15.994900') to human-readable form (e.g. 'Ox@5').
    Uses same mod_map and parsing as extract_ms1_chromatograms.py format_title().
    Returns '-' if no mods; otherwise up to 3 mods then '+N more' if needed.
    """
    if mods is None or (isinstance(mods, str) and (not mods.strip() or mods.strip() == '-')):
        return '-'
    try:
        if hasattr(mods, 'strip'):
            mods_str = str(mods).strip()
        else:
            mods_str = str(mods)
        if not mods_str or mods_str.lower() in ('nan', '-'):
            return '-'
    except Exception:
        return '-'
    # Same mod_map as chromatogram extraction plot format_title()
    mod_map = {
        '15.9949': 'Ox',       # Oxidation
        '15.994900': 'Ox',
        '57.0215': 'Carbamidomethyl',   # Carbamidomethylation (Cys)
        '57.021500': 'Carbamidomethyl',
        '0.9840': 'Deamidated',        # Deamidation (Asn/Gln)
        '0.984000': 'Deamidated',
    }
    mod_parts = []
    for mod_entry in mods_str.split(','):
        mod_entry = mod_entry.strip()
        if '_' in mod_entry:
            parts = mod_entry.split('_')
            if len(parts) >= 3:
                pos = parts[0]
                mod_type = parts[1]
                mass = parts[2].split(')')[0].strip()  # Remove any trailing )
                # Try exact match first, then try rounding (same as chromatogram filename logic)
                mod_name = mod_map.get(mass)
                if mod_name is None:
                    try:
                        mass_rounded = f"{float(mass):.4f}"
                        mod_name = mod_map.get(mass_rounded, f"+{mass}")
                    except (ValueError, TypeError):
                        mod_name = f"+{mass}"
                mod_parts.append(f"{mod_name}@{pos}")
    if not mod_parts:
        return mods_str[:30] if len(mods_str) > 30 else mods_str
    display = ','.join(mod_parts[:3])
    if len(mod_parts) > 3:
        display += f" (+{len(mod_parts)-3} more)"
    return display


def is_valid_fragment_pair_for_overhang_length(pair_str, overhang_length):
    """
    Verify that a fragment pair is valid for the given overhang length.
    For an N-AA overhang, fragments MUST differ by exactly N.
    
    Examples:
    - 1-AA overhang: c3-c4 (differ by 1) ✓, c3-c5 (differ by 2) ✗
    - 2-AA overhang: c3-c5 (differ by 2) ✓, c3-c4 (differ by 1) ✗, c3-c6 (differ by 3) ✗
    - 3-AA overhang: c3-c6 (differ by 3) ✓, c3-c5 (differ by 2) ✗
    
    Args:
        pair_str: Fragment pair string like "c3-c4", "z5-z7"
        overhang_length: Expected overhang length (1, 2, 3, etc.)
    
    Returns:
        True if fragments differ by exactly overhang_length, False otherwise
    """
    normalized = normalize_fragment_pair(pair_str)
    match = re.match(r'^([a-z])(\d+)-([a-z])(\d+)$', normalized.lower())
    if not match:
        return False
    
    s1, n1, s2, n2 = match.groups()
    # Fragments must be from the same ion series
    if s1 != s2:
        return False
    
    num1, num2 = int(n1), int(n2)
    fragment_diff = abs(num2 - num1)
    
    # For an N-AA overhang, fragments must differ by exactly N
    return fragment_diff == overhang_length


def extract_all_consecutive_fragment_pairs(matched_ions_str, peptide_start, peptide_end, peptide_length):
    """
    Extract all consecutive fragment pairs from matched fragment ions and map them to protein positions.
    
    Args:
        matched_ions_str: Comma-separated string of fragment ions (e.g., "c4,c5,c6,z8,z9,z10")
        peptide_start: Starting position of peptide in protein (1-indexed)
        peptide_end: Ending position of peptide in protein (1-indexed)
        peptide_length: Length of peptide sequence
    
    Returns:
        Dictionary mapping protein positions to sets of normalized fragment pairs
        {protein_pos: {normalized_pair1, normalized_pair2, ...}}
    """
    if not matched_ions_str or not matched_ions_str.strip():
        return {}
    
    # Parse fragment ions
    fragments = [f.strip() for f in matched_ions_str.split(',') if f.strip()]
    if not fragments:
        return {}
    
    # Group fragments by ion series
    fragments_by_series = defaultdict(list)  # series -> [(fragment_num, original_str), ...]
    
    for frag_str in fragments:
        # Parse fragment: c4, z8, z1_6, etc.
        frag_str_lower = frag_str.lower()
        
        # Handle z1_ format (z1_6 -> z6)
        if frag_str_lower.startswith('z1_'):
            # Extract number after z1_
            match = re.match(r'z1_(\d+)', frag_str_lower)
            if match:
                frag_num = int(match.group(1))
                fragments_by_series['z'].append((frag_num, frag_str))
        else:
            # Regular format: c4, z8, etc.
            match = re.match(r'^([a-z])(\d+)$', frag_str_lower)
            if match:
                series = match.group(1)
                frag_num = int(match.group(2))
                fragments_by_series[series].append((frag_num, frag_str))
    
    # Find all consecutive pairs within each ion series
    position_to_pairs = defaultdict(set)
    
    for series, frag_list in fragments_by_series.items():
        # Sort by fragment number
        frag_list.sort(key=lambda x: x[0])
        
        # Find consecutive pairs
        for i in range(len(frag_list) - 1):
            frag1_num, frag1_str = frag_list[i]
            frag2_num, frag2_str = frag_list[i + 1]
            
            # Check if consecutive (differ by exactly 1)
            if frag2_num - frag1_num == 1:
                # Create normalized pair
                pair_str = f"{frag1_str}-{frag2_str}"
                normalized_pair = normalize_fragment_pair(pair_str)
                
                # Map to protein position
                # For c-ions (N-terminal): cN covers positions peptide_start to peptide_start + N - 1
                # The overhang between cN and c(N+1) is at position peptide_start + N
                # For z-ions (C-terminal): zN covers positions peptide_end - N + 1 to peptide_end
                # The overhang between zN and z(N+1) is at position peptide_end - N
                
                if series == 'c':
                    # c-ions: overhang is at peptide_start + frag2_num - 1
                    protein_pos = peptide_start + frag2_num - 1
                elif series == 'z':
                    # z-ions: overhang is at peptide_end - frag2_num + 1
                    protein_pos = peptide_end - frag2_num + 1
                else:
                    # For other series (a, b, x, y), use same logic as c-ions for now
                    protein_pos = peptide_start + frag2_num - 1
                
                # CRITICAL: Only add if position is within the peptide boundaries
                # Single-AA overhangs MUST be within the peptide (peptide_start <= pos <= peptide_end)
                if peptide_start > 0 and peptide_end > 0 and peptide_start <= protein_pos <= peptide_end:
                    position_to_pairs[protein_pos].add(normalized_pair)
    
    return dict(position_to_pairs)


def filter_overhangs_by_ion_type(overhangs, ion_type_filter):
    """
    Filter fragment pairs in overhangs by ion type and validate fragment pair matches overhang length.
    
    Args:
        overhangs: List of overhang dictionaries with 'fragment_pairs' and 'length' keys
        ion_type_filter: 'c' for c ions only, 'z' for z/z+1 ions only, None for all
    
    Returns:
        Filtered list of overhangs (overhangs with no fragment pairs after filtering are removed)
    """
    filtered = []
    for overhang in overhangs:
        fragment_pairs = overhang.get('fragment_pairs', [])
        overhang_length = overhang.get('length', 0)
        
        if fragment_pairs:
            # CRITICAL: Only include fragment pairs where fragments differ by exactly the overhang length
            # For an N-AA overhang, fragments must differ by exactly N
            # Then filter by ion type if specified
            filtered_pairs = [p for p in fragment_pairs 
                             if is_valid_fragment_pair_for_overhang_length(p, overhang_length)
                             and filter_fragment_pair_by_ion_type(p, ion_type_filter)]
            if filtered_pairs:
                # Create a copy with filtered fragment pairs
                filtered_overhang = overhang.copy()
                filtered_overhang['fragment_pairs'] = filtered_pairs
                filtered.append(filtered_overhang)
        else:
            # Keep overhangs without fragment pairs (they'll be filtered out elsewhere if needed)
            filtered.append(overhang)
    
    return filtered


def write_filtered_peptides_csv(csv_file, peptides_for_rt_grid, output_path):
    """
    Write a CSV containing only rows for the given filtered peptide set.
    Use this CSV with extract_ms1_chromatograms.py to run chromatogram extraction on only these peptides.
    Preserves original columns; renames retention_time_sec to MS1_retention_time_sec if present (for extract script).
    """
    import pandas as pd
    def _norm_mods(m):
        if m is None or (isinstance(m, float) and np.isnan(m)):
            return '-'
        s = str(m).strip()
        return s if s and s.lower() != 'nan' else '-'
    filtered_keys = set()
    for p in peptides_for_rt_grid:
        seq = (p.get('peptide_seq') or '').strip()
        try:
            c = int(p.get('charge', 0)) if p.get('charge') not in (None, '') else 0
        except (TypeError, ValueError):
            c = 0
        mods = _norm_mods(p.get('modifications'))
        filtered_keys.add((seq, c, mods))
    with open(csv_file, 'r') as f:
        first_line = f.readline()
    skip_rows = 1 if 'CometVersion' in first_line else 0
    df = pd.read_csv(csv_file, sep=',', skiprows=skip_rows, engine='python', quotechar='"', on_bad_lines='warn')
    if 'plain_peptide' not in df.columns:
        return False
    try:
        from visualization.csv_validation import validate_and_log
        validate_and_log(
            df,
            source_name=os.path.basename(csv_file) + ' (filtered write)',
            required_columns=('plain_peptide', 'charge'),
            validate_identifiers=True,
            validate_numeric=True,
        )
    except Exception as e:
        print(f"  Validation warning: {e}")
    df['_mods_norm'] = df.apply(lambda row: _norm_mods(row.get('modifications', '-')), axis=1)
    seq_strip = df['plain_peptide'].astype(str).str.strip()
    charge_int = df['charge'].fillna(0).astype(int)
    mask = [k in filtered_keys for k in zip(seq_strip, charge_int, df['_mods_norm'])]
    df = df.drop(columns=['_mods_norm']).loc[mask]
    if 'MS1_retention_time_sec' not in df.columns and 'retention_time_sec' in df.columns:
        df = df.rename(columns={'retention_time_sec': 'MS1_retention_time_sec'})
    df.to_csv(output_path, index=False)
    return True


def get_versioned_results_directory():
    """
    Create a versioned results directory (results, results_v2, results_v3, etc.)
    Returns the path to the results directory.
    """
    base_dir = 'results'
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        print(f"Created results directory: {base_dir}")
        return base_dir
    
    # Find the highest version number
    version = 1
    while os.path.exists(f'{base_dir}_v{version}'):
        version += 1
    
    results_dir = f'{base_dir}_v{version}'
    os.makedirs(results_dir)
    print(f"Created versioned results directory: {results_dir}")
    return results_dir


def calculate_geometric_quality_metric(overhang, tau=None):
    """
    Calculate a quality/uncertainty metric based only on geometric properties:
    - Peptide length (N)
    - Fragment length (fragment ordinal)
    - Overhang length (L)
    
    Does NOT require D_total, as D content is an unknown variable.
    
    The metric is based on resolution uncertainty normalized by overhang length,
    giving a dimensionless quality score where lower values indicate better quality.
    
    Parameters:
    -----------
    overhang : dict
        Overhang dictionary with 'peptide_seq', 'length', 'fragment_ordinal'
    tau : float, optional
        Resolution heterogeneity scale (default: 0.1732)
    
    Returns:
    --------
    quality_metric : float
        Quality metric (0-100 scale, lower is better)
    """
    if tau is None:
        tau = 0.1732  # Default value
    
    peptide_seq = overhang.get('peptide_seq', '')
    clean_peptide = re.sub(r'\[.*?\]', '', peptide_seq) if peptide_seq else ''
    
    if not clean_peptide:
        return 100.0  # Worst quality if no peptide
    
    peptide_length = len(clean_peptide)
    overhang_length = overhang.get('length', 1)
    fragment_ordinal = overhang.get('fragment_ordinal', None)
    
    if peptide_length <= 0 or overhang_length <= 0:
        return 100.0
    
    N = float(peptide_length)
    L = float(overhang_length)
    
    # Calculate resolution uncertainty based only on geometric properties
    if L == 1:
        # For single-AA overhangs, resolution depends on fragment ordinal
        if fragment_ordinal is not None and fragment_ordinal > 0:
            # Longer fragments (higher ordinal) have better resolution
            sigma_res = tau / np.sqrt(fragment_ordinal)
        else:
            # Default: assume moderate fragment length
            sigma_res = tau / np.sqrt(N / 2.0)  # Assume fragment around middle of peptide
    else:
        # For multi-AA overhangs: σ_res = τ√L
        sigma_res = tau * np.sqrt(L)
    
    # Normalize by overhang length to get a dimensionless metric
    # This gives us a relative uncertainty that doesn't depend on D_total
    # For single-AA: normalize by 1 (single residue)
    # For multi-AA: normalize by L (overhang length)
    if L == 1:
        # For single-AA, use fragment ordinal as normalization factor
        if fragment_ordinal is not None and fragment_ordinal > 0:
            normalization = fragment_ordinal
        else:
            normalization = N / 2.0  # Default normalization
        relative_uncertainty = sigma_res / normalization
    else:
        # For multi-AA, normalize by overhang length
        # This gives us uncertainty per residue in the overhang
        relative_uncertainty = sigma_res / L
    
    # Convert to a quality metric (0-100 scale, lower is better)
    # Scale so that typical values are in a reasonable range
    # Use a transformation that makes longer overhangs and higher fragment ordinals have better (lower) scores
    quality_metric = relative_uncertainty * 100.0
    
    # Cap at 100 (worst quality)
    quality_metric = min(quality_metric, 100.0)
    
    return quality_metric


def calculate_composite_quality_score(peptide_dict, charge=None, peptide_length=None):
    """
    Calculate a composite quality score combining Comet metrics (no Xcorr).
    
    Combines: ΔCn (uniqueness), E-value (statistical significance), Sp rank, matched ion fraction.
    
    Args:
        peptide_dict: Dictionary containing Comet metrics (delta_cn, e_value, sp_rank, ions_matched, ions_total)
        charge: Charge state (optional, for future stratification)
        peptide_length: Peptide length (optional, for future stratification)
    
    Returns:
        Composite quality score (higher is better)
    """
    delta_cn = peptide_dict.get('delta_cn', 0.0)
    e_value = peptide_dict.get('e_value', 1.0)  # Default to worst case if missing
    sp_rank = peptide_dict.get('sp_rank', float('inf'))  # Default to worst case if missing
    ions_matched = peptide_dict.get('ions_matched', 0)
    ions_total = peptide_dict.get('ions_total', 1)  # Avoid division by zero
    
    # Calculate matched ion fraction
    ion_fraction = ions_matched / ions_total if ions_total > 0 else 0.0
    
    # ΔCn: 0-1 scale, higher is better (uniqueness)
    delta_cn_component = delta_cn * 50.0  # Weight: 50x (very important for uniqueness)
    
    # E-value: lower is better, log-transform and invert
    if e_value > 0:
        e_value_component = -np.log10(e_value + 1e-10) * 5.0  # Weight: 5x
    else:
        e_value_component = 0.0
    
    # Sp rank: lower is better, invert (1/rank)
    if sp_rank > 0 and sp_rank < float('inf'):
        sp_rank_component = (1.0 / sp_rank) * 20.0  # Weight: 20x, lower rank = higher score
    else:
        sp_rank_component = 0.0
    
    # Ion fraction: 0-1 scale, higher is better
    ion_fraction_component = ion_fraction * 15.0  # Weight: 15x
    
    composite_score = (delta_cn_component + e_value_component + sp_rank_component + ion_fraction_component)
    return composite_score


def is_c_supported(peptide_dict, fragment_pairs_dict):
    """
    Check if a peptide is c-supported (has c ion fragment pairs).
    
    Args:
        peptide_dict: Peptide dictionary
        fragment_pairs_dict: Dictionary mapping positions to sets of normalized fragment pairs
    
    Returns:
        True if peptide has at least one c ion fragment pair, False otherwise
    """
    for pos, pairs_set in fragment_pairs_dict.items():
        for pair in pairs_set:
            normalized_pair = normalize_fragment_pair(pair)
            if normalized_pair.startswith('c') and '-' in normalized_pair:
                parts = normalized_pair.split('-')
                if len(parts) == 2 and parts[1].startswith('c'):
                    return True
    return False


def is_z_supported(peptide_dict, fragment_pairs_dict):
    """
    Check if a peptide is z-supported (has z/z+1 ion fragment pairs).
    
    Args:
        peptide_dict: Peptide dictionary
        fragment_pairs_dict: Dictionary mapping positions to sets of normalized fragment pairs
    
    Returns:
        True if peptide has at least one z/z+1 ion fragment pair, False otherwise
    """
    for pos, pairs_set in fragment_pairs_dict.items():
        for pair in pairs_set:
            normalized_pair = normalize_fragment_pair(pair)
            if normalized_pair.startswith('z') and '-' in normalized_pair:
                parts = normalized_pair.split('-')
                if len(parts) == 2 and parts[1].startswith('z'):
                    return True
    return False


def get_boundary_cluster_key(peptide_dict):
    """
    Get a boundary cluster key for a peptide based on its start and end positions.
    Peptides with similar spans are grouped together.
    
    Args:
        peptide_dict: Peptide dictionary with 'start' and 'end' keys
    
    Returns:
        Tuple (start, end) representing the boundary cluster
    """
    return (peptide_dict.get('start', 0), peptide_dict.get('end', 0))


def two_pass_selection_algorithm(peptides, protein_length, Kc=3, Kz=2, max_per_residue=3, debug=False):
    """
    Implement the two-pass selection algorithm:
    Pass 1: Build c-anchored boundary set (top Kc c-supported peptides per position/cluster)
    Pass 2: Add z-supported reach peptides (top Kz per region that extend/bridge/add variety)
    Pass 3: Cap redundancy (max peptides per cluster/residue)
    
    Args:
        peptides: List of peptide dictionaries with xcorr, start, end, single_aa_fragment_pairs, etc.
        protein_length: Length of the protein
        Kc: Number of c-supported peptides to keep per position/cluster (default: 3)
        Kz: Number of z-supported peptides to add per region (default: 2)
        max_per_residue: Maximum peptides per residue after all passes (default: 3)
    
    Returns:
        List of selected peptide dictionaries
    
    Note: This algorithm uses quality_score (q-value) for ranking when available.
    """
    selected_peptides = []
    selected_peptide_keys = set()  # Track (peptide_seq, rt, charge) to avoid duplicates
    
    # Separate peptides by ion support
    c_supported_peptides = []
    z_supported_peptides = []
    
    for peptide in peptides:
        peptide_key = (peptide.get('peptide_seq', ''), peptide.get('rt', 0.0), peptide.get('charge', 0))
        
        # Collect all fragment pairs from single-AA and multi-AA overhangs
        all_fragment_pairs = {}
        
        # Add single-AA fragment pairs
        single_aa_pairs = peptide.get('single_aa_fragment_pairs', {})
        for pos, pairs_set in single_aa_pairs.items():
            if pos not in all_fragment_pairs:
                all_fragment_pairs[pos] = set()
            all_fragment_pairs[pos].update(pairs_set)
        
        # Add multi-AA fragment pairs (all lengths 1aa through 20aa)
        for overhang_type in OVERHANG_TYPE_KEYS:
            multi_aa_pairs = peptide.get(f'{overhang_type}_fragment_pairs', {})
            for pos, pairs_set in multi_aa_pairs.items():
                if pos not in all_fragment_pairs:
                    all_fragment_pairs[pos] = set()
                all_fragment_pairs[pos].update(pairs_set)
        
        # Check if peptide is c-supported or z-supported
        has_c = is_c_supported(peptide, all_fragment_pairs)
        has_z = is_z_supported(peptide, all_fragment_pairs)
        
        if has_c:
            c_supported_peptides.append((peptide, peptide_key))
        if has_z:
            z_supported_peptides.append((peptide, peptide_key))
    
    # PASS 1: Build c-anchored boundary set
    # Group by position and boundary cluster
    position_to_c_peptides = {}  # position -> list of (peptide, key, xcorr, boundary_key)
    boundary_clusters = {}  # boundary_key -> list of (peptide, key, xcorr)
    
    for peptide, peptide_key in c_supported_peptides:
        # Use quality_score (q-value) only for sorting; no Xcorr fallback
        quality_score = peptide.get('quality_score')
        if quality_score is not None:
            score_for_sorting = -quality_score  # lower q-value = higher = better
        else:
            score_for_sorting = 0.0
        
        boundary_key = get_boundary_cluster_key(peptide)
        
        # Track by position
        for pos in range(peptide.get('start', 0), peptide.get('end', 0) + 1):
            if 1 <= pos <= protein_length:
                if pos not in position_to_c_peptides:
                    position_to_c_peptides[pos] = []
                position_to_c_peptides[pos].append((peptide, peptide_key, score_for_sorting, boundary_key))
        
        # Track by boundary cluster
        if boundary_key not in boundary_clusters:
            boundary_clusters[boundary_key] = []
        boundary_clusters[boundary_key].append((peptide, peptide_key, score_for_sorting))
    
    # Select top Kc per position, prioritizing boundary diversity
    pass1_stats = {'positions_with_c': 0, 'total_unique_boundaries': 0, 'positions_limited_by_boundaries': 0}
    for pos in range(1, protein_length + 1):
        if pos not in position_to_c_peptides:
            continue
        
        pass1_stats['positions_with_c'] += 1
        candidates = position_to_c_peptides[pos]
        # Sort by score_for_sorting (descending); from quality_score (q-value) only
        candidates.sort(key=lambda x: x[2], reverse=True)
        
        # Track boundary clusters used for this position
        used_boundaries = set()
        count = 0
        all_boundaries = set(boundary_key for _, _, _, boundary_key in candidates)
        pass1_stats['total_unique_boundaries'] += len(all_boundaries)
        
        for peptide, peptide_key, score_for_sorting, boundary_key in candidates:
            if count >= Kc:
                break
            
            # Prefer peptides with unique boundaries (boundary diversity)
            if boundary_key not in used_boundaries or len(used_boundaries) < Kc:
                if peptide_key not in selected_peptide_keys:
                    selected_peptides.append(peptide)
                    selected_peptide_keys.add(peptide_key)
                    used_boundaries.add(boundary_key)
                    count += 1
        
        # Track if this position was limited by number of unique boundaries
        if len(used_boundaries) < Kc and len(used_boundaries) < len(all_boundaries):
            pass1_stats['positions_limited_by_boundaries'] += 1
    
    # PASS 2: Add z-supported reach peptides
    # Identify coverage gaps and add z-supported peptides that extend/bridge
    position_coverage = {}  # position -> count of peptides covering it
    for peptide in selected_peptides:
        for pos in range(peptide.get('start', 0), peptide.get('end', 0) + 1):
            if 1 <= pos <= protein_length:
                position_coverage[pos] = position_coverage.get(pos, 0) + 1
    
    # Find positions with low coverage (gaps)
    gap_positions = [pos for pos in range(1, protein_length + 1) 
                     if position_coverage.get(pos, 0) < 2]
    
    pass2_stats = {'gap_positions': len(gap_positions), 'positions_covered_by_pass1': len(position_coverage)}
    
    # Select z-supported peptides that cover gaps or extend spans
    z_candidates_by_position = {}  # position -> list of (peptide, key, xcorr)
    
    for peptide, peptide_key in z_supported_peptides:
        if peptide_key in selected_peptide_keys:
            continue  # Already selected
        
        # Use quality_score (q-value) only; no Xcorr fallback
        quality_score = peptide.get('quality_score')
        if quality_score is not None:
            score_for_sorting = -quality_score
        else:
            score_for_sorting = 0.0
        
        peptide_start = peptide.get('start', 0)
        peptide_end = peptide.get('end', 0)
        
        # Check if this peptide covers gaps or extends coverage
        covers_gap = any(pos in gap_positions for pos in range(peptide_start, peptide_end + 1))
        extends_span = peptide_start < min(position_coverage.keys()) if position_coverage else True
        extends_span = extends_span or peptide_end > max(position_coverage.keys()) if position_coverage else True
        
        if covers_gap or extends_span:
            for pos in range(peptide_start, peptide_end + 1):
                if 1 <= pos <= protein_length:
                    if pos not in z_candidates_by_position:
                        z_candidates_by_position[pos] = []
                    z_candidates_by_position[pos].append((peptide, peptide_key, score_for_sorting))
    
    # Select top Kz z-supported peptides per position
    for pos in range(1, protein_length + 1):
        if pos not in z_candidates_by_position:
            continue
        
        candidates = z_candidates_by_position[pos]
        candidates.sort(key=lambda x: x[2], reverse=True)
        
        count = 0
        for peptide, peptide_key, score_for_sorting in candidates:
            if count >= Kz:
                break
            if peptide_key not in selected_peptide_keys:
                selected_peptides.append(peptide)
                selected_peptide_keys.add(peptide_key)
                count += 1
    
    # PASS 3: Cap redundancy by single-AA overhangs per residue
    # max_per_residue now means maximum single-AA overhangs per position, not peptides
    # Build map of position -> list of (peptide_key, score, overhangs_at_this_position)
    position_to_peptide_data = {}  # position -> list of (peptide_key, score, overhang_pairs_set)
    
    for peptide in selected_peptides:
        peptide_key = (peptide.get('peptide_seq', ''), peptide.get('rt', 0.0), peptide.get('charge', 0))
        # Use quality_score (q-value) only; no Xcorr fallback
        quality_score = peptide.get('quality_score')
        if quality_score is not None:
            score_for_sorting = -quality_score
        else:
            score_for_sorting = 0.0
        
        single_aa_pairs = peptide.get('single_aa_fragment_pairs', {})
        for pos, pairs_set in single_aa_pairs.items():
            if 1 <= pos <= protein_length:
                if pos not in position_to_peptide_data:
                    position_to_peptide_data[pos] = []
                position_to_peptide_data[pos].append((peptide_key, score_for_sorting, pairs_set))
    
    # For each position, select peptides greedily to maximize overhang count while staying under max_per_residue
    # Sort by score (descending) and add peptides until we reach max_per_residue overhangs
    final_selected_keys = set()
    
    for pos in range(1, protein_length + 1):
        if pos not in position_to_peptide_data:
            continue
        
        # Sort peptides by score (descending)
        candidates = position_to_peptide_data[pos]
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        overhang_count = 0
        for peptide_key, score_for_sorting, pairs_set in candidates:
            if overhang_count >= max_per_residue:
                break
            # Add this peptide if it doesn't exceed the limit
            num_new_overhangs = len(pairs_set)
            if overhang_count + num_new_overhangs <= max_per_residue:
                final_selected_keys.add(peptide_key)
                overhang_count += num_new_overhangs
            elif overhang_count < max_per_residue:
                # Partially add: we can add some overhangs but not all
                # For simplicity, add the peptide if it has at least one overhang that fits
                final_selected_keys.add(peptide_key)
                overhang_count = max_per_residue  # Cap it
    
    # Filter selected peptides to only include those in final_selected_keys
    final_peptides = []
    for peptide in selected_peptides:
        peptide_key = (peptide.get('peptide_seq', ''), peptide.get('rt', 0.0), peptide.get('charge', 0))
        if peptide_key in final_selected_keys:
            final_peptides.append(peptide)
    
    if debug:
        print(f"    DEBUG: Pass 1 - {pass1_stats['positions_with_c']} positions with c-peptides, "
              f"{pass1_stats['total_unique_boundaries']} total unique boundaries, "
              f"{pass1_stats['positions_limited_by_boundaries']} positions limited by boundary diversity")
        print(f"    DEBUG: Pass 2 - {pass2_stats['gap_positions']} gap positions, "
              f"{pass2_stats['positions_covered_by_pass1']} positions covered by Pass 1")
        print(f"    DEBUG: Pass 3 - {len(selected_peptides)} peptides before capping, "
              f"{len(final_peptides)} peptides after max_per_residue={max_per_residue} capping")
    
    return final_peptides


def read_fill_times_from_mzml(mzml_file):
    """
    Read ion injection time (fill time) values from mzML file, indexed by scan number.
    Returns a dictionary mapping scan_number -> fill_time_ms
    """
    fill_times = {}
    try:
        import xml.etree.ElementTree as ET
        tree = ET.parse(mzml_file)
        root = tree.getroot()
        
        # Define namespaces
        ns = {'mzML': 'http://psi.hupo.org/ms/mzml'}
        
        for spectrum in root.findall('.//mzML:spectrum', ns):
            # Get scan number - try multiple methods
            scan_num = None
            
            # Method 1: Look for scan number in scanList (cvParam MS:1000094)
            scan_list = spectrum.find('.//mzML:scanList/mzML:scan', ns)
            if scan_list is not None:
                scan_num_elem = scan_list.find('.//mzML:cvParam[@accession="MS:1000094"]', ns)
                if scan_num_elem is not None:
                    scan_num = int(scan_num_elem.get('value', '0'))
            
            # Method 2: Parse spectrum id="... scan=783" (Thermo/ProteoWizard format)
            if (scan_num is None or scan_num == 0) and spectrum.get('id'):
                id_match = re.search(r'scan=(\d+)', spectrum.get('id', ''))
                if id_match:
                    scan_num = int(id_match.group(1))
            
            # Method 3: Use spectrum index + 1 (if scan number not found)
            if scan_num is None or scan_num == 0:
                try:
                    scan_num = int(spectrum.get('index', '0')) + 1
                except Exception:
                    continue
            
            # Get ion injection time
            iit_elem = spectrum.find('.//mzML:cvParam[@accession="MS:1000927"]', ns)
            if iit_elem is not None:
                fill_time = float(iit_elem.get('value', '0'))
                # Check unit - MS:1000927 should be in milliseconds, but check unitAccession to be sure
                # UO:0000028 = millisecond, UO:0000010 = second
                unit_accession = iit_elem.get('unitAccession', '')
                unit_name = iit_elem.get('unitName', '')
                # Only convert if unit is explicitly in seconds
                # Do NOT convert based on value size - some legitimate fill times can be < 10ms
                if unit_accession == 'UO:0000010' or unit_name.lower() == 'second' or unit_name.lower() == 'seconds':
                    fill_time = fill_time * 1000.0
                # If no unit specified, assume milliseconds (MS:1000927 standard is milliseconds)
                if fill_time > 0:
                    fill_times[scan_num] = fill_time
        
        if fill_times:
            print(f"Read {len(fill_times)} fill time values from mzML (range: {min(fill_times.values()):.2f}-{max(fill_times.values()):.2f} ms)")
        else:
            print("Warning: No fill time values found in mzML file")
    except Exception as e:
        print(f"Warning: Could not read fill times from mzML: {e}")
    
    return fill_times

def create_combined_visualization(csv_file, protein_id, fasta_file, output_file, ion_type_filter=None, no_filtering=False, use_two_pass=False, no_progressive=False, no_two_pass=False, mzml_file=None, peak_windows_csv=None, filter_by_peak_window_status=None, test=False, filter_criteria=None):
    # Debug: Print ion_type_filter to verify it's correct
    print(f"DEBUG: create_combined_visualization called with ion_type_filter={ion_type_filter}, output_file={output_file}, test={test}")
    """
    Create a combined visualization with:
    1. Multi-AA overhang heatmap (top)
    2. Histogram of fragment pairs and peptides (middle)
    3. Peptide RT grid (bottom)
    
    All plots share the same x-axis.
    
    Args:
        csv_file: Path to Comet CSV output file
        protein_id: Protein identifier
        fasta_file: Path to FASTA file
        output_file: Output PNG file path (will be saved to results directory)
        ion_type_filter: 'c' for c ions only, 'z' for z/z+1 ions only, None for all
        no_filtering: If True, skip all filtering and include all peptides with overhangs
        use_two_pass: If True, use the two-pass selection algorithm (c-anchored + z-supported)
        no_progressive: If True, skip progressive selection (gap-filling with 2-AA and 3-AA overhangs)
        no_two_pass: If True, skip two-pass selection algorithm
        mzml_file: Optional mzML file for reading fill times
        peak_windows_csv: Optional path to peak_windows CSV file (peak_windows_all.csv, peak_windows.csv, or peak_windows_rejected.csv)
        filter_by_peak_window_status: Optional filter by peak window status ('accepted' or 'rejected'). If None, no filtering by peak window status.
        test: If True, use only 30 randomly sampled peptides (same as --test in chromatograms) for all plots and comet_frags_filtered_by_qvalue_pepscore_mods_minaa.csv.
        filter_criteria: Optional dict from run_visualization.py (top_n_limit, min_single_aa_overhangs, min_scans_combined, ppm_tolerance,
            high_overhang_threshold, two_pass_Kc, two_pass_Kz, two_pass_max_per_residue, test_sample_size). If None, defaults are used.
    """
    import numpy as np
    
    # Resolve filter criteria from run_visualization.py (single source of truth); use defaults if not provided
    fc = filter_criteria if isinstance(filter_criteria, dict) else {}
    _min_scans_combined = fc.get('min_scans_combined', 1)
    _min_single_aa_overhangs = fc.get('min_single_aa_overhangs', 2)
    _ppm_tolerance = fc.get('ppm_tolerance', 5.0)
    _reject_mods = fc.get('reject_mods', True)
    _require_m0_gt_m1_gt_m2 = fc.get('require_m0_gt_m1_gt_m2', True)
    _high_overhang_threshold = fc.get('high_overhang_threshold', 3)
    _test_sample_size = fc.get('test_sample_size', 30)
    _total_area_apex_percentile = fc.get('total_area_apex_percentile', 25)
    _protect_total_area_percentile = fc.get('protect_total_area_percentile', 75)

    # Create versioned results directory (only on first call, reuse for subsequent calls)
    # Use a module-level variable to track if we've already created the directory
    if not hasattr(create_combined_visualization, '_results_dir'):
        create_combined_visualization._results_dir = get_versioned_results_directory()
    results_dir = create_combined_visualization._results_dir
    
    # Update output_file path to be in results directory; remove "combined_overhang_visualization" prefix from all plot filenames
    output_filename = os.path.basename(output_file)
    if output_filename.startswith("combined_overhang_visualization_"):
        output_filename = output_filename[len("combined_overhang_visualization_"):]
    elif output_filename.startswith("combined_overhang_visualization"):
        output_filename = output_filename[len("combined_overhang_visualization"):].lstrip("_") or "visualization"
    output_file = os.path.join(results_dir, output_filename)
    
    # Parse FASTA
    sequences = parse_fasta(fasta_file)
    if not sequences:
        print("Error: No sequences found in FASTA file")
        return
    
    if protein_id:
        clean_protein_id = re.sub(r'(\d+)$', '', protein_id)
        protein_id_alt = re.sub(r'^sp\|', '', clean_protein_id)
        if clean_protein_id in sequences:
            protein_sequence = sequences[clean_protein_id]
        elif protein_id_alt in sequences:
            protein_sequence = sequences[protein_id_alt]
        elif protein_id in sequences:
            protein_sequence = sequences[protein_id]
        else:
            print(f"Error: Protein {protein_id} not found in FASTA file")
            return
    else:
        protein_id = list(sequences.keys())[0]
        protein_sequence = sequences[protein_id]
    
    protein_length = len(protein_sequence)
    
    # Load peak window data - PREFER separate peak_windows CSV when provided (authoritative chromatogram filter)
    peak_windows_data = {}
    has_peak_window_columns = False
    
    # When user provides --peak-windows-csv, always use it for chromatogram filtering (do not use merged columns)
    if peak_windows_csv and os.path.exists(peak_windows_csv):
        print(f"Loading peak window data from separate CSV: {peak_windows_csv}")
        try:
            import pandas as pd
            df_peak_windows = pd.read_csv(peak_windows_csv)
            try:
                from visualization.csv_validation import validate_and_log
                validate_and_log(
                    df_peak_windows,
                    source_name=os.path.basename(peak_windows_csv),
                    required_columns=[],
                    validate_identifiers=False,
                    validate_numeric=True,
                    validate_windows=True,
                )
            except Exception as e:
                print(f"  Validation warning: {e}")
            # Normalize keys to match main CSV: strip whitespace, consistent mods
            # Accepted-only CSV (peak_windows.csv) has no 'status' column; treat all as 'accepted'
            has_status_column = 'status' in df_peak_windows.columns
            for _, row in df_peak_windows.iterrows():
                peptide_seq = str(row.get('peptide', '')).strip()
                charge = int(row.get('charge', 0)) if pd.notna(row.get('charge')) else 0
                mods = str(row.get('modifications', '-')).strip() if pd.notna(row.get('modifications')) else '-'
                if not mods or mods.lower() == 'nan':
                    mods = '-'
                key = (peptide_seq, charge, mods)
                peak_windows_data[key] = {
                    'status': row.get('status', 'accepted') if has_status_column else 'accepted',
                    'rejection_reason': row.get('rejection_reason', ''),
                    'detected_peak_min_rt': row.get('detected_peak_min_rt'),
                    'detected_peak_max_rt': row.get('detected_peak_max_rt'),
                    'detected_peak_window_size': row.get('detected_peak_window_window_size'),
                    'spectrum_window_min_rt': row.get('spectrum_window_min_rt'),
                    'spectrum_window_max_rt': row.get('spectrum_window_max_rt'),
                    'spectrum_window_size': row.get('spectrum_window_size'),
                    'apex_rt': row.get('apex_rt'),
                    'anchor_rt': row.get('anchor_rt'),
                    'anchor_evalue': row.get('anchor_evalue'),
                    'n_isotopic_matches': row.get('n_isotopic_matches', 0),
                    'matched_isotopes': row.get('matched_isotopes', ''),
                    'using_anchor_window': row.get('using_anchor_window', False),
                    'total_area': row.get('total_area'),
                    'apex_intensity': row.get('apex_intensity'),
                    'm0_gt_m1_gt_m2': row.get('m0_gt_m1_gt_m2')
                }
            print(f"Loaded peak window data for {len(peak_windows_data)} peptides (will restrict to accepted only)")
        except Exception as e:
            print(f"Warning: Could not load peak window CSV: {e}")
            peak_windows_data = {}
    elif peak_windows_csv:
        print(f"Warning: Peak windows CSV file not found: {peak_windows_csv}")
    
    # Only use merged CSV columns when no separate peak_windows CSV was provided
    if not peak_windows_data:
        try:
            with open(csv_file, 'r') as f:
                sample_lines = f.readlines()
                if len(sample_lines) > 0:
                    header_line_idx = 0
                    if len(sample_lines) > 1 and 'CometVersion' in sample_lines[0]:
                        header_line_idx = 1
                    if header_line_idx < len(sample_lines):
                        header_line = sample_lines[header_line_idx]
                        header = [col.strip() for col in header_line.split(',')]
                        if 'peak_window_status' in header:
                            has_peak_window_columns = True
                            print(f"Peak window columns detected in main CSV (no separate CSV provided)")
        except Exception:
            pass
        if has_peak_window_columns:
            print(f"Peak window data will be extracted from merged CSV columns")
    
    # Initialize q-value column tracking (will be set when parsing CSV)
    qvalue_idx = -1
    qvalue_column_name = None
    qvalue_is_score = False  # True if values are q-scores (log-transformed), False if true q-values (0-1)
    qvalue_is_percent = False  # True if values are percentages (1-100), False if decimal (0-1)
    
    # Parse peptides from CSV (all protein-matching rows; filtering is applied later: min 1 scan)
    # Parse peptides for RT grid first (same as create_peptide_rt_grid)
    # We need this to get filtered peptide sequences for filtering single-AA overhangs
    # Try to read fill times from mzML if provided and CSV has no/missing fill time column
    mzml_fill_times = {}
    if mzml_file:
        with open(csv_file, 'r') as f:
            sample_lines = f.readlines()
        # Detect header line (line 0 or 1)
        h_idx = 1
        if sample_lines and 'protein' in sample_lines[0]:
            h_idx = 0
        elif len(sample_lines) > 1 and 'protein' in sample_lines[1]:
            h_idx = 1
        header = [col.strip() for col in sample_lines[h_idx].split(',')] if len(sample_lines) > h_idx else []
        # Try to find fill time column (multiple possible names, then case-insensitive)
        fill_time_col_idx = -1
        fill_time_col_names = [
            'ion_injection_time_ms', 'ion_injection_time', 'IonInjectionTimeMs', 'IonInjectionTime', 'IIT',
            'fill_time', 'fill time', 'Fill Time (ms)', 'ion injection time (ms)', 'ion injection time'
        ]
        for col_name in fill_time_col_names:
            if col_name in header:
                fill_time_col_idx = header.index(col_name)
                break
        if fill_time_col_idx < 0:
            header_lower = [h.strip().strip('"').lower() for h in header]
            for col_name in fill_time_col_names:
                cn_lower = col_name.lower()
                if cn_lower in header_lower:
                    fill_time_col_idx = header_lower.index(cn_lower)
                    break
        use_mzml_for_fill = False
        if fill_time_col_idx >= 0 and len(sample_lines) > h_idx + 1:
            sample_values = []
            for line in sample_lines[h_idx + 1:min(h_idx + 12, len(sample_lines))]:
                fields = line.split(',')
                if len(fields) > fill_time_col_idx:
                    try:
                        val = float(fields[fill_time_col_idx].strip('"'))
                        if val > 0:
                            if 0 < val < 1:
                                val = val * 1000.0
                            sample_values.append(val)
                    except Exception:
                        pass
            if sample_values and max(sample_values) < 10:
                print(f"Warning: CSV fill time values seem too small (max={max(sample_values):.2f} ms).")
                use_mzml_for_fill = True
        else:
            # No fill time column in CSV - use mzML
            use_mzml_for_fill = True
        if use_mzml_for_fill:
            print(f"Reading fill times from mzML file: {mzml_file}")
            mzml_fill_times = read_fill_times_from_mzml(mzml_file)
    
    peptides = []
    
    with open(csv_file, 'r') as f:
        lines = f.readlines()
        
        if len(lines) < 1:
            print("Error: CSV file too short")
            return
        
        # Detect header line: check if line 0 or line 1 contains 'protein' column
        header_line_idx = 1  # Default: expect CometVersion on line 0, header on line 1
        if len(lines) > 0:
            # Check if line 0 contains 'protein' (header is on first line)
            if 'protein' in lines[0]:
                header_line_idx = 0
            # Otherwise, check line 1
            elif len(lines) > 1 and 'protein' in lines[1]:
                header_line_idx = 1
        
        if header_line_idx >= len(lines):
            print("Error: CSV file too short")
            return
        
        header_line = lines[header_line_idx]
        header = [col.strip() for col in header_line.split(',')]
        
        try:
            protein_idx = header.index('protein')
            scan_idx = header.index('scan')
            peptide_idx = header.index('plain_peptide')
            xcorr_idx = header.index('xcorr')
            modifications_idx = header.index('modifications')
            intensities_idx = header.index('matched fragment ion intensities')
            charge_idx = header.index('charge')
            single_aa_pairs_idx = header.index('single_aa_overhang_fragment_pairs')
            try:
                matched_ions_idx = header.index('matched fragment ions')
            except ValueError:
                matched_ions_idx = -1
            # Fill time (ion injection time): try multiple column names (including case-insensitive); fallback to mzML by scan if provided
            ion_injection_time_idx = -1
            fill_time_col_names = [
                'ion_injection_time_ms', 'ion_injection_time', 'IonInjectionTimeMs', 'IonInjectionTime', 'IIT',
                'fill_time', 'fill time', 'Fill Time (ms)', 'ion injection time (ms)', 'ion injection time'
            ]
            for col_name in fill_time_col_names:
                if col_name in header:
                    ion_injection_time_idx = header.index(col_name)
                    break
            if ion_injection_time_idx < 0:
                header_clean_lower = [h.strip().strip('"').lower() for h in header]
                for col_name in fill_time_col_names:
                    if col_name.lower() in header_clean_lower:
                        ion_injection_time_idx = header_clean_lower.index(col_name.lower())
                        break
            if ion_injection_time_idx < 0:
                print("Warning: No fill time column found (tried: ion_injection_time_ms, ion_injection_time, fill_time, etc.). Fill time will be set to 0 or from mzML if --mzml is provided.")
            # MS1 intensity at retention time (apex proxy for 25th-percentile filter)
            ms1_intensity_idx = -1
            for col_name in ['MS1_retention_time_intensity', 'ms1_retention_time_intensity', 'MS1_intensity']:
                if col_name in header:
                    ms1_intensity_idx = header.index(col_name)
                    break
            # Precursor m/z (optional): for y-axis labels on RT grid and RT windows plot
            # Comet CSV uses "mz"; also try common variants, case-insensitive and strip quotes
            mz_idx = -1
            header_clean = [h.strip().strip('"').strip() for h in header]
            for col_name in ['mz', 'spectrum precursor m/z', 'precursor_mz', 'precursor m/z', 'exp m/z', 'calc m/z', 'precursorMZ', 'precursor_m/z', 'MZ', 'Precursor MZ']:
                for i, h in enumerate(header_clean):
                    if h.lower() == col_name.lower():
                        mz_idx = i
                        break
                if mz_idx >= 0:
                    break
            if mz_idx >= 0:
                print(f"Using precursor m/z column: '{header[mz_idx].strip()}' (index {mz_idx})")
            # Theoretical m/z from calc_neutral_mass (optional): for RT windows plots
            calc_neutral_mass_idx = -1
            for col_name in ['calc_neutral_mass', 'calc neutral mass', 'CalcNeutralMass']:
                for i, h in enumerate(header_clean):
                    if h.lower() == col_name.lower():
                        calc_neutral_mass_idx = i
                        break
                if calc_neutral_mass_idx >= 0:
                    break
            if calc_neutral_mass_idx >= 0:
                print(f"Using calc_neutral_mass column for theoretical m/z (index {calc_neutral_mass_idx})")
        except ValueError as e:
            print(f"Error: Required column not found: {e}")
            return
        
        # Retention time: require MS1 (OpenMS RT) only - try MS1_retention_time_sec, then retention_time_sec (Comet/Percolator CSV)
        rt_idx = -1
        for col in ['MS1_retention_time_sec', 'retention_time_sec', 'MS1_retention_time', 'retention_time']:
            if col in header:
                rt_idx = header.index(col)
                break
        if rt_idx < 0:
            print("Error: No retention time column found (tried: MS1_retention_time_sec, retention_time_sec, MS1_retention_time, retention_time). Plots use MS1 retention time only.")
            return
        
        # Single-AA overhangs: try single_aa_overhangs, then single_aa_overhangs_protein_positions
        single_aa_idx = -1
        for col_name in ['single_aa_overhangs', 'single_aa_overhangs_protein_positions']:
            try:
                single_aa_idx = header.index(col_name)
                break
            except ValueError:
                continue
        if single_aa_idx < 0:
            print(f"Error: Required column not found: single_aa_overhangs (or single_aa_overhangs_protein_positions)")
            return
        
        # Site-specific residues: try site_specific_residues, then single_aa_overhangs_protein_positions
        site_specific_idx = -1
        for col_name in ['site_specific_residues', 'single_aa_overhangs_protein_positions']:
            try:
                site_specific_idx = header.index(col_name)
                break
            except ValueError:
                continue
        if site_specific_idx < 0:
            print(f"Error: Required column not found: site_specific_residues (or single_aa_overhangs_protein_positions)")
            return
        
        # Check for peak window columns (if merged into CSV)
        peak_window_col_indices = {}
        peak_window_columns = [
            'peak_window_status', 'peak_window_rejection_reason',
            'detected_peak_min_rt', 'detected_peak_max_rt', 'detected_peak_window_size',
            'spectrum_window_min_rt', 'spectrum_window_max_rt', 'spectrum_window_size',
            'apex_rt', 'anchor_rt', 'anchor_evalue',
            'n_isotopic_matches', 'matched_isotopes', 'using_anchor_window',
            'total_area', 'apex_intensity', 'm0_gt_m1_gt_m2'
        ]
        for col_name in peak_window_columns:
            try:
                peak_window_col_indices[col_name] = header.index(col_name)
            except ValueError:
                peak_window_col_indices[col_name] = -1
        
        if any(idx >= 0 for idx in peak_window_col_indices.values()):
            has_peak_window_columns = True
            print(f"Found peak window columns in CSV (merged data)")
            if peak_window_col_indices.get('total_area', -1) >= 0:
                print(f"  total_area column present: will color RT window plot by MS1 peak area")
        
        # Try to find q-value/FDR column (Percolator output)
        if qvalue_idx == -1:  # Only search if not already found
            for col_name in ['percolator_qvalue', 'q-value', 'qvalue', 'FDR', 'fdr', 'percolator_q-value']:
                try:
                    qvalue_idx = header.index(col_name)
                    qvalue_column_name = col_name
                    print(f"Found q-value/FDR column: {col_name}")
                    break
                except ValueError:
                    continue
        
        if qvalue_idx == -1:
            print("Warning: No q-value/FDR column found. Using XCorr for quality scores.")
            print("Available columns:", ', '.join(header[:20]))  # Show first 20 columns
        
        # PEP and e-value columns for PSM pass/fail (total vs passed scan counts)
        pep_idx = -1
        for col_name in ['percolator_PEP', 'pep', 'PEP']:
            try:
                pep_idx = header.index(col_name)
                break
            except ValueError:
                continue
        evalue_idx = -1
        for col_name in ['e-value', 'e_value', 'E-value']:
            try:
                evalue_idx = header.index(col_name)
                break
            except ValueError:
                continue
        
        # Set use_qvalue flag early so it's available in nested functions
        use_qvalue = qvalue_idx >= 0
        
        # Determine data start line (after header)
        data_start_line = header_line_idx + 1
        
        # Use csv.reader for proper CSV parsing
        import io
        csv_content = ''.join(lines[data_start_line:])
        csv_reader = csv.reader(io.StringIO(csv_content))
        run_rt_max_sec = 0.0  # max RT in run (all rows) for RT plot axis
        
        for row_idx, row_data in enumerate(csv_reader):
            if not row_data or len(row_data) == 0:
                continue
            
            if len(row_data) <= max(protein_idx, rt_idx, peptide_idx):
                continue
            
            # Track max RT in run (for RT plot to show full gradient 0 to run end)
            try:
                rt_val = float(row_data[rt_idx].strip()) if rt_idx < len(row_data) and row_data[rt_idx].strip() else None
                if rt_val is not None:
                    run_rt_max_sec = max(run_rt_max_sec, rt_val)
                for col_name in ['spectrum_window_max_rt', 'detected_peak_max_rt']:
                    idx = peak_window_col_indices.get(col_name, -1)
                    if idx >= 0 and idx < len(row_data) and row_data[idx].strip():
                        run_rt_max_sec = max(run_rt_max_sec, float(row_data[idx].strip()))
            except (ValueError, TypeError):
                pass
            
            protein = row_data[protein_idx] if protein_idx < len(row_data) else ""
            if not protein:
                continue
            
            # Get original line for quoted fields
            original_line = lines[data_start_line + row_idx] if (data_start_line + row_idx) < len(lines) else ""
            quoted_fields = re.findall(r'"([^"]*)"', original_line)
            
            clean_protein = re.sub(r'(\d+)$', '', protein)
            clean_protein_alt = re.sub(r'^sp\|', '', clean_protein)
            
            # Filter by protein_id
            if protein_id:
                protein_id_clean = re.sub(r'(\d+)$', '', protein_id)
                protein_id_alt = re.sub(r'^sp\|', '', protein_id_clean)
                if (clean_protein != protein_id_clean and 
                    clean_protein != protein_id_alt and
                    clean_protein_alt != protein_id_clean and
                    clean_protein_alt != protein_id_alt and
                    protein != protein_id):
                    continue
            
            # Get retention time
            try:
                rt_str = row_data[rt_idx] if rt_idx < len(row_data) else ""
                rt = float(rt_str)
            except (ValueError, IndexError):
                continue
            
            # Get quality score (prefer q-value/FDR if available, otherwise use xcorr)
            quality_score = None
            if qvalue_idx >= 0 and qvalue_idx < len(row_data):
                try:
                    qvalue_str = row_data[qvalue_idx].strip()
                    if qvalue_str:
                            raw_value = float(qvalue_str)
                            # Convert to true q-value (0-1) if needed
                            if qvalue_is_percent:
                                # Convert percentage to decimal (e.g., 1.0 -> 0.01)
                                quality_score = raw_value / 100.0
                            elif qvalue_is_score:
                                # Convert q-score to q-value
                                # Check if it's -10*log10(q) or -log10(q) format
                                # If values are in 1-20 range, likely -10*log10(q)
                                # If values are in 0-5 range, likely -log10(q)
                                if raw_value > 10:
                                    # Likely -10*log10(q) format: q = 10^(-qscore/10)
                                    quality_score = 10.0 ** (-raw_value / 10.0)
                                else:
                                    # Likely -log10(q) format: q = 10^(-qscore)
                                    quality_score = 10.0 ** (-raw_value)
                            else:
                                # Assume true q-value (0-1), but validate and auto-detect if needed
                                if raw_value > 1.0:
                                    # Might be percentage or score - try to detect
                                    if raw_value <= 100.0:
                                        # Likely percentage
                                        quality_score = raw_value / 100.0
                                        if not qvalue_is_percent:  # Only print once
                                            print(f"  Auto-detected: values appear to be percentages (1-100), converting to decimal")
                                        qvalue_is_percent = True
                                    elif raw_value > 10:
                                        # Likely q-score (-10*log10 format)
                                        quality_score = 10.0 ** (-raw_value / 10.0)
                                        if not qvalue_is_score:  # Only print once
                                            print(f"  Auto-detected: values appear to be q-scores (-10*log10 format), converting to q-value")
                                        qvalue_is_score = True
                                    else:
                                        # Likely q-score (-log10 format)
                                        quality_score = 10.0 ** (-raw_value)
                                        if not qvalue_is_score:  # Only print once
                                            print(f"  Auto-detected: values appear to be q-scores (-log10 format), converting to q-value")
                                        qvalue_is_score = True
                                else:
                                    # True q-value (already in [0,1])
                                    quality_score = raw_value
                except (ValueError, IndexError):
                    pass
            
            # Fall back to xcorr if q-value not available
            if quality_score is None:
                try:
                    xcorr_str = row_data[xcorr_idx] if xcorr_idx < len(row_data) else ""
                    quality_score = float(xcorr_str) if xcorr_str else 0.0
                except (ValueError, IndexError):
                    quality_score = 0.0
            
            # Store both for compatibility (xcorr for backward compatibility, quality_score for visualization)
            xcorr = quality_score  # Will be q-value if available, otherwise xcorr
            
            # Get charge state
            try:
                charge_str = row_data[charge_idx] if charge_idx < len(row_data) else ""
                charge = int(float(charge_str)) if charge_str else 0
            except (ValueError, IndexError):
                charge = 0
            
            # Get peptide sequence
            peptide_seq = ""
            if peptide_idx >= 0 and len(row_data) > peptide_idx:
                peptide_seq = row_data[peptide_idx].strip('"').strip()
            
            # Get signal intensity (sum of matched fragment ion intensities)
            total_signal_intensity = 0.0
            try:
                intensities_str = row_data[intensities_idx] if intensities_idx < len(row_data) else ""
                if intensities_str and intensities_str.strip():
                    # Parse intensities (comma-separated values)
                    intensities = [float(x.strip()) for x in intensities_str.split(',') if x.strip()]
                    total_signal_intensity = sum(intensities)
            except (ValueError, IndexError):
                total_signal_intensity = 0.0
            
            # Get fill time (ion injection time) from CSV column or mzML by scan number
            fill_time = 0.0
            if ion_injection_time_idx >= 0:
                try:
                    fill_time_str = row_data[ion_injection_time_idx] if ion_injection_time_idx < len(row_data) else ""
                    fill_time = float(fill_time_str) if fill_time_str and fill_time_str.strip() else 0.0
                    if fill_time > 0 and fill_time < 1:
                        fill_time = fill_time * 1000.0
                except (ValueError, IndexError):
                    fill_time = 0.0
            # If no CSV fill time (or suspiciously small), try mzML by scan number when available
            if mzml_fill_times and (fill_time <= 0 or fill_time < 10):
                try:
                    scan_num_str = row_data[scan_idx] if scan_idx < len(row_data) else ""
                    scan_num = int(scan_num_str.strip('"')) if scan_num_str else None
                    if scan_num and scan_num in mzml_fill_times:
                        fill_time = mzml_fill_times[scan_num]
                except (ValueError, KeyError, IndexError):
                    pass
            
            # Get matched fragment ions (from quoted fields or row_data)
            matched_ions_str = ""
            # First try to find it in quoted fields (usually one of the first quoted fields)
            for qf in quoted_fields:
                if qf and (',' in qf) and (qf.startswith('c') or qf.startswith('z') or 'z1_' in qf or qf.startswith('a') or qf.startswith('b') or qf.startswith('x') or qf.startswith('y')):
                    matched_ions_str = qf
                    break
            # Fallback to row_data if not found in quoted fields
            if not matched_ions_str and matched_ions_idx >= 0 and matched_ions_idx < len(row_data):
                matched_ions_str = row_data[matched_ions_idx].strip('"').strip()
            
            # Get single-AA overhangs, fragment pairs, and site_specific_residues (or single_aa_overhangs_protein_positions)
            # Prefer row_data when we have column indices: CSV column order is reliable; quoted_fields order may not match (e.g. unquoted 8K vs quoted "0.45,0.55")
            single_aa_str = ""
            single_aa_pairs_str = ""
            site_specific_str = ""
            if site_specific_idx >= 0 and site_specific_idx < len(row_data):
                site_specific_str = row_data[site_specific_idx].strip('"').strip()
            if single_aa_pairs_idx >= 0 and single_aa_pairs_idx < len(row_data):
                single_aa_pairs_str = row_data[single_aa_pairs_idx].strip('"').strip()
            if single_aa_idx >= 0 and single_aa_idx < len(row_data):
                single_aa_str = row_data[single_aa_idx].strip('"').strip()
            # Fallback to quoted_fields when row_data columns are empty (older CSV format)
            if not site_specific_str and len(quoted_fields) >= 1:
                site_specific_str = quoted_fields[-1]
            if not single_aa_pairs_str and len(quoted_fields) >= 2:
                single_aa_pairs_str = quoted_fields[-2]
            if not single_aa_str and len(quoted_fields) >= 3:
                single_aa_str = quoted_fields[-3]
            
            # Map peptide to protein positions
            if site_specific_str:
                protein_positions = []
                site_specific_list = []
                for item in site_specific_str.split(','):
                    item = item.strip()
                    if item and len(item) > 1:
                        match = re.match(r'^(\d+)([A-Z])$', item)
                        if match:
                            protein_pos = int(match.group(1))
                            residue = match.group(2)
                            protein_positions.append(protein_pos)
                            site_specific_list.append((protein_pos, residue))
                
                if protein_positions:
                    # ISSUE: site_specific_residues only contains positions with single-AA overhangs,
                    # NOT all positions in the peptide. So min/max gives incorrect boundaries.
                    # We need to find the actual peptide boundaries from the peptide sequence.
                    initial_start = min(protein_positions)
                    initial_end = max(protein_positions)
                    
                    # Get the actual peptide sequence to find correct boundaries
                    clean_peptide = re.sub(r'\[.*?\]', '', peptide_seq) if peptide_seq else ''
                    if clean_peptide and protein_sequence:
                        # Try to find the peptide in the protein sequence
                        protein_seq_upper = protein_sequence.upper()
                        clean_peptide_upper = clean_peptide.upper()
                        
                        # Search for the peptide sequence starting near the initial positions
                        # This handles cases where site_specific only has overhang positions
                        peptide_start_in_protein = protein_seq_upper.find(clean_peptide_upper)
                        
                        if peptide_start_in_protein >= 0:
                            # Found it! Use the actual peptide boundaries
                            peptide_start = peptide_start_in_protein + 1
                            peptide_end = peptide_start + len(clean_peptide) - 1
                            
                            # Validate: ensure the initial positions are within the peptide
                            if initial_start < peptide_start or initial_end > peptide_end:
                                # The initial positions suggest a different mapping - check if there's a better match
                                # Try searching around the initial positions
                                search_start = max(0, initial_start - len(clean_peptide) - 10)
                                search_end = min(len(protein_seq_upper), initial_end + len(clean_peptide) + 10)
                                search_region = protein_seq_upper[search_start:search_end]
                                local_match = search_region.find(clean_peptide_upper)
                                
                                if local_match >= 0:
                                    # Found a match near the initial positions
                                    peptide_start = search_start + local_match + 1
                                    peptide_end = peptide_start + len(clean_peptide) - 1
                        else:
                            # Couldn't find exact match - use initial positions as fallback
                            # But validate that the peptide sequence matches at those positions
                            peptide_start = initial_start
                            peptide_end = initial_end
                            
                            # Validate by checking if the sequence matches
                            if peptide_start <= len(protein_seq_upper) and peptide_end <= len(protein_seq_upper):
                                protein_segment = protein_seq_upper[peptide_start - 1:peptide_end]
                                if protein_segment != clean_peptide_upper:
                                    # Mismatch - try to find correct position
                                    # This will be caught and corrected by validation later
                                    pass
                    else:
                        # Fallback: use min/max of site_specific positions
                        peptide_start = initial_start
                        peptide_end = initial_end
                    
                    # CRITICAL: Only include fragment pairs where fragments differ by exactly 1 (consecutive)
                    # For single-AA overhangs, fragments MUST have all the same sequence positions except 1
                    def is_consecutive_fragment_pair(pair_str):
                        """Verify fragment pair represents consecutive fragments (differ by exactly 1)."""
                        normalized = normalize_fragment_pair(pair_str)
                        match = re.match(r'^([a-z])(\d+)-([a-z])(\d+)$', normalized.lower())
                        if not match:
                            return False
                        s1, n1, s2, n2 = match.groups()
                        if s1 != s2:
                            return False
                        return abs(int(n2) - int(n1)) == 1
                    
                    # Get single-AA overhang positions with fragment pairs
                    single_aa_positions = {}
                    single_aa_fragment_pairs = {}
                    if single_aa_str and site_specific_list:
                        single_aa_list = [s.strip() for s in single_aa_str.split(',') if s.strip()]
                        single_aa_pairs_list = [p.strip() for p in single_aa_pairs_str.split(',') if p.strip()] if single_aa_pairs_str else []
                        
                        for i, (protein_pos, protein_residue) in enumerate(site_specific_list):
                            if i < len(single_aa_list):
                                normalized_pairs = set()
                                if i < len(single_aa_pairs_list):
                                    pairs = [p.strip() for p in single_aa_pairs_list[i].split('|') if p.strip()]
                                    for pair in pairs:
                                        # CRITICAL: Only include consecutive fragment pairs (differ by exactly 1)
                                        # Then filter by ion type if specified
                                        if is_consecutive_fragment_pair(pair) and filter_fragment_pair_by_ion_type(pair, ion_type_filter):
                                            normalized_pair = normalize_fragment_pair(pair)
                                            normalized_pairs.add(normalized_pair)
                                    fragment_pair_count = len(normalized_pairs)
                                
                                if fragment_pair_count > 0:
                                    single_aa_positions[protein_pos] = fragment_pair_count
                                    single_aa_fragment_pairs[protein_pos] = normalized_pairs
                    
                    # Extract ALL consecutive fragment pairs from matched fragment ions
                    # This finds single-AA overhangs at every fragmentation site in the peptide
                    if matched_ions_str:
                        # Get peptide length for mapping
                        clean_peptide = re.sub(r'\[.*?\]', '', peptide_seq) if peptide_seq else ''
                        peptide_len = len(clean_peptide) if clean_peptide else 0
                        
                        # Extract all consecutive fragment pairs
                        all_consecutive_pairs = extract_all_consecutive_fragment_pairs(
                            matched_ions_str, peptide_start, peptide_end, peptide_len
                        )
                        
                        # Merge with existing single_aa_fragment_pairs
                        for pos, pairs_set in all_consecutive_pairs.items():
                            # Filter by ion type if specified
                            filtered_pairs = set()
                            for pair in pairs_set:
                                if filter_fragment_pair_by_ion_type(pair, ion_type_filter):
                                    filtered_pairs.add(pair)
                            
                            if filtered_pairs:
                                if pos in single_aa_fragment_pairs:
                                    # Merge with existing pairs
                                    single_aa_fragment_pairs[pos].update(filtered_pairs)
                                    single_aa_positions[pos] = len(single_aa_fragment_pairs[pos])
                                else:
                                    # New position
                                    single_aa_fragment_pairs[pos] = filtered_pairs
                                    single_aa_positions[pos] = len(filtered_pairs)
                    
                    # Get modifications string for peak window lookup
                    mods_str = ""
                    if modifications_idx >= 0 and modifications_idx < len(row_data):
                        mods_str = row_data[modifications_idx].strip('"').strip()
                    if not mods_str or mods_str == '':
                        mods_str = '-'
                    
                    # Extract peak window data - either from merged CSV columns or lookup dictionary
                    peak_window_info = {}
                    if has_peak_window_columns:
                        # Extract from merged CSV columns
                        for col_name, col_idx in peak_window_col_indices.items():
                            if col_idx >= 0 and col_idx < len(row_data):
                                value = row_data[col_idx].strip('"').strip()
                                if value and value != '' and value.lower() != 'nan':
                                    # Map column name to dict key
                                    if col_name == 'peak_window_status':
                                        peak_window_info['status'] = value
                                    elif col_name == 'peak_window_rejection_reason':
                                        peak_window_info['rejection_reason'] = value
                                    else:
                                        # Direct mapping for other columns
                                        key = col_name.replace('peak_window_', '')
                                        try:
                                            # Try to convert to appropriate type
                                            if 'rt' in key or 'size' in key or 'evalue' in key or 'matches' in key or 'score' in key or 'area' in key:
                                                peak_window_info[key] = float(value) if value else None
                                            elif 'using_anchor_window' in key:
                                                peak_window_info[key] = value.lower() in ['true', '1', 'yes']
                                            else:
                                                peak_window_info[key] = value
                                        except (ValueError, TypeError):
                                            peak_window_info[key] = value
                    else:
                        # Look up from separate CSV file (normalize mods to match peak_windows CSV keys)
                        mods_for_key = mods_str if mods_str and str(mods_str).lower() != 'nan' else '-'
                        peak_window_key = (peptide_seq, charge, mods_for_key)
                        peak_window_info = peak_windows_data.get(peak_window_key, {})
                    
                    # Precursor m/z: observed (optional) and theoretical from calc_neutral_mass (for RT windows plots)
                    mz_val = None
                    if mz_idx >= 0 and mz_idx < len(row_data):
                        raw = row_data[mz_idx]
                        if raw is not None and str(raw).strip():
                            try:
                                mz_val = float(str(raw).strip('"').strip())
                            except (ValueError, TypeError):
                                pass
                    mz_theoretical = None
                    if calc_neutral_mass_idx >= 0 and calc_neutral_mass_idx < len(row_data):
                        raw = row_data[calc_neutral_mass_idx]
                        if raw is not None and str(raw).strip():
                            try:
                                calc_mass = float(str(raw).strip('"').strip())
                                if calc_mass > 0 and charge > 0:
                                    PROTON_MASS = 1.007276466812
                                    mz_theoretical = (calc_mass + charge * PROTON_MASS) / charge
                            except (ValueError, TypeError):
                                pass
                    
                    ms1_intensity_val = None
                    if ms1_intensity_idx >= 0 and ms1_intensity_idx < len(row_data) and row_data[ms1_intensity_idx].strip():
                        try:
                            ms1_intensity_val = float(row_data[ms1_intensity_idx].strip('"').strip())
                        except (ValueError, TypeError):
                            pass
                    # PSM scores for total vs passed scan counts (same thresholds as chromatograms)
                    pep_val = None
                    if pep_idx >= 0 and pep_idx < len(row_data) and row_data[pep_idx].strip():
                        try:
                            pep_val = float(row_data[pep_idx].strip('"').strip())
                        except (ValueError, TypeError):
                            pass
                    evalue_val = None
                    if evalue_idx >= 0 and evalue_idx < len(row_data) and row_data[evalue_idx].strip():
                        try:
                            evalue_val = float(row_data[evalue_idx].strip('"').strip())
                        except (ValueError, TypeError):
                            pass
                    peptide_dict = {
                        'rt': rt,
                        'peptide_seq': peptide_seq,
                        'charge': charge,
                        'start': peptide_start,
                        'end': peptide_end,
                        'single_aa_positions': single_aa_positions,
                        'single_aa_fragment_pairs': single_aa_fragment_pairs,
                        'protein_positions': set(protein_positions),
                        'xcorr': xcorr,  # Actually contains q-value if available, otherwise xcorr
                        'quality_score': quality_score,  # Explicit quality score (q-value or xcorr)
                        'signal_intensity': total_signal_intensity,
                        'fill_time': fill_time,
                        'modifications': mods_str,
                        'mz': mz_val,
                        'mz_theoretical': mz_theoretical,
                        'ms1_intensity': ms1_intensity_val,
                        'percolator_qvalue': quality_score if qvalue_idx >= 0 else None,
                        'percolator_PEP': pep_val,
                        'evalue': evalue_val
                    }
                    
                    # Add peak window data if available
                    if peak_window_info:
                        peptide_dict.update({
                            'peak_window_status': peak_window_info.get('status', 'unknown'),
                            'peak_window_rejection_reason': peak_window_info.get('rejection_reason', ''),
                            'detected_peak_min_rt': peak_window_info.get('detected_peak_min_rt'),
                            'detected_peak_max_rt': peak_window_info.get('detected_peak_max_rt'),
                            'detected_peak_window_size': peak_window_info.get('detected_peak_window_size'),
                            'spectrum_window_min_rt': peak_window_info.get('spectrum_window_min_rt'),
                            'spectrum_window_max_rt': peak_window_info.get('spectrum_window_max_rt'),
                            'spectrum_window_size': peak_window_info.get('spectrum_window_size'),
                            'apex_rt': peak_window_info.get('apex_rt'),
                            'anchor_rt': peak_window_info.get('anchor_rt'),
                            'anchor_evalue': peak_window_info.get('anchor_evalue'),
                            'n_isotopic_matches': peak_window_info.get('n_isotopic_matches', 0),
                            'matched_isotopes': peak_window_info.get('matched_isotopes', ''),
                            'using_anchor_window': peak_window_info.get('using_anchor_window', False),
                            'total_area': peak_window_info.get('total_area'),
                            'apex_intensity': peak_window_info.get('apex_intensity'),
                            'm0_gt_m1_gt_m2': peak_window_info.get('m0_gt_m1_gt_m2')
                        })
                    
                    peptides.append(peptide_dict)
    
    if not peptides:
        print(f"No peptides found for protein {protein_id}")
        return
    
    # Print peak window statistics if available
    if peak_windows_data:
        peptides_with_peak_windows = [p for p in peptides if 'peak_window_status' in p]
        if peptides_with_peak_windows:
            accepted_count = len([p for p in peptides_with_peak_windows if p.get('peak_window_status') == 'accepted'])
            rejected_count = len([p for p in peptides_with_peak_windows if p.get('peak_window_status') == 'rejected'])
            print(f"\n{'='*60}")
            print(f"Peak Window Statistics ({len(peptides_with_peak_windows)} peptides with peak window data):")
            print(f"  Accepted: {accepted_count}")
            print(f"  Rejected: {rejected_count}")
            if rejected_count > 0:
                rejection_reasons = {}
                for p in peptides_with_peak_windows:
                    if p.get('peak_window_status') == 'rejected':
                        reason = p.get('peak_window_rejection_reason', 'Unknown')
                        rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
                print(f"  Rejection reasons:")
                for reason, count in rejection_reasons.items():
                    print(f"    {reason}: {count}")
            print(f"{'='*60}")
    
    # Reject modified peptides when filter_criteria has reject_mods True (keep only unmodified).
    def _has_modification(m):
        """True if peptide has any modification (not empty, -, or nan)."""
        if m is None or (isinstance(m, float) and np.isnan(m)):
            return False
        s = str(m).strip().lower()
        return s not in ('', '-', 'nan', 'none')
    if _reject_mods:
        before_reject_mods = len(peptides)
        peptides = [p for p in peptides if not _has_modification(p.get('modifications'))]
        if before_reject_mods > len(peptides):
            print(f"Filtered to unmodified peptides only (reject mods): {before_reject_mods} PSMs -> {len(peptides)} PSMs ({before_reject_mods - len(peptides)} modified removed)")
    
    # When peak_windows_csv is provided, always restrict to accepted peptides (from extract chromatograms).
    # Only peptides that appear in the peak windows CSV with the requested status are kept; others are excluded.
    # Require >= N scans per peptide (same seq, charge, mods) within PPM of theoretical m/z (values from filter_criteria).
    MIN_SCANS = _min_scans_combined
    PPM_TOLERANCE = _ppm_tolerance
    def _peptide_scan_key(p):
        seq = (p.get('peptide_seq') or '').strip()
        try:
            c = int(p.get('charge', 0)) if p.get('charge') not in (None, '') else 0
        except (TypeError, ValueError):
            c = 0
        m = (p.get('modifications') or '-')
        m = (str(m).strip() or '-') if m not in (None, '') else '-'
        return (seq, c, m)
    def _within_ppm_theoretical(p, ppm=5.0):
        """True if observed m/z is within ppm of theoretical m/z."""
        mz_obs = p.get('mz')
        mz_th = p.get('mz_theoretical')
        if mz_obs is None or mz_th is None or mz_th <= 0:
            return False
        try:
            mz_obs = float(mz_obs)
            mz_th = float(mz_th)
            if mz_th <= 0 or np.isnan(mz_obs) or np.isnan(mz_th):
                return False
            ppm_diff = abs(mz_obs - mz_th) / mz_th * 1e6
            return ppm_diff <= ppm
        except (TypeError, ValueError):
            return False
    # PSM pass/fail (same thresholds as chromatograms: Q<=0.01, PEP<=0.05, E<=0.01)
    FILTER_PSM_QVALUE_MAX = 0.01
    FILTER_PSM_PEP_MAX = 0.05
    FILTER_PSM_EVALUE_MAX = 0.01
    def _psm_passes(p):
        q = p.get('percolator_qvalue')
        if q is not None and np.isfinite(q) and q > FILTER_PSM_QVALUE_MAX:
            return False
        pep = p.get('percolator_PEP')
        if pep is not None and np.isfinite(pep) and pep > FILTER_PSM_PEP_MAX:
            return False
        e = p.get('evalue')
        if e is not None and np.isfinite(e) and e > FILTER_PSM_EVALUE_MAX:
            return False
        return True
    from collections import Counter
    total_scan_counts = Counter(_peptide_scan_key(p) for p in peptides)
    passed_scan_counts = Counter(_peptide_scan_key(p) for p in peptides if _psm_passes(p))
    scan_counts = Counter(_peptide_scan_key(p) for p in peptides if _within_ppm_theoretical(p, PPM_TOLERANCE))
    peptides_all_before_scan_filter = list(peptides)  # keep for RT grid pool / fill candidates (peptides before scan filter)
    before_scan_filter = len(peptides)
    peptides = [p for p in peptides if scan_counts[_peptide_scan_key(p)] >= MIN_SCANS]
    after_scan_filter = len(peptides)
    if before_scan_filter > after_scan_filter:
        n_removed = before_scan_filter - after_scan_filter
        n_peptide_ids_removed = sum(1 for k, c in scan_counts.items() if c < MIN_SCANS)
        print(f"\nFiltered to peptides with >= {MIN_SCANS} scans within {PPM_TOLERANCE:.2f} ppm of theoretical m/z: {before_scan_filter} PSMs -> {after_scan_filter} PSMs ({n_removed} removed; {n_peptide_ids_removed} peptide identities had < {MIN_SCANS} scans within {PPM_TOLERANCE:.2f} ppm)")
    
    # Require M0, M+1, M+2 all have signal > 0 and (M0 > M+1 OR M+1 > M+2) on summed MS1 (peak window sets m0_gt_m1_gt_m2 = (M0>M+1) or (M+1>M+2)).
    # Only apply when we have peak window data and require_m0_gt_m1_gt_m2 is True.
    _has_peak_window_data = bool(peak_windows_csv) or any(
        (p.get('matched_isotopes') or p.get('peak_window_matched_isotopes') or '') for p in peptides
    )
    def _has_envelope_ok(p, require_data_present):
        """Require M0, M+1, M+2 all present (signal > 0) and m0_gt_m1_gt_m2 True (M0>M+1 or M+1>M+2 from peak window)."""
        mi = p.get('matched_isotopes') or p.get('peak_window_matched_isotopes') or ''
        if not mi or not str(mi).strip():
            return False if require_data_present else True
        mi_str = str(mi).upper().replace(' ', '')
        has_m0 = 'M+0' in mi_str or 'M0' in mi_str
        has_m1 = 'M+1' in mi_str or 'M1' in mi_str
        has_m2 = 'M+2' in mi_str or 'M2' in mi_str
        if not (has_m0 and has_m1 and has_m2):
            return False  # require all three isotopes with signal
        m0_gt_m1_gt_m2 = p.get('m0_gt_m1_gt_m2')
        if m0_gt_m1_gt_m2 is None or (isinstance(m0_gt_m1_gt_m2, float) and np.isnan(m0_gt_m1_gt_m2)):
            return False if require_data_present else True
        if m0_gt_m1_gt_m2 is True:
            return True
        if isinstance(m0_gt_m1_gt_m2, str) and m0_gt_m1_gt_m2.strip().lower() in ('true', '1', 'yes'):
            return True
        if isinstance(m0_gt_m1_gt_m2, (int, float)) and m0_gt_m1_gt_m2 and m0_gt_m1_gt_m2 == m0_gt_m1_gt_m2:
            return True
        if m0_gt_m1_gt_m2 is False or (isinstance(m0_gt_m1_gt_m2, str) and m0_gt_m1_gt_m2.strip().lower() in ('false', '0', 'no', '')):
            return False
        return True  # unparseable: keep
    before_m012 = len(peptides)
    if _has_peak_window_data and _require_m0_gt_m1_gt_m2:
        peptides = [p for p in peptides if _has_envelope_ok(p, require_data_present=True)]
        if before_m012 > len(peptides):
            print(f"Filtered to peptides with M0, M+1, M+2 all >0 and (M0 > M+1 or M+1 > M+2) on summed MS1: {before_m012} PSMs -> {len(peptides)} PSMs ({before_m012 - len(peptides)} removed)")
    
    # Drop peptides below percentile in total area and/or apex peak (keep only peptides above percentile in BOTH; percentiles from filter_criteria).
    def _apex_for_peptide(p):
        """Apex intensity from peak window data only (no fallback to main CSV)."""
        a = p.get('apex_intensity')
        if a is not None and not (isinstance(a, float) and (np.isnan(a) or a < 0)):
            try:
                return float(a)
            except (TypeError, ValueError):
                pass
        return -1.0
    def _max_total_area_for_key_area_filter(k):
        areas = []
        for p in peptides:
            if _peptide_scan_key(p) != k:
                continue
            a = p.get('total_area')
            if a is not None and not (isinstance(a, float) and (np.isnan(a) or a < 0)):
                try:
                    areas.append(float(a))
                except (TypeError, ValueError):
                    pass
        return max(areas) if areas else -1.0
    def _max_apex_for_key(k):
        apexes = []
        for p in peptides:
            if _peptide_scan_key(p) != k:
                continue
            a = _apex_for_peptide(p)
            if a >= 0:
                apexes.append(a)
        return max(apexes) if apexes else -1.0
    key_to_max_area_p25 = {k: _max_total_area_for_key_area_filter(k) for k in scan_counts}
    key_to_max_apex = {k: _max_apex_for_key(k) for k in scan_counts}
    valid_areas_p25 = [v for v in key_to_max_area_p25.values() if v >= 0]
    valid_apexes = [v for v in key_to_max_apex.values() if v >= 0]
    if valid_areas_p25 and valid_apexes:
        area_thresh_p = float(np.percentile(valid_areas_p25, _total_area_apex_percentile))
        apex_thresh_p = float(np.percentile(valid_apexes, _total_area_apex_percentile))
        keep_keys_area_apex = {k for k in scan_counts
                               if key_to_max_area_p25.get(k, -1) >= area_thresh_p and key_to_max_apex.get(k, -1) >= apex_thresh_p}
        before_p25 = len(peptides)
        peptides = [p for p in peptides if _peptide_scan_key(p) in keep_keys_area_apex]
        n_removed_p25 = before_p25 - len(peptides)
        n_ids_removed_p25 = len(scan_counts) - len(keep_keys_area_apex)
        if n_removed_p25 > 0:
            print(f"Dropped below {_total_area_apex_percentile}th percentile by total area and apex peak: {before_p25} PSMs -> {len(peptides)} PSMs ({n_removed_p25} removed; {n_ids_removed_p25} peptide identities below {_total_area_apex_percentile}th percentile in either metric)")
    elif valid_areas_p25:
        # Only total_area available: drop by total area only
        area_thresh_p = float(np.percentile(valid_areas_p25, _total_area_apex_percentile))
        keep_keys_area_apex = {k for k in scan_counts if key_to_max_area_p25.get(k, -1) >= area_thresh_p}
        before_p25 = len(peptides)
        peptides = [p for p in peptides if _peptide_scan_key(p) in keep_keys_area_apex]
        if before_p25 > len(peptides):
            print(f"Dropped below {_total_area_apex_percentile}th percentile by total area (no apex intensity column): {before_p25} PSMs -> {len(peptides)} PSMs")
    
    # Optional: for c-ions or z-ions only plots, keep only peptides that have at least one fragment of that type
    if ion_type_filter is not None:
        def _peptide_has_ion_type(p, ion):
            for pos, pairs in (p.get('single_aa_fragment_pairs') or {}).items():
                for pair in (pairs if isinstance(pairs, (list, set)) else [pairs]):
                    n = normalize_fragment_pair(pair) if hasattr(pair, 'strip') else normalize_fragment_pair(str(pair))
                    if ion == 'c' and n.startswith('c') and '-' in n and n.split('-')[1].startswith('c'):
                        return True
                    if ion == 'z' and n.startswith('z') and '-' in n and n.split('-')[1].startswith('z'):
                        return True
            for key in p:
                if key.endswith('_fragment_pairs') and key != 'single_aa_fragment_pairs':
                    for pos, pairs in (p.get(key) or {}).items():
                        for pair in (pairs if isinstance(pairs, (list, set)) else [pairs]):
                            n = normalize_fragment_pair(pair) if hasattr(pair, 'strip') else normalize_fragment_pair(str(pair))
                            if ion == 'c' and n.startswith('c') and '-' in n and n.split('-')[1].startswith('c'):
                                return True
                            if ion == 'z' and n.startswith('z') and '-' in n and n.split('-')[1].startswith('z'):
                                return True
            return False
        before_ion = len(peptides)
        peptides = [p for p in peptides if _peptide_has_ion_type(p, ion_type_filter)]
        if len(peptides) < before_ion:
            print(f"Filtered to peptides with {ion_type_filter}-ions: {before_ion} -> {len(peptides)} peptides")
    
    # High-scan protection disabled: do not protect identities with 7+ scans (they are subject to majority filter and RT-overlap cap like others).
    MULTI_SCAN_THRESHOLD = MIN_SCANS - 1  # was used to protect identities with 7+ scans within 5 ppm
    protected_high_scan_keys = set()  # no longer protecting by scan count
    
    # Protect peptides with high total_area (top quartile by per-identity max total_area) so strong signals are never dropped.
    def _max_total_area_for_key(k):
        areas = []
        for p in peptides:
            if _peptide_scan_key(p) != k:
                continue
            a = p.get('total_area')
            if a is not None and not (isinstance(a, float) and (np.isnan(a) or a < 0)):
                try:
                    areas.append(float(a))
                except (TypeError, ValueError):
                    pass
        return max(areas) if areas else -1.0
    key_to_max_area = {k: _max_total_area_for_key(k) for k in scan_counts}
    valid_areas = [v for v in key_to_max_area.values() if v >= 0]
    if valid_areas and _protect_total_area_percentile < 100:
        area_threshold = float(np.percentile(valid_areas, _protect_total_area_percentile))
        protected_high_area_keys = {k for k, v in key_to_max_area.items() if v >= area_threshold}
        if protected_high_area_keys:
            print(f"Protecting {len(protected_high_area_keys)} peptide identities with high total_area (>= {_protect_total_area_percentile}th percentile = {area_threshold:.2e})")
    else:
        protected_high_area_keys = set()
    protected_peptide_keys = protected_high_scan_keys | protected_high_area_keys
    if protected_peptide_keys and len(protected_peptide_keys) > len(protected_high_scan_keys):
        print(f"Combined protected set: {len(protected_peptide_keys)} identities (high scan and/or high total_area)")
    
    # Exclude peptides where the majority of scans have fewer than threshold single-AA overhangs
    # For each (seq, charge, mods), count scans with < threshold vs >= threshold overhangs; keep only identities where majority have >= threshold
    # Always keep protected (high-scan and/or high-area) identities so strong signals remain in the RT plot
    key_to_low = defaultdict(int)   # scans with < threshold single-AA overhangs
    key_to_high = defaultdict(int)  # scans with >= threshold single-AA overhangs
    for p in peptides:
        k = _peptide_scan_key(p)
        n_overhangs = len(p.get('single_aa_positions', {}))
        if n_overhangs < _high_overhang_threshold:
            key_to_low[k] += 1
        else:
            key_to_high[k] += 1
    keep_keys = {k for k in key_to_low if key_to_high[k] > key_to_low[k]}
    keep_keys.update(k for k in key_to_high if k not in key_to_low)  # identities that only have >= threshold scans
    keep_keys.update(protected_peptide_keys)  # never drop high-scan or high-area peptides
    before_majority = len(peptides)
    peptides = [p for p in peptides if _peptide_scan_key(p) in keep_keys]
    n_removed = before_majority - len(peptides)
    n_ids_removed = sum(1 for k in key_to_low if key_to_high[k] <= key_to_low[k] and (key_to_low[k] + key_to_high[k]) > 0 and k not in protected_peptide_keys)
    if n_removed > 0:
        print(f"Filtered to peptides where majority of scans have >= {_high_overhang_threshold} single-AA overhangs: {before_majority} PSMs -> {len(peptides)} PSMs ({n_removed} removed; {n_ids_removed} peptide identities had majority of scans with <{_high_overhang_threshold} overhangs)")
    
    # In high-coverage positions, remove peptides with the lowest total_area (keep higher-abundance only)
    position_to_keys = defaultdict(set)  # pos -> set of (seq, charge, mods)
    for p in peptides:
        k = _peptide_scan_key(p)
        for pos in (p.get('single_aa_positions') or {}).keys():
            try:
                pos_int = int(pos)
                if 1 <= pos_int <= protein_length:
                    position_to_keys[pos_int].add(k)
            except (TypeError, ValueError):
                pass
    coverage_counts = [len(position_to_keys[pos]) for pos in range(1, protein_length + 1) if position_to_keys[pos]]
    median_coverage = float(np.median(coverage_counts)) if coverage_counts else 0
    high_cov_positions = {pos for pos in range(1, protein_length + 1) if len(position_to_keys[pos]) > median_coverage}
    if high_cov_positions:
        def _max_total_area_for_key(k):
            areas = []
            for p in peptides:
                if _peptide_scan_key(p) != k:
                    continue
                a = p.get('total_area')
                if a is not None and not (isinstance(a, float) and (np.isnan(a) or a < 0)):
                    try:
                        areas.append(float(a))
                    except (TypeError, ValueError):
                        pass
            return max(areas) if areas else -1.0
        keys_to_remove = set()
        for pos in high_cov_positions:
            keys_covering = list(position_to_keys[pos])
            if len(keys_covering) <= 1:
                continue
            keys_sorted_by_area = sorted(keys_covering, key=lambda k: -_max_total_area_for_key(k))
            n_keep = max(1, (len(keys_sorted_by_area) + 1) // 2)
            keys_to_drop = [k for k in keys_sorted_by_area[n_keep:] if k not in protected_peptide_keys]
            keys_to_remove.update(keys_to_drop)
        if keys_to_remove:
            before_prune = len(peptides)
            peptides = [p for p in peptides if _peptide_scan_key(p) not in keys_to_remove]
            n_psm_removed = before_prune - len(peptides)
            print(f"Pruned lowest total_area peptides in high-coverage positions: {before_prune} PSMs -> {len(peptides)} PSMs ({n_psm_removed} removed; {len(keys_to_remove)} peptide identities dropped in {len(high_cov_positions)} high-coverage positions)")
    
    # Print fill time statistics for full dataset
    all_fill_times = [p.get('fill_time', 0.0) for p in peptides if p.get('fill_time', 0.0) > 0]
    if all_fill_times:
        print(f"\n{'='*60}")
        print(f"Fill Time Statistics - Full Dataset ({len(peptides)} peptides):")
        print(f"  Valid fill times: {len(all_fill_times)} peptides")
        print(f"  Range: {min(all_fill_times):.2f} - {max(all_fill_times):.2f} ms")
        print(f"  Min: {min(all_fill_times):.2f} ms")
        print(f"  Max: {max(all_fill_times):.2f} ms")
        print(f"  Mean: {sum(all_fill_times)/len(all_fill_times):.2f} ms")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print(f"Fill Time Statistics - Full Dataset ({len(peptides)} peptides):")
        print(f"  WARNING: No valid fill times found in full dataset")
        print(f"{'='*60}")
    
    # Get set of filtered peptide keys (peptide_seq, rt, charge) for matching
    # Each peptide observation is unique based on sequence, RT, and charge
    filtered_peptide_keys = set((p['peptide_seq'], p['rt'], p.get('charge', 0)) for p in peptides)
    
    # Build row indices for overhang parsing: only include rows for our filtered peptides (>= min_scans)
    peptide_to_row_indices = {}
    with open(csv_file, 'r') as f:
        lines = f.readlines()
        if len(lines) >= 1:
            pw_header_idx = 1
            if 'protein' in lines[0]:
                pw_header_idx = 0
            elif len(lines) > 1 and 'protein' in lines[1]:
                pw_header_idx = 1
            header_line = lines[pw_header_idx]
            header = [col.strip() for col in header_line.split(',')]
            try:
                peptide_idx = header.index('plain_peptide')
                charge_idx = header.index('charge')
                modifications_idx = header.index('modifications')
            except ValueError:
                peptide_idx = -1
                charge_idx = -1
                modifications_idx = -1
            
            if peptide_idx >= 0:
                import csv as csv_module
                import io
                data_start = pw_header_idx + 1
                csv_content = ''.join(lines[data_start:])
                csv_reader = csv_module.reader(io.StringIO(csv_content))
                for row_idx, row_data in enumerate(csv_reader):
                    if len(row_data) <= max(peptide_idx, charge_idx, modifications_idx):
                        continue
                    peptide_seq = row_data[peptide_idx].strip('"').strip() if peptide_idx < len(row_data) else ""
                    charge = int(row_data[charge_idx]) if charge_idx >= 0 and charge_idx < len(row_data) and row_data[charge_idx].strip() else 0
                    mods = row_data[modifications_idx].strip('"').strip() if modifications_idx >= 0 and modifications_idx < len(row_data) else '-'
                    if peptide_seq:
                        key = (peptide_seq, charge, mods)
                        if key not in peptide_to_row_indices:
                            peptide_to_row_indices[key] = []
                        peptide_to_row_indices[key].append(row_idx)
    
    filtered_indices_for_overhangs = set()
    for p in peptides:
        key = (p.get('peptide_seq', ''), p.get('charge', 0), p.get('modifications', '-'))
        if key in peptide_to_row_indices:
            for row_idx in peptide_to_row_indices[key]:
                filtered_indices_for_overhangs.add(row_idx)
    print(f"Overhang row indices: {len(filtered_indices_for_overhangs)} rows (from {len(peptides)} peptides with >= {MIN_SCANS} scans)")
    
    # NOTE: We no longer parse unfiltered overhangs - the heatmap should only show
    # overhangs from the filtered peptides shown in the RT plot
    # This ensures consistency between what's shown in the RT plot and the heatmap
    
    # Parse overhangs with filter applied
    # For combined plot (ion_type_filter=None), we need to ensure we get ALL overhangs
    # by parsing both c and z separately and combining them
    if ion_type_filter is None:
        # For combined plot: get ALL overhangs (both c and z)
        # Parse without filtering, then we'll include everything
        multi_aa_overhangs_all = parse_multi_aa_overhangs(csv_file, protein_id, fasta_file, 
                                                          filtered_peptide_indices=filtered_indices_for_overhangs)
        # parse_multi_aa_overhangs already filters to consistent ion series, so we get both c and z
        # But we want to make sure we're not missing any - the function should return all valid overhangs
        multi_aa_overhangs = multi_aa_overhangs_all
    else:
        # For individual plots: parse and then filter by ion type
        multi_aa_overhangs = parse_multi_aa_overhangs(csv_file, protein_id, fasta_file, 
                                                      filtered_peptide_indices=filtered_indices_for_overhangs)
        # Filter multi-AA overhangs by ion type
        multi_aa_overhangs = filter_overhangs_by_ion_type(multi_aa_overhangs, ion_type_filter)
    
    # Parse single-AA overhangs and filter by matching peptide keys (seq, rt, charge)
    all_single_aa = parse_single_aa_overhangs(csv_file, protein_id, fasta_file)
    
    # CRITICAL: Filter out invalid single-AA overhangs that don't have fragment_pairs
    # A single-AA overhang MUST have fragment_pairs by definition (it's defined by consecutive fragment ions)
    all_single_aa = [o for o in all_single_aa if o.get('fragment_pairs', [])]
    
    # Create a set of accepted peptide sequences for filtering overhangs to our peptides only
    accepted_peptide_seqs = set(p.get('peptide_seq', '') for p in peptides)
    
    # Filter single-AA overhangs to only those from accepted peptides (>= min_scans)
    single_aa_overhangs = [o for o in all_single_aa if o.get('peptide_seq', '') in accepted_peptide_seqs]
    print(f"Single-AA overhangs from accepted peptides: {len(single_aa_overhangs)} (from {len(all_single_aa)} total)")
    print("Applying filters for RT grid...")
    
    # Filter single-AA overhangs by ion type if specified
    if ion_type_filter is not None:
        single_aa_overhangs = filter_overhangs_by_ion_type(single_aa_overhangs, ion_type_filter)
    
    # For coverage heatmap: include single-AA overhangs from peptides shown in plots below
    # We'll update this after peptides_for_rt_grid is defined to ensure consistency
    # For now, use all single-AA overhangs, but we'll filter later to match peptides_for_rt_grid
    
    # For RT grid: use filtered single-AA overhangs (to match selected peptides)
    all_overhangs = multi_aa_overhangs + single_aa_overhangs

    # Attach scan counts to each peptide so RT plot can show number of scans (total)
    for p in peptides:
        key = _peptide_scan_key(p)
        p['_scan_count'] = scan_counts.get(key, 0)
        p['_scan_count_total'] = total_scan_counts.get(key, 0)
        p['_scan_count_passed'] = passed_scan_counts.get(key, 0)
    
    # Filter peptides for RT grid: require at least _min_single_aa_overhangs single-AA overhangs (matches run_visualization / regenerate)
    # When min is 0, include peptides with any single_aa_positions or RT window data for backwards compatibility
    def _has_rt_window_data(p):
        return (p.get('detected_peak_min_rt') is not None or p.get('spectrum_window_min_rt') is not None or
                p.get('detected_peak_max_rt') is not None or p.get('spectrum_window_max_rt') is not None)
    if _min_single_aa_overhangs <= 0:
        peptides_for_rt_grid = [p for p in peptides if p.get('single_aa_positions', {}) or _has_rt_window_data(p)]
    else:
        peptides_for_rt_grid = [p for p in peptides if len(p.get('single_aa_positions', {})) >= _min_single_aa_overhangs]
    
    # Deduplicate by (sequence, charge, mods, MS1 RT): same peptide+charge+mods+MS1 RT = one row (ignore MS2 RT).
    # For peptides with >3 scans (4+; protected), keep every PSM as a separate row so the plot shows all scans per peptide.
    def _rt_grid_key(p):
        seq = (p.get('peptide_seq') or '').strip()
        try:
            c = int(p.get('charge', 0)) if p.get('charge') not in (None, '') else 0
        except (TypeError, ValueError):
            c = 0
        m = (p.get('modifications') or '-')
        m = (str(m).strip() or '-') if m not in (None, '') else '-'
        rt = float(p.get('rt', 0))
        return (seq, c, m, rt)
    # Do not merge by (seq, charge, mods, RT): each peptide/charge/mod has its own total area and row
    peptides_for_rt_grid = list(peptides_for_rt_grid)
    
    # No redundant low-overhang removal; selection is based on apex peak height and total area (RT-competing pass below).

    # Sort peptides by position, then length, then charge (default ordering)
    # This makes it easier to see coverage patterns along the sequence
    peptides_sorted_by_position = []
    for p in peptides:
        peptide_seq = p.get('peptide_seq', '')
        clean_peptide = re.sub(r'\[.*?\]', '', peptide_seq) if peptide_seq else ''
        peptide_length = len(clean_peptide) if clean_peptide else 0
        peptide_start = p.get('start', 0)
        charge = p.get('charge', 0)
        
        # If start is missing, try to calculate it
        if peptide_start == 0 and clean_peptide:
            protein_seq_upper = protein_sequence.upper()
            clean_peptide_upper = clean_peptide.upper()
            peptide_start_in_protein = protein_seq_upper.find(clean_peptide_upper)
            if peptide_start_in_protein >= 0:
                peptide_start = peptide_start_in_protein + 1
        
        peptides_sorted_by_position.append((peptide_start, peptide_length, charge, p))
    
    peptides_sorted_by_position.sort(key=lambda x: (x[0] if x[0] > 0 else 999999, -x[1], x[2]))
    peptides = [p for _, _, _, p in peptides_sorted_by_position]
    
    # Sort peptides_for_rt_grid by position, then length, then charge
    peptides_for_rt_grid_sorted = []
    for p in peptides_for_rt_grid:
        peptide_seq = p.get('peptide_seq', '')
        clean_peptide = re.sub(r'\[.*?\]', '', peptide_seq) if peptide_seq else ''
        peptide_length = len(clean_peptide) if clean_peptide else 0
        peptide_start = p.get('start', 0)
        charge = p.get('charge', 0)
        
        # If start is missing, try to calculate it
        if peptide_start == 0 and clean_peptide:
            protein_seq_upper = protein_sequence.upper()
            clean_peptide_upper = clean_peptide.upper()
            peptide_start_in_protein = protein_seq_upper.find(clean_peptide_upper)
            if peptide_start_in_protein >= 0:
                peptide_start = peptide_start_in_protein + 1
        
        peptides_for_rt_grid_sorted.append((peptide_start, peptide_length, charge, p))
    
    peptides_for_rt_grid_sorted.sort(key=lambda x: (x[0] if x[0] > 0 else 999999, -x[1], x[2]))
    peptides_for_rt_grid = [p for _, _, _, p in peptides_for_rt_grid_sorted]
    
    # Get RT window bounds (min_rt, max_rt) as floats for overlap checks
    def _get_window_bounds(p):
        min_rt = p.get('detected_peak_min_rt') or p.get('spectrum_window_min_rt')
        max_rt = p.get('detected_peak_max_rt') or p.get('spectrum_window_max_rt')
        rt = float(p.get('rt', 0))
        if min_rt is None or max_rt is None:
            try:
                min_rt = rt - 5.0
                max_rt = rt + 5.0
            except (TypeError, ValueError):
                min_rt, max_rt = 0.0, 10.0
        try:
            min_rt = float(min_rt)
            max_rt = float(max_rt)
        except (TypeError, ValueError):
            min_rt, max_rt = 0.0, 10.0
        return (min_rt, max_rt)
    def _rt_window_key(p):
        a, b = _get_window_bounds(p)
        return (round(a, 1), round(b, 1))
    
    # Option to remove lowest-signal peptides when competing for RT windows (overlap in time and same/superset positions).
    # Disabled by default so all peptides are kept; set REMOVE_RT_COMPETING_PEPTIDES = True to re-enable.
    REMOVE_RT_COMPETING_PEPTIDES = False
    if REMOVE_RT_COMPETING_PEPTIDES:
        def _positions_set_rt(p):
            return set((p.get('single_aa_positions') or {}).keys())
        def _signal_for_rt_compete(p):
            """Return (total_area, apex_intensity) for comparison; missing/invalid -> -1 so they sort last (not removed)."""
            a = p.get('total_area')
            if a is None or (isinstance(a, float) and (np.isnan(a) or a < 0)):
                area = -1.0
            else:
                try:
                    area = float(a)
                except (TypeError, ValueError):
                    area = -1.0
            apex = p.get('apex_intensity')
            if apex is None or (isinstance(apex, float) and (np.isnan(apex) or apex < 0)):
                apex_val = -1.0
            else:
                try:
                    apex_val = float(apex)
                except (TypeError, ValueError):
                    apex_val = -1.0
            return (area, apex_val)
        remaining_rt = list(peptides_for_rt_grid)
        rt_compete_removals = 0
        while True:
            removed_this_round = False
            for i in range(len(remaining_rt) - 1, -1, -1):
                p = remaining_rt[i]
                if _peptide_scan_key(p) in protected_peptide_keys:
                    continue
                pos_p = _positions_set_rt(p)
                if not pos_p:
                    continue
                a_p, b_p = _get_window_bounds(p)
                competitors = []
                for q in remaining_rt:
                    if q is p:
                        continue
                    pos_q = _positions_set_rt(q)
                    if not pos_p.issubset(pos_q) and pos_p != pos_q:
                        continue
                    a_q, b_q = _get_window_bounds(q)
                    if a_q <= b_p and b_q >= a_p:
                        competitors.append(q)
                if not competitors:
                    continue
                sig_p = _signal_for_rt_compete(p)
                if sig_p[0] < 0 and sig_p[1] < 0:
                    continue
                if all(sig_p <= _signal_for_rt_compete(q) for q in competitors):
                    remaining_rt.pop(i)
                    rt_compete_removals += 1
                    removed_this_round = True
                    break
            if not removed_this_round:
                break
        if rt_compete_removals:
            print(f"  [RT-competing] Removed {rt_compete_removals} lowest-signal peptides (by total_area, apex_intensity) that were competing for RT windows")
        peptides_for_rt_grid = remaining_rt
    
    # Cap by time: at any t on the RT scale, at most N peptide identities (disabled for now so all peptides are shown).
    USE_MAX_OVERLAP_CAP = False
    if USE_MAX_OVERLAP_CAP:
        MAX_OVERLAP = 3
        RT_STEP = 5.0
        RUN_RT_EXTEND_MIN_SEC = 1200  # 20 min
        run_rt_max_sec_cap = max(run_rt_max_sec, RUN_RT_EXTEND_MIN_SEC) if run_rt_max_sec else RUN_RT_EXTEND_MIN_SEC
        all_bounds = [_get_window_bounds(p) for p in peptides_for_rt_grid]
        if all_bounds:
            min_rt_cap = min(a for a, _ in all_bounds)
            max_rt_cap = max(b for _, b in all_bounds)
            min_rt_global = min(min_rt_cap, 0.0)
            if run_rt_max_sec_cap > 0:
                max_rt_global = max(max_rt_cap, run_rt_max_sec_cap)
            else:
                max_rt_global = max_rt_cap
            times = list(np.arange(min_rt_global, max_rt_global + RT_STEP, RT_STEP))
            if not times:
                times = [min_rt_global]

            def coverage_at_times(peptide_list):
                cov = [0] * len(times)
                for i, t in enumerate(times):
                    ids_at_t = {_peptide_scan_key(p) for p in peptide_list
                                if _get_window_bounds(p)[0] <= t <= _get_window_bounds(p)[1]}
                    cov[i] = len(ids_at_t)
                return cov

            S = list(peptides_for_rt_grid)
            coverage = coverage_at_times(S)
            cap_removals = 0
            max_cap_iterations = 50000  # safety: stop after this many single-identity removals
            while cap_removals < max_cap_iterations:
                max_cov = max(coverage) if coverage else 0
                if max_cov <= MAX_OVERLAP:
                    break
                removed_this_round = False
                for i, t in enumerate(times):
                    if coverage[i] <= MAX_OVERLAP:
                        continue
                    covering = [p for p in S if _get_window_bounds(p)[0] <= t <= _get_window_bounds(p)[1]]
                    ids_at_t = {_peptide_scan_key(p) for p in covering}
                    if len(ids_at_t) <= MAX_OVERLAP:
                        continue
                    def rep_row(k):
                        for p in covering:
                            if _peptide_scan_key(p) == k:
                                return p
                        return None
                    ids_can_remove = [k for k in ids_at_t if k not in protected_peptide_keys]
                    if not ids_can_remove:
                        ids_can_remove = list(ids_at_t)
                    id_to_remove = min(ids_can_remove, key=lambda k: _signal_for_rt_compete(rep_row(k)))
                    rows_to_remove = [p for p in S if _peptide_scan_key(p) == id_to_remove]
                    for p in rows_to_remove:
                        S.remove(p)
                    cap_removals += len(rows_to_remove)
                    coverage = coverage_at_times(S)
                    removed_this_round = True
                    break
                if not removed_this_round:
                    break
            if cap_removals >= max_cap_iterations:
                print(f"  [RT cap] Stopped at {max_cap_iterations} removals (safety cap); some time points may still have >{MAX_OVERLAP} peptides")
            if cap_removals:
                print(f"  [RT cap] Removed {cap_removals} rows (lowest total_area/apex_intensity) so at most {MAX_OVERLAP} peptides at any time")
            peptides_for_rt_grid = S

    # RT grid: use filtered peptides only (no fill)
    if not peptides_for_rt_grid:
        print("No peptides passed filters; cannot build RT grid. Relax filters (e.g. --no-filtering, lower overhang/scan thresholds) or check CSV and peak window data.")
        return
    print(f"  RT grid: {len(peptides_for_rt_grid)} peptides")
    # Re-sort by position after filter
    peptides_for_rt_grid_sorted = []
    for p in peptides_for_rt_grid:
        peptide_seq = p.get('peptide_seq', '')
        clean_peptide = re.sub(r'\[.*?\]', '', peptide_seq) if peptide_seq else ''
        peptide_length = len(clean_peptide) if clean_peptide else 0
        peptide_start = p.get('start', 0)
        charge = p.get('charge', 0)
        if peptide_start == 0 and clean_peptide:
            protein_seq_upper = protein_sequence.upper()
            clean_peptide_upper = clean_peptide.upper()
            peptide_start_in_protein = protein_seq_upper.find(clean_peptide_upper)
            if peptide_start_in_protein >= 0:
                peptide_start = peptide_start_in_protein + 1
        peptides_for_rt_grid_sorted.append((peptide_start, peptide_length, charge, p))
    peptides_for_rt_grid_sorted.sort(key=lambda x: (x[0] if x[0] > 0 else 999999, -x[1], x[2]))
    peptides_for_rt_grid = [p for _, _, _, p in peptides_for_rt_grid_sorted]
    
    # Test mode: use only N randomly sampled peptides (same N for all combined plots and filter CSV)
    if test and len(peptides_for_rt_grid) > _test_sample_size:
        n_orig = len(peptides_for_rt_grid)
        size = min(_test_sample_size, n_orig)
        indices = np.random.choice(n_orig, size=size, replace=False)
        sampled_tuples = [peptides_for_rt_grid_sorted[i] for i in indices]
        sampled_tuples.sort(key=lambda x: (x[0] if x[0] > 0 else 999999, -x[1], x[2]))
        peptides_for_rt_grid = [p for _, _, _, p in sampled_tuples]
        print(f"[TEST MODE] Using {size} randomly sampled peptides (of {n_orig}) for combined plots and filter CSV")
    
    num_peptides = len(peptides_for_rt_grid)  # Use filtered list for RT grid
    
    # Count unique peptides (ignoring RT) - by sequence only, sequence+charge, sequence+mods, sequence+charge+mods
    # Use canonical types so (seq, charge, mods) is the most granular and has the LARGEST count
    def _canon_charge_rt(c):
        try:
            return int(c) if c not in (None, '') else 0
        except (TypeError, ValueError):
            return 0
    def _canon_mods_rt(m):
        if m in (None, ''):
            return '-'
        s = str(m).strip() or '-'
        return s if isinstance(s, str) else '-'
    unique_peptides_by_seq_only = set()
    unique_peptides_by_seq_charge = set()
    unique_peptides_by_seq_mods = set()
    unique_peptides_by_seq_charge_mods = set()
    for p in peptides_for_rt_grid:
        peptide_seq = (p.get('peptide_seq') or '').strip()
        charge = _canon_charge_rt(p.get('charge'))
        mods = _canon_mods_rt(p.get('modifications'))
        if peptide_seq:
            unique_peptides_by_seq_only.add(peptide_seq)
            unique_peptides_by_seq_charge.add((peptide_seq, charge))
            unique_peptides_by_seq_mods.add((peptide_seq, mods))
            unique_peptides_by_seq_charge_mods.add((peptide_seq, charge, mods))
    # Invariant: (seq, charge, mods) must be >= max((seq, charge), (seq, mods)). If not, repair by adding one triple per (seq, charge) and per (seq, mods).
    n_sc = len(unique_peptides_by_seq_charge)
    n_sm = len(unique_peptides_by_seq_mods)
    n_scm = len(unique_peptides_by_seq_charge_mods)
    if n_scm < n_sc or n_scm < n_sm:
        for (seq, c) in unique_peptides_by_seq_charge:
            for p in peptides_for_rt_grid:
                ps = (p.get('peptide_seq') or '').strip()
                ch = _canon_charge_rt(p.get('charge'))
                if (ps, ch) == (seq, c):
                    mods = _canon_mods_rt(p.get('modifications'))
                    unique_peptides_by_seq_charge_mods.add((seq, c, mods))
                    break
        for (seq, m) in unique_peptides_by_seq_mods:
            for p in peptides_for_rt_grid:
                ps = (p.get('peptide_seq') or '').strip()
                mods = _canon_mods_rt(p.get('modifications'))
                if (ps, mods) == (seq, m):
                    c = _canon_charge_rt(p.get('charge'))
                    unique_peptides_by_seq_charge_mods.add((seq, c, mods))
                    break
    
    # Single line for titles: all four counts
    unique_counts_str = (f'Unique: by sequence only {len(unique_peptides_by_seq_only)} | by sequence+charge {len(unique_peptides_by_seq_charge)} | '
                        f'by sequence+mods {len(unique_peptides_by_seq_mods)} | by sequence+charge+mods {len(unique_peptides_by_seq_charge_mods)}')
    
    print(f"\n{'='*60}")
    print(f"Unique Peptide Counts (ignoring RT):")
    print(f"  Total PSMs in filtered dataset: {num_peptides}")
    print(f"  {unique_counts_str}")
    print(f"{'='*60}")
    
    # Generate unique peptide sequence grid plots whenever we're on the combined (all-ions) visualization.
    # Include ALL peptides in the unique sequences plot. Generate both full and single-AA-only grids.
    if ion_type_filter is None:
        print(f"\n{'='*60}")
        print(f"Generating unique peptides grid plots (one row per sequence+charge+modifications)...")
        print(f"{'='*60}")
        
        # Create output filename for unique peptides plot (save to results directory)
        base_filename = os.path.basename(output_file)
        if base_filename.endswith('.png'):
            unique_pep_filename = base_filename.replace('.png', '_unique_peptides.png')
        else:
            unique_pep_filename = base_filename + '_unique_peptides.png'
        unique_pep_output_file = os.path.join(results_dir, unique_pep_filename)
        
        # Full protein on x-axis, all accepted peptides on y-axis. Each row = one unique peptide (sequence + charge + mods). Each block = main row + c/z fragment rows.
        peptides_for_unique_grid = peptides_for_rt_grid
        create_accepted_peptides_protein_grid(
            peptides_for_unique_grid, protein_sequence, protein_id, protein_length,
            unique_pep_output_file, results_dir
        )
    
    # For coverage heatmap: include single-AA overhangs (in test mode only from the test 30 peptides)
    # Only filter out invalid ones (those without fragment_pairs)
    all_single_aa_for_coverage = []
    for o in all_single_aa:
        fragment_pairs = o.get('fragment_pairs', [])
        if fragment_pairs:
            all_single_aa_for_coverage.append(o)
    # In test mode restrict coverage/overhangs to only the test 30 peptides
    multi_aa_for_coverage = multi_aa_overhangs
    if test:
        test_peptide_seqs = set(p.get('peptide_seq', '') for p in peptides_for_rt_grid)
        all_single_aa_for_coverage = [o for o in all_single_aa_for_coverage if o.get('peptide_seq', '') in test_peptide_seqs]
        multi_aa_for_coverage = [o for o in multi_aa_overhangs if o.get('peptide_seq', '') in test_peptide_seqs]
    
    # For coverage heatmap: use multi-AA + single-AA overhangs (test mode = test 30 only)
    all_overhangs_for_coverage = multi_aa_for_coverage + all_single_aa_for_coverage
    print(f"Found {len(all_single_aa_for_coverage)} single-AA overhangs (for coverage heatmap)" + (" [test: 30 peptides only]" if test else " [all valid overhangs]"))
    
    # Aggregate unique fragment pairs and unique peptides per position for SINGLE-AA overhangs only
    # In test mode use only peptides_for_rt_grid (same 30 as RT grid)
    peptides_for_aggregates = peptides_for_rt_grid if test else peptides
    position_to_single_aa_pairs = {}
    position_to_single_aa_peptides = {}
    peptides_with_single_aa = set()  # Track which peptides have single-AA overhangs
    
    for peptide_idx, peptide in enumerate(peptides_for_aggregates):
        # Only count peptides that have single-AA overhangs WITH fragment pairs
        # Check both single_aa_positions and single_aa_fragment_pairs to ensure we only count peptides with actual fragment pairs
        single_aa_fragment_pairs_dict = peptide.get('single_aa_fragment_pairs', {})
        has_single_aa_with_pairs = bool(single_aa_fragment_pairs_dict)
        
        if has_single_aa_with_pairs:
            peptides_with_single_aa.add(peptide_idx)
            # Single-AA overhangs only
            # CRITICAL: Only count peptides that actually contributed fragment pairs (pairs_set must be non-empty)
            # INVARIANT: For each position, unique_fragment_pairs >= unique_peptides
            # This is because each peptide must contribute at least one fragment pair to be counted
            peptide_key = (peptide.get('peptide_seq', ''), peptide.get('rt', 0.0), peptide.get('charge', 0))
            
            for pos, pairs_set in single_aa_fragment_pairs_dict.items():
                # CRITICAL: Only process positions with non-empty fragment pair sets
                # This ensures we never count a peptide at a position where it contributes no fragment pairs
                if 1 <= pos <= protein_length and pairs_set and len(pairs_set) > 0:
                    if pos not in position_to_single_aa_pairs:
                        position_to_single_aa_pairs[pos] = set()
                        position_to_single_aa_peptides[pos] = set()
                    
                    # Store (peptide_key, pair_str) so z6 from one peptide ≠ z6 from another
                    for pair_str in pairs_set:
                        position_to_single_aa_pairs[pos].add((peptide_key, pair_str))
                    
                    # Add peptide to position count (all peptides that contribute pairs at this position)
                    # This matches the RT grid which shows all peptides with single-AA overhangs
                    if peptide_key[0]:  # Only add if peptide_seq is not empty
                        position_to_single_aa_peptides[pos].add(peptide_key)
    
    # Aggregate unique fragment pairs and unique peptides per position for MULTI-AA overhangs
    # Only include peptides that have multi-AA overhangs (exclude peptides that only have single-AA overhangs)
    
    # First, identify which peptides have multi-AA overhangs (using composite keys)
    # In test mode use only peptides_for_rt_grid (same 30 as RT grid)
    peptides_with_multi_aa = set()  # Set of (peptide_seq, rt, charge) tuples
    peptide_seq_to_keys = {}  # Map peptide_seq -> set of (peptide_seq, rt, charge) tuples
    
    for peptide in peptides_for_aggregates:
        peptide_key = (peptide.get('peptide_seq', ''), peptide.get('rt', 0.0), peptide.get('charge', 0))
        peptide_seq = peptide_key[0]
        
        if peptide_seq not in peptide_seq_to_keys:
            peptide_seq_to_keys[peptide_seq] = set()
        peptide_seq_to_keys[peptide_seq].add(peptide_key)
        
        # Check if this peptide has any multi-AA overhangs (length > 1, i.e., 2-AA through 10-AA)
        has_multi_aa = False
        for overhang_type in OVERHANG_TYPE_KEYS:
            length = int(overhang_type.replace('aa', ''))
            if length > 1:  # Only check multi-AA (not single-AA)
                if bool(peptide.get(f'{overhang_type}_overhangs', {})):
                    has_multi_aa = True
                    break
        
        # Also check if peptide has multi-AA overhangs from the original parsing
        # (peptides from multi_aa_overhangs list)
        if not has_multi_aa:
            # Check if this peptide appears in any multi-AA overhang from all_overhangs
            for overhang in all_overhangs:
                if overhang.get('length', 0) > 1 and overhang.get('peptide_seq', '') == peptide_seq:
                    has_multi_aa = True
                    break
        
        if has_multi_aa:
            peptides_with_multi_aa.add(peptide_key)
    
    position_to_multi_aa_pairs = {}
    position_to_multi_aa_peptides = {}
    multi_aa_peptide_keys = set()  # Track unique peptide keys contributing to multi-AA overhangs
    
    # Process all multi-AA overhangs from all_overhangs (exclude single-AA, length=1)
    # Only count peptides that have multi-AA overhangs
    for overhang in all_overhangs:
        if overhang['length'] == 1:
            continue  # Skip single-AA overhangs
        
        start = overhang['start']
        end = overhang['end']
        peptide_seq = overhang.get('peptide_seq', '')
        
        # Get all composite keys for peptides with this sequence that have multi-AA overhangs
        matching_keys = []
        if peptide_seq in peptide_seq_to_keys:
            for key in peptide_seq_to_keys[peptide_seq]:
                if key in peptides_with_multi_aa:
                    matching_keys.append(key)
        
        # Skip if no matching peptides with multi-AA overhangs
        if not matching_keys:
            continue
        
        # Get fragment pairs for this overhang
        fragment_pairs = overhang.get('fragment_pairs', [])
        if fragment_pairs:
            # Normalize fragment pairs
            normalized_pairs = set()
            for pair in fragment_pairs:
                normalized_pair = normalize_fragment_pair(pair)
                normalized_pairs.add(normalized_pair)
        else:
            normalized_pairs = set()
        
        # Count each position covered by this overhang
        for pos in range(start, end + 1):
            if 1 <= pos <= protein_length:
                if pos not in position_to_multi_aa_pairs:
                    position_to_multi_aa_pairs[pos] = set()
                    position_to_multi_aa_peptides[pos] = set()
                # Add fragment pairs for this position
                position_to_multi_aa_pairs[pos].update(normalized_pairs)
                # Track unique peptides using composite keys (all matching peptides)
                for key in matching_keys:
                    position_to_multi_aa_peptides[pos].add(key)
                    multi_aa_peptide_keys.add(key)
    
    # Count unique fragment pairs and unique peptides per position for SINGLE-AA overhangs
    # CRITICAL: Recalculate based on what's actually shown in RT grid to ensure consistency
    # For each position, count fragment pairs from peptides_for_rt_grid that have single-AA overhangs at that position
    # This ensures the histogram matches the RT grid: each red box = at least 1 fragment pair
    single_aa_fragment_pair_counts = []
    single_aa_unique_peptide_counts = []
    
    # Count ALL peptides with single-AA overhangs (not just those with unique pairs)
    # This matches the RT grid which shows all 778 peptides
    all_peptides_with_single_aa = set()
    
    for pos in range(1, protein_length + 1):
        # Count fragment pairs from peptides_for_rt_grid that have single-AA overhangs at this position
        pairs_at_pos = set()
        peptides_at_pos = set()
        
        for peptide in peptides_for_rt_grid:
            single_aa_fragment_pairs_dict = peptide.get('single_aa_fragment_pairs', {})
            if pos in single_aa_fragment_pairs_dict:
                pairs_set = single_aa_fragment_pairs_dict[pos]
                if pairs_set:  # Only count if there are fragment pairs
                    peptide_key = (peptide.get('peptide_seq', ''), peptide.get('rt', 0.0), peptide.get('charge', 0))
                    for pair_str in pairs_set:
                        pairs_at_pos.add((peptide_key, pair_str))  # z6 from one peptide ≠ z6 from another
                    if peptide_key[0]:  # Only add if peptide_seq is not empty
                        peptides_at_pos.add(peptide_key)
                        all_peptides_with_single_aa.add(peptide_key)
        
        single_aa_fragment_pair_counts.append(len(pairs_at_pos))
        single_aa_unique_peptide_counts.append(len(peptides_at_pos))
    
    # Count unique fragment pairs and unique peptides per position for MULTI-AA overhangs
    multi_aa_fragment_pair_counts = [len(position_to_multi_aa_pairs.get(pos, set())) for pos in range(1, protein_length + 1)]
    multi_aa_unique_peptide_counts = [len(position_to_multi_aa_peptides.get(pos, set())) for pos in range(1, protein_length + 1)]
    
    # Create figure with three subplots: heatmap (top), single-AA histogram, grid (bottom) — no scatter plot
    # Add extra width for colorbar on the left - increased to prevent overlap with sequences
    fig_width = max(20, protein_length * 0.15) + 2.5 + 2.0  # Add 2.5 inches for colorbar + 2.0 inches for right margin
    fig_height = max(12, num_peptides * 0.15) + 8 + 2.0  # Add 2.0 inches for top/bottom margins
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    # Create subplots: heatmap (15% height), single-AA histogram (8% height), grid (77% height)
    # Use 2 columns: left column for plots, right column for signal intensity histogram
    gs = fig.add_gridspec(3, 2, height_ratios=[1.5, 0.8, 6.9], width_ratios=[10, 1.5], hspace=0.12, wspace=0.05)
    ax_heatmap = fig.add_subplot(gs[0, 0])
    ax_hist_single = fig.add_subplot(gs[1, 0], sharex=ax_heatmap)
    ax_grid = fig.add_subplot(gs[2, 0], sharex=ax_heatmap)
    ax_signal = fig.add_subplot(gs[2, 1], sharey=ax_grid)  # Share y-axis with RT grid for automatic synchronization
    
    # ===== PLOT 1: Multi-AA Overhang Heatmap =====
    if all_overhangs:
        # Get all unique overhang lengths from ALL overhangs (including all single-AA for complete coverage)
        unique_lengths = sorted(set(o['length'] for o in all_overhangs_for_coverage))
        
        # Use all unique lengths for y-axis scaling (includes all single-AA for complete coverage)
        unique_lengths_for_yaxis = unique_lengths
        
        # Create 2D array: coverage[residue_position][overhang_length] = count
        # Use all unique_lengths for the actual coverage data (includes all single-AA)
        coverage = np.zeros((len(unique_lengths), protein_length), dtype=int)
        length_to_idx = {length: idx for idx, length in enumerate(unique_lengths)}
        
        # Create mapping from all lengths to y-axis positions
        yaxis_length_to_idx = {length: idx for idx, length in enumerate(unique_lengths_for_yaxis)}
        
        # Count overhangs for each residue position and length (use all overhangs for complete coverage)
        for overhang in all_overhangs_for_coverage:
            start = overhang['start']
            end = overhang['end']
            length = overhang['length']
            
            if start < 1 or end > protein_length:
                continue
            
            # For ion type filtering, check if overhang matches
            if ion_type_filter is not None:
                fragment_pairs = overhang.get('fragment_pairs', [])
                if fragment_pairs:
                    has_matching_pair = any(filter_fragment_pair_by_ion_type(p, ion_type_filter) for p in fragment_pairs)
                    if not has_matching_pair:
                        continue
                else:
                    # Overhang without fragment_pairs - skip it (shouldn't happen for valid overhangs)
                    continue
            
            start_idx = start - 1
            end_idx = end - 1
            # Add to coverage if this length is in the set
            if length in length_to_idx:
                length_idx = length_to_idx[length]
                for pos in range(start_idx, end_idx + 1):
                    coverage[length_idx, pos] += 1
        
        # Create custom colormap: blue for 0, white for 1, black for max
        # Gradient from white to black for values 1 to max
        from matplotlib.colors import ListedColormap, LinearSegmentedColormap
        import matplotlib.cm as cm
        
        max_coverage = np.max(coverage) if np.max(coverage) > 0 else 1
        
        # Create colormap with:
        # - Blue for 0
        # - White for 1 (smallest non-zero)
        # - Black for max (largest)
        # - Gradient from white to black for 1 to max
        n_colors = 1024
        colors_list = []
        
        # Color 0: Blue
        colors_list.append((0.0, 0.0, 1.0))  # Blue for 0
        
        # Colors 1 to n_colors-1: White to black gradient
        for i in range(1, n_colors):
            t = (i - 1) / (n_colors - 2)  # t goes from 0 (white) to 1 (black)
            # White (1,1,1) to Black (0,0,0)
            r = 1.0 - t
            g = 1.0 - t
            b = 1.0 - t
            colors_list.append((r, g, b))
        
        custom_cmap = ListedColormap(colors_list)
        
        # Normalize coverage: 0 -> 0 (blue), 1 -> 1 (white), max -> 1023 (black)
        # Linear mapping from 1 to max across color indices 1-1023
        coverage_normalized = coverage.copy().astype(float)
        if max_coverage > 1:
            for i in range(coverage.shape[0]):
                for j in range(coverage.shape[1]):
                    val = coverage[i, j]
                    if val == 0:
                        coverage_normalized[i, j] = 0  # Blue (index 0)
                    elif val == 1:
                        coverage_normalized[i, j] = 1  # White (index 1)
                    else:
                        # Map 1 to max linearly to indices 1 to 1023
                        # val ranges from 1 to max_coverage
                        # Map to indices 1 to 1023
                        normalized_val = 1 + ((val - 1) / (max_coverage - 1)) * 1022  # Map to 1-1023
                        coverage_normalized[i, j] = normalized_val
        elif max_coverage == 1:
            # All values are either 0 or 1
            for i in range(coverage.shape[0]):
                for j in range(coverage.shape[1]):
                    val = coverage[i, j]
                    if val == 0:
                        coverage_normalized[i, j] = 0  # Blue
                    else:
                        coverage_normalized[i, j] = 1  # White
        
        # Create full-size coverage array for y-axis (pad with zeros for missing lengths)
        # Use ALL overhangs (including all single-AA) for complete coverage visualization
        coverage_full = np.zeros((len(unique_lengths_for_yaxis), protein_length), dtype=int)
        
        # Use all overhangs for coverage (including all single-AA) to show complete fragment coverage
        overhangs_for_coverage = all_overhangs_for_coverage
        
        # Build coverage directly in the full array from overhangs
        for overhang in overhangs_for_coverage:
            start = overhang['start']
            end = overhang['end']
            length = overhang['length']
            
            if start < 1 or end > protein_length:
                continue
            
            if length not in yaxis_length_to_idx:
                continue
            
            # For combined plot, include all overhangs
            # For filtered plots, only include if they match the filter
            if ion_type_filter is not None:
                # Check if this overhang matches the ion type filter
                fragment_pairs = overhang.get('fragment_pairs', [])
                if fragment_pairs:
                    has_matching_pair = any(filter_fragment_pair_by_ion_type(p, ion_type_filter) for p in fragment_pairs)
                    if not has_matching_pair:
                        continue
                else:
                    # Overhang without fragment_pairs - skip it (shouldn't happen for valid overhangs)
                    continue
            
            start_idx = start - 1
            end_idx = end - 1
            yaxis_idx = yaxis_length_to_idx[length]
            
            for pos in range(start_idx, end_idx + 1):
                coverage_full[yaxis_idx, pos] += 1
        
        # Normalize the full coverage array: 0 -> 0 (blue), 1 -> 1 (white), max -> 1023 (black)
        max_coverage_full = np.max(coverage_full) if np.max(coverage_full) > 0 else 1
        coverage_normalized_full = coverage_full.copy().astype(float)
        if max_coverage_full > 1:
            for i in range(coverage_full.shape[0]):
                for j in range(coverage_full.shape[1]):
                    val = coverage_full[i, j]
                    if val == 0:
                        coverage_normalized_full[i, j] = 0  # Blue (index 0)
                    elif val == 1:
                        coverage_normalized_full[i, j] = 1  # White (index 1)
                    else:
                        # Map 1 to max linearly to indices 1 to 1023
                        normalized_val = 1 + ((val - 1) / (max_coverage_full - 1)) * 1022  # Map to 1-1023
                        coverage_normalized_full[i, j] = normalized_val
        elif max_coverage_full == 1:
            # All values are either 0 or 1
            for i in range(coverage_full.shape[0]):
                for j in range(coverage_full.shape[1]):
                    val = coverage_full[i, j]
                    if val == 0:
                        coverage_normalized_full[i, j] = 0  # Blue
                    else:
                        coverage_normalized_full[i, j] = 1  # White
        
        im = ax_heatmap.imshow(coverage_normalized_full, aspect='auto', cmap=custom_cmap, 
                              interpolation='nearest', origin='lower', vmin=0, vmax=1023)
        
        # Set y-axis to show overhang lengths
        # Use the unfiltered range for consistent scaling across all plots
        ax_heatmap.set_yticks(range(len(unique_lengths_for_yaxis)))
        ax_heatmap.set_yticklabels([str(l) for l in unique_lengths_for_yaxis], fontsize=8)
        # Set y-axis limits to match the full range
        ax_heatmap.set_ylim(-0.5, len(unique_lengths_for_yaxis) - 0.5)
        
        ax_heatmap.set_ylabel('Overhang Length\n(residues)', fontsize=10, fontweight='bold')
        # Update title based on ion type filter
        ion_type_label = ""
        if ion_type_filter == 'c':
            ion_type_label = " (c ions only)"
        elif ion_type_filter == 'z':
            ion_type_label = " (z/z+1 ions only)"
        
        ax_heatmap.set_title(f'Overhang Coverage Heatmap (Single-AA + Multi-AA){ion_type_label}', 
                           fontsize=11, fontweight='bold', pad=10)
        ax_heatmap.set_xticks([])  # Hide x-axis ticks (will show on bottom plot)
        # Set x-axis label with large font to match RT grid style
        ax_heatmap.set_xlabel('Protein Position (1-indexed) with Amino Acid', fontsize=18, fontweight='bold')
        ax_heatmap.set_xlim(0, protein_length)
        
        # Store heatmap image and max_coverage for colorbar creation later
        heatmap_im = im
        heatmap_max_coverage = max(max_coverage, max_coverage_full)  # Use the larger of the two
    
    # Initialize tau for quality metric calculations (geometric properties only, no D_total)
    # Must be initialized before single-AA histogram (CV / quality metric)
    tau = 0.1732  # Default value for resolution heterogeneity scale
    
    # ===== PLOT 2: Single-AA Overhang Histogram =====
    # (Scatter plot removed; figure has heatmap, single-AA histogram, RT grid only)
    # Create stacked histogram with colors matching RT grid
    # For each position, collect all CV values from single-AA overhangs (one per fragment pair)
    position_cvs = {}  # pos -> list of CV values (one per fragment pair)
    histogram_all_cvs = []  # Collect CV values for normalization
    
    for peptide in peptides_for_rt_grid:
        single_aa_positions = peptide.get('single_aa_positions', {})
        single_aa_fragment_pairs = peptide.get('single_aa_fragment_pairs', {})
        peptide_seq = peptide.get('peptide_seq', '')
        clean_peptide = re.sub(r'\[.*?\]', '', peptide_seq) if peptide_seq else ''
        
        for pos in single_aa_positions.keys():
            if 1 <= pos <= protein_length:
                if pos not in position_cvs:
                    position_cvs[pos] = []
                # Calculate CV for each fragment pair at this position
                pairs_at_pos = single_aa_fragment_pairs.get(pos, set())
                for pair_str in pairs_at_pos:
                    # Extract fragment ordinal from pair string (e.g., "c4-c5" -> 4 or 5)
                    # Use the larger fragment number as the fragment length for CV calculation
                    normalized_pair = normalize_fragment_pair(pair_str)
                    match = re.match(r'^([a-z])(\d+)-([a-z])(\d+)$', normalized_pair.lower())
                    if match:
                        series1, num1_str, series2, num2_str = match.groups()
                        num1, num2 = int(num1_str), int(num2_str)
                        # Use the larger fragment number as the effective fragment length
                        fragment_ordinal = max(num1, num2)
                    else:
                        fragment_ordinal = None
                    
                    # Create a minimal overhang dict for CV calculation
                    # For single-AA overhangs, L=1 always, but fragment ordinal affects resolution
                    overhang_dict = {
                        'peptide_seq': peptide_seq,
                        'start': pos,
                        'end': pos,
                        'length': 1,  # Single-AA overhang
                        'fragment_ordinal': fragment_ordinal  # Store fragment ordinal for CV calculation
                    }
                    # Calculate quality metric based only on geometric properties
                    # (peptide length, fragment length, overhang length) without D_total
                    cv = calculate_geometric_quality_metric(overhang_dict, tau=tau)
                    if cv is not None and not np.isnan(cv) and np.isfinite(cv):
                        position_cvs[pos].append(cv)
                        histogram_all_cvs.append(cv)
    
    # Create colormap for histogram (same as RT grid - will be defined later, but use same structure)
    from matplotlib.colors import LinearSegmentedColormap
    # Note: The colormap will be defined later for RT grid, but we use the same structure here
    # For now, create a temporary colormap that matches the RT grid colormap
    cdict_hist = {
        'red': [
            (0.0, 1.0, 1.0),      # White at 0.0
            (0.1, 0.7, 0.7),      # Gray at 0.1
            (0.15, 0.0, 0.0),     # Blue start at 0.15
            (0.5, 1.0, 1.0),      # Yellow at middle
            (1.0, 1.0, 1.0),      # Red at 1.0
        ],
        'green': [
            (0.0, 1.0, 1.0),      # White at 0.0
            (0.1, 0.7, 0.7),      # Gray at 0.1
            (0.15, 0.0, 1.0),     # Cyan start at 0.15
            (0.5, 1.0, 1.0),      # Yellow at middle
            (0.83, 0.0, 0.0),     # Red transition
            (1.0, 0.0, 0.0),      # Red at 1.0
        ],
        'blue': [
            (0.0, 1.0, 1.0),      # White at 0.0
            (0.1, 0.7, 0.7),      # Gray at 0.1
            (0.15, 1.0, 1.0),     # Blue start at 0.15
            (0.5, 0.0, 0.0),      # Yellow at middle
            (1.0, 0.0, 0.0),      # Red at 1.0
        ]
    }
    hist_cmap = LinearSegmentedColormap('hist_qvalue', cdict_hist, N=256)
    hist_norm = plt.Normalize(vmin=0.0, vmax=1.0)
    
    # Define bar width and positions for single-AA histogram (needed for bar plots)
    bar_width = 0.4
    x_positions = list(range(protein_length))
    x1 = [x - bar_width/2 for x in x_positions]
    x2 = [x + bar_width/2 for x in x_positions]
    
    # Create stacked bars with colors from RT grid colormap
    bottom_fragment_pairs = np.zeros(protein_length)
    
    # Draw fragment pairs as stacked colored segments
    if histogram_all_cvs:
        min_cv = min(histogram_all_cvs)
        max_cv = max(histogram_all_cvs)
        cv_range = max_cv - min_cv if max_cv > min_cv else 1.0
        
        for pos in range(1, protein_length + 1):
            pos_idx = pos - 1
            if pos in position_cvs:
                cvs = sorted(position_cvs[pos])  # Sort for consistent stacking
                
                # Stack each CV as a colored segment
                current_bottom = 0
                for cv_val in cvs:
                    # Normalize CV to grid value range [0.15, 1.0]
                    normalized_cv = (cv_val - min_cv) / cv_range if cv_range > 0 else 0.5
                    grid_value = 0.15 + normalized_cv * 0.85
                    
                    # Get color from colormap
                    color = hist_cmap(hist_norm(grid_value))
                    
                    # Draw segment
                    ax_hist_single.bar(pos_idx - bar_width/2, 1, width=bar_width, 
                                      bottom=current_bottom, color=color, 
                                      edgecolor='black', linewidth=0.3, align='edge')
                    current_bottom += 1
                
                bottom_fragment_pairs[pos_idx] = len(cvs)
    else:
        # Fallback: use original bars if no q-values
        bars3 = ax_hist_single.bar(x1, single_aa_fragment_pair_counts, width=bar_width, 
                            edgecolor='black', linewidth=0.5, color='steelblue', alpha=0.7, 
                            align='edge', label='Unique Fragment Pairs')
    
    # Also show unique peptides count (as a separate bar)
    bars4 = ax_hist_single.bar(x2, single_aa_unique_peptide_counts, width=bar_width, 
                        edgecolor='black', linewidth=0.5, color='black', alpha=0.7, 
                        align='edge', label='Unique Peptides')
    
    ax_hist_single.set_ylabel('Count', fontsize=10, fontweight='bold')
    ax_hist_single.set_title('Single-AA Overhangs: Unique Fragment Pairs (colored by CV) and Unique Peptides per Residue', 
                     fontsize=11, fontweight='bold', pad=10)
    ax_hist_single.legend(loc='upper right', fontsize=8)
    ax_hist_single.grid(axis='y', alpha=0.3, linestyle='--')
    ax_hist_single.set_xticks([])  # Hide x-axis ticks (will show on bottom plot)
    ax_hist_single.set_xlim(0, protein_length)
    ax_hist_single.tick_params(axis='y', labelsize=12)
    
    # Add statistics box for single-AA histogram
    # Calculate total unique peptides contributing to this histogram
    # Count ALL peptides with single-AA overhangs (matching what's shown in RT grid)
    total_single_aa_unique_peptides = len(all_peptides_with_single_aa)
    
    # Count peptides with single-AA overhangs from RT grid (use same criteria as all_peptides_with_single_aa)
    # Must have non-empty fragment pairs (same as all_peptides_with_single_aa)
    peptides_with_single_aa_only_set = set()
    for p in peptides_for_rt_grid:
        single_aa_fragment_pairs_dict = p.get('single_aa_fragment_pairs', {})
        # Check if peptide has any single-AA fragment pairs (non-empty)
        has_single_aa_pairs = False
        for pos, pairs_set in single_aa_fragment_pairs_dict.items():
            if pairs_set:  # Non-empty fragment pairs
                has_single_aa_pairs = True
                break
        if has_single_aa_pairs:
            peptide_key = (p.get('peptide_seq', ''), p.get('rt', 0.0), p.get('charge', 0))
            if peptide_key[0]:  # Only add if peptide_seq is not empty
                peptides_with_single_aa_only_set.add(peptide_key)
    num_peptides_with_single_aa = len(peptides_with_single_aa_only_set)
    
    # Total unique (peptide, position, fragment pair): sum per position (each (peptide, pair) at a position counted once)
    total_single_aa_unique_pairs = sum(len(position_to_single_aa_pairs.get(pos, set())) for pos in range(1, protein_length + 1))
    
    # Globally unique (peptide, fragment pair): union across positions (z6 from one peptide ≠ z6 from another)
    global_unique_peptide_pairs = len(set().union(*[position_to_single_aa_pairs.get(pos, set()) for pos in range(1, protein_length + 1)]))
    
    stats_text_single = f'Total unique peptides: {total_single_aa_unique_peptides}\nRT grid shows: {num_peptides} peptides\nSingle-AA only: {num_peptides_with_single_aa}\nTotal unique (peptide, pos, pair): {total_single_aa_unique_pairs}\nGlobally unique (peptide, fragment pair): {global_unique_peptide_pairs}'
    # Position box in upper left corner
    ax_hist_single.text(0.02, 0.98, stats_text_single, transform=ax_hist_single.transAxes,
                        fontsize=10, fontweight='bold', verticalalignment='top',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', linewidth=1.5, alpha=0.9),
                        zorder=100)
    
    # Add colorbar next to single-AA histogram
    # Note: Colorbar will be created later as a single horizontal colorbar between histogram and RT grid
    
    # ===== PLOT 3: RT Grid =====
    # Create grid: rows = peptides with single-AA or 2-AA overhangs (by RT), cols = protein positions
    # Grid will store: 0.0 = no coverage, 0.1 = peptide coverage (black), >0.1 = CV (normalized)
    grid = np.zeros((num_peptides, protein_length), dtype=float)
    cv_grid = np.full((num_peptides, protein_length), np.nan, dtype=float)  # Store CV values separately
    
    # Calculate quality metric for each fragment pair and average per position
    # Note: tau is already initialized earlier (before single-AA histogram)
    # We use geometric properties only (peptide length, fragment length, overhang length)
    # and do NOT use D_total, as D content is an unknown variable
    
    # Fill grid with peptide coverage (gray) and single-AA overhangs (colored by CV)
    all_cvs = []  # Collect all CV values for normalization
    for i, peptide in enumerate(peptides_for_rt_grid):
        single_aa_positions = peptide.get('single_aa_positions', {})
        single_aa_fragment_pairs = peptide.get('single_aa_fragment_pairs', {})
        
        # Get peptide start and end positions in protein coordinates
        peptide_start = peptide.get('start', 0)
        peptide_end = peptide.get('end', 0)
        peptide_seq = peptide.get('peptide_seq', '')
        clean_peptide = re.sub(r'\[.*?\]', '', peptide_seq) if peptide_seq else ''
        
        # CRITICAL: Single-AA overhangs MUST be within the peptide boundaries
        # If peptide boundaries are missing or incorrect, calculate them from the peptide sequence
        if (peptide_start == 0 or peptide_end == 0) and clean_peptide:
            # Try to find peptide in protein sequence to get actual start/end
            protein_seq_upper = protein_sequence.upper()
            clean_peptide_upper = clean_peptide.upper()
            peptide_start_in_protein = protein_seq_upper.find(clean_peptide_upper)
            if peptide_start_in_protein >= 0:
                peptide_start = peptide_start_in_protein + 1
                peptide_end = peptide_start + len(clean_peptide) - 1
        
        # If still missing, try to infer from single-AA positions (but this should be rare)
        if (peptide_start == 0 or peptide_end == 0) and single_aa_positions:
            all_single_aa_pos = list(single_aa_positions.keys())
            if all_single_aa_pos:
                peptide_start = min(all_single_aa_pos)
                peptide_end = max(all_single_aa_pos)
                # Expand to include full peptide length if we can calculate it
                if clean_peptide:
                    # Try to find peptide in protein sequence
                    protein_seq_upper = protein_sequence.upper()
                    clean_peptide_upper = clean_peptide.upper()
                    peptide_start_in_protein = protein_seq_upper.find(clean_peptide_upper)
                    if peptide_start_in_protein >= 0:
                        peptide_start = peptide_start_in_protein + 1
                        peptide_end = peptide_start + len(clean_peptide) - 1
        
        # CRITICAL: Filter out any single-AA positions that are outside the peptide boundaries
        # This ensures all displayed overhangs are within the peptide (as they should be)
        if peptide_start > 0 and peptide_end > 0:
            # Filter single-AA positions to only include those within peptide boundaries
            filtered_single_aa_positions = {}
            filtered_single_aa_fragment_pairs = {}
            for pos, count in single_aa_positions.items():
                if peptide_start <= pos <= peptide_end:
                    filtered_single_aa_positions[pos] = count
                    if pos in single_aa_fragment_pairs:
                        filtered_single_aa_fragment_pairs[pos] = single_aa_fragment_pairs[pos]
            
            # Update to use filtered positions
            single_aa_positions = filtered_single_aa_positions
            single_aa_fragment_pairs = filtered_single_aa_fragment_pairs
        
        # FIRST: Mark ALL sequence positions covered by this peptide as black (value = 0.1)
        # This shows which protein positions the peptide sequence maps to
        # We mark ALL positions first, then overlay colored single-AA positions on top
        # This ensures the entire peptide span is either black or colored, never white
        if peptide_start > 0 and peptide_end > 0:
            for pos in range(peptide_start, peptide_end + 1):
                if 1 <= pos <= protein_length:
                    # Mark ALL positions in peptide range as black (will be overwritten by colored single-AA if present)
                    grid[i, pos - 1] = 0.1  # Black for peptide sequence coverage
        
        # SECOND: Calculate CV for each single-AA position and overlay colored values
        # This will overwrite the gray values for single-AA overhang positions
        # Each fragment pair gets its own CV calculated independently based on its fragment ordinal
        for pos in single_aa_positions.keys():
            if 1 <= pos <= protein_length:
                fragment_pairs = single_aa_fragment_pairs.get(pos, set())
                if fragment_pairs:
                    # Calculate CV for each fragment pair independently
                    # Each fragment pair has its own CV based on fragment length (ordinal) and peptide properties
                    cv_values = []
                    for pair_str in fragment_pairs:
                        # Extract fragment ordinal from pair string (e.g., "c4-c5" -> 4 or 5)
                        # Use the larger fragment number as the fragment length for CV calculation
                        normalized_pair = normalize_fragment_pair(pair_str)
                        match = re.match(r'^([a-z])(\d+)-([a-z])(\d+)$', normalized_pair.lower())
                        if match:
                            series1, num1_str, series2, num2_str = match.groups()
                            num1, num2 = int(num1_str), int(num2_str)
                            # Use the larger fragment number as the effective fragment length
                            # This represents the "resolution" of the fragment pair
                            fragment_ordinal = max(num1, num2)
                        else:
                            # Fallback: use default
                            fragment_ordinal = None
                        
                        # Create a minimal overhang dict for CV calculation
                        # For single-AA overhangs, L=1 always, but fragment ordinal affects resolution
                        overhang_dict = {
                            'peptide_seq': peptide_seq,
                            'start': pos,
                            'end': pos,
                            'length': 1,  # Single-AA overhang
                            'fragment_ordinal': fragment_ordinal  # Store fragment ordinal for potential use
                        }
                        # Calculate quality metric based only on geometric properties
                        # (peptide length, fragment length, overhang length) without D_total
                        cv = calculate_geometric_quality_metric(overhang_dict, tau=tau)
                        if cv is not None and not np.isnan(cv) and np.isfinite(cv):
                            cv_values.append(cv)
                    
                    # Average CV for all fragment pairs at this position
                    # Each fragment pair's CV was calculated independently
                    if cv_values:
                        avg_cv = np.mean(cv_values)
                        cv_grid[i, pos - 1] = avg_cv
                        all_cvs.append(avg_cv)
    
    # Normalize CV values to [0.15, 1.0] range for coloring
    # 0.1 = black (peptide coverage, not single-AA), 0.15-1.0 = colored by CV
    # Lower CV (better/more reliable) -> lower values (blue), higher CV (worse/less reliable) -> higher values (red)
    if all_cvs:
        min_cv = min(all_cvs)
        max_cv = max(all_cvs)
        cv_range = max_cv - min_cv if max_cv > min_cv else 1.0
        
        # Map CV to grid values: low CV (good) -> lower values, high CV (bad) -> higher values
        # Use range [0.15, 1.0] where 0.15 = best CV (lowest), 1.0 = worst CV (highest)
        # This ensures black (0.1) is separate from CV colors
        for i in range(num_peptides):
            for pos in range(protein_length):
                if not np.isnan(cv_grid[i, pos]):
                    # Normalize CV to [0, 1] then map to [0.15, 1.0]
                    normalized_cv = (cv_grid[i, pos] - min_cv) / cv_range if cv_range > 0 else 0.5
                    grid[i, pos] = 0.15 + normalized_cv * 0.85  # Map to [0.15, 1.0]
    else:
        # No CV values, just mark single-AA positions
        for i, peptide in enumerate(peptides_for_rt_grid):
            single_aa_positions = peptide.get('single_aa_positions', {})
            for pos in single_aa_positions.keys():
                if 1 <= pos <= protein_length:
                    grid[i, pos - 1] = 1.0  # Default to max if no CV
    
    # Create colormap: white (0.0) -> black (0.1) -> blue to red (0.15-1.0 for CV)
    # 0.0 = no coverage (white)
    # 0.1 = peptide coverage (black) - NOT single-AA overhangs, NO SCORE
    # 0.15-1.0 = single-AA overhang colored by CV (blue=low CV/good/reliable, red=high CV/bad/unreliable)
    from matplotlib.colors import ListedColormap, LinearSegmentedColormap, BoundaryNorm
    import matplotlib.cm as cm
    
    # Create custom colormap with discrete boundaries
    # Use BoundaryNorm to ensure 0.1 maps exactly to black
    # Create color list: white, black, then gradient for q-values
    colors_list = [
        (1.0, 1.0, 1.0),      # White for no coverage (0.0)
        (0.0, 0.0, 0.0),      # Black for peptide coverage (0.1) - fixed black
    ]
    
    # Add gradient from blue (low q-value) to red (high q-value) for range [0.2, 1.0]
    n_colors = 200
    for i in range(n_colors):
        t = i / (n_colors - 1)  # t goes from 0 to 1
        # Blue (low q-value, good) -> Cyan -> Yellow -> Red (high q-value, bad)
        if t < 0.33:
            # Blue to cyan
            r = 0.0
            g = t * 3.0
            b = 1.0
        elif t < 0.67:
            # Cyan to yellow
            r = (t - 0.33) * 3.0
            g = 1.0
            b = 1.0 - (t - 0.33) * 3.0
        else:
            # Yellow to red
            r = 1.0
            g = 1.0 - (t - 0.67) * 3.0
            b = 0.0
        
        r = max(0.0, min(1.0, r))
        g = max(0.0, min(1.0, g))
        b = max(0.0, min(1.0, b))
        colors_list.append((r, g, b))
    
    # Create colormap with smooth gradient for q-values
    # Use LinearSegmentedColormap with specific color stops
    # Map: 0.0 -> white, 0.1 -> black, 0.15-1.0 -> smooth gradient (blue to red)
    
    # Create color dictionary for LinearSegmentedColormap
    # This creates a smooth gradient in the q-value range
    cdict = {
        'red': [
            (0.0, 1.0, 1.0),      # White at 0.0
            (0.1, 0.0, 0.0),      # Black at 0.1
            (0.15, 0.0, 0.0),     # Blue start at 0.15
            (0.5, 1.0, 1.0),      # Yellow at middle
            (1.0, 1.0, 1.0),      # Red at 1.0
        ],
        'green': [
            (0.0, 1.0, 1.0),      # White at 0.0
            (0.1, 0.0, 0.0),      # Black at 0.1
            (0.15, 0.0, 1.0),     # Cyan start at 0.15
            (0.5, 1.0, 1.0),      # Yellow at middle
            (0.83, 0.0, 0.0),     # Red transition
            (1.0, 0.0, 0.0),      # Red at 1.0
        ],
        'blue': [
            (0.0, 1.0, 1.0),      # White at 0.0
            (0.1, 0.0, 0.0),      # Black at 0.1
            (0.15, 1.0, 1.0),     # Blue start at 0.15
            (0.5, 0.0, 0.0),      # Yellow at middle
            (1.0, 0.0, 0.0),      # Red at 1.0
        ]
    }
    
    custom_cmap = LinearSegmentedColormap('white_black_qvalue', cdict, N=256)
    # Use standard normalization - the colormap handles the mapping
    norm = plt.Normalize(vmin=0.0, vmax=1.0)
    
    # Display grid with pcolormesh
    # Grid values: 0.0 = white, 0.5 = orange, 1.0 = red
    x = np.arange(protein_length + 1)
    y = np.arange(num_peptides + 1)
    X, Y = np.meshgrid(x, y)
    
    im_grid = ax_grid.pcolormesh(X, Y, grid, cmap=custom_cmap, norm=norm,
                                 edgecolors='black', linewidths=0.3, shading='flat')
    
    # Draw c/z fragment pair labels inside each grid cell; color by ion type for c vs z balance at a glance
    # c ions = cyan, z ions = amber; when both in same cell, show two lines so overlap is visible
    _c_color, _z_color = '#22d3ee', '#fbbf24'  # cyan, amber (readable on dark/colored cells)
    for i, peptide in enumerate(peptides_for_rt_grid):
        single_aa_fragment_pairs = peptide.get('single_aa_fragment_pairs', {})
        for pos, pairs_set in single_aa_fragment_pairs.items():
            if not (1 <= pos <= protein_length):
                continue
            filtered = [p for p in pairs_set if filter_fragment_pair_by_ion_type(p, ion_type_filter)]
            if not filtered:
                continue
            c_pairs = sorted(set(normalize_fragment_pair(p) for p in filtered if normalize_fragment_pair(p).startswith('c')))
            z_pairs = sorted(set(normalize_fragment_pair(p) for p in filtered if normalize_fragment_pair(p).startswith('z')))
            # When only one type (or filter is c/z only), single line in one color
            if ion_type_filter or (not c_pairs) or (not z_pairs):
                parts = c_pairs or z_pairs
                text = ",".join(parts)
                if len(text) > 14:
                    text = text[:12] + "…"
                color = _c_color if c_pairs else _z_color
                ax_grid.text(pos - 0.5, i + 0.5, text, fontsize=4, ha='center', va='center',
                             color=color, fontweight='bold')
            else:
                # Both c and z in same cell: two lines so overlap and balance are visible at a glance
                lines, colors = [], []
                if c_pairs:
                    tc = ",".join(c_pairs) if len(c_pairs) <= 2 else ",".join(c_pairs[:2]) + "…"
                    if len(tc) > 12:
                        tc = tc[:10] + "…"
                    lines.append(tc)
                    colors.append(_c_color)
                if z_pairs:
                    tz = ",".join(z_pairs) if len(z_pairs) <= 2 else ",".join(z_pairs[:2]) + "…"
                    if len(tz) > 12:
                        tz = tz[:10] + "…"
                    lines.append(tz)
                    colors.append(_z_color)
                dy = 0.2 if len(lines) == 2 else 0
                for k, (line, col) in enumerate(zip(lines, colors)):
                    y_off = dy * (0.5 - k)  # first line slightly above center, second below
                    ax_grid.text(pos - 0.5, i + 0.5 + y_off, line, fontsize=3.5, ha='center', va='center',
                                 color=col, fontweight='bold')
    
    # No multi-AA overhang markers on RT grid (only single-AA overhangs are shown)
    
    # Set limits - will be synchronized with signal plot via sharey
    ax_grid.set_ylim(0, num_peptides)
    ax_grid.set_xlim(0, protein_length)
    
    # Set x-axis to show all positions with amino acid letters (centered on columns)
    # Show all amino acid letters, but only show position numbers every 5th position
    # Always use two-line format to keep letters aligned: "number\nletter" or "\nletter"
    x_ticks = [i + 0.5 for i in range(protein_length)]
    x_labels = []
    for i in range(protein_length):
        pos_1idx = i + 1
        if pos_1idx <= len(protein_sequence):
            aa = protein_sequence[i]
            # Show position number only for multiples of 5, but always put letter on second line
            if pos_1idx % 5 == 0:
                x_labels.append(f"{pos_1idx}\n{aa}")  # Number on first line, letter on second
            else:
                x_labels.append(f"\n{aa}")  # Empty first line, letter on second line (aligned with numbered positions)
        else:
            # For positions beyond sequence, show number only if multiple of 5
            if pos_1idx % 5 == 0:
                x_labels.append(str(pos_1idx))
            else:
                x_labels.append("\n")  # Empty two-line format for alignment
    
    ax_grid.set_xticks(x_ticks)
    ax_grid.set_xticklabels(x_labels, fontsize=16, rotation=0)
    ax_grid.tick_params(axis='x', labelsize=16)
    ax_grid.set_xlabel('Protein Position (1-indexed) with Amino Acid', fontsize=18, fontweight='bold')
    
    # Legend for fragment colors so c/z balance is clear at a glance
    ax_grid.legend(
        handles=[
            mpatches.Patch(facecolor=_c_color, edgecolor='gray', label='c ions'),
            mpatches.Patch(facecolor=_z_color, edgecolor='gray', label='z ions'),
        ],
        loc='upper right', fontsize=6,
    )
    
    # Set y-axis: sequence, charge, mods, mz, MS1 RT, and c/z fragment pairs (like peptide sequence)
    y_tick_positions = [i + 0.5 for i in range(num_peptides)]
    y_labels = []
    for i in range(num_peptides):
        p = peptides_for_rt_grid[i]
        raw_mods = p.get('modifications', '-') or '-'
        mods_display = format_mods_for_display(raw_mods)
        mz = p.get('mz')
        mz_str = f"{mz:.4f}" if mz is not None and not (isinstance(mz, float) and (mz != mz or mz <= 0)) else "-"
        base = f"{p['peptide_seq']} | +{p.get('charge', 0)} | {mods_display} | m/z {mz_str} | MS1 RT {p['rt']:.2f}s"
        c_list, z_list = _peptide_c_z_fragment_lists(p.get('single_aa_fragment_pairs', {}), ion_type_filter)
        frag_parts = []
        if c_list:
            frag_parts.append("c: " + ",".join(c_list))
        if z_list:
            frag_parts.append("z: " + ",".join(z_list))
        if frag_parts:
            base += " | " + " | ".join(frag_parts)
        y_labels.append(base)
    
    ax_grid.set_ylabel('Peptide (sequence | charge | mods | m/z | MS1 RT | c/z fragments)', fontsize=10, fontweight='bold')
    
    # Prepare colorbar: show CV range for single-AA overhangs
    sm = plt.cm.ScalarMappable(cmap=custom_cmap, norm=norm)
    sm.set_array([])
    
    # ===== PLOT 5: Signal Intensity and Fill Time Histogram (right side) =====
    # IMPORTANT: peptides list is already sorted by RT (line 1258)
    # Get signal intensities and fill times for each peptide in the EXACT same order as grid
    # Grid row i corresponds to peptides_for_rt_grid[i], so signal_intensities[i] and fill_times[i] must match peptides_for_rt_grid[i]
    signal_intensities = [p.get('signal_intensity', 0.0) for p in peptides_for_rt_grid]
    fill_times = [p.get('fill_time', 0.0) for p in peptides_for_rt_grid]
    
    # Print fill time statistics for filtered peptides
    valid_fill_times = [ft for ft in fill_times if ft > 0]
    if valid_fill_times:
        print(f"\n{'='*60}")
        print(f"Fill Time Statistics - Filtered Peptides ({len(peptides_for_rt_grid)} peptides):")
        print(f"  Valid fill times: {len(valid_fill_times)} peptides")
        print(f"  Range: {min(valid_fill_times):.2f} - {max(valid_fill_times):.2f} ms")
        print(f"  Min: {min(valid_fill_times):.2f} ms")
        print(f"  Max: {max(valid_fill_times):.2f} ms")
        print(f"  Mean: {sum(valid_fill_times)/len(valid_fill_times):.2f} ms")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print(f"Fill Time Statistics - Filtered Peptides ({len(peptides_for_rt_grid)} peptides):")
        print(f"  WARNING: No valid fill times found in filtered peptides")
        print(f"{'='*60}")
    
    # Verify peptide order matches between grid and signal plot
    # Grid uses peptides[i] for row i, signal uses signal_intensities[i] and fill_times[i] for bar at y=i
    # They must be in the same order!
    if len(signal_intensities) != len(peptides_for_rt_grid) or len(fill_times) != len(peptides_for_rt_grid):
        print(f"ERROR: Mismatch! {len(signal_intensities)} intensities, {len(fill_times)} fill times, but {len(peptides_for_rt_grid)} peptides")
    else:
        print(f"Verified: {len(peptides_for_rt_grid)} peptides (with single-AA), {len(signal_intensities)} intensities, {len(fill_times)} fill times - order should match")
    
    # Match the grid plot's y-axis setup exactly:
    # Grid uses: y = np.arange(num_peptides + 1) for pcolormesh edges
    # This creates cells where cell i spans from y=i to y=i+1
    # The y-axis limits are set to (0, num_peptides)
    # y_tick_positions are at [i + 0.5 for i in range(num_peptides)] (centers of cells)
    
    # Set y-axis limits on grid (signal will automatically sync via sharey)
    exact_ylim = (0, num_peptides)
    ax_grid.set_ylim(exact_ylim)
    # Explicitly lock signal y-limits to grid (prevents drift after layout operations)
    ax_signal.set_ylim(ax_grid.get_ylim())
    
    # Create horizontal bar chart with two bars per peptide (intensity and fill time)
    # Use twin x-axis so intensity and fill time can have independent scales
    # - Bottom x-axis: Signal intensity (left to right)
    # - Top x-axis: Fill time (right to left, reversed)
    
    # Position bars: intensity bars at y=i+0.25 (lower half), fill time bars at y=i+0.75 (upper half)
    # Each bar has height=0.4 so they fit within the cell (y=i to y=i+1) with 0.1 spacing
    y_positions_intensity = [i + 0.25 for i in range(num_peptides)]  # Lower half of each cell
    y_positions_fill_time = [i + 0.75 for i in range(num_peptides)]  # Upper half of each cell
    
    # Create bars for intensity on bottom x-axis (black)
    bars_intensity = ax_signal.barh(y_positions_intensity, signal_intensities, height=0.4, align='center',
                                   color='black', alpha=0.7, edgecolor='black', linewidth=0.3, label='Signal Intensity')
    
    # Create twin x-axis for fill time (top axis, reversed direction)
    ax_signal_fill_time = ax_signal.twiny()  # Create second x-axis at top
    ax_signal_fill_time.set_ylim(ax_signal.get_ylim())  # Share y-axis with intensity plot
    ax_signal_fill_time.tick_params(axis='y', left=False, labelleft=False)  # Hide y-axis on fill time plot
    
    # Create bars for fill time on top x-axis (cobalt blue)
    # Fill time bars extend from right to left (reversed direction)
    bars_fill_time = ax_signal_fill_time.barh(y_positions_fill_time, fill_times, height=0.4, align='center',
                                              color='#0047AB', alpha=0.7, edgecolor='black', linewidth=0.3, label='Fill Time (ms)')
    
    # Reverse the fill time x-axis so it reads from right to left
    ax_signal_fill_time.invert_xaxis()
    
    # Set labels
    ax_signal.set_xlabel('Signal Intensity', fontsize=10, fontweight='bold')
    ax_signal_fill_time.set_xlabel('Fill Time (ms)', fontsize=10, fontweight='bold')
    
    # Alignment verified: 
    # - Grid row i (y from i to i+1) displays peptides[i]
    # - Signal bars at y=i+0.25 and y=i+0.75 display signal_intensities[i] and fill_times[i] for the same peptides[i]
    # - Both use the same peptides list (sorted by RT), so peptide identities match
    
    # Add legend (on the intensity axis)
    ax_signal.legend(loc='upper right', fontsize=8, framealpha=0.9)
    ax_signal.set_ylabel('')  # No y-label (shared with grid via sharey)
    # y-axis ticks/labels are handled by sharey - hide on signal side
    ax_signal.tick_params(axis='y', left=False, labelleft=False)  # Hide y-ticks and labels on right side
    ax_signal.grid(axis='x', alpha=0.3, linestyle='--')
    ax_signal.tick_params(axis='x', labelsize=8)
    
    # Set y-axis labels and ticks on grid (signal will share via sharey)
    from matplotlib.ticker import FixedLocator, FixedFormatter
    ax_grid.set_ylim(exact_ylim)  # Set limits on grid
    # Explicitly lock signal y-limits to grid (prevents drift after layout operations)
    ax_signal.set_ylim(ax_grid.get_ylim())
    ax_grid.set_yticks(y_tick_positions)
    ax_grid.set_yticklabels(y_labels, fontsize=6)
    ax_grid.tick_params(axis='y', left=True, labelleft=True)  # Show ticks/labels on left (grid)
    ax_signal.tick_params(axis='y', left=False, labelleft=False)  # Hide ticks/labels on right (signal)
    
    # CRITICAL: Re-lock y-limits after setting ticks/labels
    ax_signal.set_ylim(ax_grid.get_ylim())
    
    # Adjust subplot spacing - add more padding on all sides
    # Increased margins: left for colorbar, right for spacing, top/bottom for titles
    # Bottom margin for title box (will be positioned in figure coordinates)
    plt.subplots_adjust(bottom=0.10, top=0.92, hspace=0.14, left=0.18, right=0.86)
    
    # CRITICAL: Re-lock y-limits after subplots_adjust (layout changes may affect axes)
    ax_signal.set_ylim(ax_grid.get_ylim())
    
    # Add title below the plot
    total_peptides_after_filter = len(peptides)
    num_peptides_with_single_aa = len([p for p in peptides if p.get('single_aa_positions', {})])
    num_peptides_before = total_peptides_after_filter
    ion_type_label = ""
    if ion_type_filter == 'c':
        ion_type_label = " (C-IONS ONLY)"
    elif ion_type_filter == 'z':
        ion_type_label = " (Z-IONS ONLY)"
    
    title_lines = []
    header_text = 'Peptide Coverage Grid - Unique Fragment Pairs per Residue'
    if ion_type_label:
        header_text += ion_type_label
    title_lines.append(header_text)
    title_lines.append(f'{protein_id} | Length: {protein_length} residues')
    title_lines.append('')
    title_lines.append(f'Filtering: ≥{MIN_SCANS} scans, ≥{_min_single_aa_overhangs} single-AA overhangs')
    title_lines.append(f'Total peptides: {total_peptides_after_filter}')
    title_lines.append(f'  • Single-AA: {num_peptides_with_single_aa} | Multi-AA only: {total_peptides_after_filter - num_peptides_with_single_aa}')
    title_lines.append(f'RT grid: {num_peptides} peptides with ≥{_min_single_aa_overhangs} single-AA overhangs')
    title_lines.append(unique_counts_str)
    
    # Join lines and add to figure in a box
    title_text = '\n'.join(title_lines)
    
    # Calculate box dimensions - adjust for larger font size
    # Estimate: ~0.005 figure units per line for fontsize=18
    num_lines = len(title_lines)
    box_height = max(0.04, num_lines * 0.005)  # Adjust height for larger font
    box_width = 0.50  # Keep width the same
    
    # Position box at bottom center (in figure coordinates)
    box_x = 0.5 - box_width / 2
    box_y = 0.005  # Very small margin from bottom
    
    # Create fancy box with rounded corners and minimal padding
    fancy_box = FancyBboxPatch(
        (box_x, box_y), box_width, box_height,
        boxstyle="round,pad=0.003",
        transform=fig.transFigure,
        facecolor='white',
        edgecolor='black',
        linewidth=1.0,
        zorder=10,
        alpha=0.95
    )
    fig.patches.append(fancy_box)
    
    # Add text inside the box (centered both horizontally and vertically)
    # Calculate vertical center of box
    text_y = box_y + box_height / 2
    
    fig.text(0.5, text_y, title_text, 
             fontsize=18, fontweight='bold', ha='center', va='center',
             transform=fig.transFigure, zorder=11)
    
    # CRITICAL: Re-lock y-limits after title (title positioning may affect layout)
    ax_signal.set_ylim(ax_grid.get_ylim())
    
    # Add colorbar for coverage heatmap - vertical on the left of y-axis, above quality score colorbar
    if all_overhangs and 'heatmap_im' in locals():
        heatmap_bbox = ax_heatmap.get_position()
        # Position heatmap colorbar above the quality score colorbar
        # Use the heatmap's vertical position and height
        heatmap_cbar_ax = fig.add_axes([0.03, heatmap_bbox.y0, 0.02, heatmap_bbox.height])
        # Create ScalarMappable for the heatmap with actual data range
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
        heatmap_norm = Normalize(vmin=0, vmax=heatmap_max_coverage)
        heatmap_sm = ScalarMappable(cmap=heatmap_im.get_cmap(), norm=heatmap_norm)
        heatmap_sm.set_array([])
        heatmap_cbar = fig.colorbar(heatmap_sm, cax=heatmap_cbar_ax, orientation='vertical')
        heatmap_cbar.set_label('Number of Unique Fragment Pairs\n(incl ion type, ordinal, charge, and parent peptide)', fontsize=10, fontweight='bold', labelpad=10)
        # Add explicit tick labels showing the scale
        max_val = int(heatmap_max_coverage)
        if max_val > 0:
            # Create ticks at 0, and then evenly spaced up to max
            n_ticks = 6
            ticks = [0] + [int(max_val * i / (n_ticks - 1)) for i in range(1, n_ticks)]
            # Remove duplicates
            ticks = sorted(list(set(ticks)))
            heatmap_cbar.set_ticks(ticks)
            heatmap_cbar.set_ticklabels([f'{t}' for t in ticks])
        heatmap_cbar.ax.tick_params(labelsize=8)
    
    # Add colorbar for overhang types - vertical on the left of y-axis
    # Position it in the extra space we added to the figure width
    # Get grid plot bbox to position colorbar relative to it
    # Note: Colorbar will be created later as a single horizontal colorbar between histogram and RT grid
    
    # Note: Colorbar for heatmap only (scatter plot removed)
    # CRITICAL: Re-lock y-limits (colorbar may affect grid bbox)
    # This is essential to maintain alignment
    ax_signal.set_ylim(ax_grid.get_ylim())
    
    # Explicitly lock y-limits one final time after all layout adjustments (prevents drift)
    ax_signal.set_ylim(ax_grid.get_ylim())
    
    # CRITICAL: Final y-limit lock right before saving (prevents any drift from bbox_inches='tight')
    ax_signal.set_ylim(ax_grid.get_ylim())
    
    # Save figure (don't use tight_layout with colorbar, use manual spacing)
    plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0.25)
    print(f"Combined visualization saved to: {output_file}")
    plt.close()
    
    # RT integration windows: not generated here. Use regenerate_rt_windows_from_csv (or run_visualization --rt-windows)
    # to produce the 8 filtered_rt_windows* plots. Skipping avoids duplicate _rt_windows*.png from combined runs.
    if ion_type_filter is None:
        rt_windows_base = os.path.basename(output_file)
        # Remove "combined_overhang_visualization" prefix from derived plot filenames (scatter, CSV)
        if rt_windows_base.startswith("combined_overhang_visualization_"):
            rt_windows_base = rt_windows_base[len("combined_overhang_visualization_"):]
        elif rt_windows_base.startswith("combined_overhang_visualization"):
            rt_windows_base = rt_windows_base[len("combined_overhang_visualization"):].lstrip("_") or "visualization"
        # Scatter: total_area (y) vs sequence position (x); one point per single-AA overhang, same peptide → same y
        scatter_filename = rt_windows_base.replace('.png', '_total_area_vs_position.png') if rt_windows_base.endswith('.png') else rt_windows_base + '_total_area_vs_position.png'
        scatter_output_file = os.path.join(results_dir, scatter_filename)
        create_total_area_vs_position_scatter_plot(peptides_for_rt_grid, scatter_output_file, protein_id, protein_sequence=protein_sequence)
        
        # Write CSV with only the filtered peptides (for use with extract_ms1_chromatograms.py)
        output_basename_no_ext = os.path.splitext(os.path.basename(output_file))[0]
        is_already_filtered = (output_basename_no_ext == 'filtered' or output_basename_no_ext.endswith('_filtered'))
        if is_already_filtered and peptides_for_rt_grid:
            filtered_csv_path = os.path.join(results_dir, 'comet_frags_filtered_by_qvalue_pepscore_mods_minaa.csv')
            if write_filtered_peptides_csv(csv_file, peptides_for_rt_grid, filtered_csv_path):
                print(f"\nWrote filtered peptides CSV ({len(peptides_for_rt_grid)} peptides): {os.path.abspath(filtered_csv_path)}")
                print("  To run chromatogram extraction on these peptides: python extract_ms1_chromatograms.py <comet_frags_filtered_by_qvalue_pepscore_mods_minaa.csv> <mzML> <output.png>")
    
    if ion_type_filter is None:
        # Automatically generate filtered version with default thresholds
        # Default thresholds: Top N=42, Kc=3, Kz=3, max_per_residue=42
        # Only generate if not already a filtered version (prevent recursion / filtered_filtered duplicates)
        output_basename_no_ext = os.path.splitext(os.path.basename(output_file))[0]
        is_already_filtered = (output_basename_no_ext == 'filtered' or output_basename_no_ext.endswith('_filtered'))
        if not is_already_filtered and '_filtered' not in output_file:
            print(f"\n{'='*60}")
            print(f"Generating filtered visualization with default thresholds...")
            print(f"  Default thresholds: Top N=42, Kc=3, Kz=3, max_per_residue=42")
            print(f"{'='*60}")
            
            # Create output filename for filtered version
            # Replace "_no_filtering" with "_filtered" if present, otherwise append "_filtered"
            if '_no_filtering' in output_file:
                filtered_output_file = output_file.replace('_no_filtering', '_filtered')
            elif output_file.endswith('.png'):
                filtered_output_file = output_file.replace('.png', '_filtered.png')
            else:
                filtered_output_file = output_file + '_filtered.png'
            
            # Generate filtered visualization with default thresholds
            # Use two-pass algorithm with default parameters
            create_combined_visualization(
                csv_file, protein_id, fasta_file, filtered_output_file,
                ion_type_filter=None,  # Combined only
                no_filtering=False,     # Apply filtering (opposite of current call)
                use_two_pass=True,      # Use two-pass algorithm
                no_progressive=False,   # Allow progressive selection
                no_two_pass=False,      # Enable two-pass
                mzml_file=mzml_file,
                test=test,
            )
    else:
        print(f"DEBUG: Skipping threshold scatter plot generation. Reason: ion_type_filter={ion_type_filter} (only combined plot generates scatter)")


def create_position_sorted_visualization(peptides_for_rt_grid, peptides, all_single_aa_for_coverage, multi_aa_overhangs,
                                        protein_sequence, protein_id, protein_length, output_file,
                                        ion_type_filter, no_filtering, results_dir, all_overhangs, all_overhangs_for_coverage,
                                        position_to_single_aa_pairs, single_aa_fragment_pair_counts, single_aa_unique_peptide_counts,
                                        multi_aa_unique_peptide_counts, all_peptides_with_single_aa, tau,
                                        min_scans_combined=1, min_single_aa_overhangs=2):
    """
    Create a position-sorted version of the combined visualization.
    This recreates the entire figure with peptides sorted by sequence position, then length.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, ListedColormap
    from matplotlib.patches import FancyBboxPatch
    
    num_peptides = len(peptides_for_rt_grid)
    
    # Create figure with three subplots (same as main: heatmap, single-AA histogram, RT grid — no scatter)
    fig_width = max(20, protein_length * 0.15) + 2.5 + 2.0
    fig_height = max(12, num_peptides * 0.15) + 8 + 2.0
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    gs = fig.add_gridspec(3, 2, height_ratios=[1.5, 0.8, 6.9], width_ratios=[10, 1.5], hspace=0.12, wspace=0.05)
    ax_heatmap = fig.add_subplot(gs[0, 0])
    ax_hist_single = fig.add_subplot(gs[1, 0], sharex=ax_heatmap)
    ax_grid = fig.add_subplot(gs[2, 0], sharex=ax_heatmap)
    ax_signal = fig.add_subplot(gs[2, 1], sharey=ax_grid)
    
    # Recreate heatmap (same as main visualization - doesn't depend on peptide ordering)
    if all_overhangs:
        unique_lengths = sorted(set(o['length'] for o in all_overhangs_for_coverage))
        unique_lengths_for_yaxis = unique_lengths
        coverage = np.zeros((len(unique_lengths), protein_length), dtype=int)
        length_to_idx = {length: idx for idx, length in enumerate(unique_lengths)}
        yaxis_length_to_idx = {length: idx for idx, length in enumerate(unique_lengths_for_yaxis)}
        
        for overhang in all_overhangs_for_coverage:
            start = overhang['start']
            end = overhang['end']
            length = overhang['length']
            
            if start < 1 or end > protein_length:
                continue
            
            if ion_type_filter is not None:
                fragment_pairs = overhang.get('fragment_pairs', [])
                if fragment_pairs:
                    has_matching_pair = any(filter_fragment_pair_by_ion_type(p, ion_type_filter) for p in fragment_pairs)
                    if not has_matching_pair:
                        continue
                else:
                    continue
            
            start_idx = start - 1
            end_idx = end - 1
            if length in length_to_idx:
                length_idx = length_to_idx[length]
                for pos in range(start_idx, end_idx + 1):
                    coverage[length_idx, pos] += 1
        
        max_coverage = np.max(coverage) if np.max(coverage) > 0 else 1
        n_colors = 1024
        colors_list = [(0.0, 0.0, 1.0)]  # Blue for 0
        for i in range(1, n_colors):
            t = (i - 1) / (n_colors - 2)
            r = g = b = 1.0 - t
            colors_list.append((r, g, b))
        
        custom_cmap = ListedColormap(colors_list)
        coverage_normalized = coverage.copy().astype(float)
        if max_coverage > 1:
            for i in range(coverage.shape[0]):
                for j in range(coverage.shape[1]):
                    val = coverage[i, j]
                    if val == 0:
                        coverage_normalized[i, j] = 0
                    elif val == 1:
                        coverage_normalized[i, j] = 1
                    else:
                        normalized_val = 1 + ((val - 1) / (max_coverage - 1)) * 1022
                        coverage_normalized[i, j] = normalized_val
        elif max_coverage == 1:
            for i in range(coverage.shape[0]):
                for j in range(coverage.shape[1]):
                    val = coverage[i, j]
                    coverage_normalized[i, j] = 0 if val == 0 else 1
        
        coverage_full = np.zeros((len(unique_lengths_for_yaxis), protein_length), dtype=int)
        for overhang in all_overhangs_for_coverage:
            start = overhang['start']
            end = overhang['end']
            length = overhang['length']
            
            if start < 1 or end > protein_length:
                continue
            
            if length not in yaxis_length_to_idx:
                continue
            
            if ion_type_filter is not None:
                fragment_pairs = overhang.get('fragment_pairs', [])
                if fragment_pairs:
                    has_matching_pair = any(filter_fragment_pair_by_ion_type(p, ion_type_filter) for p in fragment_pairs)
                    if not has_matching_pair:
                        continue
                else:
                    continue
            
            start_idx = start - 1
            end_idx = end - 1
            yaxis_idx = yaxis_length_to_idx[length]
            for pos in range(start_idx, end_idx + 1):
                coverage_full[yaxis_idx, pos] += 1
        
        max_coverage_full = np.max(coverage_full) if np.max(coverage_full) > 0 else 1
        coverage_normalized_full = coverage_full.copy().astype(float)
        if max_coverage_full > 1:
            for i in range(coverage_full.shape[0]):
                for j in range(coverage_full.shape[1]):
                    val = coverage_full[i, j]
                    if val == 0:
                        coverage_normalized_full[i, j] = 0
                    elif val == 1:
                        coverage_normalized_full[i, j] = 1
                    else:
                        normalized_val = 1 + ((val - 1) / (max_coverage_full - 1)) * 1022
                        coverage_normalized_full[i, j] = normalized_val
        elif max_coverage_full == 1:
            for i in range(coverage_full.shape[0]):
                for j in range(coverage_full.shape[1]):
                    val = coverage_full[i, j]
                    coverage_normalized_full[i, j] = 0 if val == 0 else 1
        
        im = ax_heatmap.imshow(coverage_normalized_full, aspect='auto', cmap=custom_cmap, 
                              interpolation='nearest', origin='lower', vmin=0, vmax=1023)
        
        ax_heatmap.set_yticks(range(len(unique_lengths_for_yaxis)))
        ax_heatmap.set_yticklabels([str(l) for l in unique_lengths_for_yaxis], fontsize=8)
        ax_heatmap.set_ylim(-0.5, len(unique_lengths_for_yaxis) - 0.5)
        ax_heatmap.set_ylabel('Overhang Length\n(residues)', fontsize=10, fontweight='bold')
        ion_type_label = ""
        if ion_type_filter == 'c':
            ion_type_label = " (c ions only)"
        elif ion_type_filter == 'z':
            ion_type_label = " (z/z+1 ions only)"
        ax_heatmap.set_title(f'Overhang Coverage Heatmap (Single-AA + Multi-AA){ion_type_label}', 
                           fontsize=11, fontweight='bold', pad=10)
        ax_heatmap.set_xticks([])
        ax_heatmap.set_xlabel('Protein Position (1-indexed) with Amino Acid', fontsize=18, fontweight='bold')
        ax_heatmap.set_xlim(0, protein_length)
        heatmap_im = im
        heatmap_max_coverage = max(max_coverage, max_coverage_full)
    
    # Recreate single-AA histogram (same data; scatter plot removed)
    position_cvs = {}
    histogram_all_cvs = []
    
    for peptide in peptides_for_rt_grid:
        single_aa_positions = peptide.get('single_aa_positions', {})
        single_aa_fragment_pairs = peptide.get('single_aa_fragment_pairs', {})
        peptide_seq = peptide.get('peptide_seq', '')
        clean_peptide = re.sub(r'\[.*?\]', '', peptide_seq) if peptide_seq else ''
        
        for pos in single_aa_positions.keys():
            if 1 <= pos <= protein_length:
                if pos not in position_cvs:
                    position_cvs[pos] = []
                pairs_at_pos = single_aa_fragment_pairs.get(pos, set())
                for pair_str in pairs_at_pos:
                    normalized_pair = normalize_fragment_pair(pair_str)
                    match = re.match(r'^([a-z])(\d+)-([a-z])(\d+)$', normalized_pair.lower())
                    if match:
                        series1, num1_str, series2, num2_str = match.groups()
                        num1, num2 = int(num1_str), int(num2_str)
                        fragment_ordinal = max(num1, num2)
                    else:
                        fragment_ordinal = None
                    
                    overhang_dict = {
                        'peptide_seq': peptide_seq,
                        'start': pos,
                        'end': pos,
                        'length': 1,
                        'fragment_ordinal': fragment_ordinal
                    }
                    cv = calculate_geometric_quality_metric(overhang_dict, tau=tau)
                    if cv is not None and not np.isnan(cv) and np.isfinite(cv):
                        position_cvs[pos].append(cv)
                        histogram_all_cvs.append(cv)
    
    cdict_hist = {
        'red': [(0.0, 1.0, 1.0), (0.1, 0.0, 0.0), (0.15, 0.0, 0.0), (0.5, 1.0, 1.0), (1.0, 1.0, 1.0)],
        'green': [(0.0, 1.0, 1.0), (0.1, 0.0, 0.0), (0.15, 0.0, 1.0), (0.5, 1.0, 1.0), (0.83, 0.0, 0.0), (1.0, 0.0, 0.0)],
        'blue': [(0.0, 1.0, 1.0), (0.1, 0.0, 0.0), (0.15, 1.0, 1.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0)]
    }
    hist_cmap = LinearSegmentedColormap('hist_qvalue', cdict_hist, N=256)
    hist_norm = plt.Normalize(vmin=0.0, vmax=1.0)
    
    bar_width = 0.4
    x_positions = list(range(protein_length))
    x1 = [x - bar_width/2 for x in x_positions]
    x2 = [x + bar_width/2 for x in x_positions]
    
    bottom_fragment_pairs = np.zeros(protein_length)
    
    if histogram_all_cvs:
        min_cv = min(histogram_all_cvs)
        max_cv = max(histogram_all_cvs)
        cv_range = max_cv - min_cv if max_cv > min_cv else 1.0
        
        for pos in range(1, protein_length + 1):
            pos_idx = pos - 1
            if pos in position_cvs:
                cvs = sorted(position_cvs[pos])
                current_bottom = 0
                for cv_val in cvs:
                    normalized_cv = (cv_val - min_cv) / cv_range if cv_range > 0 else 0.5
                    grid_value = 0.15 + normalized_cv * 0.85
                    color = hist_cmap(hist_norm(grid_value))
                    ax_hist_single.bar(pos_idx - bar_width/2, 1, width=bar_width, 
                                      bottom=current_bottom, color=color, 
                                      edgecolor='black', linewidth=0.3, align='edge')
                    current_bottom += 1
                bottom_fragment_pairs[pos_idx] = len(cvs)
    else:
        bars3 = ax_hist_single.bar(x1, single_aa_fragment_pair_counts, width=bar_width, 
                            edgecolor='black', linewidth=0.5, color='steelblue', alpha=0.7, 
                            align='edge', label='Unique Fragment Pairs')
    
    bars4 = ax_hist_single.bar(x2, single_aa_unique_peptide_counts, width=bar_width, 
                        edgecolor='black', linewidth=0.5, color='black', alpha=0.7, 
                        align='edge', label='Unique Peptides')
    
    ax_hist_single.set_ylabel('Count', fontsize=10, fontweight='bold')
    ax_hist_single.set_title('Single-AA Overhangs: Unique Fragment Pairs (colored by CV) and Unique Peptides per Residue', 
                     fontsize=11, fontweight='bold', pad=10)
    ax_hist_single.legend(loc='upper right', fontsize=8)
    ax_hist_single.grid(axis='y', alpha=0.3, linestyle='--')
    ax_hist_single.set_xticks([])
    ax_hist_single.set_xlim(0, protein_length)
    ax_hist_single.tick_params(axis='y', labelsize=12)
    
    total_single_aa_unique_peptides = len(all_peptides_with_single_aa)
    peptides_with_single_aa_only_set = set()
    for p in peptides_for_rt_grid:
        single_aa_fragment_pairs_dict = p.get('single_aa_fragment_pairs', {})
        has_single_aa_pairs = False
        for pos, pairs_set in single_aa_fragment_pairs_dict.items():
            if pairs_set:
                has_single_aa_pairs = True
                break
        if has_single_aa_pairs:
            peptide_key = (p.get('peptide_seq', ''), p.get('rt', 0.0), p.get('charge', 0))
            if peptide_key[0]:
                peptides_with_single_aa_only_set.add(peptide_key)
    num_peptides_with_single_aa = len(peptides_with_single_aa_only_set)
    total_single_aa_unique_pairs = sum(len(position_to_single_aa_pairs.get(pos, set())) for pos in range(1, protein_length + 1))
    global_unique_peptide_pairs = len(set().union(*[position_to_single_aa_pairs.get(pos, set()) for pos in range(1, protein_length + 1)]))
    
    stats_text_single = f'Total unique peptides: {total_single_aa_unique_peptides}\nRT grid shows: {num_peptides} peptides\nSingle-AA only: {num_peptides_with_single_aa}\nTotal unique (peptide, pos, pair): {total_single_aa_unique_pairs}\nGlobally unique (peptide, fragment pair): {global_unique_peptide_pairs}'
    ax_hist_single.text(0.02, 0.98, stats_text_single, transform=ax_hist_single.transAxes,
                        fontsize=10, fontweight='bold', verticalalignment='top',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black', linewidth=1.5, alpha=0.9),
                        zorder=100)
    
    # Recreate RT grid with position-sorted peptides
    grid = np.zeros((num_peptides, protein_length), dtype=float)
    cv_grid = np.full((num_peptides, protein_length), np.nan, dtype=float)
    
    all_cvs = []
    for i, peptide in enumerate(peptides_for_rt_grid):
        single_aa_positions = peptide.get('single_aa_positions', {})
        single_aa_fragment_pairs = peptide.get('single_aa_fragment_pairs', {})
        peptide_start = peptide.get('start', 0)
        peptide_end = peptide.get('end', 0)
        peptide_seq = peptide.get('peptide_seq', '')
        clean_peptide = re.sub(r'\[.*?\]', '', peptide_seq) if peptide_seq else ''
        
        if (peptide_start == 0 or peptide_end == 0) and clean_peptide:
            protein_seq_upper = protein_sequence.upper()
            clean_peptide_upper = clean_peptide.upper()
            peptide_start_in_protein = protein_seq_upper.find(clean_peptide_upper)
            if peptide_start_in_protein >= 0:
                peptide_start = peptide_start_in_protein + 1
                peptide_end = peptide_start + len(clean_peptide) - 1
        
        if peptide_start > 0 and peptide_end > 0:
            filtered_single_aa_positions = {}
            filtered_single_aa_fragment_pairs = {}
            for pos, count in single_aa_positions.items():
                if peptide_start <= pos <= peptide_end:
                    filtered_single_aa_positions[pos] = count
                    if pos in single_aa_fragment_pairs:
                        filtered_single_aa_fragment_pairs[pos] = single_aa_fragment_pairs[pos]
            single_aa_positions = filtered_single_aa_positions
            single_aa_fragment_pairs = filtered_single_aa_fragment_pairs
        
        if peptide_start > 0 and peptide_end > 0:
            for pos in range(peptide_start, peptide_end + 1):
                if 1 <= pos <= protein_length:
                    grid[i, pos - 1] = 0.1
        
        for pos in single_aa_positions.keys():
            if 1 <= pos <= protein_length:
                fragment_pairs = single_aa_fragment_pairs.get(pos, set())
                if fragment_pairs:
                    cv_values = []
                    for pair_str in fragment_pairs:
                        normalized_pair = normalize_fragment_pair(pair_str)
                        match = re.match(r'^([a-z])(\d+)-([a-z])(\d+)$', normalized_pair.lower())
                        if match:
                            series1, num1_str, series2, num2_str = match.groups()
                            num1, num2 = int(num1_str), int(num2_str)
                            fragment_ordinal = max(num1, num2)
                        else:
                            fragment_ordinal = None
                        
                        overhang_dict = {
                            'peptide_seq': peptide_seq,
                            'start': pos,
                            'end': pos,
                            'length': 1,
                            'fragment_ordinal': fragment_ordinal
                        }
                        cv = calculate_geometric_quality_metric(overhang_dict, tau=tau)
                        if cv is not None and not np.isnan(cv) and np.isfinite(cv):
                            cv_values.append(cv)
                    
                    if cv_values:
                        avg_cv = np.mean(cv_values)
                        cv_grid[i, pos - 1] = avg_cv
                        all_cvs.append(avg_cv)
    
    if all_cvs:
        min_cv = min(all_cvs)
        max_cv = max(all_cvs)
        cv_range = max_cv - min_cv if max_cv > min_cv else 1.0
        for i in range(num_peptides):
            for pos in range(protein_length):
                if not np.isnan(cv_grid[i, pos]):
                    normalized_cv = (cv_grid[i, pos] - min_cv) / cv_range if cv_range > 0 else 0.5
                    grid[i, pos] = 0.15 + normalized_cv * 0.85
    else:
        for i, peptide in enumerate(peptides_for_rt_grid):
            single_aa_positions = peptide.get('single_aa_positions', {})
            for pos in single_aa_positions.keys():
                if 1 <= pos <= protein_length:
                    grid[i, pos - 1] = 1.0
    
    cdict = {
        'red': [(0.0, 1.0, 1.0), (0.1, 0.0, 0.0), (0.15, 0.0, 0.0), (0.5, 1.0, 1.0), (1.0, 1.0, 1.0)],
        'green': [(0.0, 1.0, 1.0), (0.1, 0.0, 0.0), (0.15, 0.0, 1.0), (0.5, 1.0, 1.0), (0.83, 0.0, 0.0), (1.0, 0.0, 0.0)],
        'blue': [(0.0, 1.0, 1.0), (0.1, 0.0, 0.0), (0.15, 1.0, 1.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0)]
    }
    custom_cmap = LinearSegmentedColormap('white_black_qvalue', cdict, N=256)
    norm = plt.Normalize(vmin=0.0, vmax=1.0)
    
    x = np.arange(protein_length + 1)
    y = np.arange(num_peptides + 1)
    X, Y = np.meshgrid(x, y)
    
    im_grid = ax_grid.pcolormesh(X, Y, grid, cmap=custom_cmap, norm=norm,
                                 edgecolors='black', linewidths=0.3, shading='flat')
    
    # Fragment pair labels: color by ion type (c=cyan, z=amber) so overlap and c/z balance visible at a glance
    _c_color, _z_color = '#22d3ee', '#fbbf24'
    for i, peptide in enumerate(peptides_for_rt_grid):
        single_aa_fragment_pairs = peptide.get('single_aa_fragment_pairs', {})
        for pos, pairs_set in single_aa_fragment_pairs.items():
            if not (1 <= pos <= protein_length):
                continue
            filtered = [p for p in pairs_set if filter_fragment_pair_by_ion_type(p, ion_type_filter)]
            if not filtered:
                continue
            c_pairs = sorted(set(normalize_fragment_pair(p) for p in filtered if normalize_fragment_pair(p).startswith('c')))
            z_pairs = sorted(set(normalize_fragment_pair(p) for p in filtered if normalize_fragment_pair(p).startswith('z')))
            if ion_type_filter or (not c_pairs) or (not z_pairs):
                parts = c_pairs or z_pairs
                text = ",".join(parts)
                if len(text) > 14:
                    text = text[:12] + "…"
                color = _c_color if c_pairs else _z_color
                ax_grid.text(pos - 0.5, i + 0.5, text, fontsize=4, ha='center', va='center',
                             color=color, fontweight='bold')
            else:
                lines, colors = [], []
                if c_pairs:
                    tc = ",".join(c_pairs) if len(c_pairs) <= 2 else ",".join(c_pairs[:2]) + "…"
                    if len(tc) > 12:
                        tc = tc[:10] + "…"
                    lines.append(tc)
                    colors.append(_c_color)
                if z_pairs:
                    tz = ",".join(z_pairs) if len(z_pairs) <= 2 else ",".join(z_pairs[:2]) + "…"
                    if len(tz) > 12:
                        tz = tz[:10] + "…"
                    lines.append(tz)
                    colors.append(_z_color)
                dy = 0.2 if len(lines) == 2 else 0
                for k, (line, col) in enumerate(zip(lines, colors)):
                    y_off = dy * (0.5 - k)
                    ax_grid.text(pos - 0.5, i + 0.5 + y_off, line, fontsize=3.5, ha='center', va='center',
                                 color=col, fontweight='bold')
    
    ax_grid.set_ylim(0, num_peptides)
    ax_grid.set_xlim(0, protein_length)
    
    x_ticks = [i + 0.5 for i in range(protein_length)]
    x_labels = []
    for i in range(protein_length):
        pos_1idx = i + 1
        if pos_1idx <= len(protein_sequence):
            aa = protein_sequence[i]
            if pos_1idx % 5 == 0:
                x_labels.append(f"{pos_1idx}\n{aa}")
            else:
                x_labels.append(f"\n{aa}")
        else:
            if pos_1idx % 5 == 0:
                x_labels.append(str(pos_1idx))
            else:
                x_labels.append("\n")
    
    ax_grid.set_xticks(x_ticks)
    ax_grid.set_xticklabels(x_labels, fontsize=16, rotation=0)
    ax_grid.tick_params(axis='x', labelsize=16)
    ax_grid.set_xlabel('Protein Position (1-indexed) with Amino Acid', fontsize=18, fontweight='bold')
    
    # Legend for fragment colors (c/z balance at a glance)
    ax_grid.legend(
        handles=[
            mpatches.Patch(facecolor=_c_color, edgecolor='gray', label='c ions'),
            mpatches.Patch(facecolor=_z_color, edgecolor='gray', label='z ions'),
        ],
        loc='upper right', fontsize=6,
    )
    
    y_tick_positions = [i + 0.5 for i in range(num_peptides)]
    y_labels = []
    for i in range(num_peptides):
        p = peptides_for_rt_grid[i]
        raw_mods = p.get('modifications', '-') or '-'
        mods_display = format_mods_for_display(raw_mods)
        mz = p.get('mz')
        mz_str = f"{mz:.4f}" if mz is not None and not (isinstance(mz, float) and (mz != mz or mz <= 0)) else "-"
        base = f"{p['peptide_seq']} | +{p.get('charge', 0)} | {mods_display} | m/z {mz_str} | MS1 RT {p['rt']:.2f}s"
        c_list, z_list = _peptide_c_z_fragment_lists(p.get('single_aa_fragment_pairs', {}), ion_type_filter)
        frag_parts = []
        if c_list:
            frag_parts.append("c: " + ",".join(c_list))
        if z_list:
            frag_parts.append("z: " + ",".join(z_list))
        if frag_parts:
            base += " | " + " | ".join(frag_parts)
        y_labels.append(base)
    
    ax_grid.set_ylabel('Peptide (sequence | charge | mods | m/z | MS1 RT | c/z fragments)', fontsize=10, fontweight='bold')
    ax_grid.set_ylim(0, num_peptides)
    ax_grid.set_yticks(y_tick_positions)
    ax_grid.set_yticklabels(y_labels, fontsize=6)
    
    # Signal intensity plot
    signal_intensities = [p.get('signal_intensity', 0.0) for p in peptides_for_rt_grid]
    fill_times = [p.get('fill_time', 0.0) for p in peptides_for_rt_grid]
    
    ax_signal.set_ylim(ax_grid.get_ylim())
    y_positions_intensity = [i + 0.25 for i in range(num_peptides)]
    y_positions_fill_time = [i + 0.75 for i in range(num_peptides)]
    
    bars_intensity = ax_signal.barh(y_positions_intensity, signal_intensities, height=0.4, align='center',
                                   color='black', alpha=0.7, edgecolor='black', linewidth=0.3, label='Signal Intensity')
    
    ax_signal_fill_time = ax_signal.twiny()
    ax_signal_fill_time.set_ylim(ax_signal.get_ylim())
    ax_signal_fill_time.tick_params(axis='y', left=False, labelleft=False)
    
    bars_fill_time = ax_signal_fill_time.barh(y_positions_fill_time, fill_times, height=0.4, align='center',
                                              color='#0047AB', alpha=0.7, edgecolor='black', linewidth=0.3, label='Fill Time (ms)')
    
    ax_signal_fill_time.invert_xaxis()
    ax_signal.set_xlabel('Signal Intensity', fontsize=10, fontweight='bold')
    ax_signal_fill_time.set_xlabel('Fill Time (ms)', fontsize=10, fontweight='bold')
    ax_signal.legend(loc='upper right', fontsize=8, framealpha=0.9)
    ax_signal.set_ylabel('')
    ax_signal.tick_params(axis='y', left=False, labelleft=False)
    ax_signal.grid(axis='x', alpha=0.3, linestyle='--')
    ax_signal.tick_params(axis='x', labelsize=8)
    
    # Add colorbar for heatmap
    if all_overhangs and 'heatmap_im' in locals():
        heatmap_bbox = ax_heatmap.get_position()
        heatmap_cbar_ax = fig.add_axes([0.03, heatmap_bbox.y0, 0.02, heatmap_bbox.height])
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
        heatmap_norm = Normalize(vmin=0, vmax=heatmap_max_coverage)
        heatmap_sm = ScalarMappable(cmap=heatmap_im.get_cmap(), norm=heatmap_norm)
        heatmap_sm.set_array([])
        heatmap_cbar = fig.colorbar(heatmap_sm, cax=heatmap_cbar_ax, orientation='vertical')
        heatmap_cbar.set_label('Number of Unique Fragment Pairs\n(incl ion type, ordinal, charge, and parent peptide)', fontsize=10, fontweight='bold', labelpad=10)
        max_val = int(heatmap_max_coverage)
        if max_val > 0:
            n_ticks = 6
            ticks = [0] + [int(max_val * i / (n_ticks - 1)) for i in range(1, n_ticks)]
            ticks = sorted(list(set(ticks)))
            heatmap_cbar.set_ticks(ticks)
            heatmap_cbar.set_ticklabels([f'{t}' for t in ticks])
        heatmap_cbar.ax.tick_params(labelsize=8)
    
    plt.subplots_adjust(bottom=0.10, top=0.92, hspace=0.14, left=0.18, right=0.86)
    ax_signal.set_ylim(ax_grid.get_ylim())
    
    # Title
    total_peptides_after_filter = len(peptides)
    num_peptides_with_single_aa = len([p for p in peptides if p.get('single_aa_positions', {})])
    title_lines = []
    title_lines.append('Peptide Coverage Grid - Unique Fragment Pairs per Residue (sorted by position, then length)')
    title_lines.append(f'{protein_id} | Length: {protein_length} residues')
    title_lines.append('')
    title_lines.append(f'Filtering: ≥{min_scans_combined} scans, ≥{min_single_aa_overhangs} single-AA overhangs')
    title_lines.append(f'Total peptides: {total_peptides_after_filter}')
    title_lines.append(f'  • Single-AA: {num_peptides_with_single_aa} | Multi-AA only: {total_peptides_after_filter - num_peptides_with_single_aa}')
    title_lines.append(f'RT grid: {num_peptides} peptides with ≥{min_single_aa_overhangs} single-AA overhangs (sorted by position, then length)')
    title_lines.append(unique_counts_str)
    
    title_text = '\n'.join(title_lines)
    num_lines = len(title_lines)
    box_height = max(0.04, num_lines * 0.005)
    box_width = 0.50
    box_x = 0.5 - box_width / 2
    box_y = 0.005
    
    fancy_box = FancyBboxPatch(
        (box_x, box_y), box_width, box_height,
        boxstyle="round,pad=0.003",
        transform=fig.transFigure,
        facecolor='white',
        edgecolor='black',
        linewidth=1.0,
        zorder=10,
        alpha=0.95
    )
    fig.patches.append(fancy_box)
    
    text_y = box_y + box_height / 2
    fig.text(0.5, text_y, title_text, 
             fontsize=18, fontweight='bold', ha='center', va='center',
             transform=fig.transFigure, zorder=11)
    
    ax_signal.set_ylim(ax_grid.get_ylim())
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0.25)
    print(f"Position-sorted visualization saved to: {output_file}")
    plt.close()


def create_fragment_pair_grid_from_peptides(peptides, single_aa_overhangs, protein_sequence, protein_id, protein_length, output_file, ion_type_filter=None):
    """
    Create fragment pair grid visualization with:
    - X-axis: length of the longer fragment in the pair
    - Y-axis: fragment pair (each pair gets a row)
    - Color: q-value using pink-orange-cyan-green colormap
    
    Args:
        peptides: List of peptide dictionaries with 'quality_score' (q-value), 'peptide_seq', 'single_aa_fragment_pairs'
        single_aa_overhangs: List of single-AA overhang dictionaries
        protein_sequence: Full protein sequence
        protein_id: Protein identifier
        protein_length: Protein length
        output_file: Output PNG file path
        ion_type_filter: 'c' for c ions only, 'z' for z/z+1 ions only, None for all
    """
    from matplotlib.colors import LinearSegmentedColormap, Normalize
    from matplotlib.cm import ScalarMappable
    
    # Helper function to extract fragment numbers from a pair string
    def get_fragment_lengths(pair_str):
        """Extract fragment numbers from a pair like 'c4-c5' or 'z8-z9'. Returns (frag1_num, frag2_num)."""
        normalized = normalize_fragment_pair(pair_str)
        # Pattern: c4-c5, z8-z9, etc.
        match = re.match(r'^([a-z])(\d+)-([a-z])(\d+)$', normalized.lower())
        if match:
            series1, num1_str, series2, num2_str = match.groups()
            num1, num2 = int(num1_str), int(num2_str)
            return (num1, num2)
        return (None, None)
    
    # Create a map from peptide sequence to q-value for quick lookup
    peptide_to_qvalue = {}
    for peptide in peptides:
        peptide_seq = peptide.get('peptide_seq', '')
        clean_peptide = re.sub(r'\[.*?\]', '', peptide_seq) if peptide_seq else ''
        qvalue = peptide.get('quality_score', 1.0)
        if clean_peptide:
            # Store best (lowest) q-value for each peptide sequence
            if clean_peptide not in peptide_to_qvalue or qvalue < peptide_to_qvalue[clean_peptide]:
                peptide_to_qvalue[clean_peptide] = qvalue
    
    # Collect data organized by position: pos -> {pair: {max_length: best_qvalue}}
    position_to_pair_data = defaultdict(lambda: defaultdict(dict))  # pos -> pair -> {max_length: best_qvalue}
    
    # Process single-AA overhangs and match with q-values from peptides
    for overhang in single_aa_overhangs:
        pos = overhang.get('start')
        if not pos or pos < 1 or pos > protein_length:
            continue
        
        fragment_pairs = overhang.get('fragment_pairs', [])
        if not fragment_pairs:
            continue
        
        # Get q-value for this peptide
        peptide_seq = overhang.get('peptide_seq', '')
        clean_peptide = re.sub(r'\[.*?\]', '', peptide_seq) if peptide_seq else ''
        if not clean_peptide:
            continue
        
        # Get q-value from peptide map
        qvalue = peptide_to_qvalue.get(clean_peptide, 1.0)
        
        # Process each fragment pair
        for pair_str in fragment_pairs:
            if not pair_str or not pair_str.strip():
                continue
            
            # Filter by ion type if specified
            if not filter_fragment_pair_by_ion_type(pair_str, ion_type_filter):
                continue
            
            normalized_pair = normalize_fragment_pair(pair_str)
            
            # Extract fragment lengths
            frag1_num, frag2_num = get_fragment_lengths(normalized_pair)
            if frag1_num is None or frag2_num is None:
                continue
            
            # Get the longer fragment length
            max_length = max(frag1_num, frag2_num)
            
            # Store best (lowest) q-value for this pair at this fragment length at this position
            if max_length not in position_to_pair_data[pos][normalized_pair] or qvalue < position_to_pair_data[pos][normalized_pair][max_length]:
                position_to_pair_data[pos][normalized_pair][max_length] = qvalue
    
    if not position_to_pair_data:
        print("Warning: No fragment pairs found for grid visualization")
        return
    
    # Find global maximums across all positions
    all_unique_pairs = set()
    max_fragment_length_global = 0
    max_num_pairs_global = 0
    
    for pos in position_to_pair_data:
        pairs_at_pos = position_to_pair_data[pos]
        all_unique_pairs.update(pairs_at_pos.keys())
        max_num_pairs_global = max(max_num_pairs_global, len(pairs_at_pos))
        for pair_data in pairs_at_pos.values():
            if pair_data:
                max_fragment_length_global = max(max_fragment_length_global, max(pair_data.keys()))
    
    if max_fragment_length_global == 0:
        print("Warning: No valid fragment lengths found")
        return
    
    # Calculate geometric quality metric for each pair and sort by it
    # The metric uses fragment ordinal (longer fragment in pair)
    # For single-AA overhangs: sigma_res = tau / sqrt(fragment_ordinal), normalized by fragment_ordinal
    # Lower metric = better quality, so we sort ascending
    pair_to_quality = {}
    tau = 0.1732  # Default value for resolution heterogeneity scale
    
    for pair in all_unique_pairs:
        # Extract fragment ordinal from pair (use max length)
        frag1_num, frag2_num = get_fragment_lengths(pair)
        if frag1_num is not None and frag2_num is not None:
            fragment_ordinal = max(frag1_num, frag2_num)
            
            # Calculate geometric quality metric for single-AA overhang
            # sigma_res = tau / sqrt(fragment_ordinal)
            # normalization = fragment_ordinal
            # relative_uncertainty = sigma_res / normalization = tau / (fragment_ordinal^1.5)
            sigma_res = tau / np.sqrt(fragment_ordinal)
            normalization = fragment_ordinal
            relative_uncertainty = sigma_res / normalization
            quality_metric = relative_uncertainty * 100.0
            quality_metric = min(quality_metric, 100.0)
            
            pair_to_quality[pair] = quality_metric
        else:
            # If we can't extract fragment numbers, assign worst quality
            pair_to_quality[pair] = 100.0
    
    # Sort pairs by geometric quality metric (lower is better, so ascending order)
    unique_pairs_global = sorted(all_unique_pairs, key=lambda p: pair_to_quality.get(p, 100.0))
    num_pairs_global = len(unique_pairs_global)
    
    print(f"Fragment pairs sorted by geometric quality metric (lower = better)")
    print(f"  Best quality pair: {unique_pairs_global[0]} (metric={pair_to_quality.get(unique_pairs_global[0], 100.0):.4f})")
    print(f"  Worst quality pair: {unique_pairs_global[-1]} (metric={pair_to_quality.get(unique_pairs_global[-1], 100.0):.4f})")
    
    # Collect all q-values for normalization
    all_qvalues = []
    for pos in position_to_pair_data:
        for pair_data in position_to_pair_data[pos].values():
            for qval in pair_data.values():
                if qval is not None and not np.isnan(qval) and np.isfinite(qval):
                    all_qvalues.append(qval)
    
    if all_qvalues:
        min_q = min(all_qvalues)
        max_q = max(all_qvalues)
        print(f"Q-value range: min={min_q:.6f}, max={max_q:.6f}")
        print(f"Global max fragment length: {max_fragment_length_global}")
        print(f"Global max number of pairs: {max_num_pairs_global}")
        if min_q == max_q:
            min_q = max(0.0, min_q - 0.01)
            max_q = min(1.0, max_q + 0.01)
    else:
        min_q = 0.0
        max_q = 1.0
    
    # Create colormap: pink -> orange -> cyan -> green
    cdict = {
        'red': [
            (0.0, 1.0, 1.0),    # Pink at 0.0
            (0.33, 1.0, 1.0),   # Orange at 0.33
            (0.67, 0.0, 0.0),   # Cyan at 0.67
            (1.0, 0.0, 0.0),    # Green at 1.0
        ],
        'green': [
            (0.0, 0.75, 0.75),  # Pink at 0.0
            (0.33, 0.65, 0.65), # Orange at 0.33
            (0.67, 1.0, 1.0),   # Cyan at 0.67
            (1.0, 0.8, 0.8),     # Green at 1.0
        ],
        'blue': [
            (0.0, 0.8, 0.8),     # Pink at 0.0
            (0.33, 0.0, 0.0),    # Orange at 0.33
            (0.67, 1.0, 1.0),   # Cyan at 0.67
            (1.0, 0.0, 0.0),     # Green at 1.0
        ]
    }
    custom_cmap = LinearSegmentedColormap('pink_orange_cyan_green', cdict, N=256)
    
    # Normalization for q-values
    class QValueNormalize(Normalize):
        def __init__(self, q_min, q_max):
            super().__init__(vmin=q_min, vmax=q_max)
            self.q_min = q_min
            self.q_max = q_max
            self.q_range = q_max - q_min if q_max > q_min else 1.0
        
        def __call__(self, value, clip=None):
            if isinstance(value, np.ndarray):
                result = np.zeros_like(value, dtype=float)
                mask = np.isnan(value)
                result[mask] = 0.0
                data_mask = ~mask
                if np.any(data_mask):
                    data_values = value[data_mask]
                    if self.q_range > 0:
                        normalized_q = (data_values - self.q_min) / self.q_range
                    else:
                        normalized_q = np.full_like(data_values, 0.5)
                    result[data_mask] = np.clip(normalized_q, 0.0, 1.0)
                return result
            else:
                if np.isnan(value):
                    return 0.0
                if self.q_range > 0:
                    normalized_q = (value - self.q_min) / self.q_range
                else:
                    normalized_q = 0.5
                return np.clip(normalized_q, 0.0, 1.0)
    
    custom_norm = QValueNormalize(q_min=min_q, q_max=max_q)
    
    # Create figure with subplots for each position
    import matplotlib.gridspec as gridspec
    n_cols = 10
    n_rows = int(np.ceil(protein_length / n_cols))
    max_width = 50
    max_height = 50
    fig_width = min(n_cols * 2.0, max_width)
    fig_height = min(n_rows * 2.5, max_height)
    
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.4, wspace=0.3)
    
    # Create grids for each position with fixed axes
    NO_COVERAGE_VALUE = np.nan
    
    # Create subplot for each position
    for pos in range(1, protein_length + 1):
        row = (pos - 1) // n_cols
        col = (pos - 1) % n_cols
        ax = fig.add_subplot(gs[row, col])
        
        # Get pairs at this position
        pairs_at_pos = position_to_pair_data.get(pos, {})
        
        if not pairs_at_pos:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{pos}:{protein_sequence[pos-1]}', fontsize=8, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
            # Set fixed axes even for empty positions (log scale)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_ylim(0.9, num_pairs_global + 0.1)
            ax.set_xlim(0.9, max_fragment_length_global + 0.1)
            continue
        
        # Create grid with fixed dimensions
        grid = np.full((num_pairs_global, max_fragment_length_global), NO_COVERAGE_VALUE, dtype=float)
        
        # Map pairs to row indices using global pair list
        pair_to_row = {pair: i for i, pair in enumerate(unique_pairs_global)}
        
        # Fill grid for this position
        for pair, length_to_qvalue in pairs_at_pos.items():
            if pair not in pair_to_row:
                continue
            row_idx = pair_to_row[pair]
            for max_length, qvalue in length_to_qvalue.items():
                if 1 <= max_length <= max_fragment_length_global:
                    # Use best (lowest) q-value if multiple entries exist
                    if np.isnan(grid[row_idx, max_length - 1]) or qvalue < grid[row_idx, max_length - 1]:
                        grid[row_idx, max_length - 1] = qvalue
        
        # Set log scale on both axes BEFORE plotting
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        # Plot grid with extent to map integer indices to actual values
        # extent = [left, right, bottom, top] in data coordinates
        # X-axis: fragment lengths 1 to max_fragment_length_global
        # Y-axis: pair indices 1 to num_pairs_global (1-based for log scale)
        im = ax.imshow(grid, aspect='auto', cmap=custom_cmap, norm=custom_norm,
                      interpolation='nearest', origin='lower',
                      extent=[0.5, max_fragment_length_global + 0.5, 0.5, num_pairs_global + 0.5])
        
        # Set fixed axes (same for all positions) - use 1-based for log scale
        # X-axis: fragment lengths from 1 to max_fragment_length_global
        # Y-axis: pair indices from 1 to num_pairs_global (1-based for log scale)
        ax.set_xlim(0.9, max_fragment_length_global + 0.1)
        ax.set_ylim(0.9, num_pairs_global + 0.1)
        
        # Set title
        num_unique_pairs_at_pos = len(pairs_at_pos)
        ax.set_title(f'{pos}:{protein_sequence[pos-1]} ({num_unique_pairs_at_pos} pairs)', fontsize=8, fontweight='bold')
        
        # Set x-axis labels (only on bottom row) - use log scale ticks
        if row == n_rows - 1:
            # Generate log scale ticks for fragment lengths
            x_ticks = []
            x_tick_labels = []
            # Add powers of 2 and other key values
            for val in [1, 2, 5, 10, 20, 50, 100]:
                if val <= max_fragment_length_global:
                    x_ticks.append(val)
                    x_tick_labels.append(str(val))
            # Also add max value if not already included
            if max_fragment_length_global not in x_ticks:
                x_ticks.append(max_fragment_length_global)
                x_tick_labels.append(str(max_fragment_length_global))
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_tick_labels, fontsize=6)
            ax.set_xlabel('Fragment Length (log scale)', fontsize=7)
        else:
            ax.set_xticks([])
        
        # Set y-axis labels (only on leftmost column) - use log scale ticks
        if col == 0:
            # Generate log scale ticks for pair indices (1-based)
            y_ticks = []
            y_tick_labels = []
            # Add powers of 2 and other key values
            for val in [1, 2, 5, 10, 20, 50, 100]:
                if val <= num_pairs_global:
                    y_ticks.append(val)
                    idx = val - 1  # Convert to 0-based index
                    if idx < len(unique_pairs_global):
                        pair = unique_pairs_global[idx]
                        if len(pair) > 12:
                            y_tick_labels.append(pair[:10] + '..')
                        else:
                            y_tick_labels.append(pair)
            # Also add max value if not already included
            if num_pairs_global not in y_ticks:
                y_ticks.append(num_pairs_global)
                idx = num_pairs_global - 1
                if idx < len(unique_pairs_global):
                    pair = unique_pairs_global[idx]
                    if len(pair) > 12:
                        y_tick_labels.append(pair[:10] + '..')
                    else:
                        y_tick_labels.append(pair)
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(y_tick_labels, fontsize=4)
        else:
            ax.set_yticks([])
    
    # Add colorbar for the entire figure (more space so colorbar doesn't overlap labels)
    fig.subplots_adjust(bottom=0.20, left=0.08, right=0.95, top=0.94)
    cbar_ax = fig.add_axes([0.15, 0.06, 0.7, 0.025])  # [left, bottom, width, height]
    sm = ScalarMappable(cmap=custom_cmap, norm=custom_norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Q-value (Percolator)', fontsize=10, fontweight='bold')
    cbar.ax.tick_params(labelsize=8)
    
    # Title
    ion_label = ''
    if ion_type_filter == 'c':
        ion_label = ' (C-IONS ONLY)'
    elif ion_type_filter == 'z':
        ion_label = ' (Z-IONS ONLY)'
    
    fig.suptitle(f'Fragment Pair Grid: Q-values by Fragment Length{ion_label}\n{protein_id} ({protein_length} residues) | Fixed axes: {num_pairs_global} pairs × {max_fragment_length_global} lengths', 
                 fontsize=12, fontweight='bold', y=0.98)
    
    # Save figure
    try:
        max_pixels_per_dim = 8000
        calculated_dpi = min(max_pixels_per_dim / max(fig_width, fig_height), 100)
        dpi = max(int(calculated_dpi), 50)
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight', pad_inches=0.25, facecolor='white')
        print(f"Fragment pair grid saved to: {output_file} (DPI: {dpi})")
    except Exception as e:
        print(f"Error saving figure: {e}")
        try:
            plt.savefig(output_file, dpi=75, bbox_inches='tight', pad_inches=0.25)
            print(f"Figure saved with fallback settings to: {output_file}")
        except Exception as e2:
            print(f"Failed to save figure: {e2}")
            raise
    finally:
        plt.close()
    
    print(f"  - Total unique fragment pairs (global): {num_pairs_global}")
    print(f"  - Fragment length range (global): 1 to {max_fragment_length_global}")


def create_accepted_peptides_protein_grid(peptides, protein_sequence, protein_id, protein_length, output_file, results_dir, channel_only=False):
    """
    Full protein on x-axis, one block per unique (sequence, charge, mods) on y-axis.
    Multiple scans of the same peptide merge into one block; different charges or modifications
    of the same sequence each get their own block (a family can have multiple blocks).
    Overhangs and fragment pairs use significant data when available (5 ppm + >0.5% intensity).
    Each block: c-fragment rows, main sequence row (deep red = single-AA overhang; elsewhere shaded by summed fragment intensity across scans so darkest where many overlapping fragments have high signal), z-fragment rows.
    When fragment_intensity_c/z are present (from CSV matched fragment ion intensities), main-row darkness = normalized summed intensity; otherwise fallback to coverage count. Fragment rows use transparent blue-black and less saturated yellow at single-AA overhang; letters are Times New Roman, bold, large.
    Writes two files: ..._unique_peptides.png (ordered by position), ..._unique_peptides_by_total_area.png (ordered by total_area).
    Each row = one unique peptide (sequence + charge + modifications).
    When channel_only=True (per-channel view): only the main grid and by_total_area/lowres are written; no family or segment breakdown plots.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import re
    from matplotlib.patches import Rectangle
    from matplotlib.colors import LogNorm, ListedColormap, Normalize
    from matplotlib.cm import ScalarMappable

    def _bounds(p):
        start = p.get('start', 0)
        end = p.get('end', 0)
        seq = p.get('peptide_seq') or p.get('plain_peptide') or ''
        clean = re.sub(r'\[.*?\]', '', seq) if seq else ''
        if (start == 0 or end == 0) and clean:
            up = protein_sequence.upper()
            i = up.find(clean.upper())
            if i >= 0:
                start, end = i + 1, i + len(clean)
        if start <= 0 or end <= 0 or not clean:
            return None, None, None
        return start, end, clean

    def _plain_sequence(p):
        """Unmodified sequence for grouping. Same sequence = one family (all charges)."""
        seq = (p.get('peptide_seq') or p.get('plain_peptide') or '').strip()
        return re.sub(r'\[.*?\]', '', seq) if seq else ''

    def _significant_overhang_positions(p, peptide_start):
        """Positions (protein) from significant_single_aa_overhangs* when present; else None."""
        for col, is_protein in [
            ('significant_single_aa_overhangs_protein_positions', True),
            ('significant_single_aa_overhangs', False),
        ]:
            val = p.get(col)
            if val is None or (isinstance(val, float) and np.isnan(val)) or not str(val).strip():
                continue
            pos_set = set()
            for part in str(val).strip().split(','):
                part = part.strip()
                if not part or not part[-1].isalpha() or not part[:-1].replace('.', '').replace('-', '').isdigit():
                    continue
                try:
                    pos = int(float(part[:-1]))
                except (TypeError, ValueError):
                    continue
                if not is_protein and peptide_start and peptide_start > 0:
                    pos = peptide_start + pos - 1  # peptide 1-based -> protein
                pos_set.add(pos)
            if pos_set:
                return pos_set
        return None

    def _parse_significant_fragment_pairs_string(s):
        """Parse 'c3|c4, z2|z3' -> set of pair strings 'c3-c4', 'z2-z3' for _c_z_numbers_from_fragment_pairs_union."""
        if s is None or (isinstance(s, float) and np.isnan(s)) or not str(s).strip():
            return set()
        out = set()
        for part in str(s).strip().split(','):
            part = part.strip().replace('|', '-')
            if '-' in part and part:
                out.add(part)
        return out

    def _overhang_positions_fallback(p):
        """Positions with fragment-based single-AA overhang (red on main row). Fallback when significant not used."""
        pos_set = set()
        for key in ('single_aa_fragment_pairs', 'single_aa_overhang_fragment_pairs'):
            pairs = p.get(key) or {}
            for pos in pairs:
                if pairs[pos]:
                    pos_set.add(int(pos) if isinstance(pos, (int, float)) else pos)
        return pos_set

    def _get_fragment_pairs_dict(p):
        """Fragment pairs from either key (CSV often uses single_aa_overhang_fragment_pairs). Fallback when significant not used."""
        for key in ('single_aa_fragment_pairs', 'single_aa_overhang_fragment_pairs'):
            d = p.get(key)
            if d and isinstance(d, dict):
                return d
        return {}

    def _total_area(p):
        v = p.get('total_area')
        if v is None or (isinstance(v, float) and (np.isnan(v) or v <= 0)):
            return -1.0
        try:
            return float(v)
        except (TypeError, ValueError):
            return -1.0

    def _canon_charge(c):
        try:
            return int(c) if c not in (None, '') else 0
        except (TypeError, ValueError):
            return 0

    def _canon_mods(m):
        if m in (None, ''):
            return '-'
        s = str(m).strip()
        return s if s and s.lower() not in ('nan', '-') else '-'

    # One block per unique peptide (sequence, charge, mods); multiple scans of same peptide merge into one block. Different charge or modifications = separate row.
    # A family (same sequence) can have multiple blocks — each (sequence, charge, mods) gets its own block.
    from collections import defaultdict
    groups_by_peptide = defaultdict(list)
    for p in peptides:
        plain = _plain_sequence(p)
        if not plain:
            continue
        charge = _canon_charge(p.get('charge'))
        mods = _canon_mods(p.get('modifications'))
        key = (plain, charge, mods)
        groups_by_peptide[key].append(p)

    blocks = []
    for _key, group_psms in groups_by_peptide.items():
        rep = group_psms[0]
        start, end, clean = _bounds(rep)
        if start is None or end is None or clean is None:
            continue
        L = len(clean)
        overhang = set()
        # Prefer significant overhang positions (5 ppm + >0.5% intensity)
        for p in group_psms:
            sig_pos = _significant_overhang_positions(p, start)
            if sig_pos is not None:
                overhang.update(sig_pos)
                break
        if not overhang:
            for p in group_psms:
                overhang.update(_overhang_positions_fallback(p))
        # Prefer significant fragment pairs only; fallback to all fragment pairs
        pairs_union = {}
        sig_pairs_merged = set()
        for p in group_psms:
            val = p.get('significant_fragment_pairs')
            sig_pairs_merged.update(_parse_significant_fragment_pairs_string(val))
        if sig_pairs_merged:
            pairs_union = {0: sig_pairs_merged}
        else:
            for p in group_psms:
                d = _get_fragment_pairs_dict(p)
                for pos, pairs_set in d.items():
                    if pos not in pairs_union:
                        pairs_union[pos] = set()
                    vals = pairs_set if isinstance(pairs_set, (set, list)) else [pairs_set]
                    for x in vals:
                        if x is not None and str(x).strip():
                            pairs_union[pos].add(x if isinstance(x, str) else str(x))
        c_list, z_list = _c_z_numbers_from_fragment_pairs_union(pairs_union)
        total_area = max(_total_area(p) for p in group_psms)
        c_intensities = rep.get('fragment_intensity_c') or {}
        z_intensities = rep.get('fragment_intensity_z') or {}
        blocks.append((rep, start, end, L, clean, overhang, c_list, z_list, total_area, c_intensities, z_intensities))

    if blocks:
        print(f"  Unique peptide grid: {len(peptides)} PSMs -> {len(blocks)} blocks (sequence+charge+mods; significant fragment pairs when available)")
    if not blocks:
        print("  No accepted peptides with valid protein boundaries; skipping accepted-peptides protein grid")
        return

    num_peptides = len(blocks)
    c_set_all = set()
    z_set_all = set()
    for b in blocks:
        c_set_all.update(b[6])
        z_set_all.update(b[7])

    # Block heights and starts (block tuple: rep, start, end, L, clean, overhang, c_list, z_list, total_area)
    block_heights = [1 + len(blocks[i][6]) + len(blocks[i][7]) for i in range(len(blocks))]
    block_starts = np.cumsum([0] + block_heights)
    total_subrows = block_starts[-1]
    # Amino acid letter font: much bigger to fill squares, Times New Roman (serif), bold
    AA_FONT_MAIN = 24   # main sequence row
    AA_FONT_FRAG = 20   # c/z fragment rows
    AA_FONT_FAMILY = 'serif'  # Times New Roman when available

    # Peptide families: cluster blocks only by same first AA or same last AA (not by overlap).
    # So R-start vs F-start vs L-start stay in different families; different-length peptides that share first or last residue are in the same family.
    def _union_find_components(n, edges):
        parent = list(range(n))
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        def union(a, b):
            pa, pb = find(a), find(b)
            if pa != pb:
                parent[pa] = pb
        for i, j in edges:
            union(i, j)
        components = {}
        for i in range(n):
            root = find(i)
            if root not in components:
                components[root] = []
            components[root].append(i)
        return list(components.values())
    def _overlap_match_count(block_i, block_j):
        """Return number of protein positions where both peptides have the same amino acid (overlap only)."""
        start_i, end_i = block_i[1], block_i[2]
        start_j, end_j = block_j[1], block_j[2]
        clean_i = block_i[4]
        clean_j = block_j[4]
        overlap_lo = max(start_i, start_j)
        overlap_hi = min(end_i, end_j)
        if overlap_lo > overlap_hi:
            return 0
        matches = 0
        for pos in range(overlap_lo, overlap_hi + 1):
            idx_i = pos - start_i
            idx_j = pos - start_j
            if 0 <= idx_i < len(clean_i) and 0 <= idx_j < len(clean_j) and clean_i[idx_i] == clean_j[idx_j]:
                matches += 1
        return matches

    family_edges = []
    for i in range(num_peptides):
        clean_i = blocks[i][4]
        len_i = len(clean_i)
        for j in range(i + 1, num_peptides):
            clean_j = blocks[j][4]
            len_j = len(clean_j)
            # Same family only if majority of amino acids match (overlap positions, same residue)
            min_len = min(len_i, len_j)
            if min_len == 0:
                continue
            n_match = _overlap_match_count(blocks[i], blocks[j])
            if n_match > min_len // 2:  # majority: more than half of shorter peptide's residues match
                family_edges.append((i, j))
    family_components = _union_find_components(num_peptides, family_edges)
    # Sort families by min(start) so leftmost protein region first; within family sort by (start, -length)
    families = []
    for comp in family_components:
        x_lo = min(blocks[i][1] for i in comp)
        x_hi = max(blocks[i][2] for i in comp)
        comp_sorted = sorted(comp, key=lambda i: (blocks[i][1], -blocks[i][3]))
        families.append((comp_sorted, x_lo, x_hi))
    families.sort(key=lambda f: f[1])  # by x_lo

    # Order by position (start, then -length)
    order_position = sorted(range(num_peptides), key=lambda i: (blocks[i][1], -blocks[i][3]))
    # Order by total_area (descending so large area at top)
    order_area = sorted(range(num_peptides), key=lambda i: (-blocks[i][8], blocks[i][1], -blocks[i][3]))  # total_area at index 8

    valid_areas = [b[8] for b in blocks if b[8] >= 0]
    if valid_areas:
        vmin_ta = max(1e4, min(valid_areas) * 0.99)
        vmax_ta = min(1e11, max(valid_areas) * 1.01)
        if vmax_ta <= vmin_ta:
            vmax_ta = vmin_ta * 10
        norm_ta = LogNorm(vmin=1e4, vmax=1e11)
        spectral = plt.get_cmap('Spectral')
        lavender = np.array([0.90, 0.90, 0.98, 1.0])
        purple = np.array([0.58, 0.44, 0.86, 1.0])
        dark_red = np.array([0.55, 0.0, 0.0, 1.0])
        ta_colors = []
        for i in range(64):
            t = i / 63
            if t < 0.33:
                c = lavender + (purple - lavender) * (t / 0.33)
            elif t < 0.66:
                c = purple + (dark_red - purple) * ((t - 0.33) / 0.33)
            else:
                c = dark_red + (np.array(spectral(0)[:4]) - dark_red) * ((t - 0.66) / 0.34)
            ta_colors.append(c)
        for i in range(192):
            ta_colors.append(spectral(0.85 * (i / 191)))
        cmap_ta = ListedColormap(ta_colors, name='total_area')
    else:
        norm_ta = None
        cmap_ta = None

    # Colors: deep desaturated red for overhang on main row; less saturated yellow for single-AA overhang in fragment rows; transparent blue-black for coverage
    OVERHANG_RED = '#7A1F2B'       # deep blue-red (single-AA overhang on main row)
    OVERHANG_YELLOW = '#C9C68F'   # less saturated yellow (single-AA overhang in c/z fragment rows only)
    BLUE_BLACK = (0.08, 0.10, 0.18)  # (r,g,b) for transparent blue-black; alpha added per layer
    BLUE_BLACK_ALPHA_ONE_LAYER = 0.5   # alpha for one fragment layer (fragment rows)
    def _fragment_coverage_per_position(peptide_start, peptide_end, L, c_list, z_list):
        """For each protein position in [peptide_start, peptide_end], return how many c/z fragments cover it (1-based)."""
        coverage = {}
        for pos in range(peptide_start, peptide_end + 1):
            coverage[pos] = 0
        for n in c_list:
            if 1 <= n <= L:
                for pos in range(peptide_start, peptide_start + n):
                    if peptide_start <= pos <= peptide_end:
                        coverage[pos] = coverage.get(pos, 0) + 1
        for n in z_list:
            if 1 <= n <= L:
                for pos in range(peptide_end - n + 1, peptide_end + 1):
                    if peptide_start <= pos <= peptide_end:
                        coverage[pos] = coverage.get(pos, 0) + 1
        return coverage

    def _fragment_intensity_per_position(peptide_start, peptide_end, L, c_list, z_list, c_intensities, z_intensities):
        """For each protein position in [peptide_start, peptide_end], return summed fragment intensity (across c/z) from c_intensities and z_intensities dicts (n -> intensity)."""
        intensity = {}
        for pos in range(peptide_start, peptide_end + 1):
            intensity[pos] = 0.0
        for n in c_list:
            if 1 <= n <= L:
                add = float(c_intensities.get(n, 0) or 0)
                for pos in range(peptide_start, peptide_start + n):
                    if peptide_start <= pos <= peptide_end:
                        intensity[pos] = intensity.get(pos, 0.0) + add
        for n in z_list:
            if 1 <= n <= L:
                add = float(z_intensities.get(n, 0) or 0)
                for pos in range(peptide_end - n + 1, peptide_end + 1):
                    if peptide_start <= pos <= peptide_end:
                        intensity[pos] = intensity.get(pos, 0.0) + add
        return intensity

    def _face_for_main_row(protein_pos, peptide_start, peptide_end, overhang_positions, c_set, z_set, L, coverage, intensity_per_pos=None, max_intensity=None):
        """Main row: overhang = deep red; else if intensity data: darkness by normalized summed intensity; else coverage = blue-black (lighter for 1, darker for 2+)."""
        is_overhang = (
            protein_pos != peptide_start
            and (
                protein_pos in overhang_positions
                or (protein_pos == peptide_end and (1 in z_set or (L - 1) in c_set))
            )
        )
        if is_overhang:
            return OVERHANG_RED
        if intensity_per_pos is not None and max_intensity is not None and max_intensity > 0:
            intensity = intensity_per_pos.get(protein_pos, 0.0) or 0.0
            if intensity <= 0:
                return (*BLUE_BLACK, 0.35)
            alpha = min(1.0, 0.3 + 0.65 * (intensity / max_intensity))
            return (*BLUE_BLACK, alpha)
        cnt = coverage.get(protein_pos, 0)
        if cnt <= 0:
            return (*BLUE_BLACK, 0.4)
        if cnt == 1:
            return (*BLUE_BLACK, 0.35)
        alpha = min(1.0, 0.35 + 0.32 * (cnt - 1))
        return (*BLUE_BLACK, alpha)

    # Cap figure size and DPI so the PNG stays loadable (target max ~4000 px on longest side)
    MAX_FIG_WIDTH_IN = 50
    MAX_FIG_HEIGHT_IN = 70
    MAX_OUTPUT_PX = 4000
    fig_width = min(max(40, protein_length / 3), MAX_FIG_WIDTH_IN)
    fig_height = min(max(24, total_subrows / 2 + 6), MAX_FIG_HEIGHT_IN)
    grid_dpi = min(150, max(72, int(MAX_OUTPUT_PX / max(fig_width, fig_height))))
    LOWRES_DPI = 72

    def draw_one_figure(block_order, out_path, title_suffix, y_range=None, dpi=None):
        use_dpi = dpi if dpi is not None else grid_dpi
        if y_range is not None:
            ymin, ymax = float(y_range[0]), float(y_range[1])
            seg_height = max(6, (ymax - ymin) / total_subrows * fig_height)
            w, h = fig_width, seg_height
        else:
            ymin, ymax = 0, total_subrows
            w, h = fig_width, fig_height
        fig = plt.figure(figsize=(w, h))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlim(-1.2, protein_length + 2.5)
        ax.set_ylim(ymin, ymax)

        for idx in block_order:
            p, peptide_start, peptide_end, L, clean_peptide, overhang_positions, c_list, z_list, area_val, c_intensities, z_intensities = blocks[idx][:11] if len(blocks[idx]) >= 11 else (*blocks[idx], {}, {})
            bstart = block_starts[idx]
            n_c, n_z = len(c_list), len(z_list)
            c_set = set(c_list)
            z_set = set(z_list)
            c_intensities = c_intensities or {}
            z_intensities = z_intensities or {}
            subrow = 0

            # Fragment coverage per position (for main row: blue-black by intensity or coverage count)
            coverage = _fragment_coverage_per_position(peptide_start, peptide_end, L, c_list, z_list)
            intensity_per_pos = _fragment_intensity_per_position(peptide_start, peptide_end, L, c_list, z_list, c_intensities, z_intensities)
            max_intensity = max(intensity_per_pos.values()) if intensity_per_pos else 0.0
            # Only positions that have yellow in fragment rows get red on main row (single-AA overhang = consecutive fragment pair).
            # This keeps red and gold/yellow in sync: no red without corresponding yellow.
            main_row_overhang_pos = set()
            for n in c_set:
                if 1 <= n <= L and (n - 1) in c_set:
                    main_row_overhang_pos.add(peptide_start + n - 1)
            for n in z_set:
                if 1 <= n <= L and (n - 1) in z_set:
                    main_row_overhang_pos.add(peptide_end - n + 1)
            if L and (1 in z_set or (L - 1) in c_set):
                main_row_overhang_pos.add(peptide_end)
            overhang_positions = main_row_overhang_pos

            # Row order (bottom to top): z-rows, main sequence, c-rows. Longest z closest to main, shortest below (ascending n)
            for n in z_list:
                if n < 1 or n > L:
                    continue
                y_row = bstart + subrow
                # z_n covers last n residues: protein peptide_end-n+1 .. peptide_end
                for col in range(protein_length):
                    protein_pos = col + 1
                    if peptide_end - n + 1 <= protein_pos <= peptide_end:
                        is_overhang = (protein_pos == peptide_end - n + 1) and ((n - 1) in z_set)
                        face = OVERHANG_YELLOW if is_overhang else (*BLUE_BLACK, BLUE_BLACK_ALPHA_ONE_LAYER)
                        rect = Rectangle((col, y_row), 1, 1, facecolor=face, edgecolor='gray', linewidth=0.5)
                        ax.add_patch(rect)
                        rel = protein_pos - peptide_start
                        if 0 <= rel < len(clean_peptide):
                            text_color = 'black' if is_overhang else 'white'
                            ax.text(col + 0.5, y_row + 0.5, clean_peptide[rel],
                                    fontsize=AA_FONT_FRAG, ha='center', va='center', color=text_color,
                                    family=AA_FONT_FAMILY, fontweight='bold', zorder=10)
                    else:
                        rect = Rectangle((col, y_row), 1, 1, facecolor='#f0f0f0', edgecolor='gray', linewidth=0.5)
                        ax.add_patch(rect)
                subrow += 1

            # main sequence row: deep red = overhang; else shade by summed fragment intensity (darkest where many overlapping fragments with high signal) or by coverage count
            y_row = bstart + subrow
            y_main_row = y_row
            for col in range(protein_length):
                protein_pos = col + 1
                if peptide_start <= protein_pos <= peptide_end:
                    face = _face_for_main_row(protein_pos, peptide_start, peptide_end, overhang_positions, c_set, z_set, L, coverage, intensity_per_pos=intensity_per_pos, max_intensity=max_intensity)
                    rect = Rectangle((col, y_row), 1, 1, facecolor=face, edgecolor='gray', linewidth=0.5)
                    ax.add_patch(rect)
                    rel = protein_pos - peptide_start
                    if 0 <= rel < len(clean_peptide):
                        ax.text(col + 0.5, y_row + 0.5, clean_peptide[rel],
                                fontsize=AA_FONT_MAIN, ha='center', va='center', color='white',
                                family=AA_FONT_FAMILY, fontweight='bold', zorder=10)
                else:
                    rect = Rectangle((col, y_row), 1, 1, facecolor='#f0f0f0', edgecolor='gray', linewidth=0.5)
                    ax.add_patch(rect)
            subrow += 1

            # Peptide label to the right: sequence text, then a bubble with charge and mods
            charge_val = p.get('charge')
            raw_mods = p.get('modifications', '-') or '-'
            mods_display = format_mods_for_display(raw_mods)
            seq_label = (clean_peptide[:35] + '…') if len(clean_peptide) > 35 else clean_peptide
            try:
                ch = int(float(charge_val)) if charge_val is not None and charge_val != '' else None
            except (TypeError, ValueError):
                ch = None
            charge_mods_parts = []
            if ch is not None:
                charge_mods_parts.append(f'z+{ch}')
            if mods_display and mods_display != '-':
                charge_mods_parts.append(mods_display)
            else:
                charge_mods_parts.append('nomod')
            charge_mods_str = '  '.join(charge_mods_parts)
            bubble_x = protein_length + 0.28
            bubble_w = max(2.0, min(4.5, len(charge_mods_str) * 0.28))
            bubble_h = 0.82
            bubble_y = y_main_row + 0.09
            bubble = FancyBboxPatch((bubble_x, bubble_y), bubble_w, bubble_h, boxstyle='round,pad=0.04,rounding_size=0.15', facecolor=(0.95, 0.95, 0.97), edgecolor=(0.3, 0.3, 0.4), linewidth=0.8, zorder=5)
            ax.add_patch(bubble)
            ax.text(bubble_x + bubble_w / 2, y_main_row + 0.5, charge_mods_str, fontsize=8, ha='center', va='center', fontweight='bold', family=AA_FONT_FAMILY, zorder=10)
            if seq_label:
                ax.text(bubble_x + bubble_w + 0.12, y_main_row + 0.5, seq_label, fontsize=8, ha='left', va='center', fontweight='bold', family=AA_FONT_FAMILY, zorder=10, clip_on=False)

            # c-rows (top of block): c_max closest to main, c_1 farthest above; less saturated yellow = overhang, blue-black = coverage
            for n in reversed(c_list):
                if n < 1 or n > L:
                    continue
                y_row = bstart + subrow
                # c_n covers protein cols peptide_start .. peptide_start+n-1 (1-based)
                for col in range(protein_length):
                    protein_pos = col + 1
                    if peptide_start <= protein_pos < peptide_start + n:
                        is_overhang = (
                            (protein_pos == peptide_start + n - 1 and (n - 1) in c_set)
                            or (protein_pos == peptide_end and n == L - 1)
                        )
                        face = OVERHANG_YELLOW if is_overhang else (*BLUE_BLACK, BLUE_BLACK_ALPHA_ONE_LAYER)
                        rect = Rectangle((col, y_row), 1, 1, facecolor=face, edgecolor='gray', linewidth=0.5)
                        ax.add_patch(rect)
                        if peptide_start <= protein_pos <= peptide_end and (protein_pos - peptide_start) < len(clean_peptide):
                            text_color = 'black' if is_overhang else 'white'
                            ax.text(col + 0.5, y_row + 0.5, clean_peptide[protein_pos - peptide_start],
                                    fontsize=AA_FONT_FRAG, ha='center', va='center', color=text_color,
                                    family=AA_FONT_FAMILY, fontweight='bold', zorder=10)
                    else:
                        rect = Rectangle((col, y_row), 1, 1, facecolor='#f0f0f0', edgecolor='gray', linewidth=0.5)
                        ax.add_patch(rect)
                subrow += 1

            # total_area strip (one color per block, left column) — same cell style as individual
            if norm_ta and cmap_ta and area_val >= 0:
                for r in range(block_heights[idx]):
                    rect_ta = Rectangle((-1, bstart + r), 1, 1, facecolor=cmap_ta(norm_ta(area_val)), edgecolor='gray', linewidth=0.5, zorder=0)
                    ax.add_patch(rect_ta)
            else:
                for r in range(block_heights[idx]):
                    rect_ta = Rectangle((-1, bstart + r), 1, 1, facecolor='#f0f0f0', edgecolor='gray', linewidth=0.5, zorder=0)
                    ax.add_patch(rect_ta)
            # Black outline around this peptide block (main row + c/z fragment rows)
            rect_outline = Rectangle((-1, bstart), protein_length + 1, block_heights[idx], linewidth=1.5, edgecolor='black', facecolor='none', zorder=3)
            ax.add_patch(rect_outline)

        ax.set_ylabel('Unique peptides (one row per sequence+charge+mods; fragments below)', fontsize=10, fontweight='bold', fontfamily='serif')
        ax.set_xlabel('Protein position', fontsize=10, fontweight='bold', fontfamily='serif')
        ax.set_title(f'{protein_id} – Unique peptides and fragments\n(one row per sequence+charge+modifications; deep red = overhang; yellow = single-AA overhang in fragment rows)\n{title_suffix}', fontsize=11, fontweight='bold', fontfamily='serif')
        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(facecolor=OVERHANG_RED, edgecolor='gray', label='Single-AA overhang (main row)'),
            Patch(facecolor=OVERHANG_YELLOW, edgecolor='gray', label='Single-AA overhang (fragment rows)'),
            Patch(facecolor=(*BLUE_BLACK, 0.8), edgecolor='gray', label='Stacked coverage'),
        ], loc='upper right', fontsize=8, prop={'family': 'serif'})
        if norm_ta and cmap_ta:
            fig.subplots_adjust(bottom=0.16, left=0.06, right=0.98, top=0.96)
            pos = ax.get_position()
            cbar_ax = fig.add_axes([pos.x0, 0.04, pos.width, 0.03])
            sm = ScalarMappable(norm=norm_ta, cmap=cmap_ta)
            sm.set_array([])
            cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
            cbar.set_label('total_area (MS1 peak)', fontsize=10, fontweight='bold')
            _set_total_area_colorbar_ticks(cbar)
        plt.savefig(out_path, dpi=use_dpi, bbox_inches='tight', pad_inches=0.25)
        plt.close()
        print(f"  Saved: {out_path} (DPI {use_dpi}, {w:.0f}x{h:.0f} in)")

    def draw_one_figure_segment(block_order, out_path, title_suffix, x_lo, x_hi):
        """Draw the grid restricted to protein positions [x_lo, x_hi] (1-based), only blocks overlapping that range."""
        # Filter to blocks overlapping this segment
        seg_indices = [i for i in block_order if blocks[i][1] <= x_hi and blocks[i][2] >= x_lo]
        if not seg_indices:
            return
        seg_heights = [block_heights[i] for i in seg_indices]
        seg_starts = np.cumsum([0] + seg_heights)
        total_seg_rows = seg_starts[-1]
        seg_width = x_hi - x_lo + 1
        seg_fig_width = min(max(14, seg_width / 2), MAX_FIG_WIDTH_IN)
        seg_fig_height = min(max(12, total_seg_rows / 2 + 4), MAX_FIG_HEIGHT_IN)
        seg_dpi = min(150, max(72, int(MAX_OUTPUT_PX / max(seg_fig_width, seg_fig_height))))
        fig = plt.figure(figsize=(seg_fig_width, seg_fig_height))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlim(x_lo - 1.2, x_hi + 2.5)
        ax.set_ylim(0, total_seg_rows)
        col_lo, col_hi = x_lo - 1, x_hi  # 0-based column range to draw

        for seg_i, idx in enumerate(seg_indices):
            b = blocks[idx]
            p, peptide_start, peptide_end, L, clean_peptide, overhang_positions, c_list, z_list, area_val = b[:9]
            c_intensities = b[9] if len(b) > 9 else {}
            z_intensities = b[10] if len(b) > 10 else {}
            c_intensities = c_intensities or {}
            z_intensities = z_intensities or {}
            bstart = seg_starts[seg_i]
            c_set = set(c_list)
            z_set = set(z_list)
            coverage = _fragment_coverage_per_position(peptide_start, peptide_end, L, c_list, z_list)
            intensity_per_pos = _fragment_intensity_per_position(peptide_start, peptide_end, L, c_list, z_list, c_intensities, z_intensities)
            max_intensity = max(intensity_per_pos.values()) if intensity_per_pos else 0.0
            # Only positions that have yellow in fragment rows get red on main row (same as draw_one_figure)
            main_row_overhang_pos = set()
            for n in c_set:
                if 1 <= n <= L and (n - 1) in c_set:
                    main_row_overhang_pos.add(peptide_start + n - 1)
            for n in z_set:
                if 1 <= n <= L and (n - 1) in z_set:
                    main_row_overhang_pos.add(peptide_end - n + 1)
            if L and (1 in z_set or (L - 1) in c_set):
                main_row_overhang_pos.add(peptide_end)
            overhang_positions = main_row_overhang_pos
            subrow = 0

            # z-rows
            for n in z_list:
                if n < 1 or n > L:
                    continue
                y_row = bstart + subrow
                for col in range(col_lo, col_hi):
                    protein_pos = col + 1
                    if peptide_end - n + 1 <= protein_pos <= peptide_end:
                        is_overhang = (protein_pos == peptide_end - n + 1) and ((n - 1) in z_set)
                        face = OVERHANG_YELLOW if is_overhang else (*BLUE_BLACK, BLUE_BLACK_ALPHA_ONE_LAYER)
                        rect = Rectangle((col, y_row), 1, 1, facecolor=face, edgecolor='gray', linewidth=0.5)
                        ax.add_patch(rect)
                        rel = protein_pos - peptide_start
                        if 0 <= rel < len(clean_peptide):
                            text_color = 'black' if is_overhang else 'white'
                            ax.text(col + 0.5, y_row + 0.5, clean_peptide[rel],
                                    fontsize=AA_FONT_FRAG, ha='center', va='center', color=text_color,
                                    family=AA_FONT_FAMILY, fontweight='bold', zorder=10)
                    else:
                        rect = Rectangle((col, y_row), 1, 1, facecolor='#f0f0f0', edgecolor='gray', linewidth=0.5)
                        ax.add_patch(rect)
                subrow += 1

            y_row = bstart + subrow
            y_main_row = y_row
            for col in range(col_lo, col_hi):
                protein_pos = col + 1
                if peptide_start <= protein_pos <= peptide_end:
                    face = _face_for_main_row(protein_pos, peptide_start, peptide_end, overhang_positions, c_set, z_set, L, coverage, intensity_per_pos=intensity_per_pos, max_intensity=max_intensity)
                    rect = Rectangle((col, y_row), 1, 1, facecolor=face, edgecolor='gray', linewidth=0.5)
                    ax.add_patch(rect)
                    rel = protein_pos - peptide_start
                    if 0 <= rel < len(clean_peptide):
                        ax.text(col + 0.5, y_row + 0.5, clean_peptide[rel],
                                fontsize=AA_FONT_MAIN, ha='center', va='center', color='white',
                                family=AA_FONT_FAMILY, fontweight='bold', zorder=10)
                else:
                    rect = Rectangle((col, y_row), 1, 1, facecolor='#f0f0f0', edgecolor='gray', linewidth=0.5)
                    ax.add_patch(rect)
            subrow += 1

            charge_val = p.get('charge')
            raw_mods = p.get('modifications', '-') or '-'
            mods_display = format_mods_for_display(raw_mods)
            seq_label = (clean_peptide[:35] + '…') if len(clean_peptide) > 35 else clean_peptide
            try:
                ch = int(float(charge_val)) if charge_val is not None and charge_val != '' else None
            except (TypeError, ValueError):
                ch = None
            charge_mods_parts = []
            if ch is not None:
                charge_mods_parts.append(f'z+{ch}')
            if mods_display and mods_display != '-':
                charge_mods_parts.append(mods_display)
            else:
                charge_mods_parts.append('nomod')
            charge_mods_str = '  '.join(charge_mods_parts)
            bubble_x = x_hi + 0.28
            bubble_w = max(2.0, min(4.5, len(charge_mods_str) * 0.28))
            bubble_h = 0.82
            bubble_y = y_main_row + 0.09
            bubble = FancyBboxPatch((bubble_x, bubble_y), bubble_w, bubble_h, boxstyle='round,pad=0.04,rounding_size=0.15', facecolor=(0.95, 0.95, 0.97), edgecolor=(0.3, 0.3, 0.4), linewidth=0.8, zorder=5)
            ax.add_patch(bubble)
            ax.text(bubble_x + bubble_w / 2, y_main_row + 0.5, charge_mods_str, fontsize=8, ha='center', va='center', fontweight='bold', family=AA_FONT_FAMILY, zorder=10)
            if seq_label:
                ax.text(bubble_x + bubble_w + 0.12, y_main_row + 0.5, seq_label, fontsize=8, ha='left', va='center', fontweight='bold', family=AA_FONT_FAMILY, zorder=10, clip_on=False)

            for n in reversed(c_list):
                if n < 1 or n > L:
                    continue
                y_row = bstart + subrow
                for col in range(col_lo, col_hi):
                    protein_pos = col + 1
                    if peptide_start <= protein_pos < peptide_start + n:
                        is_overhang = (
                            (protein_pos == peptide_start + n - 1 and (n - 1) in c_set)
                            or (protein_pos == peptide_end and n == L - 1)
                        )
                        face = OVERHANG_YELLOW if is_overhang else (*BLUE_BLACK, BLUE_BLACK_ALPHA_ONE_LAYER)
                        rect = Rectangle((col, y_row), 1, 1, facecolor=face, edgecolor='gray', linewidth=0.5)
                        ax.add_patch(rect)
                        if peptide_start <= protein_pos <= peptide_end and (protein_pos - peptide_start) < len(clean_peptide):
                            text_color = 'black' if is_overhang else 'white'
                            ax.text(col + 0.5, y_row + 0.5, clean_peptide[protein_pos - peptide_start],
                                    fontsize=AA_FONT_FRAG, ha='center', va='center', color=text_color,
                                    family=AA_FONT_FAMILY, fontweight='bold', zorder=10)
                    else:
                        rect = Rectangle((col, y_row), 1, 1, facecolor='#f0f0f0', edgecolor='gray', linewidth=0.5)
                        ax.add_patch(rect)
                subrow += 1

            if norm_ta and cmap_ta and area_val >= 0:
                for r in range(block_heights[idx]):
                    rect_ta = Rectangle((-1, bstart + r), 1, 1, facecolor=cmap_ta(norm_ta(area_val)), edgecolor='gray', linewidth=0.5, zorder=0)
                    ax.add_patch(rect_ta)
            else:
                for r in range(block_heights[idx]):
                    rect_ta = Rectangle((-1, bstart + r), 1, 1, facecolor='#f0f0f0', edgecolor='gray', linewidth=0.5, zorder=0)
                    ax.add_patch(rect_ta)
            # Black outline around this peptide block (total_area strip + main row + c/z fragment rows)
            seg_w = col_hi - (-1)  # from left column -1 to right of grid col_hi-1
            rect_outline = Rectangle((-1, bstart), seg_w, seg_heights[seg_i], linewidth=1.5, edgecolor='black', facecolor='none', zorder=3)
            ax.add_patch(rect_outline)

        ax.set_ylabel('Accepted peptides (sequence + fragments)', fontsize=10, fontweight='bold', fontfamily='serif')
        ax.set_xlabel('Protein position', fontsize=10, fontweight='bold', fontfamily='serif')
        ax.set_title(f'{protein_id} – Segment positions {x_lo}–{x_hi}\n(deep red = overhang on main row; yellow = single-AA overhang in fragment rows; blue-black = stacked coverage)\n{title_suffix}', fontsize=11, fontweight='bold', fontfamily='serif')
        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(facecolor=OVERHANG_RED, edgecolor='gray', label='Single-AA overhang (main row)'),
            Patch(facecolor=OVERHANG_YELLOW, edgecolor='gray', label='Single-AA overhang (fragment rows)'),
            Patch(facecolor=(*BLUE_BLACK, 0.8), edgecolor='gray', label='Stacked coverage'),
        ], loc='upper right', fontsize=8, prop={'family': 'serif'})
        if norm_ta and cmap_ta:
            fig.subplots_adjust(bottom=0.16, left=0.06, right=0.98, top=0.96)
            pos = ax.get_position()
            cbar_ax = fig.add_axes([pos.x0, 0.04, pos.width, 0.03])
            sm = ScalarMappable(norm=norm_ta, cmap=cmap_ta)
            sm.set_array([])
            cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
            cbar.set_label('total_area (MS1 peak)', fontsize=10, fontweight='bold')
            _set_total_area_colorbar_ticks(cbar)
        plt.savefig(out_path, dpi=seg_dpi, bbox_inches='tight', pad_inches=0.25)
        plt.close()
        print(f"  Saved segment {x_lo}–{x_hi}: {out_path}")

    # Position-ordered (full resolution)
    draw_one_figure(order_position, output_file, 'rows ordered by protein position')
    # Total-area-ordered
    out_by_area = output_file.replace('_unique_peptides.png', '_unique_peptides_by_total_area.png')
    if out_by_area == output_file:
        out_by_area = output_file.replace('.png', '_by_total_area.png')
    draw_one_figure(order_area, out_by_area, 'rows ordered by total area')
    # Low-resolution full overview (position-ordered)
    out_lowres = output_file.replace('_unique_peptides.png', '_unique_peptides_lowres.png')
    if out_lowres == output_file:
        out_lowres = output_file.replace('.png', '_lowres.png')
    draw_one_figure(order_position, out_lowres, 'rows ordered by protein position (overview)', dpi=LOWRES_DPI)

    # Family plots only for combined/total view (not for per-channel)
    if not channel_only:
        base_seg = output_file.rsplit('.', 1)[0]  # strip .png
        for fam_idx, (family_block_indices, x_lo, x_hi) in enumerate(families, start=1):
            n_pep = len(family_block_indices)
            fam_path = f"{base_seg}_family{fam_idx}_pos{x_lo}-{x_hi}.png"
            draw_one_figure_segment(family_block_indices, fam_path, f'family {fam_idx} (pos {x_lo}–{x_hi}, {n_pep} peptides)', x_lo, x_hi)


def create_rt_overlay_only_plot(peptides, output_file, protein_id, total_area_range=None, stage_label=None, peptide_traces=None, rt_max_sec=None):
    """
    Single-panel overlay: RT (x, minutes) vs intensity (y), all peptides' peaks overlaid and colored by total_area.
    Same style as the top panel of the by-channel plot. peptide_traces: optional list of (rts_sec, intensity_1d)
    in same order as peptides; if None or entry None, draw synthetic Gaussian peak from detected_peak window.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import Normalize, LogNorm, ListedColormap
    from matplotlib.cm import ScalarMappable
    from matplotlib.ticker import MultipleLocator, FuncFormatter

    if not peptides:
        return
    PEAK_DRIFT_BUFFER_SEC = 30.0

    def _get_window(p):
        min_rt = p.get('detected_peak_min_rt')
        max_rt = p.get('detected_peak_max_rt')
        if min_rt is None or max_rt is None:
            min_rt = p.get('spectrum_window_min_rt')
            max_rt = p.get('spectrum_window_max_rt')
        rt = p.get('rt', 0.0)
        if min_rt is None or max_rt is None:
            try:
                min_rt = float(rt) - 5.0
                max_rt = float(rt) + 5.0
            except (TypeError, ValueError):
                min_rt = 0.0
                max_rt = 10.0
        try:
            min_rt = float(min_rt)
            max_rt = float(max_rt)
            if np.isnan(min_rt) or np.isnan(max_rt):
                min_rt = float(rt) - 5.0
                max_rt = float(rt) + 5.0
        except (TypeError, ValueError):
            min_rt = float(rt) - 5.0
            max_rt = float(rt) + 5.0
        return (min_rt, max_rt)

    def _get_collection_window(p):
        coll_min = p.get('collection_min_rt')
        coll_max = p.get('collection_max_rt')
        if coll_min is None or coll_max is None:
            return (None, None)
        try:
            coll_min = float(coll_min)
            coll_max = float(coll_max)
            if np.isnan(coll_min) or np.isnan(coll_max) or coll_min >= coll_max:
                return (None, None)
            return (coll_min, coll_max)
        except (TypeError, ValueError):
            return (None, None)

    def _total_area_val(p):
        v = p.get('total_area')
        if v is None or (isinstance(v, float) and (np.isnan(v) or v < 0)):
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    all_areas = [_total_area_val(p) for p in peptides]
    valid_areas = [a for a in all_areas if a is not None]
    if valid_areas:
        # Same norm as channel plot overlay: use total_area_range when provided (linear), else data min/max
        if total_area_range is not None and len(total_area_range) >= 2 and total_area_range[0] is not None and total_area_range[1] is not None:
            vmin, vmax = float(total_area_range[0]), float(total_area_range[1])
            if vmax <= vmin:
                vmax = vmin + 1.0
        else:
            vmin = min(valid_areas)
            vmax = max(valid_areas)
            if vmax <= vmin:
                vmax = vmin + 1.0
        norm = Normalize(vmin=vmin, vmax=vmax)
        spectral = plt.get_cmap('Spectral')
        lavender = np.array([0.90, 0.90, 0.98, 1.0])
        purple = np.array([0.58, 0.44, 0.86, 1.0])
        dark_red = np.array([0.55, 0.0, 0.0, 1.0])
        ta_colors = []
        for i in range(64):
            t = i / 63
            if t < 0.33:
                c = lavender + (purple - lavender) * (t / 0.33)
            elif t < 0.66:
                c = purple + (dark_red - purple) * ((t - 0.33) / 0.33)
            else:
                c = dark_red + (np.array(spectral(0)[:4]) - dark_red) * ((t - 0.66) / 0.34)
            ta_colors.append(c)
        for i in range(192):
            ta_colors.append(spectral(0.85 * (i / 191)))
        cmap = ListedColormap(ta_colors, name='total_area')
    else:
        norm = None
        cmap = None

    max_area = max(valid_areas) if valid_areas else 1.0
    overlay_peptides_sorted = sorted(peptides, key=lambda p: (_total_area_val(p) or 0))
    peptide_to_idx = {id(p): i for i, p in enumerate(peptides)}
    use_traces = (peptide_traces is not None and len(peptide_traces) == len(peptides))
    rt_min_global = float('inf')
    rt_max_global = -float('inf')

    width = max(40, 36)
    height = 16.0
    fig, ax_overlay = plt.subplots(1, 1, figsize=(width, height))
    # Black background for overlay
    fig.patch.set_facecolor('black')
    ax_overlay.set_facecolor('black')
    for spine in ax_overlay.spines.values():
        spine.set_color('0.7')
    ax_overlay.tick_params(axis='both', colors='0.85')
    ax_overlay.xaxis.label.set_color('0.9')
    ax_overlay.yaxis.label.set_color('0.9')
    ax_overlay.grid(axis='x', alpha=0.25, linestyle='--', color='0.6')

    n_overlay = len(overlay_peptides_sorted)
    for zi, p in enumerate(overlay_peptides_sorted):
        min_rt, max_rt = _get_window(p)
        coll_min, coll_max = _get_collection_window(p)
        if coll_min is None or coll_max is None:
            pad = max(10.0, (max_rt - min_rt) * 0.15)
            coll_min = min_rt - pad
            coll_max = max_rt + pad
        x_lo = max(0.0, coll_min - PEAK_DRIFT_BUFFER_SEC)
        x_hi = coll_max + PEAK_DRIFT_BUFFER_SEC
        rt_min_global = min(rt_min_global, x_lo)
        rt_max_global = max(rt_max_global, x_hi)

        area_val = _total_area_val(p)
        color = cmap(norm(area_val)) if (norm and cmap and area_val is not None) else (0.7, 0.7, 0.7, 0.7)
        z = n_overlay - zi

        if use_traces:
            idx = peptide_to_idx.get(id(p))
            if idx is not None and idx < len(peptide_traces) and peptide_traces[idx] is not None:
                rts, intensity = peptide_traces[idx]
                rts = np.asarray(rts, dtype=float)
                intensity = np.asarray(intensity, dtype=float)
                if len(rts) > 0 and len(intensity) == len(rts):
                    mask = (rts >= x_lo) & (rts <= x_hi)
                    if np.any(mask):
                        x_peak = rts[mask]
                        y_peak = intensity[mask].copy()
                        y_max = np.max(y_peak)
                        if y_max > 0 and max_area > 0 and area_val is not None and area_val > 0:
                            y_peak = (y_peak / y_max) * (area_val / max_area)
                        else:
                            y_peak = y_peak / y_max
                        ax_overlay.fill_between(x_peak, 0, y_peak, color=color, alpha=0.25, linewidth=0, zorder=z)
                        ax_overlay.plot(x_peak, y_peak, color=color, alpha=0.8, linewidth=1.5, zorder=z + 0.5)
                        continue

        # Synthetic Gaussian peak (RT in seconds)
        center = (min_rt + max_rt) * 0.5
        sigma = max((max_rt - min_rt) / 4.0, 15.0)
        x = np.linspace(x_lo, x_hi, max(200, int((x_hi - x_lo))))
        amp = (float(area_val) / max_area) if (area_val is not None and max_area > 0) else 0.5
        y = amp * np.exp(-((x - center) ** 2) / (2 * sigma ** 2))
        ax_overlay.fill_between(x, 0, y, color=color, alpha=0.25, linewidth=0, zorder=z)
        ax_overlay.plot(x, y, color=color, alpha=0.8, linewidth=1.5, zorder=z + 0.5)

    if rt_max_global <= rt_min_global:
        rt_min_global = 0
        rt_max_global = 100
    rt_range = rt_max_global - rt_min_global
    x_min = rt_min_global - rt_range * 0.02
    x_max = rt_max_global + max(rt_range * 0.2, 40.0)
    if rt_max_sec is not None and rt_max_sec > 0:
        x_max = max(x_max, rt_max_sec + 60.0)
    ax_overlay.set_xlim(x_min, x_max)
    ax_overlay.set_ylim(0, 1.4)
    ax_overlay.set_ylabel('Intensity (normalized)', fontsize=12, fontweight='bold')
    ax_overlay.xaxis.set_major_locator(MultipleLocator(60))
    ax_overlay.xaxis.set_minor_locator(MultipleLocator(10))
    ax_overlay.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x/60:.0f}'))
    ax_overlay.set_xlabel('Retention Time (min)', fontsize=12, fontweight='bold')
    if stage_label:
        ax_overlay.set_title(stage_label, fontsize=14, fontweight='bold', color='0.9')
    if norm is not None and cmap is not None:
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax_overlay, orientation='horizontal', pad=0.22, shrink=0.7, aspect=35, label='Total area')
        cbar.ax.tick_params(labelsize=9, colors='0.85')
        cbar.ax.xaxis.label.set_color('0.9')
        cbar.ax.set_facecolor('black')
        cbar.ax.patch.set_facecolor('black')
    plt.tight_layout(pad=1.2)
    plt.savefig(output_file, dpi=150, bbox_inches='tight', pad_inches=0.25)
    # Also save a log-scale y-axis version (intensity) so low peaks are visible
    log_file = output_file.rsplit('.', 1)
    log_file = (log_file[0] + '_log.' + log_file[1]) if len(log_file) == 2 else (output_file + '_log.png')
    try:
        ax_overlay.set_yscale('log')
        ax_overlay.set_ylim(1e-5, 2.0)
        ax_overlay.set_ylabel('Intensity (normalized, log scale)', fontsize=12, fontweight='bold')
        plt.savefig(log_file, dpi=150, bbox_inches='tight', pad_inches=0.25)
    except Exception as e:
        import sys
        print(f"  Note: could not save log-scale overlay {log_file}: {e}", file=sys.stderr)
    plt.close()


def create_rt_windows_by_channel_plot(peptides_for_rt_grid, output_file, protein_id, rt_max_sec=None, peptide_traces=None, total_area_range=None, num_channels=1, stage_label=None, n_passed=None, n_excluded=None, use_log_scale=False):
    """
    Create a plot with: (1) top = all-peptide overlay — color-coded chromatogram peaks (RT vs intensity)
    if peptide_traces is provided, else synthetic Gaussian peaks; (2) below = num_channels horizontal strips (one per channel).
    Within each strip, draw that channel's peptides' collection windows (gray) and integration windows (colored by total_area);
    windows within a channel do not overlap. Shared RT x-axis; minutes axis has major ticks every 1 min and minor ticks every 10 s.
    peptide_traces: optional list of (rts, intensity_1d) in same order as peptides_for_rt_grid.
    total_area_range: optional (vmin, vmax) for color scale; when set, colorbar is consistent across plots.
    num_channels: number of channel strips (determined by caller from find_min_channels(peptides that pass filters)).
    stage_label, n_passed, n_excluded: optional; when set, a large "Passed / Excluded" label is drawn on the plot.
    use_log_scale: if True, color scale and overlay y-axis use log scale so shorter/smaller peaks are visible.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import Normalize, LogNorm, ListedColormap
    from matplotlib.cm import ScalarMappable
    from matplotlib.ticker import MaxNLocator, MultipleLocator, FuncFormatter

    num_channels = max(1, int(num_channels))
    channels = [[] for _ in range(num_channels)]
    dropped_no_channel = 0
    for p in peptides_for_rt_grid:
        ch = p.get('channel')
        if ch is not None and 0 <= ch < num_channels:
            channels[ch].append(p)
        else:
            dropped_no_channel += 1
    if dropped_no_channel:
        print(f"  [by-channel plot] {dropped_no_channel} peptide(s) had no valid 'channel' (None or out of range 0..{num_channels - 1}); not drawn in channel strips")

    def _get_window(p):
        min_rt = p.get('detected_peak_min_rt')
        max_rt = p.get('detected_peak_max_rt')
        if min_rt is None or max_rt is None:
            min_rt = p.get('spectrum_window_min_rt')
            max_rt = p.get('spectrum_window_max_rt')
        rt = p.get('rt', 0.0)
        if min_rt is None or max_rt is None:
            try:
                min_rt = float(rt) - 5.0
                max_rt = float(rt) + 5.0
            except (TypeError, ValueError):
                min_rt = 0.0
                max_rt = 10.0
        try:
            min_rt = float(min_rt)
            max_rt = float(max_rt)
            if np.isnan(min_rt) or np.isnan(max_rt):
                min_rt = float(rt) - 5.0
                max_rt = float(rt) + 5.0
        except (TypeError, ValueError):
            min_rt = float(rt) - 5.0
            max_rt = float(rt) + 5.0
        return (min_rt, max_rt)

    def _get_collection_window(p):
        coll_min = p.get('collection_min_rt')
        coll_max = p.get('collection_max_rt')
        if coll_min is None or coll_max is None:
            return (None, None)
        try:
            coll_min = float(coll_min)
            coll_max = float(coll_max)
            if np.isnan(coll_min) or np.isnan(coll_max) or coll_min >= coll_max:
                return (None, None)
            return (coll_min, coll_max)
        except (TypeError, ValueError):
            return (None, None)

    def _total_area_val(p):
        v = p.get('total_area')
        if v is None or (isinstance(v, float) and (np.isnan(v) or v < 0)):
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    def _total_area_sci_str(a):
        if a is None or (isinstance(a, float) and (np.isnan(a) or a < 0)):
            return "-"
        try:
            x = float(a)
            if x == 0:
                return "0"
            return f"{x:.2e}"
        except (TypeError, ValueError):
            return "-"

    all_areas = [_total_area_val(p) for p in peptides_for_rt_grid]
    valid_areas = [a for a in all_areas if a is not None and a > 0]
    if valid_areas:
        if total_area_range is not None and len(total_area_range) >= 2 and total_area_range[0] is not None and total_area_range[1] is not None:
            vmin, vmax = float(total_area_range[0]), float(total_area_range[1])
            if vmax <= vmin:
                vmax = vmin + 1.0
            if use_log_scale and vmin <= 0:
                vmin = max(1e4, min(valid_areas) * 0.5) if valid_areas else 1e4
        else:
            vmin = min(valid_areas)
            vmax = max(valid_areas)
            if vmax <= vmin:
                vmax = vmin + 1.0
        if use_log_scale:
            vmin = max(1e4, vmin * 0.5)
            vmax = min(1e12, vmax * 2.0)
            norm = LogNorm(vmin=vmin, vmax=vmax)
        else:
            norm = Normalize(vmin=vmin, vmax=vmax)
        spectral = plt.get_cmap('Spectral')
        lavender = np.array([0.90, 0.90, 0.98, 1.0])
        purple = np.array([0.58, 0.44, 0.86, 1.0])
        dark_red = np.array([0.55, 0.0, 0.0, 1.0])
        ta_colors = []
        for i in range(64):
            t = i / 63
            if t < 0.33:
                c = lavender + (purple - lavender) * (t / 0.33)
            elif t < 0.66:
                c = purple + (dark_red - purple) * ((t - 0.33) / 0.33)
            else:
                c = dark_red + (np.array(spectral(0)[:4]) - dark_red) * ((t - 0.66) / 0.34)
            ta_colors.append(c)
        for i in range(192):
            v = 0.85 * (i / 191)
            ta_colors.append(spectral(v))
        cmap = ListedColormap(ta_colors, name='total_area')
    else:
        norm = None
        cmap = None

    try:
        from matplotlib.patheffects import withStroke
        path_effects = [withStroke(linewidth=2, foreground='black')]
        overlay_outline = [withStroke(linewidth=4, foreground='black')]  # outline all peptides in black (match individual chrom overlay)
    except Exception:
        path_effects = []
        overlay_outline = []

    PEAK_DRIFT_BUFFER_SEC = 30.0       # fallback when relative buffer cannot be computed
    PEAK_DRIFT_BUFFER_FRAC = 0.5       # buffer = this fraction of collection width (each side)
    PEAK_DRIFT_BUFFER_MIN_SEC = 15.0   # minimum buffer each side (narrow peaks)
    PEAK_DRIFT_BUFFER_MAX_SEC = 90.0   # maximum buffer each side (wide peaks)
    PEAK_DRIFT_GAP_SEC = 2.0   # minimal gap between drift windows (no overlap within a channel)
    PEAK_DRIFT_TRIM_FRAC = 0.2  # allow trimming up to 20% of buffer on either side when arranging to avoid overlap
    PEAK_DRIFT_OVERLAP_FRAC = 0.0  # no overlap: drift windows are separated by at least PEAK_DRIFT_GAP_SEC
    WHITESPACE_BAR_COLOR = (1.0, 1.0, 1.0, 1.0)  # explicit white bar in gap between windows

    def _peak_drift_buffer_sec(coll_min, coll_max):
        """Buffer (seconds) each side. Fixed +30 seconds on either side."""
        return PEAK_DRIFT_BUFFER_SEC

    def _identifier_lines(p, drift_min=None, drift_max=None):
        """Return list of identifier lines for a peptide (sequence, charge, mods, m/z, integration, collection, peak drift, positions). drift_min/drift_max optional (computed per-channel with max spacing)."""
        seq = p.get('peptide_seq') or p.get('plain_peptide') or '-'
        charge = p.get('charge')
        try:
            charge_str = str(int(charge)) if charge not in (None, '') else '-'
        except (TypeError, ValueError):
            charge_str = '-'
        mods_str = format_mods_for_display(p.get('modifications'))
        mz = p.get('mz_theoretical') if p.get('mz_theoretical') is not None else p.get('mz')
        try:
            mz_str = f"{float(mz):.4f}" if mz is not None and not (isinstance(mz, float) and (mz != mz or mz <= 0)) else '-'
        except (TypeError, ValueError):
            mz_str = '-'
        min_rt, max_rt = _get_window(p)
        coll_min, coll_max = _get_collection_window(p)
        # RT labels in minutes (convert from seconds)
        int_str = f"[{min_rt/60:.2f},{max_rt/60:.2f}] min" if min_rt is not None and max_rt is not None else "-"
        coll_str = f"[{coll_min/60:.2f},{coll_max/60:.2f}] min" if coll_min is not None and coll_max is not None else "-"
        if drift_min is not None and drift_max is not None:
            drift_str = f"peak drift collection window: [{drift_min/60:.2f},{drift_max/60:.2f}] min"
        elif coll_min is not None and coll_max is not None:
            drift_str = f"peak drift collection window: [{(coll_min - PEAK_DRIFT_BUFFER_SEC)/60:.2f},{(coll_max + PEAK_DRIFT_BUFFER_SEC)/60:.2f}] min (fallback)"
        else:
            drift_str = "peak drift collection window: -"
        pos_dict = p.get('single_aa_positions') or {}
        if pos_dict:
            def _pos_key(item):
                k = item[0]
                try:
                    return (int(k), k)
                except (TypeError, ValueError):
                    return (0, str(k))
            pos_parts = [f"{k}{v}" for k, v in sorted(pos_dict.items(), key=_pos_key)]
            pos_str = ",".join(pos_parts[:10])
            if len(pos_parts) > 10:
                pos_str += f" (+{len(pos_parts)-10})"
        else:
            pos_str = "-"
        return [
            f"seq: {seq[:40]}{'...' if len(seq) > 40 else ''}",
            f"z: {charge_str}  mods: {mods_str}  m/z: {mz_str}",
            f"integration: {int_str}  collection: {coll_str}",
            drift_str,
            f"positions: {pos_str}",
        ]

    COLLECTION_GRAY = '#404040'
    PEAK_DRIFT_BLACK = '#0a0a0a'
    y_center = 0.5
    bar_height = 0.6
    width = max(40, 36)  # wider so overlay time axis is readable
    height_overlay = 16.0  # much larger overlay panel (was 6.0)
    height_per_channel = 4.0
    fig, axes_all = plt.subplots(1 + num_channels, 1, figsize=(width, height_overlay + height_per_channel * num_channels), sharex=True)
    ax_overlay = axes_all[0]
    axes = axes_all[1:1 + num_channels]
    rt_min_global = float('inf')
    rt_max_global = -float('inf')

    # Top panel: all-peptide overlay — real chromatogram traces only (no synthetic peaks), over full peak drift window
    max_area = max(valid_areas) if valid_areas else 1.0
    overlay_peptides_sorted = sorted(peptides_for_rt_grid, key=lambda p: (_total_area_val(p) or 0))
    overlay_log_ymax = 1e-10
    overlay_log_ymin = float('inf')
    n_overlay = len(overlay_peptides_sorted)
    peptide_to_idx = {id(p): i for i, p in enumerate(peptides_for_rt_grid)}
    use_traces = (peptide_traces is not None and len(peptide_traces) == len(peptides_for_rt_grid))
    # Debug: why overlay might be blank
    import sys
    n_peptides = len(peptides_for_rt_grid)
    n_traces = len(peptide_traces) if peptide_traces is not None else 0
    sys.stderr.write(f"[overlay] peptide_traces is None: {peptide_traces is None}, len(peptide_traces)={n_traces}, len(peptides_for_rt_grid)={n_peptides}, use_traces={use_traces}\n")
    n_plotted = 0
    overlay_labels = []  # (x_apex, y_apex, label_str, color, z) for overlap-aware placement after drawing traces
    OVERLAY_SEQ_MAX = 14   # truncate sequence to this many chars in overlay label
    OVERLAY_FILL_ALPHA = 0.25

    def _overlay_fill(ax, x, y, color, zorder, log_scale):
        """Single smooth fill under the peak (no step so it doesn't look blocky)."""
        floor = max(1e-10, np.finfo(float).tiny) if log_scale else 0.0
        y_plot = np.maximum(y, floor) if log_scale else np.asarray(y, dtype=float)
        ax.fill_between(x, floor, y_plot, color=color, alpha=OVERLAY_FILL_ALPHA, linewidth=0, zorder=zorder)

    for zi, p in enumerate(overlay_peptides_sorted):
        min_rt, max_rt = _get_window(p)
        coll_min, coll_max = _get_collection_window(p)
        if coll_min is None or coll_max is None:
            pad = max(10.0, (max_rt - min_rt) * 0.15)
            coll_min = min_rt - pad
            coll_max = max_rt + pad
        buf_sec = _peak_drift_buffer_sec(coll_min, coll_max)
        x_lo = max(0.0, coll_min - buf_sec)
        x_hi = coll_max + buf_sec
        rt_min_global = min(rt_min_global, x_lo)
        rt_max_global = max(rt_max_global, x_hi)
        area_val = _total_area_val(p)
        color = cmap(norm(area_val)) if (norm and cmap and area_val is not None and (not use_log_scale or area_val > 0)) else (0.7, 0.7, 0.7, 0.7)
        z = n_overlay - zi
        drawn = False
        if use_traces:
            idx = peptide_to_idx.get(id(p))
            if idx is not None and idx < len(peptide_traces):
                rts, intensity = peptide_traces[idx]
                rts = np.asarray(rts, dtype=float)
                intensity = np.asarray(intensity, dtype=float)
                if len(rts) > 0 and len(intensity) == len(rts):
                    mask = (rts >= x_lo) & (rts <= x_hi)
                    if not np.any(mask):
                        mask = np.ones(len(rts), dtype=bool)
                    if np.any(mask):
                        x_peak = rts[mask]
                        y_peak = intensity[mask].copy()
                        y_max = np.max(y_peak)
                        if use_log_scale:
                            floor = max(1e-10, np.finfo(float).tiny)
                            y_peak = np.maximum(y_peak, floor)
                            overlay_log_ymax = max(overlay_log_ymax, float(np.max(y_peak)))
                            pos_vals = y_peak[y_peak > 0]
                            if len(pos_vals) > 0:
                                overlay_log_ymin = min(overlay_log_ymin, float(np.min(pos_vals)))
                        else:
                            if y_max > 0 and max_area > 0 and area_val is not None and area_val > 0:
                                y_peak = (y_peak / y_max) * (area_val / max_area)
                            elif y_max > 0:
                                y_peak = y_peak / y_max
                        if use_log_scale:
                            floor = max(1e-10, np.finfo(float).tiny)
                            y_plot = np.maximum(y_peak, floor)
                            _overlay_fill(ax_overlay, x_peak, y_plot, color, z, log_scale=True)
                            ax_overlay.plot(x_peak, y_plot, color=color, alpha=0.8, linewidth=1.5, zorder=z + 0.5, path_effects=overlay_outline)
                        else:
                            _overlay_fill(ax_overlay, x_peak, y_peak, color, z, log_scale=False)
                            ax_overlay.plot(x_peak, y_peak, color=color, alpha=0.8, linewidth=1.5, zorder=z + 0.5, path_effects=overlay_outline)
                        apex_idx = np.argmax(y_peak)
                        drawn = True
        if not drawn:
            # Synthetic Gaussian peak (match create_rt_overlay_only_plot and individual chromatogram overlay)
            center = (min_rt + max_rt) * 0.5
            sigma = max((max_rt - min_rt) / 4.0, 15.0)
            x_peak = np.linspace(x_lo, x_hi, max(200, int((x_hi - x_lo))))
            amp = (float(area_val) / max_area) if (area_val is not None and max_area > 0) else 0.5
            y_peak = amp * np.exp(-((x_peak - center) ** 2) / (2 * sigma ** 2))
            if use_log_scale:
                floor = max(1e-10, np.finfo(float).tiny)
                y_peak = np.maximum(y_peak, floor)
                overlay_log_ymax = max(overlay_log_ymax, float(np.max(y_peak)))
                _overlay_fill(ax_overlay, x_peak, y_peak, color, z, log_scale=True)
                ax_overlay.plot(x_peak, y_peak, color=color, alpha=0.8, linewidth=1.5, zorder=z + 0.5, path_effects=overlay_outline)
            else:
                _overlay_fill(ax_overlay, x_peak, y_peak, color, z, log_scale=False)
                ax_overlay.plot(x_peak, y_peak, color=color, alpha=0.8, linewidth=1.5, zorder=z + 0.5, path_effects=overlay_outline)
            apex_idx = np.argmin(np.abs(x_peak - center))
        apex_idx = min(int(apex_idx) if hasattr(apex_idx, '__int__') else 0, len(x_peak) - 1) if len(x_peak) > 0 else 0
        apex_idx = max(0, apex_idx)
        x_apex = float(x_peak[apex_idx]) if apex_idx < len(x_peak) else (x_lo + x_hi) * 0.5
        y_apex = float(np.maximum(y_peak[apex_idx], 1e-10) if use_log_scale else y_peak[apex_idx]) if apex_idx < len(y_peak) else (0.01 if use_log_scale else 0.1)
        charge_val = p.get('charge')
        try:
            ch_int = int(charge_val) if charge_val not in (None, '') else 0
        except (TypeError, ValueError):
            ch_int = 0
        charge_str = f'z={ch_int}' if ch_int > 0 else ''
        seq_raw = (p.get('peptide_seq') or p.get('plain_peptide') or '').strip()
        seq_short = (seq_raw[:OVERLAY_SEQ_MAX] + '...') if len(seq_raw) > OVERLAY_SEQ_MAX else seq_raw
        mz_theo = p.get('mz_theoretical') or p.get('mz')
        try:
            mz_val = float(mz_theo) if mz_theo is not None and not (isinstance(mz_theo, float) and (np.isnan(mz_theo) or mz_theo <= 0)) else None
        except (TypeError, ValueError):
            mz_val = None
        mz_str = f" m/z_th {mz_val:.4f}" if mz_val is not None else ""
        if charge_str or mz_str or seq_short:
            label_str = f"{seq_short} {charge_str}{mz_str}".strip()
            if label_str:
                overlay_labels.append((x_apex, y_apex, label_str, color, z + 1))
        n_plotted += 1
    # Place overlay labels with overlap detection; offset and add arrow when overlapping
    label_dx = 48.0
    label_dy = 0.12 if not use_log_scale else 0.3   # log: use ratio-based overlap below
    placed = []   # (x_center, y_center) of each placed label
    # Candidate positions: above/below then left/right then diagonals (prefer near peak)
    def _label_candidates(x_apex, y_apex):
        if use_log_scale and y_apex > 0:
            # Multiplicative offsets for log y-axis
            return [
                (x_apex, y_apex * 1.5), (x_apex, y_apex * 2.0), (x_apex, y_apex * 2.5),
                (x_apex, y_apex * 0.7), (x_apex, y_apex * 0.5),
                (x_apex - 40, y_apex * 1.5), (x_apex + 40, y_apex * 1.5),
                (x_apex - 40, y_apex * 2.0), (x_apex + 40, y_apex * 2.0),
                (x_apex - 40, y_apex * 0.7), (x_apex + 40, y_apex * 0.7),
                (x_apex - 80, y_apex), (x_apex + 80, y_apex),
                (x_apex - 80, y_apex * 1.3), (x_apex + 80, y_apex * 1.3),
                (x_apex - 120, y_apex), (x_apex + 120, y_apex),
            ]
        return [
            (x_apex, y_apex + 0.06), (x_apex, y_apex + 0.14), (x_apex, y_apex + 0.22),
            (x_apex, y_apex - 0.08), (x_apex, y_apex - 0.16),
            (x_apex - 40, y_apex + 0.06), (x_apex + 40, y_apex + 0.06),
            (x_apex - 40, y_apex + 0.14), (x_apex + 40, y_apex + 0.14),
            (x_apex - 40, y_apex - 0.06), (x_apex + 40, y_apex - 0.06),
            (x_apex - 80, y_apex), (x_apex + 80, y_apex),
            (x_apex - 80, y_apex + 0.1), (x_apex + 80, y_apex + 0.1),
            (x_apex - 120, y_apex), (x_apex + 120, y_apex),
        ]
    for (x_apex, y_apex, label_str, color, z) in sorted(overlay_labels, key=lambda t: t[0]):
        def _overlaps(xc, yc):
            for (px, py) in placed:
                if abs(xc - px) < label_dx:
                    if use_log_scale and py > 0 and yc > 0:
                        if abs(np.log10(yc) - np.log10(py)) < 0.3:
                            return True
                    elif abs(yc - py) < label_dy:
                        return True
            return False
        if use_log_scale and y_apex > 0:
            x_txt, y_txt = x_apex, y_apex * 1.5
        else:
            x_txt, y_txt = x_apex, y_apex + 0.06
        if _overlaps(x_txt, y_txt):
            for cand in _label_candidates(x_apex, y_apex):
                if not _overlaps(cand[0], cand[1]):
                    x_txt, y_txt = cand[0], cand[1]
                    break
        placed.append((x_txt, y_txt))
        if use_log_scale and y_apex > 0:
            need_arrow = (abs(x_txt - x_apex) > 3.0 or (y_txt / y_apex) > 1.2 or (y_txt / y_apex) < 0.9)
        else:
            need_arrow = (abs(x_txt - x_apex) > 3.0 or abs(y_txt - y_apex) > 0.02)
        ax_overlay.annotate(
            label_str,
            xy=(x_apex, y_apex),
            xytext=(x_txt, y_txt),
            fontsize=7,
            ha='center',
            va='bottom',
            fontweight='bold',
            zorder=z,
            clip_on=True,
            arrowprops=dict(arrowstyle='->', color=color if len(color) >= 3 else 'gray', lw=0.8, shrinkA=2, shrinkB=2) if need_arrow else None,
        )
    sys.stderr.write(f"[overlay] plotted {n_plotted} / {n_overlay} peptides on overlay\n")
    if use_log_scale and overlay_log_ymax > 0:
        ax_overlay.set_yscale('log')
        y_lo = max(1e-10, overlay_log_ymin * 0.5) if overlay_log_ymin < float('inf') else 1e-10
        y_hi = overlay_log_ymax * 2.0
        ax_overlay.set_ylim(y_lo, y_hi)
    else:
        ax_overlay.set_ylim(0, 1.4)
    ax_overlay.set_ylabel('All peptides (chromatogram traces)' + (' [log scale]' if use_log_scale else '') if use_traces else 'All peptides (synthetic peaks; run with --mzml for real traces)', fontsize=12, fontweight='bold')
    ax_overlay.set_yticks([])
    ax_overlay.grid(axis='x', alpha=0.3, linestyle='--')
    # Global x range: always 0 to at least 20 min (full run view); overlay and RT windows share one x-axis
    RT_PLOT_XMIN_SEC = 0.0
    RT_PLOT_XMAX_SEC = 20.0 * 60.0   # 20 min so x-axis is complete
    if rt_max_global <= rt_min_global:
        rt_min_global = 0
        rt_max_global = 100
    rt_range = rt_max_global - rt_min_global
    x_min = RT_PLOT_XMIN_SEC
    x_max = rt_max_global + max(rt_range * 0.2, 40.0)
    if rt_max_sec is not None and rt_max_sec > 0:
        x_max = max(x_max, rt_max_sec + 60.0)
    x_max = max(x_max, RT_PLOT_XMAX_SEC)
    ax_overlay.set_xlim(x_min, x_max)
    # Horizontal colorbar below the overlay for total_area (same scale as channel bars)
    if norm is not None and cmap is not None:
        from matplotlib.cm import ScalarMappable
        sm_overlay = ScalarMappable(norm=norm, cmap=cmap)
        sm_overlay.set_array([])
        cbar_label = 'Total area (log)' if use_log_scale else 'Total area'
        cbar_overlay = fig.colorbar(sm_overlay, ax=ax_overlay, orientation='horizontal', pad=0.22, shrink=0.7, aspect=35, label=cbar_label)
        cbar_overlay.ax.tick_params(labelsize=9)

    # Channel axes: labels immediately above and below the window bar (bar at 0.2–0.8)
    label_fontsize = 8
    y_label_below = 0.15   # just below the window (bar bottom at 0.2)
    y_label_above = 0.85   # just above the window (bar top at 0.8)
    line_spacing = 0.055   # line height in axes coords for multi-line labels
    channel_ymin, channel_ymax = -0.22, 1.22
    for ch in range(num_channels):
        ax = axes[ch]
        peptides_ch = channels[ch]
        # Sort by collection start so alternating labels follow left-to-right (adjacent windows get opposite sides)
        peptides_ch = sorted(peptides_ch, key=lambda p: (float(p.get('collection_min_rt') or p.get('detected_peak_min_rt') or 0)))
        ax.set_ylim(channel_ymin, channel_ymax)
        ax.set_yticks([0.5])
        ax.set_yticklabels([f'Channel {ch + 1} ({len(peptides_ch)} peptides)'], fontsize=10)
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        n_ch = len(peptides_ch)
        # Use collection windows only (no drift - drift is instrument, not for channel assignment/display)
        channel_data = []
        for i, p in enumerate(peptides_ch):
            min_rt, max_rt = _get_window(p)
            coll_min, coll_max = _get_collection_window(p)
            rt_min_global = min(rt_min_global, min_rt)
            rt_max_global = max(rt_max_global, max_rt)
            if coll_min is None or coll_max is None:
                pad = max(10.0, (max_rt - min_rt) * 0.15)
                coll_min = min_rt - pad
                coll_max = max_rt + pad
            rt_min_global = min(rt_min_global, coll_min)
            rt_max_global = max(rt_max_global, coll_max)
            channel_data.append({'p': p, 'min_rt': min_rt, 'max_rt': max_rt, 'coll_min': coll_min, 'coll_max': coll_max})
        # Draw collection (grey), integration (colored) — no drift bars (drift is instrument, not channel)
        for i, d in enumerate(channel_data):
            p, min_rt, max_rt, coll_min, coll_max = d['p'], d['min_rt'], d['max_rt'], d['coll_min'], d['coll_max']
            is_repeat = p.get('is_repeat_in_channel')
            draw_coll_min = coll_min
            draw_coll_max = coll_max
            coll_w = max(0.0, draw_coll_max - draw_coll_min)
            coll_left_draw = max(0.0, draw_coll_min)
            if coll_w > 1e-6 and draw_coll_max > coll_left_draw:
                ax.barh(y_center, draw_coll_max - coll_left_draw, height=bar_height, left=coll_left_draw, color=COLLECTION_GRAY, alpha=0.6, edgecolor='none', align='center', zorder=0.5)
            int_left = max(min_rt, draw_coll_min)
            int_right = min(max_rt, draw_coll_max)
            int_left_draw = max(0.0, int_left)
            w = max(1e-6, int_right - int_left_draw)
            area_val_bar = _total_area_val(p)
            color = cmap(norm(area_val_bar)) if (norm and cmap and area_val_bar is not None and (not use_log_scale or area_val_bar > 0)) else (0.7, 0.7, 0.7, 0.7)
            ax.barh(y_center, w, height=bar_height, left=int_left_draw, color=color, alpha=0.9, edgecolor=(0.2, 0.4, 0.2, 0.9), linewidth=0.5, align='center', zorder=1)
            area_str = _total_area_sci_str(_total_area_val(p))
            rel = p.get('relative_area_within_selection')
            rel_str = f"{rel * 100:.2f}%" if (rel is not None and isinstance(rel, (int, float)) and 0 <= rel <= 1) else None
            label_str = f"{area_str}  {rel_str}" if rel_str else area_str
            if is_repeat:
                label_str = label_str + " (repeat)" if label_str else "repeat"
            ax.text(int_left_draw + w / 2.0, y_center, label_str, fontsize=8, ha='center', va='center', color='k' if is_repeat else 'white', clip_on=True, path_effects=path_effects if not is_repeat else None)
            x_anchor = max(0.0, coll_min) + 2.0
            lines = _identifier_lines(p, drift_min=coll_min, drift_max=coll_max)
            if i % 2 == 0:
                y_ref = y_label_below
                va = 'top'
                dy = -line_spacing
            else:
                y_ref = y_label_above
                va = 'bottom'
                dy = line_spacing
            for li, line in enumerate(lines):
                weight = 'bold' if li in (0, 1) else 'normal'
                y_line = y_ref + li * dy
                ax.text(x_anchor, y_line, line, fontsize=label_fontsize, ha='left', va=va, clip_on=True, fontweight=weight)
        ax.set_ylabel(f'Channel {ch + 1}', fontsize=12, fontweight='bold')

    # Apply same x-axis to all panels (overlay already set above; channel strips match like extract chromatograms)
    for ax in list(axes) + [ax_overlay]:
        ax.set_xlim(x_min, x_max)
    log_suffix = ' [log scale]' if use_log_scale else ''
    axes[0].set_title(f'{protein_id} - RT integration windows by channel (no overlap within channel){log_suffix}\nGray = collection window; colored = integration (by total_area)', fontsize=12, fontweight='bold', pad=10)
    # Big stage label: Passed / Excluded (figure coordinates, top-left of axes area)
    if stage_label is not None or n_passed is not None or n_excluded is not None:
        label_parts = []
        if stage_label:
            label_parts.append(stage_label)
        if n_passed is not None and n_excluded is not None:
            label_parts.append(f'Passed: {n_passed}  |  Excluded: {n_excluded}')
        elif n_passed is not None:
            label_parts.append(f'Passed: {n_passed}')
        elif n_excluded is not None:
            label_parts.append(f'Excluded: {n_excluded}')
        if label_parts:
            big_label = '   '.join(label_parts)
            pos = axes[0].get_position()
            fig.text(pos.x0, pos.y1 + 0.02, big_label, fontsize=16, fontweight='bold', va='bottom', ha='left',
                     bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat', edgecolor='gray', alpha=0.95))
    axes[-1].set_xlabel('Retention Time (min)', fontsize=12, fontweight='bold')
    # Primary x-axis: data in seconds, display in minutes; major = 1 min, minor = 10 s
    for ax in list(axes) + [ax_overlay]:
        ax.xaxis.set_major_locator(MultipleLocator(60))   # every 1 minute (in seconds)
        ax.xaxis.set_minor_locator(MultipleLocator(10))  # every 10 seconds
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x/60:.0f}'))
        ax.tick_params(axis='x', which='minor', size=3)
    plt.tight_layout(pad=1.2, h_pad=1.5, w_pad=1.2)
    if norm is not None and cmap is not None:
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        fig.subplots_adjust(bottom=0.18, top=0.96, left=0.06, right=0.98)
        pos = axes[min(1, len(axes) - 1)].get_position()
        cbar_ax = fig.add_axes([pos.x0, 0.04, pos.width, 0.03])
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('total_area (MS1 peak)' + (' [log]' if use_log_scale else ''), fontsize=10, fontweight='bold')
    plt.savefig(output_file, dpi=150, bbox_inches='tight', pad_inches=0.25)
    print(f"RT integration windows (by channel, {num_channels} channel(s)) saved to: {output_file}")
    plt.close()

    # Per-channel figures: one file per channel with overlay (channel peptides in black outline) + single channel row
    import os
    base, ext = os.path.splitext(output_file)
    outline_effect = [withStroke(linewidth=7.0, foreground='black')] if path_effects else None
    for ch in range(num_channels):
        channel_set = set(id(p) for p in channels[ch])
        peptides_ch = sorted(channels[ch], key=lambda p: (float(p.get('collection_min_rt') or p.get('detected_peak_min_rt') or 0)))
        n_ch = len(peptides_ch)
        # Build channel_data for this channel first (for channel RT window and to reuse for channel row)
        channel_data = []
        for i, p in enumerate(peptides_ch):
            min_rt, max_rt = _get_window(p)
            coll_min, coll_max = _get_collection_window(p)
            if coll_min is None or coll_max is None:
                pad = max(10.0, (max_rt - min_rt) * 0.15)
                coll_min = min_rt - pad
                coll_max = max_rt + pad
            buf_sec = _peak_drift_buffer_sec(coll_min, coll_max)
            drift_min_full = coll_min - buf_sec
            drift_max_full = coll_max + buf_sec
            if i == 0:
                drift_min = drift_min_full
            else:
                prev_coll_min, prev_coll_max = _get_collection_window(peptides_ch[i - 1])
                if prev_coll_min is not None and prev_coll_max is not None:
                    prev_buf = _peak_drift_buffer_sec(prev_coll_min, prev_coll_max)
                    prev_drift_max = prev_coll_max + prev_buf
                    # No overlap: leave PEAK_DRIFT_GAP_SEC between this drift and previous
                    drift_min = max(drift_min_full, prev_drift_max + PEAK_DRIFT_GAP_SEC, prev_coll_max)
                else:
                    drift_min = drift_min_full
            if i == n_ch - 1:
                drift_max = drift_max_full
            else:
                next_coll_min, next_coll_max = _get_collection_window(peptides_ch[i + 1])
                if next_coll_min is not None and next_coll_max is not None:
                    next_buf = _peak_drift_buffer_sec(next_coll_min, next_coll_max)
                    next_drift_min = next_coll_min - next_buf
                    # No overlap: leave PEAK_DRIFT_GAP_SEC between this drift and next
                    drift_max = min(drift_max_full, next_drift_min - PEAK_DRIFT_GAP_SEC, next_coll_min)
                else:
                    drift_max = drift_max_full
            if drift_max <= drift_min:
                drift_max = drift_min + max(1.0, (coll_max - coll_min) * 0.1)
            channel_data.append({'p': p, 'min_rt': min_rt, 'max_rt': max_rt, 'coll_min': coll_min, 'coll_max': coll_max, 'drift_min': drift_min, 'drift_max': drift_max})
        channel_rt_lo = min(d['drift_min'] for d in channel_data) if channel_data else 0.0
        channel_rt_hi = max(d['drift_max'] for d in channel_data) if channel_data else x_max
        # Per-channel x range: always start at 0 so first minutes are visible; extend to channel end + margin or full run
        rt_span_ch = channel_rt_hi - channel_rt_lo
        margin_ch = max(rt_span_ch * 0.05, 20.0)
        x_min_ch = 0.0
        x_max_ch = channel_rt_hi + margin_ch
        if rt_max_sec is not None and rt_max_sec > 0:
            x_max_ch = max(x_max_ch, rt_max_sec + 60.0)
        y_max_in_channel = 0.0

        fig2, (ax_overlay2, ax_ch2) = plt.subplots(2, 1, figsize=(width, height_overlay + height_per_channel), sharex=True)
        # Per-channel overlay: use the same final list as the channel strip (channel_data) so every peak is plotted.
        # Draw with black outline and highest z-order so all peaks are visible.
        channel_overlay_labels = []
        overlay_peptides = [d['p'] for d in channel_data]
        z_base = 10000
        for zi, p in enumerate(sorted(overlay_peptides, key=lambda p: (-(_total_area_val(p) or 0)))):
            min_rt, max_rt = _get_window(p)
            coll_min, coll_max = _get_collection_window(p)
            if coll_min is None or coll_max is None:
                pad = max(10.0, (max_rt - min_rt) * 0.15)
                coll_min = min_rt - pad
                coll_max = max_rt + pad
            buf_sec = _peak_drift_buffer_sec(coll_min, coll_max)
            x_lo = max(0.0, coll_min - buf_sec)
            x_hi = min(coll_max + buf_sec, channel_rt_hi + 60.0)
            area_val = _total_area_val(p)
            color = cmap(norm(area_val)) if (norm and cmap and area_val is not None and (not use_log_scale or area_val > 0)) else (0.7, 0.7, 0.7, 0.7)
            drawn_ch = False
            if use_traces and peptide_traces is not None:
                idx = peptide_to_idx.get(id(p))
                if idx is not None and idx < len(peptide_traces):
                    rts, intensity = peptide_traces[idx]
                    rts = np.asarray(rts, dtype=float)
                    intensity = np.asarray(intensity, dtype=float)
                    if len(rts) > 0 and len(intensity) == len(rts):
                        rt_lo = max(0.0, channel_rt_lo)
                        mask = (rts >= rt_lo) & (rts <= channel_rt_hi)
                        if not np.any(mask):
                            mask = (rts >= x_lo) & (rts <= x_hi)
                        if np.any(mask):
                            x_peak = rts[mask]
                            y_peak = intensity[mask].copy()
                            y_max_in_channel = max(y_max_in_channel, float(np.max(y_peak)))
                            if use_log_scale:
                                floor = max(1e-10, np.finfo(float).tiny)
                                y_plot = np.maximum(y_peak, floor)
                                _overlay_fill(ax_overlay2, x_peak, y_plot, color, z_base + zi, log_scale=True)
                                ax_overlay2.plot(x_peak, y_plot, color=color, alpha=0.8, linewidth=1.5, zorder=z_base + zi + 0.5, path_effects=outline_effect)
                                apex_i = np.argmax(y_plot)
                            else:
                                _overlay_fill(ax_overlay2, x_peak, y_peak, color, z_base + zi, log_scale=False)
                                ax_overlay2.plot(x_peak, y_peak, color=color, alpha=0.8, linewidth=1.5, zorder=z_base + zi + 0.5, path_effects=outline_effect)
                                apex_i = np.argmax(y_peak)
                            seq = (p.get('peptide_seq') or p.get('plain_peptide') or '-')[:OVERLAY_SEQ_MAX]
                            label_str = seq + ('...' if len(p.get('peptide_seq') or p.get('plain_peptide') or '') > OVERLAY_SEQ_MAX else '')
                            y_apex_val = float(y_peak[apex_i]) if not use_log_scale else float(np.maximum(y_peak[apex_i], 1e-10))
                            channel_overlay_labels.append((float(x_peak[apex_i]), y_apex_val, label_str, color, z_base + zi))
                            drawn_ch = True
            if not drawn_ch:
                # Synthetic Gaussian (match main overlay and individual chromatogram)
                center = (min_rt + max_rt) * 0.5
                sigma = max((max_rt - min_rt) / 4.0, 15.0)
                x_peak = np.linspace(x_lo, x_hi, max(200, int((x_hi - x_lo))))
                amp = (float(area_val) / max_area) if (area_val is not None and max_area > 0) else 0.5
                y_peak = amp * np.exp(-((x_peak - center) ** 2) / (2 * sigma ** 2))
                y_max_in_channel = max(y_max_in_channel, float(np.max(y_peak)))
                if use_log_scale:
                    floor = max(1e-10, np.finfo(float).tiny)
                    y_peak = np.maximum(y_peak, floor)
                    _overlay_fill(ax_overlay2, x_peak, y_peak, color, z_base + zi, log_scale=True)
                    ax_overlay2.plot(x_peak, y_peak, color=color, alpha=0.8, linewidth=1.5, zorder=z_base + zi + 0.5, path_effects=outline_effect)
                else:
                    _overlay_fill(ax_overlay2, x_peak, y_peak, color, z_base + zi, log_scale=False)
                    ax_overlay2.plot(x_peak, y_peak, color=color, alpha=0.8, linewidth=1.5, zorder=z_base + zi + 0.5, path_effects=outline_effect)
                apex_i = np.argmin(np.abs(x_peak - center))
                apex_i = min(apex_i, len(x_peak) - 1)
                seq = (p.get('peptide_seq') or p.get('plain_peptide') or '-')[:OVERLAY_SEQ_MAX]
                label_str = seq + ('...' if len(p.get('peptide_seq') or p.get('plain_peptide') or '') > OVERLAY_SEQ_MAX else '')
                y_apex_val = float(y_peak[apex_i]) if not use_log_scale else float(np.maximum(y_peak[apex_i], 1e-10))
                channel_overlay_labels.append((float(x_peak[apex_i]), y_apex_val, label_str, color, z_base + zi))
        # Y-axis: channel max determines limits so peaks are visible
        if y_max_in_channel > 0:
            if use_log_scale:
                floor = max(1e-10, np.finfo(float).tiny)
                ax_overlay2.set_yscale('log')
                ax_overlay2.set_ylim(max(floor, y_max_in_channel * 0.01), y_max_in_channel * 2.0)
            else:
                ax_overlay2.set_ylim(0, y_max_in_channel * 1.1)
        elif use_log_scale and overlay_log_ymax > 0:
            ax_overlay2.set_yscale('log')
            y_lo = max(1e-10, overlay_log_ymin * 0.5) if overlay_log_ymin < float('inf') else 1e-10
            y_hi = overlay_log_ymax * 2.0
            ax_overlay2.set_ylim(y_lo, y_hi)
        else:
            ax_overlay2.set_ylim(0, 1.4)
        ax_overlay2.set_xlim(x_min, x_max)
        ax_overlay2.set_ylabel(f'Channel {ch + 1} peptides (all outlined in black)' + (' [log scale]' if use_log_scale else '') + (', y = max in channel' if use_traces else ' (synthetic peaks)'), fontsize=12, fontweight='bold')
        ax_overlay2.set_yticks([])
        ax_overlay2.grid(axis='x', alpha=0.3, linestyle='--')
        for _ax in [ax_overlay2, ax_ch2]:
            _ax.xaxis.set_major_locator(MultipleLocator(60))
            _ax.xaxis.set_minor_locator(MultipleLocator(10))
            _ax.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x/60:.0f}'))
            _ax.tick_params(axis='x', which='minor', size=3)
        if norm is not None and cmap is not None:
            sm2 = ScalarMappable(norm=norm, cmap=cmap)
            sm2.set_array([])
            cbar2 = fig2.colorbar(sm2, ax=ax_overlay2, orientation='horizontal', pad=0.22, shrink=0.7, aspect=35, label='Total area (log)' if use_log_scale else 'Total area')
            cbar2.ax.tick_params(labelsize=9)
        # Overlay labels on ax_overlay2 (channel peptides only; positions at actual peaks)
        for (x_apex, y_apex, label_str, color, z) in sorted(channel_overlay_labels, key=lambda t: t[0]):
            if use_log_scale and y_apex > 0:
                x_txt, y_txt = x_apex, y_apex * 1.5
            else:
                x_txt, y_txt = x_apex, y_apex + (y_max_in_channel * 0.06 if y_max_in_channel > 0 else 0.06)
            need_arrow = (abs(x_txt - x_apex) > 3.0 or abs(y_txt - y_apex) > (y_max_in_channel * 0.02 if y_max_in_channel > 0 else 0.02)) if not use_log_scale else (abs(x_txt - x_apex) > 3.0 or (y_txt / y_apex) > 1.2 or (y_txt / y_apex) < 0.9)
            ax_overlay2.annotate(label_str, xy=(x_apex, y_apex), xytext=(x_txt, y_txt), fontsize=7, ha='center', va='bottom', fontweight='bold', zorder=z, clip_on=True,
                arrowprops=dict(arrowstyle='->', color=color if len(color) >= 3 else 'gray', lw=0.8, shrinkA=2, shrinkB=2) if need_arrow else None)
        # Channel row for this channel only (channel_data already built above)
        ax_ch2.set_ylim(channel_ymin, channel_ymax)
        ax_ch2.set_yticks([0.5])
        ax_ch2.set_yticklabels([f'Channel {ch + 1} ({len(peptides_ch)} peptides)'], fontsize=10)
        ax_ch2.grid(axis='x', alpha=0.3, linestyle='--')
        for i in range(len(channel_data) - 1):
            gap_left = channel_data[i]['drift_max']
            gap_right = channel_data[i + 1]['drift_min']
            left = max(0.0, gap_left)
            gap_w = gap_right - left
            if gap_w > 1e-6:
                ax_ch2.barh(y_center, gap_w, height=bar_height, left=left, color=WHITESPACE_BAR_COLOR, alpha=1.0, edgecolor='none', align='center', zorder=-0.5)
        for i, d in enumerate(channel_data):
            p, min_rt, max_rt, coll_min, coll_max, drift_min, drift_max = d['p'], d['min_rt'], d['max_rt'], d['coll_min'], d['coll_max'], d['drift_min'], d['drift_max']
            is_repeat = p.get('is_repeat_in_channel')
            # Clip to drift slot so no overlap within channel (same as main channel strip)
            draw_coll_min = max(coll_min, drift_min)
            draw_coll_max = min(coll_max, drift_max)
            coll_w = max(0.0, draw_coll_max - draw_coll_min)
            drift_left_draw = max(0.0, drift_min)
            if drift_left_draw < coll_min and (coll_min - drift_left_draw) > 1e-6:
                ax_ch2.barh(y_center, coll_min - drift_left_draw, height=bar_height, left=drift_left_draw, color=PEAK_DRIFT_BLACK, alpha=0.85, edgecolor='none', align='center', zorder=0)
            if drift_max > coll_max and (drift_max - coll_max) > 1e-6:
                ax_ch2.barh(y_center, drift_max - coll_max, height=bar_height, left=coll_max, color=PEAK_DRIFT_BLACK, alpha=0.85, edgecolor='none', align='center', zorder=0)
            coll_left_draw = max(0.0, draw_coll_min)
            if coll_w > 1e-6 and draw_coll_max > coll_left_draw:
                ax_ch2.barh(y_center, draw_coll_max - coll_left_draw, height=bar_height, left=coll_left_draw, color=COLLECTION_GRAY, alpha=0.6, edgecolor='none', align='center', zorder=0.5)
            int_left = max(min_rt, draw_coll_min)
            int_right = min(max_rt, draw_coll_max)
            int_left_draw = max(0.0, int_left)
            w = max(1e-6, int_right - int_left_draw)
            area_val_bar = _total_area_val(p)
            color_bar = cmap(norm(area_val_bar)) if (norm and cmap and area_val_bar is not None and (not use_log_scale or area_val_bar > 0)) else (0.7, 0.7, 0.7, 0.7)
            ax_ch2.barh(y_center, w, height=bar_height, left=int_left_draw, color=color_bar, alpha=0.9, edgecolor=(0.2, 0.4, 0.2, 0.9), linewidth=0.5, align='center', zorder=1)
            area_str = _total_area_sci_str(_total_area_val(p))
            rel = p.get('relative_area_within_selection')
            rel_str = f"{rel * 100:.2f}%" if (rel is not None and isinstance(rel, (int, float)) and 0 <= rel <= 1) else None
            label_str = f"{area_str}  {rel_str}" if rel_str else area_str
            if is_repeat:
                label_str = label_str + " (repeat)" if label_str else "repeat"
            ax_ch2.text(int_left_draw + w / 2.0, y_center, label_str, fontsize=8, ha='center', va='center', color='k' if is_repeat else 'white', clip_on=True, path_effects=path_effects if not is_repeat else None)
            x_anchor = max(0.0, drift_min) + 2.0
            lines = _identifier_lines(p, drift_min=drift_min, drift_max=drift_max)
            if i % 2 == 0:
                y_ref, va, dy = y_label_below, 'top', -line_spacing
            else:
                y_ref, va, dy = y_label_above, 'bottom', line_spacing
            for li, line in enumerate(lines):
                weight = 'bold' if li in (0, 1) else 'normal'
                y_line = y_ref + li * dy
                ax_ch2.text(x_anchor, y_line, line, fontsize=label_fontsize, ha='left', va=va, clip_on=True, fontweight=weight)
        ax_ch2.set_ylabel(f'Channel {ch + 1}', fontsize=12, fontweight='bold')
        ax_ch2.set_xlabel('Retention Time (min)', fontsize=12, fontweight='bold')
        ax_ch2.set_title(f'{protein_id} - Channel {ch + 1} only{log_suffix}', fontsize=12, fontweight='bold', pad=10)
        plt.tight_layout(pad=1.2, h_pad=1.5, w_pad=1.2)
        out_ch = f"{base}_ch{ch + 1}{ext}"
        fig2.savefig(out_ch, dpi=150, bbox_inches='tight', pad_inches=0.25)
        print(f"  Per-channel plot saved to: {out_ch}")
        plt.close(fig2)


def create_rt_integration_windows_plot(peptides_for_rt_grid, output_file, protein_id, dedupe_by_peptide=False, protein_sequence=None, rt_max_sec=None, peptides_fallback=None, sort_by_collection_start=False, max_zoom_segments=None, sort_by_channel=False, peptide_traces=None, zoom_segments_dir=None, total_area_range=None, num_channels=1, stage_label=None, n_peptides_previous=None, n_passed=None, n_excluded=None):
    """
    Plot peptides on the y-axis, retention time on the x-axis, and for each row draw the RT integration window in green.
    stage_label: optional (e.g. "Stage 1") to show in the plot title.
    n_peptides_previous: optional int; when set, title shows "Dropped N from previous (M → K peptides)" so viewers see loss at this stage.
    n_passed, n_excluded: optional ints; when set, a large "Passed: X | Excluded: Y" label is drawn on the plot.
    If protein_sequence is provided, position is shown as e.g. 45K-52E; otherwise 45-52.
    If rt_max_sec is set, the x-axis extends to at least that value.
    If sort_by_collection_start is True, rows are ordered by collection window start.
    If sort_by_channel is True, rows are ordered by channel then collection start.
    total_area_range: optional (vmin, vmax) for color scale. num_channels: when sort_by_channel, number of channel strips (from find_min_channels on peptides that pass).
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import MaxNLocator

    num_input = len(peptides_for_rt_grid)
    if num_input == 0:
        print("  No peptides for RT integration windows plot, skipping")
        return
    # When sort_by_channel, use num_channels horizontal strips; do NOT filter by total_area so channel plot is always created when we have peptides (by-channel plot handles missing/0 total_area for color).
    if sort_by_channel:
        try:
            create_rt_windows_by_channel_plot(peptides_for_rt_grid, output_file, protein_id, rt_max_sec=rt_max_sec, peptide_traces=peptide_traces, total_area_range=total_area_range, num_channels=num_channels, stage_label=stage_label, n_passed=n_passed, n_excluded=n_excluded)
            # Log-scale by-channel plots (_log.png and _log_ch1.png ...) are not generated (skip to avoid clutter).
        except Exception as e:
            import traceback
            print(f"  Error creating by-channel RT windows plot: {e}")
            traceback.print_exc()
            print("  Skipping by-channel plot; other plots may still be generated.")
        return

    # For non-channel plot: drop peptides with no/zero total_area (gray bars with total area 0)
    def _has_positive_total_area(p):
        v = p.get('total_area')
        if v is None or (isinstance(v, float) and (np.isnan(v) or v <= 0)):
            return False
        try:
            return float(v) > 0
        except (TypeError, ValueError):
            return False
    indices_kept = [i for i, p in enumerate(peptides_for_rt_grid) if _has_positive_total_area(p)]
    peptides_for_rt_grid = [peptides_for_rt_grid[i] for i in indices_kept]
    if peptide_traces is not None and len(peptide_traces) == num_input:
        peptide_traces = [peptide_traces[i] for i in indices_kept]
    elif peptide_traces is not None:
        peptide_traces = None
    if len(peptides_for_rt_grid) < num_input:
        print(f"  Excluded {num_input - len(peptides_for_rt_grid)} peptides with total_area 0 or missing")
    if not peptides_for_rt_grid:
        print("  No peptides left for RT integration windows plot, skipping")
        return

    def _canon_charge(c):
        try:
            return int(c) if c not in (None, '') else 0
        except (TypeError, ValueError):
            return 0
    def _canon_mods(m):
        if m in (None, ''):
            return '-'
        return (str(m).strip() or '-')
    def _mz_str(p):
        mz = p.get('mz_theoretical') if p.get('mz_theoretical') is not None else p.get('mz')
        return f"{mz:.4f}" if mz is not None and not (isinstance(mz, float) and (mz != mz or mz <= 0)) else "-"
    def _get_window(p):
        min_rt = p.get('detected_peak_min_rt')
        max_rt = p.get('detected_peak_max_rt')
        if min_rt is None or max_rt is None:
            min_rt = p.get('spectrum_window_min_rt')
            max_rt = p.get('spectrum_window_max_rt')
        rt = p.get('rt', 0.0)
        if min_rt is None or max_rt is None:
            try:
                min_rt = float(rt) - 5.0
                max_rt = float(rt) + 5.0
            except (TypeError, ValueError):
                min_rt = 0.0
                max_rt = 10.0
        try:
            min_rt = float(min_rt)
            max_rt = float(max_rt)
            if np.isnan(min_rt) or np.isnan(max_rt):
                min_rt = float(rt) - 5.0
                max_rt = float(rt) + 5.0
        except (TypeError, ValueError):
            min_rt = float(rt) - 5.0
            max_rt = float(rt) + 5.0
        return (min_rt, max_rt)
    def _get_collection_window(p):
        """Return (collection_min_rt, collection_max_rt) in seconds, or (None, None) if not in peptide dict."""
        coll_min = p.get('collection_min_rt')
        coll_max = p.get('collection_max_rt')
        if coll_min is None or coll_max is None:
            return (None, None)
        try:
            coll_min = float(coll_min)
            coll_max = float(coll_max)
            if np.isnan(coll_min) or np.isnan(coll_max) or coll_min >= coll_max:
                return (None, None)
            return (coll_min, coll_max)
        except (TypeError, ValueError):
            return (None, None)
    def _single_aa_pos_str(positions_iterable):
        """Format single-AA overhang positions (with AA letter when protein_sequence available)."""
        positions = sorted(positions_iterable)
        if not positions:
            return "?"
        if protein_sequence:
            parts = []
            for pos in positions:
                try:
                    p = int(pos)
                    if 1 <= p <= len(protein_sequence):
                        parts.append(f"{p}{protein_sequence[p-1]}")
                    else:
                        parts.append(str(p))
                except (TypeError, ValueError):
                    parts.append(str(pos))
            return ", ".join(parts)
        return ", ".join(str(pos) for pos in positions)

    def _single_aa_pos_str_bold_numbers(positions_iterable):
        """Format single-AA overhang positions with only the position numbers in bold (LaTeX).
        These are the significant sequence positions (red squares on the red/black grid)."""
        positions = sorted(positions_iterable)
        if not positions:
            return "?"
        parts = []
        for pos in positions:
            try:
                p = int(pos)
                if protein_sequence and 1 <= p <= len(protein_sequence):
                    aa = protein_sequence[p - 1]
                    parts.append(f"$\\mathbf{{{p}}}${aa}")
                else:
                    parts.append(f"$\\mathbf{{{p}}}$")
            except (TypeError, ValueError):
                parts.append(f"$\\mathbf{{{pos}}}$")
        return ", ".join(parts)
    def _rt_window_str(min_rt, max_rt):
        min_min = min_rt / 60.0
        max_min = max_rt / 60.0
        return f"[{min_rt:.1f}, {max_rt:.1f}]s ({min_min:.2f}-{max_min:.2f} min)"
    def _ms1_rt_str(rt_sec):
        rt_min = rt_sec / 60.0
        return f"{rt_sec:.2f}s ({rt_min:.2f} min)"
    def _single_aa_count(p):
        pos_dict = p.get('single_aa_positions', {})
        return len(pos_dict) if pos_dict else 0
    def _total_area_val(p):
        v = p.get('total_area')
        if v is None or (isinstance(v, float) and (v != v or v < 0)):
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None
    def _total_area_sci_str(a):
        """Format total_area in scientific notation with 3 significant figures."""
        if a is None or (isinstance(a, float) and (np.isnan(a) or a < 0)):
            return "-"
        try:
            x = float(a)
            if x == 0:
                return "0"
            return f"{x:.2e}"  # mantissa has 3 sig figs (e.g. 1.12e+06)
        except (TypeError, ValueError):
            return "-"

    def _build_rows(peptides_list, dedupe, sort_by_coll_start=False, channel_label_style=False):
        """Build sorted list of (min_rt, max_rt, label, total_area, coll_min, coll_max, relative_area) from a peptide list.
        If channel_label_style is True, labels are two-line: line1 = sequence | charge | mods | m/z | positions; line2 = MS1 RT | integration | collection."""
        if dedupe:
            MZ_GROUP_DECIMALS = 3
            def _mz_group_key(mz):
                if mz is None or (isinstance(mz, float) and (mz != mz or mz <= 0)):
                    return None
                try:
                    return round(float(mz), MZ_GROUP_DECIMALS)
                except (TypeError, ValueError):
                    return mz
            groups = {}
            for p in peptides_list:
                charge = _canon_charge(p.get('charge'))
                seq = (p.get('peptide_seq') or '').strip()
                mods = _canon_mods(p.get('modifications'))
                mz_theo = p.get('mz_theoretical')
                mz_key = _mz_group_key(mz_theo)
                if mz_key is not None:
                    key = (charge, mz_key)
                else:
                    key = (charge, seq, mods)
                if key not in groups:
                    groups[key] = []
                groups[key].append(p)
            out_rows = []
            for group_key, group_peptides in groups.items():
                charge = group_key[0]
                mz_key = group_key[1] if len(group_key) == 2 else _mz_group_key(group_peptides[0].get('mz_theoretical'))
                min_rt = min(_get_window(p)[0] for p in group_peptides)
                max_rt = max(_get_window(p)[1] for p in group_peptides)
                unique_mods = set(_canon_mods(p.get('modifications')) for p in group_peptides)
                mods_display = format_mods_for_display(next(iter(unique_mods))) if len(unique_mods) == 1 else "varied"
                mz_str = f"{mz_key:.4f}" if mz_key is not None else "-"
                all_positions = set()
                for p in group_peptides:
                    all_positions.update(p.get('single_aa_positions', {}).keys())
                pos_str_bold = _single_aa_pos_str_bold_numbers(all_positions)
                unique_seqs = sorted(set((p.get('peptide_seq') or '').strip() for p in group_peptides if (p.get('peptide_seq') or '').strip()))
                seqs_label = "; ".join(unique_seqs) if unique_seqs else "-"
                rt_win_str = _rt_window_str(min_rt, max_rt)
                areas_in_group = [_total_area_val(p) for p in group_peptides]
                total_area_row = max((a for a in areas_in_group if a is not None), default=None)
                n_scans_group = len(group_peptides)
                total_group = group_peptides[0].get('_scan_count_total') if group_peptides else None
                if total_group is not None and total_group > 0:
                    scans_str = f" | {int(total_group)} scans"
                else:
                    scans_str = f" | {n_scans_group} scans"
                coll_mins = [_get_collection_window(p)[0] for p in group_peptides if _get_collection_window(p)[0] is not None]
                coll_maxes = [_get_collection_window(p)[1] for p in group_peptides if _get_collection_window(p)[1] is not None]
                coll_min = min(coll_mins) if coll_mins else None
                coll_max = max(coll_maxes) if coll_maxes else None
                backfill_pp = next((pp for pp in group_peptides if pp.get('is_backfill')), None)
                if backfill_pp:
                    reason = backfill_pp.get('backfill_rejection_reason') or backfill_pp.get('_rejection_reason') or 'Backfill'
                    reason_short = (reason[:40] + '…') if len(reason) > 40 else reason
                    backfill_str = f" | Rejected: {reason_short}"
                else:
                    backfill_str = ""
                label = f"pos {pos_str_bold} | {seqs_label} | $\\mathbf{{+{charge}}}$ | {mods_display} | m/z {mz_str}{scans_str} | window {rt_win_str}{backfill_str}"
                rel_area = next((pp.get('relative_area_within_selection') for pp in group_peptides if pp.get('relative_area_within_selection') is not None), None)
                if rel_area is not None and isinstance(rel_area, (int, float)) and np.isfinite(rel_area):
                    pass  # keep as float 0..1
                else:
                    rel_area = None
                out_rows.append((min_rt, max_rt, label, total_area_row, coll_min, coll_max, rel_area))
        else:
            out_rows = []
            for p in peptides_list:
                raw_mods = p.get('modifications', '-') or '-'
                mods_display = format_mods_for_display(raw_mods)
                mz_str = _mz_str(p)
                min_rt, max_rt = _get_window(p)
                rt_win_str = _rt_window_str(min_rt, max_rt)
                ms1_str = _ms1_rt_str(float(p.get('rt', 0)))
                pos_str_bold = _single_aa_pos_str_bold_numbers(p.get('single_aa_positions', {}).keys())
                total_area_row = _total_area_val(p)
                charge_val = p.get('charge', 0)
                total_scans = p.get('_scan_count_total')
                n_scans = p.get('_scan_count')
                if total_scans is not None and total_scans > 0:
                    scans_str = f" | {int(total_scans)} scans"
                elif n_scans is not None and n_scans != '':
                    scans_str = f" | {n_scans} scans"
                else:
                    scans_str = ""
                coll_min, coll_max = _get_collection_window(p)
                if p.get('is_backfill'):
                    reason = p.get('backfill_rejection_reason') or p.get('_rejection_reason') or 'Backfill'
                    reason_short = (reason[:40] + '…') if len(reason) > 40 else reason
                    backfill_str = f" | Rejected: {reason_short}"
                else:
                    backfill_str = ""
                coll_str = _rt_window_str(coll_min, coll_max) if coll_min is not None and coll_max is not None else "N/A"
                # Second line: RT and window information
                line2 = f"MS1 RT {ms1_str} | integration {rt_win_str} | collection {coll_str}"
                if channel_label_style:
                    # Line 1: sequence, charge, mods, m/z, positions (bold numbers = significant single-AA positions)
                    line1 = f"{p.get('peptide_seq', '')} | +{charge_val} | {mods_display} | m/z {mz_str} | pos {pos_str_bold}{backfill_str}"
                    label = line1 + "\n" + line2
                else:
                    # Line 1: positions (bold numbers = red squares on red/black grid), sequence, charge, mods, m/z, scans
                    line1 = f"pos {pos_str_bold} | {p.get('peptide_seq', '')} | $\\mathbf{{+{charge_val}}}$ | {mods_display} | m/z {mz_str}{scans_str}{backfill_str}"
                    label = line1 + "\n" + line2
                rel_area = p.get('relative_area_within_selection')
                if rel_area is not None and isinstance(rel_area, (int, float)) and np.isfinite(rel_area):
                    pass
                else:
                    rel_area = None
                out_rows.append((min_rt, max_rt, label, total_area_row, coll_min, coll_max, rel_area))
        if sort_by_coll_start:
            # Order by collection window start (coll_min); fallback to integration min_rt if missing
            out_rows.sort(key=lambda x: (x[4] if x[4] is not None else x[0]))
        else:
            out_rows.sort(key=lambda x: x[0])
        return out_rows

    # When sort_by_channel, order by (channel, collection_min_rt) so rows are grouped by channel and by RT within channel
    if sort_by_channel:
        def _channel_sort_key(p):
            ch = p.get('channel')
            if ch is None:
                ch = -1
            coll_min, _ = _get_collection_window(p)
            t = coll_min if coll_min is not None else _get_window(p)[0]
            return (ch, t if t is not None else 0.0)
        peptides_for_rt_grid = sorted(peptides_for_rt_grid, key=_channel_sort_key)

    rows = _build_rows(peptides_for_rt_grid, dedupe_by_peptide, sort_by_collection_start, channel_label_style=sort_by_channel)
    if not rows:
        print("  No rows after building RT integration windows plot, skipping")
        return

    y_labels = [r[2] for r in rows]
    window_min = [r[0] for r in rows]
    window_max = [r[1] for r in rows]
    total_areas = [r[3] for r in rows]
    collection_mins = [r[4] for r in rows]
    collection_maxes = [r[5] for r in rows]
    relative_areas = [r[6] if len(r) > 6 else None for r in rows]
    num_peptides = len(rows)

    # Color by total_area: same custom colormap as chromatogram overlays (lavender -> purple -> dark red for lowest, then Spectral)
    from matplotlib.colors import Normalize, ListedColormap
    from matplotlib.cm import ScalarMappable

    def _total_area_colormap():
        spectral = plt.get_cmap('Spectral')
        lavender = np.array([0.90, 0.90, 0.98, 1.0])
        purple = np.array([0.58, 0.44, 0.86, 1.0])
        dark_red = np.array([0.55, 0.0, 0.0, 1.0])
        n_low = 64
        n_high = 192
        colors = []
        for i in range(n_low):
            t = i / max(n_low - 1, 1)
            if t < 0.33:
                c = lavender + (purple - lavender) * (t / 0.33)
            elif t < 0.66:
                c = purple + (dark_red - purple) * ((t - 0.33) / 0.33)
            else:
                c = dark_red + (np.array(spectral(0)[:4]) - dark_red) * ((t - 0.66) / 0.34)
            colors.append(c)
        # Cap high end at blue (spectral ~0.85) so top doesn't go into purple
        SPECTRAL_V_MAX = 0.85
        for i in range(n_high):
            v = SPECTRAL_V_MAX * (i / max(n_high - 1, 1))
            colors.append(spectral(v))
        return ListedColormap(colors, name='total_area')

    valid_areas = [a for a in total_areas if a is not None and not (isinstance(a, float) and (np.isnan(a) or a < 0))]
    if valid_areas:
        if total_area_range is not None and len(total_area_range) >= 2 and total_area_range[0] is not None and total_area_range[1] is not None:
            vmin, vmax = float(total_area_range[0]), float(total_area_range[1])
            if vmax <= vmin:
                vmax = vmin + 1.0
        else:
            vmin = min(valid_areas)
            vmax = max(valid_areas)
            if vmax <= vmin:
                vmax = vmin + 1.0
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = _total_area_colormap()
        colors = [cmap(norm(a)) if a is not None and not (isinstance(a, float) and (np.isnan(a) or a < 0)) else (0.7, 0.7, 0.7, 0.7) for a in total_areas]
        has_colorbar = True
    else:
        colors = [(0.2, 0.6, 0.2, 0.8)] * num_peptides  # default green
        norm = None
        cmap = None
        has_colorbar = False

    # Larger figure so each y-axis row is visible when zooming (more inches per peptide)
    # When sort_by_channel, use double the physical y dimension per row so overlay has room (multi-line labels, no overlap)
    inches_per_row = 3.2 if sort_by_channel else 0.4
    height = max(16, num_peptides * inches_per_row, 10)
    height = min(height, 80.0)  # cap so PNG still loads (by-channel plot)
    width = max(height * 1.5, height + 8)  # x-axis longer than y-axis tall
    fig, ax = plt.subplots(figsize=(width, height))
    # When sort_by_channel: 2 y-units per row so there is room for bar + multi-line text below within collection window
    row_height = 2.0 if sort_by_channel else 1.0
    y_centers = np.arange(num_peptides) * row_height + 0.5
    bar_height = 0.8
    widths = [window_max[i] - window_min[i] for i in range(num_peptides)]
    lefts = window_min
    COLLECTION_GRAY = '#404040'
    for i in range(num_peptides):
        # Collection window: grey bar (full peak including tails, extended); if missing, extend integration window by ~15%
        if collection_mins[i] is not None and collection_maxes[i] is not None:
            coll_left = collection_mins[i]
            coll_w = collection_maxes[i] - collection_mins[i]
        else:
            pad = max(10.0, widths[i] * 0.15)
            coll_left = lefts[i] - pad
            coll_w = widths[i] + 2 * pad
        ax.barh(y_centers[i], coll_w, height=bar_height, left=coll_left, color=COLLECTION_GRAY, alpha=0.6, edgecolor='none', align='center', zorder=0)
        # Integration window: colored bar (quantification window) on top
        ax.barh(y_centers[i], widths[i], height=bar_height, left=lefts[i], color=colors[i], alpha=0.9, edgecolor=(0.2, 0.4, 0.2, 0.9), linewidth=0.5, align='center', zorder=1)
    # Total_area in scientific notation (3 sig figs) inside each bar
    try:
        from matplotlib.patheffects import withStroke
        path_effects = [withStroke(linewidth=2, foreground='black')]
    except Exception:
        path_effects = []
    def _rel_area_str(rel):
        if rel is None or not isinstance(rel, (int, float)) or not (0 <= rel <= 1):
            return None
        return f"{rel * 100:.2f}%"

    _bar_fontsize = 8
    for i in range(num_peptides):
        bar_center_x = lefts[i] + widths[i] / 2.0
        area_str = _total_area_sci_str(total_areas[i])
        rel_str = _rel_area_str(relative_areas[i])
        label_str = f"{area_str}  {rel_str}" if rel_str else area_str
        ax.text(bar_center_x, y_centers[i], label_str, fontsize=_bar_fontsize, ha='center', va='center',
                color='white', path_effects=path_effects, clip_on=True)
    # When sort_by_channel: two-line labels under each bar (identity; RT and window on second line)
    if sort_by_channel and len(peptides_for_rt_grid) == num_peptides:
        for i in range(num_peptides):
            p = peptides_for_rt_grid[i]
            row = rows[i]
            coll_min, coll_max = row[4], row[5]
            min_rt, max_rt = row[0], row[1]
            seq = (p.get('peptide_seq') or '').strip()
            if len(seq) > 45:
                seq = seq[:42] + '...'
            charge_val = p.get('charge', 0)
            raw_mods = p.get('modifications', '-') or '-'
            mods_display = format_mods_for_display(raw_mods)
            mz_str = _mz_str(p)
            pos_str_bold = _single_aa_pos_str_bold_numbers(p.get('single_aa_positions', {}).keys())
            int_str = f"[{min_rt/60:.2f}, {max_rt/60:.2f}] min"
            coll_str = f"[{coll_min/60:.2f}, {coll_max/60:.2f}] min" if (coll_min is not None and coll_max is not None) else "N/A"
            line1 = f"{seq} | +{charge_val} | {mods_display} | m/z {mz_str} | pos {pos_str_bold}"
            line2 = f"integration {int_str} | collection {coll_str}"
            # Position text just below the bar, left-aligned within collection window
            x_pos = coll_min + (coll_max - coll_min) * 0.02 if (coll_min is not None and coll_max is not None and coll_max > coll_min) else lefts[i]
            y_pos = y_centers[i] - bar_height / 2 - 0.02
            ax.text(x_pos, y_pos, line1 + "\n" + line2, fontsize=5, va='top', ha='left', clip_on=True,
                    family='monospace', zorder=10)
        # Y-axis: add margin above and below the windows so labels don't sit on the plot edge
        y_margin = 0.8
        ax.set_ylim(num_peptides * row_height + y_margin, -y_margin)
    if has_colorbar and norm is not None and cmap is not None:
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
    if not sort_by_channel:
        ax.set_ylim(num_peptides, 0)  # Row 0 at top (non-channel plot uses row_height=1)
    ax.set_yticks(y_centers)
    ax.set_yticklabels(y_labels, fontsize=8)
    # Label to the right of each green window (within each row) so zooming shows which peptide each bar is
    rt_range = max(window_max) - min(window_min) if window_max and window_min else 100.0
    right_margin = max(rt_range * 0.2, 40.0)  # space in RT units for in-plot labels to the right of bars
    x_max = max(window_max) + right_margin if window_max else right_margin
    if rt_max_sec is not None and rt_max_sec > 0:
        x_max = max(x_max, rt_max_sec + 60.0)  # extend to full run (e.g. 20 min) so axis shows 0 to run end
    side = height  # for second plot and any code that references side
    # Base offset: use at least 2% of RT range or 25 s so narrow (low-intensity) bars get clear spacing; increase if label would overlap next bar
    base_offset = max(rt_range * 0.02, 25.0)
    min_gap_after_bar = 25.0  # minimum RT (s) between end of a bar and start of next bar before we place label
    label_x_positions = []
    for i in range(num_peptides):
        x = window_max[i] + base_offset
        # If the next row's bar starts close to or left of our label position, push label right so it doesn't overlap
        if i + 1 < num_peptides and window_min[i + 1] is not None and x < window_min[i + 1] + min_gap_after_bar:
            x = window_min[i + 1] + min_gap_after_bar
        label_x_positions.append(x)
    for i in range(num_peptides):
        ax.text(label_x_positions[i], y_centers[i], y_labels[i], fontsize=7, ha='left', va='center',
                clip_on=True)
    # Ensure x-axis extends to fit labels that were pushed right to avoid overlap
    if label_x_positions:
        x_max = max(x_max, max(label_x_positions) + rt_range * 0.08)
    ax.set_xlim(min(window_min) - rt_range * 0.02, x_max)
    ax.xaxis.set_major_locator(MaxNLocator(25, integer=False))
    ax.set_xlabel('Retention Time (s)', fontsize=12, fontweight='bold')
    # Secondary x-axis: retention time in minutes (top)
    ax_min = ax.secondary_xaxis('top', functions=(lambda s: s / 60.0, lambda m: m * 60.0))
    ax_min.set_xlabel('Retention Time (min)', fontsize=12, fontweight='bold')
    ax_min.xaxis.set_major_locator(MaxNLocator(25, integer=False))
    # Optional stage/drop line for filter-stage plots
    drop_line = ''
    if stage_label is not None or n_peptides_previous is not None:
        drop_line = '\n'
        if stage_label:
            drop_line += stage_label
        if n_peptides_previous is not None:
            delta = n_peptides_previous - num_peptides
            sep = ' | ' if drop_line.strip() else ''
            if delta > 0:
                drop_line += f'{sep}Dropped {delta} from previous ({n_peptides_previous} → {num_peptides} peptides)'
            elif delta < 0:
                drop_line += f'{sep}Added {-delta} from backfill ({n_peptides_previous} → {num_peptides} peptides)'
            else:
                drop_line += f'{sep}No change from previous ({num_peptides} peptides)'
    if dedupe_by_peptide:
        ax.set_ylabel('Peptide (sequence(s) | charge | mods | m/z)', fontsize=10, fontweight='bold')
        ax.set_title(f'{protein_id} - RT Integration Windows by m/z (colored by total_area)\nChromatogram extraction window (union over RTs); {num_peptides} rows{drop_line}', fontsize=12, fontweight='bold', pad=10)
    else:
        if sort_by_channel:
            order_note = 'channel 0/1/2 then collection start (no overlap within channel)'
        else:
            order_note = 'collection window start' if sort_by_collection_start else 'window start (min RT)'
        ax.set_ylabel('Peptide (sequence | charge | mods | m/z | MS1 RT)', fontsize=10, fontweight='bold')
        ax.set_title(f'{protein_id} - RT Integration Windows (colored by total_area)\nRows ordered by {order_note}; {num_peptides} peptides{drop_line}', fontsize=12, fontweight='bold', pad=10)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()
    # Big stage label: Passed / Excluded (above plot, after tight_layout so position is stable)
    if n_passed is not None or n_excluded is not None:
        label_parts = []
        if stage_label:
            label_parts.append(stage_label)
        if n_passed is not None and n_excluded is not None:
            label_parts.append(f'Passed: {n_passed}  |  Excluded: {n_excluded}')
        elif n_passed is not None:
            label_parts.append(f'Passed: {n_passed}')
        elif n_excluded is not None:
            label_parts.append(f'Excluded: {n_excluded}')
        if label_parts:
            big_label = '   '.join(label_parts)
            pos = ax.get_position()
            fig.text(0.5, pos.y1 + 0.02, big_label, fontsize=16, fontweight='bold', va='bottom', ha='center',
                     bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat', edgecolor='gray', alpha=0.95))
    # Horizontal colorbar below plot so it doesn't shift x-axis alignment
    if has_colorbar and norm is not None and cmap is not None:
        fig.subplots_adjust(bottom=0.14, left=0.06, right=0.98, top=0.96)
        pos = ax.get_position()
        cbar_ax = fig.add_axes([pos.x0, 0.04, pos.width, 0.025])
        cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
        cbar.set_label('total_area (MS1 peak)', fontsize=10, fontweight='bold')
    # 150 dpi: sharp enough to read when zoomed, file size still loadable
    plt.savefig(output_file, dpi=150, bbox_inches='tight', pad_inches=0.25)
    print(f"RT integration windows plot saved to: {output_file}")
    plt.close()

    # Zoomed versions: split at RT coverage gaps (merge overlapping windows, then one zoom per contiguous RT region)
    # Use actual integration + collection window extent per row so segment boundaries match what we draw
    def _extent_lo(i):
        lo = window_min[i]
        if collection_mins[i] is not None:
            lo = min(lo, collection_mins[i])
        return lo
    def _extent_hi(i):
        hi = window_max[i]
        if collection_maxes[i] is not None:
            hi = max(hi, collection_maxes[i])
        return hi
    intervals_sorted = sorted([(_extent_lo(i), _extent_hi(i)) for i in range(num_peptides)], key=lambda x: x[0])
    merged = []
    for a, b in intervals_sorted:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    # Assign each row index to the merged interval that contains its full extent (integration ∪ collection)
    def _assign_segment(i):
        lo, hi = _extent_lo(i), _extent_hi(i)
        for seg_idx, (m_lo, m_hi) in enumerate(merged):
            if lo <= m_hi and hi >= m_lo:
                return seg_idx
        return 0
    row_to_segment = [_assign_segment(i) for i in range(num_peptides)]
    # Build segments: list of (segment_index, list of row indices in order)
    segments_by_idx = {}
    for i in range(num_peptides):
        s = row_to_segment[i]
        if s not in segments_by_idx:
            segments_by_idx[s] = []
        segments_by_idx[s].append(i)
    segment_list = [segments_by_idx[k] for k in sorted(segments_by_idx.keys())]
    n_zoom_segments = len(segment_list) if max_zoom_segments is None else min(max_zoom_segments, len(segment_list))

    for seg_idx, indices_in_segment in enumerate(segment_list):
        if seg_idx >= n_zoom_segments:
            continue
        n_sub = len(indices_in_segment)
        sub_window_min = [window_min[i] for i in indices_in_segment]
        sub_window_max = [window_max[i] for i in indices_in_segment]
        sub_labels = [y_labels[i] for i in indices_in_segment]
        sub_areas = [total_areas[i] for i in indices_in_segment]
        sub_relative_areas = [relative_areas[i] for i in indices_in_segment]
        sub_coll_mins = [collection_mins[i] for i in indices_in_segment]
        sub_coll_maxes = [collection_maxes[i] for i in indices_in_segment]
        sub_colors = [colors[i] for i in indices_in_segment]
        sub_widths = [sub_window_max[j] - sub_window_min[j] for j in range(n_sub)]
        # X-axis range from this segment's RT (merged interval or peptides' windows)
        seg_start_f = min(sub_window_min[j] for j in range(n_sub))
        seg_end_f = max(sub_window_max[j] for j in range(n_sub))
        for j in range(n_sub):
            if sub_coll_mins[j] is not None:
                seg_start_f = min(seg_start_f, sub_coll_mins[j])
            if sub_coll_maxes[j] is not None:
                seg_end_f = max(seg_end_f, sub_coll_maxes[j])
        rt_span = seg_end_f - seg_start_f
        left_margin = max(10.0, rt_span * 0.02)
        label_margin_sec = max(90.0, rt_span * 0.5)
        x_min = seg_start_f - left_margin
        x_max = seg_end_f + label_margin_sec

        height_zoom = max(8, n_sub * 0.4)
        width_zoom = max(20, 18)  # wider x-axis so RT and labels are easier to read
        fig_zoom, ax_zoom = plt.subplots(figsize=(width_zoom, height_zoom))
        y_centers_zoom = np.arange(n_sub) + 0.5
        # Per row: (1) Collection window = grey bar, extended (full peak including tails); (2) Integration window = colored bar on top (quantification window)
        for j in range(n_sub):
            # Grey bar: collection window (extended); if missing, extend integration window by ~15% each side so grey is visible
            if sub_coll_mins[j] is not None and sub_coll_maxes[j] is not None:
                coll_left = sub_coll_mins[j]
                coll_right = sub_coll_maxes[j]
                cw = coll_right - coll_left
            else:
                pad = max(10.0, sub_widths[j] * 0.15)
                coll_left = sub_window_min[j] - pad
                coll_right = sub_window_max[j] + pad
                cw = coll_right - coll_left
            ax_zoom.barh(y_centers_zoom[j], cw, height=0.8, left=coll_left, color=COLLECTION_GRAY, alpha=0.6, edgecolor='none', align='center', zorder=0)
            # Colored bar: integration window (on top of grey)
            ax_zoom.barh(y_centers_zoom[j], sub_widths[j], height=0.8, left=sub_window_min[j], color=sub_colors[j], alpha=0.9, edgecolor=(0.2, 0.4, 0.2, 0.9), linewidth=0.5, align='center', zorder=1)
        _zoom_bar_fontsize = 8
        for j in range(n_sub):
            bar_center_x = sub_window_min[j] + sub_widths[j] / 2.0
            area_str = _total_area_sci_str(sub_areas[j])
            rel_str = _rel_area_str(sub_relative_areas[j])
            label_str = f"{area_str}  {rel_str}" if rel_str else area_str
            ax_zoom.text(bar_center_x, y_centers_zoom[j], label_str, fontsize=_zoom_bar_fontsize, ha='center', va='center',
                         color='white', path_effects=path_effects if path_effects else [], clip_on=True)
        for j in range(n_sub):
            int_min, int_max = sub_window_min[j], sub_window_max[j]
            int_str = f"integration [{int_min:.1f}, {int_max:.1f}]s ({int_min/60:.2f}-{int_max/60:.2f} min)"
            if sub_coll_mins[j] is not None and sub_coll_maxes[j] is not None:
                coll_str = f" | collection [{sub_coll_mins[j]:.1f}, {sub_coll_maxes[j]:.1f}]s ({sub_coll_mins[j]/60:.2f}-{sub_coll_maxes[j]/60:.2f} min)"
            else:
                coll_str = " | collection N/A"
            zoom_label = sub_labels[j] + " | " + int_str + coll_str
            ax_zoom.text(sub_window_max[j] + 2.0, y_centers_zoom[j], zoom_label, fontsize=7, ha='left', va='center', clip_on=False)
        ax_zoom.set_ylim(n_sub, 0)
        ax_zoom.set_yticks(y_centers_zoom)
        ax_zoom.set_yticklabels([])
        ax_zoom.set_ylabel('')
        ax_zoom.set_xlim(x_min, x_max)
        ax_zoom.xaxis.set_major_locator(MaxNLocator(25, integer=False))
        ax_zoom.set_xlabel('Retention Time (s)', fontsize=12, fontweight='bold')
        ax_zoom_min = ax_zoom.secondary_xaxis('top', functions=(lambda s: s / 60.0, lambda m: m * 60.0))
        ax_zoom_min.set_xlabel('Retention Time (min)', fontsize=12, fontweight='bold')
        ax_zoom_min.xaxis.set_major_locator(MaxNLocator(25, integer=False))
        pep_first = indices_in_segment[0] + 1
        pep_last = indices_in_segment[-1] + 1
        ax_zoom.set_title(f'{protein_id} - RT Integration Windows (segment {seg_idx + 1}: RT {seg_start_f:.0f}-{seg_end_f:.0f} s)\n{n_sub} rows  |  {seg_start_f/60:.2f}-{seg_end_f/60:.2f} min', fontsize=12, fontweight='bold', pad=10)
        ax_zoom.grid(axis='x', alpha=0.3, linestyle='--')
        # Same total_area colorbar as main RT windows plot
        if has_colorbar and norm is not None and cmap is not None:
            sm_zoom = ScalarMappable(norm=norm, cmap=cmap)
            sm_zoom.set_array([])
            cbar_zoom = fig_zoom.colorbar(sm_zoom, ax=ax_zoom, shrink=0.5, aspect=25, pad=0.12)
            cbar_zoom.set_label('total_area (MS1 peak)', fontsize=10, fontweight='bold')
        plt.tight_layout(pad=1.2)
        base_zoom = output_file.replace('.png', '')
        zoom_suffix = f'_zoom_peptides_segment_{seg_idx + 1}_RT_{seg_start_f:.0f}s-{seg_end_f:.0f}s.png'
        zoom_path = base_zoom + zoom_suffix
        plt.savefig(zoom_path, dpi=150, bbox_inches='tight', pad_inches=0.25)
        print(f"RT integration windows plot (segment {seg_idx + 1}, RT {seg_start_f:.0f}-{seg_end_f:.0f} s, {n_sub} rows) saved to: {zoom_path}")
        if zoom_segments_dir:
            import shutil
            os.makedirs(zoom_segments_dir, exist_ok=True)
            zoom_in_subdir = os.path.join(zoom_segments_dir, os.path.basename(zoom_path))
            try:
                shutil.copy2(zoom_path, zoom_in_subdir)
                print(f"  -> also saved to: {zoom_in_subdir}")
            except OSError as e:
                print(f"  Warning: could not copy zoom segment to {zoom_segments_dir}: {e}", file=sys.stderr)
        plt.close(fig_zoom)


def create_total_area_vs_position_scatter_plot(peptides, output_file, protein_id, protein_sequence=None):
    """
    Scatter plot: x = sequence position (protein position of each single-AA overhang),
    y = total_area of the peptide that has that overhang.
    For a peptide with several single-AA overhangs, each overhang is one point at (position, total_area);
    all overhangs from the same peptide share the same y (total_area) and have different x (positions).
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Build (position, total_area) for each single-AA overhang; skip peptides without total_area
    def _total_area_val(p):
        v = p.get('total_area')
        if v is None or (isinstance(v, float) and (np.isnan(v) or v < 0)):
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    positions = []
    areas = []
    for p in peptides:
        single_aa_positions = p.get('single_aa_positions', {})
        if not single_aa_positions:
            continue
        area = _total_area_val(p)
        if area is None:
            continue
        for pos in single_aa_positions.keys():
            try:
                pos_int = int(pos)
                if pos_int >= 1:
                    positions.append(pos_int)
                    areas.append(area)
            except (TypeError, ValueError):
                continue

    if not positions or not areas:
        print("  No (position, total_area) points for scatter plot (missing single_aa_positions or total_area), skipping")
        return

    fig, ax = plt.subplots(figsize=(max(14, (max(positions) - min(positions)) * 0.15), 8))
    ax.scatter(positions, areas, alpha=0.6, s=25, c='steelblue', edgecolors='navy', linewidths=0.5)

    ax.set_xlabel('Sequence position (protein position of single-AA overhang)', fontsize=12, fontweight='bold')
    ax.set_ylabel('total_area (MS1 peak)', fontsize=12, fontweight='bold')
    ax.set_title(f'{protein_id} - total_area vs sequence position\nOne point per single-AA overhang; same peptide → same y (total_area)', fontsize=12, fontweight='bold', pad=10)
    ax.set_xlim(0, max(positions) + 1)
    if min(areas) < max(areas):
        ax.set_ylim(0, max(areas) * 1.05)
    ax.grid(True, alpha=0.3, linestyle='--')
    if protein_sequence:
        # Optional: show residue letters on x-axis at integer positions
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True, nbins=min(50, max(positions) - min(positions) + 1)))
    plt.tight_layout(pad=1.2)
    plt.savefig(output_file, dpi=150, bbox_inches='tight', pad_inches=0.25)
    print(f"total_area vs position scatter plot saved to: {output_file}")
    plt.close()


# Good/ok/poor thresholds for peptide statistics (used in create_peptide_statistic_scatter_plots)
# Each: (column_name, higher_is_better, good_threshold, ok_threshold) -> good if value meets good_threshold, ok if meets ok_threshold, else poor
STATISTIC_THRESHOLDS = [
    ('percolator_qvalue', False, 0.01, 0.05),   # lower better: good <= 0.01, ok <= 0.05
    ('q-value', False, 0.01, 0.05),
    ('delta_cn', True, 0.1, 0.05),
    ('sp_score', True, 50.0, 20.0),
    ('e-value', False, 0.01, 0.1),               # lower better
    ('total_area', True, None, None),            # no fixed thresholds; use tertiles or leave continuous
]


def create_peptide_statistic_scatter_plots(csv_path, output_dir, protein_id=''):
    """
    Scatter plots: for each peptide statistic (q-value, total_area, etc.),
    plot vs RT and vs sequence position, with points colored by good/ok/poor.

    Reads CSV with sequence_positions (e.g. "2-9"), MS1_retention_time_sec or apex_rt,
    and statistic columns. Saves one PNG per statistic: *_vs_rt.png and *_vs_position.png.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    if not os.path.isfile(csv_path):
        print(f"[create_peptide_statistic_scatter_plots] CSV not found: {csv_path}")
        return
    try:
        with open(csv_path, 'r') as f:
            first = f.readline()
        skip = 1 if 'CometVersion' in first else 0
    except Exception:
        skip = 0
    df = pd.read_csv(csv_path, sep=',', skiprows=skip, engine='python', quotechar='"', on_bad_lines='warn')
    if df.empty:
        print(f"[create_peptide_statistic_scatter_plots] Empty CSV: {csv_path}")
        return
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
        print(f"  Validation warning: {e}")

    # Sequence position: from sequence_positions "start-end" -> start
    def _start_pos(val):
        if pd.isna(val) or val is None:
            return None
        s = str(val).strip()
        m = re.match(r'^(\d+)(?:-\d+)?$', s)
        if m:
            return int(m.group(1))
        return None
    if 'sequence_positions' not in df.columns:
        df['_seq_start'] = None
    else:
        df['_seq_start'] = df['sequence_positions'].map(_start_pos)
    # RT: prefer apex_rt (chromatogram), then MS1_retention_time_sec
    if 'apex_rt' in df.columns:
        rt_col = 'apex_rt'
    elif 'MS1_retention_time_sec' in df.columns:
        rt_col = 'MS1_retention_time_sec'
    else:
        rt_col = None
    if rt_col is None:
        print("[create_peptide_statistic_scatter_plots] No RT column (apex_rt or MS1_retention_time_sec); skipping RT scatter plots")

    os.makedirs(output_dir, exist_ok=True)
    base_title = protein_id or 'Peptides'

    for col, higher_better, good_t, ok_t in STATISTIC_THRESHOLDS:
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors='coerce')
        vals = vals.dropna()
        if vals.empty:
            continue
        # Good/ok/poor: for higher_better, good = val >= good_t, ok = val >= ok_t; for lower_better, good = val <= good_t, ok = val <= ok_t
        if good_t is None or ok_t is None:
            # Use tertiles (good = top third, poor = bottom third)
            q33 = vals.quantile(0.33)
            q67 = vals.quantile(0.67)
            if higher_better:
                def _tertile(v):
                    if pd.isna(v): return None
                    if v >= q67: return 'good'
                    if v >= q33: return 'ok'
                    return 'poor'
            else:
                def _tertile(v):
                    if pd.isna(v): return None
                    if v <= q33: return 'good'
                    if v <= q67: return 'ok'
                    return 'poor'
            df['_quality'] = df[col].map(lambda v: _tertile(pd.to_numeric(v, errors='coerce')) if pd.notna(v) else None)
        else:
            if higher_better:
                def _quality(v):
                    if pd.isna(v): return None
                    try:
                        v = float(v)
                        if v >= good_t: return 'good'
                        if v >= ok_t: return 'ok'
                        return 'poor'
                    except (TypeError, ValueError):
                        return None
            else:
                def _quality(v):
                    if pd.isna(v): return None
                    try:
                        v = float(v)
                        if v <= good_t: return 'good'
                        if v <= ok_t: return 'ok'
                        return 'poor'
                    except (TypeError, ValueError):
                        return None
            df['_quality'] = df[col].map(_quality)
        mask = df['_quality'].notna()
        if not mask.any():
            continue
        sub = df.loc[mask]
        x_rt = sub[rt_col] if rt_col else None
        x_pos = sub['_seq_start']
        y = pd.to_numeric(sub[col], errors='coerce')
        qual = sub['_quality']
        colors = qual.map({'good': '#2ecc71', 'ok': '#f39c12', 'poor': '#e74c3c'})
        col_safe = col.replace('-', '_').replace(' ', '_')

        # Scatter: RT vs statistic
        if rt_col and x_rt.notna().any():
            fig, ax = plt.subplots(figsize=(10, 6))
            for q, label, c in [('good', 'Good', '#2ecc71'), ('ok', 'OK', '#f39c12'), ('poor', 'Poor', '#e74c3c')]:
                m = qual == q
                if m.any():
                    ax.scatter(x_rt[m], y[m], alpha=0.6, s=20, c=c, label=label, edgecolors='none')
            ax.set_xlabel('Retention time (s)' if rt_col == 'apex_rt' or rt_col == 'MS1_retention_time_sec' else rt_col, fontsize=11)
            ax.set_ylabel(col, fontsize=11)
            ax.set_title(f'{base_title} – {col} vs retention time (good/ok/poor)', fontsize=12)
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            out_rt = os.path.join(output_dir, f'peptide_stats_{col_safe}_vs_rt.png')
            plt.savefig(out_rt, dpi=150, bbox_inches='tight', pad_inches=0.25)
            plt.close()
            print(f"  Saved: {out_rt}")

        # Scatter: sequence position vs statistic
        if x_pos.notna().any():
            fig, ax = plt.subplots(figsize=(max(10, (x_pos.max() - x_pos.min()) * 0.1), 6))
            for q, label, c in [('good', 'Good', '#2ecc71'), ('ok', 'OK', '#f39c12'), ('poor', 'Poor', '#e74c3c')]:
                m = qual == q
                if m.any():
                    ax.scatter(x_pos[m], y[m], alpha=0.6, s=20, c=c, label=label, edgecolors='none')
            ax.set_xlabel('Sequence position (protein)', fontsize=11)
            ax.set_ylabel(col, fontsize=11)
            ax.set_title(f'{base_title} – {col} vs sequence position (good/ok/poor)', fontsize=12)
            ax.legend(loc='best', fontsize=9)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            out_pos = os.path.join(output_dir, f'peptide_stats_{col_safe}_vs_position.png')
            plt.savefig(out_pos, dpi=150, bbox_inches='tight', pad_inches=0.25)
            plt.close()
            print(f"  Saved: {out_pos}")

    return


def create_fragment_contributors_plot(csv_file, protein_id, fasta_file, output_file, ion_type_filter=None, no_filtering=False):
    """
    Create a visualization showing all fragments that contribute to single-AA overhangs.
    
    Layout:
    - Top: Full protein sequence
    - Below: Grid where each row is a fragment pair, columns are protein positions
    - Y-axis labels: peptide sequence, charge, fragment pair
    - X-axis: protein positions
    - Overhang position is highlighted/bolded
    
    Args:
        csv_file: Path to Comet CSV output file
        protein_id: Protein identifier
        fasta_file: Path to FASTA file
        output_file: Output PNG file path
        ion_type_filter: 'c' for c ions only, 'z' for z/z+1 ions only, None for all
        no_filtering: If True, include all peptides
    """
    import numpy as np
    
    # Parse FASTA
    sequences = parse_fasta(fasta_file)
    if not sequences:
        print("Error: No sequences found in FASTA file")
        return
    
    if protein_id:
        clean_protein_id = re.sub(r'(\d+)$', '', protein_id)
        protein_id_alt = re.sub(r'^sp\|', '', clean_protein_id)
        if clean_protein_id in sequences:
            protein_sequence = sequences[clean_protein_id]
        elif protein_id_alt in sequences:
            protein_sequence = sequences[protein_id_alt]
        elif protein_id in sequences:
            protein_sequence = sequences[protein_id]
        else:
            print(f"Error: Protein {protein_id} not found in FASTA file")
            return
    else:
        protein_id = list(sequences.keys())[0]
        protein_sequence = sequences[protein_id]
    
    protein_length = len(protein_sequence)
    
    # Get filtered peptide indices if needed
    filtered_indices = None
    if not no_filtering:
        filtered_indices, _, _ = get_filtered_peptide_indices(csv_file, protein_id, protein_length, fasta_file, top_n_limit=20)
    
    # Parse all peptides with single-AA overhangs to get fragment pair information
    fragment_contributors = []  # List of (peptide_seq, charge, fragment_pair, position, rt)
    
    with open(csv_file, 'r') as f:
        lines = f.readlines()
        
        if len(lines) < 1:
            print("Error: CSV file too short")
            return
        
        # Detect header line: check if line 0 or line 1 contains 'protein'
        header_line_idx = 1
        if len(lines) > 0 and 'protein' in lines[0]:
            header_line_idx = 0
        elif len(lines) > 1 and 'protein' in lines[1]:
            header_line_idx = 1
        
        header_line = lines[header_line_idx]
        header = [col.strip() for col in header_line.split(',')]
        
        try:
            protein_idx = header.index('protein')
            peptide_idx = header.index('plain_peptide')
            charge_idx = header.index('charge')
            single_aa_pairs_idx = header.index('single_aa_overhang_fragment_pairs')
        except ValueError as e:
            print(f"Error: Required column not found: {e}")
            return
        
        # Retention time: MS1 only - try MS1_retention_time_sec, then retention_time_sec (Comet/Percolator CSV)
        rt_idx = -1
        for col in ['MS1_retention_time_sec', 'retention_time_sec', 'MS1_retention_time', 'retention_time']:
            if col in header:
                rt_idx = header.index(col)
                break
        if rt_idx < 0:
            print("Error: No retention time column found (tried: MS1_retention_time_sec, retention_time_sec, ...). Plots use MS1 retention time only.")
            return
        
        # Single-AA overhangs and site_specific: try standard names, then alternatives
        single_aa_idx = -1
        for col_name in ['single_aa_overhangs', 'single_aa_overhangs_protein_positions']:
            try:
                single_aa_idx = header.index(col_name)
                break
            except ValueError:
                continue
        site_specific_idx = -1
        for col_name in ['site_specific_residues', 'single_aa_overhangs_protein_positions']:
            try:
                site_specific_idx = header.index(col_name)
                break
            except ValueError:
                continue
        if site_specific_idx < 0:
            print(f"Error: Required column not found: site_specific_residues (or single_aa_overhangs_protein_positions)")
            return
        
        # Modifications column (optional)
        modifications_idx = header.index('modifications') if 'modifications' in header else -1
        
        # Use csv.reader for proper CSV parsing
        import io
        data_start_line = header_line_idx + 1
        csv_content = ''.join(lines[data_start_line:])
        csv_reader = csv.reader(io.StringIO(csv_content))
        
        for row_idx, row_data in enumerate(csv_reader):
            if not row_data or len(row_data) == 0:
                continue
            
            if len(row_data) <= max(protein_idx, rt_idx, peptide_idx, charge_idx):
                continue
            
            # Filter by protein
            protein = row_data[protein_idx] if protein_idx < len(row_data) else ""
            if not protein:
                continue
            
            clean_protein = re.sub(r'(\d+)$', '', protein)
            clean_protein_alt = re.sub(r'^sp\|', '', clean_protein)
            if protein_id:
                protein_id_clean = re.sub(r'(\d+)$', '', protein_id)
                protein_id_alt = re.sub(r'^sp\|', '', protein_id_clean)
                if (clean_protein != protein_id_clean and 
                    clean_protein != protein_id_alt and
                    clean_protein_alt != protein_id_clean and
                    clean_protein_alt != protein_id_alt and
                    protein != protein_id):
                    continue
            
            # Filter by indices if needed
            if not no_filtering and filtered_indices is not None and row_idx not in filtered_indices:
                continue
            
            # Get peptide sequence, charge, RT
            peptide_seq = ""
            if peptide_idx >= 0 and len(row_data) > peptide_idx:
                peptide_seq = row_data[peptide_idx].strip('"').strip()
            
            try:
                charge_str = row_data[charge_idx] if charge_idx < len(row_data) else ""
                charge = int(float(charge_str)) if charge_str else 0
            except (ValueError, IndexError):
                charge = 0
            
            try:
                rt_str = row_data[rt_idx] if rt_idx < len(row_data) else ""
                rt = float(rt_str)
            except (ValueError, IndexError):
                rt = 0.0
            
            mods_str = row_data[modifications_idx].strip('"').strip() if modifications_idx >= 0 and modifications_idx < len(row_data) and row_data[modifications_idx] else '-'
            if not mods_str or str(mods_str).lower() == 'nan':
                mods_str = '-'
            
            # Get original line for quoted fields
            original_line = lines[2 + row_idx] if (2 + row_idx) < len(lines) else ""
            quoted_fields = re.findall(r'"([^"]*)"', original_line)
            
            # Get single-AA overhangs and fragment pairs
            single_aa_str = ""
            single_aa_pairs_str = ""
            site_specific_str = ""
            if len(quoted_fields) >= 3:
                single_aa_str = quoted_fields[-3]
            if len(quoted_fields) >= 2:
                single_aa_pairs_str = quoted_fields[-2]
            if len(quoted_fields) >= 1:
                site_specific_str = quoted_fields[-1]
            
            if not site_specific_str or not single_aa_str or not single_aa_pairs_str:
                continue
            
            # Parse site-specific residues to get protein positions
            site_specific_list = []
            for item in site_specific_str.split(','):
                item = item.strip()
                if item and len(item) > 1:
                    match = re.match(r'^(\d+)([A-Z])$', item)
                    if match:
                        protein_pos = int(match.group(1))
                        residue = match.group(2)
                        site_specific_list.append((protein_pos, residue))
            
            if not site_specific_list:
                continue
            
            # Parse single-AA overhangs and fragment pairs
            single_aa_list = [s.strip() for s in single_aa_str.split(',') if s.strip()]
            single_aa_pairs_list = [p.strip() for p in single_aa_pairs_str.split(',') if p.strip()]
            
            # Match each single-AA overhang with its fragment pairs
            for i, (protein_pos, residue) in enumerate(site_specific_list):
                if i < len(single_aa_list) and i < len(single_aa_pairs_list):
                    # Get fragment pairs for this overhang (pipe-separated)
                    pairs_str = single_aa_pairs_list[i]
                    pairs = [p.strip() for p in pairs_str.split('|') if p.strip()]
                    
                    # Filter by ion type if specified
                    for pair in pairs:
                        if filter_fragment_pair_by_ion_type(pair, ion_type_filter):
                            # Normalize fragment pair
                            normalized_pair = normalize_fragment_pair(pair)
                            fragment_contributors.append({
                                'peptide_seq': peptide_seq,
                                'charge': charge,
                                'modifications': mods_str,
                                'fragment_pair': normalized_pair,
                                'position': protein_pos,
                                'rt': rt
                            })
    
    if not fragment_contributors:
        print("No fragment contributors found for single-AA overhangs")
        return
    
    # Deduplicate: keep only unique combinations of (peptide_seq, charge, fragment_pair, position)
    seen = set()
    unique_contributors = []
    for frag in fragment_contributors:
        key = (frag['peptide_seq'], frag['charge'], frag['fragment_pair'], frag['position'])
        if key not in seen:
            seen.add(key)
            unique_contributors.append(frag)
    
    fragment_contributors = unique_contributors
    
    # Sort by position, then by peptide sequence, then by fragment pair
    fragment_contributors.sort(key=lambda x: (x['position'], x['peptide_seq'], x['fragment_pair']))
    
    # Create figure
    num_fragments = len(fragment_contributors)
    fig_width = max(20, protein_length * 0.15)
    fig_height = max(8, num_fragments * 0.08) + 2  # Add space for protein sequence
    
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    # Create grid: protein sequence at top, fragment grid below
    gs = fig.add_gridspec(2, 1, height_ratios=[0.5, 9.5], hspace=0.05)
    ax_seq = fig.add_subplot(gs[0, 0])
    ax_grid = fig.add_subplot(gs[1, 0], sharex=ax_seq)
    
    # Plot 1: Protein sequence at top
    ax_seq.set_xlim(0, protein_length)
    ax_seq.set_ylim(0, 1)
    ax_seq.axis('off')
    
    # Display protein sequence with amino acid letters
    for i, aa in enumerate(protein_sequence):
        pos = i + 1
        x_pos = i + 0.5
        ax_seq.text(x_pos, 0.5, aa, ha='center', va='center', 
                   fontsize=12, fontweight='bold', 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', edgecolor='black', linewidth=1))
        # Position number below
        ax_seq.text(x_pos, 0.1, str(pos), ha='center', va='center', fontsize=8)
    
    ax_seq.set_title(f'Protein Sequence: {protein_id} ({protein_length} residues)', 
                     fontsize=14, fontweight='bold', pad=10)
    
    # Plot 2: Fragment contributors grid
    # Create grid: rows = fragment contributors, cols = protein positions
    grid = np.zeros((num_fragments, protein_length), dtype=float)
    
    # Fill grid: mark the overhang position for each fragment
    y_labels = []
    for i, frag in enumerate(fragment_contributors):
        pos = frag['position']
        if 1 <= pos <= protein_length:
            grid[i, pos - 1] = 1.0  # Mark overhang position
        
        # Create label: peptide_seq | charge | mods | fragment_pair (mods formatted like chromatogram/RT grid)
        peptide_display = frag['peptide_seq'][:30] + '...' if len(frag['peptide_seq']) > 30 else frag['peptide_seq']
        mods_display = format_mods_for_display(frag.get('modifications', '-'))
        label = f"{peptide_display} | +{frag['charge']} | {mods_display} | {frag['fragment_pair']}"
        y_labels.append(label)
    
    # Display grid
    x = np.arange(protein_length + 1)
    y = np.arange(num_fragments + 1)
    X, Y = np.meshgrid(x, y)
    
    # Create colormap: white for 0, red for 1 (overhang position)
    from matplotlib.colors import ListedColormap
    colors = ['white', 'red']
    cmap = ListedColormap(colors)
    
    ax_grid.pcolormesh(X, Y, grid, cmap=cmap, edgecolors='gray', linewidths=0.1, shading='flat')
    
    # Set x-axis to show positions
    x_ticks = [i + 0.5 for i in range(protein_length)]
    x_labels_seq = []
    for i in range(protein_length):
        pos_1idx = i + 1
        if pos_1idx <= len(protein_sequence):
            aa = protein_sequence[i]
            x_labels_seq.append(f"{pos_1idx}\n{aa}")
        else:
            x_labels_seq.append(str(pos_1idx))
    
    ax_grid.set_xticks(x_ticks)
    ax_grid.set_xticklabels(x_labels_seq, fontsize=8, rotation=0)
    ax_grid.set_xlabel('Protein Position', fontsize=12, fontweight='bold')
    
    # Set y-axis to show fragment contributors
    y_tick_positions = [i + 0.5 for i in range(num_fragments)]
    ax_grid.set_yticks(y_tick_positions)
    ax_grid.set_yticklabels(y_labels, fontsize=6)
    ax_grid.set_ylabel('Contributing Fragments\n(peptide sequence | charge | mods | fragment pair)', 
                      fontsize=12, fontweight='bold')
    ax_grid.set_ylim(0, num_fragments)
    ax_grid.set_xlim(0, protein_length)
    
    # Highlight the overhang positions in the grid with bold markers
    for i, frag in enumerate(fragment_contributors):
        pos = frag['position']
        if 1 <= pos <= protein_length:
            x_pos = pos - 0.5
            y_pos = i + 0.5
            # Add bold marker at overhang position
            ax_grid.text(x_pos, y_pos, '●', ha='center', va='center', 
                        fontsize=14, fontweight='bold', color='darkred', zorder=10)
    
    ax_grid.set_title(f'All Fragments Contributing to Single-AA Overhangs\n({num_fragments} fragment pairs, red dots = overhang positions)', 
                     fontsize=14, fontweight='bold', pad=10)
    ax_grid.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='white', edgecolor='gray', label='No overhang'),
        Patch(facecolor='red', edgecolor='black', label='Single-AA overhang position (bolded)'),
    ]
    ax_grid.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.9)
    
    plt.tight_layout(pad=1.2)
    plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0.25)
    print(f"Fragment contributors plot saved to: {output_file}")
    plt.close()


def generate_fragment_pairs_csv(csv_file, protein_id, fasta_file, output_file, no_filtering=False, 
                                ion_type_filter=None, peptides_for_rt_grid=None, peptides=None):
    """
    Generate CSV file with fragment pairs extracted from peptides.
    Uses the same peptides and filtering logic as the combined visualization.
    
    Args:
        csv_file: Input CSV file with peptide data
        protein_id: Protein ID
        fasta_file: FASTA file path
        output_file: Output CSV file path
        no_filtering: If True, include all peptides
        ion_type_filter: 'c' for c ions only, 'z' for z/z+1 ions only, None for all
        peptides_for_rt_grid: List of peptides shown in RT grid (for filtering)
        peptides: List of all peptide dictionaries with quality scores
    """
    import io
    
    # Get protein sequence for mapping positions
    sequences = parse_fasta(fasta_file)
    if not sequences:
        print("Error: No sequences found in FASTA file")
        return
    
    if protein_id:
        clean_protein_id = re.sub(r'(\d+)$', '', protein_id)
        protein_id_alt = re.sub(r'^sp\|', '', clean_protein_id)
        if clean_protein_id in sequences:
            protein_sequence = sequences[clean_protein_id]
        elif protein_id_alt in sequences:
            protein_sequence = sequences[protein_id_alt]
        elif protein_id in sequences:
            protein_sequence = sequences[protein_id]
        else:
            print(f"Error: Protein {protein_id} not found in FASTA file")
            return
    else:
        protein_sequence = list(sequences.values())[0]
    
    protein_length = len(protein_sequence)
    
    # Extract fragment pairs by length difference
    def extract_fragment_pairs_by_length(matched_ions_str):
        """Extract fragment pairs grouped by length difference (1, 2, 3)."""
        if not matched_ions_str or not matched_ions_str.strip():
            return {'single': [], 'double': [], 'triple': []}
        
        fragments = [f.strip() for f in matched_ions_str.split(',') if f.strip()]
        if not fragments:
            return {'single': [], 'double': [], 'triple': []}
        
        # Group fragments by ion series
        fragments_by_series = defaultdict(list)
        
        for frag_str in fragments:
            frag_str_lower = frag_str.lower()
            
            # Handle z1_ format (z1_6 -> z6)
            if frag_str_lower.startswith('z1_'):
                match = re.match(r'z1_(\d+)', frag_str_lower)
                if match:
                    frag_num = int(match.group(1))
                    fragments_by_series['z'].append((frag_num, frag_str))
            else:
                # Regular format: c4, z8, etc.
                match = re.match(r'^([a-z])(\d+)$', frag_str_lower)
                if match:
                    series = match.group(1)
                    frag_num = int(match.group(2))
                    fragments_by_series[series].append((frag_num, frag_str))
        
        # Find all pairs within each ion series
        single_pairs = []
        double_pairs = []
        triple_pairs = []
        
        for series, frag_list in fragments_by_series.items():
            frag_list.sort(key=lambda x: x[0])
            
            for i in range(len(frag_list)):
                for j in range(i + 1, len(frag_list)):
                    frag1_num, frag1_str = frag_list[i]
                    frag2_num, frag2_str = frag_list[j]
                    diff = frag2_num - frag1_num
                    
                    pair_str = f"{frag1_str}-{frag2_str}"
                    normalized_pair = normalize_fragment_pair(pair_str)
                    
                    # Filter by ion type if specified
                    if ion_type_filter and not filter_fragment_pair_by_ion_type(normalized_pair, ion_type_filter):
                        continue
                    
                    if diff == 1:
                        single_pairs.append(normalized_pair)
                    elif diff == 2:
                        double_pairs.append(normalized_pair)
                    elif diff == 3:
                        triple_pairs.append(normalized_pair)
        
        return {
            'single': sorted(set(single_pairs)),
            'double': sorted(set(double_pairs)),
            'triple': sorted(set(triple_pairs))
        }
    
    # Read CSV and extract data
    peptides_data = []
    
    with open(csv_file, 'r') as f:
        lines = f.readlines()
        
        if len(lines) < 1:
            print("Error: CSV file too short")
            return
        
        # Detect header line (same as main CSV: line 0 or line 1)
        header_line_idx = 1
        if len(lines) > 0 and 'protein' in lines[0]:
            header_line_idx = 0
        elif len(lines) > 1 and 'protein' in lines[1]:
            header_line_idx = 1
        if header_line_idx >= len(lines):
            print("Error: CSV file too short")
            return
        
        header_line = lines[header_line_idx]
        header = [col.strip() for col in header_line.split(',')]
        
        try:
            protein_idx = header.index('protein')
            peptide_idx = header.index('plain_peptide')
            matched_ions_idx = header.index('matched fragment ions')
            xcorr_idx = header.index('xcorr') if 'xcorr' in header else -1
            charge_idx = header.index('charge') if 'charge' in header else -1
            intensities_idx = header.index('matched fragment ion intensities') if 'matched fragment ion intensities' in header else -1
            ion_injection_time_idx = header.index('ion_injection_time_ms') if 'ion_injection_time_ms' in header else -1
            
            # Find columns for single-AA overhangs and site-specific residues
            single_aa_overhangs_idx = header.index('single_aa_overhangs') if 'single_aa_overhangs' in header else -1
            site_specific_idx = -1
            for col in ['site_specific_residues', 'single_aa_overhangs_protein_positions']:
                if col in header:
                    site_specific_idx = header.index(col)
                    break
            
            # Retention time: MS1 only - try MS1_retention_time_sec, then retention_time_sec (Comet/Percolator CSV)
            rt_idx = -1
            for col in ['MS1_retention_time_sec', 'retention_time_sec', 'MS1_retention_time', 'retention_time']:
                if col in header:
                    rt_idx = header.index(col)
                    break
            if rt_idx < 0:
                raise ValueError("No retention time column found (tried: MS1_retention_time_sec, retention_time_sec, ...). Plots use MS1 retention time only.")
            
            # Find q-value column
            qvalue_idx = -1
            for col_name in ['percolator_qvalue', 'q-value', 'qvalue', 'FDR', 'fdr', 'percolator_q-value']:
                if col_name in header:
                    qvalue_idx = header.index(col_name)
                    break
        except ValueError as e:
            print(f"Error: Required column not found: {e}")
            return
        
        data_start_line = header_line_idx + 1
        # Create set of peptides from RT grid for filtering
        rt_grid_peptide_keys = set()
        if peptides_for_rt_grid:
            for p in peptides_for_rt_grid:
                peptide_seq = p.get('peptide_seq', '')
                rt = p.get('rt', 0.0)
                charge = p.get('charge', 0)
                rt_grid_peptide_keys.add((peptide_seq, rt, charge))
        
        # Use csv.reader for proper CSV parsing
        csv_content = ''.join(lines[data_start_line:])
        csv_reader = csv.reader(io.StringIO(csv_content))
        
        for row_idx, row_data in enumerate(csv_reader):
            if not row_data or len(row_data) == 0:
                continue
            
            if len(row_data) <= max(protein_idx, rt_idx, peptide_idx):
                continue
            
            # Filter by protein_id
            protein = row_data[protein_idx] if protein_idx < len(row_data) else ""
            if not protein:
                continue
            
            if protein_id:
                clean_protein = re.sub(r'(\d+)$', '', protein)
                clean_protein_alt = re.sub(r'^sp\|', '', clean_protein)
                protein_id_clean = re.sub(r'(\d+)$', '', protein_id)
                protein_id_alt = re.sub(r'^sp\|', '', protein_id_clean)
                if (clean_protein != protein_id_clean and 
                    clean_protein != protein_id_alt and
                    clean_protein_alt != protein_id_clean and
                    clean_protein_alt != protein_id_alt and
                    protein != protein_id):
                    continue
            
            # Get peptide sequence first (needed for matching with peptides list)
            peptide_seq = ""
            if peptide_idx >= 0 and len(row_data) > peptide_idx:
                peptide_seq = row_data[peptide_idx].strip('"').strip()
            
            # Get retention time from CSV first (we need it for matching)
            rt = 0.0
            try:
                if rt_idx >= 0 and rt_idx < len(row_data):
                    rt_str = row_data[rt_idx].strip().strip('"').strip()
                    if rt_str:
                        rt = float(rt_str)
            except (ValueError, IndexError, TypeError):
                pass
            
            # If CSV parsing failed, try to get from peptides list (match by sequence and charge)
            if rt == 0.0 and peptides:
                # Get charge first for matching
                charge_for_matching = None
                if charge_idx >= 0 and charge_idx < len(row_data):
                    try:
                        charge_str = row_data[charge_idx].strip('"').strip()
                        if charge_str:
                            charge_for_matching = int(float(charge_str))
                    except (ValueError, IndexError, TypeError):
                        pass
                
                # Match by sequence and charge
                if charge_for_matching is not None:
                    for p in peptides:
                        p_seq = p.get('peptide_seq', '')
                        p_charge = p.get('charge', 0)
                        if p_seq == peptide_seq and p_charge == charge_for_matching:
                            rt = p.get('rt', 0.0)
                            if rt > 0:
                                break
            
            # Skip if still no retention time
            if rt == 0.0:
                continue
            
            # Filter: only include peptides that are in RT grid (unless no_filtering)
            if not no_filtering and peptides_for_rt_grid:
                # Check if this peptide is in RT grid (match by sequence, RT, and charge)
                in_rt_grid = False
                for p in peptides_for_rt_grid:
                    if (p.get('peptide_seq', '') == peptide_seq and 
                        p.get('rt', 0) == rt and 
                        p.get('charge', 0) == charge):
                        in_rt_grid = True
                        break
                if not in_rt_grid:
                    continue
            
            # Get q-value (use from peptides list if available, otherwise extract from CSV)
            qvalue = None
            if peptides:
                for p in peptides:
                    p_seq = p.get('peptide_seq', '')
                    p_rt = p.get('rt', 0)
                    # Match by sequence and RT (with tolerance)
                    if p_seq == peptide_seq and abs(p_rt - rt) < 0.01:
                        qvalue = p.get('quality_score')
                        if qvalue is not None:
                            break
            
            # If not found in peptides list, try to extract from CSV row
            if qvalue is None and qvalue_idx >= 0 and qvalue_idx < len(row_data):
                try:
                    qvalue_str = row_data[qvalue_idx].strip().strip('"').strip()
                    if qvalue_str:
                        raw_value = float(qvalue_str)
                        # Convert to true q-value (0-1)
                        if raw_value > 1.0:
                            if raw_value <= 100.0:
                                qvalue = raw_value / 100.0
                            elif raw_value > 10:
                                qvalue = 10.0 ** (-raw_value / 10.0)
                            else:
                                qvalue = 10.0 ** (-raw_value)
                        else:
                            qvalue = raw_value
                        qvalue = max(0.0, min(1.0, qvalue))
                except (ValueError, IndexError, TypeError):
                    pass
            
            # Get XCorr
            xcorr = None
            if xcorr_idx >= 0 and xcorr_idx < len(row_data):
                try:
                    xcorr_str = row_data[xcorr_idx].strip('"').strip()
                    if xcorr_str:
                        xcorr = float(xcorr_str)
                except (ValueError, IndexError, TypeError):
                    pass
            
            # Get charge
            charge = None
            if charge_idx >= 0 and charge_idx < len(row_data):
                try:
                    charge_str = row_data[charge_idx].strip('"').strip()
                    if charge_str:
                        charge = int(float(charge_str))
                except (ValueError, IndexError, TypeError):
                    pass
            
            # Get signal intensity
            signal_intensity = None
            if intensities_idx >= 0 and intensities_idx < len(row_data):
                try:
                    intensities_str = row_data[intensities_idx].strip('"').strip()
                    if intensities_str:
                        intensities = [float(x.strip()) for x in intensities_str.split(',') if x.strip()]
                        signal_intensity = sum(intensities)
                except (ValueError, IndexError):
                    pass
            
            # Get fill time (ion injection time)
            fill_time = None
            # Try to get from peptides list first (more reliable)
            # Match by sequence, RT (exact match), and charge
            if peptides and charge is not None:
                for p in peptides:
                    p_seq = p.get('peptide_seq', '')
                    p_rt = p.get('rt', 0.0)
                    p_charge = p.get('charge', 0)
                    # Match by sequence, RT (exact match), and charge
                    if (p_seq == peptide_seq and 
                        p_rt == rt and 
                        p_charge == charge):
                        fill_time = p.get('fill_time', None)
                        if fill_time is not None and fill_time > 0:
                            break
            
            # Fallback to CSV if not found in peptides list
            if (fill_time is None or fill_time == 0) and ion_injection_time_idx >= 0 and ion_injection_time_idx < len(row_data):
                try:
                    fill_time_str = row_data[ion_injection_time_idx].strip('"').strip()
                    if fill_time_str:
                        fill_time = float(fill_time_str)
                        # Only use if it's a valid positive value
                        if fill_time <= 0:
                            fill_time = None
                except (ValueError, IndexError, TypeError):
                    fill_time = None
            
            # Get matched fragment ions
            matched_ions_str = ""
            original_line = lines[2 + row_idx] if (2 + row_idx) < len(lines) else ""
            quoted_fields = re.findall(r'"([^"]*)"', original_line)
            
            # Try to get from quoted fields first (more reliable for CSV with quoted fields)
            for qf in quoted_fields:
                if qf and (',' in qf) and (qf.startswith('c') or qf.startswith('z') or 'z1_' in qf or 
                            qf.startswith('a') or qf.startswith('b') or qf.startswith('x') or qf.startswith('y')):
                    fragments_test = [f.strip() for f in qf.split(',') if f.strip()]
                    valid_frag = False
                    for frag in fragments_test[:3]:
                        frag_lower = frag.lower()
                        if (re.match(r'^[a-z]\d+$', frag_lower) or 
                            re.match(r'^z1_\d+$', frag_lower)):
                            valid_frag = True
                            break
                    if valid_frag:
                        matched_ions_str = qf
                        break
            
            # Fallback to column index if not found in quoted fields
            if not matched_ions_str and matched_ions_idx >= 0 and matched_ions_idx < len(row_data):
                matched_ions_str = row_data[matched_ions_idx].strip('"').strip()
            
            if not matched_ions_str:
                continue
            
            # Validate fragments
            fragments_test = [f.strip() for f in matched_ions_str.split(',') if f.strip()]
            has_valid_fragments = False
            for frag in fragments_test:
                frag_lower = frag.lower()
                if (re.match(r'^[a-z]\d+$', frag_lower) or 
                    re.match(r'^z1_\d+$', frag_lower)):
                    has_valid_fragments = True
                    break
            
            if not has_valid_fragments:
                continue
            
            # Extract fragment pairs by length
            pairs_by_length = extract_fragment_pairs_by_length(matched_ions_str)
            
            # Only include if has pairs (single-AA or double-AA, matching RT grid logic)
            if not pairs_by_length['single'] and not pairs_by_length['double']:
                continue
            
            # Extract single-AA overhang positions and map to protein positions
            single_aa_ordinals = []  # Peptide-relative positions (1-indexed within peptide)
            single_aa_protein_positions = []  # Protein positions with amino acid (e.g., "15K", "23R")
            
            # Get site-specific residues to determine peptide start/end in protein
            peptide_start = 0
            peptide_end = 0
            site_specific_str = ""
            
            # Try to get from quoted fields first (more reliable)
            # site_specific_residues is the last quoted field
            if len(quoted_fields) >= 1:
                potential_str = quoted_fields[-1].strip()
                # Check if it looks like site_specific (format: "15K,16R,17E" or "3L,4K,6I")
                if potential_str and ',' in potential_str:
                    first_item = potential_str.split(',')[0].strip()
                    if re.match(r'^\d+[A-Z]$', first_item):
                        site_specific_str = potential_str
            
            # Fallback to column index
            if not site_specific_str and site_specific_idx >= 0 and site_specific_idx < len(row_data):
                site_specific_str = row_data[site_specific_idx].strip('"').strip()
            
            # If still empty, try to get from quoted fields by checking all of them (backwards)
            if not site_specific_str:
                for qf in reversed(quoted_fields):  # Check from end to beginning
                    qf_stripped = qf.strip()
                    if qf_stripped and ',' in qf_stripped:
                        first_item = qf_stripped.split(',')[0].strip()
                        if re.match(r'^\d+[A-Z]$', first_item):
                            site_specific_str = qf_stripped
                            break
            
            if site_specific_str:
                    # Parse site_specific_residues (format: "15K,16R,17E")
                    site_specific_list = []
                    for item in site_specific_str.split(','):
                        item = item.strip()
                        if item and len(item) > 1:
                            match = re.match(r'^(\d+)([A-Z])$', item)
                            if match:
                                protein_pos = int(match.group(1))
                                residue = match.group(2)
                                site_specific_list.append((protein_pos, residue))
                    
                    if site_specific_list:
                        protein_positions = [pos for pos, res in site_specific_list]
                        peptide_start = min(protein_positions)
                        peptide_end = max(protein_positions)
            
            # Get single-AA overhangs (format: "2L,3K,4R" - peptide-relative ordinals with amino acid)
            # The number is the peptide-relative position (1-indexed), letter is the amino acid
            # NOTE: The CSV column for single_aa_overhangs actually contains fragment pairs, not the overhangs!
            # The actual single-AA overhangs are in quoted field index 3 (4th quoted field)
            single_aa_str = ""
            
            # Try to get from quoted fields first (more reliable)
            # single_aa_overhangs is the 4th quoted field (index 3)
            if len(quoted_fields) >= 4:
                potential_str = quoted_fields[3].strip()
                # Check if it looks like single-AA overhangs (format: "2L,3K,4R")
                if potential_str and ',' in potential_str:
                    first_item = potential_str.split(',')[0].strip()
                    if re.match(r'^\d+[A-Z]$', first_item):
                        single_aa_str = potential_str
            
            # Fallback to column index (but this usually contains fragment pairs, not overhangs)
            if not single_aa_str and single_aa_overhangs_idx >= 0 and single_aa_overhangs_idx < len(row_data):
                potential_str = row_data[single_aa_overhangs_idx].strip('"').strip()
                # Only use if it looks like single-AA overhangs (not fragment pairs)
                if potential_str and ',' in potential_str:
                    first_item = potential_str.split(',')[0].strip()
                    if re.match(r'^\d+[A-Z]$', first_item):
                        single_aa_str = potential_str
            
            if single_aa_str:
                clean_peptide = re.sub(r'\[.*?\]', '', peptide_seq) if peptide_seq else ''
                peptide_length = len(clean_peptide) if clean_peptide else 0
                
                # If we don't have peptide_start from site_specific, try to find it from sequence
                if peptide_start == 0 and clean_peptide:
                    protein_seq_upper = protein_sequence.upper()
                    clean_peptide_upper = clean_peptide.upper()
                    peptide_start_in_protein = protein_seq_upper.find(clean_peptide_upper)
                    if peptide_start_in_protein >= 0:
                        peptide_start = peptide_start_in_protein + 1
                        peptide_end = peptide_start + peptide_length - 1
                
                # Also try to get peptide start/end from peptides list if available
                if peptide_start == 0 and peptides:
                    for p in peptides:
                        if p.get('peptide_seq', '') == peptide_seq and abs(p.get('rt', 0) - rt) < 0.01:
                            peptide_start = p.get('start', 0)
                            peptide_end = p.get('end', 0)
                            if peptide_start > 0 and peptide_end > 0:
                                break
                
                # Parse single_aa_overhangs (format: "2L,3K,4R" where number is peptide-relative ordinal)
                for item in single_aa_str.split(','):
                    item = item.strip()
                    if item and len(item) > 1:
                        match = re.match(r'^(\d+)([A-Z])$', item)
                        if match:
                            peptide_ordinal = int(match.group(1))  # Peptide-relative position (1-indexed)
                            
                            # Map peptide ordinal to protein position
                            if peptide_start > 0 and 1 <= peptide_ordinal <= peptide_length:
                                protein_pos = peptide_start + peptide_ordinal - 1
                                
                                # Get amino acid from protein sequence
                                if 1 <= protein_pos <= protein_length:
                                    aa = protein_sequence[protein_pos - 1]
                                    single_aa_ordinals.append(str(peptide_ordinal))
                                    single_aa_protein_positions.append(f"{protein_pos}{aa}")
            
            # Format protein position range for the full peptide
            protein_position_range = ''
            if peptide_start > 0 and peptide_end > 0:
                protein_position_range = f'{peptide_start}-{peptide_end}'
            elif peptide_start > 0:
                protein_position_range = str(peptide_start)
            elif peptide_end > 0:
                protein_position_range = str(peptide_end)
            
            # Store data - ensure all values are properly formatted
            peptides_data.append({
                'retention_time': rt,  # Keep full precision, no rounding
                'sequence': peptide_seq,
                'protein_position_range': protein_position_range,
                'charge': str(charge) if charge is not None and charge != '' else '',
                'signal_intensity': f'{signal_intensity:.2f}' if signal_intensity is not None and signal_intensity != '' else '',
                'fill_time': f'{fill_time:.2f}' if fill_time is not None and fill_time != '' else '',
                'fragments': matched_ions_str,
                'single_aa_pairs': '|'.join(pairs_by_length['single']) if pairs_by_length['single'] else '',
                'double_aa_pairs': '|'.join(pairs_by_length['double']) if pairs_by_length['double'] else '',
                'triple_aa_pairs': '|'.join(pairs_by_length['triple']) if pairs_by_length['triple'] else '',
                'qvalue': f'{qvalue:.6f}' if qvalue is not None and qvalue != '' else '',
                'xcorr': f'{xcorr:.4f}' if xcorr is not None and xcorr != '' else '',
                'single_aa_ordinals': ','.join(single_aa_ordinals) if single_aa_ordinals else '',
                'single_aa_protein_positions': ','.join(single_aa_protein_positions) if single_aa_protein_positions else ''
            })
    
    # Write CSV
    with open(output_file, 'w', newline='') as f:
        fieldnames = ['retention_time', 'sequence', 'protein_position_range', 'charge', 'signal_intensity', 'fill_time', 'fragments', 'single_aa_pairs', 'double_aa_pairs', 'triple_aa_pairs', 
                     'qvalue', 'xcorr', 'single_aa_ordinals', 'single_aa_protein_positions']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(peptides_data)
    
    print(f"Extracted {len(peptides_data)} peptides with fragment pairs to {output_file}")
    print(f"  - Peptides with single AA pairs: {sum(1 for p in peptides_data if p['single_aa_pairs'])}")
    print(f"  - Peptides with double AA pairs: {sum(1 for p in peptides_data if p['double_aa_pairs'])}")
    print(f"  - Peptides with triple AA pairs: {sum(1 for p in peptides_data if p['triple_aa_pairs'])}")


def generate_threshold_scatter_plot(csv_file, protein_id, protein_length, fasta_file, base_output_file, results_dir=None):
    """
    Generate scatter plots showing:
    1. Number of peptides selected at different top-N thresholds
    2. Single-AA overhang coverage (positions covered) at different top-N thresholds
    
    Args:
        csv_file: Path to Comet CSV output file
        protein_id: Protein identifier
        protein_length: Length of the protein
        fasta_file: Path to FASTA file
        base_output_file: Base output file path (will append _threshold_scatter.png)
        results_dir: Results directory path (optional, will use base_output_file's directory if not provided)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from multi_aa_overhang_visualization import get_filtered_peptide_indices, normalize_fragment_pair
    
    # Test different top-N thresholds for first plot (score-based filtering)
    thresholds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 42, 45, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200]
    peptide_counts = []
    coverage_percentages = []  # Coverage % for each threshold
    
    # For second, third, and fourth plots: Test different two-pass algorithm parameters
    # Vary each parameter individually while keeping others at defaults
    kc_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    kz_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    max_per_residue_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 18, 20, 22, 25, 28, 30, 32, 35, 38, 40, 42, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    
    peptide_counts_kc = []  # Number of peptides for varying Kc (Kz=3, max_per_residue=10)
    peptide_counts_kz = []  # Number of peptides for varying Kz (Kc=3, max_per_residue=10)
    peptide_counts_max = []  # Number of peptides for varying max_per_residue (Kc=3, Kz=3)
    coverage_percentages_kc = []  # Coverage % for varying Kc
    coverage_percentages_kz = []  # Coverage % for varying Kz
    coverage_percentages_max = []  # Coverage % for varying max_per_residue
    
    print(f"\n{'='*60}")
    print("Generating threshold scatter plots...")
    print(f"{'='*60}")
    
    # First plot: Score-based filtering with different top-N thresholds
    print("Calculating score-based filtering results...")
    for threshold in thresholds:
        filtered_indices, covered_c_positions, covered_z_positions = get_filtered_peptide_indices(csv_file, protein_id, protein_length, fasta_file, top_n_limit=threshold)
        peptide_counts.append(len(filtered_indices))
        # Calculate coverage: union of c and z positions covered
        all_covered_positions = covered_c_positions.union(covered_z_positions)
        coverage_pct = (len(all_covered_positions) / protein_length) * 100.0 if protein_length > 0 else 0.0
        coverage_percentages.append(coverage_pct)
        print(f"  Top {threshold:3d}: {len(filtered_indices):4d} peptides, {coverage_pct:.1f}% coverage ({len(all_covered_positions)}/{protein_length} positions)")
    
    # Second plot: Two-pass algorithm with different Kc values
    # Use a fixed top-N threshold (e.g., 20) for the input peptides to two-pass
    print("\nCalculating two-pass algorithm results with varying Kc...")
    fixed_threshold = 20  # Use top 20 for input to two-pass
    filtered_indices_fixed, _, _ = get_filtered_peptide_indices(csv_file, protein_id, protein_length, fasta_file, top_n_limit=fixed_threshold)
    
    # Parse peptides for two-pass algorithm (using fixed threshold)
    peptides_for_two_pass = []
    
    with open(csv_file, 'r') as f:
        lines = f.readlines()
        if len(lines) < 1:
            return
        
        # Detect header line (same as main CSV: line 0 or line 1)
        header_line_idx = 1
        if len(lines) > 0 and 'protein' in lines[0]:
            header_line_idx = 0
        elif len(lines) > 1 and 'protein' in lines[1]:
            header_line_idx = 1
        if header_line_idx >= len(lines):
            return
        
        header_line = lines[header_line_idx]
        header = [col.strip() for col in header_line.split(',')]
        
        try:
            protein_idx = header.index('protein')
            modifications_idx = header.index('modifications')
            peptide_idx = header.index('plain_peptide')
            xcorr_idx = header.index('xcorr')
            charge_idx = header.index('charge') if 'charge' in header else -1
            single_aa_pairs_idx = header.index('single_aa_overhang_fragment_pairs')
        except ValueError as e:
            print(f"Error: Required column not found: {e}")
            return
        
        # Retention time: MS1 only - try MS1_retention_time_sec, then retention_time_sec (Comet/Percolator CSV)
        rt_idx = -1
        for col in ['MS1_retention_time_sec', 'retention_time_sec', 'MS1_retention_time', 'retention_time']:
            if col in header:
                rt_idx = header.index(col)
                break
        if rt_idx < 0:
            print("Error: No retention time column found (tried: MS1_retention_time_sec, retention_time_sec, ...). Plots use MS1 retention time only.")
            return
        
        site_specific_idx = -1
        for col in ['site_specific_residues', 'single_aa_overhangs_protein_positions']:
            if col in header:
                site_specific_idx = header.index(col)
                break
        if site_specific_idx < 0:
            print("Error: Required column not found: site_specific_residues (or single_aa_overhangs_protein_positions)")
            return
        
        data_start_line = header_line_idx + 1
        
        import io
        csv_reader = csv.reader(io.StringIO(''.join(lines[data_start_line:])))
        
        for row_idx, row_data in enumerate(csv_reader):
            if row_idx not in filtered_indices_fixed:
                continue
            
            if len(row_data) <= max(protein_idx, rt_idx, site_specific_idx):
                continue
            
            protein = row_data[protein_idx] if protein_idx < len(row_data) else ""
            if not protein:
                continue
            
            # Match protein ID
            clean_protein = re.sub(r'(\d+)$', '', protein)
            clean_protein_alt = re.sub(r'^sp\|', '', clean_protein)
            if protein_id:
                protein_id_clean = re.sub(r'(\d+)$', '', protein_id)
                protein_id_alt = re.sub(r'^sp\|', '', protein_id_clean)
                matches = (clean_protein == protein_id_clean or 
                          clean_protein == protein_id_alt or
                          clean_protein_alt == protein_id_clean or
                          clean_protein_alt == protein_id_alt or
                          protein == protein_id or
                          (clean_protein_alt and protein_id_alt and clean_protein_alt == protein_id_alt) or
                          ('UB2D3' in protein and 'UB2D3' in protein_id))
                if not matches:
                    continue
            
            # Get quoted fields
            original_line = lines[2 + row_idx] if (2 + row_idx) < len(lines) else ""
            quoted_fields = re.findall(r'"([^"]*)"', original_line)
            
            site_specific_str = ""
            single_aa_pairs_str = ""
            if len(quoted_fields) >= 2:
                single_aa_pairs_str = quoted_fields[-2]
            if len(quoted_fields) >= 1:
                site_specific_str = quoted_fields[-1]
            
            if not site_specific_str and len(row_data) > site_specific_idx:
                site_specific_str = row_data[site_specific_idx].strip('"').strip()
            if not single_aa_pairs_str and len(row_data) > single_aa_pairs_idx:
                single_aa_pairs_str = row_data[single_aa_pairs_idx].strip('"').strip()
            
            if not site_specific_str:
                continue
            
            try:
                rt = float(row_data[rt_idx]) if row_data[rt_idx] else 0.0
                peptide_seq = row_data[peptide_idx].strip('"').strip() if len(row_data) > peptide_idx else ''
                xcorr = float(row_data[xcorr_idx]) if len(row_data) > xcorr_idx and row_data[xcorr_idx] else 0.0
                charge = int(row_data[charge_idx]) if charge_idx >= 0 and len(row_data) > charge_idx and row_data[charge_idx] else 0
                
                # Parse site_specific to get protein positions
                protein_positions = []
                site_specific_list = []
                for item in site_specific_str.split(','):
                    item = item.strip()
                    if item and len(item) > 1:
                        match = re.match(r'^(\d+)([A-Z])$', item)
                        if match:
                            protein_pos = int(match.group(1))
                            residue = match.group(2)
                            protein_positions.append(protein_pos)
                            site_specific_list.append((protein_pos, residue))
                
                if not protein_positions:
                    continue
                
                peptide_start = min(protein_positions)
                peptide_end = max(protein_positions)
                
                # Parse single-AA fragment pairs
                single_aa_fragment_pairs = {}
                if single_aa_pairs_str and site_specific_list:
                    single_aa_list = [s.strip() for s in single_aa_pairs_str.split(',') if s.strip()]
                    single_aa_pairs_list = [p.strip() for p in single_aa_pairs_str.split(',') if p.strip()] if single_aa_pairs_str else []
                    
                    for i, (protein_pos, protein_residue) in enumerate(site_specific_list):
                        if 1 <= protein_pos <= protein_length:
                            normalized_pairs = set()
                            if i < len(single_aa_pairs_list):
                                pairs = [p.strip() for p in single_aa_pairs_list[i].split('|') if p.strip()]
                                for pair in pairs:
                                    normalized_pair = normalize_fragment_pair(pair)
                                    normalized_pairs.add(normalized_pair)
                            
                            if normalized_pairs:
                                single_aa_fragment_pairs[protein_pos] = normalized_pairs
                
                peptides_for_two_pass.append({
                    'rt': rt,
                    'peptide_seq': peptide_seq,
                    'charge': charge,
                    'start': peptide_start,
                    'end': peptide_end,
                    'single_aa_fragment_pairs': single_aa_fragment_pairs,
                    'xcorr': xcorr
                })
            except (ValueError, IndexError):
                continue
    
    # Test different Kc values (Kz=3, max_per_residue=42 fixed to see Kc effect)
    # Note: max_per_residue=42 means max 42 single-AA overhangs per residue
    print("  Varying Kc (Kz=3, max_per_residue=42 single-AA overhangs per residue):")
    for kc in kc_values:
        if peptides_for_two_pass:
            two_pass_selected = two_pass_selection_algorithm(peptides_for_two_pass, protein_length, Kc=kc, Kz=3, max_per_residue=42, debug=(kc == 1 or kc == 10))
            peptide_counts_kc.append(len(two_pass_selected))
            # Calculate coverage from selected peptides - only count positions with c-ion fragment pairs
            covered_positions = set()
            for peptide in two_pass_selected:
                single_aa_pairs = peptide.get('single_aa_fragment_pairs', {})
                for pos, pairs_set in single_aa_pairs.items():
                    if 1 <= pos <= protein_length:
                        # Check if this position has c-ion fragment pairs
                        has_c_ion = False
                        for pair in pairs_set:
                            normalized_pair = normalize_fragment_pair(pair)
                            if normalized_pair.startswith('c') and '-' in normalized_pair:
                                parts = normalized_pair.split('-')
                                if len(parts) == 2 and parts[1].startswith('c'):
                                    has_c_ion = True
                                    break
                        if has_c_ion:
                            covered_positions.add(pos)
            coverage_pct = (len(covered_positions) / protein_length) * 100.0 if protein_length > 0 else 0.0
            coverage_percentages_kc.append(coverage_pct)
            print(f"    Kc={kc:2d}: {len(two_pass_selected):4d} peptides, {coverage_pct:.1f}% c-ion coverage ({len(covered_positions)}/{protein_length} positions)")
        else:
            peptide_counts_kc.append(0)
            coverage_percentages_kc.append(0.0)
            print(f"    Kc={kc:2d}: 0 peptides")
    
    # Test different Kz values (Kc=3, max_per_residue=42 fixed to see Kz effect)
    # Note: max_per_residue=42 means max 42 single-AA overhangs per residue
    print("  Varying Kz (Kc=3, max_per_residue=42 single-AA overhangs per residue):")
    for kz in kz_values:
        if peptides_for_two_pass:
            two_pass_selected = two_pass_selection_algorithm(peptides_for_two_pass, protein_length, Kc=3, Kz=kz, max_per_residue=42, debug=(kz == 1 or kz == 10))
            peptide_counts_kz.append(len(two_pass_selected))
            # Calculate coverage from selected peptides - only count positions with z-ion fragment pairs
            covered_positions = set()
            for peptide in two_pass_selected:
                single_aa_pairs = peptide.get('single_aa_fragment_pairs', {})
                for pos, pairs_set in single_aa_pairs.items():
                    if 1 <= pos <= protein_length:
                        # Check if this position has z-ion fragment pairs
                        has_z_ion = False
                        for pair in pairs_set:
                            normalized_pair = normalize_fragment_pair(pair)
                            if normalized_pair.startswith('z') and '-' in normalized_pair:
                                parts = normalized_pair.split('-')
                                if len(parts) == 2 and parts[1].startswith('z'):
                                    has_z_ion = True
                                    break
                        if has_z_ion:
                            covered_positions.add(pos)
            coverage_pct = (len(covered_positions) / protein_length) * 100.0 if protein_length > 0 else 0.0
            coverage_percentages_kz.append(coverage_pct)
            print(f"    Kz={kz:2d}: {len(two_pass_selected):4d} peptides, {coverage_pct:.1f}% z-ion coverage ({len(covered_positions)}/{protein_length} positions)")
        else:
            peptide_counts_kz.append(0)
            coverage_percentages_kz.append(0.0)
            print(f"    Kz={kz:2d}: 0 peptides")
    
    # Test different max_per_residue values (Kc=3, Kz=3 fixed)
    print("  Varying max_per_residue (Kc=3, Kz=3):")
    for max_val in max_per_residue_values:
        if peptides_for_two_pass:
            two_pass_selected = two_pass_selection_algorithm(peptides_for_two_pass, protein_length, Kc=3, Kz=3, max_per_residue=max_val)
            peptide_counts_max.append(len(two_pass_selected))
            # Calculate coverage from selected peptides
            covered_positions = set()
            for peptide in two_pass_selected:
                single_aa_pairs = peptide.get('single_aa_fragment_pairs', {})
                for pos in single_aa_pairs.keys():
                    if 1 <= pos <= protein_length:
                        covered_positions.add(pos)
            coverage_pct = (len(covered_positions) / protein_length) * 100.0 if protein_length > 0 else 0.0
            coverage_percentages_max.append(coverage_pct)
            print(f"    max_per_residue={max_val:2d}: {len(two_pass_selected):4d} peptides, {coverage_pct:.1f}% coverage ({len(covered_positions)}/{protein_length} positions)")
        else:
            peptide_counts_max.append(0)
            coverage_percentages_max.append(0.0)
            print(f"    max_per_residue={max_val:2d}: 0 peptides")
    
    # Find min and max coverage across all plots for consistent colormap scale
    all_coverage = coverage_percentages + coverage_percentages_kc + coverage_percentages_kz + coverage_percentages_max
    if all_coverage:
        coverage_min = min(all_coverage)
        coverage_max = max(all_coverage)
        print(f"\nDEBUG: Coverage statistics:")
        print(f"  Coverage range: {coverage_min:.2f}% - {coverage_max:.2f}%")
        print(f"  Threshold plot coverage values: {[f'{c:.1f}%' for c in coverage_percentages[:5]]}... (showing first 5)")
        print(f"  Kc plot coverage values: {[f'{c:.1f}%' for c in coverage_percentages_kc[:5]]}... (showing first 5)")
        print(f"  Kz plot coverage values: {[f'{c:.1f}%' for c in coverage_percentages_kz[:5]]}... (showing first 5)")
        print(f"  Max_per_residue plot coverage values: {[f'{c:.1f}%' for c in coverage_percentages_max[:5]]}... (showing first 5)")
    else:
        coverage_min = 0.0
        coverage_max = 100.0
        print(f"\nWARNING: No coverage values calculated!")
    
    # Create figure with four subplots in a 2x2 grid:
    # Top row: Plot 1 (thresholds) and Plot 4 (max_per_residue)
    # Bottom row: Plot 2 (Kc) and Plot 3 (Kz)
    # Increased height to accommodate explanation text beneath Kc and Kz plots
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.3, bottom=0.15)
    ax1 = fig.add_subplot(gs[0, 0])  # Top left: thresholds
    ax4 = fig.add_subplot(gs[0, 1])  # Top right: max_per_residue
    ax2 = fig.add_subplot(gs[1, 0])  # Bottom left: Kc
    ax3 = fig.add_subplot(gs[1, 1])  # Bottom right: Kz
    
    # Create colormap for coverage percentage (viridis: low=dark purple, high=yellow)
    from matplotlib.colors import Normalize
    import matplotlib
    try:
        # matplotlib 3.7+: use colormaps registry (avoids deprecated cm.get_cmap)
        cmap = matplotlib.colormaps.get_cmap('viridis')
    except (AttributeError, TypeError):
        cmap = matplotlib.colormaps['viridis']
    norm = Normalize(vmin=coverage_min, vmax=coverage_max)
    
    # Plot 1: Number of peptides selected (colored by coverage %)
    print(f"\nDEBUG: Plot 1 (thresholds) - coverage values: {coverage_percentages}")
    print(f"  Using colormap with range: {coverage_min:.2f}% - {coverage_max:.2f}%")
    scatter1 = ax1.scatter(thresholds, peptide_counts, s=100, alpha=0.7, edgecolors='black', linewidth=1.5, 
                          c=coverage_percentages, cmap=cmap, norm=norm)
    ax1.plot(thresholds, peptide_counts, 'k-', alpha=0.3, linewidth=1)
    ax1.set_xlabel('Top N Peptides Selected per Position\n(by q-value, c-ions and z-ions independently)', 
                   fontsize=12, fontweight='bold')
    ax1.set_ylabel('Total Number of Peptides Selected', fontsize=12, fontweight='bold')
    ax1.set_title('Peptide Selection vs. Top-N Threshold\n(Progressive Coverage: Single-AA → Multi-AA)', 
                 fontsize=14, fontweight='bold', pad=15)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(0, max(thresholds) + 5)
    y1_max = (max(peptide_counts) * 1.1) if peptide_counts else 1
    ax1.set_ylim(0, max(1, y1_max))
    # Set x-axis ticks to every integer, but only label at intervals to avoid overcrowding
    max_x = int(max(thresholds)) + 5
    ax1.set_xticks(range(0, max_x + 1, 1))  # Tick marks at every integer
    # Only show labels at every 10th integer to avoid overcrowding
    ax1.set_xticklabels([str(i) if i % 10 == 0 or i in thresholds else '' for i in range(0, max_x + 1)], fontsize=8)
    ax1.tick_params(axis='x', labelsize=8, length=3)  # Smaller tick marks
    
    # Add value labels on points
    for i, (x, y) in enumerate(zip(thresholds, peptide_counts)):
        ax1.annotate(f'{y}', (x, y), textcoords="offset points", xytext=(0,10), 
                   ha='center', fontsize=8, fontweight='bold')
    
    # Add vertical dashed line at default threshold value (42)
    default_threshold = 42
    if default_threshold in thresholds:
        ax1.axvline(x=default_threshold, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Default Top N={default_threshold}')
        ax1.legend(loc='upper left', fontsize=8, framealpha=0.9)
    
    # Plot 2: Two-pass algorithm - Vary Kc (Kz=3, max_per_residue=42 fixed) (colored by coverage %)
    print(f"\nDEBUG: Plot 2 (Kc) - coverage values: {coverage_percentages_kc}")
    scatter2 = ax2.scatter(kc_values, peptide_counts_kc, s=100, alpha=0.7, edgecolors='black', linewidth=1.5,
                          c=coverage_percentages_kc, cmap=cmap, norm=norm)
    ax2.plot(kc_values, peptide_counts_kc, 'k-', alpha=0.3, linewidth=1)
    ax2.set_xlabel('Kc (C-Supported Peptides per Position)\n(Kz=3, max_per_residue=42 overhangs/residue)', 
                   fontsize=11, fontweight='bold')
    ax2.set_ylabel('Total Number of Peptides Selected', fontsize=11, fontweight='bold')
    ax2.set_title('Two-Pass: Vary Kc\n(Input: Top 20 by q-value)', 
                 fontsize=12, fontweight='bold', pad=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim(0, max(kc_values) + 1)
    y2_max = (max(peptide_counts_kc) * 1.1) if peptide_counts_kc else 1
    ax2.set_ylim(0, max(1, y2_max))
    
    # Add value labels on points
    for i, (x, y) in enumerate(zip(kc_values, peptide_counts_kc)):
        ax2.annotate(f'{y}', (x, y), textcoords="offset points", xytext=(0,10), 
                   ha='center', fontsize=7, fontweight='bold')
    
    # Add vertical dashed line at default Kc value (3)
    default_kc = 3
    if default_kc in kc_values:
        ax2.axvline(x=default_kc, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Default Kc={default_kc}')
        ax2.legend(loc='upper left', fontsize=8, framealpha=0.9)
    
    # Add explanation text for Kc beneath the plot
    kc_explanation = ("Kc is the number of c-supported peptides to keep per position/cluster in Pass 1. "
                     "C-supported peptides have at least one c ion fragment pair (e.g., c3-c4, c5-c6) "
                     "and provide precise N-terminal positioning. Higher values (Kc=5) give more "
                     "comprehensive coverage but allow more redundancy, while lower values (Kc=1-2) "
                     "produce a cleaner, more selective set but may miss important peptides.")
    kc_wrapped = '\n'.join(textwrap.wrap(kc_explanation, width=60))
    ax2.text(0.5, -0.30, kc_wrapped, transform=ax2.transAxes, 
            fontsize=8, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # Plot 3: Two-pass algorithm - Vary Kz (Kc=3, max_per_residue=42 fixed) (colored by coverage %)
    print(f"\nDEBUG: Plot 3 (Kz) - coverage values: {coverage_percentages_kz}")
    scatter3 = ax3.scatter(kz_values, peptide_counts_kz, s=100, alpha=0.7, edgecolors='black', linewidth=1.5,
                          c=coverage_percentages_kz, cmap=cmap, norm=norm)
    ax3.plot(kz_values, peptide_counts_kz, 'k-', alpha=0.3, linewidth=1)
    ax3.set_xlabel('Kz (Z-Supported Peptides per Region)\n(Kc=3, max_per_residue=42 overhangs/residue)', 
                   fontsize=11, fontweight='bold')
    ax3.set_ylabel('Total Number of Peptides Selected', fontsize=11, fontweight='bold')
    ax3.set_title('Two-Pass: Vary Kz\n(Input: Top 20 by q-value)', 
                 fontsize=12, fontweight='bold', pad=10)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.set_xlim(0, max(kz_values) + 1)
    y3_max = (max(peptide_counts_kz) * 1.1) if peptide_counts_kz else 1
    ax3.set_ylim(0, max(1, y3_max))
    
    # Add value labels on points
    for i, (x, y) in enumerate(zip(kz_values, peptide_counts_kz)):
        ax3.annotate(f'{y}', (x, y), textcoords="offset points", xytext=(0,10), 
                   ha='center', fontsize=7, fontweight='bold')
    
    # Add vertical dashed line at default Kz value (3)
    default_kz = 3
    if default_kz in kz_values:
        ax3.axvline(x=default_kz, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Default Kz={default_kz}')
        ax3.legend(loc='upper left', fontsize=8, framealpha=0.9)
    
    # Add explanation text for Kz beneath the plot
    kz_explanation = ("Kz is the number of z-supported peptides to add per region in Pass 2. "
                     "Z-supported peptides have at least one z/z+1 ion fragment pair (e.g., z4-z5, z1_6-z1_7) "
                     "and provide C-terminal validation. Higher values (Kz=3-4) provide better gap filling "
                     "and more complete coverage but result in more peptides overall, while lower values "
                     "(Kz=1) are more selective with fewer peptides but may leave some coverage gaps.")
    kz_wrapped = '\n'.join(textwrap.wrap(kz_explanation, width=60))
    ax3.text(0.5, -0.30, kz_wrapped, transform=ax3.transAxes, 
            fontsize=8, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    # Plot 4: Two-pass algorithm - Vary max_per_residue (Kc=3, Kz=3 fixed) (colored by coverage %)
    print(f"\nDEBUG: Plot 4 (max_per_residue) - coverage values: {coverage_percentages_max}")
    scatter4 = ax4.scatter(max_per_residue_values, peptide_counts_max, s=100, alpha=0.7, edgecolors='black', linewidth=1.5,
                          c=coverage_percentages_max, cmap=cmap, norm=norm)
    ax4.plot(max_per_residue_values, peptide_counts_max, 'k-', alpha=0.3, linewidth=1)
    ax4.set_xlabel('max_per_residue (Max Single-AA Overhangs per Residue)\n(Kc=3, Kz=3)', 
                   fontsize=11, fontweight='bold')
    ax4.set_ylabel('Total Number of Peptides Selected', fontsize=11, fontweight='bold')
    ax4.set_title('Two-Pass: Vary max_per_residue\n(Input: Top 20 by q-value)', 
                 fontsize=12, fontweight='bold', pad=10)
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.set_xlim(0, max(max_per_residue_values) + 1)
    y4_max = (max(peptide_counts_max) * 1.1) if peptide_counts_max else 1
    ax4.set_ylim(0, max(1, y4_max))
    
    # Add value labels on points
    for i, (x, y) in enumerate(zip(max_per_residue_values, peptide_counts_max)):
        ax4.annotate(f'{y}', (x, y), textcoords="offset points", xytext=(0,10), 
                   ha='center', fontsize=7, fontweight='bold')
    
    # Add vertical dashed line at default max_per_residue value (42)
    default_max_per_residue = 42
    if default_max_per_residue in max_per_residue_values:
        ax4.axvline(x=default_max_per_residue, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Default max_per_residue={default_max_per_residue}')
        ax4.legend(loc='upper left', fontsize=8, framealpha=0.9)
    
    # Generate output filename
    # Extract filename from base_output_file and create new filename
    base_filename = os.path.basename(base_output_file)
    if base_filename.endswith('.png'):
        scatter_filename = base_filename.replace('.png', '_threshold_scatter.png')
    else:
        scatter_filename = base_filename + '_threshold_scatter.png'
    
    # DEBUG: Print information about results_dir
    print(f"\nDEBUG: generate_threshold_scatter_plot output path determination:")
    print(f"  base_output_file: {base_output_file}")
    print(f"  base_filename: {base_filename}")
    print(f"  scatter_filename: {scatter_filename}")
    print(f"  results_dir: {results_dir}")
    print(f"  results_dir exists: {os.path.isdir(results_dir) if results_dir else False}")
    
    # Determine output directory: require results_dir (no fallback)
    if results_dir and os.path.isdir(results_dir):
        scatter_output = os.path.join(results_dir, scatter_filename)
        print(f"  Using results_dir: {scatter_output}")
    else:
        error_msg = f"ERROR: results_dir is not valid! results_dir={results_dir}, exists={os.path.isdir(results_dir) if results_dir else False}"
        print(f"  {error_msg}")
        raise ValueError(error_msg)
    
    # Add colorbar for coverage percentage (shared across all plots)
    # Position colorbar on the right side of the figure
    from matplotlib.cm import ScalarMappable
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.90, 0.12, 0.025, 0.76])  # [left, bottom, width, height] - more gap from plot
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
    cbar.set_label('Sequence Coverage (%)', fontsize=12, fontweight='bold', labelpad=15)
    cbar.ax.tick_params(labelsize=10)
    
    # Use subplots_adjust instead of tight_layout to avoid warning with manually placed colorbar axes
    fig.subplots_adjust(left=0.08, right=0.86, bottom=0.06, top=0.94, hspace=0.4, wspace=0.35)
    plt.savefig(scatter_output, dpi=300, bbox_inches='tight', pad_inches=0.25)
    print(f"Threshold scatter plots saved to: {scatter_output}")
    print(f"DEBUG: File exists after save: {os.path.exists(scatter_output)}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Create combined visualization with heatmap, histogram, and RT grid'
    )
    parser.add_argument('--csv', required=True, help='Comet CSV output file')
    parser.add_argument('--fasta', required=True, help='FASTA file with protein sequences')
    parser.add_argument('--mzml', help='mzML file (optional). Used to read ion injection/fill times by scan number when the CSV has no fill time column or values are missing/incorrect.')
    parser.add_argument('--output', default='visualization.png',
                       help='Output PNG file (default: visualization.png)')
    parser.add_argument('--protein', help='Specific protein ID to visualize (optional)')
    parser.add_argument('--c-ions-only', action='store_true',
                       help='Filter to show only c ion fragment pairs')
    parser.add_argument('--z-ions-only', action='store_true',
                       help='Filter to show only z/z+1 ion fragment pairs')
    parser.add_argument('--no-filtering', action='store_true',
                       help='Skip all filtering and include all peptides with overhangs')
    parser.add_argument('--two-pass', action='store_true',
                       help='Use two-pass selection algorithm (c-anchored + z-supported). Default: enabled when --two-pass is used')
    parser.add_argument('--no-progressive', action='store_true',
                       help='Disable progressive selection (gap-filling with 2-AA and 3-AA overhangs). Default: enabled')
    parser.add_argument('--no-two-pass', action='store_true',
                       help='Disable two-pass selection algorithm. Default: enabled when --two-pass flag is used')
    parser.add_argument('--peak-windows-csv', 
                       help='Path to peak_windows CSV file (peak_windows_all.csv, peak_windows.csv, or peak_windows_rejected.csv) from extract_ms1_chromatograms.py')
    parser.add_argument('--filter-peak-window-status', choices=['accepted', 'rejected'],
                       help='Filter peptides by peak window status (accepted or rejected). Requires --peak-windows-csv.')
    
    args = parser.parse_args()
    
    # Validate FASTA path (must be a file, not a directory)
    fasta_path = os.path.abspath(os.path.expanduser(args.fasta))
    if os.path.isdir(fasta_path):
        print(f"Error: --fasta must be a FASTA file, not a directory: {args.fasta}")
        return
    if not os.path.isfile(fasta_path):
        print(f"Error: FASTA file not found: {args.fasta}")
        return
    args.fasta = fasta_path  # Use resolved path for all subsequent use
    
    # Validate peak window filter
    if args.filter_peak_window_status and not args.peak_windows_csv:
        print("Error: --filter-peak-window-status requires --peak-windows-csv")
        return
    
    # Validate ion type flags (mutually exclusive)
    if args.c_ions_only and args.z_ions_only:
        print("Error: --c-ions-only and --z-ions-only are mutually exclusive")
        return
    
    # Determine ion type filter
    ion_type_filter = None
    if args.c_ions_only:
        ion_type_filter = 'c'
    elif args.z_ions_only:
        ion_type_filter = 'z'
    
    # Get protein ID
    sequences = parse_fasta(args.fasta)
    if not sequences:
        print("Error: No sequences found in FASTA file")
        return
    
    if args.protein:
        protein_id = args.protein
    else:
        protein_id = list(sequences.keys())[0]
        print(f"No protein specified, using first protein: {protein_id}")
    
    # Generate main plot
    create_combined_visualization(args.csv, protein_id, args.fasta, args.output, 
                                 ion_type_filter=ion_type_filter, no_filtering=args.no_filtering,
                                 use_two_pass=args.two_pass, no_progressive=args.no_progressive,
                                 no_two_pass=args.no_two_pass, mzml_file=args.mzml,
                                 peak_windows_csv=getattr(args, 'peak_windows_csv', None),
                                 filter_by_peak_window_status=getattr(args, 'filter_peak_window_status', None))
    
    # Generate fragment contributors plot
    base_output = args.output
    if base_output.endswith('.png'):
        fragment_output = base_output.replace('.png', '_fragment_contributors.png')
    else:
        fragment_output = base_output + '_fragment_contributors.png'
    
    print(f"\n{'='*60}")
    print(f"Generating fragment contributors plot: {fragment_output}")
    print(f"{'='*60}")
    create_fragment_contributors_plot(args.csv, protein_id, args.fasta, fragment_output,
                                     ion_type_filter=ion_type_filter, no_filtering=args.no_filtering)
    
    # Automatically generate c-ions-only and z-ions-only plots (unless one is already being generated)
    if ion_type_filter is None:
        base_output = args.output
        if base_output.endswith('.png'):
            base_output = base_output[:-4]
        
        c_output = base_output + '_c_ions.png'
        z_output = base_output + '_z_ions.png'
        
        # Generate c-ions-only plot
        print(f"\n{'='*60}")
        print(f"Generating c-ions-only plot: {c_output}")
        print(f"{'='*60}")
        create_combined_visualization(args.csv, protein_id, args.fasta, c_output, 
                                     ion_type_filter='c', no_filtering=args.no_filtering,
                                     use_two_pass=args.two_pass, no_progressive=args.no_progressive,
                                     no_two_pass=args.no_two_pass, mzml_file=args.mzml,
                                     peak_windows_csv=getattr(args, 'peak_windows_csv', None),
                                     filter_by_peak_window_status=getattr(args, 'filter_peak_window_status', None))
        
        # Generate z-ions-only plot
        print(f"\n{'='*60}")
        print(f"Generating z-ions-only plot: {z_output}")
        print(f"{'='*60}")
        create_combined_visualization(args.csv, protein_id, args.fasta, z_output, 
                                     ion_type_filter='z', no_filtering=args.no_filtering,
                                     use_two_pass=args.two_pass, no_progressive=args.no_progressive,
                                     no_two_pass=args.no_two_pass, mzml_file=args.mzml,
                                     peak_windows_csv=getattr(args, 'peak_windows_csv', None),
                                     filter_by_peak_window_status=getattr(args, 'filter_peak_window_status', None))


# Module mode: main() function is available for import
# For command-line usage, use the separate script files:
# - generate_combined_plot.py
# - generate_fragment_pair_grid.py  
# - generate_fragment_contributors.py
# Or run this file directly: python3 combined_overhang_visualization.py [args]
if __name__ == '__main__':
    main()