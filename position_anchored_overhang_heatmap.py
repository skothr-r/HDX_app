#!/usr/bin/env python3
"""
Position-Anchored Overhang Heatmap Visualization

Creates a grid of heatmaps (one per sequence position) showing all overhangs
that BEGIN at each position. This allows comparison of "families" of overhangs
that share the same starting position.

Each subplot:
- Shows only overhangs starting at that position
- X-axis extends to the longest overhang starting at that position
- Y-axis shows different peptides/measurements
"""

import argparse
import re
import csv
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
import numpy as np
import sys
import os

# Import functions from multi_aa_overhang_visualization.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multi_aa_overhang_visualization import (
    parse_fasta, parse_multi_aa_overhangs, parse_single_aa_overhangs
)

# Import τ estimation functions from fractional_error_coverage_heatmap.py
from fractional_error_coverage_heatmap import (
    estimate_tau_from_single_aa_overhangs,
    estimate_tau_from_overlap_inconsistency
)


def group_overhangs_by_start_position(overhangs, protein_length):
    """
    Group overhangs by their starting position.
    
    Returns:
        dict mapping start_position -> list of overhang dicts
    """
    grouped = defaultdict(list)
    
    for overhang in overhangs:
        start = overhang.get('start')
        if start and 1 <= start <= protein_length:
            grouped[start].append(overhang)
    
    return grouped


def group_overhangs_by_end_position(overhangs, protein_length):
    """
    Group overhangs by their ending position.
    
    Returns:
        dict mapping end_position -> list of overhang dicts
    """
    grouped = defaultdict(list)
    
    for overhang in overhangs:
        end = overhang.get('end')
        if end and 1 <= end <= protein_length:
            grouped[end].append(overhang)
    
    return grouped


def parse_mass_data_from_csv(csv_file, protein_id=None):
    """
    Parse mass, intensity, and quality score data from CSV for per-residue mass calculation.
    Returns:
        mass_data: dict mapping row_index -> {
            'mass': M, 
            'intensity': I, 
            'peptide': seq,
            'retention_time': rt,
            'fragment_ions': list of ion names,
            'fragment_intensities': list of intensities,
            'fragment_quality_scores': list of quality scores (0-1, higher is better)
        }
        Dtotalpepn: dict mapping (retention_time, sequence_identity) -> D_total
    """
    mass_data = {}
    Dtotalpepn = {}  # Dict mapping (retention_time, sequence_identity) -> D_total
    
    with open(csv_file, 'r') as f:
        lines = f.readlines()
        
        if len(lines) < 2:
            return mass_data, Dtotalpepn
        
        header_line = lines[1]
        header = [col.strip() for col in header_line.split(',')]
        
        try:
            exp_mass_idx = header.index('exp_neutral_mass')
            intensity_idx = header.index('matched fragment ion intensities')
            calc_mass_idx = header.index('calc_neutral_mass')
            peptide_idx = header.index('plain_peptide')
            ions_idx = header.index('matched fragment ions')
            if 'comet_matched_frags_quality_scores' in header:
                quality_idx = header.index('comet_matched_frags_quality_scores')
            else:
                quality_idx = header.index('matched fragment ion quality scores')
            rt_idx = header.index('retention_time_sec')
        except ValueError as e:
            print(f"Warning: Required columns not found: {e}, using placeholder values")
            return mass_data, Dtotalpepn
        
        # Parse data rows (starting from line 2)
        for row_idx, line in enumerate(lines[2:], start=0):
            if not line.strip():
                continue
            
            # Parse CSV row
            row_data = []
            in_quotes = False
            current_field = ""
            for char in line:
                if char == '"':
                    in_quotes = not in_quotes
                elif char == ',' and not in_quotes:
                    row_data.append(current_field.strip())
                    current_field = ""
                    continue
                current_field += char
            if current_field:
                row_data.append(current_field.strip())
            
            if len(row_data) <= max(exp_mass_idx, intensity_idx, calc_mass_idx, peptide_idx, ions_idx, quality_idx):
                continue
            
            try:
                exp_mass = float(row_data[exp_mass_idx]) if row_data[exp_mass_idx] else 0.0
                calc_mass = float(row_data[calc_mass_idx]) if row_data[calc_mass_idx] else 0.0
                intensity_str = row_data[intensity_idx].strip('"') if len(row_data) > intensity_idx else ""
                peptide_seq = row_data[peptide_idx].strip('"') if len(row_data) > peptide_idx else ""
                ions_str = row_data[ions_idx].strip('"') if len(row_data) > ions_idx else ""
                quality_str = row_data[quality_idx].strip('"') if len(row_data) > quality_idx else ""
                retention_time = float(row_data[rt_idx]) if len(row_data) > rt_idx and row_data[rt_idx] else 0.0
                
                # Parse fragment ions, intensities, and quality scores
                fragment_ions = []
                fragment_intensities = []
                fragment_quality_scores = []
                
                if ions_str and intensity_str and quality_str:
                    ions_list = [x.strip() for x in ions_str.split(',') if x.strip()]
                    intensity_list = [float(x.strip()) for x in intensity_str.split(',') if x.strip()]
                    quality_list = [float(x.strip()) for x in quality_str.split(',') if x.strip()]
                    
                    # Match up the lists (they should have same length)
                    n_fragments = min(len(ions_list), len(intensity_list), len(quality_list))
                    fragment_ions = ions_list[:n_fragments]
                    fragment_intensities = intensity_list[:n_fragments]
                    fragment_quality_scores = quality_list[:n_fragments]
                
                # Parse intensity (sum of all fragment intensities)
                total_intensity = sum(fragment_intensities) if fragment_intensities else 0.0
                
                # Use intensity as mass proxy if available, otherwise use experimental mass
                mass_value = total_intensity if total_intensity > 0 else exp_mass
                
                # D_total: total deuterium on fragment - unique value for each peptide
                # This is a distinct variable that can be set independently for each peptide
                # For now, calculate as difference between exp_mass and calc_mass (deuterium uptake)
                # This represents the total deuterium uptake on the peptide fragment
                D_total = exp_mass - calc_mass if exp_mass > 0 and calc_mass > 0 else exp_mass
                # Ensure non-negative
                D_total = max(0.0, D_total)
                
                # Store in Dtotalpepn: key = (retention_time, sequence_identity)
                sequence_identity = peptide_seq
                n_key = (retention_time, sequence_identity)
                Dtotalpepn[n_key] = D_total
                
                mass_data[row_idx] = {
                    'mass': mass_value,
                    'exp_mass': exp_mass,
                    'calc_mass': calc_mass,
                    'intensity': total_intensity,
                    'peptide': peptide_seq,
                    'retention_time': retention_time,
                    'D_total': D_total,  # Store D_total as a distinct variable
                    'fragment_ions': fragment_ions,
                    'fragment_intensities': fragment_intensities,
                    'fragment_quality_scores': fragment_quality_scores
                }
            except (ValueError, IndexError) as e:
                continue
    
    return mass_data, Dtotalpepn


def create_per_residue_mass_heatmap(csv_file, protein_id, fasta_file, output_file, 
                                   ion_type_filter=None, filtered_peptide_indices=None, tau=None):
    """
    Create a position-anchored heatmap showing per-residue mass.
    
    For each overhang:
    - Total mass: M (from fragment intensities or experimental mass)
    - Overhang length: L (number of amino acids in overhang)
    - Per-residue mass: (1/L) * M
    
    Example: 
    - 8-AA overhang with mass M: each residue gets 0.125 * M (M/8)
    - 3-AA overhang with mass M: each residue gets 0.333 * M (M/3)
    
    Note: Scrambling correction would require D_total (total deuterium on fragment),
    which is not available in the data. Therefore, we use simple per-residue mass
    without scrambling correction.
    
    Grayscale colormap: white = 0 (no mass), black = max per-residue mass
    """
    # Parse FASTA
    sequences = parse_fasta(fasta_file)
    if not sequences:
        print("Error: No sequences found in FASTA file")
        return
    
    # Get protein sequence
    if protein_id:
        protein_sequence = None
        protein_id_clean = re.sub(r'(\d+)$', '', protein_id)
        protein_id_alt = re.sub(r'^sp\|', '', protein_id_clean)
        
        for seq_id, seq in sequences.items():
            seq_id_clean = re.sub(r'(\d+)$', '', seq_id)
            seq_id_alt = re.sub(r'^sp\|', '', seq_id_clean)
            if (seq_id == protein_id or seq_id_clean == protein_id_clean or
                seq_id_alt == protein_id_alt or seq_id == protein_id_alt or
                seq_id_clean == protein_id_alt):
                protein_sequence = seq
                protein_id = seq_id
                break
        
        if not protein_sequence:
            print(f"Error: Protein {protein_id} not found in FASTA file")
            return
    else:
        protein_id = list(sequences.keys())[0]
        protein_sequence = sequences[protein_id]
    
    protein_length = len(protein_sequence)
    print(f"Protein: {protein_id}, Length: {protein_length}")
    
    # Parse mass data
    print("Parsing mass and intensity data...")
    mass_data, Dtotalpepn = parse_mass_data_from_csv(csv_file, protein_id)
    print(f"Found mass data for {len(mass_data)} peptides")
    print(f"Created Dtotalpepn lookup with {len(Dtotalpepn)} entries")
    
    # Parse overhangs
    print("Parsing multi-AA overhangs...")
    multi_aa_overhangs = parse_multi_aa_overhangs(
        csv_file, protein_id=protein_id, fasta_file=fasta_file,
        filtered_peptide_indices=filtered_peptide_indices
    )
    
    # Filter by ion type if specified
    if ion_type_filter:
        filtered_overhangs = []
        for overhang in multi_aa_overhangs:
            ion_series = overhang.get('ion_series', '')
            if ion_type_filter == 'c' and ion_series == 'c':
                filtered_overhangs.append(overhang)
            elif ion_type_filter == 'z' and ion_series in ['z', 'z+1']:
                filtered_overhangs.append(overhang)
        multi_aa_overhangs = filtered_overhangs
        print(f"Filtered to {ion_type_filter} ions: {len(multi_aa_overhangs)} overhangs")
    
    print(f"Found {len(multi_aa_overhangs)} multi-AA overhangs")
    
    # Also parse single-AA overhangs
    print("Parsing single-AA overhangs...")
    single_aa_overhangs = parse_single_aa_overhangs(
        csv_file, protein_id=protein_id, fasta_file=fasta_file
    )
    
    # Convert single-AA to same format as multi-AA
    for overhang in single_aa_overhangs:
        multi_aa_overhangs.append(overhang)
    
    # Group overhangs by start and end positions
    grouped_forward = group_overhangs_by_start_position(multi_aa_overhangs, protein_length)
    grouped_reverse = group_overhangs_by_end_position(multi_aa_overhangs, protein_length)
    
    # Create mapping from peptide sequence to mass data
    peptide_to_mass = {}
    for row_idx, data in mass_data.items():
        peptide_seq = data.get('peptide', '')
        if peptide_seq:
            peptide_to_mass[peptide_seq] = data
    
    # Need to match overhangs to CSV rows to get mass data
    # Parse CSV to get row index for each peptide
    peptide_to_row_idx = {}
    with open(csv_file, 'r') as f:
        lines = f.readlines()
        if len(lines) >= 2:
            header_line = lines[1]
            header = [col.strip() for col in header_line.split(',')]
            try:
                peptide_idx = header.index('plain_peptide')
                for row_idx, line in enumerate(lines[2:], start=0):
                    if not line.strip():
                        continue
                    # Parse CSV row
                    row_data = []
                    in_quotes = False
                    current_field = ""
                    for char in line:
                        if char == '"':
                            in_quotes = not in_quotes
                        elif char == ',' and not in_quotes:
                            row_data.append(current_field.strip())
                            current_field = ""
                            continue
                        current_field += char
                    if current_field:
                        row_data.append(current_field.strip())
                    if len(row_data) > peptide_idx:
                        peptide_seq = row_data[peptide_idx].strip('"')
                        if peptide_seq:
                            peptide_to_row_idx[peptide_seq] = row_idx
            except ValueError:
                pass
    
    # Calculate per-residue mass percentages
    # Structure: pos -> list of (row_idx, value) tuples for each overhang at that position
    per_residue_mass_forward = defaultdict(list)  # pos -> [(row_idx, value), ...]
    per_residue_mass_reverse = defaultdict(list)  # pos -> [(row_idx, value), ...]
    
    # Step 1: Assign binary 0/1 to each residue in the protein sequence (for visualization)
    # This is the "synthetic sequence" assignment
    import random
    random.seed(42)  # For reproducibility
    residue_deuteration = {}  # protein position -> 0 or 1
    for pos in range(1, protein_length + 1):
        residue_deuteration[pos] = random.choice([0, 1])  # Binary assignment
    
    # Step 2: Calculate D_total for each unique peptide by summing binary values
    # Collect all unique peptides from overhangs
    unique_peptides = set()
    for overhang in multi_aa_overhangs:
        peptide_seq = overhang.get('peptide_seq', '')
        clean_peptide = re.sub(r'\[.*?\]', '', peptide_seq) if peptide_seq else ''
        if clean_peptide:
            unique_peptides.add(clean_peptide)
    
    # Calculate D_total for each peptide by summing binary residue values
    peptide_D_total_map = {}
    protein_seq_upper = protein_sequence.upper()
    for peptide in unique_peptides:
        clean_peptide_upper = peptide.upper()
        peptide_start_in_protein = protein_seq_upper.find(clean_peptide_upper)
        
        if peptide_start_in_protein >= 0:
            D_total = 0.0
            protein_start_offset = peptide_start_in_protein + 1
            for i in range(len(peptide)):
                protein_pos = protein_start_offset + i
                if 1 <= protein_pos <= protein_length:
                    D_total += residue_deuteration.get(protein_pos, 0.0)
            peptide_D_total_map[peptide] = D_total
        else:
            # Peptide not found in protein, use default
            peptide_D_total_map[peptide] = 0.0
    
    print(f"Calculated D_total for {len(peptide_D_total_map)} unique peptides from binary assignments")
    
    # Estimate τ (within-segment heterogeneity parameter) if not provided
    if tau is None:
        print("Estimating τ from single-AA overhangs and overlap inconsistency...")
        tau_single_aa = estimate_tau_from_single_aa_overhangs(single_aa_overhangs, peptide_D_total_map, protein_sequence)
        tau_overlap = estimate_tau_from_overlap_inconsistency(multi_aa_overhangs, peptide_D_total_map, protein_sequence)
        # Use the average of the two estimates, or prefer single-AA if available
        if tau_single_aa > 0.01:  # If we got a meaningful estimate from single-AA
            tau = tau_single_aa
        else:
            tau = tau_overlap
        print(f"Estimated τ = {tau:.4f}")
    else:
        print(f"Using provided τ = {tau:.4f}")
    
    # Note: For single-AA overhangs, the single residue can be any value 0-1,
    # and the complementary fragment gets the remainder such that they sum to D_total.
    # This is handled implicitly in the fractional error calculation.
    
    # Process each overhang
    for row_idx, overhang in enumerate(multi_aa_overhangs):
        start = overhang.get('start')
        end = overhang.get('end')
        length = overhang.get('length', end - start + 1 if start and end else 1)
        peptide_seq = overhang.get('peptide_seq', '')
        fragment_pairs = overhang.get('fragment_pairs', [])
        peptide_start = overhang.get('peptide_start', 1)
        peptide_end = overhang.get('peptide_end', length)
        
        # Find corresponding mass data by matching peptide sequence to CSV row
        mass_M = 1000.0  # Default mass
        # Get peptide length from the peptide sequence
        # IMPORTANT: peptide_length is the full peptide length, not the overhang length
        clean_peptide = re.sub(r'\[.*?\]', '', peptide_seq) if peptide_seq else ''
        if clean_peptide:
            peptide_length = len(clean_peptide)
        else:
            # If peptide_seq is not available, try to get it from mass_data
            peptide_length = None
            if csv_row_idx is not None and csv_row_idx in mass_data:
                data = mass_data[csv_row_idx]
                peptide_from_data = data.get('peptide', '')
                if peptide_from_data:
                    clean_peptide = re.sub(r'\[.*?\]', '', peptide_from_data)
                    peptide_length = len(clean_peptide) if clean_peptide else None
            
            # Last resort: use a reasonable default (don't use overhang length!)
            # Using overhang length would give wrong fractional error (too high)
            if peptide_length is None or peptide_length == 0:
                peptide_length = 15  # Reasonable default for peptide length
        
        # Try to match by peptide sequence to get CSV row index
        csv_row_idx = None
        if peptide_seq in peptide_to_row_idx:
            csv_row_idx = peptide_to_row_idx[peptide_seq]
            if csv_row_idx in mass_data:
                data = mass_data[csv_row_idx]
                mass_M = data.get('mass', 1000.0)
        elif peptide_seq in peptide_to_mass:
            # Fallback: try direct peptide match
            data = peptide_to_mass[peptide_seq]
            mass_M = data.get('mass', 1000.0)
        elif mass_data:
            # Use first available mass data as fallback
            first_data = list(mass_data.values())[0]
            mass_M = first_data.get('mass', 1000.0)
        
        # Get D_total for this peptide (stable after digestion)
        # D_total is determined per peptide, not by summing residue values
        D_total = 0.0
        if peptide_seq and clean_peptide:
            # Get D_total from the peptide map
            D_total = peptide_D_total_map.get(clean_peptide, 0.0)
        
        # Calculate CV on total overhang D (K), not per-residue, to avoid tiny denominator issues
        # K = number of deuteriums in the overhang (random variable under scrambling)
        # E[K] = D_total * L/N  (expected D's in overhang)
        # Var(K) = D_total * (L/N) * (1 - L/N) * ((N - D_total) / (N - 1))  (hypergeometric variance)
        # SD(K) = sqrt(Var(K))
        # CV_K = SD(K) / E[K]  (coefficient of variation)
        
        overhang_length = length  # Get overhang length from the overhang data
        if peptide_length > 0 and overhang_length > 0:
            L = float(overhang_length)
            N = float(peptide_length)
            
            def compute_cv_overhang_total(N, L, D_total, scrambling_fraction=0.50, tau=None):
                """
                Calculate coefficient of variation for total overhang D (K) using τ-based heterogeneity model.
                
                Final uncertainty formula:
                - E_K = expected total D in overhang (S) [units: D]
                - σ_scr = SD from scrambling [units: D]
                - τ = resolution (localization) heterogeneity scale [units: D/res]
                - σ_res(L) = 0 if L=1, τ√L if L>1 [units: D]
                - σ_total = √(σ_scr² + σ_res²) [units: D]
                - CV = σ_total / E_K [dimensionless]
                
                Sanity checks:
                1. For long overhangs with tiny scrambling: σ_total ≈ τ√L, and since E_K ∝ L,
                   CV ≈ τ√L/E_K scales like ~1/√L (nice property: longer overhangs have lower CV)
                2. For L=1: σ_res = 0, so CV collapses to scrambling-only (no artificial penalty)
                
                Parameters:
                -----------
                scrambling_fraction : float
                    Fraction of deuterium that undergoes scrambling (default 0.50 = 50%).
                    Only the scrambled fraction contributes to uncertainty.
                tau : float, optional
                    Resolution (localization) heterogeneity scale in D/res.
                    If None, uses default value of 0.1.
                    Should be chosen via empirical overlap calibration or as a conservative fixed value.
                """
                if L <= 0 or N <= 1:
                    return np.nan
                
                D = np.clip(D_total, 0, N)
                p = L / N  # Probability a D is in the overhang
                
                # Expected number of D's in overhang (total D, not per-residue)
                E_K = D * p  # This is S (total D in overhang)
                
                if E_K <= 0:
                    return 0.0
                
                # Scrambling uncertainty: σ_scr (in units of D, same as E_K)
                # Variance of K (hypergeometric) - but scaled by scrambling fraction
                var_K_scrambling = D * p * (1 - p) * ((N - D) / (N - 1)) * scrambling_fraction
                var_K_scrambling = max(var_K_scrambling, 0)  # Ensure non-negative
                sigma_scr = np.sqrt(var_K_scrambling)
                
                # Resolution uncertainty: σ_res(L)
                # σ_res(L) = 0 if L=1, τ√L if L>1 (for total D form)
                if tau is None:
                    tau = 0.1  # Default τ value
                
                if L == 1:
                    sigma_res = 0.0  # No resolution uncertainty for single-AA overhangs
                else:
                    sigma_res = tau * np.sqrt(L)  # For total D form: σ_res = τ√L
                
                # Fix B: Cap resolution uncertainty by what's physically possible
                # σ_res ≤ c·E_K prevents CV explosion solely due to resolution
                resolution_cap_factor = 0.5
                sigma_res = min(sigma_res, resolution_cap_factor * E_K)
                
                # Total uncertainty: σ_total = √(σ_scr² + σ_res²)
                sigma_total = np.sqrt(sigma_scr**2 + sigma_res**2)
                
                # Fix A: Floor on usable signal
                # If E_K is too small, CV is not meaningful (essentially no signal)
                E_min = 0.2  # Minimum expected D for reliable quantification
                if E_K < E_min:
                    return np.nan  # Not reliably quantified
                
                # Coefficient of variation: CV = σ_total / E_K
                CV_K = sigma_total / E_K if E_K > 0 else 0.0
                
                return CV_K
            
            # Use 50% scrambling (0.50) instead of 11% scrambling
            # Use τ-based heterogeneity model (Option A)
            cv_overhang = compute_cv_overhang_total(N, L, D_total, scrambling_fraction=0.50, tau=tau)
            
            if not np.isnan(cv_overhang):
                # Don't cap CV - high values (>100%) are valid and meaningful
                # They indicate very high relative uncertainty when D_total is small
                # Use a reasonable upper bound for visualization (e.g., 500% = 5.0)
                # Values above this will be clipped for display but the calculation is correct
                cv_overhang = min(cv_overhang, 5.0)  # Cap at 500% for visualization
                value = cv_overhang * 100.0  # Convert to percentage
            else:
                value = 0.0
        else:
            value = 0.0
        
        # Store for forward overhangs (N-term anchored) - anchored at start position
        if start and 1 <= start <= protein_length:
            # Store value for the anchor position (start)
            per_residue_mass_forward[start].append((row_idx, value))
        
        # Store for reverse overhangs (C-term anchored) - anchored at end position
        if end and 1 <= end <= protein_length:
            # Store value for the anchor position (end)
            per_residue_mass_reverse[end].append((row_idx, value))
    
    # Create figure with same layout as single-row plot
    # Add one extra row at the top for the sequence display
    n_rows = 10
    positions_per_row = int(np.ceil(protein_length / n_rows))
    n_cols = positions_per_row * 2  # Each position has 2 columns (reverse, forward)
    
    # Limit figure size to prevent memory issues and ensure image can be loaded
    # Use more reasonable dimensions - scale down if needed
    base_width = min(n_cols * 1.5, 50)  # Reduced from 3.0 to 1.5, max 50 inches
    base_height = min(n_rows * 3.0 + 0.5, 50)  # Reduced from 6.0 to 3.0, max 50 inches
    
    # Create figure with extra space at top for sequence display
    fig = plt.figure(figsize=(base_width, base_height))
    gs = gridspec.GridSpec(n_rows + 1, n_cols, figure=fig, hspace=0.8, wspace=0.02,
                          height_ratios=[0.3] + [1.0] * n_rows)  # Top row is smaller for sequence
    
    # Add sequence display at the top
    ax_seq = fig.add_subplot(gs[0, :])  # Span all columns
    ax_seq.set_xlim(0, protein_length)
    ax_seq.set_ylim(-0.5, 0.5)
    ax_seq.axis('off')  # Hide axes
    
    # Display sequence with H/D assignments
    for pos in range(1, protein_length + 1):
        aa = protein_sequence[pos - 1] if pos <= len(protein_sequence) else "?"
        deuteration = residue_deuteration.get(pos, 0)
        label = "D" if deuteration > 0 else "H"
        
        # Position in plot (centered on position)
        x_pos = pos - 0.5
        
        # Draw background rectangle (green for D, light gray for H)
        color = 'lightgreen' if deuteration > 0 else 'lightgray'
        rect = plt.Rectangle((pos - 1, -0.3), 1, 0.6, 
                           facecolor=color, edgecolor='black', linewidth=0.5)
        ax_seq.add_patch(rect)
        
        # Add amino acid letter
        ax_seq.text(x_pos, 0.15, aa, fontsize=16, ha='center', va='center', fontweight='bold')
        
        # Add H/D label below
        ax_seq.text(x_pos, -0.15, label, fontsize=14, ha='center', va='center', fontweight='bold')
        
        # Add position number at top
        if pos % 5 == 0 or pos == 1:  # Show every 5th position and first
            ax_seq.text(x_pos, 0.4, str(pos), fontsize=12, ha='center', va='bottom', fontweight='bold')
    
    ax_seq.set_title('Protein Sequence with Random Deuteration Assignment (H=0, D=1)', 
                    fontsize=18, fontweight='bold', pad=15)
    
    # Calculate min/max values for CV
    all_values = []
    for pos_data in per_residue_mass_forward.values():
        for row_idx, val in pos_data:
            all_values.append(val)
    for pos_data in per_residue_mass_reverse.values():
        for row_idx, val in pos_data:
            all_values.append(val)
    
    if all_values:
        min_value = min(all_values)
        max_value = max(all_values)
        # For visualization, cap max at 500% but note if values exceed this
        display_max = min(max_value, 500.0)
        if max_value > 500.0:
            print(f"CV range: min={min_value:.2f}%, max={max_value:.2f}% (capped at {display_max:.2f}% for visualization)")
        else:
            print(f"CV range: min={min_value:.2f}%, max={max_value:.2f}%")
        max_value = display_max
    else:
        min_value = 0.0
        max_value = 100.0
        print(f"No values found, using default range: {min_value}% to {max_value}%")
    
    # Create custom colormap: white for no coverage, then navy blue -> crimson -> gold for CV
    # Values represent CV (coefficient of variation) computed on total overhang D (K)
    # white = no coverage (NaN/zero areas)
    # color map = CV (min to max): navy blue -> crimson -> gold
    from matplotlib.colors import LinearSegmentedColormap
    
    # Define color stops: navy blue -> crimson -> gold (matching fractional_error_times_count_coverage_heatmap)
    colors = [
        (0.0, 0.0, 0.5),       # Navy blue
        (0.863, 0.078, 0.235), # Crimson
        (1.0, 0.843, 0.0)      # Gold
    ]
    
    # Create colormap with these colors
    n_bins = 256
    custom_color_map = LinearSegmentedColormap.from_list('navy_blue_crimson_gold', colors, N=n_bins)
    
    # Create custom colormap: white first, then color map
    colormap_colors = []
    # First color: white for no coverage
    colormap_colors.append((1.0, 1.0, 1.0))  # White for no coverage
    # Then color map: get 256 colors from custom gradient
    for i in range(256):
        color = custom_color_map(i / 255.0)  # Get color from custom gradient (0 to 1)
        colormap_colors.append(color[:3])  # Take RGB, ignore alpha
    grayscale_cmap = ListedColormap(colormap_colors)
    
    # We'll use a sentinel value for no-coverage areas, which will map to the first color (white)
    # Since values are now in percentage (0-100 range), use -10 as sentinel
    NO_COVERAGE_VALUE = -10.0  # Sentinel value for no coverage (will be below min_value)
    
    # Create normalization that maps:
    # - Values < min_value -> white (index 0)
    # - min_value -> start of color map (index 1)
    # - max_value -> end of color map (index 256)
    # The colormap has 257 colors: index 0 = white, indices 1-256 = color map
    # We need to map [min_value, max_value] to normalized range [1/257, 1.0]
    # This ensures min_value maps to start of color map and max_value maps to end
    from matplotlib.colors import Normalize
    
    class CustomNormalize(Normalize):
        """Custom normalization that maps min_value to start of color map and max_value to end."""
        def __init__(self, vmin, vmax, no_coverage_value):
            # Set vmin slightly below min_value to allow clipping of no-coverage
            # But we'll override __call__ to map correctly
            super().__init__(vmin=vmin, vmax=vmax)
            self.no_coverage_value = no_coverage_value
            self.data_min = vmin
            self.data_max = vmax
        
        def __call__(self, value, clip=None):
            # Handle no-coverage values: map to 0 (white)
            if isinstance(value, np.ndarray):
                result = np.zeros_like(value, dtype=float)
                mask = value == self.no_coverage_value
                result[mask] = 0.0  # White
                # For data values, map [min_value, max_value] to [1/257, 1.0]
                data_mask = ~mask
                if np.any(data_mask):
                    data_values = value[data_mask]
                    # Linear mapping: (val - min) / (max - min) maps to [0, 1]
                    # Then shift: [0, 1] -> [1/257, 1.0]
                    normalized = (data_values - self.data_min) / (self.data_max - self.data_min)
                    result[data_mask] = (1.0 / 257.0) + normalized * (1.0 - 1.0 / 257.0)
                return np.clip(result, 0, 1)
            else:
                if value == self.no_coverage_value:
                    return 0.0  # White
                normalized = (value - self.data_min) / (self.data_max - self.data_min)
                return np.clip((1.0 / 257.0) + normalized * (1.0 - 1.0 / 257.0), 0, 1)
    
    custom_norm = CustomNormalize(vmin=min_value, vmax=max_value, no_coverage_value=NO_COVERAGE_VALUE)
    
    # Process each position
    # Note: grid_row is offset by 1 because row 0 is the sequence display
    for pos in range(1, protein_length + 1):
        grid_row = (pos - 1) // positions_per_row + 1  # +1 to account for sequence row
        col_in_row = (pos - 1) % positions_per_row
        col_reverse = col_in_row * 2
        col_forward = col_in_row * 2 + 1
        
        # Get overhangs for this position
        overhangs_reverse = grouped_reverse.get(pos, [])
        overhangs_forward = grouped_forward.get(pos, [])
        
        # Reverse subplot (C-term)
        ax_reverse = fig.add_subplot(gs[grid_row, col_reverse])
        if overhangs_reverse and pos in per_residue_mass_reverse:
            # Get unique row indices and their values, along with overhang info
            # For CV, use the first value (don't sum - each overhang has its own CV)
            row_data = []  # List of (row_idx, value, overhang) tuples
            for row_idx, value in per_residue_mass_reverse[pos]:
                # Check if we've already seen this row_idx
                if not any(rd[0] == row_idx for rd in row_data):
                    overhang = None
                    if row_idx < len(multi_aa_overhangs):
                        overhang = multi_aa_overhangs[row_idx]
                    row_data.append((row_idx, value, overhang))
            
            # Sort by overhang length (ascending), then by CV (descending) within same length
            # Descending so highest CV appears at top (row 0), lowest at bottom
            row_data.sort(key=lambda x: (x[2].get('length', 0) if x[2] else 0, -x[1]))
            
            num_rows = len(row_data)
            if num_rows > 0:
                max_length = max(oh.get('length', 1) for oh in overhangs_reverse) + 1
                # Initialize with NO_COVERAGE_VALUE (blue) instead of zero
                heatmap = np.full((num_rows, max_length), NO_COVERAGE_VALUE, dtype=float)
                
                # Create y-coordinates for linear scale (C-term uses linear)
                y_coords = np.linspace(0, num_rows, num_rows + 1)
                x_coords = np.arange(max_length + 1)
                
                # Store peptide sequences for y-axis labels
                y_labels = []
                
                for i, (row_idx, value, overhang) in enumerate(row_data):
                    # Get overhang length to create wedge shape
                    overhang_length = overhang.get('length', 1) if overhang else 1
                    peptide_seq = overhang.get('peptide_seq', '') if overhang else ''
                    
                    # Create y-axis label: peptide sequence (truncated if too long)
                    if peptide_seq:
                        # Clean peptide sequence (remove modifications)
                        clean_seq = re.sub(r'\[.*?\]', '', peptide_seq)
                        # Truncate if too long (max 30 chars for readability)
                        if len(clean_seq) > 30:
                            clean_seq = clean_seq[:27] + "..."
                        y_labels.append(clean_seq)
                    else:
                        y_labels.append(f"Row {row_idx}")
                    
                    # CV is now in percentage (0-100 range)
                    # Only fill columns up to overhang_length to create wedge shape
                    # Fill from right (C-term) - so fill last 'overhang_length' columns
                    if overhang_length > 0 and overhang_length <= max_length:
                        # For C-term: fill from the right (last columns)
                        start_col = max_length - overhang_length
                        heatmap[i, start_col:] = value
                    else:
                        # Fallback: fill entire row if length is invalid
                        heatmap[i, :] = value
                
                # Use custom normalization: min_value to max_value for color map
                # NO_COVERAGE_VALUE will map to white (first color)
                # Don't flip - keep same x-direction as N-term
                im = ax_reverse.pcolormesh(x_coords, y_coords, heatmap, 
                                          cmap=grayscale_cmap, norm=custom_norm, shading='flat')
                ax_reverse.set_xlim(-0.5, max_length - 0.5)
                # Extend y-axis to accommodate text at top
                ax_reverse.set_ylim(0, num_rows + 2)
                ax_reverse.set_yscale('linear')  # Linear scale for C-term
                ax_reverse.invert_yaxis()
                
                # Set y-axis labels (hide them, we'll show at top)
                y_tick_positions = [i + 0.5 for i in range(num_rows)]
                ax_reverse.set_yticks(y_tick_positions)
                ax_reverse.set_yticklabels([''] * num_rows)  # Hide y-axis labels
                
                # Set x-axis labels showing sequence positions (1, 2, 3, ...)
                x_tick_positions = list(range(max_length))
                x_tick_labels = [str(i + 1) for i in range(max_length)]
                ax_reverse.set_xticks(x_tick_positions)
                ax_reverse.set_xticklabels(x_tick_labels, fontsize=8)
                ax_reverse.tick_params(axis='x', which='major', length=4, width=1)
                
                # Add peptide sequences at the top of the plot
                # Position them above the heatmap (y = num_rows + 1)
                top_y = num_rows + 1
                for i, label in enumerate(y_labels):
                    # Position labels across the width, wrapping if needed
                    x_pos = (i % max_length) * (max_length / min(num_rows, max_length))
                    ax_reverse.text(x_pos, top_y, label, fontsize=4, 
                                   rotation=0, ha='left', va='bottom',
                                   bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.8, edgecolor='gray', linewidth=0.5))
        
        aa = protein_sequence[pos-1] if pos <= len(protein_sequence) else "?"
        ax_reverse.set_title(f'{pos}:{aa}\n(C-term)', fontsize=20, pad=1, fontweight='bold')
        if not (overhangs_reverse and pos in per_residue_mass_reverse):
            ax_reverse.set_xticks([])
            ax_reverse.set_yticks([])
        
        # Forward subplot (N-term)
        ax_forward = fig.add_subplot(gs[grid_row, col_forward])
        if overhangs_forward and pos in per_residue_mass_forward:
            # Get unique row indices and their values, along with overhang info
            # For CV, use the first value (don't sum - each overhang has its own CV)
            row_data = []  # List of (row_idx, value, overhang) tuples
            for row_idx, value in per_residue_mass_forward[pos]:
                # Check if we've already seen this row_idx
                if not any(rd[0] == row_idx for rd in row_data):
                    overhang = None
                    if row_idx < len(multi_aa_overhangs):
                        overhang = multi_aa_overhangs[row_idx]
                    row_data.append((row_idx, value, overhang))
            
            # Sort by overhang length (ascending), then by CV (descending) within same length
            # Descending so highest CV appears at top (row 0), lowest at bottom
            row_data.sort(key=lambda x: (x[2].get('length', 0) if x[2] else 0, -x[1]))
            
            num_rows = len(row_data)
            if num_rows > 0:
                max_length = max(oh.get('length', 1) for oh in overhangs_forward) + 1
                # Initialize with NO_COVERAGE_VALUE (blue) instead of zero
                heatmap = np.full((num_rows, max_length), NO_COVERAGE_VALUE, dtype=float)
                
                # Create y-coordinates for linear scale (same as C-term for alignment)
                y_coords = np.linspace(0, num_rows, num_rows + 1)
                x_coords = np.arange(max_length + 1)
                
                # Store peptide sequences for y-axis labels
                y_labels = []
                
                for i, (row_idx, value, overhang) in enumerate(row_data):
                    # Get overhang length to create wedge shape
                    overhang_length = overhang.get('length', 1) if overhang else 1
                    peptide_seq = overhang.get('peptide_seq', '') if overhang else ''
                    
                    # Create y-axis label: peptide sequence (truncated if too long)
                    if peptide_seq:
                        # Clean peptide sequence (remove modifications)
                        clean_seq = re.sub(r'\[.*?\]', '', peptide_seq)
                        # Truncate if too long (max 30 chars for readability)
                        if len(clean_seq) > 30:
                            clean_seq = clean_seq[:27] + "..."
                        y_labels.append(clean_seq)
                    else:
                        y_labels.append(f"Row {row_idx}")
                    
                    # CV is now in percentage (0-100 range)
                    # Only fill columns up to overhang_length to create wedge shape
                    # Fill from left (N-term) - so fill first 'overhang_length' columns
                    if overhang_length > 0 and overhang_length <= max_length:
                        # For N-term: fill from the left (first columns)
                        heatmap[i, :overhang_length] = value
                    else:
                        # Fallback: fill entire row if length is invalid
                        heatmap[i, :] = value
                
                # Use custom normalization: min_value to max_value for color map
                # NO_COVERAGE_VALUE will map to white (first color)
                # Don't flip - keep same x-direction as C-term
                im = ax_forward.pcolormesh(x_coords, y_coords, heatmap,
                                          cmap=grayscale_cmap, norm=custom_norm, shading='flat')
                ax_forward.set_xlim(-0.5, max_length - 0.5)
                # Extend y-axis to accommodate text at top
                ax_forward.set_ylim(0, num_rows + 2)
                ax_forward.set_yscale('linear')  # Linear scale to match C-term
                ax_forward.invert_yaxis()  # Invert to match C-term orientation
                
                # Set y-axis labels (hide them, we'll show at top)
                y_tick_positions = [i + 0.5 for i in range(num_rows)]
                ax_forward.set_yticks(y_tick_positions)
                ax_forward.set_yticklabels([''] * num_rows)  # Hide y-axis labels
                
                # Set x-axis labels showing sequence positions (1, 2, 3, ...)
                x_tick_positions = list(range(max_length))
                x_tick_labels = [str(i + 1) for i in range(max_length)]
                ax_forward.set_xticks(x_tick_positions)
                ax_forward.set_xticklabels(x_tick_labels, fontsize=8)
                ax_forward.tick_params(axis='x', which='major', length=4, width=1)
                
                # Add peptide sequences at the top of the plot
                # Position them above the heatmap (y = num_rows + 1)
                top_y = num_rows + 1
                for i, label in enumerate(y_labels):
                    # Position labels across the width, wrapping if needed
                    x_pos = (i % max_length) * (max_length / min(num_rows, max_length))
                    ax_forward.text(x_pos, top_y, label, fontsize=4, 
                                   rotation=0, ha='left', va='bottom',
                                   bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.8, edgecolor='gray', linewidth=0.5))
        
        ax_forward.set_title(f'{pos}:{aa}\n(N-term)', fontsize=20, pad=1, fontweight='bold')
        if not (overhangs_forward and pos in per_residue_mass_forward):
            ax_forward.set_xticks([])
            ax_forward.set_yticks([])
    
    # Add colorbar with proper scaling
    from matplotlib.colorbar import Colorbar
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    sm = ScalarMappable(cmap=grayscale_cmap, norm=custom_norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=fig.axes, orientation='horizontal', 
                        pad=0.02, shrink=0.6, aspect=30, location='top')
    cbar.set_label(f'CV (Overhang Total) / √(Fragment Count) - Uncertainty of the Mean (%)\nWhite=No Coverage, Range: {min_value:.2f}% to {max_value:.2f}%\nLower values = more trustworthy average signal', 
                   fontsize=24, fontweight='bold')
    cbar.ax.tick_params(labelsize=20)
    
    fig.suptitle(f'CV (Overhang Total) / √(Fragment Count) - Position-Anchored Overhang Heatmaps\n'
                 f'Protein: {protein_id} ({protein_length} residues) | Left: Ending at Position (C-term) | Right: Starting at Position (N-term)',
                 fontsize=12, fontweight='bold', y=0.98)
    
    # Reduce DPI to keep image size manageable (target: < 10,000 pixels per dimension)
    # Calculate max DPI based on figure size to keep total pixels reasonable
    max_pixels_per_dim = 8000  # Target max dimension in pixels
    calculated_dpi = min(max_pixels_per_dim / max(base_width, base_height), 100)
    dpi = max(int(calculated_dpi), 50)  # Minimum 50 DPI, maximum 100 DPI
    
    try:
        plt.savefig(output_file, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"CV (overhang total) heatmap saved to: {output_file} (DPI: {dpi}, size: {base_width:.1f}x{base_height:.1f} inches)")
    except Exception as e:
        print(f"Error saving figure: {e}")
        # Try saving with a simpler format
        try:
            plt.savefig(output_file, dpi=75, bbox_inches='tight')
            print(f"Figure saved with fallback settings to: {output_file}")
        except Exception as e2:
            print(f"Failed to save figure: {e2}")
            raise
    finally:
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Create position-anchored overhang heatmaps',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--csv', required=True, help='Comet CSV output file')
    parser.add_argument('--fasta', required=True, help='FASTA file with protein sequences')
    parser.add_argument('--output', required=True, help='Output PNG file')
    parser.add_argument('--protein', help='Specific protein ID (optional, uses first if not specified)')
    parser.add_argument('--c-ions-only', action='store_true', dest='c_ions_only',
                       help='Show only c-ion overhangs')
    parser.add_argument('--z-ions-only', action='store_true', dest='z_ions_only',
                       help='Show only z/z+1-ion overhangs')
    parser.add_argument('--per-residue-mass', action='store_true',
                       help='Create per-residue mass percentage heatmap with grayscale colormap')
    parser.add_argument('--tau', type=float, default=None,
                       help='Within-segment heterogeneity parameter τ (SD of per-residue deuteration). '
                            'If not provided, will be estimated from data.')
    
    args = parser.parse_args()
    
    # Determine ion type filter
    ion_type_filter = None
    if args.c_ions_only:
        ion_type_filter = 'c'
    elif args.z_ions_only:
        ion_type_filter = 'z'
    
    # Create visualization
    if args.per_residue_mass:
        create_per_residue_mass_heatmap(
            args.csv, args.protein, args.fasta, args.output,
            ion_type_filter=ion_type_filter,
            tau=args.tau
        )
    else:
        print("Error: Please specify a visualization type. Use --per-residue-mass for per-residue mass heatmap.")
        print("Note: Other visualization functions (--single-row, --combined, etc.) are not yet implemented in this version.")


if __name__ == '__main__':
    main()
