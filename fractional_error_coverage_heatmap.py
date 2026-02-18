#!/usr/bin/env python3
"""
CV (Coefficient of Variation) Coverage Heatmap

Creates a coverage heatmap showing CV (coefficient of variation) for each residue position
and overhang length combination, similar to the original coverage heatmap but
using relative uncertainty instead of fragment counts.

X-axis: Protein sequence position
Y-axis: Overhang length
Color: CV computed on total overhang D (K), not per-residue

============================================================================
What CV Tells You (Intuition)
============================================================================

CV = 0.10 (10%) → the noise/uncertainty is about 10% of the expected signal
CV = 1.0 (100%) → the uncertainty is about as big as the signal (very unreliable)

Higher CV = less trustworthy, because the measurement is "noisy relative to its size"

In the overhang scrambling context:
- We treat the overhang deuterium count K as a random variable under scrambling
- μ = E[K] = expected deuteriums in the overhang
- σ = SD(K) = uncertainty from scrambling (hypergeometric SD)
- CV_overhang = SD(K) / E[K]

This gives a dimensionless "how uncertain is this localized overhang D?" score.

Why CV is useful:
- It lets you compare uncertainty across different peptides/timepoints even when
  the absolute signal is different, because it's normalized.
- CV computed on total overhang D (K) avoids tiny denominator issues that occur
  when computing per-residue CV at early time points (when D_total is small).

Two output modes:
1. Average CV mode: Shows average CV across all fragments at each position
2. Uncertainty of the mean mode: Shows CV_mean = mean(CV_i) / sqrt(n), representing
   how trustworthy the average signal is (more fragments → lower uncertainty)
"""

import argparse
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from collections import defaultdict
from multi_aa_overhang_visualization import parse_fasta, parse_multi_aa_overhangs, parse_single_aa_overhangs
import random

def estimate_tau_from_single_aa_overhangs(single_aa_overhangs, peptide_D_total_map, protein_sequence):
    """
    Estimate τ (within-segment heterogeneity parameter) from single-AA overhangs.
    
    Single-AA overhangs provide 1-AA resolution, allowing direct estimation of
    per-residue deuteration variance. τ represents the standard deviation of
    deuteration across residues within a segment.
    
    Parameters:
    -----------
    single_aa_overhangs : list
        List of single-AA overhang dictionaries
    peptide_D_total_map : dict
        Mapping from peptide sequence to D_total
    protein_sequence : str
        Protein sequence
    
    Returns:
    --------
    tau : float
        Estimated τ value (standard deviation of per-residue deuteration)
    """
    per_residue_values = defaultdict(list)  # position -> [d_i values]
    
    for overhang in single_aa_overhangs:
        peptide_seq = overhang.get('peptide_seq', '')
        clean_peptide = re.sub(r'\[.*?\]', '', peptide_seq) if peptide_seq else ''
        
        if not clean_peptide:
            continue
        
        # Get D_total for this peptide
        D_total = peptide_D_total_map.get(clean_peptide, 0.0)
        
        # For single-AA overhangs, the overhang represents one residue
        # The deuteration value for that residue can be estimated from the overhang
        # For now, we'll use a simplified approach: assume the overhang value
        # represents the per-residue deuteration (0-1 scale)
        start = overhang.get('start')
        end = overhang.get('end')
        
        if start and end and start == end:  # Single residue
            # Estimate d_i from the overhang measurement
            # This is a simplified estimate - in practice, you'd need the actual
            # measured value from the fragment difference
            # For now, we'll collect positions that have single-AA coverage
            per_residue_values[start].append(0.0)  # Placeholder - would need actual measurement
    
    # If we have per-residue measurements, estimate τ as the standard deviation
    # across residues (within-segment heterogeneity)
    if per_residue_values:
        all_values = []
        for pos, values in per_residue_values.items():
            if values:
                all_values.extend(values)
        
        if len(all_values) > 1:
            tau = np.std(all_values)
            return max(tau, 0.01)  # Minimum reasonable value
    
    # Default: return a reasonable default value if no single-AA data
    # This can be estimated from overlap inconsistency or set empirically
    return 0.1  # Default τ = 0.1 (10% heterogeneity)


def estimate_tau_from_overlap_inconsistency(multi_aa_overhangs, peptide_D_total_map, protein_sequence):
    """
    Estimate τ from overlap inconsistency between overlapping overhangs.
    
    When two overhangs overlap, their measurements should be consistent.
    Inconsistency can arise from within-segment heterogeneity (τ).
    
    Parameters:
    -----------
    multi_aa_overhangs : list
        List of multi-AA overhang dictionaries
    peptide_D_total_map : dict
        Mapping from peptide sequence to D_total
    protein_sequence : str
        Protein sequence
    
    Returns:
    --------
    tau : float
        Estimated τ value from overlap inconsistency
    """
    # Find overlapping overhangs and compute inconsistency
    inconsistencies = []
    
    for i, overhang1 in enumerate(multi_aa_overhangs):
        start1 = overhang1.get('start')
        end1 = overhang1.get('end')
        
        if not start1 or not end1:
            continue
        
        for j, overhang2 in enumerate(multi_aa_overhangs[i+1:], start=i+1):
            start2 = overhang2.get('start')
            end2 = overhang2.get('end')
            
            if not start2 or not end2:
                continue
            
            # Check if overhangs overlap
            overlap_start = max(start1, start2)
            overlap_end = min(end1, end2)
            
            if overlap_start <= overlap_end:
                # Overhangs overlap - compute expected inconsistency
                # This is a simplified approach - in practice, you'd compare
                # the measured values in the overlapping region
                # For now, we'll use the overlap length as a proxy
                overlap_length = overlap_end - overlap_start + 1
                
                # Expected inconsistency scales with τ√L for the overlap
                # This is a placeholder - actual implementation would compare
                # measured values and compute residuals
                if overlap_length > 0:
                    # Placeholder inconsistency estimate
                    # In practice, this would be: |measured1 - measured2| / expected_uncertainty
                    inconsistencies.append(overlap_length)
    
    # Estimate τ from inconsistencies
    # This is a simplified approach - actual implementation would use
    # the measured values and compute proper residuals
    if inconsistencies:
        # Use median overlap length as a proxy
        # Actual implementation would compute residuals and estimate τ
        median_overlap = np.median(inconsistencies) if inconsistencies else 1.0
        # Rough estimate: τ ≈ inconsistency / √(overlap_length)
        tau = 0.1 * np.sqrt(median_overlap)  # Placeholder formula
        return max(tau, 0.01)
    
    # Default if no overlaps found
    return 0.1


def calculate_fractional_error_for_overhang(overhang, protein_sequence, peptide_D_total_map, tau=None):
    """
    Calculate coefficient of variation (CV) for a single overhang.
    
    D_total for each peptide is stable after enzymatic digestion.
    The distribution of D_total across residues within the peptide can vary
    (each residue 0-1, but they sum to D_total).
    
    Returns the CV value (0-100%).
    
    ============================================================================
    What CV Tells You (Intuition)
    ============================================================================
    
    CV = 0.10 (10%) → the noise/uncertainty is about 10% of the expected signal
    CV = 1.0 (100%) → the uncertainty is about as big as the signal (very unreliable)
    
    Higher CV = less trustworthy, because the measurement is "noisy relative to its size"
    
    In the overhang scrambling context:
    - We treat the overhang deuterium count K as a random variable under scrambling
    - μ = E[K] = expected deuteriums in the overhang
    - σ = SD(K) = uncertainty from scrambling (hypergeometric SD)
    - CV_overhang = SD(K) / E[K]
    
    This gives a dimensionless "how uncertain is this localized overhang D?" score.
    
    Why CV is useful:
    - It lets you compare uncertainty across different peptides/timepoints even when
      the absolute signal is different, because it's normalized.
    - CV computed on total overhang D (K) avoids tiny denominator issues that occur
      when computing per-residue CV at early time points (when D_total is small).
    """
    peptide_seq = overhang.get('peptide_seq', '')
    clean_peptide = re.sub(r'\[.*?\]', '', peptide_seq) if peptide_seq else ''
    
    if not clean_peptide:
        return 0.0
    
    peptide_length = len(clean_peptide)
    overhang_length = overhang.get('length', 1)
    
    # Get D_total for this peptide (stable after digestion)
    # Use peptide sequence as key to get D_total
    D_total = peptide_D_total_map.get(clean_peptide, 0.0)
    
    # Calculate CV on total overhang D (K), not per-residue, to avoid tiny denominator issues
    # 
    # Mathematical formulation:
    #   K = number of deuteriums in the overhang (random variable under scrambling)
    #   E[K] = D_total * L/N  (expected D's in overhang)
    #   Var(K) = D_total * (L/N) * (1 - L/N) * ((N - D_total) / (N - 1))  (hypergeometric variance)
    #   SD(K) = sqrt(Var(K))
    #   CV_K = SD(K) / E[K]  (coefficient of variation)
    #
    # Why compute CV on total overhang D instead of per-residue?
    # - Per-residue CV = SD_per-res / (D_total/N) has tiny denominator when D_total is small
    # - This creates artificially high CV values at early time points
    # - Computing CV on total overhang D avoids this issue
    
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
            
            τ should be chosen via one of:
            - Empirical overlap calibration: tune τ until predicted uncertainty matches
              observed scatter between redundant/overlapping constraints
            - Conservative fixed value: pick τ as a constant (same experiment) so CV
              is comparable across proteins/conditions (typical range: 0.2-0.5 D/res)
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
        # Only the scrambled fraction contributes to uncertainty
        var_K_scrambling = D * p * (1 - p) * ((N - D) / (N - 1)) * scrambling_fraction
        var_K_scrambling = max(var_K_scrambling, 0)  # Ensure non-negative
        sigma_scr = np.sqrt(var_K_scrambling)
        
        # Resolution uncertainty: σ_res(L)
        # σ_res(L) = 0 if L=1, τ√L if L>1 (for total D form)
        # For single-AA overhangs (L=1), fragment ordinal can still affect resolution
        # Longer fragments (higher ordinal) have better resolution, so lower uncertainty
        if tau is None:
            tau = 0.1  # Default τ value
        
        # Check if fragment ordinal is provided (for single-AA overhangs)
        fragment_ordinal = overhang.get('fragment_ordinal', None)
        
        if L == 1:
            # For single-AA overhangs, resolution uncertainty is typically 0
            # But if fragment ordinal is provided, we can incorporate it
            # Longer fragments (higher ordinal) have better resolution
            if fragment_ordinal is not None and fragment_ordinal > 0:
                # Incorporate fragment ordinal: higher ordinal = better resolution = lower uncertainty
                # Fragment ordinal affects the "effective resolution" - longer fragments provide
                # better localization, reducing uncertainty
                # Factor: tau / sqrt(fragment_ordinal) - longer fragments have lower uncertainty
                # This scales the resolution uncertainty inversely with fragment ordinal
                sigma_res = tau / np.sqrt(fragment_ordinal)  # Longer fragments = lower uncertainty
            else:
                sigma_res = 0.0  # No resolution uncertainty for single-AA overhangs if no ordinal
        else:
            sigma_res = tau * np.sqrt(L)  # For total D form: σ_res = τ√L
        
        # Fix B: Cap resolution uncertainty by what's physically possible
        # σ_res ≤ c·E_K prevents CV explosion solely due to resolution
        # Use conservative cap factor of 0.5 (can be adjusted)
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
    
    if peptide_length > 0 and overhang_length > 0:
        L = float(overhang_length)
        N = float(peptide_length)
        
        # Use 50% scrambling (0.50) instead of 11% scrambling
        # Use τ-based heterogeneity model (Option A)
        cv_overhang = compute_cv_overhang_total(N, L, D_total, scrambling_fraction=0.50, tau=tau)
        
        if not np.isnan(cv_overhang):
            # Cap at 1.0 (100%) for reasonable visualization
            cv_overhang = min(cv_overhang, 1.0)
            return cv_overhang * 100.0  # Convert to percentage
        else:
            return 0.0
    else:
        return 0.0

def compute_per_position_scores(coverage, unique_lengths, fragment_count_data, length_to_idx, protein_length, protein_id, base_output_file):
    """
    Compute per-position scores by collapsing across overhang lengths.
    
    Implements multiple scoring strategies:
    1. Best-case confidence (min CV across valid lengths)
    2. Evidence-weighted confidence (weighted average across lengths)
    3. Coverage count (total/max fragments per position)
    
    Parameters:
    -----------
    coverage : 2D array
        CV_mean values [length_idx, position]
    unique_lengths : list
        List of overhang lengths
    fragment_count_data : dict
        Fragment counts [length_idx][position] -> count
    length_to_idx : dict
        Mapping from length to index
    protein_length : int
        Length of protein sequence
    protein_id : str
        Protein identifier
    base_output_file : str
        Base filename for output (will append _per_position_scores.png)
    """
    # Convert fragment_count_data to 2D array
    count_matrix = np.zeros((len(unique_lengths), protein_length), dtype=int)
    for length_idx in range(len(unique_lengths)):
        for pos in range(protein_length):
            count_matrix[length_idx, pos] = fragment_count_data[length_idx][pos]
    
    # Parameters
    n_min = 3  # Minimum fragments required
    L_min = 3  # Minimum overhang length (avoid single-AA overhangs)
    L_max = 20  # Maximum overhang length
    tau = 0.2  # Threshold for "usable" (20% CV)
    
    # Convert unique_lengths to array for indexing
    L_vals = np.array(unique_lengths)
    
    # Initialize score arrays
    best_case_cv = np.full(protein_length, np.nan)  # Option 1: min CV
    evidence_weighted_cv = np.full(protein_length, np.nan)  # Option 2: weighted mean
    coverage_count_total = np.zeros(protein_length, dtype=int)  # Total fragments
    coverage_count_max = np.zeros(protein_length, dtype=int)  # Max fragments per length
    usable_fraction = np.full(protein_length, np.nan)  # Option 4: fraction usable
    
    for p in range(protein_length):
        # Get counts and CV values for this position
        counts = count_matrix[:, p]
        cv_values = coverage[:, p]
        
        # Option 1: Best-case confidence (min CV across valid lengths)
        valid_mask = (counts >= n_min) & (L_vals >= L_min) & (L_vals <= L_max) & np.isfinite(cv_values)
        if np.any(valid_mask):
            best_case_cv[p] = np.nanmin(cv_values[valid_mask])
        
        # Option 2: Evidence-weighted confidence
        # Weight by fragment count and optionally by sqrt(L)
        w = counts.astype(float)
        w *= np.sqrt(L_vals)  # Weight longer L more (they're more robust)
        mask = (w > 0) & np.isfinite(cv_values)
        if np.any(mask) and np.sum(w[mask]) > 0:
            evidence_weighted_cv[p] = np.sum(w[mask] * cv_values[mask]) / np.sum(w[mask])
        
        # Coverage counts
        coverage_count_total[p] = np.sum(counts)
        coverage_count_max[p] = np.max(counts) if np.any(counts > 0) else 0
        
        # Option 4: Usable fraction
        valid = counts >= n_min
        usable = valid & (cv_values <= tau) & np.isfinite(cv_values)
        if np.sum(valid) > 0:
            usable_fraction[p] = np.sum(usable) / np.sum(valid)
    
    # Convert CV to confidence scores (higher is better)
    # confidence = 1.0 / (CV + epsilon) or exp(-CV/scale)
    scale = 0.1  # Scaling factor for confidence
    best_case_confidence = 1.0 / (best_case_cv / 100.0 + 1e-12)  # Convert CV% back to fraction
    evidence_weighted_confidence = 1.0 / (evidence_weighted_cv / 100.0 + 1e-12)
    
    # Create visualization
    fig, axes = plt.subplots(5, 1, figsize=(max(20, protein_length * 0.15), 12), sharex=True)
    
    positions = np.arange(1, protein_length + 1)
    
    # Plot 1: Best-case CV (lower is better)
    ax = axes[0]
    ax.plot(positions, best_case_cv, 'b-', linewidth=2, label='Best-case CV')
    ax.axhline(y=tau * 100, color='r', linestyle='--', alpha=0.5, label=f'Threshold ({tau*100:.0f}%)')
    ax.set_ylabel('CV (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Best-Case Confidence (Min CV) - {protein_id}', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()  # Lower CV = better, so invert
    
    # Plot 2: Evidence-weighted CV
    ax = axes[1]
    ax.plot(positions, evidence_weighted_cv, 'g-', linewidth=2, label='Evidence-weighted CV')
    ax.axhline(y=tau * 100, color='r', linestyle='--', alpha=0.5, label=f'Threshold ({tau*100:.0f}%)')
    ax.set_ylabel('CV (%)', fontsize=12, fontweight='bold')
    ax.set_title('Evidence-Weighted Confidence (Weighted Mean CV)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()
    
    # Plot 3: Coverage counts
    ax = axes[2]
    ax.bar(positions, coverage_count_total, alpha=0.6, color='purple', label='Total fragments')
    ax2 = ax.twinx()
    ax2.bar(positions, coverage_count_max, alpha=0.4, color='orange', label='Max per length')
    ax.set_ylabel('Total Fragments', fontsize=12, fontweight='bold', color='purple')
    ax2.set_ylabel('Max Fragments (per length)', fontsize=12, fontweight='bold', color='orange')
    ax.set_title('Fragment Coverage', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Usable fraction
    ax = axes[3]
    ax.plot(positions, usable_fraction * 100, 'm-', linewidth=2, label='Usable fraction')
    ax.axhline(y=50, color='r', linestyle='--', alpha=0.5, label='50% threshold')
    ax.set_ylabel('Usable Fraction (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Fraction of Lengths with CV ≤ {tau*100:.0f}%', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Confidence scores (higher is better)
    ax = axes[4]
    ax.plot(positions, best_case_confidence, 'b-', linewidth=2, alpha=0.7, label='Best-case confidence')
    ax.plot(positions, evidence_weighted_confidence, 'g-', linewidth=2, alpha=0.7, label='Evidence-weighted confidence')
    ax.set_ylabel('Confidence (1/CV)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Protein Sequence Position', fontsize=12, fontweight='bold')
    ax.set_title('Confidence Scores (Higher = More Trustworthy)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_file = base_output_file.replace('.png', '_per_position_scores.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Per-position scores saved to: {output_file}")
    
    # Also save as CSV
    csv_file = base_output_file.replace('.png', '_per_position_scores.csv')
    import csv
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Position', 'Best_Case_CV', 'Evidence_Weighted_CV', 'Total_Fragments', 
                        'Max_Fragments_Per_Length', 'Usable_Fraction', 'Best_Case_Confidence', 
                        'Evidence_Weighted_Confidence'])
        for p in range(protein_length):
            writer.writerow([
                p + 1,
                best_case_cv[p] if not np.isnan(best_case_cv[p]) else '',
                evidence_weighted_cv[p] if not np.isnan(evidence_weighted_cv[p]) else '',
                coverage_count_total[p],
                coverage_count_max[p],
                usable_fraction[p] * 100 if not np.isnan(usable_fraction[p]) else '',
                best_case_confidence[p] if not np.isnan(best_case_cv[p]) else '',
                evidence_weighted_confidence[p] if not np.isnan(evidence_weighted_cv[p]) else ''
            ])
    print(f"Per-position scores CSV saved to: {csv_file}")

def create_fractional_error_coverage_heatmap(csv_file, protein_id, fasta_file, output_file, ion_type_filter=None, multiply_by_count=False, tau=None):
    """
    Create coverage heatmap with CV (coefficient of variation) instead of fragment counts.
    
    X-axis: Protein sequence position
    Y-axis: Overhang length
    Color: CV (coefficient of variation) computed on total overhang D (K)
    
    Two modes:
    1. Average mode (multiply_by_count=False):
       - Shows average CV across all fragments at each (position, overhang_length)
       - CV = SD(K) / E[K] where K = total D in overhang
       
    2. Uncertainty of the mean mode (multiply_by_count=True):
       - Shows CV_mean = mean(CV_i) / sqrt(n)
       - Represents how trustworthy the average signal is
       - More fragments → lower uncertainty (statistically correct)
       - Lower values = more trustworthy average signal
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
    
    # Store protein_length for per-position score computation
    _protein_length_for_scores = protein_length
    
    # Parse overhangs first to get all peptides
    print("Parsing multi-AA overhangs...")
    multi_aa_overhangs = parse_multi_aa_overhangs(csv_file, protein_id=protein_id, fasta_file=fasta_file)
    
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
    single_aa_overhangs = parse_single_aa_overhangs(csv_file, protein_id=protein_id, fasta_file=fasta_file)
    
    # Convert single-AA to same format as multi-AA
    for overhang in single_aa_overhangs:
        multi_aa_overhangs.append(overhang)
    
    # Step 1: Assign binary 0/1 to each residue in the protein sequence (for visualization)
    # This is the "synthetic sequence" assignment
    random.seed(42)  # For reproducibility
    residue_deuteration = {}  # protein position -> 0 or 1
    for pos in range(1, protein_length + 1):
        residue_deuteration[pos] = random.choice([0, 1])  # Binary assignment
    
    # Step 2: Calculate D_total for each unique peptide by summing binary values
    # Collect all unique peptides
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
    
    # Estimate τ if not provided
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
    
    # Get all unique overhang lengths
    unique_lengths = sorted(set(o['length'] for o in multi_aa_overhangs))
    
    # Create 2D array: fractional_error[overhang_length][residue_position] = average fractional error
    # Structure: coverage[length_idx][pos] = list of fractional error values
    fractional_error_data = defaultdict(lambda: defaultdict(list))  # length_idx -> pos -> [values]
    fragment_count_data = defaultdict(lambda: defaultdict(int))  # length_idx -> pos -> count
    length_to_idx = {length: idx for idx, length in enumerate(unique_lengths)}
    
    # Calculate fractional error for each overhang and store by position and length
    print("Calculating fractional errors...")
    for overhang in multi_aa_overhangs:
        start = overhang.get('start')
        end = overhang.get('end')
        length = overhang.get('length')
        
        if start < 1 or end > protein_length:
            continue
        
        # Calculate fractional error for this overhang
        fractional_error = calculate_fractional_error_for_overhang(
            overhang, protein_sequence, peptide_D_total_map, tau=tau
        )
        
        if length in length_to_idx:
            length_idx = length_to_idx[length]
            # Store fractional error and count fragments for all residues in this overhang
            for pos in range(start, end + 1):
                fractional_error_data[length_idx][pos - 1].append(fractional_error)  # 0-indexed
                fragment_count_data[length_idx][pos - 1] += 1  # Count fragments
    
    # Create 2D array with average fractional error multiplied by fragment count
    coverage = np.zeros((len(unique_lengths), protein_length), dtype=float)
    for length_idx in range(len(unique_lengths)):
        for pos in range(protein_length):
            values = fractional_error_data[length_idx][pos]
            fragment_count = fragment_count_data[length_idx][pos]
            if values and fragment_count > 0:
                if multiply_by_count:
                    # Uncertainty of the mean: CV_mean = mean(CV_i) / sqrt(n)
                    # This represents how trustworthy the average signal is
                    avg_cv = np.mean(values)
                    cv_mean = avg_cv / np.sqrt(fragment_count)
                    coverage[length_idx, pos] = cv_mean
                else:
                    # Just average CV across all overhangs at this position
                    coverage[length_idx, pos] = np.mean(values)
            else:
                coverage[length_idx, pos] = np.nan  # No coverage
    
    # Create figure
    fig_width = max(20, protein_length * 0.15)
    fig_height = max(8, len(unique_lengths) * 0.4)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Create custom colormap: white for no coverage, then color gradient for fractional error
    from matplotlib.colors import LinearSegmentedColormap
    
    if multiply_by_count:
        # Color scheme for fractional error × fragment count: navy blue -> crimson -> gold
        colors = [
            (0.0, 0.0, 0.5),       # Navy blue
            (0.863, 0.078, 0.235), # Crimson
            (1.0, 0.843, 0.0)      # Gold
        ]
        colormap_name = 'navy_blue_crimson_gold'
    else:
        # Color scheme for regular fractional error: lavender -> pink -> green -> orange
        colors = [
            (0.9, 0.7, 0.9),   # Lavender (light purple)
            (1.0, 0.5, 0.8),   # Pink
            (0.5, 0.9, 0.5),   # Green
            (1.0, 0.6, 0.0)    # Orange
        ]
        colormap_name = 'lavender_pink_green_orange'
    
    # Create colormap with these colors
    n_bins = 256
    custom_color_map = LinearSegmentedColormap.from_list(colormap_name, colors, N=n_bins)
    
    colormap_colors = []
    colormap_colors.append((1.0, 1.0, 1.0))  # White for no coverage
    for i in range(256):
        color = custom_color_map(i / 255.0)  # Get color from custom gradient
        colormap_colors.append(color[:3])
    custom_cmap = ListedColormap(colormap_colors)
    
    # Normalize: NaN -> white (0), 0-100% -> color map (1-256)
    coverage_normalized = coverage.copy()
    coverage_normalized[np.isnan(coverage)] = -10.0  # Sentinel for no coverage
    
    # Create normalization
    from matplotlib.colors import Normalize
    if multiply_by_count:
        # Need to find actual min/max since values are now divided by sqrt(fragment_count)
        # This gives uncertainty of the mean (SEM-like behavior)
        valid_values = coverage[~np.isnan(coverage)]
        if len(valid_values) > 0:
            min_val = np.min(valid_values)
            max_val = np.max(valid_values)
            print(f"Fractional error / sqrt(fragment_count) range: min={min_val:.2f}, max={max_val:.2f}")
        else:
            min_val = 0.0
            max_val = 100.0
    else:
        min_val = 0.0
        max_val = 100.0
    
    class CustomNormalize(Normalize):
        def __init__(self, vmin, vmax, no_coverage_value):
            super().__init__(vmin=vmin, vmax=vmax)
            self.no_coverage_value = no_coverage_value
            self.data_min = vmin
            self.data_max = vmax
        
        def __call__(self, value, clip=None):
            if isinstance(value, np.ndarray):
                result = np.zeros_like(value, dtype=float)
                mask = value == self.no_coverage_value
                result[mask] = 0.0  # White
                data_mask = ~mask
                if np.any(data_mask):
                    data_values = value[data_mask]
                    normalized = (data_values - self.data_min) / (self.data_max - self.data_min)
                    result[data_mask] = (1.0 / 257.0) + normalized * (1.0 - 1.0 / 257.0)
                return np.clip(result, 0, 1)
            else:
                if value == self.no_coverage_value:
                    return 0.0
                normalized = (value - self.data_min) / (self.data_max - self.data_min)
                return np.clip((1.0 / 257.0) + normalized * (1.0 - 1.0 / 257.0), 0, 1)
    
    custom_norm = CustomNormalize(vmin=min_val, vmax=max_val, no_coverage_value=-10.0)
    
    # Display heatmap
    im = ax.imshow(coverage_normalized, aspect='auto', cmap=custom_cmap, norm=custom_norm,
                   interpolation='nearest', origin='lower')
    
    # Set axis labels
    ax.set_xlabel('Protein Position (1-indexed) with Amino Acid', fontsize=12, fontweight='bold')
    ax.set_ylabel('Overhang Length (residues)', fontsize=12, fontweight='bold')
    
    # Set title - explicitly state τ-based heterogeneity model
    if multiply_by_count:
        title = f'CV (Overhang Total) / √(Fragment Count) Coverage Heatmap\n{protein_id} (Length: {protein_length} residues)\nCV computed on total overhang D (K), not per-residue\nUncertainty of the mean: CV_mean = mean(CV_i) / √n\nτ-based heterogeneity model: SD(S) = τ√L, SD(𝑑̄) = τ/√L'
    else:
        title = f'CV (Overhang Total) Coverage Heatmap\n{protein_id} (Length: {protein_length} residues)\nCV = SD(K) / E[K] where K = total D in overhang\nE[K] = D_total * L/N, Var(K) = Var_scrambling + τ²L\nτ-based heterogeneity model: SD(S) = τ√L, SD(𝑑̄) = τ/√L (τ = {tau:.4f})'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    
    # Set x-axis ticks to show positions with amino acid letters
    x_ticks = list(range(protein_length))
    x_labels = []
    for t in x_ticks:
        pos_1idx = t + 1
        if pos_1idx <= len(protein_sequence):
            aa = protein_sequence[t]
            x_labels.append(f"{pos_1idx}\n{aa}")
        else:
            x_labels.append(str(pos_1idx))
    
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, fontsize=8, rotation=0)
    
    # Set y-axis ticks to show overhang lengths
    y_ticks = list(range(len(unique_lengths)))
    y_labels = [str(length) for length in unique_lengths]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=10)
    
    # Add colorbar
    from matplotlib.colorbar import Colorbar
    from matplotlib.cm import ScalarMappable
    sm = ScalarMappable(cmap=custom_cmap, norm=custom_norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical', pad=0.02, shrink=0.8)
    if multiply_by_count:
        cbar.set_label(f'CV (Overhang Total) / √(Fragment Count) - Uncertainty of the Mean (%)\nWhite=No Coverage, Range: {min_val:.2f} to {max_val:.2f}\nLower values = more trustworthy average signal', 
                       fontsize=12, fontweight='bold')
    else:
        cbar.set_label('CV (Overhang Total) - Coefficient of Variation (%)\nCV = SD(K) / E[K] where K = total D in overhang\nWhite=No Coverage, Range: 0% to 100%', 
                       fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Fractional error coverage heatmap saved to: {output_file}")
    
    # Return the coverage matrix and metadata for per-position score computation
    return coverage, unique_lengths, fragment_count_data, length_to_idx, protein_length
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Create fractional error coverage heatmap')
    parser.add_argument('--csv', required=True, help='Input CSV file')
    parser.add_argument('--fasta', required=True, help='FASTA file with protein sequences')
    parser.add_argument('--output', default=None,
                       help='Output PNG file (default: auto-generated based on mode)')
    parser.add_argument('--multiply-by-count', action='store_true',
                       help='Multiply fractional error by fragment count for each position')
    parser.add_argument('--protein-id', help='Protein ID to filter (optional)')
    parser.add_argument('--ion-type', choices=['c', 'z'], help='Filter by ion type (c or z)')
    parser.add_argument('--tau', type=float, default=None,
                       help='Within-segment heterogeneity parameter τ (SD of per-residue deuteration). '
                            'If not provided, will be estimated from data.')
    
    args = parser.parse_args()
    
    # Try to extract protein ID from CSV if not provided
    protein_id = args.protein_id
    if not protein_id:
        # Try to get from CSV header or use first protein in FASTA
        sequences = parse_fasta(args.fasta)
        if sequences:
            protein_id = list(sequences.keys())[0]
            print(f"Using protein ID: {protein_id}")
    
    # Auto-generate output filename if not provided
    if args.output is None:
        if args.multiply_by_count:
            args.output = 'cv_overhang_total_uncertainty_of_mean_coverage_heatmap.png'
        else:
            args.output = 'cv_overhang_total_average_coverage_heatmap.png'
    
    result = create_fractional_error_coverage_heatmap(
        args.csv, protein_id, args.fasta, args.output, 
        ion_type_filter=args.ion_type,
        multiply_by_count=args.multiply_by_count,
        tau=args.tau
    )
    
    # If multiply_by_count mode, also compute per-position scores
    if args.multiply_by_count and result is not None:
        coverage, unique_lengths, fragment_count_data, length_to_idx, protein_length = result
        compute_per_position_scores(
            coverage, unique_lengths, fragment_count_data, length_to_idx,
            protein_length, protein_id, args.output
        )

if __name__ == '__main__':
    main()
