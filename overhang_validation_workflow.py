#!/usr/bin/env python3
"""
Overhang Validation Workflow

This script implements a validation workflow that compares single-AA overhang
estimates with multi-AA overhang block measurements to assess consistency and
identify potential issues (scrambling, contaminants, etc.).

Workflow:
Step A: Build "single-AA map" from adjacent fragments
Step B: Build "block map" from multi-AA overhangs  
Step C: Compare predicted vs observed block totals
Step D: Calculate residuals and z-scores
Step E: Two-pass optimization system
"""

import argparse
import re
import csv
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches


def parse_fasta(fasta_file):
    """Parse FASTA file and return protein sequences."""
    sequences = {}
    current_id = None
    current_seq = []
    
    with open(fasta_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_id is not None:
                    sequences[current_id] = ''.join(current_seq)
                current_id = line[1:].split()[0]  # Get first word after >
                current_seq = []
            else:
                current_seq.append(line)
        
        if current_id is not None:
            sequences[current_id] = ''.join(current_seq)
    
    return sequences


def parse_csv_data(csv_file, protein_id=None):
    """
    Parse CSV to extract:
    - Single-AA overhangs with their fragment pairs and measurements
    - Multi-AA overhangs with their fragment pairs and measurements
    - Protein mapping information
    """
    single_aa_map = defaultdict(dict)  # protein -> position -> {measurements, fragment_pairs}
    multi_aa_blocks = []  # List of {start, end, measured_value, fragment_pairs, protein}
    
    with open(csv_file, 'r') as f:
        lines = f.readlines()
        
        if len(lines) < 2:
            return single_aa_map, multi_aa_blocks
        
        header_line = lines[1]
        header = [col.strip() for col in header_line.split(',')]
        
        # Find column indices
        try:
            protein_idx = header.index('protein')
            single_aa_idx = header.index('single_aa_overhangs')
            single_aa_pairs_idx = header.index('single_aa_overhang_fragment_pairs')
            multi_aa_idx = header.index('multi_aa_overhangs')
            multi_aa_pairs_idx = header.index('multi_aa_overhang_fragment_pairs')
            site_specific_idx = header.index('site_specific_residues')
            matched_ions_idx = header.index('matched fragment ions')
            intensities_idx = header.index('matched fragment ion intensities')
        except ValueError as e:
            print(f"Error: Required column not found: {e}")
            return single_aa_map, multi_aa_blocks
        
        # Parse data rows
        for line in lines[2:]:
            if not line.strip():
                continue
            
            # Parse quoted fields
            quoted_fields = re.findall(r'"([^"]*)"', line)
            
            # Parse row_data
            row_data = []
            in_quotes = False
            current = ""
            for char in line:
                if char == '"':
                    in_quotes = not in_quotes
                elif char == ',' and not in_quotes:
                    row_data.append(current.strip())
                    current = ""
                    continue
                current += char
            if current:
                row_data.append(current.strip())
            
            if len(row_data) <= max(protein_idx, site_specific_idx):
                continue
            
            # Extract fields
            protein = row_data[protein_idx].strip('"').strip()
            if not protein:
                continue
            
            # Clean protein ID
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
            
            # Get data from quoted_fields (more reliable)
            if len(quoted_fields) >= 4:
                # Order: multi_aa_overhangs (-4), single_aa_overhangs (-3), 
                # single_aa_overhang_fragment_pairs (-2), site_specific_residues (-1)
                site_specific_str = quoted_fields[-1]
                single_aa_pairs_str = quoted_fields[-2]
                single_aa_str = quoted_fields[-3]
                multi_aa_str = quoted_fields[-4] if len(quoted_fields) >= 4 else ""
            else:
                continue
            
            # Get fragment pairs for multi-AA (from row_data due to column mix-up)
            multi_aa_pairs_str = ""
            if len(row_data) > multi_aa_idx:
                candidate = row_data[multi_aa_idx].strip('"').strip()
                if candidate and re.search(r'[cz]\d+-[cz]', candidate):
                    multi_aa_pairs_str = candidate
            
            # Get matched ions and intensities for measurements
            matched_ions_str = ""
            intensities_str = ""
            if len(quoted_fields) >= 6:
                # Find matched ions (usually one of the first quoted fields)
                for qf in quoted_fields:
                    if qf and (',' in qf) and (qf.startswith('c') or qf.startswith('z') or 'z1_' in qf):
                        matched_ions_str = qf
                        break
                # Intensities should be nearby
                if matched_ions_str:
                    ions_list = matched_ions_str.split(',')
                    # Find intensities field (should have same number of values)
                    for qf in quoted_fields:
                        if qf and ',' in qf:
                            intensity_list = qf.split(',')
                            if len(intensity_list) == len(ions_list):
                                intensities_str = qf
                                break
            
            # Calculate protein offset
            protein_start_offset = None
            if site_specific_str and single_aa_str:
                peptide_positions = []
                for item in single_aa_str.split(','):
                    item = item.strip()
                    if item and len(item) > 1:
                        match = re.match(r'^(\d+)([A-Z])$', item)
                        if match:
                            peptide_positions.append(int(match.group(1)))
                
                protein_positions = []
                for item in site_specific_str.split(','):
                    item = item.strip()
                    if item and len(item) > 1:
                        match = re.match(r'^(\d+)([A-Z])$', item)
                        if match:
                            protein_positions.append(int(match.group(1)))
                
                if peptide_positions and protein_positions and len(peptide_positions) == len(protein_positions):
                    offset = protein_positions[0] - peptide_positions[0] + 1
                    matches = sum(1 for p, pr in zip(peptide_positions[:min(3, len(peptide_positions))], 
                                                     protein_positions[:min(3, len(protein_positions))])
                                 if pr == p + offset - 1)
                    if matches >= min(2, len(peptide_positions)):
                        protein_start_offset = offset
            
            if protein_start_offset is None:
                continue  # Skip if we can't map to protein
            
            # Parse single-AA overhangs
            if single_aa_str and single_aa_pairs_str:
                single_aa_list = [s.strip() for s in single_aa_str.split(',') if s.strip()]
                single_aa_pairs_list = [p.strip() for p in single_aa_pairs_str.split(',') if p.strip()]
                
                # Map to protein positions and store
                for i, (overhang_str, pairs_str) in enumerate(zip(single_aa_list, single_aa_pairs_list)):
                    match = re.match(r'^(\d+)([A-Z])$', overhang_str)
                    if match:
                        peptide_pos = int(match.group(1))
                        residue = match.group(2)
                        protein_pos = peptide_pos + protein_start_offset - 1
                        
                        # Extract fragment pairs
                        fragment_pairs = [fp.strip() for fp in pairs_str.split('|') if fp.strip()]
                        
                        # TODO: Extract actual deuterium uptake measurements
                        # In a real HDX-MS workflow, you would:
                        # 1. Extract fragment intensities from the raw data
                        # 2. Calculate deuterium uptake: D = (m/z_observed - m/z_theoretical) / charge
                        # 3. For single-AA overhangs: d_i^(1) = D(fragment_N+1) - D(fragment_N)
                        # 4. Calculate uncertainty from replicate measurements or intensity-based error
                        # 
                        # For now, using placeholder values to demonstrate the workflow structure
                        measurement = 1.0  # Placeholder - would be actual d_i^(1) from HDX data
                        uncertainty = 0.1  # Placeholder - would be actual measurement uncertainty
                        
                        if protein_pos not in single_aa_map[clean_protein]:
                            single_aa_map[clean_protein][protein_pos] = {
                                'measurements': [],
                                'fragment_pairs': [],
                                'uncertainties': []
                            }
                        
                        single_aa_map[clean_protein][protein_pos]['measurements'].append(measurement)
                        single_aa_map[clean_protein][protein_pos]['fragment_pairs'].extend(fragment_pairs)
                        single_aa_map[clean_protein][protein_pos]['uncertainties'].append(uncertainty)
            
            # Parse multi-AA overhangs
            # Due to CSV column mix-up: multi-AA overhangs are in single_aa_overhang_fragment_pairs column (24)
            # and multi-AA fragment pairs are in multi_aa_overhangs column (25)
            multi_aa_str_actual = ""
            multi_aa_pairs_str_actual = ""
            
            # Get from row_data
            single_aa_pairs_idx = header.index('single_aa_overhang_fragment_pairs')
            if len(row_data) > single_aa_pairs_idx:
                candidate = row_data[single_aa_pairs_idx].strip('"').strip()
                if candidate and re.search(r'\d+-\d+[A-Z]{2,}', candidate):
                    multi_aa_str_actual = candidate
            
            if len(row_data) > multi_aa_idx:
                candidate = row_data[multi_aa_idx].strip('"').strip()
                if candidate and re.search(r'[cz]\d+-[cz]', candidate):
                    multi_aa_pairs_str_actual = candidate
            
            if multi_aa_str_actual and multi_aa_pairs_str_actual:
                multi_aa_list = [m.strip() for m in multi_aa_str_actual.split(',') if m.strip()]
                multi_aa_pairs_list = [p.strip() for p in multi_aa_pairs_str_actual.split(',') if p.strip()]
                
                for i, overhang_str in enumerate(multi_aa_list):
                        match = re.match(r'^(\d+)-(\d+)([A-Z]+)$', overhang_str)
                        if match:
                            peptide_start = int(match.group(1))
                            peptide_end = int(match.group(2))
                            residues = match.group(3)
                            
                            protein_start = peptide_start + protein_start_offset - 1
                            protein_end = peptide_end + protein_start_offset - 1
                            
                            # Get fragment pairs
                            fragment_pairs = []
                            if i < len(multi_aa_pairs_list):
                                fragment_pairs = [fp.strip() for fp in multi_aa_pairs_list[i].split('|') if fp.strip()]
                            
                            # TODO: Extract actual block deuterium uptake measurements
                            # For multi-AA overhangs: B^meas_{a:b} = D(fragment_M) - D(fragment_N)
                            # where fragment_M covers positions 1..b and fragment_N covers positions 1..a
                            # This gives the total deuterium in the block [a+1, b]
                            block_measurement = 1.0  # Placeholder - would be actual B^meas_{a:b}
                            block_uncertainty = 0.1  # Placeholder - would be actual block measurement uncertainty
                            
                            multi_aa_blocks.append({
                                'protein': clean_protein,
                                'start': protein_start,
                                'end': protein_end,
                                'residues': residues,
                                'measured_value': block_measurement,
                                'uncertainty': block_uncertainty,
                                'fragment_pairs': fragment_pairs
                            })
    
    return single_aa_map, multi_aa_blocks


def build_single_aa_map(single_aa_map):
    """
    Step A: Build single-AA map
    
    For each residue position, compute d_i^(1) (or its distribution across replicates/charge states)
    
    d_i^(1) is the deuterium uptake at position i, estimated from single-AA overhangs:
    - For consecutive fragments N and N+1: d_i = D(fragment_N+1) - D(fragment_N)
    - Multiple measurements (from different ion series, charge states, replicates) are combined
    
    Returns:
        single_aa_estimates: dict mapping protein -> position -> {mean, std, count, fragment_pairs}
    """
    single_aa_estimates = defaultdict(dict)  # protein -> position -> {mean, std, count}
    
    for protein, positions in single_aa_map.items():
        for pos, data in positions.items():
            measurements = data['measurements']
            uncertainties = data['uncertainties']
            
            if measurements:
                # Compute weighted mean (if we had proper uncertainties)
                # For now, simple mean
                mean_val = np.mean(measurements)
                
                # Combine measurement uncertainties
                # If measurements are independent: σ_combined = sqrt(Σ σ_i^2) / N
                # This accounts for the fact that we're averaging multiple estimates
                combined_uncertainty = np.sqrt(np.sum([u**2 for u in uncertainties])) / len(uncertainties)
                
                # Alternative: use standard deviation of measurements if we have replicates
                if len(measurements) > 1:
                    std_val = np.std(measurements, ddof=1)
                    # Use larger of combined uncertainty or observed std
                    combined_uncertainty = max(combined_uncertainty, std_val)
                
                single_aa_estimates[protein][pos] = {
                    'mean': mean_val,
                    'std': combined_uncertainty,
                    'count': len(measurements),
                    'fragment_pairs': data['fragment_pairs']
                }
    
    return single_aa_estimates


def predict_block_totals(single_aa_estimates, multi_aa_blocks):
    """
    Step C: Compare predicted vs observed block totals
    
    For each block [a,b]:
        B^pred_{a:b} = Σ_{i=a}^{b} d_i^(1)
    
    Where d_i^(1) are the single-AA estimates from Step A.
    
    Residual:
        r_{a:b} = B^meas_{a:b} - B^pred_{a:b}
    
    Z-score (normalized by uncertainty):
        z_{a:b} = r_{a:b} / σ_{a:b}
    
    Where σ_{a:b} includes:
        - Block measurement uncertainty (from experimental data)
        - Prediction uncertainty (from summing single-AA estimates)
        - σ_{a:b} = sqrt(σ_block^2 + Σ_{i=a}^{b} σ_i^2)
    
    Interpretation:
        - Small |z| (near 0): Single-AA calls are self-consistent with robust block constraints
        - Large |z|: Potential issues:
            * Scrambling differences between fragments
            * Overlapped contaminant fragment
            * Wrong deconvolution / baseline
            * Multi-population mismatch
            * Missed intensity (missing high-D signal)
    """
    results = []
    
    for block in multi_aa_blocks:
        protein = block['protein']
        start = block['start']
        end = block['end']
        measured = block['measured_value']
        
        if protein not in single_aa_estimates:
            continue
        
        # Sum single-AA estimates for this block
        # B^pred_{a:b} = Σ_{i=a}^{b} d_i^(1)
        predicted = 0.0
        predicted_uncertainty_sq = 0.0
        positions_covered = []
        
        for pos in range(start, end + 1):
            if pos in single_aa_estimates[protein]:
                predicted += single_aa_estimates[protein][pos]['mean']
                # Uncertainty propagates as: σ_sum^2 = Σ σ_i^2
                predicted_uncertainty_sq += single_aa_estimates[protein][pos]['std']**2
                positions_covered.append(pos)
        
        if len(positions_covered) == 0:
            continue  # No single-AA coverage for this block
        
        # Combined uncertainty
        predicted_uncertainty = np.sqrt(predicted_uncertainty_sq)
        block_uncertainty = block['uncertainty']
        
        # Total uncertainty: σ_{a:b} = sqrt(σ_block^2 + σ_pred^2)
        # This accounts for both block measurement error and prediction uncertainty
        total_uncertainty = np.sqrt(block_uncertainty**2 + predicted_uncertainty_sq)
        
        # Residual: r_{a:b} = B^meas_{a:b} - B^pred_{a:b}
        residual = measured - predicted
        
        # Z-score: z_{a:b} = r_{a:b} / σ_{a:b}
        z_score = residual / total_uncertainty if total_uncertainty > 0 else 0.0
        
        results.append({
            'protein': protein,
            'start': start,
            'end': end,
            'residues': block['residues'],
            'measured': measured,
            'predicted': predicted,
            'residual': residual,
            'z_score': z_score,
            'total_uncertainty': total_uncertainty,
            'predicted_uncertainty': predicted_uncertainty,
            'block_uncertainty': block_uncertainty,
            'positions_covered': positions_covered,
            'coverage_fraction': len(positions_covered) / (end - start + 1),
            'fragment_pairs': block['fragment_pairs']
        })
    
    return results


def create_validation_plots(results, protein_sequence, protein_id, output_file):
    """
    Create visualization of validation results:
    - Residual plot
    - Z-score plot
    - Coverage map
    """
    if not results:
        print("No validation results to plot")
        return
    
    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
    
    # Plot 1: Residuals vs position
    ax1 = axes[0]
    positions = [(r['start'] + r['end']) / 2 for r in results]
    residuals = [r['residual'] for r in results]
    colors = ['red' if abs(r['z_score']) > 2 else 'orange' if abs(r['z_score']) > 1 else 'green' 
              for r in results]
    
    ax1.scatter(positions, residuals, c=colors, alpha=0.6, s=50)
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax1.axhline(y=2, color='red', linestyle=':', linewidth=1, alpha=0.5)
    ax1.axhline(y=-2, color='red', linestyle=':', linewidth=1, alpha=0.5)
    ax1.set_xlabel('Protein Position (midpoint of block)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Residual (measured - predicted)', fontsize=12, fontweight='bold')
    ax1.set_title(f'Block Residuals: {protein_id}', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Z-scores vs position
    ax2 = axes[1]
    z_scores = [r['z_score'] for r in results]
    ax2.scatter(positions, z_scores, c=colors, alpha=0.6, s=50)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax2.axhline(y=2, color='red', linestyle=':', linewidth=1, alpha=0.5, label='±2σ')
    ax2.axhline(y=-2, color='red', linestyle=':', linewidth=1, alpha=0.5)
    ax2.axhline(y=1, color='orange', linestyle=':', linewidth=1, alpha=0.5, label='±1σ')
    ax2.axhline(y=-1, color='orange', linestyle=':', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Protein Position (midpoint of block)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Z-score', fontsize=12, fontweight='bold')
    ax2.set_title('Z-scores (normalized residuals)', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Coverage and z-score heatmap
    ax3 = axes[2]
    max_pos = max(r['end'] for r in results) if results else len(protein_sequence)
    
    # Create heatmap
    for r in results:
        start = r['start']
        end = r['end']
        z_score = r['z_score']
        coverage = r['coverage_fraction']
        
        # Color by z-score
        if abs(z_score) > 2:
            color = 'red'
        elif abs(z_score) > 1:
            color = 'orange'
        else:
            color = 'green'
        
        # Alpha by coverage
        alpha = 0.3 + 0.7 * coverage
        
        rect = Rectangle((start, 0), end - start + 1, 1, 
                        facecolor=color, edgecolor='black', linewidth=0.5, alpha=alpha)
        ax3.add_patch(rect)
    
    ax3.set_xlim(0, max_pos + 1)
    ax3.set_ylim(-0.1, 1.1)
    ax3.set_xlabel('Protein Position', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Block Coverage', fontsize=12, fontweight='bold')
    ax3.set_title('Block Validation Map (color=z-score, alpha=coverage)', fontsize=14, fontweight='bold')
    ax3.set_yticks([])
    
    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor='green', alpha=0.7, label='|z| ≤ 1 (consistent)'),
        mpatches.Patch(facecolor='orange', alpha=0.7, label='1 < |z| ≤ 2 (moderate discrepancy)'),
        mpatches.Patch(facecolor='red', alpha=0.7, label='|z| > 2 (large discrepancy)')
    ]
    ax3.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Validation plots saved to: {output_file}")
    plt.close()


def print_validation_summary(results):
    """Print summary statistics of validation results."""
    if not results:
        print("No validation results")
        return
    
    z_scores = [r['z_score'] for r in results]
    residuals = [r['residual'] for r in results]
    coverage_fractions = [r['coverage_fraction'] for r in results]
    
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    print(f"Total blocks validated: {len(results)}")
    print(f"\nZ-score statistics:")
    print(f"  Mean: {np.mean(z_scores):.3f}")
    print(f"  Std: {np.std(z_scores):.3f}")
    print(f"  Min: {np.min(z_scores):.3f}")
    print(f"  Max: {np.max(z_scores):.3f}")
    print(f"\nConsistency breakdown:")
    print(f"  |z| ≤ 1 (consistent): {sum(1 for z in z_scores if abs(z) <= 1)} ({100*sum(1 for z in z_scores if abs(z) <= 1)/len(z_scores):.1f}%)")
    print(f"  1 < |z| ≤ 2 (moderate): {sum(1 for z in z_scores if 1 < abs(z) <= 2)} ({100*sum(1 for z in z_scores if 1 < abs(z) <= 2)/len(z_scores):.1f}%)")
    print(f"  |z| > 2 (large discrepancy): {sum(1 for z in z_scores if abs(z) > 2)} ({100*sum(1 for z in z_scores if abs(z) > 2)/len(z_scores):.1f}%)")
    print(f"\nCoverage statistics:")
    print(f"  Mean coverage: {np.mean(coverage_fractions):.3f}")
    print(f"  Blocks with full coverage: {sum(1 for c in coverage_fractions if c == 1.0)} ({100*sum(1 for c in coverage_fractions if c == 1.0)/len(coverage_fractions):.1f}%)")
    print(f"  Blocks with partial coverage: {sum(1 for c in coverage_fractions if 0 < c < 1.0)} ({100*sum(1 for c in coverage_fractions if 0 < c < 1.0)/len(coverage_fractions):.1f}%)")
    print("="*70)


def two_pass_optimization(single_aa_estimates, multi_aa_blocks):
    """
    Two-pass optimization system (Step E)
    
    Pass 1: Trust single-AA where available, but don't overfit
        - Use single-AA estimates as initial d_i
        - Put uncertainties on them
    
    Pass 2: Solve a joint system
        - Include both constraint types:
            * Single-AA constraints (high resolution, higher fractional error)
              d_k ≈ d_k^(1)
            * Multi-AA constraints (lower resolution, lower fractional error)
              Σ_{i=a}^{b} d_i ≈ B^meas_{a:b}
        - Weight constraints by inverse fractional error
        - Solve for optimal d_i that satisfy both constraint types
    
    This is a placeholder for the optimization algorithm.
    In a real implementation, you would use a least-squares solver or
    more sophisticated optimization (e.g., with regularization).
    """
    # TODO: Implement actual optimization
    # This would involve:
    # 1. Setting up a system of linear equations/constraints
    # 2. Weighting by inverse fractional error (longer blocks = higher weight)
    # 3. Solving for optimal d_i values
    # 4. Iterating if needed
    
    print("\nStep E: Two-pass optimization (placeholder)")
    print("  Pass 1: Using single-AA estimates as initial values")
    print("  Pass 2: Joint optimization with weighted constraints")
    print("  Note: Full optimization implementation would require HDX data")
    
    # For now, return the single-AA estimates as-is
    return single_aa_estimates


def main():
    parser = argparse.ArgumentParser(
        description='Validate single-AA overhang estimates against multi-AA block measurements',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script implements a validation workflow that:

Step A: Builds a "single-AA map" from adjacent fragments
  - Computes d_i^(1) for each resolvable residue
  - Combines measurements from multiple fragments/replicates

Step B: Builds a "block map" from multi-AA overhangs
  - Each overhang gives a segment [a,b] and measured block uptake B^meas_{a:b}

Step C: Compares predicted vs observed block totals
  - B^pred_{a:b} = Σ_{i=a}^{b} d_i^(1)
  - Residual: r_{a:b} = B^meas_{a:b} - B^pred_{a:b}
  - Z-score: z_{a:b} = r_{a:b} / σ_{a:b}

Step D: Interprets results
  - Small |z|: self-consistent
  - Large |z|: potential issues (scrambling, contaminants, etc.)

Step E: Two-pass optimization (placeholder)
  - Pass 1: Use single-AA as initial estimates
  - Pass 2: Joint optimization with weighted constraints

Note: This script currently uses placeholder values for deuterium uptake.
      In a real implementation, you would extract actual HDX measurements
      from the experimental data.
        """
    )
    parser.add_argument('--csv', required=True, help='Comet CSV output file')
    parser.add_argument('--fasta', required=True, help='FASTA file with protein sequences')
    parser.add_argument('--output', default='overhang_validation.png',
                       help='Output PNG file (default: overhang_validation.png)')
    parser.add_argument('--protein', help='Specific protein ID to validate (optional)')
    
    args = parser.parse_args()
    
    # Parse FASTA
    sequences = parse_fasta(args.fasta)
    if not sequences:
        print("Error: No sequences found in FASTA file")
        return
    
    # Parse CSV data
    print("Parsing CSV data...")
    single_aa_map, multi_aa_blocks = parse_csv_data(args.csv, args.protein)
    
    if not single_aa_map and not multi_aa_blocks:
        print("No data found in CSV")
        return
    
    print(f"Found single-AA data for {len(single_aa_map)} proteins")
    print(f"Found {len(multi_aa_blocks)} multi-AA blocks")
    
    # Step A: Build single-AA map
    print("\n" + "="*70)
    print("Step A: Building single-AA map")
    print("="*70)
    single_aa_estimates = build_single_aa_map(single_aa_map)
    total_positions = sum(len(est) for est in single_aa_estimates.values())
    print(f"Built single-AA estimates for {total_positions} positions")
    
    # Step B: Multi-AA blocks are already parsed
    print(f"\nStep B: Block map contains {len(multi_aa_blocks)} blocks")
    
    # Step C: Compare predicted vs observed
    print("\n" + "="*70)
    print("Step C: Comparing predicted vs observed block totals")
    print("="*70)
    results = predict_block_totals(single_aa_estimates, multi_aa_blocks)
    print(f"Validated {len(results)} blocks")
    
    # Step D: Print summary and interpretation
    print("\n" + "="*70)
    print("Step D: Validation Results")
    print("="*70)
    print_validation_summary(results)
    
    # Step E: Two-pass optimization (placeholder)
    print("\n" + "="*70)
    print("Step E: Two-pass optimization")
    print("="*70)
    optimized_estimates = two_pass_optimization(single_aa_estimates, multi_aa_blocks)
    
    # Create visualization
    if results:
        protein_id = args.protein or list(sequences.keys())[0]
        protein_sequence = sequences.get(protein_id, '')
        
        print(f"\nCreating validation plots for {protein_id}...")
        create_validation_plots(results, protein_sequence, protein_id, args.output)


if __name__ == '__main__':
    main()
