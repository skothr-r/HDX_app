#!/usr/bin/env python3
"""
Coverage Visualization Script

This script visualizes site-specific residue coverage from Comet CSV output.
Each residue is color-coded based on how many different fragment pairs
resulted in that single amino acid resolution.

Usage:
    python coverage_visualization.py <csv_file> [options]

Example:
    python coverage_visualization.py data/WT_nep2_0MUrea_08.csv --fasta data/Ube2D3.fasta
"""

import csv
import re
import argparse
import sys
from collections import defaultdict
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors
import numpy as np

def parse_fasta(fasta_file):
    """Parse FASTA file and return sequence dictionary."""
    sequences = {}
    current_id = None
    current_seq = []
    
    with open(fasta_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_id:
                    sequences[current_id] = ''.join(current_seq)
                current_id = line[1:].split()[0]  # Get first word after >
                current_seq = []
            else:
                current_seq.append(line)
        
        if current_id:
            sequences[current_id] = ''.join(current_seq)
    
    return sequences

def parse_csv_coverage(csv_file, count_peptides=False, ion_series_filter=None):
    """
    Parse CSV file and extract site-specific residue coverage.
    
    Args:
        csv_file: Path to CSV file
        count_peptides: If True, count unique peptides per position. If False, count fragment pairs.
        ion_series_filter: If 'c' or 'z', only count overhangs from that ion series. If None, count all.
    
    Returns:
        coverage: Dictionary mapping (protein, position) -> count
        protein_to_peptides: Dictionary mapping protein -> set of peptides
        stats: Dictionary with 'total_peptides' and 'peptides_no_fragments' counts
    """
    coverage = defaultdict(lambda: defaultdict(int))  # protein -> position -> count
    peptide_coverage = defaultdict(lambda: defaultdict(set))  # protein -> position -> set of peptides
    protein_to_peptides = defaultdict(set)  # Track which proteins we've seen
    stats = {
        'total_peptides': 0,  # Total peptides processed
        'peptides_with_c_or_z': 0,  # Peptides that have c and/or z ion types
        'peptides_with_overhangs': 0,  # Peptides that contributed to single-AA overhangs
        'peptides_with_multi_aa_overhangs': 0,  # Peptides that have multi-AA overhangs
        'peptides_with_multi_aa_only': 0,  # Peptides with multi-AA but not single-AA
        'peptides_with_single_aa_only': 0,  # Peptides with single-AA but not multi-AA
        'peptides_with_c_or_z_and_overhangs': 0,  # Peptides with c/z ions that also have overhangs
        'peptides_no_fragments': 0  # Peptides with no usable fragments
    }
    
    with open(csv_file, 'r') as f:
        lines = f.readlines()
        # Comet CSV has two header lines - second one is the actual header
        if len(lines) < 2:
            return coverage, protein_to_peptides
        
        # Get header from second line
        header_line = lines[1] if lines else ""
        header = [col.strip() for col in header_line.split(',')]
        
        # Find column indices
        try:
            protein_idx = header.index('protein')
            site_specific_idx = header.index('site_specific_residues')
            overhang_pairs_idx = header.index('single_aa_overhang_fragment_pairs')
            overhangs_idx = header.index('single_aa_overhangs')
            matched_ions_idx = header.index('matched fragment ions')
            multi_aa_overhangs_idx = header.index('multi_aa_overhangs')
        except ValueError as e:
            print(f"Error: Required column not found: {e}")
            return coverage, protein_to_peptides, stats
        
        # Parse data rows (start from line 2, since line 0 is empty and line 1 is header)
        for line in lines[2:]:
            # Parse quoted fields manually to handle complex CSV
            quoted_fields = re.findall(r'"([^"]*)"', line)
            
            # Also parse unquoted fields
            # Simple approach: split by comma, but need to handle quoted fields
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
            
            # Extract fields we need - use quoted fields for complex data
            protein = ""
            site_specific = ""
            overhang_pairs = ""
            overhangs = ""
            matched_ions = ""
            multi_aa_overhangs = ""
            
            # Try to get from row_data first
            if len(row_data) > protein_idx:
                protein = row_data[protein_idx].strip('"')
            if len(row_data) > site_specific_idx:
                site_specific = row_data[site_specific_idx].strip('"')
            if len(row_data) > matched_ions_idx:
                matched_ions = row_data[matched_ions_idx].strip('"')
            if len(row_data) > multi_aa_overhangs_idx:
                multi_aa_overhangs = row_data[multi_aa_overhangs_idx].strip('"')
            
            # For complex quoted fields, use the quoted fields directly
            # The quoted fields should include: matched fragment ions, overhangs, overhang_pairs, multi_aa_overhangs, site_specific
            # Count from the end: site_specific is last, multi_aa_overhangs is second to last, etc.
            if len(quoted_fields) >= 4:
                # Last 4 quoted fields in order: multi_aa_overhangs, overhangs, overhang_pairs, site_specific
                site_specific = quoted_fields[-1]  # Last one is site_specific_residues
                overhang_pairs = quoted_fields[-2] if len(quoted_fields) >= 2 else ""
                overhangs = quoted_fields[-3] if len(quoted_fields) >= 3 else ""
                multi_aa_overhangs = quoted_fields[-4] if len(quoted_fields) >= 4 else ""
            elif len(quoted_fields) >= 3:
                # Fallback: last 3 quoted fields in order: overhangs, overhang_pairs, site_specific
                site_specific = quoted_fields[-1]
                overhang_pairs = quoted_fields[-2] if len(quoted_fields) >= 2 else ""
                overhangs = quoted_fields[-3] if len(quoted_fields) >= 3 else ""
            
            # Get matched fragment ions (usually in quoted fields, but might be earlier)
            # Try to find it in quoted fields - it's usually one of the earlier ones
            if len(quoted_fields) >= 4:
                # matched fragment ions might be earlier in the quoted fields
                # Let's check all quoted fields for one that looks like ion names
                for qf in quoted_fields:
                    if qf and (',' in qf) and (qf.startswith('c') or qf.startswith('z') or 'z1_' in qf):
                        matched_ions = qf
                        break
            
            # If still not found, try row_data
            if not matched_ions and len(row_data) > matched_ions_idx:
                matched_ions = row_data[matched_ions_idx].strip('"').strip()
            
            # Get protein from row_data (it's usually not quoted)
            if len(row_data) <= protein_idx:
                continue
            protein = row_data[protein_idx].strip('"').strip()
            
            # Get peptide sequence
            peptide_idx = header.index('plain_peptide') if 'plain_peptide' in header else -1
            peptide = ""
            if peptide_idx >= 0 and len(row_data) > peptide_idx:
                peptide = row_data[peptide_idx].strip('"').strip()
            
            # Clean up protein name (remove quotes, trailing numbers)
            protein = protein.strip('"').strip()
            # Remove trailing numbers (e.g., "sp|P61077|UB2D3_HUMAN1" -> "sp|P61077|UB2D3_HUMAN")
            protein = re.sub(r'(\d+)$', '', protein)
            
            site_specific = site_specific.strip('"').strip()
            
            if not protein:
                continue
            
            # Count all peptides (even those without usable fragments)
            stats['total_peptides'] += 1
            
            # Track this peptide
            if peptide:
                protein_to_peptides[protein].add(peptide)
            
            # Check if peptide has c or z ion types (regardless of overhangs)
            # This needs to be checked early, before any continue statements
            has_c_or_z_ions = False
            if matched_ions and matched_ions.strip():
                # Parse matched ions to check for c or z
                ions_list = matched_ions.split(',')
                for ion in ions_list:
                    ion = ion.strip()
                    # Check for c ions (but not c- which would be a negative number)
                    if ion.startswith('c') and not ion.startswith('c-'):
                        has_c_or_z_ions = True
                        break
                    # Check for z ions (but not z1_ which we check separately)
                    elif ion.startswith('z') and not ion.startswith('z1'):
                        has_c_or_z_ions = True
                        break
                    # Check for z1 ions
                    elif 'z1_' in ion:
                        has_c_or_z_ions = True
                        break
            
            if has_c_or_z_ions:
                stats['peptides_with_c_or_z'] += 1
            
            # Check if this peptide has usable fragments
            has_usable_fragments = False
            
            # If no site_specific, this peptide has no usable fragments
            if not site_specific or not site_specific.strip():
                stats['peptides_no_fragments'] += 1
                continue
            
            # Parse site-specific residues: "45R,67T,89K"
            positions = re.findall(r'(\d+)([A-Z])', site_specific)
            
            if not positions:
                # Has site_specific field but no valid positions - no usable fragments
                stats['peptides_no_fragments'] += 1
                continue
            
            # Filter by ion series if requested - need to match positions to filtered pairs
            # The single_aa_overhangs, single_aa_overhang_fragment_pairs, and site_specific_residues
            # are all comma-separated and correspond to each other by index
            if ion_series_filter and overhang_pairs:
                # Use the overhangs we already extracted
                overhangs_str = overhangs.strip().strip('"') if overhangs else ""
                if not overhangs_str:
                    continue
                    
                overhang_list = [o.strip() for o in overhangs_str.split(',') if o.strip()]
                pairs_list = [p.strip() for p in overhang_pairs.split(',') if p.strip()]
                
                # Map each overhang/pair group to its position
                # Each comma-separated group corresponds to one position
                filtered_positions = set()
                filtered_pairs_list = []
                
                for i, pair_group in enumerate(pairs_list):
                    # Each group may have multiple pairs separated by |
                    individual_pairs = pair_group.split('|')
                    filtered_group = []
                    has_matching_pair = False
                    
                    for pair in individual_pairs:
                        pair = pair.strip()
                        # Check if pair matches ion series filter
                        matches_filter = False
                        if ion_series_filter == 'c' and pair.startswith('c') and not pair.startswith('c-'):
                            matches_filter = True
                        elif ion_series_filter == 'z' and (pair.startswith('z') or 'z1_' in pair):
                            matches_filter = True
                        
                        if matches_filter:
                            filtered_group.append(pair)
                            has_matching_pair = True
                    
                    # If this overhang group has matching pairs, track its position
                    if has_matching_pair:
                        filtered_pairs_list.append('|'.join(filtered_group))
                        # The position index i corresponds to positions[i] in site_specific
                        if i < len(positions):
                            pos_str, residue = positions[i]
                            filtered_positions.add(int(pos_str))
                
                # Only proceed if we have matching pairs
                if not filtered_pairs_list:
                    # This peptide has fragments but none match the filter
                    stats['peptides_no_fragments'] += 1
                    continue
                
                # Only count positions that are resolved by filtered pairs
                positions = [(pos_str, residue) for pos_str, residue in positions 
                            if int(pos_str) in filtered_positions]
                
                if not positions:
                    # This peptide had matching pairs but no positions after filtering
                    stats['peptides_no_fragments'] += 1
                    continue
                
                # Update overhang_pairs to only include filtered ones
                overhang_pairs = ','.join(filtered_pairs_list)
                has_usable_fragments = True
            
            # Check if we have usable fragments (if not already set by filtering)
            if not has_usable_fragments:
                if overhang_pairs and overhang_pairs.strip():
                    has_usable_fragments = True
                else:
                    # No overhang pairs at all
                    stats['peptides_no_fragments'] += 1
                    continue  # Skip this peptide if no usable fragments
            
            # Check for multi-AA overhangs (check this early, before we skip peptides)
            has_multi_aa_overhangs = False
            if multi_aa_overhangs and multi_aa_overhangs.strip():
                # Check if multi-AA overhangs field has actual data
                multi_list = [m.strip() for m in multi_aa_overhangs.split(',') if m.strip()]
                if multi_list:
                    has_multi_aa_overhangs = True
                    stats['peptides_with_multi_aa_overhangs'] += 1
            
            # Now categorize peptides based on what they have
            if has_usable_fragments:
                # Has single-AA overhangs
                stats['peptides_with_overhangs'] += 1
                # If this peptide also has c or z ions, count it
                if has_c_or_z_ions:
                    stats['peptides_with_c_or_z_and_overhangs'] += 1
                
                # Check if it has single-AA only or both
                if has_multi_aa_overhangs:
                    # Has both single-AA and multi-AA overhangs
                    pass  # Counted in both categories
                else:
                    # Has single-AA but not multi-AA
                    stats['peptides_with_single_aa_only'] += 1
            elif has_multi_aa_overhangs:
                # Has multi-AA but not single-AA overhangs
                stats['peptides_with_multi_aa_only'] += 1
                # Don't count this in peptides_with_overhangs (that's for single-AA only)
                # But we've already counted it in peptides_with_multi_aa_overhangs
            
            if count_peptides:
                # Count unique peptides per position (already filtered above)
                for pos_str, residue in positions:
                    pos = int(pos_str)
                    if peptide:
                        peptide_coverage[protein][pos].add(peptide)
            else:
                # Count fragment pairs for each position
                all_pairs = []
                if overhang_pairs:
                    # Split by comma (different overhangs), then by | (different pairs for same overhang)
                    for overhang_group in overhang_pairs.split(','):
                        pairs = [p.strip() for p in overhang_group.split('|') if p.strip()]
                        all_pairs.extend(pairs)
                
                # Count unique fragment pairs per position
                pair_count = len(set(all_pairs)) if all_pairs else 0
                
                # For each position, add the count (only if we have pairs)
                if pair_count > 0:
                    for pos_str, residue in positions:
                        pos = int(pos_str)
                        # Increment count for this position
                        coverage[protein][pos] += pair_count
    
    # If counting peptides, convert sets to counts
    if count_peptides:
        for protein in peptide_coverage:
            for pos in peptide_coverage[protein]:
                coverage[protein][pos] = len(peptide_coverage[protein][pos])
    
    return coverage, protein_to_peptides, stats

def create_visualization(sequences, coverage, output_file='coverage_heatmap.png', 
                        protein_id=None, max_length=200, count_type='fragment pairs', stats=None):
    """
    Create a heatmap visualization of residue coverage.
    
    Args:
        sequences: Dictionary of protein_id -> sequence
        coverage: Dictionary of protein_id -> position -> count
        output_file: Output filename for the figure
        protein_id: Specific protein to visualize (if None, visualizes all)
        max_length: Maximum sequence length to display (truncate if longer)
        count_type: String describing what is being counted (for title/labels)
    """
    # If protein_id specified, only visualize that one
    if protein_id:
        if protein_id not in sequences:
            print(f"Error: Protein {protein_id} not found in FASTA file")
            return
        proteins_to_plot = [protein_id]
    else:
        proteins_to_plot = list(sequences.keys())
    
    # Create figure with subplots for each protein
    n_proteins = len(proteins_to_plot)
    fig, axes = plt.subplots(n_proteins, 1, figsize=(16, max(4, n_proteins * 2)))
    
    if n_proteins == 1:
        axes = [axes]
    
    # Find global max count for consistent color scaling
    global_max = 0
    for protein_id in proteins_to_plot:
        if protein_id in coverage:
            global_max = max(global_max, max(coverage[protein_id].values()) if coverage[protein_id] else 0)
    
    # If no coverage found, set a default max
    if global_max == 0:
        global_max = 1
    
    # Create custom colormap: orange for 0, blue gradient for 1+
    from matplotlib.colors import ListedColormap
    # Create color list: orange for 0, then blue gradient
    color_list = ['#FF8C00']  # Orange for 0 coverage
    # Add blue gradient colors (from light blue to dark blue)
    n_blue_colors = 255
    for i in range(n_blue_colors):
        # Map i from [0, n_blue_colors-1] to [0.3, 1.0] for blue colormap
        blue_val = 0.3 + (i / (n_blue_colors - 1)) * 0.7
        color_list.append(plt.cm.Blues(blue_val))
    
    cmap = ListedColormap(color_list)
    # Normalize: 0 maps to index 0 (orange), 1+ maps to blue gradient indices
    norm = plt.Normalize(vmin=0, vmax=global_max)
    
    for idx, protein_id in enumerate(proteins_to_plot):
        ax = axes[idx]
        sequence = sequences[protein_id]
        
        # Truncate if too long
        if len(sequence) > max_length:
            sequence = sequence[:max_length]
            print(f"Warning: Sequence truncated to {max_length} residues for visualization")
        
        # Get coverage for this protein
        protein_coverage = coverage.get(protein_id, {})
        
        # Calculate coverage statistics for legend
        total_residues = len(sequence)
        unresolved = sum(1 for i in range(1, total_residues + 1) if protein_coverage.get(i, 0) == 0)
        one_peptide = sum(1 for i in range(1, total_residues + 1) if protein_coverage.get(i, 0) == 1)
        two_peptides = sum(1 for i in range(1, total_residues + 1) if protein_coverage.get(i, 0) == 2)
        three_plus = sum(1 for i in range(1, total_residues + 1) if protein_coverage.get(i, 0) >= 3)
        
        pct_unresolved = (unresolved / total_residues * 100) if total_residues > 0 else 0
        pct_one = (one_peptide / total_residues * 100) if total_residues > 0 else 0
        pct_two = (two_peptides / total_residues * 100) if total_residues > 0 else 0
        pct_three_plus = (three_plus / total_residues * 100) if total_residues > 0 else 0
        
        # Create background rectangles for each residue
        y_pos = 0.5
        bar_height = 0.8
        
        for i, residue in enumerate(sequence, 1):
            count = protein_coverage.get(i, 0)
            # Map count to color: 0 -> orange (index 0), 1+ -> blue gradient
            color = cmap(norm(count))
            
            # Draw colored rectangle
            rect = Rectangle((i-0.5, y_pos - bar_height/2), 1, bar_height, 
                           facecolor=color, edgecolor='gray', linewidth=0.5)
            ax.add_patch(rect)
            
            # Add residue letter
            # Use white text for dark blue (high coverage), black for orange/light blue
            text_color = 'white' if count > global_max * 0.5 else 'black'
            ax.text(i, y_pos, residue, ha='center', va='center', 
                   fontsize=8, fontweight='bold' if count > 0 else 'normal',
                   color=text_color)
        
        # Set axis properties (extend y-axis downward to accommodate legend)
        ax.set_xlim(0, len(sequence) + 1)
        ax.set_ylim(-0.4, 1)  # Extended downward to make room for legend
        ax.set_xlabel('Residue Position', fontsize=12)
        ax.set_title(f'{protein_id} - Site-Specific Residue Coverage\n'
                    f'(Color intensity = number of {count_type} resolving each position)',
                    fontsize=14, fontweight='bold')
        ax.set_yticks([])
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add position numbers every 10 residues
        positions_to_label = range(10, len(sequence) + 1, 10)
        for pos in positions_to_label:
            ax.axvline(x=pos, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)
        
        # Add statistics text box
        if stats:
            stats_text = f"Total peptides (all ion types): {stats['total_peptides']}\n"
            stats_text += f"Peptides with c and/or z ions: {stats['peptides_with_c_or_z']}\n"
            if stats['peptides_with_c_or_z'] > 0:
                pct = (stats['peptides_with_c_or_z_and_overhangs'] / stats['peptides_with_c_or_z'] * 100) if stats['peptides_with_c_or_z'] > 0 else 0
                stats_text += f"  → With single-AA overhangs: {stats['peptides_with_c_or_z_and_overhangs']} ({pct:.1f}%)\n"
            stats_text += f"\nPeptides with multi-AA overhangs: {stats['peptides_with_multi_aa_overhangs']}\n"
            stats_text += f"  → Multi-AA only (no single-AA): {stats['peptides_with_multi_aa_only']}\n"
            stats_text += f"  → Single-AA only (no multi-AA): {stats['peptides_with_single_aa_only']}\n"
            stats_text += f"\nPeptides with no usable fragments: {stats['peptides_no_fragments']}"
            
            # Position text box in upper left
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   fontsize=8, verticalalignment='top', horizontalalignment='left',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Add legend text box with coverage statistics below the plot
        legend_text = (
            f'Coverage Breakdown:\n'
            f'  Unresolved (0 peptides): {pct_unresolved:.1f}%\n'
            f'  1 peptide: {pct_one:.1f}%\n'
            f'  2 peptides: {pct_two:.1f}%\n'
            f'  3+ peptides: {pct_three_plus:.1f}%'
        )
        
        # Position legend below the plot (in data coordinates, below y=0)
        # Convert to axes coordinates for consistent positioning
        ax.text(len(sequence) / 2, -0.15, legend_text, transform=ax.get_xaxis_transform(),
               fontsize=10, verticalalignment='top', horizontalalignment='center',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8, edgecolor='black', linewidth=1.5),
               family='monospace')
    
    # Add colorbar with custom colormap below the plots with extra padding
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    # Increase pad significantly to move colorbar further down
    cbar = plt.colorbar(sm, ax=axes, orientation='horizontal', pad=0.25, aspect=40)
    cbar.set_label(f'Number of {count_type.title()} Resolving Position (0=orange, 1+=blue gradient)', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Coverage visualization saved to: {output_file}")
    
    # Also print summary statistics
    print("\nCoverage Summary:")
    print("=" * 70)
    for protein_id in proteins_to_plot:
        sequence = sequences[protein_id]
        protein_coverage = coverage.get(protein_id, {})
        total_positions = len(sequence)
        covered_positions = len(protein_coverage)
        coverage_percent = (covered_positions / total_positions * 100) if total_positions > 0 else 0
        avg_count = np.mean(list(protein_coverage.values())) if protein_coverage else 0
        max_count = max(protein_coverage.values()) if protein_coverage else 0
        
        print(f"\n{protein_id}:")
        print(f"  Total residues: {total_positions}")
        print(f"  Site-specifically resolved: {covered_positions} ({coverage_percent:.1f}%)")
        print(f"  Average {count_type} per resolved position: {avg_count:.2f}")
        print(f"  Maximum {count_type} for any position: {max_count}")

def main():
    parser = argparse.ArgumentParser(
        description='Visualize site-specific residue coverage from Comet CSV output',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('csv_file', help='Input CSV file from Comet')
    parser.add_argument('--fasta', required=True, help='FASTA file with protein sequences')
    parser.add_argument('--output', '-o', default='coverage_heatmap.png',
                       help='Output filename for visualization (default: coverage_heatmap.png)')
    parser.add_argument('--protein', '-p', help='Specific protein ID to visualize (if not specified, all proteins are shown)')
    parser.add_argument('--max-length', type=int, default=200,
                       help='Maximum sequence length to display (default: 200)')
    parser.add_argument('--count-fragments', action='store_true',
                       help='Count fragment pairs per position instead of unique peptides (default: count peptides)')
    parser.add_argument('--ion-series', choices=['c', 'z', 'both'], default='both',
                       help='Filter by ion series: c, z, or both (default: both - creates separate plots)')
    
    args = parser.parse_args()
    
    # Parse FASTA file
    print(f"Reading FASTA file: {args.fasta}")
    sequences = parse_fasta(args.fasta)
    if not sequences:
        print("Error: No sequences found in FASTA file")
        sys.exit(1)
    print(f"Found {len(sequences)} protein(s) in FASTA file")
    
    # Parse CSV coverage
    print(f"Reading CSV file: {args.csv_file}")
    count_peptides = not args.count_fragments  # Default to peptides unless --count-fragments is specified
    count_type = 'peptides' if count_peptides else 'fragment pairs'
    print(f"Counting: {count_type}")
    
    # Create separate plots for c and z if requested
    if args.ion_series == 'both':
        # Create three plots: all, c-only, z-only
        print("Creating separate plots for all ions, C ions only, and Z ions only...")
        coverage_all, protein_to_peptides, stats_all = parse_csv_coverage(args.csv_file, count_peptides=count_peptides, ion_series_filter=None)
        coverage_c, _, stats_c = parse_csv_coverage(args.csv_file, count_peptides=count_peptides, ion_series_filter='c')
        coverage_z, _, stats_z = parse_csv_coverage(args.csv_file, count_peptides=count_peptides, ion_series_filter='z')
        
        print(f"Found coverage data for {len(coverage_all)} protein(s)")
        
        # Create visualizations for each
        # All ions
        create_visualization(sequences, coverage_all, args.output.replace('.png', '_all.png'), 
                           args.protein, args.max_length, count_type, stats_all)
        
        # C ions only
        create_visualization(sequences, coverage_c, args.output.replace('.png', '_c_ions.png'), 
                           args.protein, args.max_length, count_type + " (C ions only)", stats_c)
        
        # Z ions only
        create_visualization(sequences, coverage_z, args.output.replace('.png', '_z_ions.png'), 
                           args.protein, args.max_length, count_type + " (Z ions only)", stats_z)
        return
    else:
        # Single plot with filter
        coverage, protein_to_peptides, stats = parse_csv_coverage(args.csv_file, count_peptides=count_peptides, 
                                                          ion_series_filter=args.ion_series)
        print(f"Found coverage data for {len(coverage)} protein(s)")
    
    # Try to match protein IDs (handle different naming conventions)
    csv_proteins = set(coverage.keys())
    fasta_proteins = set(sequences.keys())
    
    # Try exact match first
    common_proteins = csv_proteins & fasta_proteins
    
    # If no exact match, try partial matching
    if not common_proteins:
        print("No exact protein ID matches found. Attempting partial matching...")
        print(f"CSV proteins: {list(csv_proteins)[:3]}")
        print(f"FASTA proteins: {list(fasta_proteins)[:3]}")
        
        # Try to match by partial name (e.g., "UB2D3" in both)
        protein_mapping = {}
        for csv_prot in csv_proteins:
            for fasta_prot in fasta_proteins:
                # Extract key identifier (e.g., "UB2D3" from "sp|P61077|UB2D3_HUMAN")
                csv_key = csv_prot.split('|')[-1] if '|' in csv_prot else csv_prot.split('_')[0]
                fasta_key = fasta_prot.split('|')[-1] if '|' in fasta_prot else fasta_prot.split('_')[0]
                
                # Remove trailing digits (e.g., "UB2D3_HUMAN1" -> "UB2D3_HUMAN")
                csv_key = re.sub(r'\d+$', '', csv_key)
                fasta_key = re.sub(r'\d+$', '', fasta_key)
                
                # Also try matching the base name without suffix
                csv_base = csv_key.split('_')[0] if '_' in csv_key else csv_key
                fasta_base = fasta_key.split('_')[0] if '_' in fasta_key else fasta_key
                
                if csv_key == fasta_key or csv_base == fasta_base or csv_key in fasta_key or fasta_key in csv_key:
                    protein_mapping[csv_prot] = fasta_prot
                    common_proteins.add(fasta_prot)
                    print(f"  Matched: '{csv_prot}' -> '{fasta_prot}'")
                    break
        
        # Remap coverage to use FASTA protein IDs
        if protein_mapping:
            remapped_coverage = {}
            for csv_prot, fasta_prot in protein_mapping.items():
                remapped_coverage[fasta_prot] = coverage[csv_prot]
            coverage = remapped_coverage
    
    if not common_proteins:
        print("Error: Could not match any proteins between CSV and FASTA files")
        print("Please check that protein IDs match or use --protein to specify a specific protein")
        sys.exit(1)
    else:
        print(f"Found {len(common_proteins)} matching protein(s)")
    
    # If specific protein requested, verify it exists
    if args.protein:
        if args.protein not in sequences:
            print(f"Error: Protein '{args.protein}' not found in FASTA file")
            print(f"Available proteins: {list(sequences.keys())}")
            sys.exit(1)
    
    # Create visualization
    print("\nGenerating visualization...")
    create_visualization(sequences, coverage, args.output, args.protein, args.max_length, count_type)
    
    print("\nDone!")

if __name__ == '__main__':
    main()
