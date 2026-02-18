#!/usr/bin/env python3
"""
Extract Fragment Pairs to CSV

Extracts all fragment pairs from peptides and outputs to CSV:
- Single AA pairs (consecutive, differ by 1)
- Double AA pairs (differ by 2)
- Triple AA pairs (differ by 3)
"""

import argparse
import re
import csv
from collections import defaultdict
import sys
import os

# Import functions from multi_aa_overhang_visualization.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multi_aa_overhang_visualization import normalize_fragment_pair


def extract_fragment_pairs_by_length(matched_ions_str):
    """
    Extract fragment pairs grouped by length difference (1, 2, 3).
    
    Args:
        matched_ions_str: Comma-separated string of fragment ions (e.g., "c4,c5,c6,z8,z9,z10")
    
    Returns:
        Dictionary with keys 'single', 'double', 'triple' mapping to lists of normalized pairs
    """
    if not matched_ions_str or not matched_ions_str.strip():
        return {'single': [], 'double': [], 'triple': []}
    
    # Parse fragment ions
    fragments = [f.strip() for f in matched_ions_str.split(',') if f.strip()]
    if not fragments:
        return {'single': [], 'double': [], 'triple': []}
    
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
    
    # Find all pairs within each ion series
    single_pairs = []
    double_pairs = []
    triple_pairs = []
    
    for series, frag_list in fragments_by_series.items():
        # Sort by fragment number
        frag_list.sort(key=lambda x: x[0])
        
        # Find all pairs
        for i in range(len(frag_list)):
            for j in range(i + 1, len(frag_list)):
                frag1_num, frag1_str = frag_list[i]
                frag2_num, frag2_str = frag_list[j]
                
                # Calculate difference
                diff = frag2_num - frag1_num
                
                if diff == 1:
                    # Single AA pair (consecutive)
                    pair_str = f"{frag1_str}-{frag2_str}"
                    normalized_pair = normalize_fragment_pair(pair_str)
                    single_pairs.append(normalized_pair)
                elif diff == 2:
                    # Double AA pair
                    pair_str = f"{frag1_str}-{frag2_str}"
                    normalized_pair = normalize_fragment_pair(pair_str)
                    double_pairs.append(normalized_pair)
                elif diff == 3:
                    # Triple AA pair
                    pair_str = f"{frag1_str}-{frag2_str}"
                    normalized_pair = normalize_fragment_pair(pair_str)
                    triple_pairs.append(normalized_pair)
    
    return {
        'single': sorted(set(single_pairs)),
        'double': sorted(set(double_pairs)),
        'triple': sorted(set(triple_pairs))
    }


def extract_fragment_pairs_to_csv(csv_file, protein_id, output_file, no_filtering=False):
    """
    Extract fragment pairs from CSV and output to new CSV.
    Uses the same filtering logic as combined_overhang_visualization.py
    
    Args:
        csv_file: Input CSV file with peptide data
        protein_id: Protein ID to filter by (optional, None for all)
        output_file: Output CSV file path
        no_filtering: If True, include all peptides (same as RT grid with no_filtering=True)
    """
    # Import filtering function
    from multi_aa_overhang_visualization import get_filtered_peptide_indices, parse_fasta
    
    peptides_data = []
    
    # Get protein length for filtering
    fasta_file = None  # Will need to be passed if filtering is used
    protein_length = 0
    if not no_filtering:
        # Try to find FASTA file in common locations
        import os
        csv_dir = os.path.dirname(os.path.abspath(csv_file))
        fasta_candidates = [
            os.path.join(csv_dir, 'Ube2D3.fasta'),
            os.path.join(os.path.dirname(csv_dir), 'Ube2D3.fasta'),
            os.path.join(csv_dir, '..', 'data', 'Ube2D3.fasta'),
        ]
        for candidate in fasta_candidates:
            if os.path.exists(candidate):
                fasta_file = candidate
                break
        
        if fasta_file:
            sequences = parse_fasta(fasta_file)
            if protein_id and protein_id in sequences:
                protein_length = len(sequences[protein_id])
            elif sequences:
                protein_length = len(list(sequences.values())[0])
    
    # Get filtered indices (same as RT grid)
    filtered_indices = None
    if not no_filtering and fasta_file and protein_length > 0:
        filtered_indices, _, _ = get_filtered_peptide_indices(csv_file, protein_id, protein_length, fasta_file, top_n_limit=20)
        print(f"Using {len(filtered_indices)} filtered peptides (top 10 c-ion + top 10 z-ion per position)")
    else:
        print(f"No filtering: including all peptides with fragment pairs")
    
    with open(csv_file, 'r') as f:
        lines = f.readlines()
        
        if len(lines) < 2:
            print("Error: CSV file too short")
            return
        
        header_line = lines[1]
        header = [col.strip() for col in header_line.split(',')]
        
        try:
            protein_idx = header.index('protein')
            rt_idx = header.index('retention_time_sec')
            peptide_idx = header.index('plain_peptide')
            try:
                matched_ions_idx = header.index('matched fragment ions')
            except ValueError:
                print("Error: 'matched fragment ions' column not found")
                return
            # Optional columns for metrics
            try:
                xcorr_idx = header.index('xcorr')
            except ValueError:
                xcorr_idx = -1
            try:
                charge_idx = header.index('charge')
            except ValueError:
                charge_idx = -1
            try:
                intensities_idx = header.index('matched fragment ion intensities')
            except ValueError:
                intensities_idx = -1
            # Try to find q-value column
            qvalue_idx = -1
            for col_name in ['percolator_qvalue', 'q-value', 'qvalue', 'FDR', 'fdr', 'percolator_q-value']:
                try:
                    qvalue_idx = header.index(col_name)
                    break
                except ValueError:
                    continue
        except ValueError as e:
            print(f"Error: Required column not found: {e}")
            return
        
        # Use csv.reader for proper CSV parsing
        import io
        csv_content = ''.join(lines[2:])
        csv_reader = csv.reader(io.StringIO(csv_content))
        
        for row_idx, row_data in enumerate(csv_reader):
            if not row_data or len(row_data) == 0:
                continue
            
            if len(row_data) <= max(protein_idx, rt_idx, peptide_idx):
                continue
            
            protein = row_data[protein_idx] if protein_idx < len(row_data) else ""
            if not protein:
                continue
            
            # Filter by protein_id if specified
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
            
            # Apply same filtering as RT grid (unless no_filtering is True)
            if not no_filtering and filtered_indices is not None and row_idx not in filtered_indices:
                continue
            
            # Get retention time
            try:
                rt_str = row_data[rt_idx] if rt_idx < len(row_data) else ""
                rt = float(rt_str)
            except (ValueError, IndexError):
                continue
            
            # Get peptide sequence
            peptide_seq = ""
            if peptide_idx >= 0 and len(row_data) > peptide_idx:
                peptide_seq = row_data[peptide_idx].strip('"').strip()
            
            # Get q-value (quality score)
            qvalue = None
            if qvalue_idx >= 0 and qvalue_idx < len(row_data):
                try:
                    qvalue_str = row_data[qvalue_idx].strip()
                    if qvalue_str:
                        raw_value = float(qvalue_str)
                        # Auto-detect format and convert to true q-value (0-1)
                        if raw_value > 1.0:
                            if raw_value <= 100.0:
                                # Likely percentage
                                qvalue = raw_value / 100.0
                            elif raw_value > 10:
                                # Likely q-score (-10*log10 format)
                                qvalue = 10.0 ** (-raw_value / 10.0)
                            else:
                                # Likely q-score (-log10 format)
                                qvalue = 10.0 ** (-raw_value)
                        else:
                            # True q-value (already in [0,1])
                            qvalue = raw_value
                except (ValueError, IndexError):
                    pass
            
            # Get XCorr
            xcorr = None
            if xcorr_idx >= 0 and xcorr_idx < len(row_data):
                try:
                    xcorr_str = row_data[xcorr_idx].strip('"').strip()
                    if xcorr_str:
                        xcorr = float(xcorr_str)
                except (ValueError, IndexError):
                    pass
            
            # Get charge
            charge = None
            if charge_idx >= 0 and charge_idx < len(row_data):
                try:
                    charge_str = row_data[charge_idx].strip('"').strip()
                    if charge_str:
                        charge = int(float(charge_str))
                except (ValueError, IndexError):
                    pass
            
            # Get signal intensity (sum of matched fragment ion intensities)
            signal_intensity = None
            if intensities_idx >= 0 and intensities_idx < len(row_data):
                try:
                    intensities_str = row_data[intensities_idx].strip('"').strip()
                    if intensities_str:
                        intensities = [float(x.strip()) for x in intensities_str.split(',') if x.strip()]
                        signal_intensity = sum(intensities)
                except (ValueError, IndexError):
                    pass
            
            # Get matched fragment ions from quoted fields (more reliable)
            matched_ions_str = ""
            original_line = lines[2 + row_idx] if (2 + row_idx) < len(lines) else ""
            quoted_fields = re.findall(r'"([^"]*)"', original_line)
            
            # Find matched fragment ions in quoted fields (usually one of the first fields)
            for qf in quoted_fields:
                if qf and (',' in qf) and (qf.startswith('c') or qf.startswith('z') or 'z1_' in qf or 
                            qf.startswith('a') or qf.startswith('b') or qf.startswith('x') or qf.startswith('y')):
                    # Verify it's actually fragment ions (contains at least one valid fragment pattern)
                    fragments_test = [f.strip() for f in qf.split(',') if f.strip()]
                    valid_frag = False
                    for frag in fragments_test[:3]:  # Check first 3 fragments
                        frag_lower = frag.lower()
                        if (re.match(r'^[a-z]\d+$', frag_lower) or 
                            re.match(r'^z1_\d+$', frag_lower)):
                            valid_frag = True
                            break
                    if valid_frag:
                        matched_ions_str = qf
                        break
            
            # Fallback to row_data if not found in quoted fields
            if not matched_ions_str and matched_ions_idx >= 0 and matched_ions_idx < len(row_data):
                matched_ions_str = row_data[matched_ions_idx].strip('"').strip()
            
            if not matched_ions_str:
                continue
            
            # Validate that matched_ions_str contains actual fragment ions
            # Check if it contains at least one valid fragment pattern
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
            
            # Store data with all metrics
            peptides_data.append({
                'retention_time': rt,
                'sequence': peptide_seq,
                'fragments': matched_ions_str,
                'single_aa_pairs': '|'.join(pairs_by_length['single']),
                'double_aa_pairs': '|'.join(pairs_by_length['double']),
                'triple_aa_pairs': '|'.join(pairs_by_length['triple']),
                'qvalue': qvalue if qvalue is not None else '',
                'xcorr': xcorr if xcorr is not None else '',
                'charge': charge if charge is not None else '',
                'signal_intensity': signal_intensity if signal_intensity is not None else ''
            })
    
    # Only include peptides that have single-AA or 2-AA pairs (same as RT grid logic)
    # RT grid only shows peptides with single-AA or 2-AA overhangs (not triple or higher)
    peptides_with_pairs = [p for p in peptides_data if p['single_aa_pairs'] or p['double_aa_pairs']]
    
    # Count total unique fragment pairs (matching RT grid histogram logic)
    all_single_pairs = set()
    all_double_pairs = set()
    all_triple_pairs = set()
    
    for p in peptides_with_pairs:
        if p['single_aa_pairs']:
            all_single_pairs.update(p['single_aa_pairs'].split('|'))
        if p['double_aa_pairs']:
            all_double_pairs.update(p['double_aa_pairs'].split('|'))
        if p['triple_aa_pairs']:
            all_triple_pairs.update(p['triple_aa_pairs'].split('|'))
    
    # Write only peptides with pairs
    with open(output_file, 'w', newline='') as f:
        fieldnames = ['retention_time', 'sequence', 'fragments', 'single_aa_pairs', 'double_aa_pairs', 'triple_aa_pairs', 
                     'qvalue', 'xcorr', 'charge', 'signal_intensity']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(peptides_with_pairs)
    
    print(f"Extracted {len(peptides_with_pairs)} peptides with fragment pairs to {output_file}")
    print(f"  - Peptides with single AA pairs: {sum(1 for p in peptides_with_pairs if p['single_aa_pairs'])}")
    print(f"  - Peptides with double AA pairs: {sum(1 for p in peptides_with_pairs if p['double_aa_pairs'])}")
    print(f"  - Peptides with triple AA pairs: {sum(1 for p in peptides_with_pairs if p['triple_aa_pairs'])}")
    print(f"  - Total unique single AA pairs: {len(all_single_pairs)}")
    print(f"  - Total unique double AA pairs: {len(all_double_pairs)}")
    print(f"  - Total unique triple AA pairs: {len(all_triple_pairs)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract fragment pairs to CSV')
    parser.add_argument('csv_file', help='Input CSV file')
    parser.add_argument('--protein_id', help='Protein ID to filter by (optional)')
    parser.add_argument('--output', '-o', help='Output CSV file', default='fragment_pairs_output.csv')
    parser.add_argument('--no-filtering', action='store_true', help='Include all peptides (no filtering)')
    
    args = parser.parse_args()
    
    extract_fragment_pairs_to_csv(args.csv_file, args.protein_id, args.output, no_filtering=args.no_filtering)
