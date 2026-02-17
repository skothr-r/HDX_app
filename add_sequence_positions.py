#!/usr/bin/env python3
"""
Add sequence positions column as the first column in CSV.

This script:
1. Reads the Comet CSV output
2. Reads the FASTA file to get protein sequences
3. For each peptide, finds its start position in the protein
4. Calculates all sequence positions (start, start+1, ..., start+len-1)
5. Adds them as a comma-separated list in the first column

Usage:
    python add_sequence_positions.py <comet_csv> <fasta_file> [output_csv]

Requirements:
    - pandas (pip install pandas)
"""

import sys
import os
import pandas as pd
import re

def read_fasta(fasta_file):
    """Read FASTA file and return dictionary of protein_id -> sequence."""
    protein_sequences = {}
    current_id = None
    current_seq = []
    
    with open(fasta_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                # Save previous sequence
                if current_id and current_seq:
                    protein_sequences[current_id] = ''.join(current_seq)
                
                # Parse new header
                # Format: >sp|P61077|UB2D3_HUMAN ...
                # Extract protein ID (e.g., "sp|P61077|UB2D3_HUMAN")
                header_parts = line[1:].split()
                if header_parts:
                    current_id = header_parts[0]
                else:
                    current_id = line[1:]
                current_seq = []
            else:
                if line:
                    current_seq.append(line)
        
        # Save last sequence
        if current_id and current_seq:
            protein_sequences[current_id] = ''.join(current_seq)
    
    return protein_sequences

def find_peptide_start(peptide_seq, protein_seq):
    """Find the start position (1-based) of peptide in protein sequence."""
    # Clean peptide sequence (remove modifications like [15.9949])
    clean_peptide = re.sub(r'\[.*?\]', '', peptide_seq)
    
    # Find first occurrence
    start_pos = protein_seq.find(clean_peptide)
    if start_pos == -1:
        return None  # Not found
    return start_pos + 1  # Convert to 1-based

def calculate_sequence_positions(peptide_seq, protein_seq):
    """
    Calculate all sequence positions for a peptide in the protein.
    Returns comma-separated string of positions, or empty string if not found.
    """
    start_pos = find_peptide_start(peptide_seq, protein_seq)
    if start_pos is None:
        return ""
    
    # Clean peptide to get length
    clean_peptide = re.sub(r'\[.*?\]', '', peptide_seq)
    peptide_length = len(clean_peptide)
    
    # Generate all positions: start, start+1, ..., start+length-1
    positions = [str(start_pos + i) for i in range(peptide_length)]
    return ','.join(positions)

def main():
    if len(sys.argv) < 3:
        print("Usage: python add_sequence_positions.py <comet_csv> <fasta_file> [output_csv]")
        print("\nExample:")
        print("  python add_sequence_positions.py results.csv protein.fasta results_with_positions.csv")
        sys.exit(1)
    
    comet_csv = sys.argv[1]
    fasta_file = sys.argv[2]
    output_csv = sys.argv[3] if len(sys.argv) > 3 else comet_csv.replace('.csv', '_with_positions.csv')
    
    if not os.path.exists(comet_csv):
        print(f"Error: Comet CSV file not found: {comet_csv}")
        sys.exit(1)
    
    if not os.path.exists(fasta_file):
        print(f"Error: FASTA file not found: {fasta_file}")
        sys.exit(1)
    
    print("[DEBUG] Reading FASTA file...")
    protein_sequences = read_fasta(fasta_file)
    print(f"[DEBUG] Loaded {len(protein_sequences)} protein sequences")
    for protein_id, seq in list(protein_sequences.items())[:3]:
        print(f"[DEBUG]   {protein_id}: {len(seq)} amino acids")
    
    print(f"[DEBUG] Reading Comet CSV: {comet_csv}")
    # Skip first row if it's metadata (Comet version info)
    try:
        with open(comet_csv, 'r') as f:
            first_line = f.readline()
            skip_rows = 1 if 'CometVersion' in first_line else 0
            print(f"[DEBUG] {'Skipping' if skip_rows else 'Not skipping'} first row (Comet version header)")
    except:
        skip_rows = 0
    
    # Read CSV with Python engine for robust parsing
    df = pd.read_csv(comet_csv, sep=',', skiprows=skip_rows, engine='python', quotechar='"', on_bad_lines='warn')
    
    print(f"[DEBUG] CSV loaded: {len(df)} rows, {len(df.columns)} columns")
    print(f"[DEBUG] Columns: {list(df.columns[:10])}...")
    
    # Check required columns
    required_cols = ['plain_peptide', 'protein']
    for col in required_cols:
        if col not in df.columns:
            print(f"Error: Required column '{col}' not found in CSV")
            print(f"Available columns: {list(df.columns)}")
            sys.exit(1)
    
    print("[DEBUG] Calculating sequence positions for each peptide...")
    sequence_positions = []
    not_found_count = 0
    found_count = 0
    
    for idx, row in df.iterrows():
        if idx % 500 == 0 and idx > 0:
            print(f"[DEBUG]   Processed {idx}/{len(df)} rows...")
        
        peptide = row['plain_peptide']
        protein_id = row['protein']
        
        # Handle decoy proteins (DECOY_sp|P61077|UB2D3_HUMAN -> sp|P61077|UB2D3_HUMAN)
        if protein_id.startswith('DECOY_'):
            protein_id = protein_id[6:]  # Remove 'DECOY_' prefix
        
        # Get protein sequence
        protein_seq = protein_sequences.get(protein_id)
        
        if protein_seq is None:
            if not_found_count < 5:
                print(f"[DEBUG]   Warning: Protein '{protein_id}' not found in FASTA for row {idx}")
            sequence_positions.append("")
            not_found_count += 1
            continue
        
        # Calculate sequence positions
        positions_str = calculate_sequence_positions(peptide, protein_seq)
        sequence_positions.append(positions_str)
        
        if positions_str:
            found_count += 1
            if found_count <= 3:  # Debug first few
                print(f"[DEBUG]   Row {idx}: peptide='{peptide[:20]}...', protein='{protein_id}', positions='{positions_str[:50]}...'")
        else:
            not_found_count += 1
            if not_found_count <= 5:
                print(f"[DEBUG]   Warning: Peptide '{peptide}' not found in protein '{protein_id}' (row {idx})")
    
    print(f"[DEBUG] Calculation complete:")
    print(f"[DEBUG]   Rows with positions: {found_count}/{len(df)} ({100*found_count/len(df):.1f}%)")
    print(f"[DEBUG]   Rows without positions: {not_found_count}/{len(df)}")
    
    # Add sequence_positions as the first column
    print("[DEBUG] Adding sequence_positions as first column...")
    df.insert(0, 'sequence_positions', sequence_positions)
    
    print(f"[DEBUG] Writing output to: {output_csv}")
    
    # Write CSV with Comet version header if it existed
    with open(output_csv, 'w') as f:
        if skip_rows == 1:
            # Write original header line
            with open(comet_csv, 'r') as orig:
                first_line = orig.readline()
                f.write(first_line)
        
        # Write CSV
        df.to_csv(f, index=False, sep=',')
    
    print("[DEBUG] ========================================")
    print(f"[DEBUG] Done! Added sequence positions to {len(df)} rows.")
    print(f"[DEBUG] Output written to: {output_csv}")
    print("[DEBUG] ========================================")


if __name__ == '__main__':
    main()
