#!/usr/bin/env python3
"""
Merge peak windows data from extract_ms1_chromatograms.py into the main Comet CSV file.

Usage:
    python add_peak_windows_to_csv.py <comet_csv> <peak_windows_csv> [output_csv]

Example:
    python add_peak_windows_to_csv.py data/WT_nep2_0MUrea_08_with_ms1_perc_qvalues.csv output_chromatograms_peptides/peak_windows_all.csv data/WT_nep2_0MUrea_08_with_peak_windows.csv
"""

import sys
import os
import pandas as pd

def merge_peak_windows_into_csv(comet_csv, peak_windows_csv, output_csv):
    """Merge peak windows data into Comet CSV file."""
    print(f"[DEBUG] ========================================")
    print(f"[DEBUG] Merging peak windows data into CSV")
    print(f"[DEBUG] ========================================")
    
    if not os.path.exists(comet_csv):
        print(f"Error: Comet CSV file not found: {comet_csv}")
        return False
    
    if not os.path.exists(peak_windows_csv):
        print(f"Error: Peak windows CSV file not found: {peak_windows_csv}")
        return False
    
    # Read peak windows CSV
    print(f"[DEBUG] Reading peak windows CSV: {peak_windows_csv}")
    df_peak_windows = pd.read_csv(peak_windows_csv)
    print(f"[DEBUG] Loaded {len(df_peak_windows)} peak window entries")
    
    # Create lookup dictionary: (peptide, charge, modifications) -> peak_window_row
    peak_windows_lookup = {}
    for _, row in df_peak_windows.iterrows():
        peptide_seq = str(row.get('peptide', ''))
        charge = int(row.get('charge', 0)) if pd.notna(row.get('charge')) else 0
        mods = str(row.get('modifications', '-')) if pd.notna(row.get('modifications')) else '-'
        key = (peptide_seq, charge, mods)
        peak_windows_lookup[key] = row
    
    print(f"[DEBUG] Created lookup for {len(peak_windows_lookup)} unique peptides")
    
    # Read Comet CSV
    print(f"[DEBUG] Reading Comet CSV: {comet_csv}")
    # Check if first line is CometVersion header
    skip_rows = 0
    with open(comet_csv, 'r') as f:
        first_line = f.readline()
        if 'CometVersion' in first_line:
            skip_rows = 1
            print(f"[DEBUG] Detected Comet version header, skipping first row")
    
    try:
        df = pd.read_csv(comet_csv, sep=',', skiprows=skip_rows, engine='python', quotechar='"', on_bad_lines='warn')
    except Exception as e:
        print(f"[DEBUG] Error reading CSV: {e}, trying without skiprows")
        df = pd.read_csv(comet_csv, sep=',', engine='python', quotechar='"', on_bad_lines='warn')
    
    print(f"[DEBUG] CSV loaded: {len(df)} rows, {len(df.columns)} columns")
    
    # Check required columns
    required_cols = ['plain_peptide', 'charge', 'modifications']
    for col in required_cols:
        if col not in df.columns:
            print(f"Error: Required column '{col}' not found in CSV")
            print(f"Available columns: {list(df.columns)}")
            return False
    
    # Add peak window columns
    print(f"[DEBUG] Merging peak window data...")
    
    # Initialize new columns with None/NaN
    peak_window_columns = [
        'peak_window_status', 'peak_window_rejection_reason',
        'detected_peak_min_rt', 'detected_peak_max_rt', 'detected_peak_window_size',
        'collection_min_rt', 'collection_max_rt', 'collection_window_size',
        'spectrum_window_min_rt', 'spectrum_window_max_rt', 'spectrum_window_size',
        'apex_rt', 'anchor_rt', 'anchor_evalue',
        'n_isotopic_matches', 'matched_isotopes', 'using_anchor_window',
        'precursor_mz', 'sequence_start_pos', 'best_score', 'n_candidates',
        'total_area', 'coelution_score'
    ]
    
    for col in peak_window_columns:
        if col not in df.columns:
            df[col] = None
    
    # Map peak window columns from peak_windows CSV to Comet CSV
    column_mapping = {
        'status': 'peak_window_status',
        'rejection_reason': 'peak_window_rejection_reason',
        'detected_peak_min_rt': 'detected_peak_min_rt',
        'detected_peak_max_rt': 'detected_peak_max_rt',
        'detected_peak_window_size': 'detected_peak_window_size',
        'collection_min_rt': 'collection_min_rt',
        'collection_max_rt': 'collection_max_rt',
        'collection_window_size': 'collection_window_size',
        'spectrum_window_min_rt': 'spectrum_window_min_rt',
        'spectrum_window_max_rt': 'spectrum_window_max_rt',
        'spectrum_window_size': 'spectrum_window_size',
        'apex_rt': 'apex_rt',
        'anchor_rt': 'anchor_rt',
        'anchor_evalue': 'anchor_evalue',
        'n_isotopic_matches': 'n_isotopic_matches',
        'matched_isotopes': 'matched_isotopes',
        'using_anchor_window': 'using_anchor_window',
        'precursor_mz': 'precursor_mz',
        'sequence_start_pos': 'sequence_start_pos',
        'best_score': 'best_score',
        'n_candidates': 'n_candidates',
        'total_area': 'total_area',
        'coelution_score': 'coelution_score'
    }
    
    matched_count = 0
    unmatched_count = 0
    
    for idx, row in df.iterrows():
        peptide_seq = str(row.get('plain_peptide', ''))
        charge = int(row.get('charge', 0)) if pd.notna(row.get('charge')) else 0
        mods = str(row.get('modifications', '-')) if pd.notna(row.get('modifications')) else '-'
        
        # Try exact match first
        key = (peptide_seq, charge, mods)
        if key in peak_windows_lookup:
            peak_row = peak_windows_lookup[key]
            matched_count += 1
            
            # Copy all mapped columns
            for peak_col, comet_col in column_mapping.items():
                if peak_col in peak_row.index:
                    df.at[idx, comet_col] = peak_row[peak_col]
        else:
            unmatched_count += 1
    
    print(f"[DEBUG] Matched {matched_count} rows with peak window data")
    print(f"[DEBUG] Unmatched {unmatched_count} rows (no peak window data)")
    
    # Write output CSV
    print(f"[DEBUG] Writing output to: {output_csv}")
    
    with open(output_csv, 'w') as f:
        # Write Comet version header if it existed
        if skip_rows == 1:
            with open(comet_csv, 'r') as orig:
                first_line = orig.readline()
                f.write(first_line)
        
        # Write CSV
        df.to_csv(f, index=False, sep=',')
    
    print(f"[DEBUG] ========================================")
    print(f"[DEBUG] Done! Merged peak window data into {len(df)} rows.")
    print(f"[DEBUG] Output written to: {output_csv}")
    print(f"[DEBUG] ========================================")
    
    return True


def main():
    if len(sys.argv) < 3:
        print("Usage: python add_peak_windows_to_csv.py <comet_csv> <peak_windows_csv> [output_csv]")
        print("\nExample:")
        print("  python add_peak_windows_to_csv.py data/WT_nep2_0MUrea_08_with_ms1_perc_qvalues.csv \\")
        print("    output_chromatograms_peptides/peak_windows_all.csv \\")
        print("    data/WT_nep2_0MUrea_08_with_peak_windows.csv")
        sys.exit(1)
    
    comet_csv = sys.argv[1]
    peak_windows_csv = sys.argv[2]
    output_csv = sys.argv[3] if len(sys.argv) > 3 else comet_csv.replace('.csv', '_with_peak_windows.csv')
    
    success = merge_peak_windows_into_csv(comet_csv, peak_windows_csv, output_csv)
    
    if not success:
        sys.exit(1)


if __name__ == '__main__':
    main()
