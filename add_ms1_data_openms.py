#!/usr/bin/env python3
"""
Add MS1 retention times and intensities to Comet CSV output using OpenMS.

This script:
1. Reads the Comet CSV output
2. Uses OpenMS to extract MS1 retention times and intensities for each identified peptide
3. Appends MS1 data columns to the CSV

By default all rows are preserved (only MS1 columns are added). Use --filter to drop rows
that fail ions_matched > 1 or lack single AA overhangs.

Usage:
    python add_ms1_data_openms.py <comet_csv> <raw_file> [output_csv]
    python add_ms1_data_openms.py --filter <comet_csv> <raw_file> [output_csv]   # also filter to ions_matched>1 and single AA overhangs

Requirements:
    - pyopenms (pip install pyopenms)
    - pandas (pip install pandas)
"""

import sys
import os
import pandas as pd
from pyopenms import MSExperiment, MzMLFile
import numpy as np
from collections import defaultdict

def extract_ms1_data(raw_file, comet_csv):
    """
    Extract MS1 retention times and intensities from raw file using OpenMS.
    
    Returns lists of MS1 RTs, intensities, and RT in minutes for each row in CSV.
    """
    import re
    
    print(f"[DEBUG] Loading raw file: {raw_file}")
    exp = MSExperiment()
    
    # Try different file formats
    if raw_file.endswith('.mzML') or raw_file.endswith('.mzml'):
        print(f"[DEBUG] Detected mzML format, loading...")
        MzMLFile().load(raw_file, exp)
    else:
        # Try to convert or use appropriate loader
        print(f"[DEBUG] Warning: Unsupported file format. Trying mzML loader...")
        MzMLFile().load(raw_file, exp)
    
    total_spectra = exp.size()
    print(f"[DEBUG] Loaded {total_spectra} total spectra")
    
    # Build map of MS1 scans: scan_number -> (rt, mz_intensity_map)
    ms1_data = {}
    ms1_rt_list = []  # List of (scan_index, rt) for MS1 scans
    ms2_count = 0
    
    print(f"[DEBUG] Processing spectra to identify MS1 scans...")
    for idx, spec in enumerate(exp):
        ms_level = spec.getMSLevel()
        if ms_level == 1:
            # Get scan number from native ID or use index
            scan_num = None
            native_id = spec.getNativeID()
            
            # Try to extract scan number from native ID
            if native_id:
                # Common formats: "scan=1234", "index=1234", "controllerType=0 controllerNumber=1 scan=1234"
                match = re.search(r'scan[=\s]+(\d+)', native_id, re.IGNORECASE)
                if match:
                    scan_num = int(match.group(1))
                else:
                    match = re.search(r'index[=\s]+(\d+)', native_id, re.IGNORECASE)
                    if match:
                        scan_num = int(match.group(1))
                    else:
                        # Try to find any number
                        match = re.search(r'\d+', native_id)
                        if match:
                            scan_num = int(match.group())
            
            # If no scan number found, use index + 1 (scans usually start at 1)
            if scan_num is None:
                scan_num = idx + 1
                if idx < 5:  # Debug first few
                    print(f"[DEBUG]   Spectrum {idx}: No scan number in native_id '{native_id}', using index+1={scan_num}")
            
            rt = spec.getRT()  # in seconds
            mz_intensity_map = {}
            peak_count = 0
            
            for peak in spec:
                mz = peak.getMZ()
                intensity = peak.getIntensity()
                mz_intensity_map[mz] = intensity
                peak_count += 1
            
            ms1_data[scan_num] = (rt, mz_intensity_map)
            ms1_rt_list.append((scan_num, rt))
            
            if len(ms1_data) <= 5:  # Debug first few MS1 scans
                print(f"[DEBUG]   MS1 scan {scan_num}: RT={rt:.2f}s, {peak_count} peaks, native_id='{native_id}'")
        elif ms_level == 2:
            ms2_count += 1
    
    # Sort MS1 scans by scan number
    ms1_rt_list.sort(key=lambda x: x[0])
    
    print(f"[DEBUG] Found {len(ms1_data)} MS1 scans and {ms2_count} MS2 scans")
    if len(ms1_rt_list) > 0:
        print(f"[DEBUG] MS1 scan range: {ms1_rt_list[0][0]} to {ms1_rt_list[-1][0]}")
        print(f"[DEBUG] MS1 RT range: {ms1_rt_list[0][1]:.2f}s to {ms1_rt_list[-1][1]:.2f}s ({ms1_rt_list[-1][1]/60:.2f} min)")
    
    # Now read Comet CSV and match MS2 scans to MS1 data
    print(f"[DEBUG] Reading Comet CSV: {comet_csv}")
    # Comet CSV may have metadata first line (CometVersion,...); skip it so we use the real header
    with open(comet_csv, 'r') as f:
        first_line = f.readline()
    skip = 1 if first_line.strip().startswith('CometVersion') or first_line.count(',') < 10 else 0
    df = pd.read_csv(comet_csv, sep=',', skiprows=skip, engine='python', quotechar='"', on_bad_lines='warn')
    
    print(f"[DEBUG] CSV has {len(df)} rows and columns: {list(df.columns[:10])}...")
    
    # Find MS1 retention time and intensity for each row
    ms1_rts = []
    ms1_intensities = []
    ms1_rt_mins = []
    
    rows_with_ms1 = 0
    rows_without_ms1 = 0
    rows_with_intensity = 0
    rows_without_intensity = 0
    
    print(f"[DEBUG] Matching MS2 scans to MS1 data...")
    for idx, row in df.iterrows():
        if idx % 500 == 0 and idx > 0:
            print(f"[DEBUG]   Processed {idx}/{len(df)} rows...")
        
        # Get scan number from CSV
        scan_num = None
        scan_col_used = None
        for col in ['scan', 'Scan', 'scan_number', 'ScanNumber']:
            if col in row and not pd.isna(row[col]):
                try:
                    scan_num = int(row[col])
                    scan_col_used = col
                    break
                except:
                    pass
        
        if scan_num is None:
            if idx < 3:  # Debug first few
                print(f"[DEBUG]   Row {idx}: No scan number found in columns {['scan', 'Scan', 'scan_number', 'ScanNumber']}")
            ms1_rts.append(0.0)
            ms1_intensities.append(0.0)
            ms1_rt_mins.append(0.0)
            rows_without_ms1 += 1
            continue
        
        # Find the most recent MS1 scan before this MS2 scan
        best_ms1_scan = None
        best_rt = 0.0
        best_intensity = 0.0
        
        # Search backwards through sorted MS1 scans
        for ms1_scan_num, ms1_rt in reversed(ms1_rt_list):
            if ms1_scan_num < scan_num:
                best_ms1_scan = ms1_scan_num
                best_rt = ms1_rt
                break
        
        # If found MS1 scan, get intensity at precursor m/z
        if best_ms1_scan is not None:
            rows_with_ms1 += 1
            rt, mz_intensity_map = ms1_data[best_ms1_scan]
            
            # Get precursor m/z from CSV
            precursor_mz = None
            mz_col_used = None
            for col in ['spectrum precursor m/z', 'precursor_mz', 'mz', 'precursorMZ', 'precursor_m/z']:
                if col in row and not pd.isna(row[col]):
                    try:
                        precursor_mz = float(row[col])
                        mz_col_used = col
                        break
                    except:
                        pass
            
            if idx < 3:  # Debug first few
                print(f"[DEBUG]   Row {idx}: scan={scan_num} (from '{scan_col_used}'), precursor_mz={precursor_mz} (from '{mz_col_used}'), matched MS1 scan={best_ms1_scan}, MS1 RT={best_rt:.2f}s")
            
            # Find intensity at precursor m/z (within tolerance)
            if precursor_mz and precursor_mz > 0:
                tolerance = 0.1  # Da tolerance
                matching_peaks = []
                for mz, intensity in mz_intensity_map.items():
                    if abs(mz - precursor_mz) < tolerance:
                        matching_peaks.append((mz, intensity))
                        if intensity > best_intensity:
                            best_intensity = intensity
                
                if best_intensity > 0:
                    rows_with_intensity += 1
                    if idx < 3:  # Debug first few
                        print(f"[DEBUG]     Found {len(matching_peaks)} peaks within {tolerance} Da, max intensity={best_intensity:.2e}")
                else:
                    rows_without_intensity += 1
                    if idx < 3:  # Debug first few
                        print(f"[DEBUG]     No peaks found within {tolerance} Da of {precursor_mz}")
            else:
                rows_without_intensity += 1
                if idx < 3:  # Debug first few
                    print(f"[DEBUG]     No valid precursor m/z found")
        else:
            rows_without_ms1 += 1
            if idx < 3:  # Debug first few
                print(f"[DEBUG]   Row {idx}: scan={scan_num}, no MS1 scan found before this scan")
        
        ms1_rts.append(best_rt)
        ms1_intensities.append(best_intensity)
        ms1_rt_mins.append(best_rt / 60.0 if best_rt > 0 else 0.0)
    
    print(f"[DEBUG] Matching complete:")
    print(f"[DEBUG]   Rows with MS1 RT: {rows_with_ms1}/{len(df)} ({100*rows_with_ms1/len(df):.1f}%)")
    print(f"[DEBUG]   Rows without MS1 RT: {rows_without_ms1}/{len(df)}")
    print(f"[DEBUG]   Rows with intensity: {rows_with_intensity}/{len(df)} ({100*rows_with_intensity/len(df):.1f}%)")
    print(f"[DEBUG]   Rows without intensity: {rows_without_intensity}/{len(df)}")
    
    return ms1_rts, ms1_intensities, ms1_rt_mins


def main():
    argv = [a for a in sys.argv[1:] if a != '--filter']
    apply_filter = '--filter' in sys.argv
    if not apply_filter:
        print("[DEBUG] Default: preserving all rows (only adding MS1 columns). Use --filter to apply row filters.")
    if len(argv) < 2:
        print("Usage: python add_ms1_data_openms.py [--filter] <comet_csv> <raw_file> [output_csv]")
        print("  Default: add MS1 columns only; preserve all rows.")
        print("  --filter   Also filter rows (ions_matched>1 and single AA overhangs required)")
        print("\nExample:")
        print("  python add_ms1_data_openms.py results.csv data.mzML results_with_ms1.csv")
        print("  python add_ms1_data_openms.py --filter results.csv data.mzML results_with_ms1_filtered.csv")
        sys.exit(1)
    comet_csv = argv[0]
    raw_file = argv[1]
    output_csv = argv[2] if len(argv) > 2 else comet_csv.replace('.csv', '_with_ms1.csv')
    
    if not os.path.exists(comet_csv):
        print(f"Error: Comet CSV file not found: {comet_csv}")
        sys.exit(1)
    
    if not os.path.exists(raw_file):
        print(f"Error: Raw file not found: {raw_file}")
        sys.exit(1)
    
    print("[DEBUG] ========================================")
    print("[DEBUG] Starting MS1 data extraction")
    print("[DEBUG] ========================================")
    ms1_rts, ms1_intensities, ms1_rt_mins = extract_ms1_data(raw_file, comet_csv)
    
    print(f"[DEBUG] Reading Comet CSV for column updates: {comet_csv}")
    # Skip first row if it's metadata (Comet version info)
    # Use Python engine with error handling for malformed lines
    try:
        with open(comet_csv, 'r') as f:
            first_line = f.readline()
            if 'CometVersion' in first_line:
                print(f"[DEBUG] Detected Comet version header, skipping first row")
                df = pd.read_csv(comet_csv, sep=',', skiprows=1, engine='python', quotechar='"', on_bad_lines='warn')
            else:
                df = pd.read_csv(comet_csv, sep=',', engine='python', quotechar='"', on_bad_lines='warn')
    except Exception as e:
        print(f"[DEBUG] Error reading CSV: {e}, trying without skiprows")
        df = pd.read_csv(comet_csv, sep=',', engine='python', quotechar='"', on_bad_lines='warn')
    
    input_row_count = len(df)
    print(f"[DEBUG] CSV loaded: {input_row_count} rows, {len(df.columns)} columns")
    
    # Add or update MS1 columns
    print("[DEBUG] Adding/updating MS1 columns in CSV...")
    
    # Update existing columns or add new ones
    if 'MS1_retention_time_sec' in df.columns:
        df['MS1_retention_time_sec'] = ms1_rts
    else:
        df['MS1_retention_time_sec'] = ms1_rts
    
    if 'MS1_retention_time_min' in df.columns:
        df['MS1_retention_time_min'] = ms1_rt_mins
    else:
        df['MS1_retention_time_min'] = ms1_rt_mins
    
    # Add intensity column if it doesn't exist
    if 'MS1_retention_time_intensity' not in df.columns:
        df['MS1_retention_time_intensity'] = ms1_intensities
    
    # Reorder columns to put MS1 columns after MS2 RT columns if they exist
    cols = list(df.columns)
    
    # Remove MS1 columns temporarily
    ms1_cols = ['MS1_retention_time_sec', 'MS1_retention_time_min', 'MS1_retention_time_intensity']
    for col in ms1_cols:
        if col in cols:
            cols.remove(col)
    
    # Find position after MS2 RT columns
    insert_pos = len(cols)
    for i, col in enumerate(cols):
        if ('MS2_retention_time' in col or 'retention_time_sec' in col) and 'MS1' not in col:
            insert_pos = i + 1
            break
    
    # Insert MS1 columns
    cols.insert(insert_pos, 'MS1_retention_time_sec')
    cols.insert(insert_pos + 1, 'MS1_retention_time_min')
    if 'MS1_retention_time_intensity' in df.columns:
        cols.insert(insert_pos + 2, 'MS1_retention_time_intensity')
    
    df = df[cols]
    
    print(f"[DEBUG] Column order: {list(df.columns)}")
    
    # Check empty entries in each column
    print("[DEBUG] ========================================")
    print("[DEBUG] Checking empty entries in each column...")
    print("[DEBUG] ========================================")
    total_rows = len(df)
    empty_summary = []
    
    for col in df.columns:
        # Count empty/NaN/null values
        empty_count = df[col].isna().sum()
        
        # Also count empty strings if column is string type
        if df[col].dtype == 'object':
            empty_strings = (df[col].astype(str).str.strip() == '').sum()
            empty_count = max(empty_count, empty_strings)
        
        non_empty_count = total_rows - empty_count
        empty_pct = (empty_count / total_rows * 100) if total_rows > 0 else 0
        
        empty_summary.append({
            'column': col,
            'empty': empty_count,
            'non_empty': non_empty_count,
            'empty_pct': empty_pct
        })
    
    # Sort by empty count (descending) to show columns with most empty values first
    empty_summary.sort(key=lambda x: x['empty'], reverse=True)
    
    # Print summary
    print(f"[DEBUG] Total rows: {total_rows}")
    print(f"[DEBUG] {'Column':<40} {'Empty':<10} {'Non-Empty':<12} {'Empty %':<10}")
    print(f"[DEBUG] {'-'*40} {'-'*10} {'-'*12} {'-'*10}")
    
    for item in empty_summary:
        col_name = item['column'][:39]  # Truncate long column names
        print(f"[DEBUG] {col_name:<40} {item['empty']:<10} {item['non_empty']:<12} {item['empty_pct']:>6.1f}%")
    
    # Highlight columns with all empty or mostly empty
    print(f"[DEBUG] {'-'*40} {'-'*10} {'-'*12} {'-'*10}")
    all_empty = [item for item in empty_summary if item['empty'] == total_rows]
    mostly_empty = [item for item in empty_summary if item['empty_pct'] >= 50 and item['empty'] < total_rows]
    
    if all_empty:
        print(f"[DEBUG] WARNING: {len(all_empty)} column(s) are completely empty:")
        for item in all_empty:
            print(f"[DEBUG]   - {item['column']}")
    
    if mostly_empty:
        print(f"[DEBUG] WARNING: {len(mostly_empty)} column(s) are >=50% empty:")
        for item in mostly_empty:
            print(f"[DEBUG]   - {item['column']}: {item['empty_pct']:.1f}% empty")
    
    print("[DEBUG] ========================================")
    
    initial_row_count = len(df)
    
    if not apply_filter:
        print("[DEBUG] Skipping row filters (keeping all rows)")
        df_filtered = df.copy()
    else:
        # Filter rows: exclude rows with <=1 matched fragment ion
        print("[DEBUG] Filtering rows: excluding rows with <=1 matched fragment ion...")
        # Check for ions_matched column (preferred) or matched fragment ions column
        if 'ions_matched' in df.columns:
            # Use ions_matched column (integer count)
            df_filtered = df[df['ions_matched'] > 1].copy()
            excluded_count = initial_row_count - len(df_filtered)
            print(f"[DEBUG] Using 'ions_matched' column for filtering")
            print(f"[DEBUG]   Initial rows: {initial_row_count}")
            print(f"[DEBUG]   Rows with >1 matched ions: {len(df_filtered)}")
            print(f"[DEBUG]   Rows excluded: {excluded_count} ({excluded_count/initial_row_count*100:.1f}%)")
            if 'plain_peptide' in df.columns:
                unique_before = df['plain_peptide'].nunique()
                unique_after = df_filtered['plain_peptide'].nunique()
                print(f"[DEBUG]   Unique peptides: {unique_before} -> {unique_after}")
        elif 'matched fragment ions' in df.columns:
            # Parse matched fragment ions column (comma-separated list)
            def count_fragment_ions(frag_str):
                """Count number of fragment ions in comma-separated string"""
                if pd.isna(frag_str) or frag_str == '':
                    return 0
                try:
                    frag_str = str(frag_str).strip()
                    if frag_str.startswith('"') and frag_str.endswith('"'):
                        frag_str = frag_str[1:-1]  # Remove quotes
                    if frag_str == '':
                        return 0
                    # Count comma-separated items
                    return len([x for x in frag_str.split(',') if x.strip() != ''])
                except:
                    return 0

            df['_fragment_ion_count'] = df['matched fragment ions'].apply(count_fragment_ions)
            df_filtered = df[df['_fragment_ion_count'] > 1].copy()
            excluded_count = initial_row_count - len(df_filtered)
            df_filtered = df_filtered.drop(columns=['_fragment_ion_count'])
            print(f"[DEBUG] Using 'matched fragment ions' column for filtering")
            print(f"[DEBUG]   Initial rows: {initial_row_count}")
            print(f"[DEBUG]   Rows with >1 matched ions: {len(df_filtered)}")
            print(f"[DEBUG]   Rows excluded: {excluded_count} ({excluded_count/initial_row_count*100:.1f}%)")
            if 'plain_peptide' in df.columns:
                unique_before = df['plain_peptide'].nunique()
                unique_after = df_filtered['plain_peptide'].nunique()
                print(f"[DEBUG]   Unique peptides: {unique_before} -> {unique_after}")
        else:
            print(f"[DEBUG] WARNING: Neither 'ions_matched' nor 'matched fragment ions' column found!")
            print(f"[DEBUG]   Available columns: {list(df.columns)}")
            print(f"[DEBUG]   Skipping filtering - keeping all {initial_row_count} rows")
            df_filtered = df.copy()
            excluded_count = 0

    df = df_filtered

    # Filter rows: exclude rows without single AA overhangs (only when --filter)
    if apply_filter:
        print("[DEBUG] Filtering rows: excluding rows without single AA overhangs...")
        before_overhang_filter = len(df)

        # Check for single_aa_overhang columns
        has_overhangs = False
        if 'single_aa_overhang_fragment_pairs' in df.columns:
            # Filter: keep rows where single_aa_overhang_fragment_pairs is not empty
            df_filtered = df[
                df['single_aa_overhang_fragment_pairs'].notna() &
                (df['single_aa_overhang_fragment_pairs'].astype(str).str.strip() != '') &
                (df['single_aa_overhang_fragment_pairs'].astype(str).str.strip() != '""')
            ].copy()
            has_overhangs = True
            print(f"[DEBUG] Using 'single_aa_overhang_fragment_pairs' column for filtering")
        elif 'single_aa_overhangs_protein_positions' in df.columns:
            # Filter: keep rows where single_aa_overhangs_protein_positions is not empty
            df_filtered = df[
                df['single_aa_overhangs_protein_positions'].notna() &
                (df['single_aa_overhangs_protein_positions'].astype(str).str.strip() != '') &
                (df['single_aa_overhangs_protein_positions'].astype(str).str.strip() != '""')
            ].copy()
            has_overhangs = True
            print(f"[DEBUG] Using 'single_aa_overhangs_protein_positions' column for filtering")
        else:
            print(f"[DEBUG] WARNING: Neither 'single_aa_overhang_fragment_pairs' nor 'single_aa_overhangs_protein_positions' column found!")
            print(f"[DEBUG]   Skipping single AA overhang filtering - keeping all {before_overhang_filter} rows")
            df_filtered = df.copy()

        if has_overhangs:
            excluded_overhang_count = before_overhang_filter - len(df_filtered)
            print(f"[DEBUG]   Rows before filtering: {before_overhang_filter}")
            print(f"[DEBUG]   Rows with single AA overhangs: {len(df_filtered)}")
            print(f"[DEBUG]   Rows excluded (no overhangs): {excluded_overhang_count} ({excluded_overhang_count/before_overhang_filter*100:.1f}%)")
            if 'plain_peptide' in df.columns:
                unique_before = df['plain_peptide'].nunique()
                unique_after = df_filtered['plain_peptide'].nunique()
                print(f"[DEBUG]   Unique peptides: {unique_before} -> {unique_after}")

        df = df_filtered
    
    # Sort by: sequence start position -> peptide length -> charge -> MS1 retention time
    print("[DEBUG] Sorting rows by: sequence start position, peptide length, charge, MS1 retention time...")
    
    def parse_sequence_start(seq_pos_str):
        """Extract start position from sequence_positions (e.g., '2-20' -> 2)"""
        if pd.isna(seq_pos_str) or seq_pos_str == '':
            return 999999  # Put empty ones at end
        try:
            if '-' in str(seq_pos_str):
                return int(str(seq_pos_str).split('-')[0])
            return 999999
        except:
            return 999999
    
    def parse_sequence_length(seq_pos_str):
        """Extract peptide length from sequence_positions (e.g., '2-20' -> 19)"""
        if pd.isna(seq_pos_str) or seq_pos_str == '':
            return 0
        try:
            if '-' in str(seq_pos_str):
                parts = str(seq_pos_str).split('-')
                start = int(parts[0])
                end = int(parts[1])
                return end - start + 1  # Length = end - start + 1
            return 0
        except:
            return 0
    
    # Add helper columns for sorting
    if 'sequence_positions' in df.columns:
        df['_sort_start'] = df['sequence_positions'].apply(parse_sequence_start)
        df['_sort_length'] = df['sequence_positions'].apply(parse_sequence_length)
    else:
        print("[DEBUG] Warning: 'sequence_positions' column not found, using default sorting")
        df['_sort_start'] = 999999
        df['_sort_length'] = 0
    
    # Ensure charge and MS1_retention_time_sec exist for sorting
    if 'charge' not in df.columns:
        print("[DEBUG] Warning: 'charge' column not found, using default value")
        df['charge'] = 0
    
    if 'MS1_retention_time_sec' not in df.columns:
        print("[DEBUG] Warning: 'MS1_retention_time_sec' column not found, using default value")
        df['MS1_retention_time_sec'] = 999999.0
    
    # Fill NaN values for sorting
    df['charge'] = df['charge'].fillna(0)
    df['MS1_retention_time_sec'] = df['MS1_retention_time_sec'].fillna(999999.0)
    
    # Sort by: start position -> length -> charge -> MS1 RT
    df = df.sort_values(
        by=['_sort_start', '_sort_length', 'charge', 'MS1_retention_time_sec'],
        ascending=[True, True, True, True]
    )
    
    # Remove helper columns
    df = df.drop(columns=['_sort_start', '_sort_length'])
    
    print(f"[DEBUG] Sorting complete. First few start positions: {df['sequence_positions'].head(5).tolist() if 'sequence_positions' in df.columns else 'N/A'}")
    
    output_row_count = len(df)
    print(f"[DEBUG] Writing output to: {output_csv}")
    df.to_csv(output_csv, index=False, sep=',')

    # Row preservation check
    if output_row_count == input_row_count:
        print(f"[DEBUG] Rows preserved: {input_row_count} -> {output_row_count} (all rows kept)")
    else:
        print(f"[DEBUG] Row count: input {input_row_count}, output {output_row_count} ({input_row_count - output_row_count} rows excluded by filters)")
    if not apply_filter and output_row_count != input_row_count:
        print("[DEBUG] WARNING: Filtering was not requested but row count changed; this should not happen.")
    
    print("[DEBUG] ========================================")
    
    # Count unique peptides
    unique_peptides = 0
    if 'plain_peptide' in df.columns:
        unique_peptides = df['plain_peptide'].nunique()
        total_rows = len(df)
        avg_entries_per_peptide = total_rows / unique_peptides if unique_peptides > 0 else 0
        print(f"[DEBUG] Unique peptides: {unique_peptides}")
        print(f"[DEBUG] Total rows: {total_rows}")
        print(f"[DEBUG] Average entries per peptide: {avg_entries_per_peptide:.2f}")
    else:
        print(f"[DEBUG] Warning: 'plain_peptide' column not found, cannot count unique peptides")
    
    print(f"[DEBUG] Done! Added MS1 data and sorted {len(df)} rows.")
    print(f"[DEBUG] Output written to: {output_csv}")
    print("[DEBUG] ========================================")
    
    # Print summary statistics
    if 'MS1_retention_time_sec' in df.columns:
        non_zero_rt = (df['MS1_retention_time_sec'] > 0).sum()
        print(f"[DEBUG] Summary: {non_zero_rt}/{len(df)} rows have MS1 RT > 0")
    if 'MS1_retention_time_intensity' in df.columns:
        non_zero_int = (df['MS1_retention_time_intensity'] > 0).sum()
        print(f"[DEBUG] Summary: {non_zero_int}/{len(df)} rows have MS1 intensity > 0")


if __name__ == '__main__':
    main()
