#!/usr/bin/env python3
"""
Multi-Amino Acid Overhang Block-Constraint Map Visualization

Creates a genome-browser style plot showing multi-AA overhangs as horizontal intervals.
Overhangs are grouped by length bins and colored by ion series (c vs z).
"""

import argparse
import re
import csv
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
import numpy as np


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


def parse_multi_aa_overhangs(csv_file, protein_id=None, fasta_file=None, peptides_with_single_aa=None, filtered_peptide_indices=None):
    """
    Parse multi-AA overhangs from CSV file.
    Returns a list of overhang dictionaries with:
    - start, end: peptide positions (1-indexed)
    - residues: residue string
    - fragment_pairs: list of fragment pairs
    - ion_series: 'c' or 'z' (only same-series pairs)
    - protein_start: start position in full protein
    - peptide_seq: peptide sequence
    
    Args:
        peptides_with_single_aa: Set of row indices (0-based from data rows) that have single-AA overhangs.
                                 If provided, only overhangs from these peptides will be included.
    """
    overhangs = []
    debug_count = 0
    debug_skipped_no_offset = 0
    debug_skipped_no_multi = 0
    debug_found = 0
    
    with open(csv_file, 'r') as f:
        lines = f.readlines()
        
        if len(lines) < 1:
            return overhangs
        
        # Detect header line: Comet CSV may have CometVersion on line 0 (header on line 1), or header on line 0
        header_line_idx = 1
        if 'protein' in lines[0]:
            header_line_idx = 0
        elif len(lines) > 1 and 'protein' in lines[1]:
            header_line_idx = 1
        
        header_line = lines[header_line_idx] if header_line_idx < len(lines) else ""
        header = [col.strip() for col in header_line.split(',')]
        
        # Find column indices (allow alternatives for some columns)
        try:
            protein_idx = header.index('protein')
            peptide_idx = header.index('plain_peptide') if 'plain_peptide' in header else -1
        except ValueError as e:
            print(f"Error: Required column not found: {e}")
            return overhangs
        
        single_aa_pairs_idx = header.index('single_aa_overhang_fragment_pairs') if 'single_aa_overhang_fragment_pairs' in header else -1
        multi_aa_overhangs_idx = header.index('multi_aa_overhangs') if 'multi_aa_overhangs' in header else -1
        multi_aa_pairs_idx = header.index('multi_aa_overhang_fragment_pairs') if 'multi_aa_overhang_fragment_pairs' in header else -1
        matched_ions_idx = header.index('matched fragment ions') if 'matched fragment ions' in header else -1
        site_specific_idx = -1
        for col in ['site_specific_residues', 'single_aa_overhangs_protein_positions']:
            if col in header:
                site_specific_idx = header.index(col)
                break
        if site_specific_idx < 0:
            print(f"Error: Required column not found: site_specific_residues (or single_aa_overhangs_protein_positions)")
            return overhangs
        
        # Parse data rows (starting after header)
        data_start = header_line_idx + 1
        for row_idx, line in enumerate(lines[data_start:], start=0):
            if not line.strip():
                continue
            
            # If filtering is enabled, skip peptides without single-AA overhangs
            if peptides_with_single_aa is not None and row_idx not in peptides_with_single_aa:
                continue
            
            # If filtered_peptide_indices is provided, only include peptides that passed the RT grid filter
            if filtered_peptide_indices is not None and row_idx not in filtered_peptide_indices:
                continue
            
            # Parse quoted fields manually to handle complex CSV (same as coverage_visualization.py)
            quoted_fields = re.findall(r'"([^"]*)"', line)
            
            # Also parse unquoted fields
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
            
            if len(row_data) <= protein_idx:
                continue
            
            # Extract fields (same approach as coverage_visualization.py)
            protein = ""
            multi_overhangs_str = ""
            multi_pairs_str = ""
            site_specific_str = ""
            matched_ions_str = ""
            
            # Try to get from row_data first (this is most reliable)
            if len(row_data) > protein_idx:
                protein = row_data[protein_idx].strip('"')
            
            # Matched fragment ions: used to derive multi-AA overhangs when multi-AA columns are absent
            if matched_ions_idx >= 0 and len(row_data) > matched_ions_idx:
                matched_ions_str = row_data[matched_ions_idx].strip('"').strip()
            if not matched_ions_str and len(quoted_fields) >= 4:
                # Often "matched fragment ions" is a quoted field; try by position
                for qf in quoted_fields:
                    if qf and ',' in qf and re.search(r'[cz]\d+', qf):
                        matched_ions_str = qf
                        break
            
            # Optional: legacy multi-AA columns (we now prefer deriving from matched fragment ions)
            if single_aa_pairs_idx >= 0 and len(row_data) > single_aa_pairs_idx:
                candidate = row_data[single_aa_pairs_idx].strip('"').strip()
                if candidate and re.search(r'^\d+-\d+[A-Z]{2,}', candidate):
                    multi_overhangs_str = candidate
            if not multi_overhangs_str and multi_aa_overhangs_idx >= 0 and len(row_data) > multi_aa_overhangs_idx:
                multi_overhangs_str = row_data[multi_aa_overhangs_idx].strip('"').strip()
            
            if multi_aa_overhangs_idx >= 0 and len(row_data) > multi_aa_overhangs_idx:
                candidate = row_data[multi_aa_overhangs_idx].strip('"').strip()
                if candidate and re.search(r'[cz]\d+-[cz]', candidate):
                    multi_pairs_str = candidate
            if not multi_pairs_str and multi_aa_pairs_idx >= 0 and len(row_data) > multi_aa_pairs_idx:
                multi_pairs_str = row_data[multi_aa_pairs_idx].strip('"').strip()
            
            if len(row_data) > site_specific_idx:
                site_specific_str = row_data[site_specific_idx].strip('"').strip()
            
            # For complex quoted fields, use as fallback (same as coverage_visualization.py)
            # quoted_fields[-1] is site_specific_residues
            if len(quoted_fields) >= 1:
                # Always use quoted_fields[-1] for site_specific (coverage_visualization does this)
                site_specific_str = quoted_fields[-1]  # Last one is site_specific_residues
            
            if not protein:
                continue
            
            # Clean protein ID (remove trailing numbers)
            clean_protein = re.sub(r'(\d+)$', '', protein)
            # Also try matching without the "sp|" prefix
            clean_protein_alt = re.sub(r'^sp\|', '', clean_protein)
            
            if protein_id:
                # Try multiple matching strategies
                protein_id_clean = re.sub(r'(\d+)$', '', protein_id)
                protein_id_alt = re.sub(r'^sp\|', '', protein_id_clean)
                
                if (clean_protein != protein_id_clean and 
                    clean_protein != protein_id_alt and
                    clean_protein_alt != protein_id_clean and
                    clean_protein_alt != protein_id_alt and
                    protein != protein_id):
                    continue
            else:
                # No protein filter - process all proteins
                pass
            
            # Get peptide sequence
            peptide_seq = ""
            if peptide_idx >= 0 and len(row_data) > peptide_idx:
                peptide_seq = row_data[peptide_idx].strip('"').strip()
            
            # Get single_aa_overhangs to calculate protein start position
            # Note: Due to CSV column mix-up, single_aa_overhangs column (23) has fragment pairs
            # We need to get it from quoted_fields[-3] like coverage_visualization does
            single_aa_overhangs_str = ""
            single_aa_overhangs_idx = header.index('single_aa_overhangs') if 'single_aa_overhangs' in header else -1
            if single_aa_overhangs_idx >= 0 and len(row_data) > single_aa_overhangs_idx:
                single_aa_overhangs_str = row_data[single_aa_overhangs_idx].strip('"').strip()
            
            # Also try from quoted_fields (same as coverage_visualization)
            if len(quoted_fields) >= 3:
                # quoted_fields[-3] is single_aa_overhangs
                if not single_aa_overhangs_str or not re.search(r'^\d+[A-Z]', single_aa_overhangs_str.split(',')[0] if ',' in single_aa_overhangs_str else single_aa_overhangs_str):
                    # If row_data doesn't have the right format, try quoted_fields
                    candidate = quoted_fields[-3]
                    if candidate and re.search(r'^\d+[A-Z]', candidate.split(',')[0] if ',' in candidate else candidate):
                        single_aa_overhangs_str = candidate
            
            # Calculate protein start position by comparing single_aa_overhangs (peptide positions) 
            # with site_specific_residues (protein positions)
            protein_start_offset = None
            if site_specific_str and single_aa_overhangs_str:
                # Parse single_aa_overhangs: "2L,3K,4R" -> peptide positions
                peptide_positions = []
                for item in single_aa_overhangs_str.split(','):
                    item = item.strip()
                    if item and len(item) > 1:
                        match = re.match(r'^(\d+)([A-Z])$', item)
                        if match:
                            peptide_positions.append(int(match.group(1)))
                
                # Parse site_specific_residues: "45R,46K,47R" -> protein positions
                protein_positions = []
                for item in site_specific_str.split(','):
                    item = item.strip()
                    if item and len(item) > 1:
                        match = re.match(r'^(\d+)([A-Z])$', item)
                        if match:
                            protein_positions.append(int(match.group(1)))
                
                # Calculate offset: protein_pos = peptide_pos + offset - 1
                # So: offset = protein_pos - peptide_pos + 1
                if peptide_positions and protein_positions and len(peptide_positions) == len(protein_positions):
                    # Use first matching position to calculate offset
                    offset = protein_positions[0] - peptide_positions[0] + 1
                    # Verify with a few more positions
                    matches = sum(1 for p, pr in zip(peptide_positions[:min(3, len(peptide_positions))], 
                                                     protein_positions[:min(3, len(protein_positions))])
                                 if pr == p + offset - 1)
                    if matches >= min(2, len(peptide_positions)):
                        protein_start_offset = offset
            
            # Alternative: If site_specific_residues is empty, find protein position
            # by searching for the peptide sequence in the FASTA file
            if protein_start_offset is None and fasta_file and peptide_seq:
                try:
                    sequences = parse_fasta(fasta_file)
                    # Try both clean protein ID formats
                    protein_seq = None
                    if clean_protein in sequences:
                        protein_seq = sequences[clean_protein]
                    elif clean_protein_alt in sequences:
                        protein_seq = sequences[clean_protein_alt]
                    
                    if protein_seq:
                        # Find peptide in protein sequence (case-insensitive, handle modifications)
                        # Remove any modification markers from peptide (e.g., [M+16] -> M)
                        clean_peptide = re.sub(r'\[.*?\]', '', peptide_seq).upper()
                        protein_seq_upper = protein_seq.upper()
                        
                        # Find all occurrences (in case peptide appears multiple times)
                        peptide_start_in_protein = protein_seq_upper.find(clean_peptide)
                        if peptide_start_in_protein >= 0:
                            # Found it! peptide_start_in_protein is 0-indexed, convert to 1-indexed
                            protein_start_offset = peptide_start_in_protein + 1
                        else:
                            # Try without first/last residues in case of terminal modifications
                            if len(clean_peptide) > 2:
                                # Try without first residue
                                if protein_seq_upper.find(clean_peptide[1:]) >= 0:
                                    protein_start_offset = protein_seq_upper.find(clean_peptide[1:]) + 1
                                # Try without last residue
                                elif protein_seq_upper.find(clean_peptide[:-1]) >= 0:
                                    protein_start_offset = protein_seq_upper.find(clean_peptide[:-1]) + 1
                except Exception as e:
                    # If FASTA parsing fails, skip this row
                    pass
            
            # If we still can't map to protein coordinates, skip this row
            if protein_start_offset is None:
                debug_skipped_no_offset += 1
                continue
            
            debug_count += 1
            clean_peptide = re.sub(r'\[.*?\]', '', peptide_seq) if peptide_seq else ''
            peptide_length = len(clean_peptide)
            
            # Prefer deriving multi-AA overhangs from matched fragment ions (no multi-AA columns needed)
            if matched_ions_str and matched_ions_str.strip():
                derived = derive_multi_aa_overhangs_from_fragments(
                    matched_ions_str, protein_start_offset, peptide_seq, peptide_length,
                    clean_protein, fasta_file
                )
                for o in derived:
                    if protein_id:
                        protein_id_clean = re.sub(r'(\d+)$', '', protein_id)
                        protein_id_alt = re.sub(r'^sp\|', '', protein_id_clean)
                        if (clean_protein != protein_id_clean and clean_protein != protein_id_alt and
                                clean_protein_alt != protein_id_clean and clean_protein_alt != protein_id_alt):
                            continue
                    debug_found += 1
                    overhangs.append(o)
                continue
            
            # Legacy: parse from multi-AA columns if present
            if not multi_overhangs_str:
                debug_skipped_no_multi += 1
                continue
            pattern = r'\d+-\d+[A-Z]{2,}'
            if not re.search(pattern, multi_overhangs_str):
                debug_skipped_no_multi += 1
                continue
            
            overhang_list = [o.strip() for o in multi_overhangs_str.split(',') if o.strip()]
            pairs_list = [p.strip() for p in multi_pairs_str.split(',') if p.strip()]
            
            for i, overhang_str in enumerate(overhang_list):
                # Format: "start-endRESIDUES" (e.g., "5-7RIL")
                match = re.match(r'^(\d+)-(\d+)([A-Z]+)$', overhang_str)
                if not match:
                    continue
                
                peptide_start = int(match.group(1))  # Peptide position (1-indexed)
                peptide_end = int(match.group(2))    # Peptide position (1-indexed)
                residues = match.group(3)
                
                # CRITICAL VALIDATION: Check that overhang positions are within peptide bounds
                if peptide_start < 1 or peptide_end > peptide_length or peptide_start > peptide_end:
                    # Enhanced diagnostic: try to reverse-engineer what fragment numbers could have caused this
                    fragment_pairs_str = pairs_list[i] if i < len(pairs_list) else "N/A"
                    print(f"Warning: Skipping invalid multi-AA overhang '{overhang_str}' for peptide '{peptide_seq[:50]}...' "
                          f"(peptide length: {peptide_length}, overhang: {peptide_start}-{peptide_end}, "
                          f"fragment pairs: {fragment_pairs_str})")
                    # Try to diagnose: for C-terminal ions, if end > length, what fragment numbers would cause this?
                    # For C-term: end = length - frag1, so frag1 = length - end
                    # If end > length, then frag1 would be negative, which is impossible
                    # For N-term: end = frag2, so if end > length, frag2 > length
                    if fragment_pairs_str != "N/A" and len(fragment_pairs_str) > 0:
                        # Try to extract fragment numbers from pairs
                        pair_match = re.search(r'([cz])(\d+)-([cz])(\d+)', fragment_pairs_str)
                        if pair_match:
                            ion_type = pair_match.group(1)
                            frag1 = int(pair_match.group(2))
                            frag2 = int(pair_match.group(4))
                            print(f"  Diagnostic: Fragment pair {ion_type}{frag1}-{ion_type}{frag2}, "
                                  f"peptide length {peptide_length}")
                            if ion_type == 'c':
                                # N-terminal: end should be frag2
                                print(f"  Expected: end = frag2 = {frag2}, but got {peptide_end}")
                            elif ion_type == 'z':
                                # C-terminal: end = length - frag1
                                expected_end = peptide_length - frag1
                                print(f"  Expected: end = length - frag1 = {peptide_length} - {frag1} = {expected_end}, but got {peptide_end}")
                    continue
                
                # Get corresponding fragment pairs
                fragment_pairs = []
                if i < len(pairs_list):
                    # Format: "pair1|pair2|pair3"
                    pairs = [p.strip() for p in pairs_list[i].split('|') if p.strip()]
                    fragment_pairs = pairs
                
                # Determine ion series from fragment pairs
                # Only include if all pairs are from same ion series
                ion_series = None
                for pair in fragment_pairs:
                    # Format: "c4-c6" or "z8-z10" or "z1_8-z1_10"
                    if pair.startswith('c') and not pair.startswith('c-'):
                        if ion_series is None:
                            ion_series = 'c'
                        elif ion_series != 'c':
                            ion_series = None
                            break
                    elif pair.startswith('z') or 'z1_' in pair:
                        if ion_series is None:
                            ion_series = 'z'
                        elif ion_series != 'z':
                            ion_series = None
                            break
                    else:
                        # Unknown ion series, skip this overhang
                        ion_series = None
                        break
                
                # Only include if we have a consistent ion series
                if ion_series is None:
                    continue
                
                # Map peptide positions to protein positions
                # Formula: protein_position = peptide_position + protein_start_offset - 1
                protein_start_pos = peptide_start + protein_start_offset - 1
                protein_end_pos = peptide_end + protein_start_offset - 1
                debug_found += 1
                
                overhangs.append({
                    'start': protein_start_pos,  # Use protein coordinates
                    'end': protein_end_pos,      # Use protein coordinates
                    'residues': residues,
                    'fragment_pairs': fragment_pairs,
                    'ion_series': ion_series,
                    'protein_start': protein_start_pos,
                    'peptide_start': peptide_start,  # Keep peptide coordinates for reference
                    'peptide_end': peptide_end,      # Keep peptide coordinates for reference
                    'peptide_seq': peptide_seq,
                    'protein_id': clean_protein,
                    'length': protein_end_pos - protein_start_pos + 1
                })
    
    if debug_count > 0:
        print(f"Debug: Processed {debug_count} rows with multi-AA data")
        print(f"  Found {debug_found} overhangs with protein mapping")
        print(f"  Skipped {debug_skipped_no_offset} due to no protein offset")
        print(f"  Skipped {debug_skipped_no_multi} due to no/invalid multi-AA data")
    
    return overhangs


def get_length_bin(length):
    """Return the actual overhang length (no binning)."""
    return str(length)


def derive_multi_aa_overhangs_from_fragments(matched_ions_str, protein_start_offset, peptide_seq, peptide_length, clean_protein, fasta_file=None):
    """
    Derive 2-AA and 3-AA overhangs from the list of matched fragment ions.
    Pairs of fragments that differ by 2 (e.g. c4-c6) or 3 (e.g. c4-c7) define multi-AA overhangs.
    
    Args:
        matched_ions_str: Comma-separated fragment ions (e.g., "c4,c5,c6,c7,z6,z7,z8")
        protein_start_offset: Protein position of peptide residue 1 (1-indexed)
        peptide_seq: Peptide sequence (for overhang dict)
        peptide_length: Length of peptide (no mods)
        clean_protein: Protein ID for overhang dict
        fasta_file: Optional; used to look up residue letters for overhangs
    
    Returns:
        List of overhang dicts with start, end, residues, fragment_pairs, ion_series, etc.
    """
    overhangs = []
    if not matched_ions_str or not matched_ions_str.strip():
        return overhangs
    
    fragments = [f.strip() for f in matched_ions_str.split(',') if f.strip()]
    if not fragments:
        return overhangs
    
    # Group by ion series (c, z)
    fragments_by_series = defaultdict(list)
    for frag_str in fragments:
        frag_str_lower = frag_str.lower()
        if frag_str_lower.startswith('z1_'):
            m = re.match(r'z1_(\d+)', frag_str_lower)
            if m:
                fragments_by_series['z'].append((int(m.group(1)), frag_str))
        else:
            m = re.match(r'^([a-z])(\d+)$', frag_str_lower)
            if m:
                series = m.group(1)
                fragments_by_series[series].append((int(m.group(2)), frag_str))
    
    # Residue string from protein sequence if available
    protein_sequence = None
    if fasta_file:
        try:
            sequences = parse_fasta(fasta_file)
            clean_alt = re.sub(r'^sp\|', '', clean_protein)
            protein_sequence = sequences.get(clean_protein) or sequences.get(clean_alt)
        except Exception:
            pass
    
    for series, frag_list in fragments_by_series.items():
        frag_list.sort(key=lambda x: x[0])
        for i in range(len(frag_list)):
            for j in range(i + 1, len(frag_list)):
                frag1_num, frag1_str = frag_list[i]
                frag2_num, frag2_str = frag_list[j]
                diff = frag2_num - frag1_num
                if diff < 2:
                    continue
                # Allow all multi-AA overhang lengths (2, 3, 4, 5, ...) from fragment pairs
                pair_str = f"{frag1_str}-{frag2_str}"
                normalized_pair = normalize_fragment_pair(pair_str)
                
                if series == 'c':
                    # c-ions: overhang spans peptide positions (frag1_num+1) to frag2_num (1-indexed)
                    peptide_start = frag1_num + 1
                    peptide_end = frag2_num
                else:
                    # z-ions: overhang is C-terminal; zN covers last N residues
                    peptide_start = peptide_length - frag2_num + 1
                    peptide_end = peptide_length - frag1_num
                
                if peptide_start < 1 or peptide_end > peptide_length or peptide_start >= peptide_end:
                    continue
                
                protein_start_pos = protein_start_offset + peptide_start - 1
                protein_end_pos = protein_start_offset + peptide_end - 1
                residues = ""
                if protein_sequence and 1 <= protein_start_pos <= len(protein_sequence) and 1 <= protein_end_pos <= len(protein_sequence):
                    residues = protein_sequence[protein_start_pos - 1:protein_end_pos]
                
                ion_series = 'c' if series == 'c' else 'z'
                overhangs.append({
                    'start': protein_start_pos,
                    'end': protein_end_pos,
                    'residues': residues,
                    'fragment_pairs': [normalized_pair],
                    'ion_series': ion_series,
                    'protein_start': protein_start_pos,
                    'peptide_start': peptide_start,
                    'peptide_end': peptide_end,
                    'peptide_seq': peptide_seq,
                    'protein_id': clean_protein,
                    'length': protein_end_pos - protein_start_pos + 1
                })
    
    return overhangs


def parse_single_aa_overhangs(csv_file, protein_id=None, fasta_file=None, return_peptide_indices=False):
    """
    Parse single-AA overhangs from CSV file.
    Returns a list of overhang dictionaries (same format as multi-AA overhangs).
    
    Args:
        return_peptide_indices: If True, also returns a set of row indices (0-based from data rows)
                                that have single-AA overhangs.
    
    Returns:
        If return_peptide_indices is True: (overhangs, peptide_indices_set)
        Otherwise: overhangs
    """
    overhangs = []
    peptide_indices = set()  # Track which peptides have single-AA overhangs
    
    with open(csv_file, 'r') as f:
        lines = f.readlines()
        
        if len(lines) < 1:
            return overhangs if not return_peptide_indices else (overhangs, peptide_indices)
        
        # Detect header line: Comet CSV may have header on line 0 or line 1
        header_line_idx = 1
        if 'protein' in lines[0]:
            header_line_idx = 0
        elif len(lines) > 1 and 'protein' in lines[1]:
            header_line_idx = 1
        header_line = lines[header_line_idx]
        header = [col.strip() for col in header_line.split(',')]
        
        try:
            protein_idx = header.index('protein')
            single_aa_pairs_idx = header.index('single_aa_overhang_fragment_pairs')
            peptide_idx = header.index('plain_peptide') if 'plain_peptide' in header else -1
        except ValueError as e:
            return overhangs if not return_peptide_indices else (overhangs, peptide_indices)
        
        single_aa_idx = header.index('single_aa_overhangs') if 'single_aa_overhangs' in header else -1
        site_specific_idx = -1
        for col in ['site_specific_residues', 'single_aa_overhangs_protein_positions']:
            if col in header:
                site_specific_idx = header.index(col)
                break
        if site_specific_idx < 0:
            return overhangs if not return_peptide_indices else (overhangs, peptide_indices)
        
        data_start = header_line_idx + 1
        for row_idx, line in enumerate(lines[data_start:], start=0):
            if not line.strip():
                continue
            
            quoted_fields = re.findall(r'"([^"]*)"', line)
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
            
            if len(row_data) <= max(protein_idx, single_aa_idx):
                continue
            
            protein = row_data[protein_idx].strip('"').strip()
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
            
            # Get single-AA overhangs: prefer row_data (column order is reliable); fallback to quoted_fields
            single_aa_str = ""
            single_aa_pairs_str = ""
            site_specific_str = ""
            if site_specific_idx >= 0 and len(row_data) > site_specific_idx:
                site_specific_str = row_data[site_specific_idx].strip('"').strip()
            if single_aa_pairs_idx >= 0 and len(row_data) > single_aa_pairs_idx:
                single_aa_pairs_str = row_data[single_aa_pairs_idx].strip('"').strip()
            if single_aa_idx >= 0 and len(row_data) > single_aa_idx:
                single_aa_str = row_data[single_aa_idx].strip('"').strip()
            # Fallback to quoted_fields (older CSV format)
            if not site_specific_str and len(quoted_fields) >= 1:
                site_specific_str = quoted_fields[-1]
            if not single_aa_pairs_str and len(quoted_fields) >= 2:
                single_aa_pairs_str = quoted_fields[-2]
            if not single_aa_str and len(quoted_fields) >= 3:
                single_aa_str = quoted_fields[-3]
            
            peptide_seq = ""
            if peptide_idx >= 0 and len(row_data) > peptide_idx:
                peptide_seq = row_data[peptide_idx].strip('"').strip()
            
            # BEST APPROACH: Use site_specific_residues directly if available
            # This already has protein positions mapped!
            if site_specific_str:
                # Parse site_specific_residues: "45R,67T,89K" -> protein positions
                # Parse single-AA overhangs: "2L,3K,4R" -> peptide positions
                # Parse fragment pairs: "y6-y7|z1_6-z1_7,y5-y6|z1_5-z1_6,c3-c4" -> pairs per overhang
                single_aa_list = [s.strip() for s in single_aa_str.split(',') if s.strip()] if single_aa_str else []
                single_aa_pairs_list = [p.strip() for p in single_aa_pairs_str.split(',') if p.strip()] if single_aa_pairs_str else []
                
                has_single_aa = False
                for i, item in enumerate(site_specific_str.split(',')):
                    item = item.strip()
                    if item and len(item) > 1:
                        match = re.match(r'^(\d+)([A-Z])$', item)
                        if match:
                            protein_pos = int(match.group(1))
                            residue = match.group(2)
                            
                            # Get corresponding fragment pairs for this overhang
                            fragment_pairs = []
                            if i < len(single_aa_pairs_list):
                                # Format: "pair1|pair2|pair3" (multiple pairs separated by |)
                                pairs = [p.strip() for p in single_aa_pairs_list[i].split('|') if p.strip()]
                                fragment_pairs = pairs
                            
                            # Single-AA overhang: start = end = same position, length = 1
                            overhangs.append({
                                'start': protein_pos,
                                'end': protein_pos,
                                'residues': residue,
                                'fragment_pairs': fragment_pairs,
                                'ion_series': 'mixed',  # Single-AA can come from any series
                                'protein_id': clean_protein,
                                'length': 1,
                                'peptide_seq': peptide_seq  # Store peptide sequence for unique counting
                            })
                            has_single_aa = True
                
                if has_single_aa:
                    peptide_indices.add(row_idx)
                continue  # Skip to next row if we used site_specific_residues
            
            # FALLBACK: Calculate protein offset from single_aa_overhangs + site_specific
            # (This is the old approach, kept as fallback)
            protein_start_offset = None
            if single_aa_str:
                # Try to get offset from comparing peptide and protein positions
                # But we need site_specific for this, which we already checked above
                pass
            
            # FALLBACK: Use FASTA to find peptide position
            has_single_aa = False
            if fasta_file and peptide_seq:
                try:
                    sequences = parse_fasta(fasta_file)
                    protein_seq = None
                    if clean_protein in sequences:
                        protein_seq = sequences[clean_protein]
                    elif clean_protein_alt in sequences:
                        protein_seq = sequences[clean_protein_alt]
                    
                    if protein_seq:
                        clean_peptide = re.sub(r'\[.*?\]', '', peptide_seq).upper()
                        protein_seq_upper = protein_seq.upper()
                        peptide_start_in_protein = protein_seq_upper.find(clean_peptide)
                        if peptide_start_in_protein >= 0:
                            protein_start_offset = peptide_start_in_protein + 1
                            
                            # Parse single-AA overhangs (format: "2L,3K,4R")
                            if single_aa_str:
                                for item in single_aa_str.split(','):
                                    item = item.strip()
                                    if item and len(item) > 1:
                                        match = re.match(r'^(\d+)([A-Z])$', item)
                                        if match:
                                            peptide_pos = int(match.group(1))
                                            residue = match.group(2)
                                            protein_pos = peptide_pos + protein_start_offset - 1
                                            
                                            # Single-AA overhang: start = end = same position, length = 1
                                            overhangs.append({
                                                'start': protein_pos,
                                                'end': protein_pos,
                                                'residues': residue,
                                                'fragment_pairs': [],
                                                'ion_series': 'mixed',
                                                'protein_id': clean_protein,
                                                'length': 1,
                                                'peptide_seq': peptide_seq  # Store peptide sequence for unique counting
                                            })
                                            has_single_aa = True
                except:
                    pass
            
            if has_single_aa:
                peptide_indices.add(row_idx)
    
    if return_peptide_indices:
        return overhangs, peptide_indices
    return overhangs


def create_single_aa_histogram(single_aa_overhangs, protein_sequence, protein_id, output_file):
    """
    Create a histogram showing counts of single-AA overhangs per protein position.
    The x-axis shows the protein sequence with amino acid letters.
    Counts represent how many unique peptides contribute a single-AA overhang to each position.
    
    Args:
        single_aa_overhangs: List of single-AA overhang dictionaries
        protein_sequence: Full protein sequence
        protein_id: Protein identifier
        output_file: Output PNG file path
    """
    if not single_aa_overhangs:
        print("No single-AA overhangs found to visualize.")
        return
    
    if not protein_sequence:
        print("Error: Protein sequence required for histogram")
        return
    
    protein_length = len(protein_sequence)
    
    # Count how many unique peptides contribute a single-AA overhang to each position
    # Use a set per position to track unique peptides
    # If overhang has 'peptide_id' or similar, use it; otherwise use a combination of fields
    position_peptides = {}  # position -> set of peptide identifiers
    
    for overhang in single_aa_overhangs:
        pos = overhang['start']  # Single-AA: start == end
        if 1 <= pos <= protein_length:
            # Try to get a unique peptide identifier
            # Use peptide_seq if available, or create a hash from overhang data
            peptide_id = None
            if 'peptide_seq' in overhang and overhang['peptide_seq']:
                peptide_id = overhang['peptide_seq']
            elif 'fragment_pairs' in overhang and overhang['fragment_pairs']:
                # Use fragment pairs as a proxy for peptide identity
                peptide_id = str(sorted(overhang['fragment_pairs']))
            else:
                # Fallback: use position + residues as identifier
                peptide_id = f"{pos}_{overhang.get('residues', '')}"
            
            if pos not in position_peptides:
                position_peptides[pos] = set()
            position_peptides[pos].add(peptide_id)
    
    # Convert to counts
    position_counts = {pos: len(peptides) for pos, peptides in position_peptides.items()}
    
    # Create arrays for plotting
    positions = list(range(1, protein_length + 1))
    counts = [position_counts.get(pos, 0) for pos in positions]
    
    # Create figure - make it wider to accommodate all position labels
    fig_width = max(20, protein_length * 0.15)
    fig_height = 6
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Create bar chart
    bars = ax.bar(positions, counts, width=0.8, color='steelblue', edgecolor='black', linewidth=0.5)
    
    # Set x-axis to show all positions with amino acid letters
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
    ax.set_xticklabels(x_labels, fontsize=6, rotation=0)
    
    # Set labels and title
    ax.set_xlabel('Protein Position (1-indexed) with Amino Acid', fontsize=10, fontweight='bold')
    ax.set_ylabel('Number of Peptides with Single-AA Overhang', fontsize=10, fontweight='bold')
    title = f'Single-AA Overhang Counts\n{protein_id} (Length: {protein_length} residues)'
    ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
    
    # Set y-axis to start at 0
    ax.set_ylim(bottom=0)
    
    # Add grid for easier reading
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    
    # Count total unique peptides
    unique_peptides = set()
    for overhang in single_aa_overhangs:
        peptide_id = None
        if 'peptide_seq' in overhang and overhang['peptide_seq']:
            peptide_id = overhang['peptide_seq']
        elif 'fragment_pairs' in overhang and overhang['fragment_pairs']:
            peptide_id = str(sorted(overhang['fragment_pairs']))
        else:
            peptide_id = f"{overhang['start']}_{overhang.get('residues', '')}"
        unique_peptides.add(peptide_id)
    
    total_peptides = len(unique_peptides)
    
    # Add legend with total peptide count
    legend_text = f'Total peptides: {total_peptides}'
    ax.legend([legend_text], loc='upper right', framealpha=0.9, fontsize=10)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.1, 1, 0.98])
    
    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Single-AA overhang histogram saved to: {output_file}")
    plt.close()


def normalize_fragment_pair(pair):
    """
    Normalize fragment pair to group z and z+1 ions of the same ordinal.
    For example, z5-z6 and z1_5-z1_6 should be treated as the same pair.
    
    Args:
        pair: Fragment pair string like "z5-z6" or "z1_5-z1_6"
    
    Returns:
        Normalized pair string where z+1 ions are converted to z format
    """
    # Replace z1_N with zN format
    normalized = re.sub(r'z1_(\d+)', r'z\1', pair)
    return normalized


def create_single_aa_fragment_pair_histogram(single_aa_overhangs, protein_sequence, protein_id, output_file):
    """
    Create a histogram showing counts of fragment pairs resulting in single-AA overhangs per protein position.
    The x-axis shows the protein sequence with amino acid letters.
    Counts represent how many fragment pairs contribute to a single-AA overhang at each position.
    z and z+1 ions of the same ordinal are grouped together.
    
    Args:
        single_aa_overhangs: List of single-AA overhang dictionaries with fragment_pairs
        protein_sequence: Full protein sequence
        protein_id: Protein identifier
        output_file: Output PNG file path
    """
    if not single_aa_overhangs:
        print("No single-AA overhangs found to visualize.")
        return
    
    if not protein_sequence:
        print("Error: Protein sequence required for histogram")
        return
    
    protein_length = len(protein_sequence)
    
    # Count fragment pairs per position, grouping z and z+1 ions of the same ordinal
    position_fragment_pairs = {}  # position -> set of normalized fragment pair identifiers
    
    for overhang in single_aa_overhangs:
        pos = overhang['start']  # Single-AA: start == end
        if 1 <= pos <= protein_length:
            fragment_pairs = overhang.get('fragment_pairs', [])
            
            if pos not in position_fragment_pairs:
                position_fragment_pairs[pos] = set()
            
            # Add each fragment pair, normalized to group z and z+1
            for pair in fragment_pairs:
                normalized_pair = normalize_fragment_pair(pair)
                position_fragment_pairs[pos].add(normalized_pair)
    
    # Convert to counts
    position_counts = {pos: len(pairs) for pos, pairs in position_fragment_pairs.items()}
    
    # Create arrays for plotting
    positions = list(range(1, protein_length + 1))
    counts = [position_counts.get(pos, 0) for pos in positions]
    
    # Create figure - make it wider to accommodate all position labels
    fig_width = max(20, protein_length * 0.15)
    fig_height = 6
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Create bar chart
    bars = ax.bar(positions, counts, width=0.8, color='steelblue', edgecolor='black', linewidth=0.5)
    
    # Set x-axis to show all positions with amino acid letters
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
    ax.set_xticklabels(x_labels, fontsize=6, rotation=0)
    
    # Set labels and title
    ax.set_xlabel('Protein Position (1-indexed) with Amino Acid', fontsize=10, fontweight='bold')
    ax.set_ylabel('Number of Fragment Pairs (Single-AA Overhangs)', fontsize=10, fontweight='bold')
    title = f'Single-AA Overhang Fragment Pair Counts\n{protein_id} (Length: {protein_length} residues)\n(z and z+1 ions of same ordinal grouped)'
    ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
    
    # Set y-axis to start at 0
    ax.set_ylim(bottom=0)
    
    # Add grid for easier reading
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    
    # Count total unique peptides
    unique_peptides = set()
    for overhang in single_aa_overhangs:
        peptide_id = None
        if 'peptide_seq' in overhang and overhang['peptide_seq']:
            peptide_id = overhang['peptide_seq']
        elif 'fragment_pairs' in overhang and overhang['fragment_pairs']:
            peptide_id = str(sorted(overhang['fragment_pairs']))
        else:
            peptide_id = f"{overhang['start']}_{overhang.get('residues', '')}"
        unique_peptides.add(peptide_id)
    
    total_peptides = len(unique_peptides)
    
    # Add legend with total peptide count
    legend_text = f'Total peptides: {total_peptides}'
    ax.legend([legend_text], loc='upper right', framealpha=0.9, fontsize=10)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.1, 1, 0.98])
    
    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Single-AA overhang fragment pair histogram saved to: {output_file}")
    plt.close()


def create_peptide_rt_grid(csv_file, protein_id, fasta_file, output_file):
    """
    Create a grid plot showing peptides ordered by retention time on y-axis,
    protein positions on x-axis, with single-AA overhangs marked.
    
    Args:
        csv_file: Comet CSV output file
        protein_id: Protein identifier to filter
        fasta_file: FASTA file with protein sequences
        output_file: Output PNG file path
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
    
    # Parse peptides with retention times and single-AA overhangs
    peptides = []
    
    with open(csv_file, 'r') as f:
        lines = f.readlines()
        
        if len(lines) < 2:
            print("Error: CSV file too short")
            return
        
        header_line = lines[1]
        header = [col.strip() for col in header_line.split(',')]
        
        try:
            protein_idx = header.index('protein')
            # Retention time: MS1 only - try MS1_retention_time_sec, then retention_time_sec (Comet/Percolator CSV)
            rt_idx = -1
            for col in ['MS1_retention_time_sec', 'retention_time_sec', 'MS1_retention_time', 'retention_time']:
                if col in header:
                    rt_idx = header.index(col)
                    break
            if rt_idx < 0:
                raise ValueError("No retention time column found (tried: MS1_retention_time_sec, retention_time_sec, ...). Plots use MS1 retention time only.")
            peptide_idx = header.index('plain_peptide')
            single_aa_idx = header.index('single_aa_overhangs')
            single_aa_pairs_idx = header.index('single_aa_overhang_fragment_pairs')
            site_specific_idx = header.index('site_specific_residues')
            sp_score_idx = header.index('sp_score')
            modifications_idx = header.index('modifications')
        except ValueError as e:
            print(f"Error: Required column not found: {e}")
            return
        
        # Use csv.reader for proper CSV parsing (handles quoted fields correctly)
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
            
            # Also get quoted fields for single_aa_overhangs, etc. by re-parsing the original line
            # We need to get the original line to extract quoted fields properly
            original_line = lines[2 + row_idx] if (2 + row_idx) < len(lines) else ""
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
            
            # Get retention time (use csv.reader result which is already parsed)
            try:
                rt_str = row_data[rt_idx] if rt_idx < len(row_data) else ""
                rt = float(rt_str)
            except (ValueError, IndexError):
                continue
            
            # Get quality score (sp_score) - use csv.reader result
            try:
                sp_score_str = row_data[sp_score_idx] if sp_score_idx < len(row_data) else ""
                sp_score = float(sp_score_str)
            except (ValueError, IndexError):
                sp_score = 0.0
            
            # Get peptide sequence
            peptide_seq = ""
            if peptide_idx >= 0 and len(row_data) > peptide_idx:
                peptide_seq = row_data[peptide_idx].strip('"').strip()
            
            # Get single-AA overhangs, fragment pairs, and site_specific_residues
            single_aa_str = ""
            single_aa_pairs_str = ""
            site_specific_str = ""
            if len(quoted_fields) >= 3:
                single_aa_str = quoted_fields[-3]  # single_aa_overhangs
            if len(quoted_fields) >= 2:
                single_aa_pairs_str = quoted_fields[-2]  # single_aa_overhang_fragment_pairs
            if len(quoted_fields) >= 1:
                site_specific_str = quoted_fields[-1]  # site_specific_residues
            
            # Also try from row_data as fallback
            if single_aa_pairs_idx >= 0 and len(row_data) > single_aa_pairs_idx:
                if not single_aa_pairs_str:
                    single_aa_pairs_str = row_data[single_aa_pairs_idx].strip('"').strip()
            
            # Map peptide to protein positions using site_specific_residues
            if site_specific_str:
                # Parse site_specific_residues to get protein positions
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
                    peptide_start = min(protein_positions)
                    peptide_end = max(protein_positions)
                    
                    # Get single-AA overhang positions with fragment pair counts and actual pairs
                    # single_aa_overhangs are in peptide coordinates (e.g., "2L,3K")
                    # single_aa_overhang_fragment_pairs are like "y6-y7|z1_6-z1_7,y5-y6|z1_5-z1_6"
                    # site_specific_residues are in protein coordinates (e.g., "45L,46K")
                    # They should align by index
                    single_aa_positions = {}  # position -> count of unique fragment pairs
                    single_aa_fragment_pairs = {}  # position -> set of normalized fragment pairs
                    if single_aa_str and site_specific_list:
                        single_aa_list = [s.strip() for s in single_aa_str.split(',') if s.strip()]
                        single_aa_pairs_list = [p.strip() for p in single_aa_pairs_str.split(',') if p.strip()] if single_aa_pairs_str else []
                        
                        # Align by index: single_aa_list[i] corresponds to site_specific_list[i] and single_aa_pairs_list[i]
                        for i, (protein_pos, protein_residue) in enumerate(site_specific_list):
                            if i < len(single_aa_list):
                                # single_aa_list[i] is like "2L" (peptide position + residue)
                                # We already have the protein position from site_specific
                                # Count unique fragment pairs for this overhang
                                fragment_pair_count = 0
                                normalized_pairs = set()
                                if i < len(single_aa_pairs_list):
                                    # Format: "pair1|pair2|pair3" (pipe-separated)
                                    pairs = [p.strip() for p in single_aa_pairs_list[i].split('|') if p.strip()]
                                    # Normalize pairs to group z and z+1 ions of same ordinal
                                    for pair in pairs:
                                        normalized_pair = normalize_fragment_pair(pair)
                                        normalized_pairs.add(normalized_pair)
                                    fragment_pair_count = len(normalized_pairs)
                                
                                if fragment_pair_count > 0:
                                    single_aa_positions[protein_pos] = fragment_pair_count
                                    single_aa_fragment_pairs[protein_pos] = normalized_pairs
                    
                    peptides.append({
                        'rt': rt,
                        'peptide_seq': peptide_seq,
                        'start': peptide_start,
                        'end': peptide_end,
                        'single_aa_positions': single_aa_positions,  # Now a dict: position -> fragment pair count
                        'single_aa_fragment_pairs': single_aa_fragment_pairs,  # position -> set of normalized fragment pairs
                        'protein_positions': set(protein_positions),
                        'sp_score': sp_score
                    })
    
    if not peptides:
        print(f"No peptides found for protein {protein_id}")
        return
    
    # Filter: First exclude ALL peptides with duplicate retention times BEFORE position filtering
    num_peptides_before = len(peptides)
    print(f"Before filtering: {num_peptides_before} peptides")
    
    # Step 1: Identify retention times that appear only once (unique RTs)
    rt_to_peptides = {}  # rt -> list of (peptide_index, sp_score)
    for idx, peptide in enumerate(peptides):
        rt = peptide['rt']
        if rt not in rt_to_peptides:
            rt_to_peptides[rt] = []
        rt_to_peptides[rt].append((idx, peptide['sp_score']))
    
    # Keep only peptides with retention times that appear exactly once (unique RTs)
    unique_rt_peptides = set()
    for rt, peptide_list in rt_to_peptides.items():
        if len(peptide_list) == 1:
            # This RT appears only once, keep this peptide
            unique_rt_peptides.add(peptide_list[0][0])
    
    print(f"After excluding duplicate RTs: {len(unique_rt_peptides)} peptides with unique retention times")
    
    # Step 2: Filter peptides list to only those with unique RTs
    peptides = [peptides[i] for i in sorted(unique_rt_peptides)]
    
    # Step 3: Now build position map from unique RT peptides and ensure at least 3 per position
    position_to_peptides = {}
    for idx, peptide in enumerate(peptides):
        for pos in peptide['single_aa_positions'].keys():
            if 1 <= pos <= protein_length:
                if pos not in position_to_peptides:
                    position_to_peptides[pos] = []
                position_to_peptides[pos].append((idx, peptide['sp_score']))
    
    # Step 4: For each position, keep at least top 10 peptides by sp_score
    peptides_to_keep = set()
    for pos, peptide_list in position_to_peptides.items():
        # Sort by sp_score (descending) and take at least top 10
        peptide_list.sort(key=lambda x: x[1], reverse=True)
        top_peptides = peptide_list[:10] if len(peptide_list) >= 10 else peptide_list
        for idx, _ in top_peptides:
            peptides_to_keep.add(idx)
    
    # Filter peptides list
    peptides = [peptides[i] for i in sorted(peptides_to_keep)]
    num_peptides_after = len(peptides)
    print(f"After filtering (unique RT only, at least 10 per position): {num_peptides_after} peptides")
    
    # Sort peptides by retention time
    peptides.sort(key=lambda x: x['rt'])
    
    num_peptides = len(peptides)
    
    # Aggregate unique fragment pairs and unique peptides per position across all filtered peptides
    position_to_all_pairs = {}  # position -> set of all unique normalized fragment pairs
    position_to_peptides = {}  # position -> set of unique peptide indices
    for peptide_idx, peptide in enumerate(peptides):
        for pos, pairs_set in peptide.get('single_aa_fragment_pairs', {}).items():
            if 1 <= pos <= protein_length:
                if pos not in position_to_all_pairs:
                    position_to_all_pairs[pos] = set()
                    position_to_peptides[pos] = set()
                position_to_all_pairs[pos].update(pairs_set)
                position_to_peptides[pos].add(peptide_idx)
    
    # Count unique fragment pairs and unique peptides per position
    fragment_pair_counts = [len(position_to_all_pairs.get(pos, set())) for pos in range(1, protein_length + 1)]
    unique_peptide_counts = [len(position_to_peptides.get(pos, set())) for pos in range(1, protein_length + 1)]
    
    # Create figure with stacked subplots: histogram on top, grid below
    fig_width = max(20, protein_length * 0.15)
    fig_height = max(8, num_peptides * 0.1) + 3  # Add space for histogram
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    # Create subplots: histogram on top (20% height), grid below (80% height)
    # Use sharex=True to ensure they share the exact same x-axis
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 4], hspace=0.1)
    ax_hist = fig.add_subplot(gs[0])
    ax = fig.add_subplot(gs[1], sharex=ax_hist)
    
    # Create grid: rows = peptides (by RT), cols = protein positions
    # Mark only single-AA overhang positions, color by peptide quality score
    grid = np.zeros((num_peptides, protein_length), dtype=float)  # Will store sp_score for coloring
    
    # Get sp_score distribution for percentile-based normalization
    import numpy as np
    sp_scores = [p['sp_score'] for p in peptides]
    
    for i, peptide in enumerate(peptides):
        # Mark single-AA overhang positions with the peptide's sp_score
        for pos in peptide['single_aa_positions'].keys():
            if 1 <= pos <= protein_length:
                grid[i, pos - 1] = peptide['sp_score']
    
    # Calculate percentiles for better color distribution
    if sp_scores:
        sp_scores_array = np.array(sp_scores)
        p25 = np.percentile(sp_scores_array, 25)
        p50 = np.percentile(sp_scores_array, 50)
        p75 = np.percentile(sp_scores_array, 75)
        p90 = np.percentile(sp_scores_array, 90)
        min_sp_score = min(sp_scores)
        max_sp_score = max(sp_scores)
        
        print(f"sp_score distribution: min={min_sp_score:.4f}, p25={p25:.4f}, p50={p50:.4f}, p75={p75:.4f}, p90={p90:.4f}, max={max_sp_score:.4f}")
    else:
        min_sp_score = 0
        max_sp_score = 1
        p25 = p50 = p75 = p90 = 0.5
    
    # Create colormap for quality scores
    # Use a colormap where 0 (no coverage) is white, and scores are colored
    from matplotlib.colors import ListedColormap, LinearSegmentedColormap
    import matplotlib.cm as cm
    
    # Create custom colormap: white for 0, then orange -> pink -> purple -> blue for scores
    # Orange (lowest) -> medium pink -> purple (high) -> blue (very high)
    colors_list = ['white']  # White for no coverage
    
    # Create custom colormap with specific colors
    n_colors = 255
    for i in range(n_colors):
        t = i / (n_colors - 1)  # 0 to 1
        
        if t < 0.33:
            # Orange (lowest scores) - RGB: (255, 165, 0) / (1.0, 0.65, 0.0)
            # Keep orange throughout first third
            r = 1.0
            g = 0.65
            b = 0.0
        elif t < 0.66:
            # Orange to medium pink (33-66%)
            # Medium pink RGB: approximately (255, 105, 180) / (1.0, 0.41, 0.71)
            local_t = (t - 0.33) / 0.33
            r = 1.0
            g = 0.65 - local_t * 0.24  # 0.65 -> 0.41
            b = local_t * 0.71  # 0.0 -> 0.71
        elif t < 0.85:
            # Medium pink to purple (66-85%)
            # Purple RGB: approximately (128, 0, 128) / (0.5, 0.0, 0.5)
            local_t = (t - 0.66) / 0.19
            r = 1.0 - local_t * 0.5  # 1.0 -> 0.5
            g = max(0.0, 0.41 - local_t * 0.41)  # 0.41 -> 0.0, ensure >= 0
            b = 0.71 - local_t * 0.21  # 0.71 -> 0.5
        else:
            # Purple to blue (85-100%)
            # Blue RGB: (0, 0, 255) / (0.0, 0.0, 1.0)
            local_t = (t - 0.85) / 0.15
            r = max(0.0, 0.5 - local_t * 0.5)  # 0.5 -> 0.0, ensure >= 0
            g = 0.0  # Stay at 0
            b = min(1.0, 0.5 + local_t * 0.5)  # 0.5 -> 1.0, ensure <= 1
        
        # Ensure all values are in [0, 1] range
        r = max(0.0, min(1.0, r))
        g = max(0.0, min(1.0, g))
        b = max(0.0, min(1.0, b))
        
        colors_list.append((r, g, b))
    
    custom_cmap = ListedColormap(colors_list)
    
    # Use percentile-based normalization to spread colors more evenly
    # Map percentiles to color ranges: p25->orange/pink boundary, p50->pink/purple, p75->purple/blue, p90->blue
    grid_normalized = grid.copy()
    if max_sp_score > min_sp_score:
        mask = grid > 0
        scores_to_normalize = grid[mask]
        
        # Create a function to map scores to normalized values based on percentiles
        def normalize_score(score):
            if score <= p25:
                # Map [min, p25] to [1, 85] (orange range: first 33% of colors)
                if p25 > min_sp_score:
                    t = (score - min_sp_score) / (p25 - min_sp_score)
                else:
                    t = 0.0
                return 1 + int(t * 84)  # 1 to 85
            elif score <= p50:
                # Map [p25, p50] to [85, 168] (orange-pink range: 33-66% of colors)
                if p50 > p25:
                    t = (score - p25) / (p50 - p25)
                else:
                    t = 0.0
                return 85 + int(t * 83)  # 85 to 168
            elif score <= p75:
                # Map [p50, p75] to [168, 217] (pink-purple range: 66-85% of colors)
                if p75 > p50:
                    t = (score - p50) / (p75 - p50)
                else:
                    t = 0.0
                return 168 + int(t * 49)  # 168 to 217
            elif score <= p90:
                # Map [p75, p90] to [217, 243] (purple range: 85-95% of colors)
                if p90 > p75:
                    t = (score - p75) / (p90 - p75)
                else:
                    t = 0.0
                return 217 + int(t * 26)  # 217 to 243
            else:
                # Map [p90, max] to [243, 256] (blue range: top 5% of colors)
                if max_sp_score > p90:
                    t = (score - p90) / (max_sp_score - p90)
                else:
                    t = 0.0
                return 243 + int(t * 12)  # 243 to 256
        
        # Apply normalization
        normalized_scores = np.array([normalize_score(s) for s in scores_to_normalize])
        grid_normalized[mask] = normalized_scores
    
    # Display grid with quality score coloring using pcolormesh for black edges
    # Create mesh coordinates - pcolormesh uses edges, so we need num_peptides+1 rows
    x = np.arange(protein_length + 1)
    y = np.arange(num_peptides + 1)
    X, Y = np.meshgrid(x, y)
    
    # Use pcolormesh to get black edges around each pixel
    im = ax.pcolormesh(X, Y, grid_normalized, cmap=custom_cmap, 
                       vmin=0, vmax=256, edgecolors='black', linewidths=0.3, shading='flat')
    
    # Set y-axis limits to match the mesh
    ax.set_ylim(0, num_peptides)
    
    # Set x-axis limits (both plots share this via sharex)
    ax.set_xlim(0, protein_length)
    
    # Create histogram on top (stacked above grid plot) with two bars per position
    # Align bars exactly with grid columns: bars should span from integer to integer (0-1, 1-2, etc.)
    # to match pcolormesh cells which have edges at integer positions
    x_positions = list(range(protein_length))  # Bars start at 0, 1, 2, etc. (left edge of each grid cell)
    
    # Create grouped bars: two bars per position (fragment pairs and unique peptides)
    # Each bar will be 0.4 wide, positioned side by side within each column
    bar_width = 0.4
    x1 = [x - bar_width/2 for x in x_positions]  # Left bars (fragment pairs)
    x2 = [x + bar_width/2 for x in x_positions]  # Right bars (unique peptides)
    
    bars1 = ax_hist.bar(x1, fragment_pair_counts, width=bar_width, edgecolor='black', linewidth=0.5, 
                        color='steelblue', alpha=0.7, align='edge', label='Unique Fragment Pairs')
    bars2 = ax_hist.bar(x2, unique_peptide_counts, width=bar_width, edgecolor='black', linewidth=0.5, 
                        color='green', alpha=0.7, align='edge', label='Unique Peptides')
    ax_hist.set_ylabel('Count', fontsize=10, fontweight='bold')
    ax_hist.set_title('Number of Unique Fragment Pairs and Unique Peptides per Residue (across all filtered peptides)', 
                     fontsize=11, fontweight='bold', pad=20)
    ax_hist.legend(loc='upper right', fontsize=8)
    ax_hist.grid(axis='y', alpha=0.3, linestyle='--')
    ax_hist.set_xticks([])  # Hide x-axis ticks on histogram (will show on main plot)
    ax_hist.set_xlim(0, protein_length)  # Same x-axis limits as main plot (shared via sharex)
    # Match y-axis tick label font size to grid plot x-axis font size
    ax_hist.tick_params(axis='y', labelsize=6)
    
    # Set x-axis to show all positions with amino acid letters (on main plot only, since they share x-axis)
    # Position ticks at the center of each column (0.5, 1.5, 2.5, ...) not at the edges
    x_ticks = [i + 0.5 for i in range(protein_length)]
    x_labels = []
    for i in range(protein_length):
        pos_1idx = i + 1
        if pos_1idx <= len(protein_sequence):
            aa = protein_sequence[i]
            x_labels.append(f"{pos_1idx}\n{aa}")
        else:
            x_labels.append(str(pos_1idx))
    
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, fontsize=6, rotation=0)
    # Ensure x-axis tick label font size matches on both plots (they share x-axis via sharex)
    ax.tick_params(axis='x', labelsize=6)
    ax_hist.tick_params(axis='x', labelsize=6)  # Even though hidden, ensure consistency
    
    # Set y-axis to show peptide sequences and retention times
    # For pcolormesh, the y-axis is in data coordinates (0 to num_peptides)
    # We need to set ticks at the center of each row - label every row with its peptide sequence and RT
    # Add 0.5 to center the tick on the row (since pcolormesh uses edges)
    y_tick_positions = [i + 0.5 for i in range(num_peptides)]
    y_labels = [f"{peptides[i]['peptide_seq']} | {peptides[i]['rt']:.2f}" for i in range(num_peptides)]
    
    # Don't set y-axis labels yet - wait until after colorbar
    
    # Set labels
    ax.set_xlabel('Protein Position (1-indexed) with Amino Acid', fontsize=10, fontweight='bold')
    ax.set_ylabel('Retention Time (sec)', fontsize=10, fontweight='bold')
    
    # Add title below the plot (below colorbar)
    title = f'Peptide Coverage Grid (Ordered by Retention Time) - Unique Fragment Pairs per Residue\n{protein_id} (Length: {protein_length} residues)\nBefore filtering: {num_peptides_before} peptides | After filtering (unique RT, at least 10 per position): {num_peptides_after} peptides'
    # Position title below the colorbar with padding (no background)
    fig.text(0.5, 0.005, title, fontsize=12, fontweight='bold', ha='center', va='bottom')
    
    # Add colorbar for quality scores at the bottom of the plot
    from matplotlib.patches import Patch
    from matplotlib.colors import BoundaryNorm
    
    # Create a normalization that shows the actual score values but uses percentile-based mapping
    # We'll use a custom normalization that maps the actual scores
    sm = plt.cm.ScalarMappable(cmap=custom_cmap, 
                               norm=plt.Normalize(vmin=min_sp_score, vmax=max_sp_score))
    sm.set_array([])
    # Position colorbar at the bottom, outside the plot area
    cbar = plt.colorbar(sm, ax=ax, orientation='horizontal', pad=0.15, shrink=0.8)
    cbar.set_label('Peptide Quality Score (sp_score)', fontsize=10, fontweight='bold', labelpad=10)
    cbar.ax.tick_params(labelsize=8)
    # For horizontal colorbar, label should be above the colorbar
    cbar.ax.xaxis.set_label_position('top')
    
    # Remove percentile tick marks - they were confusing
    # The colorbar now only shows the actual score values
    
    # NOW set y-axis labels after colorbar (which might reset them)
    from matplotlib.ticker import FixedLocator, FixedFormatter
    
    # Ensure y-axis limits are correct
    ax.set_ylim(0, num_peptides)
    
    # Set ticks and labels explicitly
    ax.set_yticks(y_tick_positions)
    ax.set_yticklabels(y_labels, fontsize=6)
    
    # Use FixedLocator/Formatter to ensure they stick and don't get reset
    locator = FixedLocator(y_tick_positions)
    formatter = FixedFormatter(y_labels)
    ax.yaxis.set_major_locator(locator)
    ax.yaxis.set_major_formatter(formatter)
    
    # Disable minor ticks to avoid confusion
    ax.yaxis.set_minor_locator(plt.NullLocator())
    
    # Force update
    plt.draw()
    
    # Add simple legend
    legend_elements = [
        Patch(facecolor='white', edgecolor='black', label='No single-AA overhang'),
        Patch(facecolor='yellow', edgecolor='black', label='Single-AA overhang (colored by peptide quality)'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', framealpha=0.9, fontsize=9)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.1, 1, 0.98])
    
    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Peptide RT grid saved to: {output_file}")
    plt.close()


def get_filtered_peptide_indices(csv_file, protein_id, protein_length, fasta_file=None, top_n_limit=20):
    """
    Apply progressive coverage filtering logic:
    1. (Unique RT filter disabled - using all peptides)
    2. Include top 20 highest q-score peptides that result in single-AA or multi-AA overhangs, separately for c-ions and z-ions
    3. Uses q-values for ranking if available, otherwise falls back to sp_score
    4. Progressive coverage: Start with single-AA overhangs, then progressively try
       longer overhang lengths (2-AA, 3-AA, etc.) for uncovered positions until
       maximum coverage is achieved.
    
    Filters c-ions and z-ions independently:
    - Top 20 c-ion peptides per position (by q-score or sp_score)
    - Top 20 z-ion peptides per position (by q-score or sp_score)
    - Uses shorter overhangs first, then longer ones for uncovered positions
    
    Returns a set of row indices (0-based from data rows) that pass the filter.
    """
    # Parse peptides with single-AA overhangs (similar to create_peptide_rt_grid)
    peptides = []
    
    with open(csv_file, 'r') as f:
        lines = f.readlines()
        
        if len(lines) < 1:
            return set(), set(), set()
        
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
            return set(), set(), set()
        
        header_line = lines[header_line_idx]
        header = [col.strip() for col in header_line.split(',')]
        
        try:
            protein_idx = header.index('protein')
            modifications_idx = header.index('modifications')
            peptide_idx = header.index('plain_peptide')
            sp_score_idx = header.index('sp_score')
            single_aa_pairs_idx = header.index('single_aa_overhang_fragment_pairs')
        except ValueError as e:
            print(f"Error: Required column not found: {e}")
            return set(), set(), set()
        
        # Retention time: MS1 only - try MS1_retention_time_sec, then retention_time_sec (Comet/Percolator CSV)
        rt_idx = -1
        for col in ['MS1_retention_time_sec', 'retention_time_sec', 'MS1_retention_time', 'retention_time']:
            if col in header:
                rt_idx = header.index(col)
                break
        if rt_idx < 0:
            print("Error: No retention time column found (tried: MS1_retention_time_sec, retention_time_sec, ...). Plots use MS1 retention time only.")
            return set(), set(), set()
        
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
            return set(), set(), set()
        
        try:
            single_aa_overhangs_idx = header.index('single_aa_overhangs') if 'single_aa_overhangs' in header else -1
            multi_aa_overhangs_idx = header.index('multi_aa_overhangs') if 'multi_aa_overhangs' in header else -1
            multi_aa_pairs_idx = header.index('multi_aa_overhang_fragment_pairs') if 'multi_aa_overhang_fragment_pairs' in header else -1
        except ValueError:
            single_aa_overhangs_idx = -1
            multi_aa_overhangs_idx = -1
            multi_aa_pairs_idx = -1
        
        sequence_positions_idx = header.index('sequence_positions') if 'sequence_positions' in header else -1
        
        # Try to find q-value column (prefer q-values over sp_score for filtering)
        qvalue_idx = -1
        qvalue_column_name = None
        qvalue_is_score = False
        qvalue_is_percent = False
        for col_name in ['percolator_qvalue', 'q-value', 'qvalue', 'FDR', 'fdr', 'percolator_q-value', 'qscore', 'q_score', 'q-value(%)', 'FDR%']:
            try:
                qvalue_idx = header.index(col_name)
                qvalue_column_name = col_name
                # Detect format from column name
                if 'qscore' in col_name.lower() or 'q_score' in col_name.lower():
                    qvalue_is_score = True
                elif '%' in col_name or 'percent' in col_name.lower():
                    qvalue_is_percent = True
                break
            except ValueError:
                continue
        
        use_qvalue = qvalue_idx >= 0
        if use_qvalue:
            print(f"Using q-value column '{qvalue_column_name}' for score-based filtering")
        else:
            print(f"No q-value column found, using sp_score for filtering")
        
        # Determine data start line (after header)
        data_start_line = header_line_idx + 1
        
        # Use csv.reader for proper CSV parsing
        import io
        csv_reader = csv.reader(io.StringIO(''.join(lines[data_start_line:])))
        
        for row_idx, row_data in enumerate(csv_reader):
            if len(row_data) <= max(protein_idx, rt_idx, site_specific_idx):
                continue
            
            protein = row_data[protein_idx] if protein_idx < len(row_data) else ""
            if not protein:
                continue
            
            # Match protein ID (same logic as parse_multi_aa_overhangs and create_peptide_rt_grid)
            clean_protein = re.sub(r'(\d+)$', '', protein)
            clean_protein_alt = re.sub(r'^sp\|', '', clean_protein)
            if protein_id:
                protein_id_clean = re.sub(r'(\d+)$', '', protein_id)
                protein_id_alt = re.sub(r'^sp\|', '', protein_id_clean)
                # Check if they match (allowing for trailing numbers and sp| prefix differences)
                # Also check if protein contains UB2D3 or similar base name
                matches = (clean_protein == protein_id_clean or 
                          clean_protein == protein_id_alt or
                          clean_protein_alt == protein_id_clean or
                          clean_protein_alt == protein_id_alt or
                          protein == protein_id or
                          (clean_protein_alt and protein_id_alt and clean_protein_alt == protein_id_alt) or
                          # Check if base protein name matches (e.g., both contain UB2D3_HUMAN)
                          ('UB2D3' in protein and 'UB2D3' in protein_id))
                if not matches:
                    continue
            
            # Prefer row_data by column index (CSV column order varies; quoted_fields order is unreliable)
            site_specific = ""
            single_aa_str = ""
            single_aa_pairs_str = ""
            if len(row_data) > site_specific_idx and row_data[site_specific_idx]:
                site_specific = row_data[site_specific_idx].strip('"').strip()
            if len(row_data) > single_aa_pairs_idx and row_data[single_aa_pairs_idx]:
                single_aa_pairs_str = row_data[single_aa_pairs_idx].strip('"').strip()
            if single_aa_overhangs_idx >= 0 and len(row_data) > single_aa_overhangs_idx and row_data[single_aa_overhangs_idx]:
                single_aa_str = row_data[single_aa_overhangs_idx].strip('"').strip()
            elif site_specific and sequence_positions_idx >= 0 and len(row_data) > sequence_positions_idx and row_data[sequence_positions_idx]:
                # Derive peptide-relative single_aa_str from sequence_positions and protein positions (single_aa_overhangs_protein_positions)
                seq_pos_str = row_data[sequence_positions_idx].strip('"').strip()
                match = re.match(r'^(\d+)-(\d+)$', seq_pos_str)
                if match:
                    peptide_start = int(match.group(1))
                    for item in site_specific.split(','):
                        item = item.strip()
                        m = re.match(r'^(\d+)([A-Z])$', item)
                        if m:
                            protein_pos = int(m.group(1))
                            res = m.group(2)
                            peptide_pos = protein_pos - peptide_start + 1
                            single_aa_str = (single_aa_str + ',' if single_aa_str else '') + f"{peptide_pos}{res}"
            # Fallback to quoted_fields only when row_data did not provide values (older CSV formats)
            original_line = lines[data_start_line + row_idx] if (data_start_line + row_idx) < len(lines) else ""
            quoted_fields = re.findall(r'"([^"]*)"', original_line)
            if not site_specific and len(quoted_fields) >= 1:
                site_specific = quoted_fields[-1]
            if not single_aa_pairs_str and len(quoted_fields) >= 2:
                single_aa_pairs_str = quoted_fields[-2]
            if not single_aa_str and len(quoted_fields) >= 3:
                single_aa_str = quoted_fields[-3]
            
            # Don't skip rows without site_specific - they might have multi-AA overhangs
            # We'll handle protein offset calculation separately for multi-AA overhangs
            
            try:
                rt = float(row_data[rt_idx]) if row_data[rt_idx] else 0.0
                peptide_seq = row_data[peptide_idx].strip('"').strip() if len(row_data) > peptide_idx else ''
                sp_score = float(row_data[sp_score_idx]) if len(row_data) > sp_score_idx and row_data[sp_score_idx] else 0.0
                
                # Get q-value if available, otherwise use sp_score
                quality_score = None
                if use_qvalue and qvalue_idx < len(row_data):
                    try:
                        qvalue_str = row_data[qvalue_idx].strip()
                        if qvalue_str:
                            raw_value = float(qvalue_str)
                            # Convert to true q-value (0-1) if needed
                            if qvalue_is_percent:
                                quality_score = raw_value / 100.0
                            elif qvalue_is_score:
                                if raw_value > 10:
                                    quality_score = 10.0 ** (-raw_value / 10.0)
                                else:
                                    quality_score = 10.0 ** (-raw_value)
                            else:
                                if raw_value > 1.0:
                                    if raw_value <= 100.0:
                                        quality_score = raw_value / 100.0
                                        qvalue_is_percent = True
                                    elif raw_value > 10:
                                        quality_score = 10.0 ** (-raw_value / 10.0)
                                        qvalue_is_score = True
                                    else:
                                        quality_score = 10.0 ** (-raw_value)
                                        qvalue_is_score = True
                                else:
                                    quality_score = raw_value
                    except (ValueError, IndexError):
                        pass
                
                # Use q-value if available, otherwise fall back to sp_score
                # For q-values: lower is better, so we'll use negative for sorting (higher = better)
                # For sp_score: higher is better, so use as-is
                if quality_score is not None:
                    # Use negative q-value for sorting (so lower q-value = higher sort value)
                    score_for_sorting = -quality_score
                else:
                    score_for_sorting = sp_score
            except (ValueError, IndexError):
                continue
            
            # Parse single-AA overhangs to get positions and fragment pairs by ion type
            single_aa_positions = {}
            single_aa_c_ion_positions = {}  # Positions with c-ion fragment pairs
            single_aa_z_ion_positions = {}  # Positions with z-ion fragment pairs
            
            # Also parse multi-AA overhangs
            multi_aa_overhangs_data = []  # List of {length, positions, c_ion_positions, z_ion_positions}
            
            # Get multi-AA overhang strings (similar to parse_multi_aa_overhangs)
            multi_overhangs_str = ""
            multi_pairs_str = ""
            
            # Due to CSV column mix-up, multi-AA overhangs are in single_aa_overhang_fragment_pairs column
            # and multi-AA fragment pairs are in multi_aa_overhangs column
            # Check both locations to be safe
            if single_aa_pairs_idx >= 0 and len(row_data) > single_aa_pairs_idx:
                candidate = row_data[single_aa_pairs_idx].strip('"').strip()
                if candidate and re.search(r'\d+-\d+[A-Z]{2,}', candidate):
                    multi_overhangs_str = candidate
            if not multi_overhangs_str and multi_aa_overhangs_idx >= 0 and len(row_data) > multi_aa_overhangs_idx:
                candidate = row_data[multi_aa_overhangs_idx].strip('"').strip()
                if candidate and re.search(r'\d+-\d+[A-Z]{2,}', candidate):
                    multi_overhangs_str = candidate
            
            # Multi-AA fragment pairs are in multi_aa_overhangs column (due to CSV mix-up)
            if multi_aa_overhangs_idx >= 0 and len(row_data) > multi_aa_overhangs_idx:
                candidate = row_data[multi_aa_overhangs_idx].strip('"').strip()
                if candidate and re.search(r'[cz]\d+-[cz]', candidate):
                    multi_pairs_str = candidate
            if not multi_pairs_str and multi_aa_pairs_idx >= 0 and len(row_data) > multi_aa_pairs_idx:
                multi_pairs_str = row_data[multi_aa_pairs_idx].strip('"').strip()
            
            # Get single_aa_overhangs to calculate protein start offset (for multi-AA mapping)
            if single_aa_overhangs_idx >= 0 and len(row_data) > single_aa_overhangs_idx:
                single_aa_str = row_data[single_aa_overhangs_idx].strip('"').strip()
            if len(quoted_fields) >= 3:
                if not single_aa_str or not re.search(r'^\d+[A-Z]', single_aa_str.split(',')[0] if ',' in single_aa_str else single_aa_str):
                    candidate = quoted_fields[-3]
                    if candidate and re.search(r'^\d+[A-Z]', candidate.split(',')[0] if ',' in candidate else candidate):
                        single_aa_str = candidate
            
            # Calculate protein start offset for multi-AA overhangs
            protein_start_offset = None
            if site_specific and single_aa_str:
                # Parse single_aa_overhangs: "2L,3K,4R" -> peptide positions
                peptide_positions = []
                for item in single_aa_str.split(','):
                    item = item.strip()
                    if item and len(item) > 1:
                        match = re.match(r'^(\d+)([A-Z])$', item)
                        if match:
                            peptide_positions.append(int(match.group(1)))
                
                # Parse site_specific_residues: "45R,46K,47R" -> protein positions
                protein_positions = []
                for item in site_specific.split(','):
                    item = item.strip()
                    if item and len(item) > 1:
                        match = re.match(r'^(\d+)([A-Z])$', item)
                        if match:
                            protein_positions.append(int(match.group(1)))
                
                # Calculate offset: protein_pos = peptide_pos + offset - 1
                if peptide_positions and protein_positions and len(peptide_positions) == len(protein_positions):
                    offset = protein_positions[0] - peptide_positions[0] + 1
                    matches = sum(1 for p, pr in zip(peptide_positions[:min(3, len(peptide_positions))], 
                                                     protein_positions[:min(3, len(protein_positions))])
                                 if pr == p + offset - 1)
                    if matches >= min(2, len(peptide_positions)):
                        protein_start_offset = offset
            
            if site_specific:
                site_specific_list = []
                for item in site_specific.split(','):
                    item = item.strip()
                    match = re.match(r'^(\d+)([A-Z])$', item)
                    if match:
                        protein_pos = int(match.group(1))
                        protein_residue = match.group(2)
                        site_specific_list.append((protein_pos, protein_residue))
                        if 1 <= protein_pos <= protein_length:
                            single_aa_positions[protein_pos] = 1
                
                # Parse fragment pairs to identify c-ions vs z-ions per position
                if single_aa_str and single_aa_pairs_str and site_specific_list:
                    single_aa_list = [s.strip() for s in single_aa_str.split(',') if s.strip()]
                    single_aa_pairs_list = [p.strip() for p in single_aa_pairs_str.split(',') if p.strip()] if single_aa_pairs_str else []
                    
                    for i, (protein_pos, protein_residue) in enumerate(site_specific_list):
                        if i < len(single_aa_pairs_list):
                            pairs_str = single_aa_pairs_list[i]
                            if pairs_str:
                                # Parse pairs (pipe-separated)
                                pairs = [p.strip() for p in pairs_str.split('|') if p.strip()]
                                has_c_ion = False
                                has_z_ion = False
                                
                                for pair in pairs:
                                    # Normalize pair to check ion type
                                    normalized_pair = normalize_fragment_pair(pair)
                                    # Check if it's a c-ion pair (both fragments start with 'c')
                                    if normalized_pair.startswith('c') and '-' in normalized_pair:
                                        parts = normalized_pair.split('-')
                                        if len(parts) == 2 and parts[1].startswith('c'):
                                            has_c_ion = True
                                    # Check if it's a z-ion pair (both fragments start with 'z')
                                    elif normalized_pair.startswith('z') and '-' in normalized_pair:
                                        parts = normalized_pair.split('-')
                                        if len(parts) == 2 and parts[1].startswith('z'):
                                            has_z_ion = True
                                
                                if has_c_ion:
                                    single_aa_c_ion_positions[protein_pos] = 1
                                if has_z_ion:
                                    single_aa_z_ion_positions[protein_pos] = 1
            
            # Parse multi-AA overhangs if available
            # Try to get protein_start_offset even if we don't have single-AA overhangs
            # (needed for multi-AA overhangs that don't have single-AA counterparts)
            # First try from single-AA overhangs if available
            if not protein_start_offset and site_specific and single_aa_str:
                # Try to calculate offset from single-AA overhangs if available
                peptide_positions = []
                for item in single_aa_str.split(','):
                    item = item.strip()
                    if item and len(item) > 1:
                        match = re.match(r'^(\d+)([A-Z])$', item)
                        if match:
                            peptide_positions.append(int(match.group(1)))
                
                protein_positions = []
                for item in site_specific.split(','):
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
            
            # If we still don't have offset, try using FASTA file to find peptide in protein
            if not protein_start_offset and fasta_file and peptide_seq:
                try:
                    sequences = parse_fasta(fasta_file)
                    clean_protein = re.sub(r'(\d+)$', '', protein)
                    clean_protein_alt = re.sub(r'^sp\|', '', clean_protein)
                    protein_seq = None
                    if clean_protein in sequences:
                        protein_seq = sequences[clean_protein]
                    elif clean_protein_alt in sequences:
                        protein_seq = sequences[clean_protein_alt]
                    elif protein_id:
                        protein_id_clean = re.sub(r'(\d+)$', '', protein_id)
                        protein_id_alt = re.sub(r'^sp\|', '', protein_id_clean)
                        if protein_id_clean in sequences:
                            protein_seq = sequences[protein_id_clean]
                        elif protein_id_alt in sequences:
                            protein_seq = sequences[protein_id_alt]
                    
                    if protein_seq:
                        # Find peptide in protein sequence
                        clean_peptide = re.sub(r'\[.*?\]', '', peptide_seq).upper()
                        protein_seq_upper = protein_seq.upper()
                        peptide_start_in_protein = protein_seq_upper.find(clean_peptide)
                        if peptide_start_in_protein >= 0:
                            # Found it! peptide_start_in_protein is 0-indexed, convert to 1-indexed
                            protein_start_offset = peptide_start_in_protein + 1
                        elif len(clean_peptide) > 2:
                            # Try without first/last residues in case of terminal modifications
                            if protein_seq_upper.find(clean_peptide[1:]) >= 0:
                                protein_start_offset = protein_seq_upper.find(clean_peptide[1:]) + 1
                            elif protein_seq_upper.find(clean_peptide[:-1]) >= 0:
                                protein_start_offset = protein_seq_upper.find(clean_peptide[:-1]) + 1
                except Exception as e:
                    # If FASTA parsing fails, continue without offset
                    pass
            
            # Parse multi-AA overhangs if available (even without single-AA overhangs)
            if multi_overhangs_str and protein_start_offset is not None:
                # Calculate peptide length (removing modifications like [M+16])
                clean_peptide = re.sub(r'\[.*?\]', '', peptide_seq) if peptide_seq else ''
                peptide_length = len(clean_peptide)
                
                overhang_list = [o.strip() for o in multi_overhangs_str.split(',') if o.strip()]
                pairs_list = [p.strip() for p in multi_pairs_str.split(',') if p.strip()] if multi_pairs_str else []
                
                for i, overhang_str in enumerate(overhang_list):
                    # Format: "start-endRESIDUES" (e.g., "5-7RIL")
                    match = re.match(r'^(\d+)-(\d+)([A-Z]+)$', overhang_str)
                    if not match:
                        continue
                    
                    peptide_start = int(match.group(1))  # Peptide position (1-indexed)
                    peptide_end = int(match.group(2))    # Peptide position (1-indexed)
                    residues = match.group(3)
                    overhang_length = peptide_end - peptide_start + 1
                    
                    # CRITICAL VALIDATION: Check that overhang positions are within peptide bounds
                    if peptide_start < 1 or peptide_end > peptide_length or peptide_start > peptide_end:
                        # Enhanced diagnostic: try to reverse-engineer what fragment numbers could have caused this
                        fragment_pairs_str = pairs_list[i] if i < len(pairs_list) else "N/A"
                        print(f"Warning: Skipping invalid multi-AA overhang '{overhang_str}' for peptide '{peptide_seq[:50]}...' "
                              f"(peptide length: {peptide_length}, overhang: {peptide_start}-{peptide_end}, "
                              f"fragment pairs: {fragment_pairs_str})")
                        # Try to diagnose: extract fragment numbers from pairs if available
                        if fragment_pairs_str != "N/A" and len(fragment_pairs_str) > 0:
                            # Try to extract fragment numbers from pairs (handle z1_ format too)
                            pair_match = re.search(r'([cz])(?:\d+|1_(\d+))-([cz])(?:\d+|1_(\d+))', fragment_pairs_str)
                            if pair_match:
                                ion_type = pair_match.group(1)
                                frag1_str = pair_match.group(2) if pair_match.group(2) else (pair_match.group(1) if pair_match.group(1) else "")
                                frag2_str = pair_match.group(4) if pair_match.group(4) else ""
                                # Try simpler pattern
                                simple_match = re.search(r'([cz])(\d+)-([cz])(\d+)', fragment_pairs_str)
                                if simple_match:
                                    ion_type = simple_match.group(1)
                                    frag1 = int(simple_match.group(2))
                                    frag2 = int(simple_match.group(4))
                                    print(f"  Diagnostic: Fragment pair {ion_type}{frag1}-{ion_type}{frag2}, "
                                          f"peptide length {peptide_length}")
                                    if ion_type == 'c':
                                        # N-terminal: end should be frag2
                                        print(f"  Expected: end = frag2 = {frag2}, but got {peptide_end}")
                                    elif ion_type == 'z':
                                        # C-terminal: end = length - frag1
                                        expected_end = peptide_length - frag1
                                        print(f"  Expected: end = length - frag1 = {peptide_length} - {frag1} = {expected_end}, but got {peptide_end}")
                        continue
                    
                    # Map to protein positions
                    protein_start_pos = peptide_start + protein_start_offset - 1
                    protein_end_pos = peptide_end + protein_start_offset - 1
                    
                    # Skip if out of bounds
                    if protein_start_pos < 1 or protein_end_pos > protein_length:
                        continue
                    
                    # Get positions covered by this overhang
                    positions_covered = set(range(protein_start_pos, protein_end_pos + 1))
                    
                    # Get fragment pairs and determine ion series
                    fragment_pairs = []
                    if i < len(pairs_list):
                        pairs = [p.strip() for p in pairs_list[i].split('|') if p.strip()]
                        fragment_pairs = pairs
                    
                    # Determine ion series
                    ion_series = None
                    for pair in fragment_pairs:
                        if pair.startswith('c') and not pair.startswith('c-'):
                            if ion_series is None:
                                ion_series = 'c'
                            elif ion_series != 'c':
                                ion_series = None
                                break
                        elif pair.startswith('z') or 'z1_' in pair:
                            if ion_series is None:
                                ion_series = 'z'
                            elif ion_series != 'z':
                                ion_series = None
                                break
                        else:
                            ion_series = None
                            break
                    
                    if ion_series:
                        multi_aa_overhangs_data.append({
                            'length': overhang_length,
                            'positions': positions_covered,
                            'c_ion_positions': positions_covered if ion_series == 'c' else set(),
                            'z_ion_positions': positions_covered if ion_series == 'z' else set()
                        })
            
            # Store peptide data with both single-AA and multi-AA overhangs
            # Include peptides that have EITHER single-AA OR multi-AA overhangs (or both)
            if single_aa_positions or multi_aa_overhangs_data:
                peptides.append({
                    'row_idx': row_idx,
                    'rt': rt,
                    'peptide_seq': peptide_seq,
                    'single_aa_positions': single_aa_positions,
                    'single_aa_c_ion_positions': single_aa_c_ion_positions,
                    'single_aa_z_ion_positions': single_aa_z_ion_positions,
                    'multi_aa_overhangs': multi_aa_overhangs_data,
                    'sp_score': sp_score,
                    'quality_score': quality_score if quality_score is not None else None,
                    'score_for_sorting': score_for_sorting
                })
    
    if not peptides:
        return set(), set(), set()
    
    # Step 1: Skip unique RT filtering - use all peptides with overhangs
    # (Unique RT filter disabled per user request)
    
    # Step 2: Group peptides by overhang length and build position maps
    # Structure: {length: {position: [(row_idx, score), ...]}}
    # Separate for c-ions and z-ions
    peptides_by_length_c = {}  # {length: {position: [(row_idx, score), ...]}}
    peptides_by_length_z = {}  # {length: {position: [(row_idx, score), ...]}}
    
    for peptide in peptides:
        row_idx = peptide['row_idx']
        score = peptide['score_for_sorting']
        
        # Process single-AA overhangs (length 1)
        for pos in peptide.get('single_aa_c_ion_positions', {}).keys():
            if 1 <= pos <= protein_length:
                if 1 not in peptides_by_length_c:
                    peptides_by_length_c[1] = {}
                if pos not in peptides_by_length_c[1]:
                    peptides_by_length_c[1][pos] = []
                peptides_by_length_c[1][pos].append((row_idx, score))
        
        for pos in peptide.get('single_aa_z_ion_positions', {}).keys():
            if 1 <= pos <= protein_length:
                if 1 not in peptides_by_length_z:
                    peptides_by_length_z[1] = {}
                if pos not in peptides_by_length_z[1]:
                    peptides_by_length_z[1][pos] = []
                peptides_by_length_z[1][pos].append((row_idx, score))
        
        # Process multi-AA overhangs
        for overhang_data in peptide.get('multi_aa_overhangs', []):
            length = overhang_data['length']
            positions = overhang_data['positions']
            c_ion_positions = overhang_data.get('c_ion_positions', set())
            z_ion_positions = overhang_data.get('z_ion_positions', set())
            
            # Add to c-ion map
            for pos in c_ion_positions:
                if 1 <= pos <= protein_length:
                    if length not in peptides_by_length_c:
                        peptides_by_length_c[length] = {}
                    if pos not in peptides_by_length_c[length]:
                        peptides_by_length_c[length][pos] = []
                    peptides_by_length_c[length][pos].append((row_idx, score))
            
            # Add to z-ion map
            for pos in z_ion_positions:
                if 1 <= pos <= protein_length:
                    if length not in peptides_by_length_z:
                        peptides_by_length_z[length] = {}
                    if pos not in peptides_by_length_z[length]:
                        peptides_by_length_z[length][pos] = []
                    peptides_by_length_z[length][pos].append((row_idx, score))
    
    # Step 3: Progressive coverage - try shorter overhangs first, then longer ones for uncovered positions
    # For each ion type (c and z), independently:
    # 1. Start with length 1, get top 10 peptides per position
    # 2. For uncovered positions, try length 2, get top 10 peptides
    # 3. Continue with longer lengths until maximum coverage
    
    def apply_progressive_coverage(peptides_by_length, ion_type_name, top_n_limit):
        """Apply progressive coverage filtering for one ion type."""
        filtered_row_indices = set()
        covered_positions = set()  # Track which positions are covered
        
        # Get all available lengths, sorted
        all_lengths = sorted(peptides_by_length.keys())
        
        # For each length (starting with shortest)
        for length in all_lengths:
            position_map = peptides_by_length[length]
            
            # For each position at this length
            for pos, peptide_list in position_map.items():
                # Skip if position is already covered
                if pos in covered_positions:
                    continue
                
                # Sort by score (higher = better)
                # For q-values: score_for_sorting is negative q-value (lower q-value = higher sort value)
                # For sp_score: score_for_sorting is sp_score (higher = better)
                peptide_list.sort(key=lambda x: x[1], reverse=True)
                
                # Take top N peptides for this position (based on q-score or sp_score)
                top_peptides = peptide_list[:top_n_limit] if len(peptide_list) >= top_n_limit else peptide_list
                
                # Add to filtered set
                for row_idx, _ in top_peptides:
                    filtered_row_indices.add(row_idx)
                
                # Mark this position as covered
                covered_positions.add(pos)
        
        return filtered_row_indices, covered_positions
    
    # Apply progressive coverage for c-ions
    filtered_c_indices, covered_c_positions = apply_progressive_coverage(peptides_by_length_c, "c-ions", top_n_limit)
    
    # Apply progressive coverage for z-ions
    filtered_z_indices, covered_z_positions = apply_progressive_coverage(peptides_by_length_z, "z-ions", top_n_limit)
    
    # Combine results (union)
    filtered_row_indices = filtered_c_indices.union(filtered_z_indices)
    
    # Print statistics
    print(f"Score-based filtering (progressive coverage, top {top_n_limit} highest q-score per position, c-ions and z-ions independently): {len(filtered_row_indices)} peptides pass")
    print(f"  - C-ion positions covered: {len(covered_c_positions)}/{protein_length} positions")
    print(f"  - Z-ion positions covered: {len(covered_z_positions)}/{protein_length} positions")
    print(f"  - C-ion peptides: {len(filtered_c_indices)} (top {top_n_limit} per position by q-score/sp_score)")
    print(f"  - Z-ion peptides: {len(filtered_z_indices)} (top {top_n_limit} per position by q-score/sp_score)")
    
    # Print coverage by length for debugging
    if peptides_by_length_c:
        lengths_used_c = sorted(peptides_by_length_c.keys())
        print(f"  - C-ion overhang lengths used: {lengths_used_c}")
    if peptides_by_length_z:
        lengths_used_z = sorted(peptides_by_length_z.keys())
        print(f"  - Z-ion overhang lengths used: {lengths_used_z}")
    
    # Return filtered indices and coverage information
    # covered_c_positions and covered_z_positions are sets of positions (1-indexed) that have single-AA overhangs
    return filtered_row_indices, covered_c_positions, covered_z_positions


def create_block_constraint_map(overhangs, protein_sequence, protein_id, output_file):
    """
    Create coverage heatmap of multi-AA and single-AA overhangs.
    
    For each residue position and each overhang length, count how many overhangs include that residue.
    This shows redundancy/robustness: regions with many overlapping blocks stand out.
    """
    if not overhangs:
        print("No overhangs found to visualize.")
        return
    
    if not protein_sequence:
        print("Error: Protein sequence required for coverage heatmap")
        return
    
    protein_length = len(protein_sequence)
    
    # Get all unique overhang lengths (including length 1 for single-AA)
    unique_lengths = sorted(set(o['length'] for o in overhangs))
    
    # Create 2D array: coverage[residue_position][overhang_length] = count
    # Use 1-indexed positions (1 to protein_length)
    coverage = np.zeros((len(unique_lengths), protein_length), dtype=int)
    
    # Map length to index
    length_to_idx = {length: idx for idx, length in enumerate(unique_lengths)}
    
    # Count overhangs for each residue position and length
    for overhang in overhangs:
        start = overhang['start']
        end = overhang['end']
        length = overhang['length']
        
        # Skip if out of bounds
        if start < 1 or end > protein_length:
            continue
        
        # Convert to 0-indexed for array
        start_idx = start - 1
        end_idx = end - 1
        
        # Get length index
        length_idx = length_to_idx[length]
        
        # Increment count for all residues in this overhang
        for pos in range(start_idx, end_idx + 1):
            coverage[length_idx, pos] += 1
    
    # Create figure - make it wider to accommodate all position labels
    # For ~147 residues, we need more width
    fig_width = max(20, protein_length * 0.15)  # More width per residue
    fig_height = max(8, len(unique_lengths) * 0.4)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    
    # Create custom colormap: gray for 0, then YlOrRd for 1+
    from matplotlib.colors import ListedColormap
    import matplotlib.cm as cm
    
    # Get max coverage for normalization
    max_coverage = np.max(coverage) if np.max(coverage) > 0 else 1
    
    # Create custom colormap: gray for 0, YlOrRd for 1+
    # Use a colormap with gray at the bottom, then YlOrRd
    colors_list = ['#808080']  # Gray for 0
    # Get YlOrRd colormap colors
    ylorrd = cm.get_cmap('YlOrRd', 255)
    for i in range(255):
        colors_list.append(ylorrd(i))
    
    custom_cmap = ListedColormap(colors_list)
    
    # Normalize coverage: 0 stays 0, 1+ maps to 1-255
    # Shift non-zero values: 0 -> 0, 1 -> 1, max -> 255
    coverage_normalized = coverage.copy().astype(float)
    if max_coverage > 1:
        # Map non-zero values: 1 -> 1, max -> 255
        coverage_normalized[coverage > 0] = 1 + (coverage[coverage > 0] - 1) * 254 / (max_coverage - 1)
    
    im = ax.imshow(coverage_normalized, aspect='auto', cmap=custom_cmap, 
                   interpolation='nearest', origin='lower', vmin=0, vmax=255)
    
    # Set axis labels
    ax.set_xlabel('Protein Position (1-indexed) with Amino Acid', fontsize=10, fontweight='bold')
    ax.set_ylabel('Overhang Length (residues)', fontsize=10, fontweight='bold')
    
    # Set title
    title = f'Overhang Coverage Heatmap (Single-AA + Multi-AA)\n{protein_id} (Length: {protein_length} residues)'
    ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
    
    # Set x-axis ticks to show ALL positions with amino acid letters
    x_ticks = list(range(protein_length))
    
    # Create labels with position number and amino acid letter
    x_labels = []
    for t in x_ticks:
        pos_1idx = t + 1  # Convert to 1-indexed
        if pos_1idx <= len(protein_sequence):
            aa = protein_sequence[t]  # t is 0-indexed, so use directly
            x_labels.append(f"{pos_1idx}\n{aa}")
        else:
            x_labels.append(str(pos_1idx))
    
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, fontsize=6, rotation=0)  # Small font, no rotation
    
    # Set y-axis ticks (overhang lengths)
    # Show all lengths, but if too many, show every Nth
    if len(unique_lengths) <= 30:
        ax.set_yticks(range(len(unique_lengths)))
        ax.set_yticklabels([str(l) for l in unique_lengths])
    else:
        # Show every 5th length
        step = max(1, len(unique_lengths) // 30)
        y_ticks = list(range(0, len(unique_lengths), step))
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([str(unique_lengths[i]) for i in y_ticks])
    
    # Add colorbar with custom ticks
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label('Number of Overhangs', fontsize=11, fontweight='bold')
    
    # Set colorbar ticks to show actual coverage values
    # Map normalized values back to actual coverage
    if max_coverage > 0:
        # Show 0, then a few intermediate values, then max
        if max_coverage <= 5:
            tick_values = list(range(int(max_coverage) + 1))
        else:
            tick_values = [0, 1, 2, 3, 5, 10]
            if max_coverage > 10:
                tick_values.append(int(max_coverage))
        
        # Normalize tick values to colormap positions
        tick_positions = []
        for v in tick_values:
            if v == 0:
                tick_positions.append(0)
            elif max_coverage > 1:
                tick_positions.append(1 + (v - 1) * 254 / (max_coverage - 1))
            else:
                tick_positions.append(1)
        
        cbar.set_ticks(tick_positions)
        cbar.set_ticklabels([str(int(v)) for v in tick_values])
    
    # Remove the duplicate amino acid letters below (they're now in the tick labels)
    
    # Add text annotation explaining the plot
    textstr = 'Color intensity = number of overhangs including each residue\n' \
              'Darker = more redundant/robust regions'
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Adjust layout to make room for all position labels
    plt.tight_layout(rect=[0, 0.1, 1, 0.98])  # Leave more space at bottom for all labels
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Coverage heatmap saved to: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Create genome-browser style visualization of multi-AA overhangs'
    )
    parser.add_argument('--csv', required=True, help='Comet CSV output file')
    parser.add_argument('--fasta', required=True, help='FASTA file with protein sequences')
    parser.add_argument('--output', default='multi_aa_overhang_map.png',
                       help='Output PNG file (default: multi_aa_overhang_map.png)')
    parser.add_argument('--protein', help='Specific protein ID to visualize (optional)')
    parser.add_argument('--demo', action='store_true',
                       help='Create demo plot with synthetic data (for testing)')
    parser.add_argument('--only-peptides-with-single-aa', action='store_true',
                       help='Only include peptides that have fragments contributing to single-AA overhangs')
    parser.add_argument('--single-aa-histogram', action='store_true',
                       help='Create a histogram of single-AA overhang counts per position (instead of heatmap)')
    parser.add_argument('--single-aa-fragment-pairs', action='store_true',
                       help='Create a histogram of fragment pair counts per position (instead of peptide counts)')
    parser.add_argument('--peptide-rt-grid', action='store_true',
                       help='Create a grid plot with peptides ordered by retention time, showing single-AA overhangs')
    
    args = parser.parse_args()
    
    # Demo mode: create synthetic data
    if args.demo:
        print("Creating demo visualization with synthetic multi-AA overhang data...")
        sequences = parse_fasta(args.fasta)
        if not sequences:
            print("Error: No sequences found in FASTA file")
            return
        
        protein_id = list(sequences.keys())[0]
        protein_sequence = sequences[protein_id]
        
        # Create synthetic overhangs
        demo_overhangs = [
            {'start': 10, 'end': 12, 'residues': 'ABC', 'fragment_pairs': ['c10-c12'], 
             'ion_series': 'c', 'protein_start': 10, 'peptide_seq': 'TEST', 
             'protein_id': protein_id, 'length': 3},
            {'start': 15, 'end': 18, 'residues': 'DEFG', 'fragment_pairs': ['c15-c18'], 
             'ion_series': 'c', 'protein_start': 15, 'peptide_seq': 'TEST', 
             'protein_id': protein_id, 'length': 4},
            {'start': 25, 'end': 30, 'residues': 'HIJKLM', 'fragment_pairs': ['z25-z30'], 
             'ion_series': 'z', 'protein_start': 25, 'peptide_seq': 'TEST', 
             'protein_id': protein_id, 'length': 6},
            {'start': 35, 'end': 42, 'residues': 'NOPQRSTU', 'fragment_pairs': ['c35-c42'], 
             'ion_series': 'c', 'protein_start': 35, 'peptide_seq': 'TEST', 
             'protein_id': protein_id, 'length': 8},
            {'start': 50, 'end': 55, 'residues': 'VWXYZ', 'fragment_pairs': ['z50-z55'], 
             'ion_series': 'z', 'protein_start': 50, 'peptide_seq': 'TEST', 
             'protein_id': protein_id, 'length': 6},
        ]
        
        create_block_constraint_map(demo_overhangs, protein_sequence, protein_id, args.output)
        return
    
    # Parse FASTA
    sequences = parse_fasta(args.fasta)
    if not sequences:
        print("Error: No sequences found in FASTA file")
        return
    
    # Parse multi-AA overhangs from CSV
    if args.protein:
        protein_id = args.protein
    else:
        # Use first protein in FASTA
        protein_id = list(sequences.keys())[0]
        print(f"No protein specified, using first protein: {protein_id}")
    
    # Try to find matching protein ID in CSV
    print(f"Looking for overhangs for protein: {protein_id}")
    
    # First, parse single-AA overhangs to identify which peptides have them
    print("Parsing single-AA overhangs...")
    if args.only_peptides_with_single_aa:
        single_aa_overhangs, peptides_with_single_aa = parse_single_aa_overhangs(
            args.csv, protein_id, args.fasta, return_peptide_indices=True)
        print(f"Found {len(single_aa_overhangs)} single-AA overhangs from {len(peptides_with_single_aa)} peptides")
    else:
        single_aa_overhangs = parse_single_aa_overhangs(args.csv, protein_id, args.fasta)
        peptides_with_single_aa = None
        print(f"Found {len(single_aa_overhangs)} single-AA overhangs")
    
    # Parse multi-AA overhangs (optionally filtered to only peptides with single-AA overhangs)
    print("Parsing multi-AA overhangs...")
    multi_overhangs = parse_multi_aa_overhangs(
        args.csv, protein_id, args.fasta, 
        peptides_with_single_aa=peptides_with_single_aa if args.only_peptides_with_single_aa else None)
    print(f"Found {len(multi_overhangs)} multi-AA overhangs")
    
    # Combine all overhangs
    all_overhangs = list(multi_overhangs) + list(single_aa_overhangs)
    print(f"Total overhangs (multi-AA + single-AA): {len(all_overhangs)}")
    
    # If no overhangs found, try without protein filter to see what's available
    if not all_overhangs:
        print("\nTrying to find any overhangs (without protein filter)...")
        if args.only_peptides_with_single_aa:
            all_single, all_peptide_indices = parse_single_aa_overhangs(
                args.csv, None, args.fasta, return_peptide_indices=True)
            all_multi = parse_multi_aa_overhangs(
                args.csv, None, args.fasta, peptides_with_single_aa=all_peptide_indices)
        else:
            all_multi = parse_multi_aa_overhangs(args.csv, None, args.fasta)
            all_single = parse_single_aa_overhangs(args.csv, None, args.fasta)
        if all_multi or all_single:
            print(f"Found {len(all_multi)} multi-AA and {len(all_single)} single-AA overhangs total")
            # Use first protein from overhangs
            if all_multi:
                protein_id = all_multi[0]['protein_id']
            elif all_single:
                protein_id = all_single[0]['protein_id']
            multi_overhangs = [o for o in all_multi if o['protein_id'] == protein_id]
            single_aa_overhangs = [o for o in all_single if o['protein_id'] == protein_id]
            all_overhangs = list(multi_overhangs) + list(single_aa_overhangs)
    
    if not all_overhangs:
        print(f"No overhangs found for protein {protein_id}")
        return
    
    # Get protein sequence
    protein_sequence = sequences.get(protein_id, '')
    if not protein_sequence:
        print(f"Warning: Protein {protein_id} not found in FASTA file")
        protein_sequence = ''
    
    # Create visualization
    if args.peptide_rt_grid:
        # Create grid plot with peptides ordered by retention time
        create_peptide_rt_grid(args.csv, protein_id, args.fasta, args.output)
    elif args.single_aa_fragment_pairs:
        # Create histogram of fragment pair counts for single-AA overhangs
        create_single_aa_fragment_pair_histogram(single_aa_overhangs, protein_sequence, protein_id, args.output)
    elif args.single_aa_histogram:
        # Create histogram of single-AA overhangs only (peptide counts)
        create_single_aa_histogram(single_aa_overhangs, protein_sequence, protein_id, args.output)
    else:
        # Create heatmap with combined overhangs
        # Apply the same filtering as peptide_rt_grid
        if protein_sequence:
            filtered_indices = get_filtered_peptide_indices(args.csv, protein_id, len(protein_sequence))
            print(f"Applying RT grid filter: {len(filtered_indices)} peptides pass the filter")
            
            # Re-parse overhangs with the filter applied
            multi_aa_overhangs = parse_multi_aa_overhangs(args.csv, protein_id, args.fasta, 
                                                          filtered_peptide_indices=filtered_indices)
            single_aa_overhangs = parse_single_aa_overhangs(args.csv, protein_id, args.fasta)
            # Filter single-AA overhangs too
            if filtered_indices:
                single_aa_overhangs = [o for o in single_aa_overhangs if o.get('row_idx', -1) in filtered_indices]
            all_overhangs = multi_aa_overhangs + single_aa_overhangs
        else:
            all_overhangs = multi_aa_overhangs + single_aa_overhangs
        
        create_block_constraint_map(all_overhangs, protein_sequence, protein_id, args.output)


if __name__ == '__main__':
    main()
