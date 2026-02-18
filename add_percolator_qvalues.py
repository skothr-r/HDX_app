#!/usr/bin/env python3
"""
Add Percolator q-values to Comet CSV output.

This script:
1. Checks if Percolator is available
2. Runs Percolator on a .pin file (if available) or generates one from Comet search
3. Merges q-values and PEP scores into the CSV file

Usage:
    python3 add_percolator_qvalues.py --csv input.csv --pin input.pin --output output.csv
    python3 add_percolator_qvalues.py --csv input.csv --comet-pin-dir . --output output.csv
"""

import argparse
import subprocess
import sys
import os
import csv
import re
from collections import defaultdict

import numpy as np
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Warning: pandas not available, using manual CSV parsing")

UNUSED_WORKFLOW_COLUMNS = {
    'exp_neutral_mass',
    'ions_total',
    'modified_peptide',
    'protein_count',
    'sp_rank',
    'single_aa_overhang_fragment_pairs',
    'single_aa_overhangs_protein_positions',
}

def _find_percolator():
    """Return path to Percolator executable, or None if not found.
    Checks PATH first, then bundled percolator.linux (for Streamlit Cloud).
    """
    import shutil
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # PATH first
    exe = shutil.which('percolator')
    if exe:
        return exe
    # Bundled Linux binary (Streamlit Cloud) - check multiple locations
    if sys.platform.startswith('linux'):
        candidates = [
            os.path.join(script_dir, 'percolator.linux'),
            os.path.join(os.path.dirname(script_dir), 'percolator.linux'),  # parent (repo root)
            os.path.join(os.getcwd(), 'percolator.linux'),
        ]
        # Streamlit Cloud: app root may be /mount/src/<app_name>
        for env in ('STREAMLIT_APP_ROOT', 'PWD'):
            root = os.environ.get(env)
            if root:
                candidates.append(os.path.join(root, 'percolator.linux'))
        for bundled in candidates:
            if bundled and os.path.exists(bundled) and os.access(bundled, os.X_OK):
                return bundled
    return None


def check_percolator():
    """Check if Percolator is available."""
    exe = _find_percolator()
    if not exe:
        print("Percolator not found in PATH.")
        print("To install Percolator:")
        print("  Option 1: conda install -c bioconda percolator")
        print("  Option 2: Download from https://github.com/percolator/percolator/releases")
        return False

    try:
        result = subprocess.run([exe],
                              capture_output=True, text=True, timeout=5)
        if 'Percolator version' in result.stderr or 'Percolator version' in result.stdout:
            version_line = [line for line in (result.stderr + result.stdout).split('\n')
                          if 'Percolator version' in line]
            if version_line:
                print(f"Found Percolator: {version_line[0].strip()}")
                return True
        print("Found Percolator (version check format unexpected)")
        return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    print("Percolator not found in PATH.")
    print("To install Percolator:")
    print("  Option 1: conda install -c bioconda percolator")
    print("  Option 2: Download from https://github.com/percolator/percolator/releases")
    return False

def _log_pin_file_info(pin_path):
    """Print .pin line count and mtime so user can verify it was regenerated."""
    try:
        with open(pin_path, 'r') as f:
            first = f.readline()
            n_lines = 1 + sum(1 for _ in f)
        # First line is header; data lines = n_lines - 1
        n_psms = n_lines - 1
        mtime = os.path.getmtime(pin_path)
        import time
        mtime_str = time.strftime('%Y-%m-%d %H:%M', time.localtime(mtime))
        print(f"  .pin has {n_psms} PSM lines (modified {mtime_str})")
    except Exception:
        pass


def find_pin_file(csv_file, pin_dir=None):
    """Find .pin file corresponding to CSV file."""
    csv_basename = os.path.splitext(os.path.basename(csv_file))[0]
    
    # Remove common suffixes
    csv_basename = re.sub(r'_PP$', '', csv_basename)  # Remove _PP suffix
    
    search_dirs = []
    if pin_dir:
        search_dirs.append(pin_dir)
    search_dirs.append(os.path.dirname(csv_file))
    search_dirs.append(os.path.dirname(os.path.abspath(csv_file)))
    
    # Try different .pin file names
    pin_candidates = [
        f"{csv_basename}.pin",
        f"{csv_basename}_PP.pin",
        f"{os.path.basename(csv_file).replace('.csv', '.pin')}",
    ]
    
    for search_dir in search_dirs:
        if not search_dir:
            continue
        for pin_name in pin_candidates:
            pin_path = os.path.join(search_dir, pin_name)
            if os.path.exists(pin_path):
                print(f"Found .pin file: {pin_path}")
                return pin_path
    
    print(f"Could not find .pin file for {csv_file}")
    print(f"Searched in: {', '.join(search_dirs)}")
    print(f"Looking for files matching: {', '.join(pin_candidates)}")
    return None

def run_percolator(pin_file, output_prefix=None, test_fdr=None):
    """Run Percolator on a .pin file.
    test_fdr: if set (e.g. 1.0), use -t to relax FDR threshold so more PSMs get q-values.
    """
    if output_prefix is None:
        output_prefix = os.path.splitext(pin_file)[0]
    
    # Percolator -m option creates output file (may be without extension or with .psms)
    # Check multiple possible output file names
    possible_outputs = [
        output_prefix,  # No extension (most common with -m flag)
        f"{output_prefix}.psms",  # With .psms extension
        f"{output_prefix}.percolator",  # Alternative format
        f"{output_prefix}.pout",  # XML output format
    ]
    
    print(f"Running Percolator on {pin_file}...")
    print(f"Output will be written to one of: {', '.join(possible_outputs)}")
    
    try:
        # Run Percolator with default settings
        # -Y: use target-decoy competition for FDR estimation
        # -U: keep all PSMs (do not deduplicate per scan); without this only one PSM per (ScanNr, ExpMass) is output
        # -m: output tab-delimited PSM results to file
        # -t: testFDR; default 0.01; use 1.0 to report q-values for all PSMs (no FDR filtering)
        percolator_exe = _find_percolator()
        if not percolator_exe:
            print("Percolator not found.")
            return None
        cmd = [percolator_exe, '-Y', '-U', '-m', output_prefix, pin_file]
        if test_fdr is not None:
            cmd = [percolator_exe, '-Y', '-U', '-t', str(test_fdr), '-m', output_prefix, pin_file]
            print(f"  Using -t {test_fdr} to include more PSMs in output")
        print(f"Command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        
        if result.returncode != 0:
            print(f"Error running Percolator:")
            print(result.stderr)
            if result.stdout:
                print("Stdout:", result.stdout)
            return None
        
        # Check for output file (try all possible names)
        for percolator_output in possible_outputs:
            if os.path.exists(percolator_output):
                print(f"Percolator completed successfully. Output: {percolator_output}")
                return percolator_output
        
        # If none found, report error
        print(f"Warning: Percolator output file not found. Checked:")
        for alt in possible_outputs:
            print(f"  - {alt}")
        return None
            
    except subprocess.TimeoutExpired:
        print("Error: Percolator timed out after 1 hour")
        return None
    except Exception as e:
        print(f"Error running Percolator: {e}")
        return None

def parse_percolator_output(percolator_file):
    """Parse Percolator output file and extract q-values and PEPs."""
    qvalues = {}  # SpecId -> (qvalue, PEP)
    
    print(f"Parsing Percolator output: {percolator_file}")
    
    try:
        with open(percolator_file, 'r') as f:
            # Skip header lines (version info, etc.) until we find the column header
            header_line = None
            for line in f:
                line = line.strip()
                # Look for the header row (contains PSMId or SpecId)
                if 'PSMId' in line or 'SpecId' in line:
                    header_line = line
                    break
            
            if not header_line:
                print(f"Error: Could not find PSMId or SpecId header row in Percolator output")
                return {}
            
            header = header_line.split('\t')
            
            # Find column indices
            try:
                specid_idx = header.index('PSMId')
            except ValueError:
                try:
                    specid_idx = header.index('SpecId')
                except ValueError:
                    print(f"Error: Could not find PSMId or SpecId column in Percolator output")
                    print(f"Available columns: {', '.join(header)}")
                    return {}
            
            try:
                qvalue_idx = header.index('q-value')
            except ValueError:
                try:
                    qvalue_idx = header.index('percolator q-value')
                except ValueError:
                    print(f"Error: Could not find q-value column in Percolator output")
                    print(f"Available columns: {', '.join(header)}")
                    return {}
            
            try:
                pep_idx = header.index('posterior_error_prob')
            except ValueError:
                try:
                    pep_idx = header.index('PEP')
                except ValueError:
                    pep_idx = None
            
            # Parse data
            for line in f:
                fields = line.strip().split('\t')
                if len(fields) <= max(specid_idx, qvalue_idx):
                    continue
                
                specid = fields[specid_idx]
                try:
                    qvalue = float(fields[qvalue_idx])
                    pep = float(fields[pep_idx]) if pep_idx and pep_idx < len(fields) else None
                    qvalues[specid] = (qvalue, pep)
                except (ValueError, IndexError):
                    continue
        
        print(f"Parsed {len(qvalues)} PSMs from Percolator output")
        if qvalues:
            sample_ids = list(qvalues.keys())[:3]
            print(f"  Example Percolator SpecIds: {sample_ids}")
        return qvalues
        
    except Exception as e:
        print(f"Error parsing Percolator output: {e}")
        return {}

def create_specid_from_csv_row(row_data, header):
    """Create SpecId from CSV row to match Percolator format."""
    try:
        scan_idx = header.index('scan')
        charge_idx = header.index('charge')
        num_idx = header.index('num') if 'num' in header else None
        
        scan = row_data[scan_idx] if scan_idx < len(row_data) else ""
        charge = row_data[charge_idx] if charge_idx < len(row_data) else ""
        num = row_data[num_idx] if num_idx and num_idx < len(row_data) else "1"
        
        # Get base name from CSV (first line)
        # Format: base_name_scan_charge_num
        # We'll need to extract base name from the file or use a default
        return f"{scan}_{charge}_{num}"
    except (ValueError, IndexError) as e:
        return None

def _infer_specid_base_names(qvalues):
    """Infer possible base names from Percolator SpecIds (e.g. mzML base from .pin)."""
    bases = set()
    for specid in qvalues:
        # Format: base_scan_charge_num or data/base_scan_charge_num; scan/charge/num are last 3 parts
        s = specid.replace('data/', '').strip()
        parts = s.split('_')
        if len(parts) >= 4:
            base = '_'.join(parts[:-3])  # everything before scan_charge_num
            if base:
                bases.add(base)
    return list(bases)


def _sort_by_sequence_start_len_charge(df):
    """Sort rows by sequence start, peptide length (short->long), then charge."""
    pos_col = 'protein_position' if 'protein_position' in df.columns else ('sequence_positions' if 'sequence_positions' in df.columns else None)
    seq_col = 'peptide_sequence' if 'peptide_sequence' in df.columns else ('plain_peptide' if 'plain_peptide' in df.columns else None)
    if pos_col is None and seq_col is None:
        return df

    def _start_and_len(row):
        pos_val = row.get(pos_col, '') if pos_col else ''
        s = str(pos_val).strip() if pos_val is not None else ''
        if s and '-' in s:
            try:
                a, b = s.split('-', 1)
                start = int(str(a).strip())
                end = int(str(b).strip().split(',')[0])
                ln = max(0, end - start + 1)
                return start, ln
            except Exception:
                pass
        seq = str(row.get(seq_col, '')).strip() if seq_col else ''
        return 999999, (len(seq) if seq else 999999)

    start_len = df.apply(_start_and_len, axis=1, result_type='expand')
    df['_sort_start'] = start_len[0]
    df['_sort_len'] = start_len[1]
    df['_sort_charge'] = pd.to_numeric(df.get('charge'), errors='coerce').fillna(999999)
    df = df.sort_values(by=['_sort_start', '_sort_len', '_sort_charge'], ascending=[True, True, True], kind='mergesort')
    df = df.drop(columns=['_sort_start', '_sort_len', '_sort_charge'])
    return df


def _move_tail_columns_for_display_df(df):
    """
    Keep MS1_RT_minutes/intensity early; move MS1_RT_sec+MS2_RT* to very end.
    """
    # Backward compatibility for old column name
    if 'MS1_mz_error' in df.columns and 'MS1_mz_error_ppm' not in df.columns:
        df = df.rename(columns={'MS1_mz_error': 'MS1_mz_error_ppm'})
    front_cols = [c for c in ['protein_position', 'peptide_sequence', 'charge', 'observed_mz', 'theoretical_mz', 'MS1_mz_error_ppm', 'MS1_RT_minutes'] if c in df.columns]
    ms1_cols = [c for c in ['MS1_RT_intensity'] if c in df.columns]
    frag_tail = [c for c in ['comet_matched_frags', 'comet_matched_frags_mz', 'comet_matched_frags_intensities', 'comet_matched_frags_quality_scores'] if c in df.columns]
    ms2_tail = [c for c in ['MS1_RT_sec', 'MS2_RT_sec', 'MS2_RT_minutes'] if c in df.columns]
    if not (front_cols or ms1_cols or frag_tail or ms2_tail):
        return df
    rest = [c for c in df.columns if c not in front_cols and c not in ms1_cols and c not in frag_tail and c not in ms2_tail]
    return df[front_cols + ms1_cols + rest + frag_tail + ms2_tail]


def merge_qvalues_into_csv(csv_file, qvalues, output_file, diagnose_unmatched=False, fill_unmatched=None, specid_bases=None):
    """Merge q-values into CSV file. If diagnose_unmatched=True, print why each unmatched row failed.
    If fill_unmatched=(qval, pep) is set, use those values for rows that have no Percolator match (so every row has a value)."""
    print(f"Merging q-values into {csv_file}...")
    
    # Try to use pandas if available
    # Use global to access module-level HAS_PANDAS
    global HAS_PANDAS
    use_pandas = HAS_PANDAS
    df = None
    
    if use_pandas:
        try:
            # Comet CSV has a metadata first line (CometVersion, file, date, ...); skip it so line 2 is the real header
            with open(csv_file, 'r') as f:
                first_line = f.readline()
            skip = 0
            if first_line.strip().startswith('CometVersion') or (first_line.count(',') + 1) < 10:
                skip = 1
            # Get header to know expected column count (bad lines are then kept, not dropped)
            with open(csv_file, 'r') as f:
                for _ in range(skip):
                    f.readline()
                header_line = f.readline()
            ncols = len([c for c in header_line.split(',')])
            bad_line_count = [0]  # mutable so callable can increment

            def keep_bad_line(bad_line):
                """Salvage bad line so we keep the row instead of dropping it (pad/truncate to ncols)."""
                bad_line_count[0] += 1
                if isinstance(bad_line, list):
                    parts = bad_line
                else:
                    parts = str(bad_line).split(',')
                if len(parts) >= ncols:
                    return parts[:ncols]
                return parts + [''] * (ncols - len(parts))

            df = pd.read_csv(csv_file, sep=',', skiprows=skip, engine='python', quotechar='"', on_bad_lines=keep_bad_line)
            if bad_line_count[0] > 0:
                print(f"  Note: {bad_line_count[0]} row(s) had CSV parse issues (e.g. unescaped quotes); kept with salvaged fields to avoid dropping rows.")
        except Exception as e:
            print(f"Error reading CSV with pandas: {e}, falling back to manual parsing")
            use_pandas = False

    input_row_count = len(df) if use_pandas else None  # track for row-preservation check
    
    if not use_pandas:
        # Fallback to manual parsing
        with open(csv_file, 'r') as f:
            lines = f.readlines()
        
        if len(lines) < 1:
            print("Error: CSV file too short")
            return False
        
        # Parse header
        header_line = lines[0]
        header = [col.strip() for col in header_line.split(',')]
        return merge_qvalues_into_csv_manual(csv_file, qvalues, output_file, header, lines)
    
    # Get base name(s) to try: CSV filename, plus any inferred from Percolator SpecIds (e.g. mzML base in .pin)
    csv_base = os.path.splitext(os.path.basename(csv_file))[0]
    csv_base = csv_base.replace('_PP', '').replace('_with_ms1', '').replace('_with_qvalues', '')
    base_names = [csv_base]
    if specid_bases:
        for b in specid_bases:
            if b and b not in base_names:
                base_names.append(b)
    
    # Normalize q-value/PEP column names to canonical workflow names.
    if 'percolator_qvalue' in df.columns and 'perc_qvalue' not in df.columns:
        df = df.rename(columns={'percolator_qvalue': 'perc_qvalue'})
    if 'q-value' in df.columns and 'perc_qvalue' not in df.columns:
        df = df.rename(columns={'q-value': 'perc_qvalue'})
    if 'percolator_PEP' in df.columns and 'perc_PEP' not in df.columns:
        df = df.rename(columns={'percolator_PEP': 'perc_PEP'})
    if 'PEP' in df.columns and 'perc_PEP' not in df.columns:
        df = df.rename(columns={'PEP': 'perc_PEP'})
    # Standardize legacy retention_time headers to RT headers.
    for old, new in [
        ('MS1_retention_time_sec', 'MS1_RT_sec'),
        ('MS1_retention_time_min', 'MS1_RT_minutes'),
        ('MS1_retention_time_intensity', 'MS1_RT_intensity'),
        ('MS2_retention_time_sec', 'MS2_RT_sec'),
        ('MS2_retention_time_min', 'MS2_RT_minutes'),
        ('retention_time_sec', 'RT_sec'),
        ('retention_time_min', 'RT_minutes'),
    ]:
        if old in df.columns:
            if new in df.columns:
                df[new] = df[old].where(pd.notna(df[old]), df[new])
                df = df.drop(columns=[old])
            else:
                df = df.rename(columns={old: new})

    has_qvalue = 'perc_qvalue' in df.columns
    has_pep = 'perc_PEP' in df.columns
    if not has_qvalue:
        df['perc_qvalue'] = None
    if not has_pep:
        df['perc_PEP'] = None
    
    # Build lookup by (scan, charge) for fallback matching
    # Percolator outputs one PSM per (scan, charge), but Comet may have multiple num values
    # Comet PIN writes SpecId as baseName_scan_charge_num (no "data/" prefix)
    scan_charge_lookup = {}
    for specid_key, (qv, p) in qvalues.items():
        parts = specid_key.split('_')
        scan_str = None
        charge_str = None
        if len(parts) >= 4:
            # Format: data/WT_nep2_0MUrea_08_scan_charge_num or WT_nep2_0MUrea_08_scan_charge_num
            scan_str = parts[-3]
            charge_str = parts[-2]
        elif len(parts) == 3:
            # Format: scan_charge_num only
            scan_str = parts[0]
            charge_str = parts[1]
        if scan_str is not None and charge_str is not None:
            for key in [(scan_str, charge_str)]:
                if key not in scan_charge_lookup:
                    scan_charge_lookup[key] = (qv, p)
            # Normalized numeric key so "1234"/"1234.0" and "2"/"2.0" match
            try:
                key_norm = (str(int(float(scan_str))), str(int(float(charge_str))))
                if key_norm not in scan_charge_lookup:
                    scan_charge_lookup[key_norm] = (qv, p)
            except (ValueError, TypeError):
                pass
    
    # Resolve scan/charge/num column names (CSV may use different casing or names)
    def _get_col(row, candidates, default=None):
        for c in candidates:
            if c in row.index and pd.notna(row.get(c)) and row.get(c) != '':
                try:
                    return row[c]
                except (TypeError, KeyError):
                    pass
        return default
    scan_candidates = ['scan', 'Scan', 'scan_number', 'scan_id', 'scannum']
    charge_candidates = ['charge', 'Charge', 'charge_state']
    num_candidates = ['num', 'Num', 'rank', 'hit_rank']

    matched_count = 0
    unmatched_count = 0
    unmatched_sample_variants = []  # first few unmatched SpecId variants we tried (for debug)
    unmatched_details = []  # for --diagnose-unmatched: (scan, charge, num_str, in_lookup, peptide_key)

    # Match rows to Percolator SpecIds (use row_num for fallback scan - idx can be tuple if index is non-integer)
    for row_num, (idx, row) in enumerate(df.iterrows(), start=1):
        qvalue = None
        pep = None
        
        # Create SpecId variants (Comet PIN uses baseName_scan_charge_num, no "data/" prefix)
        # Use flexible column names so Scan/scan, etc. all work
        scan_val = _get_col(row, scan_candidates)
        try:
            scan = str(int(float(scan_val))) if scan_val is not None else str(row_num)
        except (ValueError, TypeError):
            scan = str(row_num)
        charge_val = _get_col(row, charge_candidates)
        try:
            charge = str(int(float(charge_val))) if charge_val is not None else "1"
        except (ValueError, TypeError):
            charge = "1"
        num_val = _get_col(row, num_candidates)
        try:
            num_str = str(int(float(num_val))) if num_val is not None else "1"
        except (ValueError, TypeError):
            num_str = "1"
        try:
            num_int = int(float(num_val)) if num_val is not None else 1
        except (ValueError, TypeError):
            num_int = 1
        
        # First try exact match with CSV's num value; try all base names (CSV base + inferred from .pin SpecIds)
        specid_variants = [f"{scan}_{charge}_{num_str}"]
        for bn in base_names:
            specid_variants.extend([
                f"{bn}_{scan}_{charge}_{num_str}",
                f"data/{bn}_{scan}_{charge}_{num_str}",
                f"data/{bn}_PP_{scan}_{charge}_{num_str}",
                f"{bn}_PP_{scan}_{charge}_{num_str}",
            ])
        
        matched = False
        for variant in specid_variants:
            if variant in qvalues:
                qvalue, pep = qvalues[variant]
                matched_count += 1
                matched = True
                break
        
        # If not matched and CSV has num > 1, try with num=1 (Percolator often only outputs best PSM)
        if not matched and num_int > 1:
            specid_variants_num1 = [f"{scan}_{charge}_1"]
            for bn in base_names:
                specid_variants_num1.extend([
                    f"data/{bn}_PP_{scan}_{charge}_1",
                    f"data/{bn}_{scan}_{charge}_1",
                    f"{bn}_{scan}_{charge}_1",
                ])
            for variant in specid_variants_num1:
                if variant in qvalues:
                    qvalue, pep = qvalues[variant]
                    matched_count += 1
                    matched = True
                    break
        
        # If still not matched, try matching by (scan, charge) with any num value
        if not matched:
            for key in [(scan, charge), (str(int(float(scan))), str(int(float(charge))))]:
                try:
                    if key in scan_charge_lookup:
                        qvalue, pep = scan_charge_lookup[key]
                        matched_count += 1
                        matched = True
                        break
                except (ValueError, TypeError):
                    pass
        
        # Last resort: try without num (just scan_charge) - this is less reliable
        if not matched:
            specid_variants_no_num = [f"{scan}_{charge}"]
            for bn in base_names:
                specid_variants_no_num.extend([
                    f"data/{bn}_PP_{scan}_{charge}",
                    f"data/{bn}_{scan}_{charge}",
                    f"{bn}_{scan}_{charge}",
                ])
            for variant in specid_variants_no_num:
                if variant in qvalues:
                    qvalue, pep = qvalues[variant]
                    matched_count += 1
                    matched = True
                    break
        
        if not matched:
            unmatched_count += 1
            if fill_unmatched is not None:
                fill_q, fill_pep = fill_unmatched
                qvalue, pep = fill_q, fill_pep
            if len(unmatched_sample_variants) < 3:
                unmatched_sample_variants.append(specid_variants[0] if specid_variants else f"{scan}_{charge}_{num_str}")
            if diagnose_unmatched:
                try:
                    key_norm = (str(int(float(scan))), str(int(float(charge))))
                except (ValueError, TypeError):
                    key_norm = (scan, charge)
                in_lookup = ((scan, charge) in scan_charge_lookup or
                             (key_norm in scan_charge_lookup))
                pep_str = str(row.get('plain_peptide', row.get('sequence', '')))[:40]
                mod_str = str(row.get('modifications', '-'))[:20]
                peptide_key = f"{pep_str}_{charge}_{mod_str}"
                unmatched_details.append((scan, charge, num_str, in_lookup, peptide_key))

        # Update q-value columns
        df.at[idx, 'perc_qvalue'] = qvalue
        df.at[idx, 'perc_PEP'] = pep
    
    # Unmatched rows are left with missing/NaN percolator_qvalue and percolator_PEP (no E-value fallback).

    # Drop columns unused in this workflow to reduce CSV noise.
    drop_cols = [c for c in UNUSED_WORKFLOW_COLUMNS if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        print(f"Dropped unused columns: {', '.join(drop_cols)}")

    # Reorder: place Percolator score columns directly after observed_mz.
    if 'observed_mz' in df.columns:
        preferred = ['perc_qvalue', 'perc_PEP']
        present = [c for c in preferred if c in df.columns]
        if present:
            remainder = [c for c in df.columns if c not in present]
            anchor_idx = remainder.index('observed_mz')
            reordered = remainder[:anchor_idx + 1] + present + remainder[anchor_idx + 1:]
            df = df[reordered]

    df = _sort_by_sequence_start_len_charge(df)
    df = _move_tail_columns_for_display_df(df)

    # Write output (same number of rows as input; we only add/update columns)
    output_row_count = len(df)
    df.to_csv(output_file, index=False)

    if input_row_count is not None:
        if output_row_count == input_row_count:
            print(f"Rows preserved: {input_row_count} -> {output_row_count} (all rows kept)")
        else:
            print(f"WARNING: Row count changed: input {input_row_count}, output {output_row_count}. Rows should not be dropped.")

    num_percolator_scan_charge = len(scan_charge_lookup)
    print(f"Merged q-values: {matched_count} matched, {unmatched_count} unmatched")
    if fill_unmatched is not None and unmatched_count > 0:
        print(f"  Filled {unmatched_count} unmatched rows with q-value={fill_unmatched[0]}, PEP={fill_unmatched[1]} (--fill-unmatched).")
    if unmatched_count == 0:
        print("  All rows have Percolator q-value and PEP.")
    if unmatched_count > 0 and fill_unmatched is None:
        print("")
        print("  Those unmatched rows have (scan, charge) that do NOT appear in Percolator's output.")
        print(f"  Percolator output had {num_percolator_scan_charge} (scan, charge) pairs; the CSV has {unmatched_count} rows whose (scan, charge) are not in that set.")
        print("  So the .pin file that was fed to Percolator contained fewer spectra than your CSV (e.g. .pin was built with fewer PSMs per spectrum, or from a different run).")
        print("")
        print("  To get q-values for (almost) all rows:")
        print("    1. Rebuild Comet:  make clean && make")
        print("    2. Re-run Comet on the same input file as your CSV, with  output_percolatorfile = 1  in comet.params.")
        print("    3. Re-run this script on the new .pin and the same CSV. This can help with missing rows.")
        print("  Or use  --fill-unmatched 1.0  to assign q-value 1.0 to unmatched rows so downstream does not drop them.")
        print("")
        print("  If you already re-ran Comet and still see many unmatched:")
        print("    Run this script with  --diagnose-unmatched  to see whether the unmatched (scan, charge) appear in Percolator output.")
        print("    If they do NOT appear there, the .pin file may be from a different run or input file than the CSV (check the 'Found .pin file: ...' line above).")
        if unmatched_sample_variants:
            print(f"  Example unmatched SpecIds (compare to Percolator SpecIds above): {unmatched_sample_variants}")
        if diagnose_unmatched and unmatched_details:
            percolator_scan_charge_set = set(scan_charge_lookup.keys())
            print("\n--- Diagnose unmatched (--diagnose-unmatched) ---")
            print(f"  Percolator has {len(percolator_scan_charge_set)} unique (scan, charge) pairs in lookup.")
            print(f"  Sample Percolator SpecIds (from qvalues): {list(qvalues.keys())[:5]}")
            print(f"  CSV base names tried for SpecId: {base_names}")
            print(f"  Unmatched rows ({len(unmatched_details)}):")
            in_lookup_count = sum(1 for _s, _c, _n, in_lu, _p in unmatched_details if in_lu)
            not_in_lookup_count = len(unmatched_details) - in_lookup_count
            if in_lookup_count > 0:
                print(f"    BUG: {in_lookup_count} unmatched rows have (scan, charge) IN Percolator lookup but still did not match (check key normalization).")
            if not_in_lookup_count > 0:
                print(f"    {not_in_lookup_count} unmatched rows have (scan, charge) NOT in Percolator output (those spectra were not in .pin or were filtered by Percolator).")
            for i, (s, c, n, in_lu, pk) in enumerate(unmatched_details[:25]):
                print(f"    [{i+1}] scan={s!r} charge={c!r} num={n!r}  (scan,charge) in Percolator lookup: {in_lu}  peptide_key={pk[:50]}...")
            if len(unmatched_details) > 25:
                print(f"    ... and {len(unmatched_details) - 25} more.")
            unmatched_sc = set((s, c) for s, c, n, in_lu, pk in unmatched_details)
            print(f"  Unmatched (scan, charge) pairs: {len(unmatched_sc)} unique. Sample: {list(unmatched_sc)[:10]}")
            print("--- end diagnose ---\n")
    print(f"Output written to: {output_file}")
    
    return True

def merge_qvalues_into_csv_manual(csv_file, qvalues, output_file, header, lines):
    """Fallback manual CSV merging (for non-standard formats)."""
    # This is the old implementation - kept for compatibility
    # Normalize legacy names to canonical workflow names.
    header = [('perc_qvalue' if c in ('percolator_qvalue', 'q-value') else c) for c in header]
    header = [('perc_PEP' if c in ('percolator_PEP', 'PEP') else c) for c in header]
    # Check if q-value columns already exist
    has_qvalue = 'perc_qvalue' in header
    has_pep = 'perc_PEP' in header
    
    # Add columns if they don't exist
    if not has_qvalue:
        header.append('perc_qvalue')
    if not has_pep:
        header.append('perc_PEP')

    # Move Percolator columns directly after observed_mz when present.
    if 'observed_mz' in header:
        percs = [c for c in ('perc_qvalue', 'perc_PEP') if c in header]
        if percs:
            rest = [c for c in header if c not in percs]
            i = rest.index('observed_mz')
            header = rest[:i + 1] + percs + rest[i + 1:]
    
    # Get base name from CSV filename
    base_name = os.path.splitext(os.path.basename(csv_file))[0]
    base_name = base_name.replace('_PP', '').replace('_with_ms1', '').replace('_with_qvalues', '')
    
    # Parse CSV rows and add q-values
    output_lines = []
    output_lines.append(','.join(header) + '\n')  # Updated header
    
    # Build lookup by (scan, charge) for fallback matching
    scan_charge_lookup = {}
    for specid_key, (qv, p) in qvalues.items():
        parts = specid_key.split('_')
        if len(parts) >= 4:
            try:
                scan_str = parts[-3]
                charge_str = parts[-2]
                key = (scan_str, charge_str)
                if key not in scan_charge_lookup:
                    scan_charge_lookup[key] = (qv, p)
            except (ValueError, IndexError):
                continue
    
    matched_count = 0
    unmatched_count = 0
    
    # Use csv.reader for proper parsing
    import io
    csv_content = ''.join(lines[1:])  # Skip header
    csv_reader = csv.reader(io.StringIO(csv_content))
    
    for row_idx, row_data in enumerate(csv_reader):
        if len(row_data) < len(header) - 2:  # Before adding q-value columns
            # Pad row if needed
            row_data.extend([''] * (len(header) - len(row_data)))
        
        # Create SpecId to match Percolator format
        try:
            scan_idx = header.index('scan') if 'scan' in header else None
            charge_idx = header.index('charge') if 'charge' in header else None
            num_idx = header.index('num') if 'num' in header else None
            
            if scan_idx is None or charge_idx is None:
                # Fallback: use row index
                specid = f"{base_name}_{row_idx+1}_1_1"
            else:
                scan = row_data[scan_idx] if scan_idx < len(row_data) else str(row_idx+1)
                charge = row_data[charge_idx] if charge_idx < len(row_data) else "1"
                num = row_data[num_idx] if num_idx and num_idx < len(row_data) else "1"
                # Percolator format: basename_scan_charge_num
                specid = f"{base_name}_{scan}_{charge}_{num}"
                specid_with_path = f"data/{base_name}_{scan}_{charge}_{num}"
                specid_with_path_pp = f"data/{base_name}_PP_{scan}_{charge}_{num}"
            
            # Try to find matching q-value
            qvalue = None
            pep = None
            
            # First try exact match with CSV's num value
            specid_variants = [
                specid_with_path_pp,  # data/basename_PP_scan_charge_num (most likely)
                specid_with_path,     # data/basename_scan_charge_num
                specid,               # basename_scan_charge_num
                f"{scan}_{charge}_{num}",  # scan_charge_num
            ]
            
            matched = False
            for variant in specid_variants:
                if variant in qvalues:
                    qvalue, pep = qvalues[variant]
                    matched_count += 1
                    matched = True
                    break
            
            # If not matched and CSV has num > 1, try with num=1 (Percolator often only outputs best PSM)
            if not matched and int(num) > 1:
                specid_variants_num1 = [
                    f"data/{base_name}_PP_{scan}_{charge}_1",
                    f"data/{base_name}_{scan}_{charge}_1",
                    f"{base_name}_{scan}_{charge}_1",
                    f"{scan}_{charge}_1",
                ]
                for variant in specid_variants_num1:
                    if variant in qvalues:
                        qvalue, pep = qvalues[variant]
                        matched_count += 1
                        matched = True
                        break
            
            # If still not matched, try matching by (scan, charge) with any num value
            if not matched:
                key = (scan, charge)
                if key in scan_charge_lookup:
                    qvalue, pep = scan_charge_lookup[key]
                    matched_count += 1
                    matched = True
            
            # Last resort: try without num (just scan_charge) - this is less reliable
            if not matched:
                specid_variants_no_num = [
                    f"data/{base_name}_PP_{scan}_{charge}",
                    f"data/{base_name}_{scan}_{charge}",
                    f"{base_name}_{scan}_{charge}",
                    f"{scan}_{charge}",
                ]
                for variant in specid_variants_no_num:
                    if variant in qvalues:
                        qvalue, pep = qvalues[variant]
                        matched_count += 1
                        matched = True
                        break
            
            if not matched:
                unmatched_count += 1
            
            # Add q-value columns
            qvalue_str = f"{qvalue:.6e}" if qvalue is not None else ""
            pep_str = f"{pep:.6e}" if pep is not None else ""
            
            # Ensure row has enough columns
            while len(row_data) < len(header):
                row_data.append("")
            
            # Update or add q-value columns
            if not has_qvalue:
                row_data.append(qvalue_str)
            else:
                if 'perc_qvalue' in header:
                    qvalue_col_idx = header.index('perc_qvalue')
                elif 'percolator_qvalue' in header:
                    qvalue_col_idx = header.index('percolator_qvalue')
                else:
                    qvalue_col_idx = header.index('q-value')
                if qvalue_col_idx < len(row_data):
                    row_data[qvalue_col_idx] = qvalue_str
            
            if not has_pep:
                row_data.append(pep_str)
            else:
                if 'perc_PEP' in header:
                    pep_col_idx = header.index('perc_PEP')
                elif 'percolator_PEP' in header:
                    pep_col_idx = header.index('percolator_PEP')
                else:
                    pep_col_idx = header.index('PEP')
                if pep_col_idx < len(row_data):
                    row_data[pep_col_idx] = pep_str
            
            # Write row (handle quoted fields properly)
            output_row = []
            for val in row_data:
                val_str = str(val) if val else ""
                # Quote if contains comma, quote, or newline
                if ',' in val_str or '"' in val_str or '\n' in val_str:
                    val_str = '"' + val_str.replace('"', '""') + '"'
                output_row.append(val_str)
            
            output_lines.append(','.join(output_row) + '\n')
            
        except Exception as e:
            print(f"Warning: Error processing row {row_idx}: {e}")
            # Write row as-is
            output_lines.append(','.join(row_data) + '\n')
    
    # Write output
    with open(output_file, 'w') as f:
        f.writelines(output_lines)
    
    print(f"Merged q-values: {matched_count} matched, {unmatched_count} unmatched")
    print(f"Output written to: {output_file}")
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description='Add Percolator q-values to Comet CSV output'
    )
    parser.add_argument('--csv', required=True, help='Input CSV file from Comet')
    parser.add_argument('--pin', help='Input .pin file (if not provided, will search for it)')
    parser.add_argument('--comet-pin-dir', help='Directory to search for .pin files')
    parser.add_argument('--percolator-output', help='Percolator output file (if already run)')
    parser.add_argument('--output', help='Output CSV file (default: same dir as input with _with_qvalues suffix; this file is overwritten)')
    parser.add_argument('--skip-percolator', action='store_true',
                       help='Skip running Percolator, only merge existing results')
    parser.add_argument('--diagnose-unmatched', action='store_true',
                       help='Print why each unmatched row failed (scan, charge, whether in Percolator lookup)')
    parser.add_argument('--fill-unmatched', nargs='?', const=1.0, type=float, metavar='Q',
                       help='Give unmatched rows this q-value (and same for PEP) so every row has a value (default: 1.0). E.g. --fill-unmatched or --fill-unmatched 1.0')
    
    args = parser.parse_args()
    
    # Determine output file
    if args.output:
        output_file = args.output
    else:
        # Use input CSV name with _with_qvalues suffix, in same directory as input
        input_dir = os.path.dirname(args.csv) if os.path.dirname(args.csv) else '.'
        input_basename = os.path.basename(args.csv)
        base_name_no_ext = os.path.splitext(input_basename)[0]
        output_file = os.path.join(input_dir, f"{base_name_no_ext}_with_qvalues.csv")
    
    print(f"Output CSV: {os.path.abspath(output_file)} (will overwrite if it exists)")
    
    # Check if Percolator is available
    if not args.skip_percolator and not args.percolator_output:
        if not check_percolator():
            print("\nCannot proceed without Percolator.")
            print("Please install Percolator or use --percolator-output to provide existing results.")
            return 1
    
    # Find or use .pin file
    pin_file = args.pin
    if not pin_file and not args.percolator_output:
        pin_file = find_pin_file(args.csv, args.comet_pin_dir)
        if not pin_file:
            print("\nCannot find .pin file.")
            print("To generate a .pin file, re-run Comet with:")
            print("  output_percolatorfile = 1")
            return 1
    if pin_file and os.path.exists(pin_file):
        _log_pin_file_info(pin_file)
    
    # Run Percolator if needed
    percolator_output = args.percolator_output
    if not percolator_output and not args.skip_percolator:
        percolator_output = run_percolator(pin_file)
        if not percolator_output:
            print("\nFailed to run Percolator.")
            return 1
    
    # Parse Percolator output
    if percolator_output:
        qvalues = parse_percolator_output(percolator_output)
        if not qvalues:
            print("\nFailed to parse Percolator output.")
            return 1
        
        # Warn if pin had far fewer PSMs than CSV (pin was likely generated with num_output_lines=1)
        try:
            with open(args.csv, 'r') as f:
                first = f.readline()
            skip = 1 if first.strip().startswith('CometVersion') or (first.count(',') + 1) < 10 else 0
            with open(args.csv, 'r') as f:
                for _ in range(skip + 1):
                    f.readline()  # skip metadata and header
                csv_data_lines = sum(1 for _ in f)
            if csv_data_lines > 0 and len(qvalues) < 0.6 * csv_data_lines:
                print(f"\n  Note: Percolator output has {len(qvalues)} PSMs but the CSV has {csv_data_lines} rows.")
                print("  If the .pin above has many PSM lines: Percolator keeps only one PSM per (scan, ExpMass).")
                print("  Rebuild Comet (make clean && make) and re-run the search; the updated Comet writes unique ExpMass per PSM so Percolator will output all. Then re-run this script.")
                print("  If the .pin has only ~1000–1200 lines: regenerate it by running Comet with output_percolatorfile = 1 on the same input as the CSV.")
        except Exception:
            pass

        # Infer SpecId base names from Percolator output (e.g. mzML base when CSV was renamed to fasta_comet)
        specid_bases = _infer_specid_base_names(qvalues)
        if specid_bases:
            print(f"  Inferred SpecId base(s) from .pin: {specid_bases[:3]}{'...' if len(specid_bases) > 3 else ''}")

        # Merge into CSV (with inferred SpecId bases for better matching)
        fill_unmatched = None
        if getattr(args, 'fill_unmatched', None) is not None:
            q = float(args.fill_unmatched)
            fill_unmatched = (q, q)  # same value for PEP
        if not merge_qvalues_into_csv(args.csv, qvalues, output_file, diagnose_unmatched=getattr(args, 'diagnose_unmatched', False), fill_unmatched=fill_unmatched, specid_bases=specid_bases):
            return 1

        print(f"\nSuccess! Q-values added to {output_file}")
        return 0
    else:
        print("\nNo Percolator output available.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
