#!/usr/bin/env python3
"""
Run Comet search and automatically add Percolator q-values to CSV output.

This script:
1. Runs Comet search (generates CSV and .pin files)
2. Runs Percolator on the .pin file
3. Merges q-values into the CSV file

Usage:
    python3 run_comet_with_percolator.py --mzml input.mzML --fasta database.fasta --params comet.params
    python3 run_comet_with_percolator.py --mzml input.mzML --fasta database.fasta --params comet.params --skip-percolator
"""

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Import functions from add_percolator_qvalues.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from add_percolator_qvalues import (
        check_percolator, find_pin_file, run_percolator,
        parse_percolator_output, merge_qvalues_into_csv,
        _infer_specid_base_names
    )
except ImportError:
    print("Error: Could not import functions from add_percolator_qvalues.py")
    print("Make sure add_percolator_qvalues.py is in the same directory")
    sys.exit(1)

def find_comet_executable():
    """Find Comet executable."""
    # Check common locations
    candidates = [
        './comet.exe',
        './comet',
        'comet.exe',
        'comet',
    ]
    
    for candidate in candidates:
        if os.path.exists(candidate) and os.access(candidate, os.X_OK):
            return candidate
    
    # Try in PATH
    try:
        result = subprocess.run(['which', 'comet'], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
    except:
        pass
    
    return None

def prepare_params_for_run(params_file, fasta_file):
    """
    Prepare a params file for Comet: override database_name with the actual FASTA path
    (Comet reads database from params, not command line) and ensure Percolator output.
    Returns path to a temp params file; caller should delete when done.
    """
    if not os.path.exists(params_file):
        print(f"Error: Parameter file not found: {params_file}")
        return None
    if not os.path.exists(fasta_file):
        print(f"Error: FASTA file not found: {fasta_file}")
        return None

    with open(params_file, 'r') as f:
        content = f.read()

    # Override database_name with absolute FASTA path (Comet uses params, not CLI)
    fasta_abs = os.path.abspath(fasta_file)
    if re.search(r'^database_name\s*=', content, re.MULTILINE):
        content = re.sub(
            r'^database_name\s*=\s*.*$',
            f'database_name = {fasta_abs}',
            content,
            count=1,
            flags=re.MULTILINE
        )
    else:
        content = f'database_name = {fasta_abs}\n' + content

    # Ensure output_percolatorfile = 1
    if re.search(r'output_percolatorfile\s*=\s*0', content):
        content = re.sub(
            r'output_percolatorfile\s*=\s*0',
            'output_percolatorfile = 1',
            content
        )
    elif not re.search(r'output_percolatorfile\s*=\s*1', content):
        if re.search(r'output_', content):
            lines = content.split('\n')
            last_output_idx = -1
            for i, line in enumerate(lines):
                if 'output_' in line:
                    last_output_idx = i
            if last_output_idx >= 0:
                lines.insert(last_output_idx + 1, 'output_percolatorfile = 1')
            else:
                lines.append('output_percolatorfile = 1')
            content = '\n'.join(lines)
        else:
            content += '\noutput_percolatorfile = 1\n'

    fd, tmp_path = tempfile.mkstemp(suffix='.params', prefix='comet_')
    try:
        with os.fdopen(fd, 'w') as f:
            f.write(content)
    except Exception:
        os.unlink(tmp_path)
        raise
    return tmp_path

def run_comet(mzml_file, fasta_file, params_file, comet_exe=None):
    """Run Comet search."""
    if comet_exe is None:
        comet_exe = find_comet_executable()
        if not comet_exe:
            print("Error: Comet executable not found.")
            print("Please specify --comet-exe or ensure comet.exe is in the current directory or PATH")
            return None
    
    if not os.path.exists(comet_exe):
        print(f"Error: Comet executable not found: {comet_exe}")
        return None

    # Prepare params with database_name = fasta path (Comet reads DB from params)
    params_to_use = prepare_params_for_run(params_file, fasta_file)
    if not params_to_use:
        return None

    print("=" * 60)
    print("Running Comet search...")
    print("=" * 60)
    print(f"Input file: {mzml_file}")
    print(f"Database: {fasta_file}")
    print(f"Parameters: {params_file}")
    print()

    try:
        cmd = [comet_exe, f'-P{params_to_use}', mzml_file]
        print(f"Command: {' '.join(cmd)}")
        print()

        # Run Comet from mzML dir so output lands there
        mzml_dir = os.path.dirname(os.path.abspath(mzml_file))
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200, cwd=mzml_dir)
        
        if result.returncode != 0:
            print("Error running Comet:")
            if result.stderr:
                print(result.stderr)
            if result.stdout:
                print("Stdout:", result.stdout)
            return None

        print("Comet search completed successfully")

        # Comet writes next to input file; also check cwd
        cwd = os.getcwd()
        mzml_dir = os.path.dirname(os.path.abspath(mzml_file))
        base_name = os.path.splitext(os.path.basename(mzml_file))[0]
        search_dirs = list(dict.fromkeys([cwd, mzml_dir]))

        csv_file = None
        for d in search_dirs:
            for name in (f"{base_name}.csv", f"{base_name}_PP.csv"):
                p = os.path.join(d, name)
                if os.path.exists(p):
                    csv_file = p
                    break
            if csv_file:
                break

        if not csv_file:
            # Fallback: Comet may output .txt only; convert to .csv
            txt_file = None
            for d in search_dirs:
                for name in (f"{base_name}.txt", f"{base_name}_PP.txt"):
                    p = os.path.join(d, name)
                    if os.path.exists(p):
                        txt_file = p
                        break
                if txt_file:
                    break
            if txt_file:
                csv_file = os.path.join(os.path.dirname(txt_file), f"{base_name}.csv")
                with open(txt_file, 'r') as f_in, open(csv_file, 'w') as f_out:
                    for line in f_in:
                        f_out.write(line.rstrip().replace('\t', ',') + '\n')
                print(f"Converted {os.path.basename(txt_file)} to {os.path.basename(csv_file)}")

        if not csv_file or not os.path.exists(csv_file):
            print("Error: Could not find CSV output file")
            print(f"  Searched in: {search_dirs}")
            print(f"  Looking for: {base_name}.csv or {base_name}_PP.csv")
            return None

        pin_file = None
        for d in search_dirs:
            for name in (f"{base_name}.pin", f"{base_name}_PP.pin"):
                p = os.path.join(d, name)
                if os.path.exists(p):
                    pin_file = p
                    break
            if pin_file:
                break

        return {
            'csv': csv_file,
            'pin': pin_file,
            'base_name': base_name
        }

    except subprocess.TimeoutExpired:
        print("Error: Comet search timed out after 2 hours")
        return None
    except Exception as e:
        print(f"Error running Comet: {e}")
        return None
    finally:
        if params_to_use and os.path.exists(params_to_use):
            try:
                os.unlink(params_to_use)
            except OSError:
                pass


def _prune_comet_output_columns(csv_file, drop_columns):
    """
    Remove selected columns from Comet CSV output while preserving row count.
    Supports optional Comet metadata first line.
    Returns number of columns removed.
    """
    if not csv_file or not os.path.exists(csv_file):
        return 0
    try:
        with open(csv_file, 'r', newline='') as f:
            rows = list(csv.reader(f))
        if not rows:
            return 0

        has_meta = bool(rows and rows[0] and str(rows[0][0]).startswith('CometVersion'))
        header_idx = 1 if has_meta else 0
        if header_idx >= len(rows):
            return 0

        header = rows[header_idx]
        drop_set = set(drop_columns)
        drop_idx = [i for i, c in enumerate(header) if c in drop_set]
        if not drop_idx:
            return 0

        keep_idx = [i for i in range(len(header)) if i not in set(drop_idx)]
        new_rows = []
        if has_meta:
            new_rows.append(rows[0])
        new_rows.append([header[i] for i in keep_idx])

        for row in rows[header_idx + 1:]:
            if not row:
                continue
            new_rows.append([row[i] if i < len(row) else '' for i in keep_idx])

        with open(csv_file, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerows(new_rows)

        removed = len(drop_idx)
        print(f"Pruned {removed} column(s) from Comet output: {', '.join(drop_columns)}")
        return removed
    except Exception as e:
        print(f"Warning: Could not prune Comet output columns from {csv_file}: {e}")
        return 0


def _rename_comet_output_columns(csv_file, rename_map):
    """Rename selected columns in Comet CSV output while preserving row count."""
    if not csv_file or not os.path.exists(csv_file):
        return 0
    try:
        with open(csv_file, 'r', newline='') as f:
            rows = list(csv.reader(f))
        if not rows:
            return 0
        has_meta = bool(rows and rows[0] and str(rows[0][0]).startswith('CometVersion'))
        header_idx = 1 if has_meta else 0
        if header_idx >= len(rows):
            return 0
        header = rows[header_idx]
        renamed = 0
        for i, c in enumerate(header):
            if c in rename_map:
                header[i] = rename_map[c]
                renamed += 1
        if renamed == 0:
            return 0
        rows[header_idx] = header
        with open(csv_file, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerows(rows)
        print(f"Renamed {renamed} Comet output column(s): {rename_map}")
        return renamed
    except Exception as e:
        print(f"Warning: Could not rename Comet output columns in {csv_file}: {e}")
        return 0


def _reorder_comet_output_columns(csv_file):
    """Move core score columns to immediately follow observed_mz."""
    if not csv_file or not os.path.exists(csv_file):
        return 0
    try:
        with open(csv_file, 'r', newline='') as f:
            rows = list(csv.reader(f))
        if not rows:
            return 0

        has_meta = bool(rows and rows[0] and str(rows[0][0]).startswith('CometVersion'))
        header_idx = 1 if has_meta else 0
        if header_idx >= len(rows):
            return 0

        header = rows[header_idx]
        if 'observed_mz' not in header:
            return 0

        metric_cols = ['e-value', 'xcorr', 'delta_cn', 'sp_score']
        metrics_present = [c for c in metric_cols if c in header]
        if not metrics_present:
            return 0

        header_without_metrics = [c for c in header if c not in metrics_present]
        anchor_idx = header_without_metrics.index('observed_mz')
        new_header = (
            header_without_metrics[:anchor_idx + 1]
            + metrics_present
            + header_without_metrics[anchor_idx + 1:]
        )
        if new_header == header:
            return 0

        idx_by_name = {c: i for i, c in enumerate(header)}
        new_idx = [idx_by_name[c] for c in new_header]

        new_rows = []
        if has_meta:
            new_rows.append(rows[0])
        new_rows.append(new_header)
        for row in rows[header_idx + 1:]:
            if not row:
                continue
            new_rows.append([row[i] if i < len(row) else '' for i in new_idx])

        with open(csv_file, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerows(new_rows)

        print("Reordered Comet columns: observed_mz, e-value, xcorr, delta_cn, sp_score")
        return len(metrics_present)
    except Exception as e:
        print(f"Warning: Could not reorder Comet output columns in {csv_file}: {e}")
        return 0


def _ensure_theoretical_mz_and_ms1_error(csv_file):
    """
    Ensure Comet output has:
      - theoretical_mz (derived from calc_neutral_mass + charge when available)
      - MS1_mz_error_ppm = signed ppm error of observed vs theoretical
    And place them immediately after observed_mz in this order:
      observed_mz, theoretical_mz, MS1_mz_error_ppm
    """
    if not csv_file or not os.path.exists(csv_file):
        return 0
    try:
        with open(csv_file, 'r', newline='') as f:
            rows = list(csv.reader(f))
        if not rows:
            return 0

        has_meta = bool(rows and rows[0] and str(rows[0][0]).startswith('CometVersion'))
        header_idx = 1 if has_meta else 0
        if header_idx >= len(rows):
            return 0

        old_header = rows[header_idx]
        header = old_header[:]
        changed = 0

        if 'observed_mz' not in header:
            return 0

        if 'theoretical_mz' not in header:
            header.append('theoretical_mz')
            changed += 1
        # Backward compatibility: old name -> canonical name
        if 'MS1_mz_error' in header and 'MS1_mz_error_ppm' not in header:
            header[header.index('MS1_mz_error')] = 'MS1_mz_error_ppm'
            changed += 1
        if 'MS1_mz_error_ppm' not in header:
            header.append('MS1_mz_error_ppm')
            changed += 1

        block_cols = [c for c in ['theoretical_mz', 'MS1_mz_error_ppm'] if c in header]
        header_wo_block = [c for c in header if c not in block_cols]
        obs_idx = header_wo_block.index('observed_mz')
        new_header = header_wo_block[:obs_idx + 1] + block_cols + header_wo_block[obs_idx + 1:]
        if new_header != header:
            header = new_header
            changed += 1

        if changed == 0:
            return 0

        old_idx = {c: i for i, c in enumerate(old_header)}
        proton_mass = 1.007276466812

        def _to_float(v):
            try:
                s = str(v).strip()
                if not s:
                    return None
                return float(s)
            except (TypeError, ValueError):
                return None

        new_rows = [rows[0]] if has_meta else []
        new_rows.append(header)

        for row in rows[header_idx + 1:]:
            if not row:
                continue
            out_row = []
            # compute once per row
            obs_val = _to_float(row[old_idx['observed_mz']]) if 'observed_mz' in old_idx and old_idx['observed_mz'] < len(row) else None

            # theoretical_mz priority: existing value -> calc_neutral_mass/charge -> empty
            theo_val = None
            if 'theoretical_mz' in old_idx and old_idx['theoretical_mz'] < len(row):
                theo_val = _to_float(row[old_idx['theoretical_mz']])
            if theo_val is None:
                cnm = _to_float(row[old_idx['calc_neutral_mass']]) if 'calc_neutral_mass' in old_idx and old_idx['calc_neutral_mass'] < len(row) else None
                z = _to_float(row[old_idx['charge']]) if 'charge' in old_idx and old_idx['charge'] < len(row) else None
                if cnm is not None and z is not None and z > 0:
                    theo_val = (cnm + (z * proton_mass)) / z

            # ppm error (signed): (observed - theoretical) / theoretical * 1e6
            ppm_val = None
            if obs_val is not None and theo_val is not None and theo_val != 0:
                ppm_val = ((obs_val - theo_val) / theo_val) * 1e6

            for col in header:
                if col == 'theoretical_mz':
                    if theo_val is None:
                        out_row.append('')
                    else:
                        out_row.append(f"{theo_val:.6f}")
                elif col == 'MS1_mz_error_ppm':
                    if ppm_val is None:
                        out_row.append('')
                    else:
                        out_row.append(f"{ppm_val:.3f}")
                else:
                    src_i = old_idx.get(col)
                    out_row.append(row[src_i] if src_i is not None and src_i < len(row) else '')
            new_rows.append(out_row)

        with open(csv_file, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerows(new_rows)
        print("Ensured mz columns: observed_mz, theoretical_mz, MS1_mz_error_ppm")
        return 1
    except Exception as e:
        print(f"Warning: Could not ensure theoretical_mz/MS1_mz_error_ppm in {csv_file}: {e}")
        return 0


def _ensure_ms2_rt_columns(csv_file):
    """
    Normalize MS2 RT columns:
      - rename MS2_retention_time_sec -> MS2_RT_sec
      - ensure MS2_RT_minutes exists
      - place MS2_RT_minutes immediately after MS2_RT_sec
    """
    if not csv_file or not os.path.exists(csv_file):
        return 0
    try:
        with open(csv_file, 'r', newline='') as f:
            rows = list(csv.reader(f))
        if not rows:
            return 0

        has_meta = bool(rows and rows[0] and str(rows[0][0]).startswith('CometVersion'))
        header_idx = 1 if has_meta else 0
        if header_idx >= len(rows):
            return 0

        header = rows[header_idx][:]
        changed = 0

        # Normalize seconds column name
        if 'MS2_retention_time_sec' in header and 'MS2_RT_sec' not in header:
            sec_idx = header.index('MS2_retention_time_sec')
            header[sec_idx] = 'MS2_RT_sec'
            changed += 1
        elif 'MS2_RT_sec' in header:
            sec_idx = header.index('MS2_RT_sec')
        else:
            sec_idx = None

        # Add minutes column if missing and seconds is available
        if sec_idx is not None and 'MS2_RT_minutes' not in header:
            header.insert(sec_idx + 1, 'MS2_RT_minutes')
            changed += 1

        # Re-position minutes column immediately after seconds
        if sec_idx is not None and 'MS2_RT_minutes' in header:
            mins_idx = header.index('MS2_RT_minutes')
            target_idx = header.index('MS2_RT_sec') + 1
            if mins_idx != target_idx:
                col = header.pop(mins_idx)
                header.insert(target_idx, col)
                changed += 1

        if changed == 0:
            return 0

        old_header = rows[header_idx]
        old_idx = {c: i for i, c in enumerate(old_header)}
        new_rows = [rows[0]] if has_meta else []
        new_rows.append(header)

        for row in rows[header_idx + 1:]:
            if not row:
                continue
            out_row = []
            for col in header:
                if col == 'MS2_RT_minutes':
                    if 'MS2_RT_minutes' in old_idx and old_idx['MS2_RT_minutes'] < len(row):
                        out_row.append(row[old_idx['MS2_RT_minutes']])
                    else:
                        # Derive from whichever seconds column exists in source
                        sec_val = ''
                        if 'MS2_RT_sec' in old_idx and old_idx['MS2_RT_sec'] < len(row):
                            sec_val = row[old_idx['MS2_RT_sec']]
                        elif 'MS2_retention_time_sec' in old_idx and old_idx['MS2_retention_time_sec'] < len(row):
                            sec_val = row[old_idx['MS2_retention_time_sec']]
                        try:
                            out_row.append(f"{float(sec_val) / 60.0:.6f}" if str(sec_val).strip() else '')
                        except (TypeError, ValueError):
                            out_row.append('')
                elif col == 'MS2_RT_sec' and 'MS2_RT_sec' not in old_idx and 'MS2_retention_time_sec' in old_idx:
                    src_i = old_idx['MS2_retention_time_sec']
                    out_row.append(row[src_i] if src_i < len(row) else '')
                else:
                    src_i = old_idx.get(col)
                    out_row.append(row[src_i] if src_i is not None and src_i < len(row) else '')
            new_rows.append(out_row)

        with open(csv_file, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerows(new_rows)
        print("Normalized MS2 RT columns: MS2_RT_sec + MS2_RT_minutes")
        return 1
    except Exception as e:
        print(f"Warning: Could not normalize MS2 RT columns in {csv_file}: {e}")
        return 0


def _sort_comet_rows(csv_file):
    """
    Sort rows by sequence start, peptide length (short->long), then charge.
    Used for Comet/Percolator alignment across outputs.
    """
    if not csv_file or not os.path.exists(csv_file):
        return 0
    try:
        with open(csv_file, 'r', newline='') as f:
            rows = list(csv.reader(f))
        if not rows:
            return 0
        has_meta = bool(rows and rows[0] and str(rows[0][0]).startswith('CometVersion'))
        header_idx = 1 if has_meta else 0
        if header_idx >= len(rows):
            return 0

        header = rows[header_idx]
        data_rows = rows[header_idx + 1:]
        if not data_rows:
            return 0
        idx = {c: i for i, c in enumerate(header)}
        pos_i = idx.get('protein_position', idx.get('sequence_positions'))
        seq_i = idx.get('peptide_sequence', idx.get('plain_peptide'))
        ch_i = idx.get('charge')

        def _parse_start_and_len(pos_val, seq_val):
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
            seq = str(seq_val).strip() if seq_val is not None else ''
            return 999999, (len(seq) if seq else 999999)

        keyed = []
        for i, row in enumerate(data_rows):
            pos_val = row[pos_i] if pos_i is not None and pos_i < len(row) else ''
            seq_val = row[seq_i] if seq_i is not None and seq_i < len(row) else ''
            start, ln = _parse_start_and_len(pos_val, seq_val)
            try:
                ch = float(row[ch_i]) if ch_i is not None and ch_i < len(row) and str(row[ch_i]).strip() else 999999.0
            except Exception:
                ch = 999999.0
            keyed.append((start, ln, ch, i, row))
        keyed.sort(key=lambda t: (t[0], t[1], t[2], t[3]))
        sorted_rows = [t[4] for t in keyed]

        out_rows = [rows[0]] if has_meta else []
        out_rows.append(header)
        out_rows.extend(sorted_rows)
        with open(csv_file, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerows(out_rows)
        print("Sorted Comet rows: sequence start, length, charge")
        return len(sorted_rows)
    except Exception as e:
        print(f"Warning: Could not sort Comet rows in {csv_file}: {e}")
        return 0


def _move_tail_columns_for_display(csv_file):
    """Keep MS1_RT_minutes/intensity early; move MS1_RT_sec+MS2_RT* to very end."""
    if not csv_file or not os.path.exists(csv_file):
        return 0
    try:
        with open(csv_file, 'r', newline='') as f:
            rows = list(csv.reader(f))
        if not rows:
            return 0
        has_meta = bool(rows and rows[0] and str(rows[0][0]).startswith('CometVersion'))
        header_idx = 1 if has_meta else 0
        if header_idx >= len(rows):
            return 0
        header = rows[header_idx]
        front_cols = [c for c in ['protein_position', 'peptide_sequence', 'charge', 'observed_mz', 'theoretical_mz', 'MS1_mz_error_ppm', 'MS1_RT_minutes'] if c in header]
        ms1_cols = [c for c in ['MS1_RT_intensity'] if c in header]
        frag_tail = [c for c in ['comet_matched_frags', 'comet_matched_frags_mz', 'comet_matched_frags_intensities', 'comet_matched_frags_quality_scores'] if c in header]
        ms2_tail = [c for c in ['MS1_RT_sec', 'MS2_RT_sec', 'MS2_RT_minutes'] if c in header]
        if not (front_cols or ms1_cols or frag_tail or ms2_tail):
            return 0
        head_cols = [c for c in header if c not in front_cols and c not in ms1_cols and c not in frag_tail and c not in ms2_tail]
        new_head = front_cols + ms1_cols + head_cols
        new_header = new_head + frag_tail + ms2_tail
        if new_header == header:
            return 0
        idx_by_name = {c: i for i, c in enumerate(header)}
        new_idx = [idx_by_name[c] for c in new_header]
        out_rows = [rows[0]] if has_meta else []
        out_rows.append(new_header)
        for row in rows[header_idx + 1:]:
            if not row:
                continue
            out_rows.append([row[i] if i < len(row) else '' for i in new_idx])
        with open(csv_file, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerows(out_rows)
        print("Moved MS1_RT_sec and MS2_RT* to end; kept minutes early")
        return 1
    except Exception as e:
        print(f"Warning: Could not reorder tail columns in {csv_file}: {e}")
        return 0

def main():
    parser = argparse.ArgumentParser(
        description='Run Comet search and automatically add Percolator q-values to CSV output',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script automates the complete workflow:
1. Runs Comet search (generates CSV and .pin files)
2. Runs Percolator on the .pin file
3. Merges q-values into the CSV file

The output CSV will have q-values automatically added, ready for visualization.

Example:
    python3 run_comet_with_percolator.py \\
        --mzml data/sample.mzML \\
        --fasta data/database.fasta \\
        --params comet.params.new
        """
    )
    
    parser.add_argument('--mzml', required=True, help='Input mzML/mzXML file')
    parser.add_argument('--fasta', required=True, help='FASTA database file')
    parser.add_argument('--params', required=True, help='Comet parameter file')
    parser.add_argument('--comet-exe', help='Path to Comet executable (default: auto-detect)')
    parser.add_argument('--output-dir', help='Directory for output [fasta]_comet.csv and .pin')
    parser.add_argument('--output-base', help='Base name for output (default: FASTA basename)')
    parser.add_argument('--output-csv', help='Output CSV file name (default: auto-detect from input)')
    parser.add_argument('--skip-percolator', action='store_true',
                       help='Skip Percolator step (only run Comet)')
    parser.add_argument('--skip-comet', action='store_true',
                       help='Skip Comet step (assume CSV and .pin already exist)')
    parser.add_argument('--pin-dir', help='Directory to search for .pin files')
    
    args = parser.parse_args()
    
    # Step 1: Run Comet (unless skipped)
    comet_output = None
    if not args.skip_comet:
        comet_output = run_comet(args.mzml, args.fasta, args.params, args.comet_exe)
        if not comet_output:
            print("\nFailed to run Comet search.")
            return 1
    else:
        # Find existing CSV and .pin files
        base_name = os.path.splitext(os.path.basename(args.mzml))[0]
        csv_file = args.output_csv or f"{base_name}.csv"
        pin_file = f"{base_name}.pin"
        
        if not os.path.exists(csv_file):
            csv_file_pp = f"{base_name}_PP.csv"
            if os.path.exists(csv_file_pp):
                csv_file = csv_file_pp
            else:
                print(f"Error: CSV file not found: {csv_file}")
                return 1
        
        comet_output = {
            'csv': csv_file,
            'pin': pin_file if os.path.exists(pin_file) else None,
            'base_name': base_name
        }
    
    csv_file = comet_output['csv']
    pin_file = comet_output.get('pin')

    # Copy to [fasta]_comet.csv and [fasta]_comet.pin if --output-dir given
    if args.output_dir:
        out_base = args.output_base or os.path.splitext(os.path.basename(args.fasta))[0]
        dst_csv = os.path.join(args.output_dir, f"{out_base}_comet.csv")
        dst_pin = os.path.join(args.output_dir, f"{out_base}_comet.pin")
        os.makedirs(args.output_dir, exist_ok=True)
        if os.path.exists(csv_file):
            shutil.copy2(csv_file, dst_csv)
            csv_file = dst_csv
        if pin_file and os.path.exists(pin_file):
            shutil.copy2(pin_file, dst_pin)
            pin_file = dst_pin

    # Remove legacy overhang columns from Comet output; downstream relies on
    # significance-derived overhang mappings instead.
    _prune_comet_output_columns(
        csv_file,
        [
            'single_aa_overhang_fragment_pairs',
            'single_aa_overhangs_protein_positions',
            # Audited as unused in current workflow
            'exp_neutral_mass',
            'ions_total',
            'modified_peptide',
            'protein_count',
            'sp_rank',
        ]
    )
    _rename_comet_output_columns(
        csv_file,
        {
            'matched fragment ion intensities': 'comet_matched_frags_intensities',
            'matched fragment ion quality scores': 'comet_matched_frags_quality_scores',
            'mz': 'observed_mz',
            'sequence_positions': 'protein_position',
            'plain_peptide': 'peptide_sequence',
        }
    )
    _ensure_ms2_rt_columns(csv_file)
    _reorder_comet_output_columns(csv_file)
    _ensure_theoretical_mz_and_ms1_error(csv_file)
    _sort_comet_rows(csv_file)
    _move_tail_columns_for_display(csv_file)
    
    # Step 2: Run Percolator (unless skipped)
    percolator_output = None
    if not args.skip_percolator:
        if not pin_file:
            # Try to find .pin file
            pin_file = find_pin_file(csv_file, args.pin_dir)
            if not pin_file:
                print("\nWarning: .pin file not found. Skipping Percolator step.")
                print("To generate q-values, re-run Comet with output_percolatorfile = 1")
                return 0
        
        # Check if Percolator is available
        if not check_percolator():
            print("\nWarning: Percolator not found. Skipping Percolator step.")
            print("Install Percolator to add q-values: conda install -c bioconda percolator")
            return 0
        
        print()
        print("=" * 60)
        print("Running Percolator...")
        print("=" * 60)
        
        percolator_output = run_percolator(pin_file)
        if not percolator_output:
            print("\nWarning: Failed to run Percolator. CSV file created without q-values.")
            return 0
    
    # Step 3: Merge q-values into CSV
    if percolator_output:
        print()
        print("=" * 60)
        print("Merging q-values into CSV...")
        print("=" * 60)
        
        # Determine output file
        if args.output_csv:
            output_csv = args.output_csv
        else:
            base_name = os.path.splitext(csv_file)[0]
            # Remove _with_qvalues if already present
            if base_name.endswith('_with_qvalues'):
                base_name = base_name[:-12]
            output_csv = f"{base_name}_with_qvalues.csv"
        
        # Parse Percolator output
        qvalues = parse_percolator_output(percolator_output)
        if not qvalues:
            print("\nWarning: Failed to parse Percolator output. CSV file created without q-values.")
            return 0
        
        # Merge into CSV (use fill_unmatched=1.0 so every row has a value; downstream won't drop unmatched rows)
        specid_bases = _infer_specid_base_names(qvalues)
        if merge_qvalues_into_csv(csv_file, qvalues, output_csv, specid_bases=specid_bases, fill_unmatched=(1.0, 1.0)):
            print()
            print("=" * 60)
            print("Success!")
            print("=" * 60)
            print(f"CSV with q-values: {output_csv}")
            print()
            print("You can now use this file with the visualization script:")
            print(f"  python3 combined_overhang_visualization.py --csv {output_csv} --fasta {args.fasta} --two-pass")
            return 0
        else:
            print("\nWarning: Failed to merge q-values. CSV file created without q-values.")
            return 0
    else:
        print()
        print("=" * 60)
        print("Comet search completed")
        print("=" * 60)
        print(f"CSV file: {csv_file}")
        if pin_file:
            print(f".pin file: {pin_file}")
            print()
            print("To add q-values, run:")
            print(f"  python3 add_percolator_qvalues.py --csv {csv_file} --pin {pin_file}")
        return 0

if __name__ == '__main__':
    sys.exit(main())
