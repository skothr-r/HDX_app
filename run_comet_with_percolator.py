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
import subprocess
import sys
import os
import re
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
