#!/usr/bin/env python3
"""
Convert Comet .txt output to .csv format.
Replaces tabs with commas and handles the header row properly.
"""

import sys
import os

def convert_txt_to_csv(txt_file, csv_file=None):
    """Convert tab-delimited Comet output to CSV format."""
    if csv_file is None:
        base = os.path.splitext(txt_file)[0]
        csv_file = f"{base}.csv"
    
    with open(txt_file, 'r') as f_in, open(csv_file, 'w') as f_out:
        for line_num, line in enumerate(f_in, 1):
            # Remove trailing newline
            line = line.rstrip('\n\r')
            
            # Skip empty lines
            if not line.strip():
                f_out.write('\n')
                continue
            
            # Replace tabs with commas
            csv_line = line.replace('\t', ',')
            f_out.write(csv_line + '\n')
    
    print(f"Converted {txt_file} to {csv_file}")
    return csv_file

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 convert_txt_to_csv.py <input.txt> [output.csv]")
        sys.exit(1)
    
    txt_file = sys.argv[1]
    csv_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(txt_file):
        print(f"Error: File not found: {txt_file}")
        sys.exit(1)
    
    convert_txt_to_csv(txt_file, csv_file)
