#!/usr/bin/env python3
"""
Generate Fragment Contributors Plot

Creates a visualization showing all fragments that contribute to single-AA overhangs:
- Top: Full protein sequence
- Below: Grid where each row is a fragment pair, columns are protein positions
- Y-axis labels: peptide sequence, charge, fragment pair
- Overhang position is highlighted
"""

import argparse
import sys
import os

# Add current directory to path so visualization package is found
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from visualization import create_fragment_contributors_plot, parse_fasta


def main():
    parser = argparse.ArgumentParser(
        description='Generate fragment contributors plot showing all fragments contributing to single-AA overhangs'
    )
    parser.add_argument('--csv', required=True, help='Comet CSV output file')
    parser.add_argument('--fasta', required=True, help='FASTA file with protein sequences')
    parser.add_argument('--output', default='fragment_contributors.png',
                       help='Output PNG file (default: fragment_contributors.png)')
    parser.add_argument('--protein', help='Specific protein ID to visualize (optional)')
    parser.add_argument('--c-ions-only', action='store_true',
                       help='Filter to show only c ion fragment pairs')
    parser.add_argument('--z-ions-only', action='store_true',
                       help='Filter to show only z/z+1 ion fragment pairs')
    parser.add_argument('--no-filtering', action='store_true',
                       help='Skip all filtering and include all peptides with overhangs')
    
    args = parser.parse_args()
    
    # Validate ion type flags (mutually exclusive)
    if args.c_ions_only and args.z_ions_only:
        print("Error: --c-ions-only and --z-ions-only are mutually exclusive")
        return
    
    # Determine ion type filter
    ion_type_filter = None
    if args.c_ions_only:
        ion_type_filter = 'c'
    elif args.z_ions_only:
        ion_type_filter = 'z'
    
    # Get protein ID
    sequences = parse_fasta(args.fasta)
    if not sequences:
        print("Error: No sequences found in FASTA file")
        return
    
    if args.protein:
        protein_id = args.protein
    else:
        protein_id = list(sequences.keys())[0]
        print(f"No protein specified, using first protein: {protein_id}")
    
    # Generate fragment contributors plot
    create_fragment_contributors_plot(args.csv, protein_id, args.fasta, args.output,
                                      ion_type_filter=ion_type_filter, no_filtering=args.no_filtering)


if __name__ == '__main__':
    main()
