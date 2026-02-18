#!/usr/bin/env python3
"""
Generate Combined Overhang Visualization

Creates the main combined visualization with:
- Coverage heatmap
- Q-value vs Signal Intensity scatter plot
- Single-AA histogram
- RT grid

Also automatically generates the per-residue fragment pair grid.
"""

import argparse
import sys
import os

# Add current directory to path so visualization package is found
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from visualization import create_combined_visualization, parse_fasta


def main():
    parser = argparse.ArgumentParser(
        description='Generate combined overhang visualization with heatmap, scatter plot, histogram, and RT grid'
    )
    parser.add_argument('--csv', required=True, help='Comet CSV output file')
    parser.add_argument('--fasta', required=True, help='FASTA file with protein sequences')
    parser.add_argument('--mzml', help='mzML file (optional, used to read correct fill time values if CSV values are incorrect)')
    parser.add_argument('--output', default='combined_overhang_visualization.png',
                       help='Output PNG file (default: combined_overhang_visualization.png)')
    parser.add_argument('--protein', help='Specific protein ID to visualize (optional)')
    parser.add_argument('--c-ions-only', action='store_true',
                       help='Filter to show only c ion fragment pairs')
    parser.add_argument('--z-ions-only', action='store_true',
                       help='Filter to show only z/z+1 ion fragment pairs')
    parser.add_argument('--no-filtering', action='store_true',
                       help='Skip all filtering and include all peptides with overhangs')
    parser.add_argument('--two-pass', action='store_true',
                       help='Use two-pass selection algorithm (c-anchored + z-supported)')
    parser.add_argument('--no-progressive', action='store_true',
                       help='Disable progressive selection (gap-filling with 2-AA and 3-AA overhangs)')
    parser.add_argument('--no-two-pass', action='store_true',
                       help='Disable two-pass selection algorithm')
    
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
    
    # Generate main combined plot (this also generates the fragment pair grid automatically)
    create_combined_visualization(args.csv, protein_id, args.fasta, args.output, 
                                 ion_type_filter=ion_type_filter, no_filtering=args.no_filtering,
                                 use_two_pass=args.two_pass, no_progressive=args.no_progressive,
                                 no_two_pass=args.no_two_pass, mzml_file=args.mzml)
    
    # Automatically generate c-ions-only and z-ions-only plots (unless one is already being generated)
    if ion_type_filter is None:
        base_output = args.output
        if base_output.endswith('.png'):
            base_output = base_output[:-4]
        
        c_output = base_output + '_c_ions.png'
        z_output = base_output + '_z_ions.png'
        
        # Generate c-ions-only plot
        print(f"\n{'='*60}")
        print(f"Generating c-ions-only plot: {c_output}")
        print(f"{'='*60}")
        create_combined_visualization(args.csv, protein_id, args.fasta, c_output, 
                                     ion_type_filter='c', no_filtering=args.no_filtering,
                                     use_two_pass=args.two_pass, no_progressive=args.no_progressive,
                                     no_two_pass=args.no_two_pass, mzml_file=args.mzml)
        
        # Generate z-ions-only plot
        print(f"\n{'='*60}")
        print(f"Generating z-ions-only plot: {z_output}")
        print(f"{'='*60}")
        create_combined_visualization(args.csv, protein_id, args.fasta, z_output, 
                                     ion_type_filter='z', no_filtering=args.no_filtering,
                                     use_two_pass=args.two_pass, no_progressive=args.no_progressive,
                                     no_two_pass=args.no_two_pass, mzml_file=args.mzml)


if __name__ == '__main__':
    main()
