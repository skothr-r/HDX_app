"""
Visualization module: combined overhang plots, RT windows, and peak-window (chromatogram) production.

Submodules:
- combined: create_combined_visualization, create_rt_integration_windows_plot,
            create_fragment_contributors_plot, create_peptide_statistic_scatter_plots,
            generate_threshold_scatter_plot
- chromatograms: run_chromatograms (MS1 chromatogram extraction and peak_windows CSV)
- multi_aa: parse_fasta, parse_multi_aa_overhangs, parse_single_aa_overhangs,
            normalize_fragment_pair, get_filtered_peptide_indices
"""

from .multi_aa import (
    parse_fasta,
    parse_multi_aa_overhangs,
    parse_single_aa_overhangs,
    normalize_fragment_pair,
    get_filtered_peptide_indices,
)
from .combined import (
    create_combined_visualization,
    create_rt_integration_windows_plot,
    create_rt_overlay_only_plot,
    create_peptide_statistic_scatter_plots,
    create_fragment_contributors_plot,
    generate_threshold_scatter_plot,
)
from .chromatograms import run_chromatograms

__all__ = [
    'parse_fasta',
    'parse_multi_aa_overhangs',
    'parse_single_aa_overhangs',
    'normalize_fragment_pair',
    'get_filtered_peptide_indices',
    'create_combined_visualization',
    'create_rt_integration_windows_plot',
    'create_rt_overlay_only_plot',
    'create_peptide_statistic_scatter_plots',
    'create_fragment_contributors_plot',
    'generate_threshold_scatter_plot',
    'run_chromatograms',
]
