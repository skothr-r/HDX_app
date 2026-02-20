#!/usr/bin/env python3
"""
Extract MS1 chromatograms for each unique peptide and determine peak windows.

This script:
1. Reads the Comet CSV output with MS1 retention times
2. Groups by unique peptide
3. Extracts MS1 chromatogram data from mzML file for each peptide
4. Determines peak boundaries (min-max RT window with non-zero intensities)
5. Creates visualization with subplots showing chromatograms and peak windows

Usage:
    python extract_ms1_chromatograms.py <comet_csv> <raw_file> [output_png] [--test]
    python extract_ms1_chromatograms.py <comet_csv> <raw_file> [output_png] --filter-csv <comet_frags_filtered_by_qvalue_pepscore_mods_minaa.csv>

    Use the same full CSV as the visualization script (comet_csv). Optionally pass --filter-csv
    (e.g. comet_frags_filtered_by_qvalue_pepscore_mods_minaa.csv from the visualization) to process only those peptides; data
    is still read from comet_csv so all columns are preserved.

Requirements:
    - pyopenms (pip install pyopenms)
    - pandas (pip install pandas)
    - matplotlib (pip install matplotlib)
    - numpy (pip install numpy)
"""

import sys
import os
import argparse
import warnings
import textwrap

# Ensure OpenMS/pyopenms shared libraries can be found (macOS Homebrew, conda, etc.)
def _prepend_library_paths():
    paths = []
    if sys.platform == 'darwin':
        # macOS: Homebrew (Intel and Apple Silicon) and conda env lib
        for d in ('/opt/homebrew/lib', '/usr/local/lib', os.path.join(sys.prefix, 'lib')):
            if d and os.path.isdir(d):
                paths.append(d)
        key = 'DYLD_LIBRARY_PATH'
    else:
        # Linux: conda env and common locations
        for d in (os.path.join(sys.prefix, 'lib'), '/usr/local/lib'):
            if d and os.path.isdir(d):
                paths.append(d)
        key = 'LD_LIBRARY_PATH'
    if paths:
        existing = os.environ.get(key, '')
        new = os.pathsep.join(paths)
        os.environ[key] = new + (os.pathsep + existing if existing else '')
_prepend_library_paths()

# Suppress tight_layout warnings for complex subplot layouts
warnings.filterwarnings('ignore', category=UserWarning, message='.*tight_layout.*')
warnings.filterwarnings('ignore', category=UserWarning, message='.*Tight layout.*')
warnings.filterwarnings('ignore', category=UserWarning, message='.*not compatible with tight_layout.*')
import pandas as pd
import numpy as np
# NumPy 2.0 removed np.trapz; use np.trapezoid when available
_np_trapz = getattr(np, 'trapezoid', None) or getattr(np, 'trapz')
import matplotlib
import matplotlib.pyplot as plt
# Serif font: DejaVu Serif first (Linux/Streamlit Cloud), Times New Roman on macOS/Windows
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['DejaVu Serif', 'Times New Roman', 'Liberation Serif', 'serif']
matplotlib.rcParams['font.size'] = 10
# Allow many figures open at once (combined figures + one per peptide); we close each peptide fig after save (0 = disable warning)
matplotlib.rcParams['figure.max_open_warning'] = 0
try:
    from pyopenms import (
        MSExperiment, MzMLFile, FeatureMap, FeatureFinderAlgorithmPicked
    )
except (ModuleNotFoundError, ImportError) as e:
    print("Error: Could not import pyopenms.", file=sys.stderr)
    print("", file=sys.stderr)
    print("If pyopenms is installed in a conda environment, activate it first:", file=sys.stderr)
    print("  conda activate <your_env>", file=sys.stderr)
    print("", file=sys.stderr)
    print("Or ensure the correct Python is on your PATH (which python).", file=sys.stderr)
    if sys.platform == 'darwin':
        print("On macOS, you may need to set DYLD_LIBRARY_PATH so OpenMS libs are found:", file=sys.stderr)
        print("  export DYLD_LIBRARY_PATH=$(brew --prefix openms)/lib:$DYLD_LIBRARY_PATH", file=sys.stderr)
        print("  (or add your conda env lib: export DYLD_LIBRARY_PATH=$CONDA_PREFIX/lib:$DYLD_LIBRARY_PATH)", file=sys.stderr)
    print("", file=sys.stderr)
    raise
import bisect
import re
from collections import defaultdict
from matplotlib.patches import Rectangle
from scipy import signal

# Collection window background: lighter gray used consistently on main chromatogram, window bar, and overlay
COLLECTION_WINDOW_COLOR = '#A8A8A8'
COLLECTION_WINDOW_ALPHA = 0.3   # Transparent over black background
INTEGRATION_WINDOW_ALPHA = 0.35  # Transparent green over black
# Fragment row colors in sequence panel (match unique-sequences combined plot: coverage/overhangs/opacity)
OVERHANG_RED = '#7A1F2B'       # deep blue-red (single-AA overhang on main row)
OVERHANG_YELLOW = '#C9C68F'    # less saturated yellow (single-AA overhang in c/z fragment rows)
BLUE_BLACK = (0.08, 0.10, 0.18)
BLUE_BLACK_ALPHA_ONE_LAYER = 0.5   # alpha for one fragment layer (fragment rows)
# Legacy names for backward compat; coverage now uses semi-opaque blue-black
C_FRAGMENT_COVERAGE_COLOR = (*BLUE_BLACK, BLUE_BLACK_ALPHA_ONE_LAYER)
Z_FRAGMENT_COVERAGE_COLOR = (*BLUE_BLACK, BLUE_BLACK_ALPHA_ONE_LAYER)
# Peak drift window = collection window ± this buffer (seconds). Match regenerate_rt_windows_from_csv.PEAK_DRIFT_BUFFER_SEC.
PEAK_DRIFT_BUFFER_SEC = 30.0
# Collection window extends beyond integration window by this much (seconds each side) so grey band is visibly wider.
COLLECTION_EXTENSION_SEC = 30.0
# Peak boundary detection (integration window): baseline threshold = max(10th pct, apex_frac, noise_mult × noise)
PEAK_BOUNDARY_APEX_FRAC = 0.01
PEAK_BOUNDARY_NOISE_MULTIPLIER = 3.0
# rel_height for peak_widths: higher = wider boundaries (measure further down the peak)
# 0.5 = half prominence (FWHM-like); 1.0 = full prominence (widest, at peak base)
REL_HEIGHT_INTEGRATION = 0.5
REL_HEIGHT_COLLECTION = 0.8
# Shape correlation minimum: reject peptides whose best peak has shape_corr < this threshold
SHAPE_CORR_MIN = 0.8


def _total_area_colormap_for_overlay():
    """Colormap for overlay/tracks: lavender -> purple -> dark red -> Spectral (matches RT windows)."""
    from matplotlib.colors import ListedColormap
    spectral = plt.get_cmap('Spectral')
    lavender = np.array([0.90, 0.90, 0.98, 1.0])
    purple = np.array([0.58, 0.44, 0.86, 1.0])
    dark_red = np.array([0.55, 0.0, 0.0, 1.0])
    n_low, n_high = 64, 192
    colors = []
    for i in range(n_low):
        t = i / max(n_low - 1, 1)
        if t < 0.33:
            c = lavender + (purple - lavender) * (t / 0.33)
        elif t < 0.66:
            c = purple + (dark_red - purple) * ((t - 0.33) / 0.33)
        else:
            c = dark_red + (np.array(spectral(0)[:4]) - dark_red) * ((t - 0.66) / 0.34)
        colors.append(c)
    SPECTRAL_V_MAX = 0.85
    for i in range(n_high):
        v = SPECTRAL_V_MAX * (i / max(n_high - 1, 1))
        colors.append(spectral(v))
    return ListedColormap(colors, name='total_area')


# Stage 2 significant fragments: 5 ppm (from Step 5 accuracy filter) + optional % max MS2 intensity. Step 6 (extract) uses 5 ppm only (skip_significance_intensity_filter=True); Step 7 (refine_significant_frags) applies 0.5% max MS2. Used for placed_ion_labels, red squares, and output significant_fragment_* columns.
SIGNIFICANT_FRAGMENT_PPM_MS2 = 5.0
SIGNIFICANT_FRAGMENT_MIN_INTENSITY_FRAC = 0.01   # 1% of max MS2 intensity (Step 7 only; Step 6 skips this)

# Global variable to cache MSExperiment and FeatureMap (load once, reuse many times)
_cached_experiment = None
_cached_raw_file = None
_cached_feature_map = None
# MS2 spectra indexed by RT for fast window lookup (avoids O(n) scan per row)
_cached_ms2_rt_list = None  # list of (rt, spec) sorted by rt
# MS1 spectra indexed by RT for fast chromatogram extraction
_cached_ms1_rt_list = None  # list of (rt, spec) sorted by rt


def clear_chromatograms_cache():
    """Clear mzML/MSExperiment cache. Call before run/generate to force fresh load."""
    global _cached_experiment, _cached_raw_file, _cached_feature_map, _cached_ms1_rt_list, _cached_ms2_rt_list
    _cached_experiment = None
    _cached_raw_file = None
    _cached_feature_map = None
    _cached_ms1_rt_list = None
    _cached_ms2_rt_list = None


def extract_ms1_features_openms(raw_file):
    """
    Extract MS1 chromatographic features using OpenMS FeatureFinder.
    For proteomics, attempts FeatureFinderAlgorithmPicked but falls back to
    chromatogram-based peak detection (which works well for this use case).
    
    Note: FeatureFinderAlgorithmPicked typically requires mass trace extraction
    first, which may not be straightforward in pyopenms. We use chromatogram-based
    peak detection instead, which directly extracts XIC and finds peak apexes.
    
    Args:
        raw_file: Path to mzML file
    
    Returns:
        FeatureMap (currently empty, as we use chromatogram-based detection)
    """
    global _cached_experiment, _cached_raw_file, _cached_feature_map, _cached_ms1_rt_list, _cached_ms2_rt_list
    # Load experiment once and cache it
    if _cached_experiment is None or _cached_raw_file != raw_file:
        print(f"[DEBUG] Loading mzML file for chromatogram extraction...")
        _cached_experiment = MSExperiment()
        
        if raw_file.endswith('.mzML') or raw_file.endswith('.mzml'):
            MzMLFile().load(raw_file, _cached_experiment)
        else:
            MzMLFile().load(raw_file, _cached_experiment)
        
        _cached_raw_file = raw_file
        _cached_ms1_rt_list = None
        _cached_ms2_rt_list = None
        print(f"[DEBUG] Loaded {_cached_experiment.size()} spectra")
        
        # Note: FeatureFinderAlgorithmPicked is the correct tool for proteomics,
        # but it requires mass trace extraction which isn't implemented here.
        # Instead, we use direct chromatogram extraction and peak apex detection,
        # which works well for finding MS1 chromatographic feature RTs.
        print(f"[DEBUG] Using chromatogram-based peak detection (direct XIC extraction)")
        feature_map = FeatureMap()  # Empty - we use chromatogram method instead
        
        _cached_feature_map = feature_map
    
    return _cached_feature_map

def find_feature_apex_rt(feature_map, precursor_mz, ms1_rts_table=None, mz_tolerance_ppm=20.0, rt_tolerance_sec=60.0):
    """
    Find the chromatographic feature (peak) apex RT for a given precursor m/z.
    Uses both m/z and RT constraints to avoid misassignment in crowded MS1 data.
    
    Args:
        feature_map: FeatureMap from FeatureFinder
        precursor_mz: Precursor m/z to search for
        ms1_rts_table: Optional list of MS1 RTs from table (for RT constraint)
        mz_tolerance_ppm: m/z tolerance in ppm (default 20 ppm)
        rt_tolerance_sec: RT tolerance in seconds (default 60 sec)
    
    Returns:
        (apex_rt, intensity) tuple, or (None, None) if not found
    """
    if feature_map is None:
        return (None, None)
    try:
        if feature_map.size() == 0:
            return (None, None)
    except Exception:
        return (None, None)
    
    mz_tolerance_da = precursor_mz * mz_tolerance_ppm / 1e6
    
    # Calculate anchor RT from table MS1 RTs (median)
    anchor_rt = None
    if ms1_rts_table and len(ms1_rts_table) > 0:
        valid_rts = [rt for rt in ms1_rts_table if rt > 0]
        if valid_rts:
            anchor_rt = np.median(valid_rts)
    
    best_feature = None
    best_intensity = 0.0
    
    for feature in feature_map:
        feature_mz = feature.getMZ()
        feature_rt = feature.getRT()
        feature_intensity = feature.getIntensity()
        
        # Check if m/z matches
        mz_match = abs(feature_mz - precursor_mz) <= mz_tolerance_da
        
        # Check if RT matches (if anchor RT available)
        rt_match = True
        if anchor_rt is not None:
            rt_match = abs(feature_rt - anchor_rt) <= rt_tolerance_sec
        
        # Both m/z and RT must match
        if mz_match and rt_match:
            # Use feature with highest intensity if multiple matches
            if feature_intensity > best_intensity:
                best_intensity = feature_intensity
                best_feature = feature
    
    if best_feature:
        return (best_feature.getRT(), best_feature.getIntensity())
    
    return (None, None)

def find_chromatogram_peak_apex(chromatogram, ms1_rts, window_size=30.0):
    """
    Find the peak apex RT from chromatogram data around MS1 RTs.
    This is a fallback when FeatureFinder doesn't work.
    
    Args:
        chromatogram: list of (rt, intensity) tuples
        ms1_rts: list of MS1 retention times from table
        window_size: window size in seconds around MS1 RTs
    
    Returns:
        (apex_rt, apex_intensity) tuple, or (None, None) if not found
    """
    if not ms1_rts or not chromatogram:
        return (None, None)
    
    # Find center RT from MS1 RTs
    center_rt = sum(ms1_rts) / len(ms1_rts)
    
    # Extract chromatogram points within window
    half_window = window_size / 2.0
    window_start = center_rt - half_window
    window_end = center_rt + half_window
    
    peak_points = [(rt, intensity) for rt, intensity in chromatogram 
                   if window_start <= rt <= window_end]
    
    if not peak_points:
        return (None, None)
    
    # Find point with maximum intensity (peak apex)
    apex_point = max(peak_points, key=lambda x: x[1])
    return apex_point

# Mass constants - aligned with Comet (CometMassSpecUtils.h, CometDataInternal.h)
PROTON_MASS = 1.00727646688   # Comet PROTON_MASS
_HYDROGEN_MONO = 1.007825035  # Comet Hydrogen_Mono
_OXYGEN_MONO = 15.99491463
_CARBON_MONO = 12.00000000
_NITROGEN_MONO = 14.0030740
_SULPHUR_MONO = 31.9720707
C13_C12_DIFF = 1.0033548378  # Mass difference between C13 and C12

# Comet fragment offsets (from GetFragmentIonMass: c=b+NH3, z=y-NH2, z1=y-NH2+H)
_dNH3 = _NITROGEN_MONO + 3 * _HYDROGEN_MONO   # 17.026549
_dNH2 = _NITROGEN_MONO + 2 * _HYDROGEN_MONO   # 16.018724
_dOH2 = _HYDROGEN_MONO + _HYDROGEN_MONO + _OXYGEN_MONO  # H2O

# Monoisotopic residue masses - Comet AssignMass (pdAAMassFragment)
_AA_MASS = {
    'G': _CARBON_MONO*2 + _HYDROGEN_MONO*3 + _NITROGEN_MONO + _OXYGEN_MONO,
    'A': _CARBON_MONO*3 + _HYDROGEN_MONO*5 + _NITROGEN_MONO + _OXYGEN_MONO,
    'S': _CARBON_MONO*3 + _HYDROGEN_MONO*5 + _NITROGEN_MONO + _OXYGEN_MONO*2,
    'P': _CARBON_MONO*5 + _HYDROGEN_MONO*7 + _NITROGEN_MONO + _OXYGEN_MONO,
    'V': _CARBON_MONO*5 + _HYDROGEN_MONO*9 + _NITROGEN_MONO + _OXYGEN_MONO,
    'T': _CARBON_MONO*4 + _HYDROGEN_MONO*7 + _NITROGEN_MONO + _OXYGEN_MONO*2,
    'C': _CARBON_MONO*3 + _HYDROGEN_MONO*5 + _NITROGEN_MONO + _OXYGEN_MONO + _SULPHUR_MONO,
    'L': _CARBON_MONO*6 + _HYDROGEN_MONO*11 + _NITROGEN_MONO + _OXYGEN_MONO,
    'I': _CARBON_MONO*6 + _HYDROGEN_MONO*11 + _NITROGEN_MONO + _OXYGEN_MONO,
    'N': _CARBON_MONO*4 + _HYDROGEN_MONO*6 + _NITROGEN_MONO*2 + _OXYGEN_MONO*2,
    'D': _CARBON_MONO*4 + _HYDROGEN_MONO*5 + _NITROGEN_MONO + _OXYGEN_MONO*3,
    'Q': _CARBON_MONO*5 + _HYDROGEN_MONO*8 + _NITROGEN_MONO*2 + _OXYGEN_MONO*2,
    'K': _CARBON_MONO*6 + _HYDROGEN_MONO*12 + _NITROGEN_MONO*2 + _OXYGEN_MONO,
    'E': _CARBON_MONO*5 + _HYDROGEN_MONO*7 + _NITROGEN_MONO + _OXYGEN_MONO*3,
    'M': _CARBON_MONO*5 + _HYDROGEN_MONO*9 + _NITROGEN_MONO + _OXYGEN_MONO + _SULPHUR_MONO,
    'H': _CARBON_MONO*6 + _HYDROGEN_MONO*7 + _NITROGEN_MONO*3 + _OXYGEN_MONO,
    'F': _CARBON_MONO*9 + _HYDROGEN_MONO*9 + _NITROGEN_MONO + _OXYGEN_MONO,
    'R': _CARBON_MONO*6 + _HYDROGEN_MONO*12 + _NITROGEN_MONO*4 + _OXYGEN_MONO,
    'Y': _CARBON_MONO*9 + _HYDROGEN_MONO*9 + _NITROGEN_MONO + _OXYGEN_MONO*2,
    'W': _CARBON_MONO*11 + _HYDROGEN_MONO*10 + _NITROGEN_MONO*2 + _OXYGEN_MONO,
}

# y-ion: C-term mass + H2O + H+ (dCtermOH2Proton)
_Y_ION_OFFSET = _dOH2 + PROTON_MASS  # 19.01784

# Comet formulas: c = b + NH3 (b = sum + PROTON), z = y - NH2 (y = sum + dOH2 + PROTON), z1 = z + H
_C_ION_OFFSET_NEUTRAL = PROTON_MASS + _dNH3   # 18.03383
_Z_ION_OFFSET_NEUTRAL = _dOH2 + PROTON_MASS - _dNH2   # 2.99912
_Z1_ION_OFFSET_NEUTRAL = _Z_ION_OFFSET_NEUTRAL + _HYDROGEN_MONO   # 4.00694


def theoretical_by_ion_mz(sequence, charge=1):
    """Return (b_mz_list, y_mz_list) for sequence; each list is [(mz, label), ...]. Strips non-letter."""
    seq = ''.join(c for c in str(sequence).upper() if c in _AA_MASS)
    if not seq:
        return [], []
    n = len(seq)
    b_mzs, y_mzs = [], []
    mass_b = PROTON_MASS  # b1 = residue1 + H+
    for i in range(n - 1):
        mass_b += _AA_MASS.get(seq[i], 0)
        b_mzs.append((mass_b / charge, f'b{i+1}'))
    mass_y = _Y_ION_OFFSET  # y1 = C-term residue + H2O + H (matches Comet dCtermOH2Proton)
    for i in range(n - 1):
        mass_y += _AA_MASS.get(seq[n - 1 - i], 0)
        y_mzs.append((mass_y / charge, f'y{i+1}'))
    return b_mzs, y_mzs


def theoretical_c_z_z1_ion_mz(sequence, fragment_charge=1):
    """Return (c_list, z_list, z1_list) for sequence; each list is [(mz, label), ...]. Strips non-letter.
    Only c, z, and z+1 ions (for ETD/ECD-style labeling on MS2).
    Matches Comet: m/z = (neutral + (z-1)*PROTON) / z."""
    seq = ''.join(c for c in str(sequence).upper() if c in _AA_MASS)
    if not seq:
        return [], [], []
    n = len(seq)
    c_mzs, z_mzs, z1_mzs = [], [], []
    # c: N-term; neutral = sum(residues 1..k) + C_OFFSET; m/z = (neutral + (z-1)*PROTON)/z
    protons = (fragment_charge - 1) * PROTON_MASS
    mass_c = _C_ION_OFFSET_NEUTRAL + protons
    for i in range(n - 1):
        mass_c += _AA_MASS.get(seq[i], 0)
        c_mzs.append((mass_c / fragment_charge, f'c{i+1}'))
    # z and z+1: C-term; same residue sums as y
    mass_z = _Z_ION_OFFSET_NEUTRAL + protons
    mass_z1 = _Z1_ION_OFFSET_NEUTRAL + protons
    for i in range(n - 1):
        mass_z += _AA_MASS.get(seq[n - 1 - i], 0)
        mass_z1 += _AA_MASS.get(seq[n - 1 - i], 0)
        z_mzs.append((mass_z / fragment_charge, f'z{i+1}'))
        z1_mzs.append((mass_z1 / fragment_charge, f'z{i+1}+1'))
    return c_mzs, z_mzs, z1_mzs


def parse_comet_c_z_z1_matched_ions(group):
    """
    Parse significant fragment ions from CSV (prefer canonical significant columns).
    Return set of ion names that are c, z, or z+1 only (e.g. {'c4', 'z6', 'z1_7'}).
    Ion format: c2, z4, z1_6 (z+1 fragment 6).
    """
    col = None
    for candidate in ['significant_frags', 'refined_significant_frags', 'significant_fragment_ions']:
        if candidate in group.columns:
            col = candidate
            break
    if col not in group.columns:
        return set()
    out = set()
    for _, row in group.iterrows():
        val = row.get(col, '')
        if pd.isna(val) or val == '':
            continue
        s = str(val).strip().strip('"')
        for part in s.split(','):
            part = part.strip()
            if not part:
                continue
            # c1, c2, ... ; z1, z2, ... ; z1_1, z1_2, ... (z+1)
            if part.startswith('c') and part[1:].isdigit():
                out.add(part)
            elif part.startswith('z1_') and part.split('_')[-1].isdigit():
                out.add(part)
            elif part.startswith('z') and not part.startswith('z1_') and part[1:].isdigit():
                out.add(part)
    return out


def comet_matched_ions_to_theoretical_mz(comet_ion_names, peptide, fragment_charge=1):
    """
    Map Comet-matched ion names (c/z/z+1 only) to (mz, display_label).
    comet_ion_names: set/list of strings e.g. {'c4', 'z6', 'z1_7'}.
    Returns list of (mz, display_label) for each name that has a valid theoretical m/z.
    """
    c_list, z_list, z1_list = theoretical_c_z_z1_ion_mz(peptide, fragment_charge)
    result = []
    for name in comet_ion_names:
        mz, label = None, None
        if name.startswith('c') and name[1:].isdigit():
            k = int(name[1:])
            if 1 <= k <= len(c_list):
                mz, label = c_list[k - 1]
        elif name.startswith('z1_'):
            k = int(name.split('_')[-1])
            if 1 <= k <= len(z1_list):
                mz, label = z1_list[k - 1]  # label is e.g. "z6+1"
        elif name.startswith('z') and name[1:].isdigit():
            k = int(name[1:])
            if 1 <= k <= len(z_list):
                mz, label = z_list[k - 1]
        if mz is not None and label is not None:
            result.append((mz, label))
    return result


def get_best_theoretical_match(ion_name, peptide, observed_mz, charges=(1, 2, 3, 4, 5)):
    """
    Charge-agnostic match: try all plausible fragment charges (z=1..5), pick the
    hypothesis with minimum |ppm|. ppm is a property of the match, not the instrument—
    wrong charge hypothesis explodes ppm even with perfect measurement.
    Returns (theoretical_mz, best_charge, ppm) or None if no valid match.
    """
    best_theo, best_charge, best_ppm = None, None, float('inf')
    for z in charges:
        theo_list = comet_matched_ions_to_theoretical_mz([ion_name], peptide, fragment_charge=z)
        if not theo_list:
            continue
        theo_mz = theo_list[0][0]
        if theo_mz <= 0:
            continue
        ppm = abs(observed_mz - theo_mz) / theo_mz * 1e6
        if ppm < best_ppm:
            best_ppm = ppm
            best_theo = theo_mz
            best_charge = z
    if best_theo is not None:
        return (best_theo, best_charge, best_ppm)
    return None


def matches_b_or_y_ion(ion_name, peptide, observed_mz, ppm_threshold, charges=(1, 2, 3, 4, 5)):
    """
    For Comet c/z ion at same cleavage, check if observed m/z matches b or y theoretical.
    Returns True if observed matches b or y within ppm → reject as CID/HCD mis-assignment.
    """
    seq = ''.join(c for c in str(peptide).upper() if c in _AA_MASS)
    if not seq:
        return False
    n = len(seq)
    k = None
    if ion_name.startswith('c') and ion_name[1:].isdigit():
        k = int(ion_name[1:])
    elif ion_name.startswith('z1_'):
        k = n - int(ion_name.split('_')[-1])
    elif ion_name.startswith('z') and ion_name[1:].isdigit():
        k = n - int(ion_name[1:])
    if k is None or k < 1 or k >= n:
        return False
    mass_b = PROTON_MASS
    for i in range(k):
        mass_b += _AA_MASS.get(seq[i], 0)
    mass_y = _Y_ION_OFFSET
    for i in range(k, n):
        mass_y += _AA_MASS.get(seq[i], 0)
    for z in charges:
        protons = (z - 1) * PROTON_MASS
        b_mz = (mass_b + protons) / z
        y_mz = (mass_y + protons) / z
        for theo in (b_mz, y_mz):
            if theo <= 0:
                continue
            ppm = abs(observed_mz - theo) / theo * 1e6
            if ppm <= ppm_threshold:
                return True
    return False


def _display_label_for_comet_ion(name):
    """Convert Comet ion name to display label: c4 -> c4, z6 -> z6, z1_6 -> z6+1."""
    if name.startswith('z1_') and name.split('_')[-1].isdigit():
        return f"z{name.split('_')[-1]}+1"
    return name


def _pair_token_to_display_label(token):
    """Convert fragment pair token (e.g. c3, z5, z1_6, z6+1) to MS2 display label (c3, z5, z6+1)."""
    if not token or not isinstance(token, str):
        return token
    t = token.strip()
    if t.startswith('z1_') and t.split('_')[-1].isdigit():
        return f"z{t.split('_')[-1]}+1"
    # Already in display format z6+1
    if re.match(r'^z\d+\+1$', t):
        return t
    return t


def _overhang_position_from_single_aa_pair(pair_str, L):
    """
    Given a fragment pair string (e.g. 'c3|c4', 'c3-c4', 'z5|z6') and peptide length L, return the
    1-based peptide position of the single-AA overhang, or None if the pair is not a valid
    single-AA overhang (fragments must be same series and differ by exactly 1).
    Used so red squares come only from significant fragment pairs with single-AA overhangs.
    """
    if not pair_str or not isinstance(pair_str, str) or L <= 0:
        return None
    tokens = [t.strip() for t in pair_str.replace('-', '|').split('|') if t.strip()]
    expanded = []
    for t in tokens:
        expanded.extend(_expand_pair_tokens(t))
    if len(expanded) < 2:
        return None
    a_tok, b_tok = expanded[0], expanded[1]
    # Get fragment numbers and series
    def _num_series(t):
        if not t or not isinstance(t, str):
            return None, None
        t = t.strip()
        if t.startswith('c') and t[1:].isdigit():
            return int(t[1:]), 'c'
        if re.match(r'^z(\d+)\+1$', t):
            return int(re.match(r'^z(\d+)\+1$', t).group(1)), 'z1'
        if t.startswith('z1_') and t.split('_')[-1].isdigit():
            return int(t.split('_')[-1]), 'z1'
        if t.startswith('z') and t[1:].isdigit():
            return int(t[1:]), 'z'
        return None, None
    na, sa = _num_series(a_tok)
    nb, sb = _num_series(b_tok)
    if na is None or nb is None or sa != sb or sa is None:
        return None
    lo, hi = min(na, nb), max(na, nb)
    if hi - lo != 1 or lo < 1 or hi > L:
        return None
    if sa == 'c':
        return hi  # c_n and c_{n+1} -> overhang at n+1
    return L - lo  # z_n and z_{n+1} -> overhang at L-n


def _pair_token_to_position(token, L):
    """
    Given a fragment token (c3, z5, z1_6, z6+1) and peptide length L, return the 1-based overhang position or None.
    c_n -> n; z_n / z1_n / z n+1 -> L - n + 1.
    """
    if not token or not isinstance(token, str) or L <= 0:
        return None
    t = token.strip()
    # c_n
    if t.startswith('c') and t[1:].isdigit():
        n = int(t[1:])
        return n if 1 <= n <= L else None
    # z1_n or z1_n
    if t.startswith('z1_') and t.split('_')[-1].isdigit():
        n = int(t.split('_')[-1])
        return (L - n + 1) if 1 <= n <= L else None
    # z n+1 (display format)
    m = re.match(r'^z(\d+)\+1$', t)
    if m:
        n = int(m.group(1))
        return (L - n + 1) if 1 <= n <= L else None
    # z_n
    if t.startswith('z') and not t.startswith('z1_') and t[1:].isdigit():
        n = int(t[1:])
        return (L - n + 1) if 1 <= n <= L else None
    return None


def _expand_pair_tokens(token):
    """
    Expand a pair token that may be hyphenated (e.g. z1_6-z1_7, c3-c4) into a list of
    single fragment tokens. If not hyphenated or only one part is valid, return [token] or list of valid parts.
    """
    if not token or not isinstance(token, str):
        return []
    t = token.strip()
    if '-' not in t:
        return [t]
    # CSV format can be "z1_6-z1_7" or "y6-y7" (we only parse c/z/z+1)
    parts = [p.strip() for p in t.split('-') if p.strip()]
    out = []
    for p in parts:
        # Keep only c/z/z+1-looking tokens so _pair_token_to_position can parse them
        if (p.startswith('c') and p[1:].isdigit()) or (p.startswith('z1_') and p.split('_')[-1].isdigit()) or re.match(r'^z\d+\+1$', p) or (p.startswith('z') and p[1:].isdigit()):
            out.append(p)
    return out if out else [t]


def _parse_single_aa_overhang_positions(group, peptide, placed_ion_labels=None):
    """
    Return set of 1-based peptide positions that are single-AA overhangs.
    When placed_ion_labels is provided (ions with significant MS2 signal above cutoff), only overhangs
    from those fragments are included. If placed_ion_labels is explicitly provided but empty (no
    significant fragments), returns empty set—do not fall back to CSV so red squares match actual
    MS2 evidence. When placed_ion_labels is None, uses CSV: single_aa_overhang_fragment_pairs,
    matched fragment ions, and single_aa_overhangs_protein_positions.
    """
    overhang = set()
    L = len(peptide) if peptide else 0
    if L == 0:
        return overhang
    # Caller passed significant-fragment set but it's empty: no red squares from CSV
    if placed_ion_labels is not None and len(placed_ion_labels) == 0:
        return overhang
    row0 = group.iloc[0] if hasattr(group, 'iloc') else group
    use_placed_only = placed_ion_labels is not None and len(placed_ion_labels) > 0

    # When use_placed_only: red = only where a fragment has "one more than one next to it" (consecutive pair).
    # Same rule for peptide row: if it's red in any fragment row (c or z), it's red in the sequence.
    if use_placed_only:
        c_nums, z_nums, z1_nums = [], [], []  # z and z+1 are separate chemistry; do not pair across series
        for lbl in placed_ion_labels:
            if not lbl or not isinstance(lbl, str):
                continue
            if lbl.startswith('c') and lbl[1:].isdigit():
                n = int(lbl[1:])
                if 1 <= n <= L:
                    c_nums.append(n)
            elif re.match(r'^z\d+\+1$', lbl):
                n = int(re.match(r'^z(\d+)\+1$', lbl).group(1))
                if 1 <= n <= L:
                    z1_nums.append(n)
            elif lbl.startswith('z') and lbl[1:].isdigit():
                n = int(lbl[1:])
                if 1 <= n <= L:
                    z_nums.append(n)
        # Consecutive c: c_n and c_{n+1} both displayed → overhang at position n+1 (red)
        c_set = set(c_nums)
        for n in sorted(c_set):
            if (n + 1) in c_set:
                overhang.add(n + 1)
        # Consecutive z only: z_n and z_{n+1} both displayed → overhang at 1-based position L-n+1 (first residue of z_n)
        z_set = set(z_nums)
        for n in sorted(z_set):
            if (n + 1) in z_set:
                overhang.add(L - n + 1)
        # Consecutive z+1 only: z_n+1 and z_{n+1}+1 both displayed → overhang at L-n+1
        z1_set = set(z1_nums)
        for n in sorted(z1_set):
            if (n + 1) in z1_set:
                overhang.add(L - n + 1)
        # C-terminal overhang: position L from c_{L-1} or from z_1 / z_1+1 (last residue is single-AA fragment)
        if L >= 2 and (L - 1) in c_set:
            overhang.add(L)
        if 1 in z_set:
            overhang.add(L)
        if 1 in z1_set:
            overhang.add(L)
        return overhang
    # Not placed-only: prefer significant_single_aa_overhangs when available (>= 2 significant rule); else CSV pairs and single_aa_overhangs_protein_positions
    val_sig = row0.get('significant_single_aa_overhangs') if hasattr(row0, 'get') else None
    if pd.notna(val_sig) and str(val_sig).strip():
        # Format: "2L,3K,5F" (position + AA); extract 1-based positions; N-term (1) discarded for consistency with placed-ion path
        for part in str(val_sig).strip().split(','):
            part = part.strip()
            if len(part) >= 2 and part[:-1].isdigit() and part[-1].isalpha():
                try:
                    pos = int(part[:-1])
                    if 1 <= pos <= L:
                        overhang.add(pos)
                except (TypeError, ValueError):
                    pass
        overhang.discard(1)
        if overhang:
            return overhang
    val = row0.get('single_aa_overhang_fragment_pairs') if hasattr(row0, 'get') else None
    if pd.notna(val) and str(val).strip():
        s = str(val).strip().strip('"').replace(';', ',').replace(',', ' ')
        for part in s.split():
            part = part.strip()
            if not part:
                continue
            tokens = [t.strip() for t in part.split('|') if t.strip()]
            for token in tokens:
                # Hyphenated tokens (e.g. z1_6-z1_7) need to be expanded so each fragment gets an overhang position
                for sub in _expand_pair_tokens(token):
                    pos = _pair_token_to_position(sub, L)
                    if pos is not None:
                        overhang.add(pos)
    else:
        comet_ions = parse_comet_c_z_z1_matched_ions(group)
        for name in comet_ions:
            if name.startswith('c') and name[1:].isdigit():
                n = int(name[1:])
                if n == L - 1 and L >= 2:
                    overhang.add(L)
    # CSV red-bubble positions: only when not restricting to placed ions
    if not use_placed_only:
        val = row0.get('single_aa_overhangs_protein_positions') if hasattr(row0, 'get') else None
        if pd.notna(val) and str(val).strip():
            start = None
            seq_start = row0.get('sequence_start_pos') if hasattr(row0, 'get') else None
            if pd.notna(seq_start) and seq_start is not None:
                try:
                    start = int(seq_start)
                except (TypeError, ValueError):
                    pass
            if start is None or start <= 0:
                sp = row0.get('sequence_positions') if hasattr(row0, 'get') else None
                if pd.notna(sp) and str(sp).strip():
                    sp_str = str(sp).strip()
                    if '-' in sp_str:
                        try:
                            start = int(sp_str.split('-')[0])
                        except (ValueError, TypeError):
                            pass
                    elif sp_str.isdigit():
                        try:
                            start = int(sp_str)
                        except (ValueError, TypeError):
                            pass
            for part in str(val).strip().split(','):
                part = part.strip()
                if len(part) >= 2 and part[:-1].isdigit() and part[-1].isalpha():
                    try:
                        protein_pos = int(part[:-1])
                        if start is not None and start > 0:
                            peptide_pos = protein_pos - start + 1
                        else:
                            peptide_pos = protein_pos
                        if 1 <= peptide_pos <= L:
                            overhang.add(peptide_pos)
                        # Fallback: some CSVs use peptide-relative positions in this column (e.g. "2L,3K")
                        elif 1 <= protein_pos <= L:
                            overhang.add(protein_pos)
                    except (TypeError, ValueError):
                        pass
    return overhang


def get_single_aa_overhang_positions_from_row(row, peptide, placed_ion_labels=None):
    """
    Same logic as _parse_single_aa_overhang_positions but for a single row (dict-like).
    Used by sequence coverage / combined pipeline so overhang positions match the chromatogram plots.
    row: dict with keys single_aa_overhang_fragment_pairs, single_aa_overhangs_protein_positions,
         sequence_start_pos, sequence_positions (optional).
    Returns: set of 1-based peptide positions that are single-AA overhangs.
    """
    if row is None:
        return set()
    try:
        one_row = pd.DataFrame([row])
    except Exception:
        return set()
    return _parse_single_aa_overhang_positions(one_row, peptide, placed_ion_labels=placed_ion_labels)


def _get_c_z_z1_lists_from_csv(group, L):
    """
    Extract c_list, z_list, z1_list (fragment ordinals) from CSV so we can draw fragment rows
    when MS2 placed_ion_labels is missing but overhang positions come from CSV (red squares).
    Uses single_aa_overhang_fragment_pairs first; fallback to parse_comet_c_z_z1_matched_ions.
    Returns (c_list, z_list, z1_list) each sorted list of ints in [1, L].
    """
    c_nums, z_nums, z1_nums = [], [], []
    row0 = group.iloc[0] if hasattr(group, 'iloc') else group
    val = row0.get('single_aa_overhang_fragment_pairs') if hasattr(row0, 'get') else None
    if pd.notna(val) and str(val).strip():
        s = str(val).strip().strip('"').replace(';', ',').replace(',', ' ')
        for part in s.split():
            part = part.strip()
            if not part:
                continue
            tokens = [t.strip() for t in part.split('|') if t.strip()]
            for token in tokens:
                for sub in _expand_pair_tokens(token):
                    if not sub or not isinstance(sub, str):
                        continue
                    if sub.startswith('c') and sub[1:].isdigit():
                        n = int(sub[1:])
                        if 1 <= n <= L:
                            c_nums.append(n)
                    elif re.match(r'^z\d+\+1$', sub):
                        n = int(re.match(r'^z(\d+)\+1$', sub).group(1))
                        if 1 <= n <= L:
                            z1_nums.append(n)
                    elif sub.startswith('z1_') and sub.split('_')[-1].isdigit():
                        n = int(sub.split('_')[-1])
                        if 1 <= n <= L:
                            z1_nums.append(n)
                    elif sub.startswith('z') and sub[1:].isdigit():
                        n = int(sub[1:])
                        if 1 <= n <= L:
                            z_nums.append(n)
    if not c_nums and not z_nums and not z1_nums:
        comet_ions = parse_comet_c_z_z1_matched_ions(group)
        for name in comet_ions:
            if name.startswith('c') and name[1:].isdigit():
                n = int(name[1:])
                if 1 <= n <= L:
                    c_nums.append(n)
            elif re.match(r'^z\d+\+1$', name):
                n = int(re.match(r'^z(\d+)\+1$', name).group(1))
                if 1 <= n <= L:
                    z1_nums.append(n)
            elif name.startswith('z') and name[1:].isdigit():
                n = int(name[1:])
                if 1 <= n <= L:
                    z_nums.append(n)
    c_list = sorted(set(c_nums))
    z_list = sorted(set(z_nums))
    z1_list = sorted(set(z1_nums))
    return (c_list, z_list, z1_list)


def get_comet_c_z_z1_ion_mz_from_csv(group):
    """
    If CSV has significant_frags and significant_frags_mz (same order), return
    list of (mz, display_label) for c/z/z+1 only. Otherwise return None.
    """
    col_ions = None
    col_mz = None
    if 'significant_frags' in group.columns and 'significant_frags_mz' in group.columns:
        col_ions = 'significant_frags'
        col_mz = 'significant_frags_mz'
    elif 'refined_significant_frags' in group.columns and 'refined_significant_frags_mz' in group.columns:
        col_ions = 'refined_significant_frags'
        col_mz = 'refined_significant_frags_mz'
    elif 'significant_fragment_ions' in group.columns and 'significant_fragment_mz' in group.columns:
        col_ions = 'significant_fragment_ions'
        col_mz = 'significant_fragment_mz'
    if col_ions not in group.columns or col_mz not in group.columns:
        return None
    # Use first row that has both columns (they are per-PSM; order is same within a row)
    for _, row in group.iterrows():
        ions_val = row.get(col_ions, '')
        mz_val = row.get(col_mz, '')
        if pd.isna(ions_val) or pd.isna(mz_val) or not str(ions_val).strip() or not str(mz_val).strip():
            continue
        ions_str = str(ions_val).strip().strip('"')
        mz_str = str(mz_val).strip().strip('"')
        ions_list = [p.strip() for p in ions_str.split(',') if p.strip()]
        mz_list = [p.strip() for p in mz_str.split(',') if p.strip()]
        if len(ions_list) != len(mz_list):
            continue
        result = []
        for name, mz_s in zip(ions_list, mz_list):
            if not name or not mz_s:
                continue
            # Only c, z, z+1
            if name.startswith('c') and name[1:].isdigit():
                pass
            elif name.startswith('z1_') and name.split('_')[-1].isdigit():
                pass
            elif name.startswith('z') and name[1:].isdigit():
                pass
            else:
                continue
            try:
                mz = float(mz_s)
            except ValueError:
                continue
            if mz <= 0:
                continue
            result.append((mz, _display_label_for_comet_ion(name)))
        if result:
            return result
    return None


def _ms2_intensity_within_ppm(ms2_mzs, ms2_ints, theoretical_mz, ppm=5.0):
    """Return intensity from summed MS2 spectrum for peak within ppm of theoretical m/z, or 0 if none."""
    if ms2_mzs is None or ms2_ints is None or len(ms2_mzs) == 0 or theoretical_mz <= 0:
        return 0.0
    ms2_mzs_arr = np.asarray(ms2_mzs)
    ms2_ints_arr = np.asarray(ms2_ints)
    tol = max(0.02, theoretical_mz * ppm * 1e-6)
    idx = np.argmin(np.abs(ms2_mzs_arr - theoretical_mz))
    if np.abs(ms2_mzs_arr[idx] - theoretical_mz) <= tol:
        return float(ms2_ints_arr[idx]) if idx < len(ms2_ints_arr) else 0.0
    return 0.0


def _ms2_label_positions_no_overlap(ion_mz_labels, ms2_mzs, ms2_ints, y_max_ms2, ppm_ms2=None, label_offset_frac=0.16,
                                    mz_near=100.0, min_y_sep_frac=0.20, min_intensity_frac=None, horizontal_offset_mz=28.0):
    """
    Stage 2 significant-fragment filter: find ions that match a peak in (ms2_mzs, ms2_ints) within ppm_ms2
    AND with intensity >= min_intensity_frac of max (default 0.5%). Return list of (x_text, y_text, label, xy_peak).
    Positions are chosen so labels do not overlap. Defaults use SIGNIFICANT_FRAGMENT_PPM_MS2 and
    SIGNIFICANT_FRAGMENT_MIN_INTENSITY_FRAC for pipeline consistency.
    """
    import numpy as np
    if ppm_ms2 is None:
        ppm_ms2 = SIGNIFICANT_FRAGMENT_PPM_MS2
    if min_intensity_frac is None:
        min_intensity_frac = SIGNIFICANT_FRAGMENT_MIN_INTENSITY_FRAC
    out = []
    ms2_ints_arr = np.asarray(ms2_ints) if len(ms2_ints) else np.array([])
    max_int = float(np.max(ms2_ints_arr)) if len(ms2_ints_arr) > 0 else 0.0
    noise_floor = max_int * min_intensity_frac if max_int > 0 else 0.0
    for mz_theo, ion_label in ion_mz_labels:
        if mz_theo <= 0:
            continue
        tol = max(0.02, mz_theo * ppm_ms2 * 1e-6)
        idx = np.argmin(np.abs(np.asarray(ms2_mzs) - mz_theo))
        peak_int = ms2_ints_arr[idx] if idx < len(ms2_ints_arr) else 0.0
        if np.abs(ms2_mzs[idx] - mz_theo) <= tol and peak_int >= noise_floor:
            out.append((ms2_mzs[idx], peak_int, ion_label))
    if not out:
        return out
    # Sort by m/z so labels are in order
    out.sort(key=lambda t: t[0])
    min_y_sep = min_y_sep_frac * y_max_ms2
    # Approximate label extent in data coords (bubble + text; use generous margin)
    x_margin = 24.0   # min horizontal gap between label centers (m/z units)
    y_margin = min_y_sep  # min vertical gap
    y_cap = y_max_ms2 * 1.02  # above this, use new column instead of stacking higher

    def overlaps_any(x_text, y_text, placed_list):
        for (px, py, _, _) in placed_list:
            if abs(x_text - px) < x_margin and abs(y_text - py) < y_margin:
                return True
        return False

    placed = []
    for i, (mz, intensity, label) in enumerate(out):
        y0 = intensity + label_offset_frac * y_max_ms2
        x_text = mz
        stack_idx = 0
        col = 0
        while overlaps_any(x_text, y0, placed):
            if y0 > y_cap:
                # Start a new column to the right instead of stacking higher
                col += 1
                y0 = intensity + label_offset_frac * y_max_ms2
                x_text = mz + col * horizontal_offset_mz
                stack_idx = 0
            else:
                y0 += y_margin
                stack_idx += 1
                x_text = mz + (col + (stack_idx % 2)) * horizontal_offset_mz
        placed.append((x_text, y0, label, (mz, intensity)))
    return [(x, y, lbl, xy) for x, y, lbl, xy in placed]


def create_ms2_spectrum_figure(row, mzml_path, figsize=(12, 6), ppm_ms2=5.0):
    """
    Create a standalone MS2 spectrum figure for a peptide (same style as chromatogram extraction).
    Labels significant fragment peaks, pairs, and protein positions.
    
    Args:
        row: dict-like with plain_peptide, charge, detected_peak_min_rt/max_rt (or collection_*, anchor_rt),
             significant_frags, single_aa_overhang_fragment_pairs, single_aa_overhangs_protein_positions,
             sequence_start_pos or sequence_positions
        mzml_path: path to mzML file
        figsize: (width, height) for the figure
        ppm_ms2: PPM tolerance for peak matching
    
    Returns:
        matplotlib Figure or None if MS2 cannot be extracted
    """
    peptide = row.get('plain_peptide') or row.get('sequence') or ''
    if not peptide or not isinstance(peptide, str):
        return None
    peptide = str(peptide).strip()
    charge = int(row.get('charge', 1)) if pd.notna(row.get('charge')) else 1
    
    min_rt = row.get('detected_peak_min_rt') or row.get('collection_min_rt')
    max_rt = row.get('detected_peak_max_rt') or row.get('collection_max_rt')
    if pd.isna(min_rt) or pd.isna(max_rt):
        anchor = row.get('anchor_rt') or row.get('MS1_retention_time_sec')
        if pd.notna(anchor) and anchor is not None:
            anchor = float(anchor)
            min_rt, max_rt = max(0, anchor - 15), anchor + 15  # ±15 s window when only anchor available
        else:
            min_rt, max_rt = None, None
    min_rt = float(min_rt) if min_rt is not None and not pd.isna(min_rt) else None
    max_rt = float(max_rt) if max_rt is not None and not pd.isna(max_rt) else None
    if min_rt is None or max_rt is None or max_rt < min_rt:
        return None
    
    ms2_mzs, ms2_ints = get_averaged_ms2_spectrum_in_window(mzml_path, min_rt, max_rt, ppm_tolerance=20.0)
    if ms2_mzs is None or len(ms2_mzs) == 0 or ms2_ints is None or len(ms2_ints) == 0:
        return None
    
    clean_peptide = re.sub(r'\[.*?\]', '', peptide) if peptide else ''
    sig_frags_val = row.get('significant_frags') or row.get('significant_fragment_ions') or row.get('refined_significant_frags') or ''
    if pd.isna(sig_frags_val) or not str(sig_frags_val).strip():
        return None
    
    sig_ion_names = set(p.strip() for p in str(sig_frags_val).replace(',', ' ').split() if p.strip())
    ion_mz_labels = comet_matched_ions_to_theoretical_mz(sig_ion_names, peptide, fragment_charge=1)
    if not ion_mz_labels:
        return None
    
    mz_max_plot = max(ms2_mzs) * 1.02 if len(ms2_mzs) > 0 else 500
    ion_mz_labels = [(mz, lbl) for mz, lbl in ion_mz_labels if 0 < mz <= mz_max_plot]
    if not ion_mz_labels:
        return None
    
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    
    ax.stem(ms2_mzs, ms2_ints, basefmt=' ', linefmt='#888888', markerfmt=' ')
    ax.set_xlabel('m/z', fontsize=14, fontfamily='serif', color='0.85')
    ax.set_ylabel('Intensity', fontsize=14, fontfamily='serif', color='0.85')
    y_max_ms2 = float(np.max(ms2_ints)) * 1.05 if len(ms2_ints) > 0 else 1.0
    ax.set_ylim([0, y_max_ms2])
    ax.set_xlim([0, mz_max_plot])
    ax.tick_params(labelsize=11, colors='0.85')
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily('serif')
        label.set_color('0.85')
    for spine in ax.spines.values():
        spine.set_color('0.6')
    ax.grid(True, alpha=0.25)
    
    min_rt_min = min_rt / 60.0
    max_rt_min = max_rt / 60.0
    mods = str(row.get('modifications', '-')).strip() if pd.notna(row.get('modifications')) else '-'
    if not mods or mods.lower() == 'nan':
        mods = '-'
    title = f"{peptide[:40]}{'...' if len(peptide) > 40 else ''} +{charge} | {mods} | RT {min_rt_min:.2f}-{max_rt_min:.2f} min"
    ax.set_title(title, fontsize=12, fontweight='bold', fontfamily='serif', color='0.9')
    
    label_placements = _ms2_label_positions_no_overlap(
        ion_mz_labels, ms2_mzs, ms2_ints, y_max_ms2, ppm_ms2=ppm_ms2, label_offset_frac=0.14,
        mz_near=80.0, min_y_sep_frac=0.12, min_intensity_frac=0.0, horizontal_offset_mz=18.0)
    
    placed_ion_labels = {lbl for (_, _, lbl, _) in label_placements}
    L = len(clean_peptide) if clean_peptide else 0
    c_set = set(int(lbl[1:]) for lbl in placed_ion_labels if lbl.startswith('c') and lbl[1:].isdigit())
    z_set = set(int(lbl[1:]) for lbl in placed_ion_labels if re.match(r'^z\d+$', lbl))
    z1_set = set(int(re.match(r'^z(\d+)\+1$', lbl).group(1)) for lbl in placed_ion_labels if re.match(r'^z\d+\+1$', lbl))
    c_set = {n for n in c_set if 1 <= n <= L}
    z_set = {n for n in z_set if 1 <= n <= L}
    z1_set = {n for n in z1_set if 1 <= n <= L}
    
    ions_in_pair = set()
    for n in c_set:
        if (n - 1) in c_set or n == L:
            ions_in_pair.add(f'c{n}')
    for n in z_set:
        if (n - 1) in z_set:
            ions_in_pair.add(f'z{n}')
    for n in z1_set:
        if (n - 1) in z1_set:
            ions_in_pair.add(f'z{n}+1')
    
    significant_pairs = [f'c{n}|c{n+1}' for n in sorted(c_set) if (n + 1) in c_set]
    significant_pairs += [f'z{n}|z{n+1}' for n in sorted(z_set) if (n + 1) in z_set]
    significant_pairs += [f'z{n}+1|z{n+1}+1' for n in sorted(z1_set) if (n + 1) in z1_set]
    
    overhang_positions = set()
    for n in sorted(c_set):
        if (n + 1) in c_set:
            overhang_positions.add(n + 1)
    for n in sorted(z_set):
        if (n + 1) in z_set:
            overhang_positions.add(L - n)
    for n in sorted(z1_set):
        if (n + 1) in z1_set:
            overhang_positions.add(L - n)
    if L >= 2 and (L - 1) in c_set:
        overhang_positions.add(L)
    if 1 in z_set or 1 in z1_set:
        overhang_positions.add(L)
    overhang_positions.discard(1)
    
    for x_text, y_text, ion_label, xy_peak in label_placements:
        if ion_label in ions_in_pair:
            bubble_face = OVERHANG_YELLOW
            bubble_text_color = 'black'
            arrow_color = OVERHANG_YELLOW
        elif ion_label.startswith('c'):
            bubble_face = C_FRAGMENT_COVERAGE_COLOR
            bubble_text_color = 'white'
            arrow_color = C_FRAGMENT_COVERAGE_COLOR
        else:
            bubble_face = Z_FRAGMENT_COVERAGE_COLOR
            bubble_text_color = 'white'
            arrow_color = Z_FRAGMENT_COVERAGE_COLOR
        ax.annotate(ion_label, xy_peak, xytext=(x_text, y_text), fontsize=11, fontfamily='serif', fontweight='bold',
            ha='center', va='bottom', rotation=0, textcoords='data', color=bubble_text_color,
            bbox=dict(boxstyle='round,pad=0.4', facecolor=bubble_face, alpha=0.95, edgecolor='black', linewidth=1.0),
            arrowprops=dict(arrowstyle='-', lw=1.0, color=arrow_color))
    
    if label_placements:
        y_max_label = max(y for (_, y, _, _) in label_placements)
        ax.set_ylim([0, max(y_max_ms2, y_max_label * 1.08)])
    
    pairs_val = row.get('single_aa_overhang_fragment_pairs')
    if pairs_val is None or pd.isna(pairs_val):
        pairs_str = ''
    else:
        pairs_str = str(pairs_val)
    if significant_pairs:
        fragment_pairs_text = '\n'.join(p.replace('|', ' | ') for p in significant_pairs)
    else:
        pairs_safe = str(pairs_str) if pairs_str is not None and not (isinstance(pairs_str, float) and pd.isna(pairs_str)) else ''
        fragment_pairs_text = pairs_safe.replace('|', ' | ') if pairs_safe else '—'
    ax.text(0.98, 0.98, fragment_pairs_text, transform=ax.transAxes,
        fontsize=8, fontfamily='serif', va='top', ha='right', wrap=True,
        bbox=dict(boxstyle='round,pad=0.25', facecolor=OVERHANG_YELLOW, alpha=0.9, edgecolor='black', linewidth=0.5),
        zorder=20)
    
    pos_aa_parts = []
    seq_start = row.get('sequence_start_pos') or row.get('sequence_positions')
    try:
        start = int(float(str(seq_start).split('-')[0].split()[0])) if pd.notna(seq_start) and str(seq_start).strip() else None
    except (ValueError, TypeError):
        start = None
    prot_pos_str = row.get('single_aa_overhangs_protein_positions') or ''
    if prot_pos_str and str(prot_pos_str).strip():
        pos_aa_parts = [p.strip() for p in str(prot_pos_str).split(',') if p.strip()]
    elif overhang_positions and clean_peptide and start:
        for pos in sorted(overhang_positions):
            if 1 <= pos <= len(clean_peptide):
                pos_aa_parts.append(f'{start + pos - 1}{clean_peptide[pos - 1]}')
    if pos_aa_parts:
        ax.text(0.98, 0.72, ', '.join(pos_aa_parts), transform=ax.transAxes,
            fontsize=12, fontfamily='serif', fontweight='bold', va='top', ha='right', wrap=True, color='white',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=OVERHANG_RED, alpha=0.95, edgecolor=OVERHANG_RED, linewidth=0.8),
            zorder=20)
    
    plt.tight_layout()
    return fig


def calculate_theoretical_mz(calc_neutral_mass, charge):
    """
    Calculate theoretical m/z from calculated neutral mass and charge.
    
    Formula: mz = (calc_neutral_mass + z * PROTON) / z
    
    Args:
        calc_neutral_mass: Calculated neutral mass (monoisotopic)
        charge: Charge state
    
    Returns:
        Theoretical m/z (monoisotopic)
    """
    if pd.isna(calc_neutral_mass) or calc_neutral_mass <= 0:
        return None
    if pd.isna(charge) or charge <= 0:
        return None
    
    return (calc_neutral_mass + charge * PROTON_MASS) / charge

def isotope_mz_list(mz_mono, charge, n_isos=4):
    """
    Calculate m/z values for isotopic peaks (M, M+1, M+2, ...).
    
    Args:
        mz_mono: Monoisotopic m/z
        charge: Charge state
        n_isos: Number of isotopes to calculate (default 4: M, M+1, M+2, M+3)
    
    Returns:
        List of m/z values [M, M+1, M+2, ...]
    """
    if pd.isna(mz_mono) or mz_mono <= 0 or charge <= 0:
        return []
    
    return [mz_mono + k * (C13_C12_DIFF / charge) for k in range(n_isos)]

def extract_aligned_isotope_chromatograms(raw_file, mz_targets, ppm_tolerance=20.0, rt_min_sec=None, rt_max_sec=None):
    """
    Extract aligned MS1 chromatograms for multiple isotope m/z targets.
    Returns aligned RT grid and intensity matrix (n_scans x n_isos) with zeros preserved.
    
    Peak selection when multiple m/z peaks fall within tolerance:
    - Below 6 ppm: choose the peak with highest intensity
    - At or above 6 ppm: choose the peak closest to theoretical (lowest ppm)
    This favors mass accuracy (closest to 5 ppm) when ppm >= 6, and signal strength when ppm < 6.
    """
    global _cached_experiment, _cached_raw_file, _cached_ms1_rt_list
    
    # Load experiment once and cache it
    if _cached_experiment is None or _cached_raw_file != raw_file:
        print(f"[DEBUG] Loading mzML file (will cache for subsequent peptides)...")
        _cached_experiment = MSExperiment()
        
        if raw_file.endswith('.mzML') or raw_file.endswith('.mzml'):
            MzMLFile().load(raw_file, _cached_experiment)
        else:
            MzMLFile().load(raw_file, _cached_experiment)
        
        _cached_raw_file = raw_file
        _cached_ms1_rt_list = None
        print(f"[DEBUG] Loaded {_cached_experiment.size()} spectra, cached for reuse")
    
    # Build MS1 RT-indexed list once (avoids O(n) scan per peptide)
    if _cached_ms1_rt_list is None:
        _cached_ms1_rt_list = [(s.getRT(), s) for s in _cached_experiment if s.getMSLevel() == 1]
        _cached_ms1_rt_list.sort(key=lambda x: x[0])
    
    ms1_scans = _cached_ms1_rt_list
    
    # Restrict to full peak drift window when requested (for overlay) - use bisect for speed
    if rt_min_sec is not None or rt_max_sec is not None:
        rt_vals = [r[0] for r in ms1_scans]
        left = bisect.bisect_left(rt_vals, rt_min_sec) if rt_min_sec is not None else 0
        right = bisect.bisect_right(rt_vals, rt_max_sec) if rt_max_sec is not None else len(ms1_scans)
        ms1_scans = ms1_scans[left:right]
    
    n_scans = len(ms1_scans)
    n_isos = len(mz_targets)
    
    # Initialize aligned arrays
    rts = np.array([scan[0] for scan in ms1_scans])
    intensity_matrix = np.zeros((n_scans, n_isos), dtype=np.float64)
    measured_mz_matrix = np.full((n_scans, n_isos), np.nan, dtype=np.float64)
    
    # Extract intensities for each isotope at each MS1 scan
    # Peak selection when multiple candidates: favor lowest ppm down to 6 ppm; below 6 ppm, choose highest intensity
    PPM_INTENSITY_THRESHOLD = 6.0  # Below this ppm, prefer highest intensity; at/above, prefer lowest ppm
    for scan_idx, (rt, spec) in enumerate(ms1_scans):
        try:
            mzs, ints = spec.get_peaks()
            mzs = np.asarray(mzs)
            ints = np.asarray(ints)
        except Exception:
            mzs = np.array([p.getMZ() for p in spec])
            ints = np.array([p.getIntensity() for p in spec])
        
        # Extract intensity for each isotope m/z target
        # When multiple peaks within tolerance: pick one by (1) ppm < 6 → highest intensity; (2) ppm >= 6 → lowest ppm
        for iso_idx, mz0 in enumerate(mz_targets):
            tolerance_da = mz0 * ppm_tolerance * 1e-6
            mask = (mzs >= mz0 - tolerance_da) & (mzs <= mz0 + tolerance_da)
            cand_mzs = mzs[mask]
            cand_ints = ints[mask]
            
            if len(cand_mzs) == 0:
                intensity = 0.0
                measured_mz = np.nan
            elif len(cand_mzs) == 1:
                intensity = float(cand_ints[0])
                measured_mz = float(cand_mzs[0])
            else:
                # Multiple candidates: compute ppm for each
                ppms = np.abs(cand_mzs - mz0) / mz0 * 1e6
                # Below 6 ppm: choose highest intensity; at/above 6 ppm: choose lowest ppm
                below_thresh = ppms < PPM_INTENSITY_THRESHOLD
                if np.any(below_thresh):
                    best_idx = np.where(below_thresh)[0][np.argmax(cand_ints[below_thresh])]
                else:
                    best_idx = np.argmin(ppms)
                intensity = float(cand_ints[best_idx])
                measured_mz = float(cand_mzs[best_idx])
            
            intensity_matrix[scan_idx, iso_idx] = intensity
            measured_mz_matrix[scan_idx, iso_idx] = measured_mz
    
    return rts, intensity_matrix, measured_mz_matrix

def get_ms1_spectrum_at_rt(raw_file, target_rt, rt_tolerance_sec=2.0):
    """Extract MS1 spectrum closest to a target retention time. Uses RT-indexed cache."""
    global _cached_experiment, _cached_raw_file, _cached_ms1_rt_list
    
    if _cached_experiment is None or _cached_raw_file != raw_file:
        if _cached_experiment is None:
            _cached_experiment = MSExperiment()
            if raw_file.endswith('.mzML') or raw_file.endswith('.mzml'):
                MzMLFile().load(raw_file, _cached_experiment)
            else:
                MzMLFile().load(raw_file, _cached_experiment)
            _cached_raw_file = raw_file
        _cached_ms1_rt_list = None
    
    if _cached_ms1_rt_list is None:
        _cached_ms1_rt_list = [(s.getRT(), s) for s in _cached_experiment if s.getMSLevel() == 1]
        _cached_ms1_rt_list.sort(key=lambda x: x[0])
    
    if not _cached_ms1_rt_list:
        return (None, None)
    rt_vals = [r[0] for r in _cached_ms1_rt_list]
    idx = bisect.bisect_left(rt_vals, target_rt)
    best_spec = None
    best_rt_diff = float('inf')
    for i in [idx - 1, idx]:
        if 0 <= i < len(_cached_ms1_rt_list):
            rt, spec = _cached_ms1_rt_list[i]
            rt_diff = abs(rt - target_rt)
            if rt_diff <= rt_tolerance_sec and rt_diff < best_rt_diff:
                best_rt_diff = rt_diff
                best_spec = spec
    if best_spec is None:
        return (None, None)
    try:
        mzs, ints = best_spec.get_peaks()
        return (np.asarray(mzs), np.asarray(ints))
    except Exception:
        mzs = np.array([p.getMZ() for p in best_spec])
        ints = np.array([p.getIntensity() for p in best_spec])
        return (mzs, ints)

def get_ms2_spectrum_at_rt(raw_file, target_rt, rt_tolerance_sec=30.0):
    """
    Extract MS2 spectrum closest to a target retention time (e.g. peak apex).
    Uses the same cached MSExperiment as MS1 extraction.

    Args:
        raw_file: Path to mzML file
        target_rt: Target retention time in seconds
        rt_tolerance_sec: Max RT difference to consider (default 30 s)

    Returns:
        (mzs, intensities, spectrum_rt) tuple of numpy arrays and float, or (None, None, None) if not found
    """
    global _cached_experiment, _cached_raw_file

    if _cached_experiment is None or _cached_raw_file != raw_file:
        if _cached_experiment is None:
            _cached_experiment = MSExperiment()
            if raw_file.endswith('.mzML') or raw_file.endswith('.mzml'):
                MzMLFile().load(raw_file, _cached_experiment)
            else:
                MzMLFile().load(raw_file, _cached_experiment)
            _cached_raw_file = raw_file

    exp = _cached_experiment
    best_spec = None
    best_rt_diff = float('inf')

    for spec in exp:
        if spec.getMSLevel() == 2:
            spec_rt = spec.getRT()
            rt_diff = abs(spec_rt - target_rt)
            if rt_diff < best_rt_diff and rt_diff <= rt_tolerance_sec:
                best_rt_diff = rt_diff
                best_spec = spec

    if best_spec is None:
        return (None, None, None)

    mzs = []
    ints = []
    for peak in best_spec:
        mzs.append(peak.getMZ())
        ints.append(peak.getIntensity())
    spec_rt = best_spec.getRT()
    return (np.array(mzs), np.array(ints), spec_rt)

def get_averaged_ms1_spectrum_in_window(raw_file, min_rt, max_rt, mz_min=None, mz_max=None, ppm_tolerance=7.0):
    """
    Extract and sum MS1 spectra across a retention time window (Freestyle-style).
    Uses RT-indexed cache for O(log n) window lookup.
    """
    global _cached_experiment, _cached_raw_file, _cached_ms1_rt_list
    
    # Use cached experiment if available
    if _cached_experiment is None or _cached_raw_file != raw_file:
        if _cached_experiment is None:
            _cached_experiment = MSExperiment()
            if raw_file.endswith('.mzML') or raw_file.endswith('.mzml'):
                MzMLFile().load(raw_file, _cached_experiment)
            else:
                MzMLFile().load(raw_file, _cached_experiment)
            _cached_raw_file = raw_file
        _cached_ms1_rt_list = None
    
    # Build MS1 RT-indexed list once
    if _cached_ms1_rt_list is None:
        _cached_ms1_rt_list = [(s.getRT(), s) for s in _cached_experiment if s.getMSLevel() == 1]
        _cached_ms1_rt_list.sort(key=lambda x: x[0])
    
    rt_list = _cached_ms1_rt_list
    if not rt_list:
        return (None, None)
    rt_values = [r[0] for r in rt_list]
    left = bisect.bisect_left(rt_values, min_rt)
    right = bisect.bisect_right(rt_values, max_rt)
    spectra_in_window = [r[1] for r in rt_list[left:right]]
    
    if len(spectra_in_window) == 0:
        return (None, None)
    
    # Collect all unique m/z values from all spectra (use get_peaks for speed)
    all_mzs = []
    for spec in spectra_in_window:
        try:
            mz_arr, _ = spec.get_peaks()
        except Exception:
            for peak in spec:
                mz = peak.getMZ()
                if mz_min is None or mz >= mz_min:
                    if mz_max is None or mz <= mz_max:
                        all_mzs.append(mz)
            continue
        mz_arr = np.asarray(mz_arr)
        if mz_min is not None:
            mz_arr = mz_arr[mz_arr >= mz_min]
        if mz_max is not None:
            mz_arr = mz_arr[mz_arr <= mz_max]
        all_mzs.extend(mz_arr.tolist())
    
    if len(all_mzs) == 0:
        return (None, None)
    
    # Build common m/z grid using binning approach (more efficient than interpolation)
    # Sort and remove duplicates, then bin nearby peaks
    all_mzs_sorted = np.sort(np.unique(all_mzs))
    
    # Bin m/z values within ppm tolerance
    mz_bins = []
    current_bin_center = None
    
    for mz in all_mzs_sorted:
        if current_bin_center is None:
            # Start new bin
            current_bin_center = mz
            mz_bins.append([mz])
        else:
            # Check if mz is within ppm tolerance of current bin center
            tolerance_da = current_bin_center * ppm_tolerance * 1e-6
            if abs(mz - current_bin_center) <= tolerance_da:
                # Add to current bin
                mz_bins[-1].append(mz)
            else:
                # Start new bin
                current_bin_center = mz
                mz_bins.append([mz])
    
    # Create final m/z grid (use median of each bin as representative m/z)
    mz_grid = np.array([np.median(bin_mzs) for bin_mzs in mz_bins])
    
    # Sum intensities across all spectra for each m/z bin (Freestyle-style: sum, not average)
    intensity_sum = np.zeros(len(mz_grid))
    
    for spec in spectra_in_window:
        try:
            spec_mzs, spec_ints = spec.get_peaks()
            spec_mzs = np.asarray(spec_mzs)
            spec_ints = np.asarray(spec_ints)
        except Exception:
            spec_mzs = np.array([p.getMZ() for p in spec])
            spec_ints = np.array([p.getIntensity() for p in spec])
        if mz_min is not None:
            mask = spec_mzs >= mz_min
            spec_mzs, spec_ints = spec_mzs[mask], spec_ints[mask]
        if mz_max is not None:
            mask = spec_mzs <= mz_max
            spec_mzs, spec_ints = spec_mzs[mask], spec_ints[mask]
        if len(spec_mzs) == 0:
            continue
        # Use searchsorted for O(log n) bin lookup instead of O(n) argmin
        idx_hi = np.searchsorted(mz_grid, spec_mzs)
        idx_lo = np.maximum(0, idx_hi - 1)
        idx_hi = np.minimum(idx_hi, len(mz_grid) - 1)
        tol = spec_mzs * ppm_tolerance * 1e-6
        dist_lo = np.abs(mz_grid[idx_lo] - spec_mzs)
        dist_hi = np.abs(mz_grid[idx_hi] - spec_mzs)
        closer_is_lo = dist_lo <= dist_hi
        bin_indices = np.where(closer_is_lo, idx_lo, idx_hi)
        dist_chosen = np.where(closer_is_lo, dist_lo, dist_hi)
        in_tol = dist_chosen <= tol
        np.add.at(intensity_sum, bin_indices[in_tol], spec_ints[in_tol])
    
    # Return summed intensities (not averaged)
    # This represents total integrated signal across the peak window
    return (mz_grid, intensity_sum)

def count_ms2_scans_in_window(raw_file, min_rt, max_rt):
    """
    Count MS2 scans in a retention time window. Used to verify that more fragments
    come from summing multiple scans (not overcounting).
    """
    global _cached_experiment, _cached_raw_file
    if _cached_experiment is None or _cached_raw_file != raw_file:
        if _cached_experiment is None:
            _cached_experiment = MSExperiment()
        if raw_file.endswith('.mzML') or raw_file.endswith('.mzml'):
            MzMLFile().load(raw_file, _cached_experiment)
        else:
            MzMLFile().load(raw_file, _cached_experiment)
        _cached_raw_file = raw_file
    n = 0
    for spec in _cached_experiment:
        if spec.getMSLevel() == 2 and min_rt <= spec.getRT() <= max_rt:
            n += 1
    return n


def get_averaged_ms2_spectrum_in_window(raw_file, min_rt, max_rt, mz_min=None, mz_max=None, ppm_tolerance=20.0, max_scans=None):
    """
    Sum MS2 spectra across a retention time window (same approach as MS1).
    All MS2 scans with RT in [min_rt, max_rt] are binned by m/z and intensities summed.
    Uses RT-indexed cache for O(log n) window lookup instead of O(n) scan per call.

    Args:
        raw_file: Path to mzML file
        min_rt: Minimum retention time (seconds) for window
        max_rt: Maximum retention time (seconds) for window
        mz_min: Optional minimum m/z (default None = no filter)
        mz_max: Optional maximum m/z (default None = no filter)
        ppm_tolerance: m/z tolerance for binning (default 20 ppm)
        max_scans: If set, only sum first N scans (for testing single-scan vs summed)

    Returns:
        (mzs, intensities) tuple of numpy arrays (summed MS2 spectrum), or (None, None) if no spectra found
    """
    global _cached_experiment, _cached_raw_file, _cached_ms2_rt_list

    if _cached_experiment is None or _cached_raw_file != raw_file:
        if _cached_experiment is None:
            _cached_experiment = MSExperiment()
        if raw_file.endswith('.mzML') or raw_file.endswith('.mzml'):
            MzMLFile().load(raw_file, _cached_experiment)
        else:
            MzMLFile().load(raw_file, _cached_experiment)
        _cached_raw_file = raw_file
        _cached_ms2_rt_list = None  # invalidate when file changes

    # Build RT-indexed list of MS2 spectra (one-time O(n) cost for fast lookups)
    if _cached_ms2_rt_list is None:
        _cached_ms2_rt_list = [(s.getRT(), s) for s in _cached_experiment if s.getMSLevel() == 2]
        _cached_ms2_rt_list.sort(key=lambda x: x[0])

    rt_list = _cached_ms2_rt_list
    if not rt_list:
        return (None, None)
    rt_values = [r[0] for r in rt_list]
    left = bisect.bisect_left(rt_values, min_rt)
    right = bisect.bisect_right(rt_values, max_rt)
    spectra_in_window = [r[1] for r in rt_list[left:right]]

    if max_scans is not None and len(spectra_in_window) > max_scans:
        spectra_in_window = spectra_in_window[:max_scans]

    if len(spectra_in_window) == 0:
        return (None, None)

    # Use get_peaks() for fast numpy arrays (avoid slow peak-by-peak iteration)
    all_mzs = []
    for spec in spectra_in_window:
        try:
            mz_arr, _ = spec.get_peaks()
        except Exception:
            for peak in spec:
                mz = peak.getMZ()
                if mz_min is None or mz >= mz_min:
                    if mz_max is None or mz <= mz_max:
                        all_mzs.append(mz)
            continue
        mz_arr = np.asarray(mz_arr)
        if mz_min is not None:
            mz_arr = mz_arr[mz_arr >= mz_min]
        if mz_max is not None:
            mz_arr = mz_arr[mz_arr <= mz_max]
        all_mzs.extend(mz_arr.tolist())

    if len(all_mzs) == 0:
        return (None, None)

    all_mzs_sorted = np.sort(np.unique(all_mzs))
    mz_bins = []
    current_bin_center = None

    for mz in all_mzs_sorted:
        if current_bin_center is None:
            current_bin_center = mz
            mz_bins.append([mz])
        else:
            tolerance_da = current_bin_center * ppm_tolerance * 1e-6
            if abs(mz - current_bin_center) <= tolerance_da:
                mz_bins[-1].append(mz)
            else:
                current_bin_center = mz
                mz_bins.append([mz])

    mz_grid = np.array([np.median(bin_mzs) for bin_mzs in mz_bins])
    intensity_sum = np.zeros(len(mz_grid))

    # Use get_peaks() + searchsorted for fast vectorized binning
    for spec in spectra_in_window:
        try:
            spec_mzs, spec_ints = spec.get_peaks()
            spec_mzs = np.asarray(spec_mzs)
            spec_ints = np.asarray(spec_ints)
        except Exception:
            spec_mzs = np.array([p.getMZ() for p in spec])
            spec_ints = np.array([p.getIntensity() for p in spec])
        if mz_min is not None:
            mask = spec_mzs >= mz_min
            spec_mzs, spec_ints = spec_mzs[mask], spec_ints[mask]
        if mz_max is not None:
            mask = spec_mzs <= mz_max
            spec_mzs, spec_ints = spec_mzs[mask], spec_ints[mask]
        if len(spec_mzs) == 0:
            continue
        idx_hi = np.searchsorted(mz_grid, spec_mzs)
        idx_lo = np.maximum(0, idx_hi - 1)
        idx_hi = np.minimum(idx_hi, len(mz_grid) - 1)
        tol = spec_mzs * ppm_tolerance * 1e-6
        dist_lo = np.abs(mz_grid[idx_lo] - spec_mzs)
        dist_hi = np.abs(mz_grid[idx_hi] - spec_mzs)
        closer_is_lo = dist_lo <= dist_hi
        bin_indices = np.where(closer_is_lo, idx_lo, idx_hi)
        dist_chosen = np.where(closer_is_lo, dist_lo, dist_hi)
        in_tol = dist_chosen <= tol
        np.add.at(intensity_sum, bin_indices[in_tol], spec_ints[in_tol])

    return (mz_grid, intensity_sum)

def baseline_correct_trace(intensities, window_size=5, method='percentile'):
    """
    Baseline correction for chromatographic traces.
    
    Args:
        intensities: numpy array of intensities
        window_size: window size (for rolling methods, not used for percentile)
        method: 'percentile' (robust, Skyline-like) or 'rolling_min' (legacy)
    
    Returns:
        Baseline-corrected intensities
    """
    if method == 'percentile':
        # Robust method: use low percentile as baseline (Skyline-like)
        # Prevents rolling min from tracking peak edges
        non_zero = intensities[intensities > 0]
        if len(non_zero) > 0:
            baseline = np.percentile(non_zero, 5)  # 5th percentile
        else:
            baseline = 0.0
        corrected = intensities - baseline
        return np.maximum(corrected, 0.0)
    else:
        # Legacy rolling minimum method (can track peak edges)
        if len(intensities) < window_size:
            return intensities - np.min(intensities)
        
        baseline = np.zeros_like(intensities)
        half_window = window_size // 2
        
        for i in range(len(intensities)):
            start_idx = max(0, i - half_window)
            end_idx = min(len(intensities), i + half_window + 1)
            baseline[i] = np.min(intensities[start_idx:end_idx])
        
        corrected = intensities - baseline
        return np.maximum(corrected, 0.0)

def smooth_trace(intensities, window_size=3):
    """
    Simple moving average smoothing.
    
    Args:
        intensities: numpy array of intensities
        window_size: window size for smoothing
    
    Returns:
        Smoothed intensities
    """
    if len(intensities) < window_size:
        return intensities
    
    smoothed = np.convolve(intensities, np.ones(window_size)/window_size, mode='same')
    return smoothed

def find_collection_window_boundaries(rts, intensities, rt_anchor, window_sec=180.0):
    """
    Find peak boundaries (integration window) using the same logic as the chromatogram extraction step.
    Delegates to detect_chromatographic_peaks_windowed and returns the best candidate's boundaries.

    Args:
        rts: numpy array of RT values (seconds)
        intensities: numpy array of intensities (baseline-corrected, smoothed)
        rt_anchor: Anchor RT around which to search
        window_sec: Search window size in seconds (default 180, matches extraction)

    Returns:
        (min_rt, max_rt, apex_rt, apex_intensity) or (None, None, None, None) if not found
    """
    candidates = detect_chromatographic_peaks_windowed(
        rts, intensities,
        rt_anchor=rt_anchor,
        window_sec=window_sec,
        max_peaks=10,
        min_prominence=None,
        min_width_scans=2
    )
    if not candidates:
        return None, None, None, None
    # Best candidate = closest to anchor
    best = min(candidates, key=lambda c: abs(c['apex_rt'] - rt_anchor))
    return (
        best['left_base_rt'],
        best['right_base_rt'],
        best['apex_rt'],
        best['apex_intensity']
    )

def integrate_isotope_area(rts, intensities, min_rt, max_rt):
    """
    Integrate area under trace within RT window using trapezoid rule.
    Baseline correction is assumed to be done beforehand.
    
    Args:
        rts: numpy array of RT values
        intensities: numpy array of intensities (baseline-corrected)
        min_rt: Start RT for integration
        max_rt: End RT for integration
    
    Returns:
        Integrated area
    """
    mask = (rts >= min_rt) & (rts <= max_rt)
    window_rts = rts[mask]
    window_ints = intensities[mask]
    
    if len(window_rts) < 2:
        return 0.0
    
    return float(_np_trapz(window_ints, window_rts))

def compute_coelution_variance(intensity_matrix, start_idx, end_idx, rts):
    """
    Compute co-elution score: variance of isotope apex times (Skyline metric A).
    Lower variance = better co-elution.
    Uses smoothed traces within candidate window to find apex (reduces noise effects).
    
    Args:
        intensity_matrix: numpy array of shape (n_scans, n_isos)
        start_idx: Start index for peak window
        end_idx: End index for peak window
        rts: numpy array of RT values
    
    Returns:
        (variance, score) where score is exp(-variance/sigma^2) for normalization
    """
    if end_idx <= start_idx or start_idx < 0 or end_idx > intensity_matrix.shape[0]:
        return (float('inf'), 0.0)
    
    window_matrix = intensity_matrix[start_idx:end_idx, :]
    window_rts = rts[start_idx:end_idx]
    
    # Find apex time for each isotope trace (use smoothed trace to reduce noise)
    apex_times = []
    for iso_idx in range(window_matrix.shape[1]):
        iso_trace = window_matrix[:, iso_idx]
        if np.max(iso_trace) > 0:
            # Light smoothing to reduce noise before finding apex
            if len(iso_trace) >= 3:
                iso_smoothed = smooth_trace(iso_trace, window_size=3)
            else:
                iso_smoothed = iso_trace
            apex_idx_local = np.argmax(iso_smoothed)
            apex_time = window_rts[apex_idx_local]
            apex_times.append(apex_time)
    
    if len(apex_times) < 2:
        return (float('inf'), 0.0)
    
    # Compute variance (in seconds^2) of isotope apex times
    apex_times_arr = np.array(apex_times)
    variance = np.var(apex_times_arr)
    
    # Score: exp(-variance/sigma^2). Use sigma_t = 2 s so the score spreads:
    # variance 0 -> 1.0, ~4 s^2 -> ~0.37, ~1 s^2 -> ~0.78 (was sigma=5, so score was ~1 for any small variance)
    sigma_t = 2.0
    score = np.exp(-variance / (sigma_t ** 2))
    
    return (variance, score)

def compute_coelution_correlation(intensity_matrix, start_idx, end_idx):
    """
    Compute correlation between isotope traces within peak window.
    
    Args:
        intensity_matrix: numpy array of shape (n_scans, n_isos)
        start_idx: Start index for peak window
        end_idx: End index for peak window
    
    Returns:
        Average pairwise correlation coefficient
    """
    if end_idx <= start_idx or start_idx < 0 or end_idx > intensity_matrix.shape[0]:
        return 0.0
    
    window_matrix = intensity_matrix[start_idx:end_idx, :]
    
    # Remove isotopes with all zeros
    non_zero_cols = []
    for col_idx in range(window_matrix.shape[1]):
        if np.any(window_matrix[:, col_idx] > 0):
            non_zero_cols.append(col_idx)
    
    if len(non_zero_cols) < 2:
        return 0.0
    
    # Compute pairwise correlations (require non-zero std to avoid divide-by-zero in corrcoef)
    MIN_STD = 1e-12
    correlations = []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for i in range(len(non_zero_cols)):
            for j in range(i + 1, len(non_zero_cols)):
                col1 = window_matrix[:, non_zero_cols[i]]
                col2 = window_matrix[:, non_zero_cols[j]]
                std1, std2 = np.std(col1), np.std(col2)
                if std1 > MIN_STD and std2 > MIN_STD:
                    with np.errstate(invalid='ignore', divide='ignore'):
                        corr = np.corrcoef(col1, col2)[0, 1]
                    if not np.isnan(corr) and not np.isinf(corr):
                        correlations.append(float(corr))
    
    return np.mean(correlations) if correlations else 0.0

def compute_shape_similarity(intensity_matrix, total_intensities, start_idx, end_idx):
    """
    Compute shape similarity: correlation of each isotope trace to summed trace (Skyline metric B).
    Uses z-scored window to reduce scale issues.
    
    Args:
        intensity_matrix: numpy array of shape (n_scans, n_isos)
        total_intensities: numpy array of summed intensities (should be same processed version used for detection)
        start_idx: Start index for peak window
        end_idx: End index for peak window
    
    Returns:
        Mean correlation coefficient
    """
    if end_idx <= start_idx or start_idx < 0 or end_idx > intensity_matrix.shape[0]:
        return 0.0
    
    window_matrix = intensity_matrix[start_idx:end_idx, :]
    window_total = total_intensities[start_idx:end_idx]
    
    MIN_STD = 1e-12
    std_total = np.std(window_total)
    if std_total <= MIN_STD:
        return 0.0
    
    # Z-score the total trace for better correlation
    window_total_z = (window_total - np.mean(window_total)) / (std_total + MIN_STD)
    
    correlations = []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for iso_idx in range(window_matrix.shape[1]):
            iso_trace = window_matrix[:, iso_idx]
            std_iso = np.std(iso_trace)
            if std_iso > MIN_STD and np.max(iso_trace) > 0:
                # Z-score isotope trace too
                iso_trace_z = (iso_trace - np.mean(iso_trace)) / (std_iso + MIN_STD)
                # Avoid corrcoef divide-by-zero when z-scored array is constant (numerical edge case)
                std_iso_z = np.std(iso_trace_z)
                std_total_z = np.std(window_total_z)
                if std_iso_z > MIN_STD and std_total_z > MIN_STD:
                    with np.errstate(invalid='ignore', divide='ignore'):
                        corr = np.corrcoef(iso_trace_z, window_total_z)[0, 1]
                    if not np.isnan(corr) and not np.isinf(corr):
                        correlations.append(float(corr))
    
    return np.mean(correlations) if correlations else 0.0

def compute_isotope_ratio_stability(intensity_matrix, start_idx, end_idx, eps=1e-12):
    """
    Compute isotope ratio stability: consistency of ratios across scans within peak (Skyline metric C).
    True stability = do isotope ratios stay constant as the peak elutes?
    
    Returns 0 when: window too small, too few scans above M0 noise threshold, or no isotope
    has enough non-zero ratio points (M+1/M0, M+2/M0, ...). Relaxed to allow 2+ valid scans
    and 2+ ratio points per isotope so narrow or noisy peaks still get a non-zero score.
    
    Args:
        intensity_matrix: numpy array of shape (n_scans, n_isos)
        start_idx: Start index for peak window
        end_idx: End index for peak window
        eps: Small epsilon to avoid division by zero
    
    Returns:
        Score (0-1, higher = more stable ratios across scans)
    """
    if end_idx <= start_idx or start_idx < 0 or end_idx > intensity_matrix.shape[0]:
        return 0.0
    
    W = intensity_matrix[start_idx:end_idx, :]
    if W.shape[0] < 2 or W.shape[1] < 2:
        return 0.0
    
    # Get monoisotopic (M) trace
    m = W[:, 0]
    
    # Only consider scans where M is above noise threshold
    valid = m > np.percentile(m[m > 0], 20) if np.any(m > 0) else np.zeros_like(m, dtype=bool)
    if np.sum(valid) < 2:
        return 0.0
    
    # Compute per-scan ratios to M for each isotope (M+1, M+2, ...)
    ratios = []
    for j in range(1, W.shape[1]):
        r = W[valid, j] / (m[valid] + eps)
        # Ignore scans where isotope is basically zero
        r = r[r > 0]
        # Need at least 2 ratio points to compute CV (was 3; many peaks had 0 due to sparse isotope signal)
        if len(r) >= 2:
            # Coefficient of variation of this isotope's ratio across scans
            cv = np.std(r) / (np.mean(r) + eps)
            ratios.append(cv)
    
    if not ratios:
        return 0.0
    
    # Average CV across isotopes (lower = more stable)
    mean_cv = float(np.mean(ratios))
    
    # Score: exp(-CV) (penalize high variation)
    score = float(np.exp(-mean_cv))
    
    return score

def compute_isotope_score(intensity_matrix, start_idx, end_idx):
    """
    Compute isotope dot-product score (observed vs expected pattern).
    Simple version: compare relative intensities.
    
    Args:
        intensity_matrix: numpy array of shape (n_scans, n_isos)
        start_idx: Start index for peak window
        end_idx: End index for peak window
    
    Returns:
        Isotope score (0-1, higher is better)
    """
    if end_idx <= start_idx or start_idx < 0 or end_idx > intensity_matrix.shape[0]:
        return 0.0
    
    window_matrix = intensity_matrix[start_idx:end_idx, :]
    
    # Sum intensities across RT for each isotope
    isotope_sums = np.sum(window_matrix, axis=0)
    total_sum = np.sum(isotope_sums)
    
    if total_sum == 0:
        return 0.0
    
    # Normalize to get observed pattern
    observed_pattern = isotope_sums / total_sum
    
    # Expected pattern: M > M+1 > M+2 > M+3 (simplified, could use theoretical)
    n_isos = len(isotope_sums)
    expected_pattern = np.array([0.5, 0.3, 0.15, 0.05][:n_isos])
    expected_pattern = expected_pattern / np.sum(expected_pattern)
    
    # Dot product (cosine similarity)
    dot_product = np.dot(observed_pattern, expected_pattern)
    norm_obs = np.linalg.norm(observed_pattern)
    norm_exp = np.linalg.norm(expected_pattern)
    
    if norm_obs == 0 or norm_exp == 0:
        return 0.0
    
    score = dot_product / (norm_obs * norm_exp)
    return float(score)


def _expand_peak_boundaries_log_noise(rts, intensities, apex_idx, left_idx, right_idx,
                                       epsilon=5.0, k_sigma=2.0, use_linear_cap=True, slope_min=0.02):
    """
    Expand peak boundaries using noise-threshold crossing in log space (HDX-friendly).
    Log scale reveals low-intensity tails; this uses baseline + k*sigma in log space
    so boundaries are not chosen visually on log but algorithmically.

    Steps:
    1. logI = log10(intensity + epsilon) to avoid log(0).
    2. Estimate baseline and sigma from flanking regions (excluding current peak).
    3. Cutoff = baseline_log + k*sigma; walk left/right until logI < cutoff.
    4. Optional: cap with linear sanity (intensity > linear_baseline + 3*sigma_linear).
    5. Slope persistence: stop only when intensity is noise-like AND local |d(logI)/dt| < slope_min
       (so real tails that decay smoothly are kept; baseline chatter is not).

    Args:
        rts: 1D array of RT (seconds)
        intensities: 1D array of intensities (baseline-corrected, same length as rts)
        apex_idx: index of peak apex
        left_idx: current left boundary index
        right_idx: current right boundary index
        epsilon: added to intensity before log (default 5, ~1-10 counts)
        k_sigma: cutoff = baseline_log + k_sigma * sigma_log (2=inclusive, 3=conservative, 4=Skyline-like)
        use_linear_cap: if True, also require linear I > linear_baseline + 3*sigma (prevent absurd expansion)
        slope_min: min |d(logI)/dt| (log10 units per second) to consider "still peak"; below this + below cutoff => stop (default 0.02)

    Returns:
        (new_left_idx, new_right_idx) (may be same or expanded)
    """
    n = len(intensities)
    if n == 0 or apex_idx < 0 or apex_idx >= n:
        return left_idx, right_idx
    rts = np.asarray(rts, dtype=float)
    y = np.asarray(intensities, dtype=float)
    y_safe = np.maximum(y, 0) + epsilon
    logI = np.log10(y_safe)

    # Flanking regions: exclude [left_idx, right_idx] so we don't use peak itself for baseline
    exclude_left = max(0, left_idx)
    exclude_right = min(n - 1, right_idx)
    mask = np.ones(n, dtype=bool)
    mask[exclude_left : exclude_right + 1] = False
    flank = logI[mask]
    if len(flank) < 5:
        return left_idx, right_idx
    baseline_log = float(np.median(flank))
    mad = np.median(np.abs(flank - np.median(flank)))
    sigma_log = 1.4826 * mad if mad > 0 else 0.1
    cutoff_log = baseline_log + k_sigma * sigma_log

    # Optional linear cap: baseline and sigma from same flanking region (linear space)
    if use_linear_cap:
        flank_linear = y[mask]
        flank_linear = flank_linear[flank_linear >= 0]
        if len(flank_linear) >= 5:
            baseline_linear = float(np.median(flank_linear))
            mad_linear = np.median(np.abs(flank_linear - np.median(flank_linear)))
            sigma_linear = 1.4826 * mad_linear if mad_linear > 0 else 0
            linear_floor = baseline_linear + 3.0 * sigma_linear
        else:
            linear_floor = 0.0
    else:
        linear_floor = 0.0

    # Helper: |d(logI)/dt| at i when walking toward apex. Left walk: slope from i to i+1; right walk: slope from i-1 to i.
    def slope_left(i):
        if i + 1 >= n:
            return 0.0
        dt = rts[i + 1] - rts[i]
        if dt <= 0:
            return 0.0
        return abs(logI[i + 1] - logI[i]) / dt

    def slope_right(i):
        if i - 1 < 0:
            return 0.0
        dt = rts[i] - rts[i - 1]
        if dt <= 0:
            return 0.0
        return abs(logI[i] - logI[i - 1]) / dt

    # Walk left from current left boundary. Stop only when (below noise or linear cap) AND slope flat.
    new_left = 0
    for i in range(left_idx, -1, -1):
        below_noise = logI[i] < cutoff_log
        hit_linear_cap = use_linear_cap and linear_floor > 0 and y[i] <= linear_floor
        flat = slope_left(i) < slope_min
        if (below_noise or hit_linear_cap) and flat:
            new_left = i + 1
            break
        new_left = i
    new_left = max(0, new_left)

    # Walk right from current right boundary
    new_right = n - 1
    for i in range(right_idx, n):
        below_noise = logI[i] < cutoff_log
        hit_linear_cap = use_linear_cap and linear_floor > 0 and y[i] <= linear_floor
        flat = slope_right(i) < slope_min
        if (below_noise or hit_linear_cap) and flat:
            new_right = i - 1
            break
        new_right = i
    new_right = min(n - 1, new_right)

    return new_left, new_right


def has_log_plateau(rts, intensities, left_idx, right_idx, apex_idx,
                    min_points_near_top=1, rel_height=0.3, eps=1e-10):
    """
    True if the peak has at least min_points_near_top points in the "near top" band on log scale (real peak);
    False if too few points (e.g. single-point noise spike).
    Relaxed defaults: min_points_near_top=1 (allow single-point peaks), rel_height=0.3 (wide band: 70%% below max).
    """
    if left_idx is None or right_idx is None or apex_idx is None:
        return False
    left_idx = int(left_idx)
    right_idx = int(right_idx)
    apex_idx = int(apex_idx)
    if left_idx >= right_idx or apex_idx < left_idx or apex_idx > right_idx:
        return False
    rts = np.asarray(rts, dtype=float)
    intensities = np.asarray(intensities, dtype=float)
    n = len(rts)
    if right_idx >= n or left_idx < 0:
        return False
    I_slice = intensities[left_idx:right_idx + 1]
    if np.max(I_slice) <= 0:
        return False
    log_I = np.log10(I_slice + eps)
    max_log = np.max(log_I)
    min_log = np.min(log_I[I_slice > 0]) if np.any(I_slice > 0) else max_log
    thresh = max_log - (1.0 - rel_height) * (max_log - min_log) if max_log > min_log else max_log
    above = log_I >= thresh
    n_near_top = np.sum(above)
    return n_near_top >= min_points_near_top


def detect_chromatographic_peaks_windowed(rts, intensities, rt_anchor, window_sec=180.0,
                                          max_peaks=10, min_prominence=None, min_width_scans=2):
    """
    Detect chromatographic peaks within a windowed region around anchor RT.
    Used by the chromatogram extraction step (Step 6). For single-peak use, call find_collection_window_boundaries.
    This addresses gappy/sparse XIC traces by:
    - Restricting search to local RT window (not entire run)
    - Using relaxed width constraints (handles narrow peaks)
    - Handling flat-topped peaks after smoothing
    - Expanding boundaries via log-space noise threshold (baseline + k*sigma in log(I+eps)),
      with slope persistence (stop only when below cutoff AND |d(logI)/dt| < slope_min) and optional linear cap,
      to capture HDX-relevant tails without including baseline chatter
    
    Args:
        rts: numpy array of RT values (seconds)
        intensities: numpy array of intensities (summed trace, baseline-corrected, smoothed)
        rt_anchor: Anchor RT around which to search (seconds)
        window_sec: Search window size in seconds (default 180s = 3 minutes)
        max_peaks: Maximum number of peaks to return
        min_prominence: Minimum peak prominence (None = auto from window data)
        min_width_scans: Minimum peak width in scans (default 2, very relaxed for gappy data)
    
    Returns:
        List of peak candidates, each with:
        {
            'apex_idx': int (global index),
            'apex_rt': float,
            'apex_intensity': float,
            'left_base_idx': int (global index),
            'right_base_idx': int (global index),
            'left_base_rt': float,
            'right_base_rt': float,
            'prominence': float,
            'width_scans': float
        }
    """
    if len(rts) == 0 or len(intensities) == 0 or np.max(intensities) <= 0:
        return []
    
    if rt_anchor is None:
        # Fallback: use median RT if no anchor provided
        rt_anchor = np.median(rts)
    
    # Restrict to a local window around anchor (Skyline-style)
    w0 = rt_anchor - window_sec / 2.0
    w1 = rt_anchor + window_sec / 2.0
    mask = (rts >= w0) & (rts <= w1)
    
    if np.sum(mask) < 5:
        # Very sparse data - use fallback: find max in window
        if np.sum(mask) > 0:
            y_w = intensities[mask]
            max_idx_local = np.argmax(y_w)
            global_indices = np.where(mask)[0]
            apex_idx_global = int(global_indices[max_idx_local])
            # Use fixed width around apex
            half_width = min(5, len(global_indices) // 2)
            left_global = max(0, apex_idx_global - half_width)
            right_global = min(len(rts) - 1, apex_idx_global + half_width)
            return [{
                'apex_idx': apex_idx_global,
                'apex_rt': float(rts[apex_idx_global]),
                'apex_intensity': float(intensities[apex_idx_global]),
                'left_base_idx': left_global,
                'right_base_idx': right_global,
                'left_base_rt': float(rts[left_global]),
                'right_base_rt': float(rts[right_global]),
                'collection_left_base_rt': float(rts[left_global]),
                'collection_right_base_rt': float(rts[right_global]),
                'prominence': float(intensities[apex_idx_global]),
                'width_scans': float(half_width * 2)
            }]
        return []
    
    rts_w = rts[mask]
    y_w = intensities[mask]
    
    # Robust prominence: scale to local window, not whole run
    # Use MAD or low percentile for noise (not high percentile which can be signal)
    if min_prominence is None:
        max_local = np.max(y_w)
        if max_local <= 0:
            return []
        
        # Robust noise estimation: use median or low percentile (true chromatograms are mostly baseline)
        # If many points are nonzero due to interference, percentile(10) can still be signal
        # Use median of all points (including zeros) or 25th percentile
        noise = np.percentile(y_w, 25)  # 25th percentile (more robust than 10th)
        # Cap noise estimate - if it's too high relative to max, it's probably not noise
        noise = min(noise, max_local * 0.1)
        
        # Prominence threshold: smaller of relative (1% of max) or absolute (2x noise)
        relative_prominence = max_local * 0.01
        absolute_prominence = 2.0 * noise
        min_prominence = min(relative_prominence, absolute_prominence)
    
    # Find peaks with relaxed width and plateau handling
    try:
        peaks, props = signal.find_peaks(
            y_w,
            prominence=min_prominence,
            width=min_width_scans,
            plateau_size=1,  # Handle flat-topped peaks after smoothing
            distance=max(2, min_width_scans)  # Minimum distance between peaks
        )
    except Exception:
        # Fallback if prominence/width calculation fails
        peaks, props = signal.find_peaks(
            y_w,
            height=np.max(y_w) * 0.01,
            distance=2
        )
        if len(peaks) == 0:
            return []
    
    if len(peaks) == 0:
        return []
    
    # Pick top N by height
    heights = y_w[peaks]
    order = np.argsort(heights)[::-1][:max_peaks]
    peaks_ordered = peaks[order]
    
    # Reorder props arrays to match peaks_ordered (fix indexing bug)
    if 'prominences' in props:
        prominences_ordered = props['prominences'][order]
    else:
        prominences_ordered = heights[order] - min_prominence
    
    # Use peak_widths for boundaries: integration at 50% prominence, collection at 80% (wider, captures tails)
    try:
        widths_int, _, left_ips_int, right_ips_int = signal.peak_widths(y_w, peaks_ordered, rel_height=REL_HEIGHT_INTEGRATION)
        widths_col, _, left_ips_col, right_ips_col = signal.peak_widths(y_w, peaks_ordered, rel_height=REL_HEIGHT_COLLECTION)
        widths = widths_int
    except Exception:
        widths = np.full(len(peaks_ordered), min_width_scans * 2)
        left_ips_int = peaks_ordered - min_width_scans
        right_ips_int = peaks_ordered + min_width_scans
        left_ips_col = peaks_ordered - min_width_scans
        right_ips_col = peaks_ordered + min_width_scans
    
    # Map local indices back to global indices
    global_idxs = np.where(mask)[0]
    
    def _compute_boundaries(li_init, ri_init, apex_idx_local, apex_rt_local):
        """Walk + log expansion from initial boundaries; return (li, ri) in local indices."""
        li, ri = int(np.floor(li_init)), int(np.ceil(ri_init))
        li = max(0, li)
        ri = min(len(rts_w) - 1, ri)
        apex_intensity = y_w[apex_idx_local]
        noise_estimate = np.percentile(y_w[y_w > 0], 10) if np.any(y_w > 0) else 0
        baseline_threshold = max(
            np.percentile(y_w, 10),
            apex_intensity * PEAK_BOUNDARY_APEX_FRAC,
            PEAK_BOUNDARY_NOISE_MULTIPLIER * noise_estimate
        )
        while li > 0:
            if li < len(y_w) - 1 and y_w[li] < y_w[li - 1] and y_w[li] < y_w[li + 1]:
                break
            if y_w[li] <= baseline_threshold:
                break
            li -= 1
        while ri < len(rts_w) - 1:
            if ri > 0 and y_w[ri] < y_w[ri - 1] and y_w[ri] < y_w[ri + 1]:
                break
            if y_w[ri] <= baseline_threshold:
                break
            ri += 1
        li, ri = _expand_peak_boundaries_log_noise(rts_w, y_w, apex_idx_local, li, ri,
            epsilon=5.0, k_sigma=1.5, use_linear_cap=True, slope_min=0.01)
        max_half_span_sec = 90.0
        li_min = np.searchsorted(rts_w, apex_rt_local - max_half_span_sec, side='left')
        ri_max = np.searchsorted(rts_w, apex_rt_local + max_half_span_sec, side='right') - 1
        li = max(li, li_min, 0)
        ri = min(ri, ri_max, len(rts_w) - 1)
        return max(0, li), min(len(rts_w) - 1, ri)
    
    candidates = []
    for i, p in enumerate(peaks_ordered):
        apex_idx_local = int(p)
        apex_rt_local = float(rts_w[apex_idx_local])
        
        li, ri = _compute_boundaries(left_ips_int[i], right_ips_int[i], apex_idx_local, apex_rt_local)
        li_col, ri_col = _compute_boundaries(left_ips_col[i], right_ips_col[i], apex_idx_local, apex_rt_local)
        
        apex_idx_global = int(global_idxs[apex_idx_local])
        left_global = int(global_idxs[li])
        right_global = int(global_idxs[ri])
        left_col_global = int(global_idxs[li_col])
        right_col_global = int(global_idxs[ri_col])
        
        prominence_val = float(prominences_ordered[i])
        
        candidates.append({
            'apex_idx': apex_idx_global,
            'apex_rt': float(rts[apex_idx_global]),
            'apex_intensity': float(intensities[apex_idx_global]),
            'left_base_idx': left_global,
            'right_base_idx': right_global,
            'left_base_rt': float(rts[left_global]),
            'right_base_rt': float(rts[right_global]),
            'collection_left_base_rt': float(rts[left_col_global]),
            'collection_right_base_rt': float(rts[right_col_global]),
            'prominence': prominence_val,
            'width_scans': float(widths[i])
        })
    
    return candidates


def detect_chromatographic_peaks(rts, intensities, min_prominence=None, min_width=None, 
                                  max_peaks=10, rt_anchor=None, rt_prior_sigma=30.0):
    """
    Detect multiple chromatographic peaks on summed trace (Skyline-style).
    Uses scipy.signal.find_peaks with prominence/width detection.
    
    NOTE: This function is kept for backward compatibility but detect_chromatographic_peaks_windowed
    is preferred for gappy/sparse XIC data.
    
    Args:
        rts: numpy array of RT values (seconds)
        intensities: numpy array of intensities (summed trace, baseline-corrected, smoothed)
        min_prominence: Minimum peak prominence (None = auto from data)
        min_width: Minimum peak width in scans (None = auto)
        max_peaks: Maximum number of peaks to return
        rt_anchor: Optional anchor RT (for RT prior scoring, not peak detection)
        rt_prior_sigma: Sigma for RT prior gaussian (seconds)
    
    Returns:
        List of peak candidates, each with:
        {
            'apex_idx': int,
            'apex_rt': float,
            'apex_intensity': float,
            'left_base_idx': int,
            'right_base_idx': int,
            'left_base_rt': float,
            'right_base_rt': float,
            'prominence': float,
            'width_scans': float
        }
    """
    if len(rts) == 0 or len(intensities) == 0 or np.max(intensities) <= 0:
        return []
    
    # Auto-determine prominence if not provided
    if min_prominence is None:
        max_intensity = np.max(intensities)
        
        # Better noise estimate: use median of lower 50% of non-zero values, or use a wider window
        non_zero = intensities[intensities > 0]
        if len(non_zero) > 10:
            # Use lower quartile for noise estimate (more robust than 10th percentile)
            noise = np.percentile(non_zero, 25)
            # Cap noise estimate - if it's too high relative to max, it's probably not noise
            noise = min(noise, max_intensity * 0.1)  # Cap at 10% of max
        else:
            # Very sparse data - use a small fraction of max as noise estimate
            noise = max_intensity * 0.01
        
        # Use relative prominence: smaller of absolute threshold or relative to max
        # This handles both high-intensity and low-intensity peaks better
        relative_prominence = max_intensity * 0.01  # 1% of max
        absolute_prominence = 2.0 * noise
        min_prominence = min(relative_prominence, absolute_prominence) if max_intensity > 0 else 0
    
    # Auto-determine width if not provided
    if min_width is None:
        if len(rts) > 1:
            dt = np.median(np.diff(rts))
            # Use 5 seconds minimum (was 10 sec) - better for sparse data
            min_width = max(3, int(5.0 / dt))  # >=5 sec, minimum 3 scans
        else:
            min_width = 3
    
    # Find all peaks using scipy
    peaks_all, props = signal.find_peaks(
        intensities,
        prominence=min_prominence,
        width=min_width,
        distance=min_width
    )
    
    if len(peaks_all) == 0:
        return []
    
    # Choose top peaks by height
    heights = intensities[peaks_all]
    order = np.argsort(heights)[::-1][:max_peaks]
    peaks = peaks_all[order]
    
    # Compute widths/bounds at fixed relative height (10% of peak height) - captures full elution including tails
    widths, width_heights, left_ips, right_ips = signal.peak_widths(
        intensities, peaks, rel_height=0.1
    )
    
    # Build candidate peaks with boundaries
    candidates = []
    for i, apex_idx in enumerate(peaks):
        apex_idx = int(apex_idx)
        li = int(np.floor(left_ips[i]))
        ri = int(np.ceil(right_ips[i]))
        li = max(0, li)
        ri = min(len(rts) - 1, ri)
        
        # Get prominence from original peaks_all order
        prominence_idx = np.where(peaks_all == apex_idx)[0][0]
        peak_prominence = float(props['prominences'][prominence_idx])
        
        candidates.append({
            'apex_idx': apex_idx,
            'apex_rt': float(rts[apex_idx]),
            'apex_intensity': float(intensities[apex_idx]),
            'left_base_idx': li,
            'right_base_idx': ri,
            'left_base_rt': float(rts[li]),
            'right_base_rt': float(rts[ri]),
            'prominence': peak_prominence,
            'width_scans': float(widths[i])
        })
    
    return candidates

def select_best_rt_cluster(clusters, group_data=None, rt_anchor=None):
    """
    Select the best RT cluster from multiple clusters.
    
    Criteria (in order):
    1. Cluster nearest to anchor RT (if provided)
    2. Cluster with most PSMs
    3. Cluster with highest median MS1 intensity (if available)
    4. Cluster with best scores (if available)
    5. Largest cluster (fallback)
    
    Args:
        clusters: list of RT clusters (each is a list of RTs)
        group_data: Optional DataFrame with quality metrics (e-value, intensity, etc.)
        rt_anchor: Optional anchor RT (feature apex, chrom apex, or median) to guide selection
    
    Returns:
        Best cluster (list of RTs)
    """
    if not clusters:
        return []
    
    if len(clusters) == 1:
        return clusters[0]
    
    # If anchor RT provided, prioritize cluster nearest to anchor
    if rt_anchor is not None:
        best_cluster = None
        min_distance = float('inf')
        
        for cluster in clusters:
            # Calculate distance from anchor to cluster center
            cluster_center = np.median(cluster)
            distance = abs(cluster_center - rt_anchor)
            
            if distance < min_distance:
                min_distance = distance
                best_cluster = cluster
        
        if best_cluster is not None:
            return best_cluster
    
    # If we have quality metrics, use them to select best cluster
    if group_data is not None and len(group_data) > 0:
        best_cluster = None
        best_score = -1
        
        for cluster in clusters:
            # Score based on: number of PSMs, median intensity, best e-value
            cluster_rt_set = set(cluster)
            
            # Find rows matching this cluster
            cluster_rows = group_data[
                group_data['MS1_retention_time_sec'].apply(
                    lambda rt: rt in cluster_rt_set if pd.notna(rt) else False
                )
            ]
            
            if len(cluster_rows) == 0:
                continue
            
            # Score: prioritize more PSMs, higher intensity, better e-value
            num_psms = len(cluster_rows)
            
            # Median MS1 intensity
            intensities = cluster_rows['MS1_retention_time_intensity'].dropna()
            median_intensity = intensities.median() if len(intensities) > 0 else 0
            
            # Best (lowest) e-value
            evalues = cluster_rows['e-value'].dropna()
            best_evalue = evalues.min() if len(evalues) > 0 else 1.0
            
            # Composite score: more PSMs + higher intensity + better e-value
            # Normalize: intensity/1e6, e-value inverted (1/e-value)
            score = num_psms + (median_intensity / 1e6) + (1.0 / (best_evalue + 1e-10))
            
            if score > best_score:
                best_score = score
                best_cluster = cluster
        
        if best_cluster is not None:
            return best_cluster
    
    # Fallback: largest cluster
    return max(clusters, key=len)

def find_peak_boundaries(chromatogram, ms1_rts, group_data=None, rt_anchor=None, window_size=30.0, gap_threshold=60.0):
    """
    Find peak boundaries using RT clustering to identify distinct elution peaks.
    Selects the best cluster based on anchor RT proximity and quality metrics.
    
    Args:
        chromatogram: list of (rt, intensity) tuples (not used but kept for compatibility)
        ms1_rts: list of MS1 retention times from the table
        group_data: Optional DataFrame with quality metrics for cluster selection
        rt_anchor: Optional anchor RT (feature apex, chrom apex, or median) to guide cluster selection
        window_size: minimum window size in seconds (default 30 seconds)
        gap_threshold: minimum gap in seconds to split into different clusters
    
    Returns:
        (min_rt, max_rt) tuple for the best cluster, or (None, None) if no MS1 RTs found
    """
    if not ms1_rts:
        return None, None
    
    # Cluster RTs to find distinct elution peaks
    clusters = cluster_rts(ms1_rts, gap_threshold)
    
    if not clusters:
        return None, None
    
    # Select best cluster based on anchor RT proximity and quality metrics
    primary_cluster = select_best_rt_cluster(clusters, group_data, rt_anchor=rt_anchor)
    
    if not primary_cluster:
        return None, None
    
    # Find the range of MS1 RTs in primary cluster
    min_ms1_rt = min(primary_cluster)
    max_ms1_rt = max(primary_cluster)
    ms1_rt_range = max_ms1_rt - min_ms1_rt
    
    # Create window that encompasses the primary cluster
    if ms1_rt_range >= window_size:
        # Add small padding (5 seconds on each side)
        padding = 5.0
        peak_start = min_ms1_rt - padding
        peak_end = max_ms1_rt + padding
    else:
        # Center a 30-second window on the cluster
        center_rt = (min_ms1_rt + max_ms1_rt) / 2.0
        half_window = window_size / 2.0
        peak_start = center_rt - half_window
        peak_end = center_rt + half_window
        
        # Ensure all MS1 RTs in cluster are within the window
        if min_ms1_rt < peak_start:
            peak_end = peak_end + (peak_start - min_ms1_rt)
            peak_start = min_ms1_rt
        if max_ms1_rt > peak_end:
            peak_end = max_ms1_rt
    
    return peak_start, peak_end


# Filtering criteria string for plot titles (same as in run_chromatograms)
FILTERING_CRITERIA_PLOT = (
    "XIC ppm=20 | anchor mz/rt=20ppm/60s | pre-filter: (no E-value threshold) | "
    "score=0.28 coel + 0.28 shape + 0.14 ratio + 0.12 RT_cov + 0.05 rt_prior + 0.05 quality + 0.20 strength | "
    "accept: M0, M+1, M+2, M+3 present; M0 and M+1 > M+2 and M+3 | PSM: q≤0.05, PEP≤0.05"
)


def _break_chromatogram_at_gaps(rt, I, gap_factor=2.5):
    """Insert NaNs at MS1 gaps so matplotlib breaks the line (no false valleys).
    rt, I are arrays (I can be 1d or 2d). Returns (rt, I_plot)."""
    rt = np.asarray(rt, dtype=float)
    I = np.asarray(I, dtype=float)
    if rt.size < 2:
        return rt, I.copy()
    dt = np.diff(rt)
    med = np.median(dt)
    if med <= 0:
        return rt, I.copy()
    gap_thresh = gap_factor * med
    I_plot = I.copy()
    mask = dt > gap_thresh
    if I_plot.ndim == 1:
        I_plot[1:][mask] = np.nan
    else:
        I_plot[1:][mask, :] = np.nan
    return rt, I_plot


def _draw_peptide_figure_from_dataframe(row, rts, intensity_matrix, df_all, output_path, is_rejected, rejection_reason):
    """Create and save one peptide figure from dataframe row and stored traces. Uses same style as chromatograms_peptides (black bg, integration/collection windows)."""
    min_rt = row.get('detected_peak_min_rt') or row.get('collection_min_rt')
    max_rt = row.get('detected_peak_max_rt') or row.get('collection_max_rt')
    collection_min_rt = row.get('collection_min_rt')
    collection_max_rt = row.get('collection_max_rt')
    peptide_key = row.get('peptide_key', row.get('peptide', ''))
    if is_rejected and rejection_reason:
        peptide_key = f"{peptide_key} [REJECTED: {rejection_reason}]"
    fig = create_chromatogram_subplot_figure(
        row, rts, intensity_matrix,
        min_rt=min_rt, max_rt=max_rt,
        collection_min_rt=collection_min_rt, collection_max_rt=collection_max_rt,
        peptide_key=peptide_key,
        figsize=(12, 5)
    )
    try:
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='black', pad_inches=0.25)
        print(f"[DEBUG] Plot-from-dataframe: saved -> {os.path.abspath(output_path)}")
    except Exception as e:
        print(f"[ERROR] Failed to save plot-from-dataframe: {os.path.abspath(output_path)}: {e}", file=sys.stderr)
    finally:
        plt.close(fig)


def _get_precursor_mz_from_row(row):
    """Get monoisotopic m/z from a row. Tries mz_theoretical, theoretical_mz, mz, exp_mz, calc_neutral_mass/charge."""
    for col in ('mz_theoretical', 'theoretical_mz', 'mz', 'exp_mz'):
        if col in row.index:
            v = row.get(col)
            if pd.notna(v) and v != '' and float(v) > 0:
                return float(v)
    if 'calc_neutral_mass' in row.index and 'charge' in row.index:
        return calculate_theoretical_mz(row.get('calc_neutral_mass'), row.get('charge'))
    return None


def create_chromatogram_subplot_figure(row, rts, intensity_matrix, min_rt=None, max_rt=None,
                                      collection_min_rt=None, collection_max_rt=None, peptide_key=None,
                                      figsize=(12, 5)):
    """
    Create a standalone chromatogram subplot figure (same style as chromatograms_peptides).
    Shows isotopes, total trace, integration window (green), collection window (gray), buffer window (drift).
    Uses raw data (no gap-breaking) and explicit axis limits to match chromatograms_peptides.
    Black background, black total trace, same window alpha as full workflow.
    """
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    rts = np.asarray(rts, dtype=float).flatten().copy()
    intensity_matrix = np.asarray(intensity_matrix, dtype=float).copy()
    if len(rts) < 2 or (len(rts) > 0 and np.ptp(rts) < 0.01):
        ax.text(0.5, 0.5, 'Degenerate RT vector (too few points or constant)', ha='center', va='center', transform=ax.transAxes, color='0.9', fontsize=11)
        return fig
    if intensity_matrix.ndim == 1:
        intensity_matrix = intensity_matrix.flatten()
    elif intensity_matrix.ndim == 2:
        if intensity_matrix.shape[1] > intensity_matrix.shape[0]:
            intensity_matrix = intensity_matrix.T
        intensity_matrix = np.ascontiguousarray(intensity_matrix)
    n_rts = len(rts)
    n_int = intensity_matrix.shape[0] if intensity_matrix.ndim >= 1 else len(intensity_matrix)
    if n_rts != n_int:
        n = min(n_rts, n_int)
        rts = rts[:n].copy()
        intensity_matrix = intensity_matrix[:n] if intensity_matrix.ndim == 1 else intensity_matrix[:n, :].copy()
    rts_min = rts / 60.0
    if intensity_matrix.ndim == 1:
        intensity_matrix = np.atleast_2d(intensity_matrix).T
    n_scans = intensity_matrix.shape[0]
    n_isos = intensity_matrix.shape[1] if intensity_matrix.ndim > 1 else 1
    if n_scans < 2:
        ax.text(0.5, 0.5, 'Insufficient scans (need per-scan intensities, not aggregates)', ha='center', va='center', transform=ax.transAxes, color='0.9', fontsize=11)
        return fig
    total_intensities = np.sum(intensity_matrix, axis=1)
    if len(rts_min) != len(total_intensities):
        n = min(len(rts_min), len(total_intensities))
        rts_min = rts_min[:n]
        intensity_matrix = intensity_matrix[:n, :]
        total_intensities = np.sum(intensity_matrix, axis=1)
    min_rt_min = min_rt / 60.0 if min_rt is not None else None
    max_rt_min = max_rt / 60.0 if max_rt is not None else None
    coll_min_min = collection_min_rt / 60.0 if collection_min_rt is not None else min_rt_min
    coll_max_min = collection_max_rt / 60.0 if collection_max_rt is not None else max_rt_min
    # Fallback: if stored window doesn't contain the trace's peak, use trace-derived window (fixes misaligned metrics)
    if len(total_intensities) > 0 and np.nanmax(total_intensities) > 0:
        apex_idx = int(np.nanargmax(total_intensities))
        trace_apex_rt_sec = float(rts[apex_idx]) if apex_idx < len(rts) else None
        if trace_apex_rt_sec is not None and min_rt is not None and max_rt is not None:
            if trace_apex_rt_sec < min_rt or trace_apex_rt_sec > max_rt:
                dt = np.median(np.diff(rts)) if len(rts) > 1 else 1.0
                half_span = max(15.0, dt * 15)
                min_rt = max(0.0, trace_apex_rt_sec - half_span)
                max_rt = trace_apex_rt_sec + half_span
                min_rt_min = min_rt / 60.0
                max_rt_min = max_rt / 60.0
                coll_min_min = min_rt_min
                coll_max_min = max_rt_min
                collection_min_rt = min_rt
                collection_max_rt = max_rt
    drift_min_min = (max(0.0, float(collection_min_rt or min_rt or 0) - PEAK_DRIFT_BUFFER_SEC) / 60.0) if (collection_min_rt or min_rt) is not None else None
    drift_max_min = ((float(collection_max_rt or max_rt or 0) + PEAK_DRIFT_BUFFER_SEC) / 60.0) if (collection_max_rt or max_rt) is not None else None
    # Draw window shading FIRST (zorder < 0) so it stays behind the signal.
    # Match chromatograms_peptides: COLLECTION_WINDOW_ALPHA, INTEGRATION_WINDOW_ALPHA
    if drift_min_min is not None and drift_max_min is not None and drift_max_min > drift_min_min:
        ax.axvspan(drift_min_min, drift_max_min, alpha=0.15, color='#606060', zorder=-2, label='Buffer window')
    if coll_min_min is not None and coll_max_min is not None and coll_max_min > coll_min_min:
        ax.axvspan(coll_min_min, coll_max_min, alpha=COLLECTION_WINDOW_ALPHA, color=COLLECTION_WINDOW_COLOR, zorder=-2, label='Collection window')
    if min_rt_min is not None and max_rt_min is not None and max_rt_min > min_rt_min:
        ax.axvspan(min_rt_min, max_rt_min, alpha=INTEGRATION_WINDOW_ALPHA, color='#00FF7F', zorder=-1, label='Integration window')
        ax.axvline(x=min_rt_min, color='#00CC66', linestyle='-', linewidth=4.0, alpha=1.0, zorder=5)
        ax.axvline(x=max_rt_min, color='#00CC66', linestyle='-', linewidth=4.0, alpha=1.0, zorder=5)
    # Plot isotope traces AFTER windows so signal is always on top (zorder 10+)
    # Ensure x and y are 1D contiguous arrays of same length (avoid broadcasting scalar/wrong-length)
    x_plot = np.ascontiguousarray(rts_min.ravel(), dtype=float)
    n_plot = len(x_plot)
    colors = ['#0066FF', '#9933FF', '#FF00CC', '#FF66CC', '#FF9900']
    labels_map = {0: 'M', 1: 'M+1', 2: 'M+2', 3: 'M+3', 4: 'M+4'}
    for iso_idx in range(min(intensity_matrix.shape[1], 5)):
        y_iso = np.ascontiguousarray(intensity_matrix[:, iso_idx].ravel(), dtype=float)
        n_y = min(n_plot, len(y_iso))
        ax.plot(x_plot[:n_y], y_iso[:n_y], color=colors[iso_idx % len(colors)],
                linewidth=3.0, alpha=1.0, label=labels_map.get(iso_idx, f'M+{iso_idx}'), linestyle='-', zorder=15 + iso_idx)
        ax.fill_between(x_plot[:n_y], y_iso[:n_y], alpha=0.15, color=colors[iso_idx % len(colors)], zorder=12 + iso_idx)
    # Total trace: near-black for visibility on black background (chromatograms_peptides uses #000000; slight lightening for standalone subplot)
    y_total = np.ascontiguousarray(total_intensities.ravel(), dtype=float)
    n_y = min(n_plot, len(y_total))
    ax.plot(x_plot[:n_y], y_total[:n_y], color='#222222', linewidth=3.0, alpha=1.0, label='Total', linestyle='-', zorder=25)
    # X-axis limits: replicate full chromatograms_peptides logic (dynamic buffer, extend for integration/anchor/apex/collection/drift)
    ms1_rts_table = np.asarray(rts, dtype=float)  # rts in seconds
    rt_anchor = row.get('anchor_rt') if hasattr(row, 'get') else None
    if rt_anchor is not None and (pd.isna(rt_anchor) or rt_anchor == ''):
        rt_anchor = None
    apex_rt = row.get('apex_rt') if hasattr(row, 'get') else None
    if apex_rt is not None and (pd.isna(apex_rt) or apex_rt == ''):
        apex_rt = None
    drift_min_rt = None
    drift_max_rt = None
    if collection_min_rt is not None and collection_max_rt is not None:
        drift_min_rt = max(0.0, float(collection_min_rt) - PEAK_DRIFT_BUFFER_SEC)
        drift_max_rt = float(collection_max_rt) + PEAK_DRIFT_BUFFER_SEC
    if (drift_min_rt is None or drift_max_rt is None) and hasattr(row, 'get'):
        if row.get('drift_min_rt') is not None and row.get('drift_max_rt') is not None:
            try:
                drift_min_rt = float(row.get('drift_min_rt'))
                drift_max_rt = float(row.get('drift_max_rt'))
            except (TypeError, ValueError):
                pass
    global_rt_min_sec = float(np.min(ms1_rts_table)) if len(ms1_rts_table) > 0 else (min_rt if min_rt is not None else 0)
    global_rt_max_sec = float(np.max(ms1_rts_table)) if len(ms1_rts_table) > 0 else (max_rt if max_rt is not None else 0)
    span_sec = global_rt_max_sec - global_rt_min_sec
    collection_span_min = (max_rt - min_rt) / 60.0 if (min_rt is not None and max_rt is not None) else span_sec / 60.0
    buffer_sec = max(90.0, span_sec * 0.25, collection_span_min * 60.0 * 0.5)
    buffer_min = buffer_sec / 60.0
    x_min = max(0, global_rt_min_sec / 60.0 - buffer_min)
    x_max = global_rt_max_sec / 60.0 + buffer_min
    if min_rt_min is not None and min_rt_min < x_min:
        x_min = max(0, min_rt_min - buffer_min * 1.5)
    if max_rt_min is not None and max_rt_min > x_max:
        x_max = max_rt_min + buffer_min * 1.5
    if rt_anchor is not None:
        anchor_rt_min = float(rt_anchor) / 60.0
        x_min = min(x_min, anchor_rt_min - buffer_min * 1.5)
        x_max = max(x_max, anchor_rt_min + buffer_min * 1.5)
    apex_rt_min = float(apex_rt) / 60.0 if apex_rt is not None else None
    if apex_rt_min is not None:
        x_min = min(x_min, apex_rt_min - buffer_min * 1.5)
        x_max = max(x_max, apex_rt_min + buffer_min * 1.5)
    if x_max <= x_min and min_rt_min is not None and max_rt_min is not None:
        x_min = max(0, min_rt_min - buffer_min)
        x_max = max_rt_min + buffer_min
    if len(rts_min) > 0:
        data_min = float(np.min(rts_min))
        data_max = float(np.max(rts_min))
        if data_max < x_min or data_min > x_max:
            x_min = max(0, data_min - buffer_min)
            x_max = data_max + buffer_min
    margin_coll_min = max(0.008, (max_rt_min - min_rt_min) * 0.02) if (min_rt_min is not None and max_rt_min is not None) else 0.01
    if coll_min_min is not None and coll_min_min < x_min:
        x_min = max(0, coll_min_min - margin_coll_min)
    if coll_max_min is not None and coll_max_min > x_max:
        x_max = coll_max_min + margin_coll_min
    if drift_min_rt is not None and drift_max_rt is not None and drift_max_rt > drift_min_rt:
        drift_min_min = drift_min_rt / 60.0
        drift_max_min = drift_max_rt / 60.0
        x_min = min(x_min, max(0, drift_min_min - buffer_min))
        x_max = max(x_max, drift_max_min + buffer_min)
    ax.set_xlim([x_min, x_max])
    # Y limits from data in view (match full path: window_mask, y_max = max * 1.1)
    if len(total_intensities) > 0:
        window_mask = (rts_min >= x_min) & (rts_min <= x_max)
        window_intensities = total_intensities[window_mask] if np.any(window_mask) else total_intensities
        if len(window_intensities) > 0 and np.nanmax(window_intensities) > 0:
            y_max = float(np.nanmax(window_intensities)) * 1.1
            ax.set_ylim([0, y_max])
        elif np.nanmax(total_intensities) > 0:
            y_max = float(np.nanmax(total_intensities)) * 1.1
            ax.set_ylim([0, y_max])
        else:
            ax.set_ylim([-0.1, 0.1])
    ax.set_xlabel('Retention time (min)', fontsize=12, fontfamily='serif', color='0.9')
    ax.set_ylabel('Intensity', fontsize=12, fontfamily='serif', color='0.9')
    ax.set_title(peptide_key or row.get('peptide_key', row.get('plain_peptide', 'Chromatogram')), fontsize=12, fontweight='bold', fontfamily='serif', color='0.9')
    ax.tick_params(labelsize=10, colors='0.9')
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily('serif')
        label.set_color('0.9')
    for spine in ax.spines.values():
        spine.set_color('0.7')
    ax.legend(loc='upper right', fontsize=9, prop={'family': 'serif'}, facecolor='0.15', edgecolor='0.5', labelcolor='0.9')
    ax.grid(False)
    plt.tight_layout()
    return fig


def create_summed_ms1_spectrum_figure(row, mzml_path, min_rt=None, max_rt=None, peptide_key=None, figsize=(10, 5), return_metrics=False):
    """
    Create a standalone summed MS1 spectrum figure (same style as individual chromatogram extraction plots).
    Sums MS1 spectra across the integration/collection window and plots with isotope labels.
    """
    min_rt = min_rt or row.get('detected_peak_min_rt') or row.get('collection_min_rt')
    max_rt = max_rt or row.get('detected_peak_max_rt') or row.get('collection_max_rt')
    if min_rt is None or max_rt is None:
        return None
    min_rt, max_rt = float(min_rt), float(max_rt)
    precursor_mz = _get_precursor_mz_from_row(row)
    charge = int(row.get('charge', 1)) if pd.notna(row.get('charge')) else 1
    if precursor_mz is None or precursor_mz <= 0:
        return None
    isotope_mzs = isotope_mz_list(precursor_mz, charge, n_isos=4)
    if not isotope_mzs:
        return None
    mz_min_window = min(isotope_mzs) - 2.0
    mz_max_window = max(isotope_mzs) + 2.0
    spec_mzs, spec_ints = get_averaged_ms1_spectrum_in_window(mzml_path, min_rt, max_rt, mz_min=mz_min_window, mz_max=mz_max_window, ppm_tolerance=7.0)
    if spec_mzs is None or len(spec_mzs) == 0:
        return None
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    markerline, stemlines, baseline = ax.stem(spec_mzs, spec_ints, basefmt=' ', linefmt='#888888', markerfmt=' ')
    plt.setp(stemlines, linewidth=0.6)
    plt.setp(markerline, markersize=2)
    iso_colors = ['#0066FF', '#9933FF', '#FF00CC', '#FF66CC']
    y_max_spec = np.max(spec_ints) * 1.1 if len(spec_ints) > 0 else 1.0
    y_min_spec = 0
    matched_iso_indices = []
    observed_iso = {}
    for iso_idx, iso_mz in enumerate(isotope_mzs):
        if iso_idx >= len(iso_colors):
            break
        iso_tolerance = iso_mz * 5e-6
        iso_mask = (spec_mzs >= iso_mz - iso_tolerance) & (spec_mzs <= iso_mz + iso_tolerance)
        if np.any(iso_mask):
            iso_mzs_obs = spec_mzs[iso_mask]
            iso_intensities = spec_ints[iso_mask]
            max_idx_local = int(np.argmax(iso_intensities)) if len(iso_intensities) > 0 else 0
            max_iso_intensity = float(iso_intensities[max_idx_local]) if len(iso_intensities) > 0 else 0.0
            obs_mz = float(iso_mzs_obs[max_idx_local]) if len(iso_mzs_obs) > 0 else float(iso_mz)
            ppm_err = ((obs_mz - iso_mz) / iso_mz) * 1e6 if iso_mz else np.nan
            matched_iso_indices.append(iso_idx)
            observed_iso[iso_idx] = {
                'theoretical_mz': float(iso_mz),
                'observed_mz': obs_mz,
                'max_intensity': max_iso_intensity,
                'ppm_error': float(ppm_err) if not np.isnan(ppm_err) else np.nan,
            }
            line_color = iso_colors[iso_idx % len(iso_colors)]
            ax.vlines(iso_mz, 0, max_iso_intensity, colors=line_color, linewidth=2.5, alpha=0.95, zorder=20,
                     label=f'M+{iso_idx}')
            # Label each matched isotope with ppm error based on highest-intensity peak within +/-5 ppm.
            if max_iso_intensity > 0 and not np.isnan(ppm_err):
                y_text = min(y_max_spec * 0.97, max_iso_intensity * 1.05 + y_max_spec * 0.02)
                ax.text(
                    iso_mz,
                    y_text,
                    f"M+{iso_idx} {ppm_err:+.1f} ppm",
                    ha='center',
                    va='bottom',
                    fontsize=9,
                    color=line_color,
                    fontfamily='serif',
                    zorder=30,
                )
    ax.set_xlabel('m/z', fontsize=15, fontfamily='serif', color='0.85')
    ax.set_ylabel('Intensity', fontsize=15, fontfamily='serif', color='0.85')
    title_rt_range = f'{min_rt:.1f}-{max_rt:.1f}s'
    ax.set_title(f'Summed MS1 Spectrum (RT={title_rt_range})', fontsize=15, fontweight='bold', fontfamily='serif', color='0.9')
    ax.tick_params(labelsize=14, colors='0.85')
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily('serif')
        label.set_color('0.85')
    for spine in ax.spines.values():
        spine.set_color('0.6')
    ax.grid(False)
    if matched_iso_indices:
        ax.legend(loc='upper right', fontsize=11, prop={'family': 'serif'}, facecolor='#1a1a1a', edgecolor='0.6', labelcolor='0.9')
    ax.set_xlim([mz_min_window, mz_max_window])
    ax.set_ylim([y_min_spec, y_max_spec])
    ax.text(ax.get_xlim()[0] + 0.02 * (ax.get_xlim()[1] - ax.get_xlim()[0]),
            y_max_spec * 0.95, 'Isotope matching: ±5 ppm', fontsize=12, ha='left', va='top', fontfamily='serif',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='lightblue', alpha=0.8, edgecolor='black', linewidth=0.8), zorder=25)
    plt.tight_layout()
    if return_metrics:
        m0 = observed_iso.get(0, {}).get('max_intensity', 0.0)
        m1 = observed_iso.get(1, {}).get('max_intensity', 0.0)
        m2 = observed_iso.get(2, {}).get('max_intensity', 0.0)
        m3 = observed_iso.get(3, {}).get('max_intensity', 0.0)
        has_required_isotopes = all(i in observed_iso for i in (0, 1, 2, 3))
        m0_m1_gt_m2_m3 = bool(has_required_isotopes and (m0 > m2) and (m0 > m3) and (m1 > m2) and (m1 > m3))
        envelope_ok = bool(m0_m1_gt_m2_m3)
        metrics = {
            'matched_iso_indices': sorted(matched_iso_indices),
            'observed_iso': observed_iso,
            'has_required_isotopes': has_required_isotopes,
            'm0_m1_gt_m2_m3': m0_m1_gt_m2_m3 if has_required_isotopes else False,
            'm0_gt_m1_gt_m2': bool((m0 > m1) or (m1 > m2)) if has_required_isotopes else False,
            'm2_le_m0': bool(m2 <= m0) if has_required_isotopes else False,
            'passes_envelope': envelope_ok,
        }
        return fig, metrics
    return fig


def _create_overlay_figures(overlay_data, output_dir, area_vmin_global=None, area_vmax_global=None):
    """Create MS1 overlay figures (all_peptides_overlay.png, tracks_only) from overlay_data.
    Used by run_plots_from_dataframe (Generate plots) and run_chromatograms (Extraction).
    area_vmin_global/area_vmax_global: optional; if not provided, computed from overlay_data."""
    if len(overlay_data) == 0:
        return
    overlay_data = sorted(overlay_data, key=lambda item: (item.get('sequence_start_pos', 999999), str(item.get('label', ''))))
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    from matplotlib.gridspec import GridSpec
    from matplotlib.ticker import MultipleLocator as _ML
    try:
        from matplotlib.patheffects import withStroke
        overlay_path_effects = [withStroke(linewidth=1.5, foreground='black')]
    except Exception:
        overlay_path_effects = []
    valid_areas = [item['total_area'] for item in overlay_data if item.get('total_area') is not None and not (isinstance(item['total_area'], float) and (np.isnan(item['total_area']) or item['total_area'] < 0))]
    if area_vmin_global is not None and area_vmax_global is not None and area_vmax_global > area_vmin_global:
        area_norm_overview = Normalize(vmin=area_vmin_global, vmax=area_vmax_global)
    elif valid_areas:
        area_vmin = min(valid_areas)
        area_vmax = max(valid_areas)
        if area_vmax <= area_vmin:
            area_vmax = area_vmin + 1.0
        area_norm_overview = Normalize(vmin=area_vmin, vmax=area_vmax)
    else:
        area_norm_overview = None
    cmap_overview = _total_area_colormap_for_overlay()
    if overlay_data:
        all_rts_sec = np.concatenate([np.asarray(item['rts'], dtype=float).ravel() for item in overlay_data])
        rt_min_sec, rt_max_sec = float(np.min(all_rts_sec)), float(np.max(all_rts_sec))
        if rt_max_sec <= rt_min_sec:
            rt_max_sec = rt_min_sec + 60.0
        buffer_min = max(0.5, (rt_max_sec - rt_min_sec) / 60.0 * 0.1)
        x_min_ov = max(0.0, rt_min_sec / 60.0 - buffer_min)
        x_max_ov = rt_max_sec / 60.0 + buffer_min
        wins_min = [item['min_rt'] for item in overlay_data if item.get('min_rt') is not None]
        wins_max = [item['max_rt'] for item in overlay_data if item.get('max_rt') is not None]
        if wins_min and wins_max and max(wins_max) > min(wins_min):
            active_min_sec, active_max_sec = float(min(wins_min)), float(max(wins_max))
            buf_tracks = max(0.5, (active_max_sec - active_min_sec) / 60.0 * 0.15)
            x_min_tracks = max(0.0, active_min_sec / 60.0 - buf_tracks)
            x_max_tracks = active_max_sec / 60.0 + buf_tracks
        else:
            x_min_tracks, x_max_tracks = x_min_ov, x_max_ov
    else:
        x_min_ov, x_max_ov = 0.0, 1.0
        x_min_tracks, x_max_tracks = 0.0, 1.0
    x_max_ov_plot = x_max_ov if x_max_ov > x_min_ov else x_min_ov + 1.0
    y_max_ov = 0.0
    for item in overlay_data:
        rts_min_ov = np.asarray(item['rts'], dtype=float) / 60.0
        total_ints_ov = np.asarray(item['total_intensities'], dtype=float)
        mask = (rts_min_ov >= x_min_ov) & (rts_min_ov <= x_max_ov)
        if np.any(mask) and len(total_ints_ov) == len(rts_min_ov):
            m = float(np.max(total_ints_ov[mask]))
            if np.isfinite(m):
                y_max_ov = max(y_max_ov, m)
    if y_max_ov <= 0:
        for item in overlay_data:
            total_ints_ov = np.asarray(item['total_intensities'], dtype=float)
            if len(total_ints_ov) > 0:
                m = float(np.nanmax(total_ints_ov))
                if np.isfinite(m) and m > 0:
                    y_max_ov = max(y_max_ov, m)
    y_floor = 1.0
    track_height, track_gap = 2.0, 0.25
    label_pad_min = 1.2
    n_tracks = len(overlay_data)
    fig_overview = plt.figure(figsize=(22, 72))
    fig_overview.patch.set_facecolor('black')
    gs_overview = GridSpec(3, 1, figure=fig_overview, height_ratios=[5, 5, 6], hspace=0.35)
    ax_overview = fig_overview.add_subplot(gs_overview[0])
    ax_overview_log = fig_overview.add_subplot(gs_overview[1], sharex=ax_overview)
    ax_overview_tracks = fig_overview.add_subplot(gs_overview[2])
    for ax in (ax_overview, ax_overview_log, ax_overview_tracks):
        ax.set_facecolor('black')
    for item in overlay_data:
        min_rt_m, max_rt_m = item.get('min_rt'), item.get('max_rt')
        cmin, cmax = item.get('collection_min_rt'), item.get('collection_max_rt')
        if min_rt_m is not None and max_rt_m is not None:
            ax_overview.axvspan(float(min_rt_m) / 60.0, float(max_rt_m) / 60.0, alpha=0.06, color='green', zorder=0)
        if cmin is not None and cmax is not None:
            ax_overview.axvspan(float(cmin) / 60.0, float(cmax) / 60.0, alpha=0.04, color=COLLECTION_WINDOW_COLOR, zorder=0)
    for item in overlay_data:
        rts_min_overlay = np.asarray(item['rts'], dtype=float) / 60.0
        total_ints = np.asarray(item['total_intensities'], dtype=float)
        if len(rts_min_overlay) < 2 or len(total_ints) < 2:
            continue
        ta = item.get('total_area')
        color = cmap_overview(area_norm_overview(ta)) if area_norm_overview is not None and ta is not None and not (isinstance(ta, float) and (np.isnan(ta) or ta < 0)) else (0.6, 0.6, 0.6, 0.8)
        color_rgb = color[:3] if len(color) >= 3 else color
        low_ints = np.where(total_ints < 1e5, total_ints, 0)
        high_ints = np.where(total_ints >= 1e5, total_ints, 0)
        ax_overview.fill_between(rts_min_overlay, 0, low_ints, color=color_rgb, alpha=0.12, zorder=5)
        ax_overview.fill_between(rts_min_overlay, 0, high_ints, color=color_rgb, alpha=0.35, zorder=5)
        low_plot = np.where(total_ints < 1e5, total_ints, np.nan)
        high_plot = np.where(total_ints >= 1e5, total_ints, np.nan)
        ax_overview.plot(rts_min_overlay, low_plot, color=color_rgb, alpha=0.25, linewidth=1.0, path_effects=overlay_path_effects)
        ax_overview.plot(rts_min_overlay, high_plot, color=color_rgb, alpha=0.95, linewidth=2.5, label=item['label'], path_effects=overlay_path_effects)
    ax_overview.set_ylabel('Intensity', fontsize=13, fontfamily='serif', color='0.85')
    ax_overview.set_title('Accepted Peptides – Raw Intensity (green = integration window, gray = collection)', fontsize=14, fontweight='bold', fontfamily='serif', color='0.9')
    ax_overview.set_xlim(left=x_min_ov, right=x_max_ov_plot)
    ax_overview.xaxis.set_major_locator(_ML(max(0.5, (x_max_ov_plot - x_min_ov) / 10)))
    ax_overview.xaxis.set_minor_locator(_ML(max(0.1, (x_max_ov_plot - x_min_ov) / 30)))
    if y_max_ov > 0:
        ax_overview.set_ylim(0, y_max_ov * 1.1)
    ax_overview.tick_params(labelsize=11, colors='0.85')
    for label in ax_overview.get_xticklabels() + ax_overview.get_yticklabels():
        label.set_fontfamily('serif')
    ax_overview.grid(True, alpha=0.25)
    for spine in ax_overview.spines.values():
        spine.set_color('0.6')
    ax_overview.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8, prop={'family': 'serif'}, ncol=1, labelcolor='0.9')
    for item in overlay_data:
        min_rt_m, max_rt_m = item.get('min_rt'), item.get('max_rt')
        cmin, cmax = item.get('collection_min_rt'), item.get('collection_max_rt')
        if min_rt_m is not None and max_rt_m is not None:
            ax_overview_log.axvspan(float(min_rt_m) / 60.0, float(max_rt_m) / 60.0, alpha=0.06, color='green', zorder=0)
        if cmin is not None and cmax is not None:
            ax_overview_log.axvspan(float(cmin) / 60.0, float(cmax) / 60.0, alpha=0.04, color=COLLECTION_WINDOW_COLOR, zorder=0)
    for item in overlay_data:
        rts_min_overlay = np.asarray(item['rts'], dtype=float) / 60.0
        total_ints = np.asarray(item['total_intensities'], dtype=float)
        total_ints_log = np.maximum(total_ints, y_floor)
        if len(rts_min_overlay) < 2 or len(total_ints_log) < 2:
            continue
        ta = item.get('total_area')
        color = cmap_overview(area_norm_overview(ta)) if area_norm_overview is not None and ta is not None and not (isinstance(ta, float) and (np.isnan(ta) or ta < 0)) else (0.6, 0.6, 0.6, 0.8)
        color_rgb = color[:3] if len(color) >= 3 else color
        low_log = np.where(total_ints_log < 1e5, total_ints_log, y_floor)
        high_log = np.where(total_ints_log >= 1e5, total_ints_log, y_floor)
        ax_overview_log.fill_between(rts_min_overlay, y_floor, low_log, color=color_rgb, alpha=0.12, zorder=5)
        ax_overview_log.fill_between(rts_min_overlay, y_floor, high_log, color=color_rgb, alpha=0.35, zorder=5)
        low_plot_log = np.where(total_ints_log < 1e5, total_ints_log, np.nan)
        high_plot_log = np.where(total_ints_log >= 1e5, total_ints_log, np.nan)
        ax_overview_log.plot(rts_min_overlay, low_plot_log, color=color_rgb, alpha=0.25, linewidth=1.0, path_effects=overlay_path_effects)
        ax_overview_log.plot(rts_min_overlay, high_plot_log, color=color_rgb, alpha=0.95, linewidth=2.5, label=item['label'], path_effects=overlay_path_effects)
    ax_overview_log.set_ylabel('Intensity (log)', fontsize=13, fontfamily='serif', color='0.85')
    ax_overview_log.set_yscale('log')
    ax_overview_log.set_title('Accepted Peptides – Raw Intensity (log; green = integration, gray = collection)', fontsize=14, fontweight='bold', fontfamily='serif', color='0.9')
    ax_overview_log.set_xlim(left=x_min_ov, right=x_max_ov_plot)
    ax_overview_log.xaxis.set_major_locator(_ML(max(0.5, (x_max_ov_plot - x_min_ov) / 10)))
    ax_overview_log.xaxis.set_minor_locator(_ML(max(0.1, (x_max_ov_plot - x_min_ov) / 30)))
    if y_max_ov > 0:
        ax_overview_log.set_ylim(y_floor, y_max_ov * 2.0)
    ax_overview_log.tick_params(labelsize=11, colors='0.85')
    for label in ax_overview_log.get_xticklabels() + ax_overview_log.get_yticklabels():
        label.set_fontfamily('serif')
    ax_overview_log.grid(True, alpha=0.25, which='both')
    for spine in ax_overview_log.spines.values():
        spine.set_color('0.6')
    ax_overview_log.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8, prop={'family': 'serif'}, ncol=1, labelcolor='0.9')
    for i, item in enumerate(overlay_data):
        rts_min = np.asarray(item['rts'], dtype=float) / 60.0
        total_ints = np.asarray(item['total_intensities'], dtype=float)
        if len(rts_min) < 2 or len(total_ints) < 2:
            continue
        imax = float(np.nanmax(total_ints)) if len(total_ints) > 0 else 1.0
        imax = imax if imax > 0 and not np.isnan(imax) else 1.0
        I_norm = total_ints / imax
        y_offset = i * (track_height + track_gap)
        y_plot = I_norm * track_height + y_offset
        ta = item.get('total_area')
        color = cmap_overview(area_norm_overview(ta)) if area_norm_overview is not None and ta is not None and not (isinstance(ta, float) and (np.isnan(ta) or ta < 0)) else (0.5, 0.5, 0.5, 0.9)
        ax_overview_tracks.fill_between(rts_min, y_offset, y_plot, color=color, alpha=0.95)
        ax_overview_tracks.plot(rts_min, y_plot, color=color, linewidth=1.5, alpha=1.0)
        lab = item.get('label') or f'#{i+1}'
        if len(str(lab)) > 22:
            lab = str(lab)[:19] + '…'
        ax_overview_tracks.text(x_min_tracks - label_pad_min, y_offset + track_height * 0.5, lab, va='center', ha='right', fontsize=7, fontfamily='serif')
    for item in overlay_data:
        min_rt_m, max_rt_m = item.get('min_rt'), item.get('max_rt')
        cmin, cmax = item.get('collection_min_rt'), item.get('collection_max_rt')
        if min_rt_m is not None and max_rt_m is not None:
            ax_overview_tracks.axvspan(float(min_rt_m) / 60.0, float(max_rt_m) / 60.0, alpha=0.06, color='green', zorder=0)
        if cmin is not None and cmax is not None:
            ax_overview_tracks.axvspan(float(cmin) / 60.0, float(cmax) / 60.0, alpha=0.04, color=COLLECTION_WINDOW_COLOR, zorder=0)
    ax_overview_tracks.set_xlabel('MS1 Retention Time (min)', fontsize=13, fontfamily='serif', color='0.85')
    ax_overview_tracks.set_ylabel('Track (shape norm.)', fontsize=11, fontfamily='serif', color='0.85')
    ax_overview_tracks.set_title('Accepted Peptides – Tracks (shape only: each trace max=1; x zoomed to integration windows)', fontsize=14, fontweight='bold', fontfamily='serif', color='0.9')
    ax_overview_tracks.set_xlim(left=x_min_tracks - label_pad_min - 0.5, right=x_max_tracks + 0.3)
    ax_overview_tracks.set_ylim(-0.3, n_tracks * (track_height + track_gap) - track_gap + 0.3)
    ax_overview_tracks.tick_params(labelsize=11, colors='0.85')
    for label in ax_overview_tracks.get_xticklabels() + ax_overview_tracks.get_yticklabels():
        label.set_fontfamily('serif')
    ax_overview_tracks.set_yticks([])
    ax_overview_tracks.tick_params(axis='y', left=False)
    ax_overview_tracks.grid(True, alpha=0.25, axis='x')
    for spine in ax_overview_tracks.spines.values():
        spine.set_color('0.6')
    for txt in ax_overview_tracks.texts:
        txt.set_color('0.9')
    if area_norm_overview is not None:
        sm_overview = ScalarMappable(norm=area_norm_overview, cmap=cmap_overview)
        sm_overview.set_array([])
        cbar_overview = fig_overview.colorbar(sm_overview, ax=[ax_overview, ax_overview_log, ax_overview_tracks], shrink=0.5, aspect=25, pad=0.08)
        cbar_overview.ax.set_facecolor('black')
        cbar_overview.set_label('total_area (MS1 peak)', fontsize=12, fontweight='bold', fontfamily='serif', color='0.9')
        cbar_overview.ax.tick_params(labelsize=10, colors='0.85')
        for label in cbar_overview.ax.get_xticklabels() + cbar_overview.ax.get_yticklabels():
            label.set_fontfamily('serif')
            label.set_color('0.85')
    fig_overview.tight_layout(rect=[0, 0, 0.82, 1], pad=1.2)
    overview_path = os.path.join(output_dir, 'all_peptides_overlay.png')
    fig_overview.savefig(overview_path, dpi=150, pad_inches=0.25, facecolor='black')
    plt.close(fig_overview)
    print(f"[DEBUG] Overview overlay plot saved to: {os.path.abspath(overview_path)}")


def run_plots_from_dataframe(args):
    """
    Load chromatogram_metrics_all.csv and chromatogram_traces.npz; apply filtering criteria;
    write accepted/rejected dataframes and plots.
    If filter_csv is provided, only plot peptides that appear in that CSV (thins down plots).
    """
    metrics_csv = getattr(args, 'chromatogram_metrics_csv', None) or getattr(args, 'chromatogram_metrics', None)
    traces_path = getattr(args, 'chromatogram_traces', None)
    filter_csv = getattr(args, 'filter_csv', None)
    exclude_mods = getattr(args, 'exclude_mods', False)
    output_dir = getattr(args, 'output_accepted_dir', None) or getattr(args, 'output_dir', None)
    rejected_dir = getattr(args, 'output_rejected_dir', None) or getattr(args, 'rejected_dir', None)

    if not metrics_csv or not os.path.exists(metrics_csv):
        print(f"Error: chromatogram_metrics_csv not found: {metrics_csv}")
        return
    if not traces_path or not os.path.exists(traces_path):
        print(f"Error: chromatogram_traces not found: {traces_path}")
        return

    df = pd.read_csv(metrics_csv)
    # Optionally restrict to peptides in filter CSV to reduce plotted set
    if filter_csv and os.path.exists(filter_csv):
        try:
            with open(filter_csv, 'r') as f:
                first = f.readline()
            filter_skip = 1 if 'CometVersion' in first else 0
        except Exception:
            filter_skip = 0
        df_filter = pd.read_csv(filter_csv, sep=',', skiprows=filter_skip, engine='python', quotechar='"', on_bad_lines='warn')
        if 'plain_peptide' in df_filter.columns:
            def _norm_mods(m):
                if m is None or (isinstance(m, float) and pd.isna(m)):
                    return '-'
                s = str(m).strip()
                return s if s and s.lower() != 'nan' else '-'
            df_filter['_mods'] = df_filter.apply(lambda r: _norm_mods(r.get('modifications', '-')), axis=1)
            filter_keys = set(zip(
                df_filter['plain_peptide'].astype(str).str.strip(),
                df_filter['charge'].fillna(0).astype(int),
                df_filter['_mods']
            ))
            # Metrics CSV may have 'peptide' or 'plain_peptide'
            pep_col = 'plain_peptide' if 'plain_peptide' in df.columns else 'peptide'
            df['_mods'] = df.apply(lambda r: _norm_mods(r.get('modifications', r.get('mods', '-'))), axis=1)
            mask = df.apply(lambda r: (str(r.get(pep_col, '')).strip(), int(r.get('charge', 0)) if pd.notna(r.get('charge')) else 0, r['_mods']) in filter_keys, axis=1)
            n_before = len(df)
            df = df[mask].copy()
            df = df.drop(columns=['_mods'], errors='ignore')
            print(f"[DEBUG] Filter CSV: {len(df)} / {n_before} peptides to plot (restricted to peptides present in {os.path.basename(filter_csv)})")
    data = np.load(traces_path)

    def _has_modifications(m):
        if m is None or (isinstance(m, float) and pd.isna(m)):
            return False
        s = str(m).strip().lower()
        return s not in ('', '-', 'nan', 'none')

    def _to_bool(v):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return False
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return bool(v)
        s = str(v).strip().lower()
        return s in ('true', '1', 'yes')

    # Apply filtering criteria from the dataframe columns (no mzML needed).
    # Require: (1) M0, M+1, M+2, M+3 present (has_required_isotopes);
    #          (2) M0 and M+1 > M+2 and M+3.
    status_list = []
    rejection_reasons = []
    for _, row in df.iterrows():
        has_required = _to_bool(row.get('has_required_isotopes', False))
        if 'm0_m1_gt_m2_m3' in df.columns and not pd.isna(row.get('m0_m1_gt_m2_m3')):
            envelope_ok = _to_bool(row.get('m0_m1_gt_m2_m3', False))
        else:
            i0 = pd.to_numeric(pd.Series([row.get('m0_intensity')]), errors='coerce').iloc[0]
            i1 = pd.to_numeric(pd.Series([row.get('m1_intensity')]), errors='coerce').iloc[0]
            i2 = pd.to_numeric(pd.Series([row.get('m2_intensity')]), errors='coerce').iloc[0]
            i3 = pd.to_numeric(pd.Series([row.get('m3_intensity')]), errors='coerce').iloc[0]
            if pd.notna(i0) and pd.notna(i1) and pd.notna(i2) and pd.notna(i3):
                envelope_ok = bool((float(i0) > float(i2)) and (float(i0) > float(i3)) and (float(i1) > float(i2)) and (float(i1) > float(i3)))
            else:
                envelope_ok = _to_bool(row.get('envelope_ok', False))
        mods = row.get('modifications', '-')
        if has_required and envelope_ok and not (exclude_mods and _has_modifications(mods)):
            status_list.append('accepted')
            rejection_reasons.append('')
        else:
            status_list.append('rejected')
            if not has_required:
                rejection_reasons.append('Summed MS1 must contain M0, M+1, M+2, and M+3')
            elif not envelope_ok:
                rejection_reasons.append(str(row.get('rejection_reason', 'Isotopic envelope must satisfy M0 and M+1 > M+2 and M+3')) or 'Isotopic envelope must satisfy M0 and M+1 > M+2 and M+3')
            else:
                rejection_reasons.append('Excluded: peptide has modifications (--exclude-mods)')

    df['status'] = status_list
    df['rejection_reason'] = rejection_reasons
    accepted_df = df[df['status'] == 'accepted'].copy()
    rejected_df = df[df['status'] == 'rejected'].copy()

    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(metrics_csv), 'accepted')
    if rejected_dir is None:
        rejected_dir = os.path.join(os.path.dirname(metrics_csv), 'rejected')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(rejected_dir, exist_ok=True)

    accepted_csv = os.path.join(output_dir, 'chromatogram_metrics_accepted.csv')
    rejected_csv = os.path.join(rejected_dir, 'chromatogram_metrics_rejected.csv')
    accepted_no_status = accepted_df.drop(columns=['status', 'rejection_reason'], errors='ignore')
    accepted_no_status.to_csv(accepted_csv, index=False)
    rejected_df.to_csv(rejected_csv, index=False)
    print(f"[DEBUG] Plot-from-dataframe: accepted CSV -> {os.path.abspath(accepted_csv)} ({len(accepted_df)} rows)")
    print(f"[DEBUG] Plot-from-dataframe: rejected CSV -> {os.path.abspath(rejected_csv)} ({len(rejected_df)} rows)")

    # Build scatter data from df for all peptides (coelution_score, shape_corr, etc.)
    df_all = df.copy()

    if 'trace_index' not in df.columns:
        print("[DEBUG] WARNING: chromatogram_metrics_all.csv missing 'trace_index' column. Using row index (may misalign if filter_csv changes row order). Re-run extraction to add trace_index.")
    n_saved = 0
    for orig_idx, row in df.iterrows():
        trace_idx = int(row['trace_index']) if 'trace_index' in df.columns and pd.notna(row.get('trace_index')) else orig_idx
        rts_key = f'trace_{trace_idx}_rts'
        int_key = f'trace_{trace_idx}_int'
        if rts_key not in data or int_key not in data:
            print(f"[DEBUG] Skipping row {orig_idx} (missing traces)")
            continue
        rts = np.asarray(data[rts_key], dtype=float).flatten().copy()
        intensity_matrix = np.asarray(data[int_key], dtype=float).copy()
        n_rts, n_int = len(rts), intensity_matrix.shape[0] if intensity_matrix.ndim >= 1 else len(intensity_matrix)
        if n_rts != n_int:
            print(f"[DEBUG] Row {orig_idx}: RT length {n_rts} != intensity length {n_int}, truncating to match")
            n = min(n_rts, n_int)
            rts = rts[:n].copy()
            intensity_matrix = intensity_matrix[:n] if intensity_matrix.ndim == 1 else intensity_matrix[:n, :].copy()
        if len(rts) < 2 or np.ptp(rts) < 0.01:
            print(f"[DEBUG] Row {orig_idx}: Skipping degenerate RT vector (len={len(rts)}, ptp={np.ptp(rts):.4f})")
            continue
        n_int_loaded = intensity_matrix.shape[0] if intensity_matrix.ndim >= 1 else len(intensity_matrix)
        if n_int_loaded < 2:
            print(f"[DEBUG] Row {orig_idx}: Skipping aggregate/single-point intensity (need per-scan data, got {n_int_loaded} points)")
            continue
        is_rejected = row['status'] == 'rejected'
        rejection_reason = row['rejection_reason'] if is_rejected else ''
        peptide_key = row.get('peptide_key', str(orig_idx))
        safe_key = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in peptide_key)[:50]
        seq_start = int(row.get('sequence_start_pos', 999999))
        filename = f"{seq_start:05d}_{orig_idx+1:04d}_{safe_key}_{'rejected' if is_rejected else 'accepted'}.png"
        out_path = os.path.join(rejected_dir if is_rejected else output_dir, filename)
        _draw_peptide_figure_from_dataframe(row, rts, intensity_matrix, df_all, out_path, is_rejected, rejection_reason)
        n_saved += 1
        if n_saved % 10 == 0:
            print(f"[DEBUG] Saved {n_saved}/{len(df)} figures")

    # Build overlay_data from accepted_df and create MS1 overlay (all_peptides_overlay.png) so "Generate plots" produces it
    overlay_data = []
    for orig_idx, row in accepted_df.iterrows():
        trace_idx = int(row['trace_index']) if 'trace_index' in accepted_df.columns and pd.notna(row.get('trace_index')) else orig_idx
        rts_key = f'trace_{trace_idx}_rts'
        int_key = f'trace_{trace_idx}_int'
        if rts_key not in data or int_key not in data:
            continue
        rts = np.asarray(data[rts_key], dtype=float).flatten().copy()
        intensity_matrix = np.asarray(data[int_key], dtype=float).copy()
        n_rts, n_int = len(rts), intensity_matrix.shape[0] if intensity_matrix.ndim >= 1 else len(intensity_matrix)
        if n_rts != n_int:
            n = min(n_rts, n_int)
            rts = rts[:n].copy()
            intensity_matrix = intensity_matrix[:n] if intensity_matrix.ndim == 1 else intensity_matrix[:n, :].copy()
        if len(rts) < 2 or np.ptp(rts) < 0.01:
            continue
        total_intensities = np.sum(intensity_matrix, axis=1) if intensity_matrix.ndim > 1 else np.asarray(intensity_matrix, dtype=float).flatten()
        if len(total_intensities) != len(rts):
            continue
        pep = str(row.get('plain_peptide', row.get('peptide', ''))).strip()
        ch = int(row.get('charge', 0)) if pd.notna(row.get('charge')) else 0
        mods = str(row.get('modifications', '-')).strip() if pd.notna(row.get('modifications')) else '-'
        min_rt = row.get('detected_peak_min_rt') or row.get('collection_min_rt')
        max_rt = row.get('detected_peak_max_rt') or row.get('collection_max_rt')
        min_rt = float(min_rt) if min_rt is not None and not (isinstance(min_rt, float) and np.isnan(min_rt)) else None
        max_rt = float(max_rt) if max_rt is not None and not (isinstance(max_rt, float) and np.isnan(max_rt)) else None
        coll_min = row.get('collection_min_rt')
        coll_max = row.get('collection_max_rt')
        coll_min = float(coll_min) if coll_min is not None and not (isinstance(coll_min, float) and np.isnan(coll_min)) else None
        coll_max = float(coll_max) if coll_max is not None and not (isinstance(coll_max, float) and np.isnan(coll_max)) else None
        ta = row.get('total_area')
        ta = float(ta) if ta is not None and not (isinstance(ta, float) and np.isnan(ta)) else None
        seq_start = int(row.get('sequence_start_pos', 999999))
        min_rt_min = min_rt / 60.0 if min_rt is not None else None
        max_rt_min = max_rt / 60.0 if max_rt is not None else None
        label = f"{pep[:40]}{'...' if len(pep) > 40 else ''} +{ch} | {mods} | RT {min_rt_min:.2f}-{max_rt_min:.2f} min" if min_rt_min is not None and max_rt_min is not None else f"{pep[:40]}{'...' if len(pep) > 40 else ''} +{ch} | {mods}"
        overlay_data.append({
            'rts': rts, 'total_intensities': total_intensities, 'label': label, 'total_area': ta,
            'sequence_start_pos': seq_start, 'min_rt': min_rt, 'max_rt': max_rt,
            'collection_min_rt': coll_min, 'collection_max_rt': coll_max,
        })
    if len(overlay_data) > 0:
        _create_overlay_figures(overlay_data, output_dir)

    print(f"[DEBUG] Plot-from-dataframe: done. Accepted -> {output_dir}, Rejected -> {rejected_dir}")


def run_chromatograms(args):
    """Run MS1 chromatogram extraction and plots. args: comet_csv (or csv), raw_file (or mzml), output_png (optional), filter_csv, exclude_mods, test, no_psm_filter."""
    plot_only = getattr(args, 'plot_only', False)
    if plot_only:
        run_plots_from_dataframe(args)
        return

    comet_csv = getattr(args, 'comet_csv', None) or getattr(args, 'csv', None)
    raw_file = getattr(args, 'raw_file', None) or getattr(args, 'mzml', None)
    filter_csv = getattr(args, 'filter_csv', None)
    output_png = getattr(args, 'output_png', None)
    if not output_png and comet_csv:
        output_png = comet_csv.replace('.csv', '_chromatograms.png')
    test_mode = getattr(args, 'test', False)
    exclude_mods = getattr(args, 'exclude_mods', False)
    extract_only = getattr(args, 'extract_only', False)
    no_psm_filter = getattr(args, 'no_psm_filter', False)
    # Step 6 (extract): skip 0.5% intensity filter; significance filter applied in Step 7 (refine_significant_frags)
    skip_significance_intensity_filter = getattr(args, 'skip_significance_intensity_filter', False)

    if not comet_csv or not raw_file:
        print("Error: comet_csv and raw_file are required unless --plot-only is used.")
        sys.exit(1)
    if not os.path.exists(comet_csv):
        print(f"Error: Comet CSV file not found: {comet_csv}")
        sys.exit(1)
    
    if not os.path.exists(raw_file):
        print(f"Error: Raw file not found: {raw_file}")
        sys.exit(1)
    
    if filter_csv and not os.path.exists(filter_csv):
        print(f"Error: Filter CSV file not found: {filter_csv}")
        sys.exit(1)
    
    print("[DEBUG] ========================================")
    print("[DEBUG] Extracting MS1 chromatograms")
    print("[DEBUG] ========================================")
    
    # Read CSV (full data source)
    print(f"[DEBUG] Reading CSV: {comet_csv}")
    try:
        with open(comet_csv, 'r') as f:
            first_line = f.readline()
            skip_rows = 1 if 'CometVersion' in first_line else 0
    except:
        skip_rows = 0
    
    df = pd.read_csv(comet_csv, sep=',', skiprows=skip_rows, engine='python', quotechar='"', on_bad_lines='warn')
    # Compatibility aliases for renamed workflow identifiers/RT columns.
    if 'peptide_sequence' in df.columns and 'plain_peptide' not in df.columns:
        df['plain_peptide'] = df['peptide_sequence']
    if 'protein_position' in df.columns and 'sequence_positions' not in df.columns:
        df['sequence_positions'] = df['protein_position']
    if 'MS2_RT_sec' in df.columns and 'MS2_retention_time_sec' not in df.columns:
        df['MS2_retention_time_sec'] = df['MS2_RT_sec']
    if 'MS2_RT_minutes' in df.columns and 'MS2_retention_time_min' not in df.columns:
        df['MS2_retention_time_min'] = df['MS2_RT_minutes']
    if 'MS1_RT_sec' in df.columns and 'MS1_retention_time_sec' not in df.columns:
        df['MS1_retention_time_sec'] = df['MS1_RT_sec']
    if 'MS1_RT_minutes' in df.columns and 'MS1_retention_time_min' not in df.columns:
        df['MS1_retention_time_min'] = df['MS1_RT_minutes']
    print(f"[DEBUG] CSV loaded: {len(df)} rows")
    try:
        from visualization.csv_validation import validate_and_log
        validate_and_log(
            df,
            source_name=os.path.basename(comet_csv),
            required_columns=('plain_peptide', 'charge'),
            validate_identifiers=True,
            validate_numeric=True,
        )
    except Exception as e:
        print(f"[DEBUG] Validation warning: {e}")
    # Fallback: use retention_time_sec or MS2_retention_time_sec as MS1_retention_time_sec if required column missing
    if 'MS1_retention_time_sec' not in df.columns:
        if 'retention_time_sec' in df.columns:
            df['MS1_retention_time_sec'] = df['retention_time_sec']
            print("[DEBUG] Using retention_time_sec as MS1_retention_time_sec")
        elif 'MS2_retention_time_sec' in df.columns:
            df['MS1_retention_time_sec'] = df['MS2_retention_time_sec']
            print("[DEBUG] Using MS2_retention_time_sec as MS1_retention_time_sec (run add_ms1_data_openms.py for true MS1 RT)")
    
    # If filter CSV provided: restrict to peptides that appear in the filter list (same CSV format / keys)
    if filter_csv:
        print(f"[DEBUG] Filtering to peptides listed in: {filter_csv}")
        try:
            with open(filter_csv, 'r') as f:
                filter_first = f.readline()
            filter_skip = 1 if 'CometVersion' in filter_first else 0
        except Exception:
            filter_skip = 0
        df_filter = pd.read_csv(filter_csv, sep=',', skiprows=filter_skip, engine='python', quotechar='"', on_bad_lines='warn')
        if 'plain_peptide' not in df_filter.columns:
            print("Error: Filter CSV must contain column 'plain_peptide'")
            sys.exit(1)
        try:
            from visualization.csv_validation import validate_and_log
            validate_and_log(
                df_filter,
                source_name=os.path.basename(filter_csv),
                required_columns=('plain_peptide', 'charge'),
                validate_identifiers=True,
                validate_numeric=True,
            )
        except Exception as e:
            print(f"[DEBUG] Validation warning: {e}")
        def _norm_mods(m):
            if m is None or (isinstance(m, float) and pd.isna(m)):
                return '-'
            s = str(m).strip()
            return s if s and s.lower() != 'nan' else '-'
        df_filter['_mods_norm'] = df_filter.apply(lambda row: _norm_mods(row.get('modifications', '-')), axis=1)
        filtered_keys = set(zip(
            df_filter['plain_peptide'].astype(str).str.strip(),
            df_filter['charge'].fillna(0).astype(int),
            df_filter['_mods_norm']
        ))
        df['_mods_norm'] = df.apply(lambda row: _norm_mods(row.get('modifications', '-')), axis=1)
        mask = [k in filtered_keys for k in zip(df['plain_peptide'].astype(str).str.strip(), df['charge'].fillna(0).astype(int), df['_mods_norm'])]
        df = df.loc[mask].reset_index(drop=True)
        # Merge drift window and m0_gt_m1_gt_m2 from filter CSV (for re-extraction: x-axis and skip envelope re-check)
        if 'drift_min_rt' in df_filter.columns and 'drift_max_rt' in df_filter.columns:
            merge_cols = ['plain_peptide', 'charge', '_mods_norm', 'drift_min_rt', 'drift_max_rt']
            if 'm0_gt_m1_gt_m2' in df_filter.columns:
                merge_cols.append('m0_gt_m1_gt_m2')
            drift_cols = df_filter[merge_cols].drop_duplicates(subset=['plain_peptide', 'charge', '_mods_norm'])
            df = df.merge(drift_cols, on=['plain_peptide', 'charge', '_mods_norm'], how='left')
        df = df.drop(columns=['_mods_norm'])
        print(f"[DEBUG] After filter: {len(df)} rows (peptides from filter CSV only)")
    print("[DEBUG] Keeping all rows - using quality scores for cluster selection only")

    # Ensure every row has sequence_start_pos when possible (from sequence_positions), so plots and downstream CSVs all have it
    if 'sequence_positions' in df.columns:
        if 'sequence_start_pos' not in df.columns:
            df['sequence_start_pos'] = np.nan
        def _parse_start_from_seq_pos(val):
            if pd.isna(val) or val is None or str(val).strip() == '':
                return np.nan
            s = str(val).strip()
            if '-' in s:
                try:
                    return int(s.split('-')[0])
                except (ValueError, TypeError):
                    return np.nan
            if ',' in s:
                try:
                    return int(s.split(',')[0].strip())
                except (ValueError, TypeError):
                    return np.nan
            try:
                return int(s)
            except (ValueError, TypeError):
                return np.nan
        missing = df['sequence_start_pos'].isna()
        if missing.any():
            parsed = df.loc[missing, 'sequence_positions'].map(_parse_start_from_seq_pos)
            df.loc[missing, 'sequence_start_pos'] = parsed
            n_filled = parsed.notna().sum()
            if n_filled > 0:
                print(f"[DEBUG] Filled sequence_start_pos from sequence_positions for {n_filled} rows")
    
    # Load mzML file for chromatogram extraction (cached for reuse)
    print("[DEBUG] ========================================")
    print("[DEBUG] Loading mzML file for chromatogram extraction...")
    print("[DEBUG] ========================================")
    feature_map = extract_ms1_features_openms(raw_file)
    print("[DEBUG] mzML file loaded and cached for chromatogram extraction")
    print("[DEBUG] ========================================")
    
    # Check required columns
    required_cols = ['plain_peptide', 'charge', 'MS1_retention_time_sec']
    for col in required_cols:
        if col not in df.columns:
            print(f"Error: Required column '{col}' not found in CSV")
            print(f"Available columns: {list(df.columns)}")
            sys.exit(1)
    
    # Normalize m/z column: prefilter renames mz -> theoretical_mz; support both
    if 'mz' in df.columns and 'theoretical_mz' not in df.columns:
        df['theoretical_mz'] = df['mz']
    if 'theoretical_mz' in df.columns and 'mz' not in df.columns:
        df['mz'] = df['theoretical_mz']

    # Check for calc_neutral_mass (required for theoretical m/z)
    if 'calc_neutral_mass' not in df.columns:
        print(f"[DEBUG] Warning: 'calc_neutral_mass' column not found, falling back to observed m/z column")
        use_theoretical_mz = False
    else:
        use_theoretical_mz = True
        print(f"[DEBUG] Using theoretical m/z calculated from calc_neutral_mass")

    # Calculate theoretical m/z for each row
    if use_theoretical_mz:
        print("[DEBUG] Calculating theoretical m/z from calc_neutral_mass...")
        df['mz_theoretical'] = df.apply(
            lambda row: calculate_theoretical_mz(row.get('calc_neutral_mass'), row.get('charge')),
            axis=1
        )
        mz_col = 'mz_theoretical'
    else:
        # Fallback: use mz or theoretical_mz (prefilter output uses theoretical_mz)
        mz_col = 'theoretical_mz' if 'theoretical_mz' in df.columns else 'mz'
        if mz_col not in df.columns:
            print(f"Error: Neither 'calc_neutral_mass' nor 'mz'/'theoretical_mz' column found. Columns: {list(df.columns)}")
            sys.exit(1)
    
    # Create grouping key by peptide identity only (sequence, charge, mods).
    # This ensures one plot per peptide; M0/M1/etc. are already plotted together as isotope traces.
    # Representative m/z for XIC is taken as min(m/z) in the group (monoisotopic).
    # Normalize mods so the same biological (seq, charge, mods) always gets one key (avoids one key per PSM when mod strings differ slightly).
    print("[DEBUG] Creating peptide keys by identity (sequence + charge + mods, no m/z)...")
    
    def _norm_mods_for_key(m):
        """Normalize modifications for grouping: empty/nan -> '-', strip; round masses to 4 decimals so 15.9949 and 15.994900 collapse."""
        if m is None or (isinstance(m, float) and pd.isna(m)):
            return '-'
        s = str(m).strip()
        if not s or s.lower() == 'nan':
            return '-'
        # Round any decimal numbers in the string (e.g. "1_V_15.994900" -> "1_V_15.9949") so same mod doesn't split by float representation
        def round_match(mo):
            try:
                return f"{float(mo.group(0)):.4f}".rstrip('0').rstrip('.')
            except ValueError:
                return mo.group(0)
        s = re.sub(r'\d+\.\d+', round_match, s)
        return s
    
    def create_peptide_key(row):
        """Create a unique key for peptide grouping: sequence, charge, mods only (no m/z). Mods normalized so one key per biological peptide."""
        peptide = row['plain_peptide']
        charge = row['charge'] if 'charge' in row else 0
        mods = _norm_mods_for_key(row.get('modifications', '-'))
        return f"{peptide}_{charge}_{mods}"
    
    df['peptide_key'] = df.apply(create_peptide_key, axis=1)
    
    # Group by peptide identity (one plot per peptide; all isotope traces on same plot)
    print("[DEBUG] Grouping by peptide identity (sequence+charge+mods)...")
    peptide_groups = df.groupby('peptide_key')
    unique_peptide_keys = list(peptide_groups.groups.keys())
    num_peptides = len(unique_peptide_keys)
    
    def representative_mz(g):
        """Representative (monoisotopic) m/z for a peptide group: min m/z in group."""
        mz_vals = g[mz_col].dropna() if mz_col in g.columns else pd.Series(dtype=float)
        mz_vals = mz_vals[mz_vals > 0]
        if len(mz_vals) > 0:
            return float(mz_vals.min())
        if 'mz' in g.columns:
            mz_vals = g['mz'].dropna()
            mz_vals = mz_vals[mz_vals > 0]
            if len(mz_vals) > 0:
                return float(mz_vals.min())
        return None
    
    print(f"[DEBUG] Found {num_peptides} unique peptides (sequence+charge+mods)")
    print(f"[DEBUG] (vs {df['plain_peptide'].nunique()} unique sequences)")
    mz_col_display = mz_col if use_theoretical_mz else 'mz'
    if mz_col_display in df.columns:
        print(f"[DEBUG] (vs {df[mz_col_display].nunique()} unique {mz_col_display} values)")
    
    # Analyze MS1 RT distribution for each group
    print("[DEBUG] Analyzing MS1 RT distribution per peptide species...")
    rt_distribution_issues = []
    for key in unique_peptide_keys[:20]:  # Check first 20
        group = peptide_groups.get_group(key)
        ms1_rts = group['MS1_retention_time_sec'].dropna().tolist()
        ms1_rts = [rt for rt in ms1_rts if rt > 0]
        
        if len(ms1_rts) > 1:
            rt_range = max(ms1_rts) - min(ms1_rts)
            rt_std = np.std(ms1_rts) if len(ms1_rts) > 1 else 0
            
            # Flag groups with wide RT distribution (>30 seconds range)
            if rt_range > 30:
                peptide = group['plain_peptide'].iloc[0]
                charge = group['charge'].iloc[0] if 'charge' in group.columns else '?'
                mods = group['modifications'].iloc[0] if 'modifications' in group.columns else '-'
                mz_col_used = mz_col if use_theoretical_mz else 'mz'
                mz_values = group[mz_col_used].dropna().unique().tolist() if mz_col_used in group.columns else []
                observed_mz_values = group['mz'].dropna().unique().tolist() if 'mz' in group.columns else []
                
                rt_distribution_issues.append({
                    'key': key,
                    'peptide': peptide,
                    'charge': charge,
                    'modifications': mods,
                    'num_spectra': len(group),  # number of spectra (rows) for this unique peptide (sequence+charge+mods)
                    'rt_range': rt_range,
                    'rt_std': rt_std,
                    'rt_min': min(ms1_rts),
                    'rt_max': max(ms1_rts),
                    'mz_values': mz_values,
                    'observed_mz_values': observed_mz_values,
                    'unique_mz': len(mz_values),
                    'num_psms': len(group)  # keep for CSV/output compatibility
                })
    
    if rt_distribution_issues:
        print(f"[DEBUG] Found {len(rt_distribution_issues)} peptide species with wide MS1 RT distribution (>30s range):")
        for issue in rt_distribution_issues[:5]:  # Show first 5
            print(f"[DEBUG]   {issue['peptide']} (charge={issue['charge']}, mods={issue['modifications']})")
            print(f"[DEBUG]     RT range: {issue['rt_min']:.1f} - {issue['rt_max']:.1f} ({issue['rt_range']:.1f}s)")
            mz_label = 'theoretical m/z' if use_theoretical_mz else 'm/z'
            n_spectra = issue.get('num_spectra', issue.get('num_psms', len(peptide_groups.get_group(issue['key']))))
            print(f"[DEBUG]     This unique peptide (same seq+charge+mods): {n_spectra} spectra, {issue['unique_mz']} unique {mz_label} values: {issue['mz_values'][:3]}")
            obs = issue.get('observed_mz_values', [])
            if obs:
                n_obs = len(obs)
                # Each spectrum (row) has one observed m/z; we report unique values across spectra for this unique peptide
                print(f"[DEBUG]     observed m/z (one per spectrum, {n_obs} unique across {n_spectra} spectra): {[round(x, 6) for x in obs[:3]]}{' ...' if n_obs > 3 else ''}")
            
            # Detailed investigation for problematic cases
            group = peptide_groups.get_group(issue['key'])
            print(f"[DEBUG]     Detailed breakdown:")
            
            # Group by theoretical m/z to see if different m/z values have different RTs
            mz_col_used = mz_col if use_theoretical_mz else 'mz'
            if mz_col_used in group.columns:
                mz_groups = group.groupby(mz_col_used)
                mz_label_used = 'theoretical m/z' if use_theoretical_mz else 'observed m/z'
                print(f"[DEBUG]       Grouped by {mz_col_used} ({mz_label_used}, {len(mz_groups)} unique):")
                
                # Flag m/z groups with wide RT ranges
                wide_rt_mz_groups = []
                for mz_val, mz_group in list(mz_groups)[:10]:  # First 10 m/z groups
                    mz_rts = mz_group['MS1_retention_time_sec'].dropna().tolist()
                    mz_rts = [rt for rt in mz_rts if rt > 0]
                    if mz_rts:
                        rt_range = max(mz_rts) - min(mz_rts) if len(mz_rts) > 1 else 0
                        print(f"[DEBUG]         m/z {mz_val:.6f}: {len(mz_group)} spectra, RT range: {min(mz_rts):.1f}-{max(mz_rts):.1f}s (span: {rt_range:.1f}s)")
                        
                        # Show a few example RTs
                        if len(mz_rts) <= 5:
                            print(f"[DEBUG]           RTs: {[f'{rt:.1f}' for rt in sorted(mz_rts)]}")
                        else:
                            print(f"[DEBUG]           RTs (sample): {[f'{rt:.1f}' for rt in sorted(mz_rts)[:3]]} ... {[f'{rt:.1f}' for rt in sorted(mz_rts)[-2:]]}")
                        
                        # Flag if this m/z has wide RT range (>10s)
                        if rt_range > 10:
                            wide_rt_mz_groups.append((mz_val, rt_range, len(mz_group)))
                
                if wide_rt_mz_groups:
                    print(f"[DEBUG]       WARNING: {len(wide_rt_mz_groups)} m/z values have wide RT ranges (>10s) within same m/z!")
                    print(f"[DEBUG]       This suggests MS1 matching may be incorrect or multiple chromatographic peaks exist.")
                    for mz_val, rt_range, count in wide_rt_mz_groups[:3]:
                        print(f"[DEBUG]         m/z {mz_val:.6f}: {count} spectra, RT span: {rt_range:.1f}s")
            
            # Check scan numbers and MS2 RTs to see correlation
            if 'scan' in group.columns and 'MS2_retention_time_sec' in group.columns:
                scans = group['scan'].dropna().tolist()
                ms2_rts = group['MS2_retention_time_sec'].dropna().tolist()
                if scans and ms2_rts:
                    print(f"[DEBUG]       Scan range: {min(scans)} - {max(scans)}")
                    print(f"[DEBUG]       MS2 RT range: {min(ms2_rts):.1f} - {max(ms2_rts):.1f}s")
                    # Check correlation between MS1 and MS2 RT
                    ms1_ms2_pairs = list(zip(group['MS1_retention_time_sec'].dropna(), group['MS2_retention_time_sec'].dropna()))
                    if ms1_ms2_pairs:
                        ms1_ms2_diffs = [abs(ms1 - ms2) for ms1, ms2 in ms1_ms2_pairs if ms1 > 0]
                        if ms1_ms2_diffs:
                            avg_diff = np.mean(ms1_ms2_diffs)
                            print(f"[DEBUG]       MS1-MS2 RT difference: avg {avg_diff:.1f}s (should be small, <10s)")
            
            # Check if there are different protein contexts
            if 'sequence_positions' in group.columns:
                seq_positions = group['sequence_positions'].dropna().unique().tolist()
                if len(seq_positions) > 1:
                    print(f"[DEBUG]       WARNING: Multiple sequence positions: {seq_positions[:3]}")
            
            # Check precursor m/z precision - are they really the same? (report both theoretical and observed when available)
            mz_col_used = mz_col if use_theoretical_mz else 'mz'
            if mz_col_used in group.columns:
                mz_vals = group[mz_col_used].dropna().tolist()
                if len(mz_vals) > 1:
                    mz_range = max(mz_vals) - min(mz_vals)
                    mz_std = np.std(mz_vals)
                    mz_mean = np.mean(mz_vals)
                    ppm_range = (mz_range / mz_mean * 1e6) if mz_mean and mz_mean > 0 else 0.0
                    ppm_std = (mz_std / mz_mean * 1e6) if mz_mean and mz_mean > 0 else 0.0
                    label_used = 'theoretical' if use_theoretical_mz else 'observed'
                    print(f"[DEBUG]       Precursor m/z ({label_used}, {mz_col_used}): range={mz_range:.6f} Da ({ppm_range:.2f} ppm), std={mz_std:.6f} Da ({ppm_std:.2f} ppm)")
                    if mz_range > 0.01:  # More than 0.01 Da difference
                        print(f"[DEBUG]       WARNING: Large m/z variation - may indicate different species!")
            if use_theoretical_mz and 'mz' in group.columns:
                obs_vals = group['mz'].dropna().tolist()
                if obs_vals:
                    obs_range = max(obs_vals) - min(obs_vals) if len(obs_vals) > 1 else 0.0
                    obs_std = np.std(obs_vals) if len(obs_vals) > 1 else 0.0
                    obs_mean = np.mean(obs_vals)
                    obs_ppm_range = (obs_range / obs_mean * 1e6) if obs_mean and obs_mean > 0 else 0.0
                    obs_ppm_std = (obs_std / obs_mean * 1e6) if obs_mean and obs_mean > 0 else 0.0
                    print(f"[DEBUG]       Precursor m/z (observed, mz): range={obs_range:.6f} Da ({obs_ppm_range:.2f} ppm), std={obs_std:.6f} Da ({obs_ppm_std:.2f} ppm)")
            
            print()
    
    # Sort peptides by apex peak RT (proxy: median MS1_retention_time_sec) for combined figure
    def get_apex_rt_sort_key(key):
        """Get sort key: apex RT (proxy = median MS1_retention_time_sec), then sequence start as tiebreaker."""
        group = peptide_groups.get_group(key)
        ms1_rts = group['MS1_retention_time_sec'].dropna()
        ms1_rts = ms1_rts[ms1_rts > 0]
        apex_proxy = float(np.median(ms1_rts)) if len(ms1_rts) > 0 else 999999.0
        seq_start = 999999
        if 'sequence_positions' in group.columns:
            seq_pos = group['sequence_positions'].iloc[0]
            if pd.notna(seq_pos) and seq_pos != '':
                try:
                    if '-' in str(seq_pos):
                        seq_start = int(str(seq_pos).split('-')[0])
                except:
                    pass
        return (apex_proxy, seq_start)
    
    print("[DEBUG] Sorting peptides by apex peak RT (proxy: median MS1 RT)...")
    unique_peptide_keys = sorted(unique_peptide_keys, key=get_apex_rt_sort_key)
    print(f"[DEBUG] Peptides sorted by apex peak RT")
    
    # Test mode: take a smaller random sample of peptides (faster run)
    TEST_SAMPLE_SIZE = 30
    if test_mode:
        import random
        random.seed(42)  # reproducible test runs
        n_sample = min(TEST_SAMPLE_SIZE, len(unique_peptide_keys))
        unique_peptide_keys = random.sample(unique_peptide_keys, n_sample)
        print(f"[DEBUG] TEST MODE: Random sample of {n_sample} peptides (faster run)")
    
    # Update num_peptides after test mode filtering
    num_peptides = len(unique_peptide_keys)
    
    # Create output directory for individual peptide figures (accepted vs rejected separate when provided)
    output_accepted_dir = getattr(args, 'output_accepted_dir', None)
    output_rejected_dir = getattr(args, 'output_rejected_dir', None)
    if output_accepted_dir and output_rejected_dir:
        output_dir = output_accepted_dir
        rejected_dir = output_rejected_dir
    else:
        output_dir = output_png.replace('.png', '_peptides')
        rejected_dir = output_png.replace('.png', '_peptides_rejected')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(rejected_dir, exist_ok=True)
    # Accepted/rejected PNGs use sortable names (seq_position, index); filter by sorting: sorted(os.listdir(...)) or ls *.png | sort
    # Convert to absolute paths for clarity
    output_dir_abs = os.path.abspath(output_dir)
    rejected_dir_abs = os.path.abspath(rejected_dir)
    output_png_abs = os.path.abspath(output_png)
    print(f"[DEBUG] ========== CHROMATOGRAMS_PEPTIDES: PLOT GENERATION ==========")
    print(f"[DEBUG] Creating separate figure for each peptide (num_peptides={num_peptides})")
    print(f"[DEBUG] CHROMATOGRAMS_PEPTIDES: Accepted plots will be written to: {output_dir_abs}")
    print(f"[DEBUG] CHROMATOGRAMS_PEPTIDES: Rejected plots will be written to: {rejected_dir_abs}")
    print(f"[DEBUG] Output directory (accepted): {output_dir_abs}")
    print(f"[DEBUG] Rejected directory: {rejected_dir_abs}")
    if num_peptides == 0:
        print(f"[DEBUG] CHROMATOGRAMS_PEPTIDES: WARNING num_peptides=0 - no individual peptide plots will be generated.")
    
    # Also create combined figure(s) with all peptides arranged by sequence start position
    # Fixed layout: 10 plots per row, 10 rows per file (100 plots max per PNG)
    print(f"[DEBUG] Creating combined figure(s) with all peptides...")
    
    # Fixed grid dimensions
    cols_per_file = 10
    rows_per_file = 10
    plots_per_file = cols_per_file * rows_per_file
    
    # Calculate how many files we need
    num_files = int(np.ceil(num_peptides / plots_per_file))
    print(f"[DEBUG] Will create {num_files} combined figure file(s) ({plots_per_file} plots per file)")
    
    # Subplot size - make them larger since we're limiting to 10 per row
    subplot_width = 4.5  # Larger subplots
    subplot_height = 3.5
    
    fig_width = cols_per_file * subplot_width
    fig_height = rows_per_file * subplot_height
    
    # Spacing between subplots
    hspace, wspace = 0.3, 0.3
    
    peak_windows = []
    
    # Track window sizes for calculating median and detecting suspicious windows
    window_sizes = []  # Will collect (min_rt, max_rt) tuples to calculate window widths
    
    # Track rejected peptides for separate plotting
    rejected_peptides = []  # List of (peptide_key, rejection_reason, peptide_data) tuples
    
    # Filtering criteria string for figure titles (extraction + acceptance)
    FILTERING_CRITERIA = (
        "XIC ppm=20 | anchor mz/rt=20ppm/60s | pre-filter: (no E-value threshold) | "
        "score=0.28 coel + 0.28 shape + 0.14 ratio + 0.12 RT_cov + 0.05 rt_prior + 0.05 quality + 0.20 strength | "
        "reject: modifications, apex < 10^5 | accept: apex ≥ 10^5 | PSM: q≤0.05, PEP≤0.05 | envelope filter: Step 7"
    )
    MIN_SCANS_REQUIRED = 1  # need >= 1 scan with valid MS1 RT; envelope computed for Step 7 (not rejected here)
    MIN_APEX_INTENSITY = 1e5  # apex peak below this is treated as noise; pretty much never select below 10^5

    # Allowed modification masses (fixed mods from sample prep); peptides with ONLY these are not rejected
    ALLOWED_MODIFICATION_MASSES = frozenset(['57.0215', '57.021500'])  # Carbamidomethyl (C) - fixed

    def _has_modifications_for_reject(m):
        """True if peptide has variable modifications (reject). Empty, -, or only Carbamidomethyl -> False (accept)."""
        if m is None or (isinstance(m, float) and pd.isna(m)):
            return False
        s = str(m).strip()
        if s.lower() in ('', '-', 'nan', 'none'):
            return False
        # Parse mass values from format "9_V_15.994900" or "1_V_15.994900,3_C_57.0215" (position_letter_mass)
        masses = []
        for part in s.split(','):
            part = part.strip()
            if '_' in part:
                mass = part.rsplit('_', 1)[-1].strip()
                if mass and mass.replace('.', '').isdigit():
                    masses.append(mass.rstrip('0').rstrip('.') if '.' in mass else mass)
        if not masses:
            return True  # Unknown format, treat as variable mod
        def _norm_mass(m):
            return m.rstrip('0').rstrip('.') if '.' in m else m

        # Reject if ANY mass is not in allowed list (e.g. 15.9949 = Oxidation)
        allowed_norm = frozenset(_norm_mass(am) for am in ALLOWED_MODIFICATION_MASSES)
        for mass in masses:
            norm = _norm_mass(mass)
            if norm not in allowed_norm:
                return True
        return False
    # PSM score thresholds: reject peptide if its best PSM exceeds any of these (aligned with scatter-plot reference lines)
    FILTER_PSM_QVALUE_MAX = 0.05   # reject if best PSM q-value > this (q ≤ 0.05 acceptable)
    FILTER_PSM_PEP_MAX = 0.05      # reject if best PSM PEP > this (PEP ≤ 0.05 acceptable)
    FILTER_PSM_EVALUE_MAX = float('inf')  # no E-value threshold (do not reject based on E-value)
    
    # Track peak window data for ALL peptides (accepted and rejected) for CSV output
    all_peak_windows = []  # List of dicts with peak window info for CSV export
    # Collect (rts, intensity) per peptide in same order as all_peak_windows for npz (extract_only: full matrix; normal: total trace for downstream by-channel/overlay)
    traces_list = []

    # Debug: track where each peptide figure is saved (for "nothing in accepted/rejected" diagnosis)
    saved_accepted_paths = []
    saved_rejected_paths = []
    early_rejected_paths = []

    # Process peptides in batches for multiple combined figure files
    combined_figures = []
    combined_axes_list = []
    
    # Also create combined figures for rejected peptides
    rejected_figures = []
    rejected_axes_list = []
    
    # Collect (rts, total_intensities) per peptide for overview overlay plot
    overlay_data = []
    # Collect chromatogram data for accepted peptides only (for 7-peptide zoom chromatogram figures, shared y-axis)
    chrom_zoom_data = []
    
    for file_idx in range(num_files):
        start_idx = file_idx * plots_per_file
        end_idx = min(start_idx + plots_per_file, num_peptides)
        peptides_in_file = end_idx - start_idx
        
        if peptides_in_file == 0:
            continue
        
        # Calculate rows needed for this file (each peptide gets 2 subplots: chromatogram + summed spectrum)
        rows_this_file = int(np.ceil(peptides_in_file / cols_per_file))
        
        # Create figure for this batch with gridspec to allow 2 subplots per peptide
        # Each peptide will have: chromatogram (top), summed spectrum (bottom)
        fig_combined = plt.figure(figsize=(fig_width, rows_this_file * subplot_height * 2.0))  # 2x height for 2 subplots per peptide
        fig_combined.patch.set_facecolor('black')
        
        file_title = f'MS1 Chromatograms for All Peptides (Sorted by Apex Peak RT)'
        if num_files > 1:
            file_title += f' - Part {file_idx + 1}/{num_files}'
        
        fig_combined.suptitle(file_title, fontsize=14, y=0.995, fontfamily='serif', color='0.9')
        
        # Create gridspec with explicit bounds so subplots fill the figure
        gs_combined = fig_combined.add_gridspec(rows_this_file * 2, cols_per_file,
                                                 hspace=0.4, wspace=wspace,
                                                 height_ratios=[2.5, 1] * rows_this_file,  # chrom + summed spectrum
                                                 left=0.04, right=0.96, bottom=0.03, top=0.94)
        
        # Create axes arrays: one for chromatograms, one for summed spectra
        axes_combined_chrom = []
        axes_combined_spec = []
        
        for peptide_idx in range(peptides_in_file):
            row_idx = peptide_idx // cols_per_file
            col_idx = peptide_idx % cols_per_file
            
            # Chromatogram subplot (top, row 0 of this peptide's 2-row block) — black background to match overlay
            ax_chrom = fig_combined.add_subplot(gs_combined[row_idx * 2, col_idx])
            ax_chrom.set_facecolor('black')
            axes_combined_chrom.append(ax_chrom)
            
            # Summed spectrum subplot (bottom, row 1 of this peptide's 2-row block) — black background
            ax_spec = fig_combined.add_subplot(gs_combined[row_idx * 2 + 1, col_idx])
            ax_spec.set_facecolor('black')
            axes_combined_spec.append(ax_spec)
        
        # Store all axes lists
        combined_figures.append(fig_combined)
        combined_axes_list.append((axes_combined_chrom, axes_combined_spec))
        
        print(f"[DEBUG] Combined figure {file_idx + 1}/{num_files}: {rows_this_file}x{cols_per_file} ({peptides_in_file} plots)")
    
    print(f"[DEBUG] This may take a while...")
    
    # Ensure combined_axes_list is initialized even if num_files is 0 (shouldn't happen, but safety check)
    if not combined_axes_list:
        print("[DEBUG] WARNING: No combined figures created!")
        return
    
    # Build FULL cross-reference from ALL non-decoy peptides in CSV (for interference detection)
    # This includes peptides that may not be in unique_peptide_keys (e.g., filtered out earlier)
    print("[DEBUG] Building full cross-reference of ALL non-decoy peptides for interference detection...")
    
    # Identify decoy peptides (check for protein column with DECOY_ prefix, or decoy column)
    def is_decoy(row):
        """Check if peptide is a decoy"""
        # Check for decoy column
        if 'decoy' in df.columns:
            decoy_val = row.get('decoy', None)
            if pd.notna(decoy_val):
                if isinstance(decoy_val, (int, float)):
                    return decoy_val < 0 or decoy_val == -1  # Common: -1 = decoy, 1 = target
                elif isinstance(decoy_val, str):
                    return decoy_val.lower() in ['true', 'decoy', '-1']
        if 'target_decoy' in df.columns:
            td_val = row.get('target_decoy', None)
            if pd.notna(td_val):
                if isinstance(td_val, (int, float)):
                    return td_val < 0 or td_val == -1
                elif isinstance(td_val, str):
                    return td_val.lower() in ['decoy', '-1']
        # Check protein column for DECOY_ prefix
        if 'protein' in df.columns:
            protein = row.get('protein', '')
            if pd.notna(protein) and isinstance(protein, str):
                if protein.startswith('DECOY_'):
                    return True
        return False
    
    # Filter out decoys
    df_nondecoy = df[~df.apply(is_decoy, axis=1)].copy()
    print(f"[DEBUG] Filtered out decoys: {len(df)} total rows -> {len(df_nondecoy)} non-decoy rows")
    
    # Group non-decoy peptides by peptide_key
    if len(df_nondecoy) > 0:
        df_nondecoy['peptide_key'] = df_nondecoy.apply(create_peptide_key, axis=1)
        nondecoy_groups = df_nondecoy.groupby('peptide_key')
        
        # Build full cross-reference from all non-decoy peptide groups
        full_crossref = {}  # peptide_key -> {'isotope_mzs': [...], 'rt_anchor': float, 'rt_range': (min, max)}
        for peptide_key, group in nondecoy_groups:
            precursor_mz = representative_mz(group)
            if precursor_mz is None or precursor_mz <= 0:
                continue
            
            charge = group['charge'].iloc[0] if 'charge' in group.columns else 1
            isotope_mzs = isotope_mz_list(precursor_mz, charge, n_isos=4)
            
            ms1_rts_raw = group['MS1_retention_time_sec'].dropna().tolist()
            ms1_rts_table = [rt for rt in ms1_rts_raw if rt > 0]
            
            if not ms1_rts_table or not isotope_mzs:
                continue
            
            # Get anchor RT (best e-value if available and valid, else median)
            rt_anchor = None
            if 'e-value' in group.columns:
                evals = pd.to_numeric(group['e-value'], errors='coerce').dropna()
                if len(evals) > 0 and np.isfinite(evals).any():
                    best_idx = evals.idxmin()
                    rt_anchor = group.loc[best_idx, 'MS1_retention_time_sec']
                if pd.isna(rt_anchor) or rt_anchor <= 0:
                    rt_anchor = np.median(ms1_rts_table)
            else:
                rt_anchor = np.median(ms1_rts_table)
            
            rt_range = (min(ms1_rts_table), max(ms1_rts_table))
            full_crossref[peptide_key] = {
                'isotope_mzs': isotope_mzs,
                'rt_anchor': rt_anchor,
                'rt_range': rt_range,
                'precursor_mz': precursor_mz,
                'charge': charge
            }
    else:
        full_crossref = {}
    
    # Also build cross-reference for filtered peptides (for backward compatibility)
    print("[DEBUG] Building cross-reference of filtered peptide isotopic peaks and RT values...")
    peptide_crossref = {}  # peptide_key -> {'isotope_mzs': [...], 'rt_anchor': float, 'rt_range': (min, max)}
    for peptide_key in unique_peptide_keys:
        group = peptide_groups.get_group(peptide_key)
        precursor_mz = representative_mz(group)
        if precursor_mz is None or precursor_mz <= 0:
            continue
        
        charge = group['charge'].iloc[0] if 'charge' in group.columns else 1
        isotope_mzs = isotope_mz_list(precursor_mz, charge, n_isos=4)
        
        ms1_rts_raw = group['MS1_retention_time_sec'].dropna().tolist()
        ms1_rts_table = [rt for rt in ms1_rts_raw if rt > 0]
        
        if not ms1_rts_table or not isotope_mzs:
            continue
        
        # Get anchor RT (best e-value if available and valid, else median; Byonic-only CSVs have no e-value)
        rt_anchor = None
        if 'e-value' in group.columns:
            evals = pd.to_numeric(group['e-value'], errors='coerce').dropna()
            if len(evals) > 0 and np.isfinite(evals).any():
                best_idx = evals.idxmin()
                rt_anchor = group.loc[best_idx, 'MS1_retention_time_sec']
            if pd.isna(rt_anchor) or rt_anchor <= 0:
                rt_anchor = np.median(ms1_rts_table)
        else:
            rt_anchor = np.median(ms1_rts_table)
        
        rt_range = (min(ms1_rts_table), max(ms1_rts_table))
        peptide_crossref[peptide_key] = {
            'isotope_mzs': isotope_mzs,
            'rt_anchor': rt_anchor,
            'rt_range': rt_range,
            'precursor_mz': precursor_mz,
            'charge': charge
        }
    
    print(f"[DEBUG] Full cross-reference (all non-decoy): {len(full_crossref)} peptides")
    print(f"[DEBUG] Filtered cross-reference: {len(peptide_crossref)} peptides")
    
    # First pass: collect peptide-level scores (one per unique peptide, not per scan).
    # For each peptide we take the best PSM (e.g. by E-value) and use that row's E-value, Sp, Q-value, PEP for scatter plots.
    def _get_psm_val(row, col_candidates, default=np.nan):
        for c in col_candidates:
            if c in row.index:
                v = row[c]
                if pd.notna(v) and v != '':
                    try:
                        return float(v)
                    except (ValueError, TypeError):
                        pass
        return default
    all_psm_scores = []  # one entry per PSM: peptide_key, e_value, qvalue, pep (Sp score not used)
    # Case-insensitive column detection (CSV headers can vary)
    def _find_col(names):
        for n in names:
            n_lower = n.lower().replace('-', '_')
            for c in df.columns:
                if c is not None and str(c).strip().lower().replace('-', '_') == n_lower:
                    return c
        for n in names:
            if n in df.columns:
                return n
        return None
    evalue_col = _find_col(['e-value', 'e_value', 'E-value'])
    qvalue_col = _find_col(['percolator_qvalue', 'q-value', 'qvalue', 'Q-value'])
    pep_col = _find_col(['percolator_PEP', 'pep', 'PEP'])
    def _to_float(v):
        if v is None or (isinstance(v, float) and np.isnan(v)) or v == '':
            return np.nan
        try:
            return float(v)
        except (TypeError, ValueError):
            return np.nan
    print("[DEBUG] Collecting PSM-level scores (E-value, Q-value, PEP) for all peptides...")
    psm_count = 0
    for key in unique_peptide_keys:
        group = peptide_groups.get_group(key)
        for _, row in group.iterrows():
            try:
                e_val = row[evalue_col] if evalue_col and evalue_col in row.index else _get_psm_val(row, ['e-value', 'e_value', 'E-value'])
                qv = row[qvalue_col] if qvalue_col and qvalue_col in row.index else _get_psm_val(row, ['percolator_qvalue', 'q-value', 'qvalue', 'Q-value'])
                pep_val = row[pep_col] if pep_col and pep_col in row.index else _get_psm_val(row, ['percolator_PEP', 'pep', 'PEP'])
            except (KeyError, TypeError):
                e_val = _get_psm_val(row, ['e-value', 'e_value', 'E-value'])
                qv = _get_psm_val(row, ['percolator_qvalue', 'q-value', 'qvalue', 'Q-value'])
                pep_val = _get_psm_val(row, ['percolator_PEP', 'pep', 'PEP'])
            raw_e = e_val
            raw_qv = qv
            raw_pep = pep_val
            e_val = _to_float(e_val)
            qv = _to_float(qv)
            pep_val = _to_float(pep_val)
            all_psm_scores.append({
                'peptide_key': key,
                'e_value': e_val,
                'qvalue': qv,
                'pep': pep_val,
                'raw_qvalue': raw_qv,
                'raw_evalue': raw_e,
                'raw_pep': raw_pep,
            })
            psm_count += 1
    print(f"[DEBUG] Collected {len(all_psm_scores)} PSM-level scores.")
    def _invalid_reason(raw_val):
        """Classify why a score is invalid for debug messages."""
        if raw_val is None or raw_val == '' or (isinstance(raw_val, float) and np.isnan(raw_val)):
            return 'missing or NaN'
        if isinstance(raw_val, float) and np.isinf(raw_val):
            return 'inf' if raw_val > 0 else '-inf'
        try:
            float(raw_val)
            return 'invalid'
        except (TypeError, ValueError):
            return 'non-numeric'
    n_with_e = sum(1 for x in all_psm_scores if pd.notna(x.get('e_value')) and np.isfinite(x.get('e_value')))
    n_with_q = sum(1 for x in all_psm_scores if pd.notna(x.get('qvalue')) and np.isfinite(x.get('qvalue')))
    q_invalid = len(all_psm_scores) - n_with_q
    if q_invalid > 0:
        q_reasons = {}
        q_examples = []
        for x in all_psm_scores:
            if pd.isna(x.get('qvalue')) or not np.isfinite(x.get('qvalue')):
                r = _invalid_reason(x.get('raw_qvalue'))
                q_reasons[r] = q_reasons.get(r, 0) + 1
                if len(q_examples) < 5:
                    q_examples.append((x.get('peptide_key'), r, x.get('raw_qvalue')))
        reason_str = ', '.join(f"{cnt} {label}" for label, cnt in sorted(q_reasons.items(), key=lambda t: -t[1]))
        print(f"[DEBUG] Collected PSM-level scores (one per PSM) for {len(all_psm_scores)} PSMs (E-value, Sp, Q-value, PEP); {n_with_e} with valid E-value, {n_with_q} with valid Q-value ({q_invalid} invalid Q-value: {reason_str})")
        if no_psm_filter and q_invalid > 0:
            print(f"[DEBUG]   No PSM filter: all {len(all_psm_scores)} PSMs (including {q_invalid} without Q-value/PEP) are still used for plots and overlay.")
        else:
            print(f"[DEBUG]   Invalid Q-value usually means: PSM not matched to Percolator output (SpecId mismatch) and/or E-value FDR fallback not run (need decoy or protein column). Re-run add_percolator_qvalues.py on the CSV to see matched/unmatched and FDR fill.")
        for ex_key, ex_reason, ex_raw in q_examples:
            raw_str = f" (raw={ex_raw!r})" if ex_raw is not None and ex_raw != '' else ""
            print(f"[DEBUG]   Example invalid Q-value: {ex_key} -> {ex_reason}{raw_str}")
    else:
        print(f"[DEBUG] Collected PSM-level scores (one per PSM) for {len(all_psm_scores)} PSMs (E-value, Sp, Q-value, PEP); {n_with_e} with valid E-value, {n_with_q} with valid Q-value")
    e_invalid = len(all_psm_scores) - n_with_e
    if e_invalid > 0:
        e_reasons = {}
        for x in all_psm_scores:
            if pd.isna(x.get('e_value')) or not np.isfinite(x.get('e_value')):
                r = _invalid_reason(x.get('raw_evalue'))
                e_reasons[r] = e_reasons.get(r, 0) + 1
        e_reason_str = ', '.join(f"{cnt} {label}" for label, cnt in sorted(e_reasons.items(), key=lambda t: -t[1]))
        print(f"[DEBUG]   E-value: {e_invalid} invalid ({e_reason_str})")
    pep_invalid = len(all_psm_scores) - sum(1 for x in all_psm_scores if pd.notna(x.get('pep')) and np.isfinite(x.get('pep')))
    if pep_invalid > 0:
        pep_reasons = {}
        for x in all_psm_scores:
            if pd.isna(x.get('pep')) or not np.isfinite(x.get('pep')):
                r = _invalid_reason(x.get('raw_pep'))
                pep_reasons[r] = pep_reasons.get(r, 0) + 1
        pep_reason_str = ', '.join(f"{cnt} {label}" for label, cnt in sorted(pep_reasons.items(), key=lambda t: -t[1]))
        print(f"[DEBUG]   PEP: {pep_invalid} invalid ({pep_reason_str})")
    
    # First pass: extract chromatograms for ALL peptides for the overlay plot (use group directly so we don't skip any)
    n_keys = len(unique_peptide_keys)
    print(f"[DEBUG] Extracting chromatograms for all peptides (overlay plot): {n_keys} peptides...")
    all_peptide_overlay_data = []
    overlay_skipped_no_mz = 0
    overlay_skipped_empty = 0
    for key_idx, key in enumerate(unique_peptide_keys):
        if (key_idx + 1) % 50 == 0 or key_idx == 0 or key_idx == n_keys - 1:
            print(f"[DEBUG]   Overlay: peptide {key_idx + 1}/{n_keys} ({key[:40]}...)" if len(str(key)) > 40 else f"[DEBUG]   Overlay: peptide {key_idx + 1}/{n_keys} ({key})")
        try:
            group = peptide_groups.get_group(key)
            mods_raw = group['modifications'].iloc[0] if 'modifications' in group.columns else '-'
            precursor_mz = representative_mz(group)
            if precursor_mz is None or precursor_mz <= 0:
                overlay_skipped_no_mz += 1
                continue
            charge = group['charge'].iloc[0] if 'charge' in group.columns else 1
            isotope_mzs = isotope_mz_list(precursor_mz, charge, n_isos=4)
            if not isotope_mzs:
                overlay_skipped_no_mz += 1
                continue
            rts_ov, intensity_matrix_ov, _ = extract_aligned_isotope_chromatograms(raw_file, isotope_mzs, ppm_tolerance=20.0)
            if len(rts_ov) == 0:
                overlay_skipped_empty += 1
                continue
            total_ints_ov = np.sum(intensity_matrix_ov, axis=1)  # raw total for overlay lines
            proxy_area = float(np.sum(total_ints_ov))
            max_intensity = float(np.max(total_ints_ov)) if len(total_ints_ov) > 0 else 0.0
            seq_start = 999999
            if 'sequence_positions' in group.columns:
                seq_pos = group['sequence_positions'].iloc[0]
                if pd.notna(seq_pos) and seq_pos != '' and '-' in str(seq_pos):
                    try:
                        seq_start = int(str(seq_pos).split('-')[0])
                    except (ValueError, TypeError):
                        pass
            all_peptide_overlay_data.append({
                'peptide_key': key,
                'mods': mods_raw,
                'rts': np.asarray(rts_ov, dtype=float).copy(),
                'total_intensities': np.asarray(total_ints_ov, dtype=float).copy(),
                'proxy_area': proxy_area,
                'max_intensity': max_intensity,
                'seq_start': seq_start,
            })
        except Exception:
            continue
    if overlay_skipped_no_mz or overlay_skipped_empty:
        print(f"[DEBUG] Overlay skip counts: {overlay_skipped_no_mz} no m/z/isotopes, {overlay_skipped_empty} empty chromatogram")
    if all_peptide_overlay_data:
        overlay_proxy_min = min(item['proxy_area'] for item in all_peptide_overlay_data)
        overlay_proxy_max = max(item['proxy_area'] for item in all_peptide_overlay_data)
        if overlay_proxy_max <= overlay_proxy_min:
            overlay_proxy_max = overlay_proxy_min + 1.0
        # Full RT range for overlay so all peptides' peaks are visible (not zoomed to current peptide)
        overlay_rt_min_sec = min(float(np.min(item['rts'])) for item in all_peptide_overlay_data)
        overlay_rt_max_sec = max(float(np.max(item['rts'])) for item in all_peptide_overlay_data)
        if overlay_rt_max_sec <= overlay_rt_min_sec:
            overlay_rt_max_sec = overlay_rt_min_sec + 60.0
        # Global y-max across all overlay traces so all peaks are visible (not just current peptide dominating)
        overlay_ymax_global = max(float(np.max(item['total_intensities'])) for item in all_peptide_overlay_data)
        if overlay_ymax_global <= 0:
            overlay_ymax_global = 1.0
    else:
        overlay_proxy_min, overlay_proxy_max = 0.0, 1.0
        overlay_rt_min_sec, overlay_rt_max_sec = 0.0, 60.0
        overlay_ymax_global = 1.0
    print(f"[DEBUG] Overlay: chromatograms for {len(all_peptide_overlay_data)} peptides (proxy_area range {overlay_proxy_min:.2e}-{overlay_proxy_max:.2e}, RT {overlay_rt_min_sec/60:.1f}-{overlay_rt_max_sec/60:.1f} min, y_max {overlay_ymax_global:.2e})")
    # Map peptide_key -> proxy_area so rejected peptides still get a total_area for scatter/overlay coloring
    overlay_by_key = {item['peptide_key']: item['proxy_area'] for item in all_peptide_overlay_data} if all_peptide_overlay_data else {}
    
    # No intensity filter: save one individual plot per peptide (all peptides get a figure).
    num_peptides = len(unique_peptide_keys)
    
    # Combined figure slot index: only non-rejected (picked) peptides get a slot; no gaps
    picked_idx = 0
    # One entry per peptide (by peptide_key) for scatter plots so every figure shows N dots (e.g. 30 in test mode)
    all_peptide_scatter_data = {}

    # Running global total_area range for window-bar colorscale (same as RT windows / overview)
    area_vmin_global = np.inf
    area_vmax_global = -np.inf
    
    # Custom total_area colormap (same as RT windows): lavender -> purple -> dark red -> Spectral
    def _total_area_colormap_early():
        from matplotlib.colors import ListedColormap
        spectral = plt.get_cmap('Spectral')
        lavender = np.array([0.90, 0.90, 0.98, 1.0])
        purple = np.array([0.58, 0.44, 0.86, 1.0])
        dark_red = np.array([0.55, 0.0, 0.0, 1.0])
        n_low, n_high = 64, 192
        colors = []
        for i in range(n_low):
            t = i / max(n_low - 1, 1)
            if t < 0.33:
                c = lavender + (purple - lavender) * (t / 0.33)
            elif t < 0.66:
                c = purple + (dark_red - purple) * ((t - 0.33) / 0.33)
            else:
                c = dark_red + (np.array(spectral(0)[:4]) - dark_red) * ((t - 0.66) / 0.34)
            colors.append(c)
        # Cap high end at blue (spectral ~0.85) so top doesn't go into purple
        SPECTRAL_V_MAX = 0.85
        for i in range(n_high):
            v = SPECTRAL_V_MAX * (i / max(n_high - 1, 1))
            colors.append(spectral(v))
        return ListedColormap(colors, name='total_area')
    
    # Sentinel for scatter when peptide is skipped (no peak / rejected); total_area from overlay when available so dots get colored
    def _scatter_sentinel():
        return {'coelution_score': None, 'shape_corr': None, 'ratio_stability': None, 'peak_score': None, 'total_area': None}
    def _scatter_sentinel_with_overlay(key):
        e = _scatter_sentinel()
        e['total_area'] = overlay_by_key.get(key)
        return e

    def _save_early_rejected_figure(group, peptide_key, peptide_display_1based, num_peptides, rejection_reason, rejected_dir, filtering_criteria, early_rejected_paths_ref):
        """Save a minimal 'rejected' figure for peptides that hit early continue (no chromatogram, etc.) so they count as rejected.
        peptide_display_1based: 1-based index for display (e.g. 1 to num_peptides)."""
        peptide = group['plain_peptide'].iloc[0]
        charge = group['charge'].iloc[0] if 'charge' in group.columns else 1
        seq_start_pos = 99999
        if 'sequence_positions' in group.columns:
            sp = group['sequence_positions'].iloc[0]
            if pd.notna(sp) and str(sp).strip() and '-' in str(sp):
                try:
                    seq_start_pos = int(str(sp).strip().split('-')[0])
                except (ValueError, TypeError):
                    pass
        safe_peptide = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in peptide)[:40]
        reason_slug = "".join(c if c.isalnum() or c == '_' else '_' for c in rejection_reason.replace(' ', '_'))[:30]
        # Sortable name: seq_position (5), index (4); filter by sorting image names
        filename = f"{seq_start_pos:05d}_{peptide_display_1based:04d}_{safe_peptide}_z{charge}_rejected_{reason_slug}.png"
        filepath = os.path.join(rejected_dir, filename)
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111)
        ax.set_axis_off()
        ax.text(0.5, 0.7, f"Rejected: {rejection_reason}", ha='center', va='center', fontsize=14, fontfamily='serif')
        ax.text(0.5, 0.5, str(peptide_key), ha='center', va='center', fontsize=10, fontfamily='serif', wrap=True)
        ax.text(0.5, 0.25, "Filtering criteria:", ha='center', va='center', fontsize=9, fontfamily='serif', fontweight='bold')
        ax.text(0.5, 0.12, filtering_criteria, ha='center', va='center', fontsize=7, fontfamily='serif', wrap=True)
        fig.suptitle(f'Peptide {peptide_display_1based}/{num_peptides} – skipped: {rejection_reason}\n{filtering_criteria}', fontsize=9, fontfamily='serif')
        try:
            os.makedirs(rejected_dir, exist_ok=True)
            plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
            abspath = os.path.abspath(filepath)
            early_rejected_paths_ref.append(abspath)
            print(f"[DEBUG] CHROMATOGRAMS_PEPTIDES: early-rejected -> {abspath}")
            print(f"[DEBUG] Saved early-rejected plot: {abspath}")
        except Exception as e:
            print(f"[ERROR] Failed to save early-rejected plot: {os.path.abspath(filepath)}: {e}", file=sys.stderr)
        finally:
            plt.close(fig)

    # Track rejected peptide keys so pass 1 can exclude them from scatter plots, overlay, and PSM scatter
    rejected_peptide_keys = set()
    # Count rejections by reason (pass 1 only) for end-of-run summary
    rejection_counts = defaultdict(int)
    def _rejection_category(reason):
        if reason is None or not reason:
            return None
        r = str(reason).strip()
        if 'No valid m/z' in r or r == 'No valid m/z':
            return 'No valid m/z'
        if 'Need >=1' in r or 'need >=1' in r.lower() or 'Need >=2' in r or 'need >=2' in r.lower():
            return 'Need >=1 scan'
        if 'No valid MS1 RTs' in r:
            return 'No valid MS1 RTs'
        if 'No isotope m/z' in r:
            return 'No isotope m/z'
        if 'No MS1 scans' in r:
            return 'No MS1 scans'
        if 'Summed MS1' in r or 'M0, M+1' in r:
            return 'Summed MS1 M0/M+1/M+2'
        if 'Isotopic envelope' in r:
            return 'Isotopic envelope'
        if 'PSM filters' in r:
            return 'PSM filters (Q/PEP/E)'
        if 'single-AA overhang' in r or 'single-AA overhangs' in r:
            return 'No significant single-AA overhang'
        if 'Excluded:' in r and 'modification' in r.lower():
            return 'Excluded mods'
        if 'modification' in r.lower() or 'has mods' in r.lower():
            return 'Excluded mods'
        if 'Apex intensity' in r or '10^5' in r or 'noise' in r.lower():
            return 'Apex intensity < 10^5 (noise)'
        if 'proline' in r.lower():
            return 'Starts/ends with proline'
        if 'Shape correlation' in r or 'shape_corr' in r.lower():
            return 'Shape correlation < 0.8'
        if 'ppm' in r.lower() and 'theoretical' in r.lower():
            return 'Peak m/z >6 ppm from theoretical'
        return 'Other'
    # Significant-fragment data (5 ppm + 0.5% intensity): populated in pass 1 when we have MS2; merged into output CSVs.
    # Keep one canonical overhang mapping column: single_aa_overhangs_protein_positions.
    significant_data = {}  # key (plain_peptide, charge, mods_norm) -> {significant_fragment_ions, significant_fragment_pairs, single_aa_overhangs_protein_positions}
    def _norm_mods_sig(m):
        if m is None or (isinstance(m, float) and pd.isna(m)):
            return '-'
        s = str(m).strip()
        return s if s and s.lower() != 'nan' else '-'
    # Two passes: pass 0 fills all_peptide_scatter_data for every peptide so scatter plots show all N dots; pass 1 draws and saves figures
    for pass_num in [0, 1]:
        if pass_num == 0:
            print("[DEBUG] First pass: filling scatter data for all peptides (no figures saved yet; pass 2 will save)...")
        if pass_num == 1:
            print("")
            print("[DEBUG] Pass 0 complete. All scatter data collected.")
            print("[DEBUG] ========== PASS 2 (DRAW & SAVE): Writing figures to accepted/ and rejected/ ==========")
            print(f"[DEBUG] CHROMATOGRAMS_PEPTIDES: Pass 2 starting - will write one PNG per peptide to:")
            print(f"[DEBUG] CHROMATOGRAMS_PEPTIDES:   accepted -> {output_dir_abs}")
            print(f"[DEBUG] CHROMATOGRAMS_PEPTIDES:   rejected -> {rejected_dir_abs}")
            # Log which peptides we're processing (to verify list isn't always the same when using --test)
            head = unique_peptide_keys[:5]
            tail = f" ... and {len(unique_peptide_keys) - 5} more" if len(unique_peptide_keys) > 5 else ""
            print(f"[DEBUG] Peptides to process this run ({len(unique_peptide_keys)}): {head}{tail}")
            # Ensure every peptide has scatter data (backfill any missed by early continues in pass 0)
            for key in unique_peptide_keys:
                if key not in all_peptide_scatter_data:
                    all_peptide_scatter_data[key] = _scatter_sentinel_with_overlay(key)
            # Use global total_area range for overlay and scatter so colors are consistent across all plots
            if pass_num == 1:
                from matplotlib.colors import Normalize
                if area_vmax_global > area_vmin_global:
                    area_norm_global = Normalize(vmin=area_vmin_global, vmax=area_vmax_global)
                else:
                    valid_ta = []
                    for k in unique_peptide_keys:
                        ta = all_peptide_scatter_data.get(k, {}).get('total_area')
                        if ta is not None and not (isinstance(ta, float) and (np.isnan(ta) or ta < 0)):
                            try:
                                v = float(ta)
                                if np.isfinite(v) and v >= 0:
                                    valid_ta.append(v)
                            except (TypeError, ValueError):
                                pass
                    if len(valid_ta) > 1:
                        area_norm_global = Normalize(vmin=min(valid_ta), vmax=max(valid_ta))
                    elif len(valid_ta) == 1:
                        area_norm_global = Normalize(vmin=0, vmax=max(valid_ta[0], 1))
                    else:
                        area_norm_global = None
                norm_overlay = area_norm_global if area_norm_global is not None else norm_overlay
        for idx, peptide_key in enumerate(unique_peptide_keys):
                # Use 1-based peptide number for display (captured here so it can't be overwritten by nested loops)
                current_peptide_1based = idx + 1
                if pass_num == 1:
                    print(f"  Peptide {current_peptide_1based}/{num_peptides}: {peptide_key}")
            
                group = peptide_groups.get_group(peptide_key)
                peptide = group['plain_peptide'].iloc[0]  # Get the sequence for display

                # Reject peptides that start or end with proline
                seq = str(peptide).strip() if peptide else ''
                if seq and (seq[0].upper() == 'P' or seq[-1].upper() == 'P'):
                    rejected_peptide_keys.add(peptide_key)
                    if pass_num == 0:
                        all_peptide_scatter_data[peptide_key] = _scatter_sentinel_with_overlay(peptide_key)
                    elif pass_num == 1:
                        _save_early_rejected_figure(group, peptide_key, current_peptide_1based, num_peptides,
                            "Peptide starts or ends with proline",
                            rejected_dir, FILTERING_CRITERIA, early_rejected_paths)
                        rejection_counts['Starts/ends with proline'] += 1
                        print(f"[DEBUG] Peptide {current_peptide_1based}/{num_peptides}: {peptide_key} -> early-rejected (starts or ends with proline)")
                    continue

                # Reject peptides with modifications
                mods = group['modifications'].iloc[0] if 'modifications' in group.columns else '-'
                if _has_modifications_for_reject(mods):
                    rejected_peptide_keys.add(peptide_key)
                    if pass_num == 0:
                        all_peptide_scatter_data[peptide_key] = _scatter_sentinel_with_overlay(peptide_key)
                    elif pass_num == 1:
                        _save_early_rejected_figure(group, peptide_key, current_peptide_1based, num_peptides,
                            "Peptide has modifications",
                            rejected_dir, FILTERING_CRITERIA, early_rejected_paths)
                        rejection_counts['Excluded mods'] += 1
                        print(f"[DEBUG] Peptide {current_peptide_1based}/{num_peptides}: {peptide_key} -> early-rejected (has modifications)")
                    continue

                # Representative (monoisotopic) m/z for XIC: min m/z in group so one plot per peptide.
                precursor_mz = representative_mz(group)
                if precursor_mz is None or precursor_mz <= 0:
                    rejected_peptide_keys.add(peptide_key)
                    if pass_num == 0:
                        all_peptide_scatter_data[peptide_key] = _scatter_sentinel_with_overlay(peptide_key)
                    elif pass_num == 1:
                        _save_early_rejected_figure(group, peptide_key, current_peptide_1based, num_peptides, "No valid m/z", rejected_dir, FILTERING_CRITERIA, early_rejected_paths)
                        rejection_counts['No valid m/z'] += 1
                        print(f"[DEBUG] Peptide {current_peptide_1based}/{num_peptides}: {peptide_key} -> early-rejected (No valid m/z)")
                    print(f"[DEBUG]   Warning: No valid m/z for peptide {peptide}, skipping")
                    continue
            
                # Get MS1 retention times from table (one per spectrum/row for this unique peptide)
                ms1_rts_raw = group['MS1_retention_time_sec'].dropna().tolist()
                ms1_rts_table = [rt for rt in ms1_rts_raw if rt > 0]  # Filter out zeros and negatives
                n_spectra = len(ms1_rts_table)
                if idx < 5:
                    print(f"[DEBUG]   Unique peptide {peptide[:25]}...: {n_spectra} spectra -> drawing {n_spectra} RT lines")
                if n_spectra < MIN_SCANS_REQUIRED:
                    rejected_peptide_keys.add(peptide_key)
                    if pass_num == 0:
                        all_peptide_scatter_data[peptide_key] = _scatter_sentinel_with_overlay(peptide_key)
                    elif pass_num == 1:
                        _save_early_rejected_figure(group, peptide_key, current_peptide_1based, num_peptides,
                            f"Need >=1 scan with MS1 RT (has {n_spectra})",
                            rejected_dir, FILTERING_CRITERIA, early_rejected_paths)
                        rejection_counts['Need >=1 scan'] += 1
                        print(f"[DEBUG] Peptide {current_peptide_1based}/{num_peptides}: {peptide_key} -> early-rejected (need >=1 scan, has {n_spectra})")
                    print(f"[DEBUG]   Warning: peptide has {n_spectra} scans (need >=1 with MS1 RT), skipping")
                    continue
                if n_spectra >= 2 and ms1_rts_table and idx < 3:
                    rt_span_sec = max(ms1_rts_table) - min(ms1_rts_table)
                    print(f"    RT range {min(ms1_rts_table):.0f}-{max(ms1_rts_table):.0f} s (span {rt_span_sec:.0f} s)")
            
                if not ms1_rts_table:
                    rejected_peptide_keys.add(peptide_key)
                    if pass_num == 0:
                        all_peptide_scatter_data[peptide_key] = _scatter_sentinel_with_overlay(peptide_key)
                    elif pass_num == 1:
                        _save_early_rejected_figure(group, peptide_key, current_peptide_1based, num_peptides, "No valid MS1 RTs", rejected_dir, FILTERING_CRITERIA, early_rejected_paths)
                        rejection_counts['No valid MS1 RTs'] += 1
                        print(f"[DEBUG] Peptide {current_peptide_1based}/{num_peptides}: {peptide_key} -> early-rejected (No valid MS1 RTs)")
                    print(f"[DEBUG]   Warning: No valid MS1 RTs for peptide {peptide}, skipping")
                    print(f"[DEBUG]     Raw MS1 RTs: {ms1_rts_raw}")
                    continue
            
                # Calculate isotope m/z values (M, M+1, M+2, M+3) for Skyline-like extraction
                charge = group['charge'].iloc[0] if 'charge' in group.columns else 1
                mods = group['modifications'].iloc[0] if 'modifications' in group.columns else '-'
                isotope_mzs = isotope_mz_list(precursor_mz, charge, n_isos=4)
            
                if not isotope_mzs:
                    rejected_peptide_keys.add(peptide_key)
                    if pass_num == 0:
                        all_peptide_scatter_data[peptide_key] = _scatter_sentinel_with_overlay(peptide_key)
                    elif pass_num == 1:
                        _save_early_rejected_figure(group, peptide_key, current_peptide_1based, num_peptides, "No isotope m/z", rejected_dir, FILTERING_CRITERIA, early_rejected_paths)
                        rejection_counts['No isotope m/z'] += 1
                        print(f"[DEBUG] Peptide {current_peptide_1based}/{num_peptides}: {peptide_key} -> early-rejected (No isotope m/z)")
                    print(f"[DEBUG]   Warning: Could not calculate isotope m/z values for peptide {peptide}, skipping")
                    continue
            
                # Extract aligned isotope chromatograms from mzML (Skyline-like: aligned RT grid, zeros preserved).
                # Same extraction used for: main chromatogram plot (ax), overlay traces (all_peptide_overlay_data), and window bar (min_rt/max_rt).
                rts, intensity_matrix, measured_mz_matrix = extract_aligned_isotope_chromatograms(raw_file, isotope_mzs, ppm_tolerance=20.0)
            
                if len(rts) == 0:
                    rejected_peptide_keys.add(peptide_key)
                    if pass_num == 0:
                        all_peptide_scatter_data[peptide_key] = _scatter_sentinel_with_overlay(peptide_key)
                    elif pass_num == 1:
                        _save_early_rejected_figure(group, peptide_key, current_peptide_1based, num_peptides, "No MS1 scans", rejected_dir, FILTERING_CRITERIA, early_rejected_paths)
                        rejection_counts['No MS1 scans'] += 1
                        print(f"[DEBUG] Peptide {current_peptide_1based}/{num_peptides}: {peptide_key} -> early-rejected (No MS1 scans)")
                    print(f"[DEBUG]   Warning: No MS1 scans found for peptide {peptide}, skipping")
                    continue
            
                # Create a new figure for this peptide (individual): chromatogram, spectrum, window bar, overlay; no scatter plots.
                fig = plt.figure(figsize=(16, 28), constrained_layout=False)
                fig.patch.set_facecolor('black')
                # Gridspec: left column = chrom, window bar, overlay (linear), overlay (log); right column = MS1 + MS2 stacked, then sequence panel.
                gs = fig.add_gridspec(4, 2, height_ratios=[2.5, 0.28, 2.5, 2.5], width_ratios=[1.5, 1], hspace=0.24, wspace=0.32, bottom=0.05, top=0.88, left=0.06, right=0.98)
                ax = fig.add_subplot(gs[0, 0])  # Chromatogram (left)
                ax_windows = fig.add_subplot(gs[1, 0], sharex=ax)  # Window bar (left)
                ax_overlay = fig.add_subplot(gs[2, 0], sharex=ax)  # All-peptides overlay linear (left); own y-scale so traces aren't squashed
                ax_overlay_log = fig.add_subplot(gs[3, 0], sharex=ax)  # All-peptides overlay log (left)
                for _ax in (ax, ax_windows, ax_overlay, ax_overlay_log):
                    _ax.set_facecolor('black')
                # Right column: MS1 + MS2 stacked (top), sequence panel (bottom) — black background for entire figure
                gs_right_spectra = gs[0:2, 1].subgridspec(2, 1, hspace=0.35, height_ratios=[1, 1])
                ax_spec = fig.add_subplot(gs_right_spectra[0])   # Summed MS1 spectrum (top)
                ax_spec_ms2 = fig.add_subplot(gs_right_spectra[1])  # Summed MS2 spectrum (bottom)
                ax_seq = fig.add_subplot(gs[2:4, 1])  # Sequence panel spans lower-right (c-ions, sequence, z-ions) — fills corner for long peptides
                for _ax in (ax_spec, ax_spec_ms2, ax_seq):
                    _ax.set_facecolor('black')
        
                # Set explicit axis limits immediately to prevent resizing
                # These will be updated later with actual data ranges, but initial limits prevent layout shifts
                ax.set_xlim([0, 20])  # Initial x-limits (will be updated)
                ax.set_ylim([0, 1])  # Initial y-limits (will be updated)
                ax_spec.set_xlim([0, 1000])  # Initial x-limits (will be updated)
                ax_spec.set_ylim([0, 1])  # Initial y-limits (will be updated)
                ax_spec_ms2.set_xlim([0, 500])  # MS2 m/z range (will be updated)
                ax_spec_ms2.set_ylim([0, 1])  # Initial y-limits (will be updated)
        
                # Combined-figure axes are assigned only for non-rejected peptides (using picked_idx); set after rejection check
        
                # Compute Total trace as sum across isotopes on aligned grid (no interpolation needed)
                total_intensities = np.sum(intensity_matrix, axis=1)
        
                # overlay_data and chrom_zoom_data are filled only for accepted peptides (when we increment picked_idx)
        
                # Get monoisotopic chromatogram for backward compatibility (for old functions)
                chromatogram = list(zip(rts, intensity_matrix[:, 0]))
        
                # Find anchor RT (feature apex, chrom apex, or median of table RTs)
                # This will be used for visualization and to guide cluster selection
                feature_apex_rt, feature_intensity = find_feature_apex_rt(
                feature_map, precursor_mz, ms1_rts_table=ms1_rts_table, mz_tolerance_ppm=20.0, rt_tolerance_sec=60.0
                )
        
                # Find anchor RT: use median of table MS1 RTs or best-scoring spectrum RT (Skyline-like)
                # Also get E-value for color-coding the anchor RT line
                # Also collect fragment counts for each RT value
                anchor_evalue = None
                rt_to_frag_count = {}  # Map RT value to fragment count for labeling
        
                # Helper function to count fragments from various possible column formats
                def count_fragments_from_row(row):
                    """Count number of matched fragments from various column formats"""
                    frag_count = 0
                    # Try different possible column names
                    for col_name in ['significant_frags', 'significant_fragment_ions', 'refined_significant_frags']:
                        if col_name in group.columns:
                            frag_val = row[col_name] if isinstance(row, pd.Series) else row.get(col_name, '')
                            if pd.notna(frag_val) and frag_val != '':
                                frag_str = str(frag_val).strip()
                                # Remove quotes if present
                                if frag_str.startswith('"') and frag_str.endswith('"'):
                                    frag_str = frag_str[1:-1]
                                # Count comma-separated items
                                if frag_str:
                                    frag_count = len([x for x in frag_str.split(',') if x.strip() != ''])
                                    break
                    return frag_count
        
                # Build RT to fragment count mapping
                # For each unique RT value, get the maximum fragment count across all PSMs at that RT
                for row_idx, row in group.iterrows():
                    rt_val = row['MS1_retention_time_sec']
                    if pd.notna(rt_val) and rt_val > 0:
                        frag_count = count_fragments_from_row(row)
                        if rt_val not in rt_to_frag_count or frag_count > rt_to_frag_count[rt_val]:
                            rt_to_frag_count[rt_val] = frag_count
        
                if 'e-value' in group.columns:
                    # Anchor RT = RT of the spectrum (row) with the lowest E-value (used as prior to pick which chromatographic peak)
                    # The matched peak (apex_rt) is then chosen from ALL detected peaks in the XIC and scored by
                    # coelution, shape, ratio stability, table RT coverage, and distance to this anchor.
                    evals = pd.to_numeric(group['e-value'], errors='coerce').dropna()
                    if len(evals) > 0 and np.isfinite(evals).any():
                        best_idx = evals.idxmin()
                        rt_anchor = group.loc[best_idx, 'MS1_retention_time_sec']
                        anchor_evalue = group.loc[best_idx, 'e-value']
                    else:
                        rt_anchor = np.median(ms1_rts_table)
                        anchor_evalue = None
                    if pd.isna(rt_anchor) or rt_anchor <= 0:
                        rt_anchor = np.median(ms1_rts_table)
                        # Get E-value closest to median RT
                        if rt_anchor > 0 and len(ms1_rts_table) > 0:
                            group_with_rt = group[group['MS1_retention_time_sec'].notna()]
                            if len(group_with_rt) > 0:
                                rt_diffs = abs(group_with_rt['MS1_retention_time_sec'] - rt_anchor)
                                closest_to_median = rt_diffs.idxmin()
                                anchor_evalue = group_with_rt.loc[closest_to_median, 'e-value'] if pd.notna(group_with_rt.loc[closest_to_median, 'e-value']) else None
                else:
                    rt_anchor = np.median(ms1_rts_table)
                    anchor_evalue = None
        
                # (No E-value threshold: do not reject based on E-value or RT=0+E-value)

                if idx < 3:
                    print(f"[DEBUG]   Peptide {peptide[:30]}: Anchor RT={rt_anchor:.1f}s")
        
                if not chromatogram:
                    if pass_num == 0:
                        all_peptide_scatter_data[peptide_key] = _scatter_sentinel_with_overlay(peptide_key)
                    elif pass_num == 1:
                        plt.close(fig)
                        _save_early_rejected_figure(group, peptide_key, current_peptide_1based, num_peptides, "No chromatogram data", rejected_dir, FILTERING_CRITERIA, early_rejected_paths)
                        print(f"[DEBUG] Peptide {current_peptide_1based}/{num_peptides}: {peptide_key} -> early-rejected (No chromatogram data)")
                    print(f"[DEBUG]   Warning: No chromatogram data for peptide {peptide}, skipping")
                    continue
        
                # Baseline correct and smooth Total trace for peak picking (Skyline-style)
                # Use percentile method for more robust baseline correction
                total_baseline_corrected = baseline_correct_trace(total_intensities, method='percentile')
                total_smoothed = smooth_trace(total_baseline_corrected, window_size=3)
        
                # Detect multiple chromatographic peaks on summed trace (Skyline-style)
                # Explore +/-90 s around anchor RT (from CSV) to find best peak
                window_span_sec = 180.0  # 90 s each side of anchor
                peak_candidates = detect_chromatographic_peaks_windowed(
                rts, total_smoothed,
                rt_anchor=rt_anchor,
                window_sec=window_span_sec,
                max_peaks=10,
                min_prominence=None,  # Auto-detect from window data
                min_width_scans=2  # Very relaxed: handles narrow/gappy peaks
                )
        
                if idx < 2 and len(rts) > 1:
                    dt = np.median(np.diff(rts))
                    print(f"    Scan spacing {dt:.2f}s, min_width_scans=2")
        
                # If no peaks found on smoothed trace, try baseline-corrected (less smoothing)
                if not peak_candidates and len(total_baseline_corrected) > 0:
                    peak_candidates = detect_chromatographic_peaks_windowed(
                        rts, total_baseline_corrected,
                        rt_anchor=rt_anchor,
                        window_sec=window_span_sec,
                        max_peaks=10,
                        min_prominence=None,
                        min_width_scans=2
                    )
                if peak_candidates and idx < 2:
                    print(f"    Found {len(peak_candidates)} peak(s) (baseline-corrected trace)")
        
                if not peak_candidates:
                    # Debug: check why no peaks detected
                    max_intensity = np.max(total_smoothed) if len(total_smoothed) > 0 else 0
                    
                    if idx < 3:
                        print(f"    No peaks for {peptide[:20]}: max_intensity={max_intensity:.1f}, non-zero points {np.sum(total_smoothed > 0)}/{len(total_smoothed)}")
                
                # Try with more lenient parameters if initial detection failed
                if max_intensity > 0:
                    # Very relaxed prominence: use height-based instead of prominence-based
                    # Try using 'height' parameter which is simpler than prominence
                    relaxed_height = max_intensity * 0.005  # 0.5% of max
                
                    # Also relax width requirement
                    if len(rts) > 1:
                        dt = np.median(np.diff(rts))
                        relaxed_width = max(3, int(5.0 / dt))  # Minimum 5 sec instead of 10 sec
                    else:
                        relaxed_width = 3
                
                    # Try with height-based detection (simpler than prominence)
                    try:
                        # First try with width requirement
                        peaks_all, props = signal.find_peaks(
                            total_smoothed,
                            height=relaxed_height,
                            width=relaxed_width,
                            distance=relaxed_width
                        )
                    
                        # If that fails, try without width requirement (just height and distance)
                        if len(peaks_all) == 0:
                            peaks_all, props = signal.find_peaks(
                                total_smoothed,
                                height=relaxed_height,
                                distance=relaxed_width
                            )
                    
                        # If still no peaks, try even simpler: just local maxima above threshold
                        if len(peaks_all) == 0:
                            local_maxima = signal.argrelextrema(total_smoothed, np.greater, order=5)[0]
                            peaks_all = local_maxima[total_smoothed[local_maxima] > relaxed_height]
                            if len(peaks_all) > 0:
                                # Create minimal properties dict
                                props = {'prominences': total_smoothed[peaks_all] - relaxed_height}
                    
                        if len(peaks_all) > 0:
                            # Convert to candidate format
                            heights = total_smoothed[peaks_all]
                            order = np.argsort(heights)[::-1][:10]
                            peaks = peaks_all[order]
                        
                            # Try to get widths, but handle gracefully if it fails
                            # Use same rel_height as main path for consistent integration
                            try:
                                widths, width_heights, left_ips, right_ips = signal.peak_widths(
                                    total_smoothed, peaks, rel_height=REL_HEIGHT_INTEGRATION
                                )
                            except:
                                # Fallback: use fixed width around peak
                                widths = np.full(len(peaks), relaxed_width)
                                left_ips = peaks - relaxed_width / 2
                                right_ips = peaks + relaxed_width / 2
                        
                            peak_candidates = []
                            for i, apex_idx in enumerate(peaks):
                                apex_idx = int(apex_idx)
                                li = int(np.floor(left_ips[i]))
                                ri = int(np.ceil(right_ips[i]))
                                li = max(0, li)
                                ri = min(len(rts) - 1, ri)
                            
                                prominence_val = float(props.get('prominences', [total_smoothed[apex_idx] - relaxed_height])[i if i < len(props.get('prominences', [])) else 0])
                            
                                peak_candidates.append({
                                    'apex_idx': apex_idx,
                                    'apex_rt': float(rts[apex_idx]),
                                    'apex_intensity': float(total_smoothed[apex_idx]),
                                    'left_base_idx': li,
                                    'right_base_idx': ri,
                                    'left_base_rt': float(rts[li]),
                                    'right_base_rt': float(rts[ri]),
                                    'collection_left_base_rt': float(rts[li]),
                                    'collection_right_base_rt': float(rts[ri]),
                                    'prominence': prominence_val,
                                    'width_scans': float(widths[i])
                                })
                        
                            if peak_candidates and idx < 2:
                                print(f"    Height-based detection found {len(peak_candidates)} peak(s)")
                    except Exception:
                        pass
                    if not peak_candidates:
                        if idx < 3:
                            print(f"    No peaks for {peptide[:20]}, using fallback window")
                        # Find max over full chromatogram range (consistent with full-range detection)
                        w0 = float(np.min(rts))
                        w1 = float(np.max(rts))
                        mask_window = (rts >= w0) & (rts <= w1)
                        if np.sum(mask_window) > 0:
                            y_window = total_smoothed[mask_window]
                            max_idx_local = np.argmax(y_window)
                            global_indices = np.where(mask_window)[0]
                            apex_idx_global = int(global_indices[max_idx_local])
                            # Use fixed width around apex
                            half_width = max(5, int(10.0 / np.median(np.diff(rts))))  # ~10 seconds
                            left_global = max(0, apex_idx_global - half_width)
                            right_global = min(len(rts) - 1, apex_idx_global + half_width)
                            peak_candidates = [{
                                'apex_idx': apex_idx_global,
                                'apex_rt': float(rts[apex_idx_global]),
                                'apex_intensity': float(total_smoothed[apex_idx_global]),
                                'left_base_idx': left_global,
                                'right_base_idx': right_global,
                                'left_base_rt': float(rts[left_global]),
                                'right_base_rt': float(rts[right_global]),
                                'collection_left_base_rt': float(rts[left_global]),
                                'collection_right_base_rt': float(rts[right_global]),
                                'prominence': float(total_smoothed[apex_idx_global]),
                                'width_scans': float(half_width * 2)
                            }]
                        else:
                            if pass_num == 0:
                                all_peptide_scatter_data[peptide_key] = _scatter_sentinel_with_overlay(peptide_key)
                            elif pass_num == 1:
                                _save_early_rejected_figure(group, peptide_key, current_peptide_1based, num_peptides, "No data in window", rejected_dir, FILTERING_CRITERIA, early_rejected_paths)
                                print(f"[DEBUG] Peptide {current_peptide_1based}/{num_peptides}: {peptide_key} -> early-rejected (No data in window)")
                            print(f"[DEBUG]   Error: No data in window for peptide {peptide}, skipping")
                            plt.close(fig)
                            continue
        
                # Process isotope traces for scoring (same processing as detection)
                # Baseline correct and smooth each isotope trace globally for consistent scoring
                intensity_matrix_processed = np.zeros_like(intensity_matrix)
                for iso_idx in range(intensity_matrix.shape[1]):
                    iso_trace = intensity_matrix[:, iso_idx]
                    # Baseline correct using percentile method (more robust than rolling min)
                    non_zero = iso_trace[iso_trace > 0]
                    if len(non_zero) > 0:
                        baseline = np.percentile(non_zero, 5)  # 5th percentile as baseline
                    else:
                        baseline = 0.0
                    iso_baseline_corrected = np.maximum(iso_trace - baseline, 0.0)
                    # Light smoothing (same as detection)
                    iso_smoothed = smooth_trace(iso_baseline_corrected, window_size=3)
                    intensity_matrix_processed[:, iso_idx] = iso_smoothed
        
                # Recompute total_smoothed from processed isotope traces for consistency
                total_smoothed_processed = np.sum(intensity_matrix_processed, axis=1)
        
                # Score each candidate peak using Skyline metrics (on processed traces)
                scored_candidates = []
                n_skip_invalid = 0
                n_skip_no_plateau = 0
                for cand_idx, candidate in enumerate(peak_candidates):
                    start_idx_candidate = candidate['left_base_idx']
                    end_idx_candidate = candidate['right_base_idx']
                    apex_rt_candidate = candidate['apex_rt']
                    apex_idx_candidate = candidate.get('apex_idx', None)  # Get apex_idx if available
                    
                    if end_idx_candidate <= start_idx_candidate:
                        n_skip_invalid += 1
                        continue
                    # No plateau on log scale => noise spike; exclude from candidates
                    if not has_log_plateau(rts, total_smoothed_processed, start_idx_candidate, end_idx_candidate, apex_idx_candidate):
                        n_skip_no_plateau += 1
                        continue
                    
                    # If apex_idx not in candidate, find it from apex_rt
                    if apex_idx_candidate is None:
                        # Find the index closest to apex_rt
                        apex_idx_candidate = np.argmin(np.abs(rts - apex_rt_candidate))
                    
                    # Skyline scoring metrics (using processed traces):
                    # A) Co-elution variance (lower is better, converted to score)
                    coelution_var, coelution_score = compute_coelution_variance(
                        intensity_matrix_processed, start_idx_candidate, end_idx_candidate, rts
                    )
                    
                    # B) Shape similarity (correlation to summed trace)
                    # Use processed total trace for consistency
                    shape_corr = compute_shape_similarity(
                        intensity_matrix_processed, total_smoothed_processed, start_idx_candidate, end_idx_candidate
                    )
                    
                    # C) Isotope ratio stability (on processed traces)
                    ratio_stability = compute_isotope_ratio_stability(
                        intensity_matrix_processed, start_idx_candidate, end_idx_candidate
                    )
                    
                    # D) RT prior - disabled: prefer highest/best peak regardless of anchor RT distance
                    rt_offset = abs(apex_rt_candidate - rt_anchor) if rt_anchor else 0.0
                    rt_prior_sigma = 30.0
                    rt_prior_score = 0.5  # Neutral: no penalty for being far from anchor (was: exp(-rt_offset²/(2σ²)))
                    
                    # D2) Table RT coverage: fraction of all scan RTs that fall inside this candidate window
                    n_table_inside = sum(1 for rt in ms1_rts_table if rt > 0 and candidate['left_base_rt'] <= rt <= candidate['right_base_rt'])
                    table_rt_coverage = n_table_inside / len(ms1_rts_table) if ms1_rts_table else 0.5
                    
                    # E) Peak quality (S/N: prominence / noise)
                    # Estimate noise robustly from the candidate window (use low percentile, not 10th)
                    window_mask_candidate = (rts >= candidate['left_base_rt']) & (rts <= candidate['right_base_rt'])
                    window_trace = total_smoothed_processed[window_mask_candidate] if len(total_smoothed_processed) > 0 else total_smoothed[window_mask_candidate]
                    if len(window_trace) > 0:
                        noise = np.percentile(window_trace, 25)  # 25th percentile (more robust)
                        noise = min(noise, np.max(window_trace) * 0.1)  # Cap at 10% of max
                    else:
                        noise = 1.0
                    peak_quality_raw = candidate['prominence'] / (noise + 1e-12)
                    # Saturating score: 1.0 - exp(-quality/10)
                    peak_quality_score = 1.0 - np.exp(-peak_quality_raw / 10.0)
                    
                    # Compute peak area for normalization
                    window_mask_candidate = (rts >= candidate['left_base_rt']) & (rts <= candidate['right_base_rt'])
                    window_total_candidate = total_intensities[window_mask_candidate]
                    peak_area = np.sum(window_total_candidate) if len(window_total_candidate) > 0 else 0
                    log_area = np.log(peak_area + 1.0)
                    
                    # Composite score (Skyline-like weights - isotope evidence dominates)
                    # Co-elution (0.35), Shape (0.35), Ratio stability (0.15), RT prior (0.05 tie-breaker), Quality (0.10)
                    # Skyline principle: trust isotope chemistry more than RT anchor
                    
                    # Safeguard: if shape evidence is very strong, reduce RT penalty
                    strong_isotope_evidence = (shape_corr > 0.85)
                    if strong_isotope_evidence:
                        # Boost score for strong isotope evidence (Skyline: trust chemistry over anchor)
                        rt_prior_boost = rt_prior_score * 0.5  # Reduce RT penalty impact
                    else:
                        rt_prior_boost = rt_prior_score
                    
                    # Weights: co-elution + shape + ratio + table RT coverage + RT prior + quality + peak strength
                    # (strength added after loop when we have max_apex/max_area; base weights sum to 0.92)
                    composite_score = (
                        0.28 * coelution_score +      # Isotope co-elution
                        0.28 * shape_corr +           # Shape similarity
                        0.14 * ratio_stability +     # Ratio consistency
                        0.12 * table_rt_coverage +   # Favor peak that contains most table RTs
                        0.05 * rt_prior_boost +      # Tie-breaker
                        0.05 * peak_quality_score   # Quality check
                    )
                    
                    collection_min_rt_cand = candidate.get('collection_left_base_rt', candidate['left_base_rt'])
                    collection_max_rt_cand = candidate.get('collection_right_base_rt', candidate['right_base_rt'])
                    scored_candidates.append({
                        'candidate_idx': cand_idx,
                        'min_rt': candidate['left_base_rt'],
                        'max_rt': candidate['right_base_rt'],
                        'collection_min_rt': collection_min_rt_cand,
                        'collection_max_rt': collection_max_rt_cand,
                        'apex_rt': apex_rt_candidate,
                        'apex_idx': apex_idx_candidate,  # Add apex_idx to scored candidate
                        'apex_intensity': candidate['apex_intensity'],
                        'start_idx': start_idx_candidate,
                        'end_idx': end_idx_candidate,
                        'coelution_var': coelution_var,
                        'coelution_score': coelution_score,
                        'shape_corr': shape_corr,
                        'ratio_stability': ratio_stability,
                        'rt_offset': rt_offset,
                        'rt_prior_score': rt_prior_score,
                        'table_rt_coverage': table_rt_coverage,
                        'peak_quality': peak_quality_raw,
                        'peak_quality_score': peak_quality_score,
                        'peak_area': peak_area,
                        'score': composite_score
                    })
        
                if not scored_candidates:
                    n_cand = len(peak_candidates)
                    reason = (
                        f"no peaks detected" if n_cand == 0 else
                        f"{n_cand} candidate(s) rejected: {n_skip_invalid} invalid window, {n_skip_no_plateau} no log plateau (single-point spike)"
                    )
                    if pass_num == 0:
                        all_peptide_scatter_data[peptide_key] = _scatter_sentinel_with_overlay(peptide_key)
                    elif pass_num == 1:
                        _save_early_rejected_figure(group, peptide_key, current_peptide_1based, num_peptides, "No scored candidate peaks", rejected_dir, FILTERING_CRITERIA, early_rejected_paths)
                        print(f"[DEBUG] Peptide {current_peptide_1based}/{num_peptides}: {peptide_key} -> early-rejected (No scored candidate peaks)")
                    print(f"  No scored peaks for {peptide[:25]}: {reason}")
                    plt.close(fig)
                    continue

                # Add peak strength term so real signal (apex + area) is prioritized: stronger peak can outweigh RT/table bias
                max_apex = max(c['apex_intensity'] for c in scored_candidates)
                max_area = max(c['peak_area'] for c in scored_candidates) or 1.0
                STRENGTH_WEIGHT = 0.20  # Prioritize peaks with real signal (XIC-matched) when multiple peaks pass apex >= 10^5
                for c in scored_candidates:
                    strength_score = 0.5 * (
                        (c['apex_intensity'] / (max_apex + 1e-12)) +
                        (c['peak_area'] / (max_area + 1e-12))
                    )
                    c['strength_score'] = strength_score
                    c['score'] = c['score'] + STRENGTH_WEIGHT * strength_score
        
                # Sort by score (highest first)
                scored_candidates.sort(key=lambda x: x['score'], reverse=True)
                # Restrict to peaks with apex intensity >= MIN_APEX_INTENSITY (avoid assigning noise); pretty much never select below 10^5
                acceptable_candidates = [c for c in scored_candidates if c['apex_intensity'] >= MIN_APEX_INTENSITY]
                if not acceptable_candidates:
                    if pass_num == 0:
                        all_peptide_scatter_data[peptide_key] = _scatter_sentinel_with_overlay(peptide_key)
                    elif pass_num == 1:
                        _save_early_rejected_figure(group, peptide_key, current_peptide_1based, num_peptides, "Apex intensity < 10^5 (noise); no higher peak", rejected_dir, FILTERING_CRITERIA, early_rejected_paths)
                        rejection_counts['Apex intensity < 10^5 (noise)'] += 1
                        print(f"[DEBUG] Peptide {current_peptide_1based}/{num_peptides}: {peptide_key} -> early-rejected (no peak >= 10^5)")
                    plt.close(fig)
                    continue
                # Best = highest score among peaks >= 10^5; tie-break by apex so we prioritize real signal when scores are close
                best_peak = acceptable_candidates[0]
                top_score = best_peak['score']
                for c in acceptable_candidates[1:]:
                    if c['apex_intensity'] > best_peak['apex_intensity'] and c['score'] >= top_score - 0.12:
                        best_peak = c
                # Drift buffer (collection ± 30s) is for instrument calibration: when re-measuring, RT may shift by that much.
                # It is NOT used for peak selection—we pick the best peak regardless of drift.
                # Apex dominance override: when anchor is wrong, a clearly dominant peak (>> intensity) should win
                # If the highest-apex peak has apex > 5x the selected peak's apex, prefer it (dominant signal over anchor bias)
                APEX_DOMINANCE_RATIO = 5.0
                max_apex_candidate = max(acceptable_candidates, key=lambda c: c['apex_intensity'])
                if max_apex_candidate['apex_intensity'] > APEX_DOMINANCE_RATIO * best_peak['apex_intensity']:
                    best_peak = max_apex_candidate
                apex_rt = best_peak['apex_rt']
                apex_idx = best_peak['apex_idx']
                # ppm at apex (monoisotope): measured m/z vs theoretical
                ppm_at_apex = None
                if measured_mz_matrix is not None and precursor_mz and precursor_mz > 0 and apex_idx is not None:
                    if apex_idx < measured_mz_matrix.shape[0] and measured_mz_matrix.shape[1] > 0:
                        measured_mz_apex = float(measured_mz_matrix[apex_idx, 0])
                        if not np.isnan(measured_mz_apex) and measured_mz_apex > 0:
                            ppm_at_apex = abs(measured_mz_apex - precursor_mz) / precursor_mz * 1e6
        
                # (No E-value threshold: do not reject based on E-value)

                # Skyline-like window strategy: apex-centered, FWHM-based, resistant to merged peaks
                # 1. Compute FWHM (Full Width at Half Maximum) at the selected apex
                apex_intensity = total_smoothed[apex_idx]
                half_max = apex_intensity / 2.0
        
                # Find FWHM boundaries by walking left and right from apex
                fwhm_left_idx = apex_idx
                fwhm_right_idx = apex_idx
        
                # Walk left to find where intensity drops to half-max (interpolate if needed)
                found_left = False
                for i in range(apex_idx, -1, -1):
                    if i == 0:
                        fwhm_left_idx = 0
                        break
                    if total_smoothed[i] <= half_max:
                        # Interpolate between this point and previous
                        if i < apex_idx:
                            # Linear interpolation
                            y1 = total_smoothed[i]
                            y2 = total_smoothed[i+1] if i+1 < len(total_smoothed) else y1
                            if y2 > y1:
                                frac = (half_max - y1) / (y2 - y1) if (y2 - y1) > 0 else 0
                                fwhm_left_idx = i + frac
                            else:
                                fwhm_left_idx = i
                        else:
                            fwhm_left_idx = apex_idx
                        found_left = True
                        break
        
                if not found_left:
                    fwhm_left_idx = 0
        
                # Walk right to find where intensity drops to half-max (interpolate if needed)
                found_right = False
                for i in range(apex_idx, len(total_smoothed)):
                    if i == len(total_smoothed) - 1:
                        fwhm_right_idx = len(total_smoothed) - 1
                        break
                    if total_smoothed[i] <= half_max:
                        # Interpolate between previous point and this one
                        if i > apex_idx:
                            y1 = total_smoothed[i-1] if i-1 >= 0 else total_smoothed[i]
                            y2 = total_smoothed[i]
                            if y1 > y2:
                                frac = (half_max - y2) / (y1 - y2) if (y1 - y2) > 0 else 0
                                fwhm_right_idx = i - frac
                            else:
                                fwhm_right_idx = i
                        else:
                            fwhm_right_idx = apex_idx
                        found_right = True
                        break
        
                if not found_right:
                    fwhm_right_idx = len(total_smoothed) - 1
        
                # Calculate FWHM in seconds (interpolate RT if needed)
                if fwhm_left_idx < len(rts):
                    if fwhm_left_idx == int(fwhm_left_idx):
                        fwhm_left_rt = rts[int(fwhm_left_idx)]
                    else:
                        # Interpolate RT
                        idx_low = int(np.floor(fwhm_left_idx))
                        idx_high = int(np.ceil(fwhm_left_idx))
                        idx_low = max(0, min(idx_low, len(rts)-1))
                        idx_high = max(0, min(idx_high, len(rts)-1))
                        frac = fwhm_left_idx - idx_low
                        fwhm_left_rt = rts[idx_low] + frac * (rts[idx_high] - rts[idx_low])
                else:
                    fwhm_left_rt = rts[0]
        
                if fwhm_right_idx < len(rts):
                    if fwhm_right_idx == int(fwhm_right_idx):
                        fwhm_right_rt = rts[int(fwhm_right_idx)]
                    else:
                        # Interpolate RT
                        idx_low = int(np.floor(fwhm_right_idx))
                        idx_high = int(np.ceil(fwhm_right_idx))
                        idx_low = max(0, min(idx_low, len(rts)-1))
                        idx_high = max(0, min(idx_high, len(rts)-1))
                        frac = fwhm_right_idx - idx_low
                        fwhm_right_rt = rts[idx_low] + frac * (rts[idx_high] - rts[idx_low])
                else:
                    fwhm_right_rt = rts[-1]
        
                fwhm_sec = fwhm_right_rt - fwhm_left_rt
        
                # Fallback: if FWHM is too small or invalid, use a default based on scan spacing
                if fwhm_sec <= 0 or fwhm_sec < 5.0:
                    dt = np.median(np.diff(rts)) if len(rts) > 1 else 1.0
                    fwhm_sec = max(10.0, dt * 20)  # Default to ~20 scans or 10s, whichever is larger
                    if idx < 5:
                        print(f"[DEBUG]   Using default FWHM={fwhm_sec:.1f}s for {peptide[:30]} (computed FWHM was invalid)")
        
                # Integration and collection windows = detected peak boundaries (5% and 3% rel_height)
                min_rt = float(best_peak['min_rt'])
                max_rt = float(best_peak['max_rt'])
                collection_min_rt = float(best_peak.get('collection_min_rt', min_rt))
                collection_max_rt = float(best_peak.get('collection_max_rt', max_rt))
                # Optionally include anchor RT if just outside (e.g. table RT slightly outside log-noise boundary)
                if rt_anchor is not None and (rt_anchor < min_rt or rt_anchor > max_rt):
                    half_span = (max_rt - min_rt) / 2.0
                    if rt_anchor < min_rt and abs(rt_anchor - apex_rt) < half_span * 1.5:
                        min_rt = min(min_rt, rt_anchor)
                    elif rt_anchor > max_rt and abs(rt_anchor - apex_rt) < half_span * 1.5:
                        max_rt = max(max_rt, rt_anchor)
                min_rt = max(0.0, min_rt)
                max_rt = min(float(rts[-1]), max_rt)
                collection_min_rt = max(0.0, min(collection_min_rt, min_rt))  # collection extends left of integration
                collection_max_rt = min(float(rts[-1]), max(collection_max_rt, max_rt))  # collection extends right of integration
        
                # Snap to scan indices
                start_idx = np.argmin(np.abs(rts - min_rt))
                end_idx = np.argmin(np.abs(rts - max_rt))
                start_idx = max(0, start_idx)
                end_idx = min(len(rts) - 1, end_idx)
                min_rt = float(rts[start_idx])
                max_rt = float(rts[end_idx])
        
                # Store window size for median calculation
                window_sizes.append((min_rt, max_rt))
        
                # Debug output for first few peptides
                if idx < 3:
                    window_width = max_rt - min_rt
                    print(f"    Apex {apex_rt:.1f}s, window {window_width:.1f}s")
        
                # Keep top 2 peaks for comparison (Skyline-style)
                n_candidates = len(scored_candidates)
                best_score = best_peak['score']
                second_best_peak = scored_candidates[1] if n_candidates > 1 else None
                second_best_score = second_best_peak['score'] if second_best_peak else 0.0
                delta_score = best_score - second_best_score
        
                def _fmt_ratio_stability_plot(v):
                    """Format ratio stability for plot text: 'n/a' when not computed (None/NaN/0), else numeric."""
                    if v is None or (isinstance(v, float) and np.isnan(v)):
                        return 'n/a'
                    if isinstance(v, (int, float)) and v == 0:
                        return 'n/a'
                    try:
                        return f'{float(v):.2f}'
                    except (TypeError, ValueError):
                        return 'n/a'
        
                # Drift window for alternative_reasons (collection ± buffer); fallback to group CSV if available
                drift_min_rt = None
                drift_max_rt = None
                if collection_min_rt is not None and collection_max_rt is not None:
                    drift_min_rt = max(0.0, float(collection_min_rt) - PEAK_DRIFT_BUFFER_SEC)
                    drift_max_rt = float(collection_max_rt) + PEAK_DRIFT_BUFFER_SEC
                if (drift_min_rt is None or drift_max_rt is None) and 'drift_min_rt' in group.columns and 'drift_max_rt' in group.columns:
                    vmin = group['drift_min_rt'].dropna()
                    vmax = group['drift_max_rt'].dropna()
                    if len(vmin) > 0 and len(vmax) > 0:
                        try:
                            drift_min_rt = float(vmin.iloc[0])
                            drift_max_rt = float(vmax.iloc[0])
                        except (TypeError, ValueError):
                            pass
        
                # Analyze why peak 2 lost (if exists)
                peak2_loss_reasons = []
                if second_best_peak:
                    if best_peak['shape_corr'] > second_best_peak['shape_corr'] + 0.1:
                        peak2_loss_reasons.append(f"shape ({second_best_peak['shape_corr']:.2f} vs {best_peak['shape_corr']:.2f})")
                    if best_peak['rt_prior_score'] > second_best_peak['rt_prior_score'] + 0.2:
                        peak2_loss_reasons.append(f"RT prior ({second_best_peak['rt_offset']:.1f}s vs {best_peak['rt_offset']:.1f}s)")
                    if best_peak['ratio_stability'] > second_best_peak['ratio_stability'] + 0.1:
                        peak2_loss_reasons.append(f"ratio stability ({_fmt_ratio_stability_plot(second_best_peak['ratio_stability'])} vs {_fmt_ratio_stability_plot(best_peak['ratio_stability'])})")

                # For each alternative peak (not chosen), build a short "why not chosen" label for the plot
                alternative_reasons = []
                for c in scored_candidates:
                    if c is best_peak:
                        alternative_reasons.append(None)
                        continue
                    parts = [f"Score {c['score']:.3f} < best {best_peak['score']:.3f}"]
                    if drift_min_rt is not None and drift_max_rt is not None and (c['apex_rt'] < drift_min_rt or c['apex_rt'] > drift_max_rt):
                        parts.append("Outside drift RT window")
                    if best_peak['shape_corr'] > c['shape_corr'] + 0.1:
                        parts.append(f"shape ({c['shape_corr']:.2f} vs {best_peak['shape_corr']:.2f})")
                    if best_peak['rt_prior_score'] > c['rt_prior_score'] + 0.2:
                        parts.append(f"RT prior ({c['rt_offset']:.1f}s vs {best_peak['rt_offset']:.1f}s)")
                    if best_peak['ratio_stability'] > c['ratio_stability'] + 0.1:
                        parts.append(f"ratio ({_fmt_ratio_stability_plot(c['ratio_stability'])} vs {_fmt_ratio_stability_plot(best_peak['ratio_stability'])})")
                    alternative_reasons.append("; ".join(parts))
        
                # Check for interference/manual review flags (Skyline-style)
                interference_flag = False
                rt_deviation_flag = False
        
                # Flag if multiple candidates have similar scores (ambiguous)
                if n_candidates > 1 and delta_score < 0.1:  # Scores too close
                    interference_flag = True
        
                # Flag if shape evidence is poor
                if best_peak['shape_corr'] < 0.5:
                    interference_flag = True
        
                # Flag large RT deviation (but don't fail - Skyline treats as "needs review")
                rt_offset_sec = best_peak['rt_offset']
                if rt_offset_sec > 30.0:  # >30 seconds deviation
                    rt_deviation_flag = True
                strong_isotope = (best_peak['shape_corr'] > 0.85)
                if not strong_isotope and rt_offset_sec > 30.0:
                    interference_flag = True  # Large RT deviation + weak shape evidence = suspicious
        
                if idx < 3:
                    print(f"[DEBUG]   Peptide {peptide[:30]}: Found {n_candidates} candidate peak(s), best score: {best_score:.3f}, interference: {interference_flag}")
        
                # Baseline correct each isotope trace within peak window
                isotope_areas = {}
                isotope_traces_corrected = {}
                for iso_idx in range(intensity_matrix.shape[1]):
                    iso_intensities = intensity_matrix[:, iso_idx]
                    window_iso = iso_intensities[start_idx:end_idx+1]
                    if len(window_iso) > 0:
                        iso_baseline_corrected = baseline_correct_trace(window_iso, window_size=min(5, len(window_iso)))
                        isotope_traces_corrected[iso_idx] = iso_baseline_corrected
                        window_rts_iso = rts[start_idx:end_idx+1]
                        if len(window_rts_iso) >= 2:
                            area = float(_np_trapz(iso_baseline_corrected, window_rts_iso))
                        else:
                            area = 0.0
                        isotope_areas[iso_idx] = area
        
                # Total area = sum of isotope areas
                total_area = sum(isotope_areas.values())
                if total_area is not None and not (isinstance(total_area, float) and (np.isnan(total_area) or total_area < 0)):
                    try:
                        ta_val = float(total_area)
                        if ta_val < area_vmin_global:
                            area_vmin_global = ta_val
                        if ta_val > area_vmax_global:
                            area_vmax_global = ta_val
                    except (TypeError, ValueError):
                        pass
        
                # Extract QC metrics (for best peak - already computed above)
                coelution_var = best_peak['coelution_var']
                coelution_score = best_peak['coelution_score']
                shape_corr = best_peak['shape_corr']
                ratio_stability = best_peak['ratio_stability']
                rt_offset = best_peak['rt_offset']
                peak_quality_raw = best_peak['peak_quality']
                peak_quality_score = best_peak['peak_quality_score']
        
                # First pass: store scatter data for this peptide so every figure can show all N dots, then skip drawing/saving
                if pass_num == 0:
                    all_peptide_scatter_data[peptide_key] = {
                        'coelution_score': coelution_score,
                        'shape_corr': shape_corr,
                        'ratio_stability': ratio_stability,
                        'peak_score': best_score,
                        'total_area': total_area,
                    }
                    plt.close(fig)
                    continue
        
                # Convert RT to minutes for plotting
                rts_min = rts / 60.0
                min_rt_min = min_rt / 60.0
                max_rt_min = max_rt / 60.0
                collection_min_rt_min = collection_min_rt / 60.0
                collection_max_rt_min = collection_max_rt / 60.0
                apex_rt_min = apex_rt / 60.0 if apex_rt else None
                # Drift window (for positioning frag-count bubble over black drift band)
                drift_min_rt_min = (max(0.0, float(collection_min_rt) - PEAK_DRIFT_BUFFER_SEC) / 60.0) if collection_min_rt is not None else None
                drift_max_rt_min = ((float(collection_max_rt) + PEAK_DRIFT_BUFFER_SEC) / 60.0) if collection_max_rt is not None else None
        
                # Helper function to plot isotope traces on an axis (raw MS1 extraction — no gap-breaking so peaks always show)
                def plot_isotope_traces(axis, rts_min, intensity_matrix, total_intensities):
                    """Plot isotope traces and total trace. Uses raw rts/intensity so the chromatogram line is always drawn."""
                    rts_plot = np.asarray(rts_min, dtype=float)
                    im_plot = np.asarray(intensity_matrix, dtype=float)
                    tot_plot = np.asarray(total_intensities, dtype=float)
                    if im_plot.ndim == 1:
                        im_plot = np.atleast_2d(im_plot).T
                    colors = ['#0066FF', '#9933FF', '#FF00CC', '#FF66CC', '#FF9900']
                    labels_map = {0: 'M', 1: 'M+1', 2: 'M+2', 3: 'M+3', 4: 'M+4'}
                    for iso_idx in range(im_plot.shape[1]):
                        iso_intensities = im_plot[:, iso_idx]
                        color = colors[iso_idx % len(colors)]
                        label = labels_map.get(iso_idx, f'M+{iso_idx}')
                        axis.plot(rts_plot, iso_intensities, color=color, linewidth=3.0, alpha=1.0, label=label, linestyle='-', zorder=15+iso_idx)
                        axis.fill_between(rts_plot, iso_intensities, alpha=0.15, color=color, zorder=12+iso_idx)
                    axis.plot(rts_plot, tot_plot, color='#000000', linewidth=3.0, alpha=1.0, label='Total', linestyle='-', zorder=25)
        
                def add_ppm_labels_above_peaks(axis, rts_min, intensity_matrix, total_intensities, measured_mz_matrix, isotope_mzs, start_idx, end_idx):
                    """Label ppm (measured vs theoretical) above each total peak within the integration window.
                    Finds peaks in the summed trace; for each peak apex, uses the isotope with highest intensity there for ppm."""
                    if measured_mz_matrix is None or isotope_mzs is None or len(isotope_mzs) == 0:
                        return
                    rts_plot = np.asarray(rts_min, dtype=float)
                    im_plot = np.asarray(intensity_matrix, dtype=float)
                    tot_plot = np.asarray(total_intensities, dtype=float)
                    mz_plot = np.asarray(measured_mz_matrix, dtype=float)
                    if im_plot.ndim == 1:
                        im_plot = np.atleast_2d(im_plot).T
                    if mz_plot.ndim == 1:
                        mz_plot = np.atleast_2d(mz_plot).T
                    s0 = max(0, start_idx)
                    e0 = min(len(tot_plot), end_idx + 1)
                    if e0 <= s0:
                        return
                    window_tot = tot_plot[s0:e0]
                    if len(window_tot) == 0 or np.nanmax(window_tot) <= 0:
                        return
                    # Find peaks in total trace (multiple peaks possible, e.g. two separated by a few seconds)
                    min_prom = max(np.nanmax(window_tot) * 0.01, 1e-6)
                    try:
                        peaks_local, _ = signal.find_peaks(
                            window_tot, prominence=min_prom, width=2, distance=max(2, 3)
                        )
                    except Exception:
                        peaks_local = np.array([np.argmax(window_tot)])  # fallback: single max
                    if len(peaks_local) == 0:
                        peaks_local = np.array([np.argmax(window_tot)])
                    for apex_local in peaks_local:
                        apex_idx = s0 + int(apex_local)
                        if apex_idx >= mz_plot.shape[0]:
                            continue
                        # Use isotope with highest intensity at this apex for ppm
                        iso_ints = im_plot[apex_idx, :] if apex_idx < im_plot.shape[0] else np.zeros(im_plot.shape[1])
                        best_iso = int(np.argmax(iso_ints))
                        if best_iso >= len(isotope_mzs) or best_iso >= mz_plot.shape[1]:
                            continue
                        measured_mz = mz_plot[apex_idx, best_iso] if apex_idx < mz_plot.shape[0] else np.nan
                        if np.isnan(measured_mz) or measured_mz <= 0:
                            continue
                        theo_mz = float(isotope_mzs[best_iso])
                        if theo_mz <= 0:
                            continue
                        ppm = (measured_mz - theo_mz) / theo_mz * 1e6
                        rt_apex = rts_plot[apex_idx] if apex_idx < len(rts_plot) else rts_plot[-1]
                        int_apex = tot_plot[apex_idx] if apex_idx < len(tot_plot) else 0
                        axis.text(rt_apex, int_apex * 1.05, f'{ppm:+.1f} ppm', fontsize=9, ha='center', va='bottom',
                                 color='white', fontweight='bold', fontfamily='serif', zorder=100,
                                 bbox=dict(boxstyle='round,pad=0.35', facecolor='black', edgecolor='white',
                                          linewidth=1.2, alpha=0.95))
        
                # Plot on individual figure: raw extracted MS1 chromatogram (rts_min, intensity_matrix, total_intensities from extraction above)
                plot_isotope_traces(ax, rts_min, intensity_matrix, total_intensities)
                add_ppm_labels_above_peaks(ax, rts_min, intensity_matrix, total_intensities, measured_mz_matrix, isotope_mzs, start_idx, end_idx)
        
                # Combined-figure chromatogram is drawn only when not rejected (inside "if not is_rejected" below)
        
                # Create E-value colormap and normalization for colorbar (shared across all plots)
                from matplotlib.colors import LinearSegmentedColormap, LogNorm
                from matplotlib.cm import ScalarMappable
                evalue_min = 1e-10
                evalue_max = 100.0
                evalue_norm = LogNorm(vmin=evalue_min, vmax=evalue_max)
                evalue_colors = ['#00FF00', '#FFFF00', '#FF8800']  # Green, Yellow, Orange
                evalue_cmap = LinearSegmentedColormap.from_list('green_orange', evalue_colors, N=100)
        
                # Helper function to add peak windows and annotations to an axis
                def add_peak_annotations(axis, scored_candidates, best_peak, min_rt_min, max_rt_min, 
                                    apex_rt_min, rts, rts_min, total_intensities, min_rt, max_rt, 
                                    rt_anchor=None, anchor_evalue=None, evalue_cmap=None, evalue_norm=None,
                                    peptide_crossref=None, peptide_key=None, rt_to_frag_count=None,
                                    ms1_rts_table=None, ms1_rt_evalue_pairs=None,
                                    collection_min_rt_min=None, collection_max_rt_min=None,
                                    drift_min_rt_min=None, drift_max_rt_min=None,
                                    alternative_reasons=None, significant_frag_count=None,
                                    measured_mz_matrix_ann=None, isotope_mzs_ann=None, intensity_matrix_ann=None, total_intensities_ann=None):
                    """Add peak windows, apex line, and fill area. Draw ppm labels for alternative peaks (not chosen)."""
                    mz_mat = measured_mz_matrix_ann if measured_mz_matrix_ann is not None else measured_mz_matrix
                    iso_mzs = isotope_mzs_ann if isotope_mzs_ann is not None else isotope_mzs
                    imat = intensity_matrix_ann if intensity_matrix_ann is not None else intensity_matrix
                    tot_ints = total_intensities_ann if total_intensities_ann is not None else total_intensities
                    # Draw one dashed vertical line per spectrum (each row = one spectrum) for this unique peptide, color by E-value (low=green, high=orange/red)
                    # Source: ms1_rt_evalue_pairs (rejected) or built from group (main loop); always one line per scan
                    rt_evalue_list = ms1_rt_evalue_pairs
                    if rt_evalue_list is None and ms1_rts_table and len(ms1_rts_table) > 0:
                        # Build full list from every row (one entry per spectrum) so we draw one dashed line per spectrum
                        rt_evalue_list = []
                        for _, row in group.iterrows():
                            rt_sec = row.get('MS1_retention_time_sec')
                            if pd.isna(rt_sec) or rt_sec <= 0:
                                continue
                            evalue = row.get('e-value')
                            rt_evalue_list.append((float(rt_sec), evalue))
                    if rt_evalue_list:
                        drawn = 0
                        for rt_sec, evalue in rt_evalue_list:
                            rt_min_val = rt_sec / 60.0
                            if evalue is not None and pd.notna(evalue) and evalue > 0 and evalue_cmap is not None and evalue_norm is not None:
                                try:
                                    color = evalue_cmap(evalue_norm(evalue))
                                except Exception:
                                    color = '#0066CC'
                            else:
                                color = '#0066CC'
                            label = 'Scan RT (E-value)' if drawn == 0 and (evalue_cmap is not None and evalue_norm is not None) else ('Scan RT' if drawn == 0 else '')
                            axis.axvline(x=rt_min_val, color=color, linestyle='--', linewidth=2.5,
                                         alpha=0.95, zorder=32, label=label)
                            drawn += 1
                    elif ms1_rts_table and len(ms1_rts_table) > 0:
                        # Fallback: one dashed line per RT in table (e.g. when group not in closure)
                        for i, rt_sec in enumerate([rt for rt in ms1_rts_table if rt and rt > 0]):
                            rt_min_val = rt_sec / 60.0
                            label = 'Scan RT (this peptide)' if i == 0 else ''
                            axis.axvline(x=rt_min_val, color='#0066CC', linestyle='--', linewidth=2.0,
                                         alpha=0.95, zorder=32, label=label)
                    
                    # Shade background: collection window (one opaque gray band) then integration window (green)
                    # Collection window always includes the entire integration window and more (one band from coll_left to coll_right)
                    coll_left = collection_min_rt_min
                    coll_right = collection_max_rt_min
                    if coll_left is None:
                        coll_left = min_rt_min
                    if coll_right is None:
                        coll_right = max_rt_min
                    # Ensure collection always contains integration (and extends beyond)
                    if min_rt_min is not None and coll_left is not None and coll_left > min_rt_min:
                        coll_left = min_rt_min
                    if max_rt_min is not None and coll_right is not None and coll_right < max_rt_min:
                        coll_right = max_rt_min
                    if coll_left is not None and coll_right is not None:
                        # If collection equals integration (no tails), extend slightly so grey band is visible
                        window_min = (max_rt_min - min_rt_min) * 0.02 if (min_rt_min is not None and max_rt_min is not None) else 0.008
                        if coll_left >= min_rt_min:
                            coll_left = min_rt_min - max(window_min, 0.008)
                        if coll_right <= max_rt_min:
                            coll_right = max_rt_min + max(window_min, 0.008)
                        # Clamp to chromatogram range so grey stays within plot
                        if len(rts_min) > 0:
                            coll_left = max(float(rts_min[0]), coll_left)
                            coll_right = min(float(rts_min[-1]), coll_right)
                        # One opaque gray band: full collection window (includes integration + tails)
                        if coll_right > coll_left:
                            axis.axvspan(coll_left, coll_right, alpha=COLLECTION_WINDOW_ALPHA, color=COLLECTION_WINDOW_COLOR, zorder=0,
                                        label='Collection window')
                    
                    # Shade non-best candidates with hatched fill (so only one solid gray = collection)
                    for peak_idx, candidate in enumerate(scored_candidates):
                        if candidate != best_peak:  # Skip best peak for now
                            candidate_min_rt_min = candidate['min_rt'] / 60.0
                            candidate_max_rt_min = candidate['max_rt'] / 60.0
                            axis.axvspan(candidate_min_rt_min, candidate_max_rt_min,
                                      alpha=0.4, facecolor='#E8E8E8', hatch='///', zorder=0,
                                      label='Alternative Peak' if peak_idx == 1 and len(scored_candidates) > 1 else '')
                            # Simple ppm label for alternative peak (replaces "Not chosen" text)
                            apex_idx_cand = candidate.get('apex_idx')
                            if (apex_idx_cand is not None and mz_mat is not None and iso_mzs is not None
                                    and len(iso_mzs) > 0 and imat is not None and tot_ints is not None
                                    and apex_idx_cand < mz_mat.shape[0] and apex_idx_cand < imat.shape[0]):
                                im_cand = imat[apex_idx_cand, :]
                                best_iso = int(np.argmax(im_cand)) if len(im_cand) > 0 else 0
                                if best_iso < len(iso_mzs) and best_iso < mz_mat.shape[1]:
                                    meas_mz = mz_mat[apex_idx_cand, best_iso]
                                    theo_mz = float(iso_mzs[best_iso])
                                    if not (np.isnan(meas_mz) or meas_mz <= 0 or theo_mz <= 0):
                                        ppm_val = (meas_mz - theo_mz) / theo_mz * 1e6
                                        apex_min = candidate['apex_rt'] / 60.0
                                        int_apex = tot_ints[apex_idx_cand] if (tot_ints is not None and apex_idx_cand < len(tot_ints)) else 0
                                        axis.text(apex_min, int_apex * 1.05, f'{ppm_val:+.1f} ppm', fontsize=9, ha='center', va='bottom',
                                                 color='white', fontweight='bold', fontfamily='serif', zorder=100,
                                                 bbox=dict(boxstyle='round,pad=0.35', facecolor='black', edgecolor='white',
                                                          linewidth=1.2, alpha=0.95), clip_on=True)
                    
                    # Integration window (green): apex-centered, used for quantification
                    axis.axvspan(min_rt_min, max_rt_min, alpha=INTEGRATION_WINDOW_ALPHA, color='#00FF7F', zorder=1, 
                              label='[OK] Selected Peak' if len(scored_candidates) > 1 else 'Integration Window')
                    
                    # Add a prominent border to the selected peak window (bright, vibrant green)
                    # Border lines should be above the shaded area but below the data lines
                    axis.axvline(x=min_rt_min, color='#00CC66', linestyle='-', linewidth=4.0, alpha=1.0, zorder=5)
                    axis.axvline(x=max_rt_min, color='#00CC66', linestyle='-', linewidth=4.0, alpha=1.0, zorder=5)
                    
                    # Plot anchor RT (MS1 RT from table) - dashed line colored by E-value (good = green)
                    # Plot peak apex RT (detected peak) - green dashed line (see below)
                    
                    # Plot anchor RT (MS1 RT from table) - dashed, color by E-value (low = green, high = orange/red)
                    anchor_rt_min = None
                    if rt_anchor:
                        anchor_rt_min = rt_anchor / 60.0
                    
                        # Collect other peptides' RT values within the chromatogram window (for gray lines only)
                        overlapping_rt_values_chrom = []
                        if peptide_crossref is not None and peptide_key is not None:
                            for other_key, other_data in peptide_crossref.items():
                                if other_key == peptide_key:
                                    continue
                                other_rt_anchor = other_data.get('rt_anchor', None)
                                if other_rt_anchor is not None:
                                    if min_rt <= other_rt_anchor <= max_rt:
                                        if other_rt_anchor not in overlapping_rt_values_chrom:
                                            overlapping_rt_values_chrom.append(other_rt_anchor)
                                    elif (min_rt - 30.0) <= other_rt_anchor <= (max_rt + 30.0):
                                        if other_rt_anchor not in overlapping_rt_values_chrom:
                                            overlapping_rt_values_chrom.append(other_rt_anchor)
                                other_rt_min, other_rt_max = other_data['rt_range']
                                if not (other_rt_max < min_rt or other_rt_min > max_rt) and other_rt_anchor is not None:
                                    if (min_rt - 30.0) <= other_rt_anchor <= (max_rt + 30.0) and other_rt_anchor not in overlapping_rt_values_chrom:
                                        overlapping_rt_values_chrom.append(other_rt_anchor)
                    
                        # Anchor RT: color by E-value (good = green, high = orange/red), same colormap as scan RTs
                        if anchor_evalue is not None and pd.notna(anchor_evalue) and anchor_evalue > 0 and evalue_cmap is not None and evalue_norm is not None:
                            try:
                                line_color = evalue_cmap(evalue_norm(anchor_evalue))
                            except Exception:
                                line_color = '#666666'
                        else:
                            line_color = '#666666'
                    
                        axis.axvline(x=anchor_rt_min, color=line_color, linestyle='--', linewidth=2.5, 
                                  alpha=1.0, zorder=30, label='OpenMS_MS1_Retention_Time')
                    
                        # Add fragment count label: yellow bubble = significant fragments only (5 ppm + >0.5% intensity)
                        frag_count = significant_frag_count
                        if frag_count is not None and frag_count > 0:
                            y_lim = axis.get_ylim()
                            x_lim = axis.get_xlim()
                            coll_left = collection_min_rt_min if collection_min_rt_min is not None else min_rt_min
                            coll_right = collection_max_rt_min if collection_max_rt_min is not None else max_rt_min
                            if coll_left is None:
                                coll_left = min_rt_min
                            if coll_right is None:
                                coll_right = max_rt_min
                            # Position x at center of left black drift buffer, or right buffer, or anchor
                            if drift_min_rt_min is not None and drift_max_rt_min is not None and coll_left is not None and coll_right is not None:
                                left_buf_end = min(coll_left, drift_max_rt_min)
                                right_buf_start = max(coll_right, drift_min_rt_min)
                                if left_buf_end > drift_min_rt_min and coll_left > drift_min_rt_min:
                                    label_x_anchor = (drift_min_rt_min + left_buf_end) * 0.5
                                elif drift_max_rt_min > right_buf_start and coll_right < drift_max_rt_min:
                                    label_x_anchor = (right_buf_start + drift_max_rt_min) * 0.5
                                else:
                                    label_x_anchor = anchor_rt_min
                            else:
                                label_x_anchor = anchor_rt_min
                            # Vertical position: middle of plot so bubble sits over the black drift band
                            label_y_anchor = y_lim[0] + (y_lim[1] - y_lim[0]) * 0.5
                            fontsize_anchor = 10 if hasattr(axis, '_is_combined') else 12
                            axis.text(label_x_anchor, label_y_anchor, f'{frag_count} frags', 
                                    fontsize=fontsize_anchor, ha='center', va='center', fontfamily='serif',
                                    bbox=dict(boxstyle='round,pad=0.25', facecolor='yellow', alpha=0.9, edgecolor='black', linewidth=0.8),
                                    zorder=32, clip_on=False)
                    
                        # Draw off-target (other peptides') RTs as triangle symbols along the x-axis
                        if overlapping_rt_values_chrom:
                            overlapping_rt_values_chrom = sorted(list(set(overlapping_rt_values_chrom)))
                            x_lim = axis.get_xlim()
                            y_lim = axis.get_ylim()
                            x_min_sec = x_lim[0] * 60.0
                            x_max_sec = x_lim[1] * 60.0
                            x_min_extended = x_min_sec - 10.0
                            x_max_extended = x_max_sec + 10.0
                            other_rt_mins = [other_rt / 60.0 for other_rt in overlapping_rt_values_chrom
                                            if x_min_extended <= other_rt <= x_max_extended]
                            if other_rt_mins:
                                y_bottom = y_lim[0]
                                axis.scatter(other_rt_mins, [y_bottom] * len(other_rt_mins),
                                            marker='^', s=144, color='#F0A030', edgecolors='#C07820', linewidths=1.5,
                                            alpha=0.9, zorder=28, label='Other peptide RT (below)', clip_on=True)
                    
                    # Plot peak apex RT (detected peak) - green dashed line with thin black outline
                    if apex_rt_min:
                        axis.axvline(x=apex_rt_min, color='black', linestyle='--', linewidth=4.5, 
                                  alpha=1.0, zorder=30, label='Matched_Peak_Apex')
                        axis.axvline(x=apex_rt_min, color='#00FF7F', linestyle='--', linewidth=3.0, 
                                  alpha=1.0, zorder=31, label='Matched_Peak_Apex')
                        if anchor_rt_min:
                            apex_rt_sec = apex_rt_min * 60.0
                            rt_offset_sec = abs(apex_rt_sec - rt_anchor)
                            rt_offset_min = rt_offset_sec / 60.0
                            label_x = max_rt_min + (max_rt_min - min_rt_min) * 0.05
                            y_lim = axis.get_ylim()
                            x_lim = axis.get_xlim()
                            if hasattr(axis, '_is_combined'):
                                label_x = max_rt_min + (max_rt_min - min_rt_min) * 0.02
                                label_y_data = y_lim[0] + (y_lim[1] - y_lim[0]) * 0.95
                                ha_align = 'left'
                            else:
                                label_x = x_lim[0] + (x_lim[1] - x_lim[0]) * 0.02
                                label_y_data = y_lim[0] + (y_lim[1] - y_lim[0]) * 0.98
                                ha_align = 'left'
                            if x_lim[0] <= label_x <= x_lim[1] and y_lim[0] <= label_y_data <= y_lim[1]:
                                fontsize = 10 if hasattr(axis, '_is_combined') else 13
                                axis.text(label_x, label_y_data, f'ΔRT: {rt_offset_sec:.1f}s', 
                                        fontsize=fontsize, ha=ha_align, va='top', fontfamily='serif',
                                        bbox=dict(boxstyle='round,pad=0.25', facecolor='yellow', alpha=0.7),
                                        zorder=21, clip_on=True)
                                if hasattr(axis, '_is_combined') and anchor_evalue is not None and pd.notna(anchor_evalue) and anchor_evalue > 0:
                                    evalue_label_y = label_y_data - (y_lim[1] - y_lim[0]) * 0.08
                                    if anchor_evalue < 0.001:
                                        evalue_text = f'E-value: {anchor_evalue:.2e}'
                                    else:
                                        evalue_text = f'E-value: {anchor_evalue:.4f}'
                                    axis.text(label_x, evalue_label_y, evalue_text, 
                                            fontsize=9, ha=ha_align, va='top', fontfamily='serif',
                                            bbox=dict(boxstyle='round,pad=0.25', facecolor='orange', alpha=0.7),
                                            zorder=21, clip_on=True)
                    
                    # Fill area under Total trace within boundaries (best peak only)
                    window_mask_best = (rts >= min_rt) & (rts <= max_rt)
                    window_rts_min_best = rts_min[window_mask_best]
                    if len(window_rts_min_best) > 1:
                        window_total_best = total_intensities[window_mask_best]
                        axis.fill_between(window_rts_min_best, window_total_best, alpha=0.2, color='gray', zorder=8)
        
                # Number of fragments that passed 5 ppm + >0.5% intensity (for yellow bubble on chromatogram)
                significant_frag_count = None
                if raw_file and min_rt is not None and max_rt is not None:
                    ms2_mzs_pre, ms2_ints_pre = get_averaged_ms2_spectrum_in_window(raw_file, min_rt, max_rt, ppm_tolerance=20.0)
                    if ms2_mzs_pre is not None and len(ms2_mzs_pre) > 0 and ms2_ints_pre is not None and len(ms2_ints_pre) > 0:
                        ion_mz_labels_pre = get_comet_c_z_z1_ion_mz_from_csv(group)
                        if ion_mz_labels_pre is None:
                            comet_c_z_z1_pre = parse_comet_c_z_z1_matched_ions(group)
                            ion_mz_labels_pre = comet_matched_ions_to_theoretical_mz(comet_c_z_z1_pre, peptide, fragment_charge=1)
                        mz_max_pre = max(ms2_mzs_pre) * 1.02 if len(ms2_mzs_pre) > 0 else 500
                        ion_mz_labels_pre = [(mz, lbl) for mz, lbl in (ion_mz_labels_pre or []) if 0 < mz <= mz_max_pre]
                        y_max_pre = np.max(ms2_ints_pre) * 1.05 if len(ms2_ints_pre) > 0 else 1.0
                        label_placements_pre = _ms2_label_positions_no_overlap(
                            ion_mz_labels_pre, ms2_mzs_pre, ms2_ints_pre, y_max_pre, ppm_ms2=SIGNIFICANT_FRAGMENT_PPM_MS2, label_offset_frac=0.06,
                            mz_near=40.0, min_y_sep_frac=0.045, min_intensity_frac=(0.0 if skip_significance_intensity_filter else SIGNIFICANT_FRAGMENT_MIN_INTENSITY_FRAC))
                        significant_frag_count = len(label_placements_pre)
                # Add annotations to individual figure (pass all table RTs so both clusters visible)
                n_scan_rts = len([rt for rt in ms1_rts_table if rt and rt > 0])
                if idx < 15 or n_scan_rts <= 3:
                    x_min_vals = [round(rt/60, 2) for rt in ms1_rts_table[:5] if rt and rt > 0]
                    print(f"[DEBUG]   This unique peptide: drawing {n_scan_rts} spectrum RT line(s) at x (min) = {x_min_vals}{'...' if n_scan_rts > 5 else ''}")
                add_peak_annotations(ax, scored_candidates, best_peak, min_rt_min, max_rt_min, 
                               apex_rt_min, rts, rts_min, total_intensities, min_rt, max_rt,
                               rt_anchor=rt_anchor, anchor_evalue=anchor_evalue, 
                               evalue_cmap=evalue_cmap, evalue_norm=evalue_norm,
                               peptide_crossref=peptide_crossref, peptide_key=peptide_key,
                               rt_to_frag_count=rt_to_frag_count, ms1_rts_table=ms1_rts_table,
                               collection_min_rt_min=collection_min_rt_min, collection_max_rt_min=collection_max_rt_min,
                               drift_min_rt_min=drift_min_rt_min, drift_max_rt_min=drift_max_rt_min,
                               alternative_reasons=alternative_reasons, significant_frag_count=significant_frag_count)
        
                # Helper function to count isotopic peak matches in a spectrum
                def count_isotopic_matches(spec_mzs_arr, spec_ints_arr, isotope_mzs_list, ppm_tol=5.0):
                    """Count how many expected isotopic peaks are found in the spectrum."""
                    matches = 0
                    for iso_mz in isotope_mzs_list:
                        iso_tolerance = iso_mz * ppm_tol * 1e-6
                        iso_mask = (spec_mzs_arr >= iso_mz - iso_tolerance) & (spec_mzs_arr <= iso_mz + iso_tolerance)
                        if np.any(iso_mask):
                            iso_intensities = spec_ints_arr[iso_mask]
                            max_intensity = np.max(spec_ints_arr) if len(spec_ints_arr) > 0 else 0
                            if max_intensity > 0 and np.max(iso_intensities) > max_intensity * 0.01:
                                matches += 1
                    return matches
        
                def get_matched_isotope_indices(spec_mzs_arr, spec_ints_arr, isotope_mzs_list, ppm_tol=5.0):
                    """Return list of isotope indices (0=M, 1=M+1, 2=M+2, 3=M+3) that are matched."""
                    matched_indices = []
                    for iso_idx, iso_mz in enumerate(isotope_mzs_list):
                        iso_tolerance = iso_mz * ppm_tol * 1e-6
                        iso_mask = (spec_mzs_arr >= iso_mz - iso_tolerance) & (spec_mzs_arr <= iso_mz + iso_tolerance)
                        if np.any(iso_mask):
                            iso_intensities = spec_ints_arr[iso_mask]
                            max_intensity = np.max(spec_ints_arr) if len(spec_ints_arr) > 0 else 0
                            if max_intensity > 0 and np.max(iso_intensities) > max_intensity * 0.01:
                                matched_indices.append(iso_idx)
                    return matched_indices

                def get_isotope_peak_metrics(spec_mzs_arr, spec_ints_arr, isotope_mzs_list, ppm_tol=5.0):
                    """
                    For each theoretical isotope m/z, find the highest-intensity observed peak within +/-ppm_tol.
                    Returns dict: iso_idx -> {'observed_mz','intensity','ppm_error'}.
                    """
                    metrics = {}
                    if spec_mzs_arr is None or spec_ints_arr is None or isotope_mzs_list is None:
                        return metrics
                    if len(spec_mzs_arr) == 0 or len(spec_ints_arr) == 0:
                        return metrics
                    for iso_idx, iso_mz in enumerate(isotope_mzs_list):
                        iso_tolerance = iso_mz * ppm_tol * 1e-6
                        iso_mask = (spec_mzs_arr >= iso_mz - iso_tolerance) & (spec_mzs_arr <= iso_mz + iso_tolerance)
                        if not np.any(iso_mask):
                            continue
                        iso_mzs_near = spec_mzs_arr[iso_mask]
                        iso_ints_near = spec_ints_arr[iso_mask]
                        if len(iso_ints_near) == 0:
                            continue
                        idx_max = int(np.argmax(iso_ints_near))
                        obs_mz = float(iso_mzs_near[idx_max])
                        obs_int = float(iso_ints_near[idx_max])
                        ppm_err = (obs_mz - iso_mz) / iso_mz * 1e6 if iso_mz > 0 else np.nan
                        metrics[iso_idx] = {
                            'observed_mz': obs_mz,
                            'intensity': obs_int,
                            'ppm_error': ppm_err,
                        }
                    return metrics
        
                def get_observed_centroid_near_theoretical(spec_mzs_arr, spec_ints_arr, theoretical_mz, ppm_tolerance=20.0):
                    """Find m/z at max intensity within ±ppm_tolerance of theoretical_mz. Returns (observed_mz, ppm_diff) or (None, None)."""
                    if spec_mzs_arr is None or len(spec_mzs_arr) == 0 or theoretical_mz <= 0:
                        return None, None
                    tol_da = theoretical_mz * ppm_tolerance * 1e-6
                    mask = (spec_mzs_arr >= theoretical_mz - tol_da) & (spec_mzs_arr <= theoretical_mz + tol_da)
                    if not np.any(mask):
                        return None, None
                    mzs_near = spec_mzs_arr[mask]
                    ints_near = spec_ints_arr[mask]
                    idx_max = np.argmax(ints_near)
                    observed_mz = float(mzs_near[idx_max])
                    ppm_diff = (observed_mz - theoretical_mz) / theoretical_mz * 1e6
                    return observed_mz, ppm_diff
        
                # Note: We no longer filter out peptides with overlapping isotopic peaks here
                # Instead, we'll visualize them with red lines in the spectrum and let the user decide
                # The overlap detection will happen later when plotting the spectrum
        
                # Initialize spectrum variables (needed for rejected peptide collection)
                spec_mzs_windowed = None
                spec_ints_windowed = None
                spec_mzs_windowed_integration = None
                spec_ints_windowed_integration = None
                spectrum_rt_min = None
                spectrum_rt_max = None
                using_anchor_window = False
                mz_min_window = None
                mz_max_window = None
                n_matches_final = 0
                matched_iso_indices_final = []
        
                # Extract and plot MS1 spectrum averaged across peak window (for individual figure only)
                # Use averaged spectrum across the entire peak window for better isotope pattern representation
                if apex_rt and min_rt and max_rt:
                    # Calculate m/z window around isotopic peaks (M, M+1, M+2, M+3) for efficiency
                    if isotope_mzs and len(isotope_mzs) > 0:
                        isotope_mz_min = min(isotope_mzs)
                        isotope_mz_max = max(isotope_mzs)
                        mz_buffer = 0.5
                        mz_min_window = isotope_mz_min - mz_buffer
                        mz_max_window = isotope_mz_max + mz_buffer
                    else:
                        mz_buffer = 2.0
                        mz_min_window = precursor_mz - mz_buffer
                        mz_max_window = precursor_mz + mz_buffer
                    spec_mzs, spec_ints = get_averaged_ms1_spectrum_in_window(
                        raw_file, min_rt, max_rt, 
                        mz_min=mz_min_window, mz_max=mz_max_window, 
                        ppm_tolerance=7.0
                    )
                    # Initialize best spectrum variables
                    best_spec_mzs_windowed = None
                    best_spec_ints_windowed = None
                    best_spectrum_rt_min = min_rt
                    best_spectrum_rt_max = max_rt
                    best_n_matches = 0
                    best_matched_iso_indices = []
                    best_window_source = "detected_peak"
                    best_summed_intensity = 0.0  # tie-breaker: prefer higher intensity when match count is equal
                    
                    if spec_mzs is not None and len(spec_mzs) > 0:
                        # Filter to the m/z window of interest
                        mask = (spec_mzs >= mz_min_window) & (spec_mzs <= mz_max_window)
                        spec_mzs_windowed = spec_mzs[mask]
                        spec_ints_windowed = spec_ints[mask]
                        # Keep a strict integration-window copy for envelope logic written to CSV.
                        spec_mzs_windowed_integration = spec_mzs_windowed.copy()
                        spec_ints_windowed_integration = spec_ints_windowed.copy()
                        if len(spec_mzs_windowed) > 0 and isotope_mzs and len(isotope_mzs) > 0:
                            n_matches = count_isotopic_matches(spec_mzs_windowed, spec_ints_windowed, isotope_mzs, ppm_tol=5.0)
                            matched_iso_indices = get_matched_isotope_indices(spec_mzs_windowed, spec_ints_windowed, isotope_mzs, ppm_tol=5.0)
                            best_spec_mzs_windowed = spec_mzs_windowed
                            best_spec_ints_windowed = spec_ints_windowed
                            best_spectrum_rt_min = min_rt
                            best_spectrum_rt_max = max_rt
                            best_n_matches = n_matches
                            best_matched_iso_indices = matched_iso_indices.copy()
                            best_window_source = "detected_peak"
                            best_summed_intensity = float(np.sum(spec_ints_windowed))
                            rt_window_sec = 30.0
                            unique_table_rts = sorted(set(ms1_rts_table))
                            detected_peak_center = (min_rt + max_rt) / 2.0
                            detected_peak_width = max_rt - min_rt
                            for table_rt in unique_table_rts:
                                if table_rt <= 0:
                                    continue
                        
                                # Skip if this RT is already covered by the detected peak window (within 10% margin)
                                margin = detected_peak_width * 0.1
                                if (min_rt - margin) <= table_rt <= (max_rt + margin):
                                    continue
                        
                                # Extract spectrum around this table RT
                                table_min_rt = max(0, table_rt - rt_window_sec)
                                table_max_rt = table_rt + rt_window_sec
                        
                                spec_mzs_table, spec_ints_table = get_averaged_ms1_spectrum_in_window(
                                    raw_file, table_min_rt, table_max_rt,
                                    mz_min=mz_min_window, mz_max=mz_max_window,
                                    ppm_tolerance=7.0
                                )
                        
                                if spec_mzs_table is not None and len(spec_mzs_table) > 0:
                                    mask_table = (spec_mzs_table >= mz_min_window) & (spec_mzs_table <= mz_max_window)
                                    spec_mzs_table_windowed = spec_mzs_table[mask_table]
                                    spec_ints_table_windowed = spec_ints_table[mask_table]
                            
                                    if len(spec_mzs_table_windowed) > 0:
                                        # Count matches in this table RT window
                                        n_matches_table = count_isotopic_matches(
                                            spec_mzs_table_windowed, spec_ints_table_windowed, isotope_mzs, ppm_tol=5.0
                                        )
                                        matched_iso_indices_table = get_matched_isotope_indices(
                                            spec_mzs_table_windowed, spec_ints_table_windowed, isotope_mzs, ppm_tol=5.0
                                        )
                                
                                        # Distance from detected peak center and anchor RT (for logging; no hard cap)
                                        table_window_center = (table_min_rt + table_max_rt) / 2.0
                                        dist_from_detected_peak = abs(table_window_center - detected_peak_center)
                                        dist_from_anchor = abs(table_rt - rt_anchor) if rt_anchor else float('inf')
                                
                                        # Compare peaks: the superior one wins. Don't default to the weaker (detected) peak.
                                        # Superior = more isotopic matches, or same matches but higher summed intensity.
                                        # Envelope, ppm, and matched-fragment checks still apply when accepting the peptide.
                                        summed_intensity_table = float(np.sum(spec_ints_table_windowed))
                                        should_use = False
                                        reason = ""
                                        if n_matches_table > best_n_matches:
                                            should_use = True
                                            reason = f"{n_matches_table} matches (vs {best_n_matches})"
                                        elif n_matches_table == best_n_matches and summed_intensity_table > best_summed_intensity:
                                            should_use = True
                                            reason = f"same {n_matches_table} matches, higher intensity ({summed_intensity_table:.0f} vs {best_summed_intensity:.0f})"
                                        elif best_n_matches < 2 and n_matches_table >= 2:
                                            should_use = True
                                            reason = f"{n_matches_table} matches (detected peak had only {best_n_matches})"
                                
                                        if should_use:
                                            best_spec_mzs_windowed = spec_mzs_table_windowed
                                            best_spec_ints_windowed = spec_ints_table_windowed
                                            best_spectrum_rt_min = table_min_rt
                                            best_spectrum_rt_max = table_max_rt
                                            best_n_matches = n_matches_table
                                            best_matched_iso_indices = matched_iso_indices_table.copy()
                                            best_window_source = f"table_rt_{table_rt:.1f}s"
                                            best_summed_intensity = summed_intensity_table
                                            if idx < 5:
                                                print(f"[DEBUG]   Using table RT window ({table_min_rt:.1f}-{table_max_rt:.1f}s, RT={table_rt:.1f}s) for spectrum - {reason}, dist from detected peak: {dist_from_detected_peak:.1f}s, from anchor: {dist_from_anchor:.1f}s")
                    
                            # Use the best spectrum found
                            spec_mzs_windowed = best_spec_mzs_windowed
                            spec_ints_windowed = best_spec_ints_windowed
                            spectrum_rt_min = best_spectrum_rt_min
                            spectrum_rt_max = best_spectrum_rt_max
                            n_matches_final = best_n_matches
                            matched_iso_indices_final = best_matched_iso_indices.copy()
                            using_anchor_window = (best_window_source != "detected_peak")
                    
                            # Validate that selected window makes sense
                            selected_window_center = (spectrum_rt_min + spectrum_rt_max) / 2.0
                            detected_peak_center = (min_rt + max_rt) / 2.0
                            window_distance = abs(selected_window_center - detected_peak_center)
                    
                            # Warn if selected window is far from detected peak (unless detected peak had very few matches)
                            if best_window_source != "detected_peak" and window_distance > 60.0:
                                if idx < 10:
                                    print(f"[DEBUG]   WARNING [{peptide[:30]}]: Selected window ({spectrum_rt_min:.1f}-{spectrum_rt_max:.1f}s) is {window_distance:.1f}s away from detected peak ({min_rt:.1f}-{max_rt:.1f}s)")
                    
                            if idx < 5:
                                if best_window_source == "detected_peak":
                                    print(f"[DEBUG]   Using detected peak window ({spectrum_rt_min:.1f}-{spectrum_rt_max:.1f}s) for spectrum - {n_matches_final} isotopic peaks matched")
                                else:
                                    print(f"[DEBUG]   Selected {best_window_source} window ({spectrum_rt_min:.1f}-{spectrum_rt_max:.1f}s) for spectrum - {n_matches_final} isotopic peaks matched (detected peak: {min_rt:.1f}-{max_rt:.1f}s)")
                        else:
                            # No spectrum data or isotope m/z values available
                            n_matches_final = 0
                            matched_iso_indices_final = []
                
                    # Quick test: theoretical M0 vs actual centroid m/z at peak (within ±20 ppm)
                    # If mismatch, possible wrong precursor m/z, charge, or mod state for XIC extraction.
                    if spec_mzs_windowed is not None and len(spec_mzs_windowed) > 0 and isotope_mzs and len(isotope_mzs) > 0:
                        theoretical_m0 = isotope_mzs[0]
                        observed_mz, ppm_diff = get_observed_centroid_near_theoretical(
                            spec_mzs_windowed, spec_ints_windowed, theoretical_m0, ppm_tolerance=20.0
                        )
                        if observed_mz is not None and ppm_diff is not None:
                            if idx < 5:
                                print(f"[DEBUG]   XIC m/z check (peak apex spectrum): theoretical M0={theoretical_m0:.6f}, observed centroid={observed_mz:.6f}, ppm={ppm_diff:+.1f}")
                            if abs(ppm_diff) > 20:
                                print(f"[DEBUG]   WARNING [{peptide[:25]}...]: theoretical vs observed M0 differ by {abs(ppm_diff):.1f} ppm (>20 ppm) — possible wrong precursor m/z, charge, or mod state for XIC")
                
                    # Extract snapshot spectrum at anchor RT for filtering check
                    # This allows us to accept peptides if EITHER the summed spectrum OR snapshot spectrum passes
                    matched_iso_indices_snapshot = []
                    n_matches_snapshot = 0
                    if rt_anchor and rt_anchor > 0 and isotope_mzs and len(isotope_mzs) > 0:
                        spec_mzs_snapshot_check, spec_ints_snapshot_check = get_ms1_spectrum_at_rt(raw_file, rt_anchor, rt_tolerance_sec=2.0)
                        if spec_mzs_snapshot_check is not None and len(spec_mzs_snapshot_check) > 0:
                            # Check isotopic matches in snapshot spectrum
                            for iso_idx, iso_mz in enumerate(isotope_mzs):
                                iso_tolerance = iso_mz * 5e-6  # 5 ppm tolerance
                                iso_mask = (spec_mzs_snapshot_check >= iso_mz - iso_tolerance) & (spec_mzs_snapshot_check <= iso_mz + iso_tolerance)
                                if np.any(iso_mask):
                                    iso_intensities = spec_ints_snapshot_check[iso_mask]
                                    max_iso_intensity = np.max(iso_intensities) if len(iso_intensities) > 0 else 0
                                    # Check if signal is significant (>1% of max spectrum intensity)
                                    max_spec_intensity = np.max(spec_ints_snapshot_check) if len(spec_ints_snapshot_check) > 0 else 0
                                    if max_spec_intensity > 0 and max_iso_intensity > max_spec_intensity * 0.01:
                                        matched_iso_indices_snapshot.append(iso_idx)
                            n_matches_snapshot = len(matched_iso_indices_snapshot)
                
                    # Track if this peptide should be rejected
                    is_rejected = False
                    rejection_reason = None
                    # (Apex < 10^5 is handled earlier: we only assign from acceptable_candidates with apex >= 10^5)
                    def _has_modifications(m):
                        """True if peptide has any modification (not empty, -, or nan)."""
                        if m is None or (isinstance(m, float) and pd.isna(m)):
                            return False
                        s = str(m).strip().lower()
                        return s not in ('', '-', 'nan', 'none')
                
                    # Required isotope indices: M0, M+1, M+2, M+3 (summed MS1 must contain all four)
                    REQUIRED_ISOTOPE_INDICES = (0, 1, 2, 3)  # M0, M+1, M+2, M+3
    
                    # Helper function to check if isotopic matches pass filtering criteria
                    def check_isotopic_matches_passed(matched_indices, spectrum_name):
                        """Check if matched isotopic indices include M0, M+1, M+2, and M+3.
                        Returns (passed, reason) tuple."""
                        if len(matched_indices) == 0:
                            return False, f"No isotopic peaks matched in {spectrum_name}"
                        sorted_indices = sorted(matched_indices)
                        matched_iso_names = [f'M+{i}' for i in sorted_indices]
                        # Require M0, M+1, M+2, and M+3
                        missing = [i for i in REQUIRED_ISOTOPE_INDICES if i not in matched_indices]
                        if missing:
                            missing_names = [f'M+{i}' for i in sorted(missing)]
                            return False, f"Summed MS1 must contain M0, M+1, M+2, and M+3 (missing: {', '.join(missing_names)}; matched: {', '.join(matched_iso_names)})"
                        return True, None
    
                    # Check summed spectrum in the INTEGRATION WINDOW (detected_peak_min_rt..max_rt) only.
                    # Core envelope logic (for CSV columns):
                    #   1) M0, M+1, M+2, M+3 present (highest-intensity peak within +/-5 ppm per isotope)
                    #   2) M0 and M+1 are both > M+2 and M+3
                    core_peak_metrics = get_isotope_peak_metrics(
                        spec_mzs_windowed_integration, spec_ints_windowed_integration, isotope_mzs, ppm_tol=5.0
                    )
                    core_matched_indices = sorted(core_peak_metrics.keys())
                    summed_passed, summed_reason = check_isotopic_matches_passed(core_matched_indices, "summed spectrum (integration window)")
                    has_required_in_window = all(i in core_matched_indices for i in REQUIRED_ISOTOPE_INDICES)
                    I0 = float(core_peak_metrics.get(0, {}).get('intensity', 0.0))
                    I1 = float(core_peak_metrics.get(1, {}).get('intensity', 0.0))
                    I2 = float(core_peak_metrics.get(2, {}).get('intensity', 0.0))
                    I3 = float(core_peak_metrics.get(3, {}).get('intensity', 0.0))
                    m0_gt_m1 = has_required_in_window and I0 > 0 and I1 > 0 and I0 > I1
                    m1_gt_m2 = has_required_in_window and I1 > 0 and I2 > 0 and I1 > I2
                    m0_gt_m1_gt_m2 = bool(m0_gt_m1 or m1_gt_m2)
                    m2_le_m0 = bool(has_required_in_window and I2 <= I0)
                    m0_m1_gt_m2_m3 = bool(has_required_in_window and (I0 > I2) and (I0 > I3) and (I1 > I2) and (I1 > I3))
                    envelope_ok = bool(m0_m1_gt_m2_m3)

                    # When re-extracting with a primary/alternate filter CSV, trust envelope criterion from the filter
                    # (those peptides already passed the envelope criterion in the initial run; re-checking in a different
                    # extraction window can falsely reject due to window shift/narrowing.)
                    filter_envelope_trusted = False
                    if filter_csv and len(group) > 0:
                        if 'm0_m1_gt_m2_m3' in group.columns:
                            v = group['m0_m1_gt_m2_m3'].iloc[0]
                        elif 'm0_gt_m1_gt_m2' in group.columns:
                            v = group['m0_gt_m1_gt_m2'].iloc[0]
                        else:
                            v = None
                        if v is not None and not (isinstance(v, float) and pd.isna(v)):
                            if isinstance(v, bool) and v:
                                filter_envelope_trusted = True
                            elif isinstance(v, (int, float)) and v:
                                filter_envelope_trusted = True
                            elif str(v).strip().lower() in ('true', '1', 'yes'):
                                filter_envelope_trusted = True

                    if filter_envelope_trusted:
                        # Require M0, M+1, M+2 present even when trusting filter envelope
                        envelope_ok = has_required_in_window
                        if not envelope_ok:
                            rejection_reason = "Summed MS1 must contain M0, M+1, M+2, and M+3"
                            is_rejected = True
                        else:
                            is_rejected = False if not (exclude_mods and _has_modifications(mods)) else True
                            rejection_reason = None if not is_rejected else "Excluded: peptide has modifications (--exclude-mods)"
                    elif no_psm_filter:
                        # Require M0, M+1, M+2 present even in no-filter run
                        envelope_ok = has_required_in_window
                        if not envelope_ok:
                            rejection_reason = "Summed MS1 must contain M0, M+1, M+2, and M+3"
                            is_rejected = True
                        else:
                            is_rejected = False if not (exclude_mods and _has_modifications(mods)) else True
                            rejection_reason = None if not is_rejected else "Excluded: peptide has modifications (--exclude-mods)"
                    else:
                        # Do not reject for envelope here; Step 7 handles filtering.
                        # Keep extraction acceptance independent from envelope ordering.
                        # Do NOT reject for envelope here; Step 7 (filter_envelope.py) handles that
                        is_rejected = False if not (exclude_mods and _has_modifications(mods)) else True
                        rejection_reason = None if not is_rejected else "Excluded: peptide has modifications (--exclude-mods)"

                    # Reject if peak m/z >6 ppm from theoretical
                    PPM_REJECT_THRESHOLD = 6.0
                    if ppm_at_apex is not None and ppm_at_apex > PPM_REJECT_THRESHOLD:
                        is_rejected = True
                        rejection_reason = f"Peak m/z >6 ppm from theoretical ({ppm_at_apex:.2f} ppm)"

                    # Helper to get numeric score from a row (used for PSM filter and for best-PSM display below)
                    def _score_from_row(row, col, fallback_keys):
                        if col and col in row.index:
                            v = row[col]
                            if pd.notna(v) and v != '':
                                try:
                                    return float(v)
                                except (TypeError, ValueError):
                                    pass
                        for k in fallback_keys:
                            if k in row.index:
                                v = row[k]
                                if pd.notna(v) and v != '':
                                    try:
                                        return float(v)
                                    except (TypeError, ValueError):
                                        pass
                        return None
                    def _psm_passes_all(row):
                        """Match Step 4 (filter_comet_frags_confidence): pass if Q≤0.05 OR PEP≤0.05 (OR logic)."""
                        q = _score_from_row(row, qvalue_col, ['percolator_qvalue', 'q-value', 'qvalue', 'Q-value'])
                        p = _score_from_row(row, pep_col, ['percolator_PEP', 'pep', 'PEP'])
                        e = _score_from_row(row, evalue_col, ['e-value', 'e_value', 'E-value'])
                        if e is not None and np.isfinite(e) and e > FILTER_PSM_EVALUE_MAX:
                            return False
                        q_ok = (q is None or not np.isfinite(q) or q <= FILTER_PSM_QVALUE_MAX)
                        p_ok = (p is None or not np.isfinite(p) or p <= FILTER_PSM_PEP_MAX)
                        return q_ok or p_ok  # Same as Step 4: Q≤0.05 OR PEP≤0.05
                    # Apply PSM score filters (Q-value, PEP, E-value): accept peptide if ANY PSM passes all metrics (skip when no_psm_filter)
                    if not is_rejected and not no_psm_filter:
                        any_psm_passes = any(_psm_passes_all(row) for _, row in group.iterrows())
                        if not any_psm_passes:
                            is_rejected = True
                            rejection_reason = "PSM filters: no PSM passed (Q≤0.05 OR PEP≤0.05)"
                            print(f"[DEBUG]   Rejecting peptide {peptide[:30]}: {rejection_reason}", file=sys.stderr)
                    # Shape correlation threshold: require best peak to have shape_corr >= SHAPE_CORR_MIN
                    if not is_rejected:
                        sc = best_peak.get('shape_corr')
                        if sc is not None and not (isinstance(sc, float) and np.isnan(sc)) and float(sc) < SHAPE_CORR_MIN:
                            is_rejected = True
                            rejection_reason = f"Shape correlation {float(sc):.3f} < {SHAPE_CORR_MIN}"
                            print(f"[DEBUG]   Rejecting peptide {peptide[:30]}: {rejection_reason}", file=sys.stderr)
                    # Significant single-AA overhang is enforced per-peptide from MS2 (placed_ion_labels); no CSV fallback
                    # Best PSM for display (lowest q-value if available, else lowest E-value); prefer one that passes if any
                    best_psm_row = None
                    if qvalue_col and qvalue_col in group.columns:
                        qmask = group[qvalue_col].apply(lambda v: pd.notna(v) and v != '' and (isinstance(v, (int, float)) or (isinstance(v, str) and v.strip())))
                        try:
                            qnums = group.loc[qmask, qvalue_col].astype(float)
                            if len(qnums) > 0:
                                best_psm_idx = qnums.idxmin()
                                best_psm_row = group.loc[best_psm_idx]
                        except (TypeError, ValueError):
                            pass
                    if best_psm_row is None and evalue_col and evalue_col in group.columns:
                        emask = group[evalue_col].apply(lambda v: pd.notna(v) and v != '' and (isinstance(v, (int, float)) or (isinstance(v, str) and v.strip())))
                        try:
                            enums = group.loc[emask, evalue_col].astype(float)
                            if len(enums) > 0:
                                best_psm_idx = enums.idxmin()
                                best_psm_row = group.loc[best_psm_idx]
                        except (TypeError, ValueError):
                            pass
                    if best_psm_row is None:
                        best_psm_row = group.iloc[0]
                    psm_q = _score_from_row(best_psm_row, qvalue_col, ['percolator_qvalue', 'q-value', 'qvalue', 'Q-value'])
                    psm_pep = _score_from_row(best_psm_row, pep_col, ['percolator_PEP', 'pep', 'PEP'])
                    psm_e = _score_from_row(best_psm_row, evalue_col, ['e-value', 'e_value', 'E-value'])

                    # Assign combined-figure slot only for non-rejected peptides (no gaps; proper axes/labels for every panel)
                    if not is_rejected:
                        file_idx = picked_idx // plots_per_file
                        local_idx = picked_idx % plots_per_file
                        axes_combined_chrom_list, axes_combined_spec_list = combined_axes_list[file_idx]
                        ax_combined_chrom = axes_combined_chrom_list[local_idx]
                        ax_combined_spec = axes_combined_spec_list[local_idx]
                        drew_combined_spec = False
                        # Plot isotope traces on combined chromatogram (first thing drawn on this slot)
                        plot_isotope_traces(ax_combined_chrom, rts_min, intensity_matrix, total_intensities)
                        add_ppm_labels_above_peaks(ax_combined_chrom, rts_min, intensity_matrix, total_intensities, measured_mz_matrix, isotope_mzs, start_idx, end_idx)
                
                    # If rejected, skip plotting to accepted combined figure but continue to plot individual figure and spectrum
                    # We'll collect rejected peptides and plot them separately at the end
                
                    # Only proceed with plotting spectrum if we have spectrum data
                    # Even if rejected, we still want to plot the spectrum for visualization in the individual figure
                    if spec_mzs_windowed is not None and len(spec_mzs_windowed) > 0:
                            # Plot summed spectrum as stem plot (Freestyle-style: sum intensities)
                            # Use matplotlib's stem function - black bars with finer lines
                            markerline, stemlines, baseline = ax_spec.stem(spec_mzs_windowed, spec_ints_windowed, 
                                                                           basefmt=' ', linefmt='#888888', markerfmt=' ')
                            plt.setp(stemlines, linewidth=0.6)
                            plt.setp(markerline, markersize=2)
                    
                            # Cross-reference: Check if other peptides have overlapping isotopic peaks or RT values WITHIN THE WINDOW
                            # Use full_crossref (all non-decoy peptides from CSV) for comprehensive interference detection
                            overlapping_peptides_iso = []  # List of peptide keys with overlapping isotopic peaks
                            overlapping_peptides_rt = []  # List of peptide keys with overlapping RT values
                            overlapping_iso_mzs = set()  # Set of current peptide's iso_mz values that overlap with other peptides
                            other_peptides_iso_mzs = {}  # Dict mapping OTHER peptides' isotopic m/z values to peptide sequence (for gray dashed lines with labels)
                            overlapping_rt_values = []  # List of RT values from other peptides within window
                    
                            for other_key, other_data in full_crossref.items():
                                if other_key == peptide_key:
                                    continue
                        
                                # Get peptide sequence and charge from group data for labeling
                                other_group = df[df['peptide_key'] == other_key] if 'peptide_key' in df.columns else None
                                other_peptide_seq = None
                                other_charge = None
                                if other_group is not None and len(other_group) > 0:
                                    other_peptide_seq = other_group['plain_peptide'].iloc[0] if 'plain_peptide' in other_group.columns else None
                                    other_charge = other_group['charge'].iloc[0] if 'charge' in other_group.columns else None
                                if other_peptide_seq is None:
                                    # Fallback: try to extract from key (less reliable)
                                    parts = other_key.split('_')
                                    if len(parts) >= 4:
                                        # Assume peptide is the second part (after mz), charge is second-to-last
                                        other_peptide_seq = parts[1] if len(parts) > 1 else other_key[:20]
                                        try:
                                            other_charge = int(parts[-2]) if len(parts) >= 3 else None
                                        except:
                                            other_charge = None
                        
                                # Check isotopic peak overlap (within 5 ppm tolerance) AND within m/z window
                                for iso_mz in isotope_mzs:
                                    # Only check if this iso_mz is within the spectrum window
                                    if mz_min_window <= iso_mz <= mz_max_window:
                                        iso_tolerance = iso_mz * 5e-6  # 5 ppm tolerance
                                        for other_iso_mz in other_data['isotope_mzs']:
                                            # Check if other peptide's isotopic peak is within window AND overlaps
                                            if mz_min_window <= other_iso_mz <= mz_max_window:
                                                if abs(iso_mz - other_iso_mz) <= iso_tolerance:
                                                    overlapping_iso_mzs.add(iso_mz)
                                                    if other_key not in overlapping_peptides_iso:
                                                        overlapping_peptides_iso.append(other_key)
                                                    break
                        
                                # RT overlap: track for chromatogram
                                other_rt_anchor = other_data.get('rt_anchor', None)
                                other_rt_min, other_rt_max = other_data['rt_range']
                                if other_rt_anchor is not None and min_rt <= other_rt_anchor <= max_rt:
                                    overlapping_rt_values.append(other_rt_anchor)
                                if not (other_rt_max < min_rt or other_rt_min > max_rt):
                                    if other_key not in overlapping_peptides_rt:
                                        overlapping_peptides_rt.append(other_key)
                        
                                # Collect other peptides' isotopic m/z in window (gray dashed = interfering positions)
                                # Include all others in m/z window so we show every potential interference; co-eluting ones are the true interferences
                                # Store mapping of m/z to (peptide sequence, charge) tuple for labeling
                                for other_iso_mz in other_data['isotope_mzs']:
                                    if mz_min_window <= other_iso_mz <= mz_max_window:
                                        if other_iso_mz not in other_peptides_iso_mzs:
                                            other_peptides_iso_mzs[other_iso_mz] = (
                                                other_peptide_seq if other_peptide_seq else other_key[:20],
                                                other_charge if other_charge else other_data.get('charge', None)
                                            )
                    
                            # Note: Overlap filtering already happened earlier (before spectrum extraction)
                            # This overlap detection is now only for visualization purposes (though overlaps should be filtered out already)
                            # Highlight precursor and isotope m/z values - thinner, more saturated, darker colors
                            iso_colors = ['#0033CC', '#6600CC', '#CC0099', '#CC0066']  # More saturated and darker
                    
                            # Collect maximum intensities of matched isotopic peaks for y-axis scaling
                            matched_iso_intensities = []
                            matched_iso_indices = []  # Track which isotopes have matches for legend filtering
                    
                            # First pass: collect matched intensities to compute y_max
                            for iso_idx, iso_mz in enumerate(isotope_mzs):
                                if iso_idx < len(iso_colors):
                                    iso_tolerance = iso_mz * 5e-6  # 5 ppm tolerance for isotope matching
                                    iso_mask = (spec_mzs_windowed >= iso_mz - iso_tolerance) & (spec_mzs_windowed <= iso_mz + iso_tolerance)
                                    if np.any(iso_mask):
                                        iso_intensities = spec_ints_windowed[iso_mask]
                                        max_iso_intensity = np.max(iso_intensities) if len(iso_intensities) > 0 else np.max(spec_ints_windowed) if len(spec_ints_windowed) > 0 else 0
                                        matched_iso_intensities.append(max_iso_intensity)
                                        matched_iso_indices.append(iso_idx)
                    
                            # Compute y-axis max for drawing off-target lines
                            if len(matched_iso_intensities) > 0:
                                max_matched_intensity = np.max(matched_iso_intensities)
                                y_max_spec = max_matched_intensity * 1.2
                            else:
                                y_max_spec = np.max(spec_ints_windowed) * 1.1 if len(spec_ints_windowed) > 0 else 1.0
                    
                            # Draw gray dashed lines only where there is signal within 5 ppm of the expected (other) isotopic m/z
                            max_spec_intensity = np.max(spec_ints_windowed) if len(spec_ints_windowed) > 0 else 0
                            for other_iso_mz, (other_peptide_seq, other_charge_val) in other_peptides_iso_mzs.items():
                                other_tol = other_iso_mz * 5e-6  # 5 ppm
                                other_mask = (spec_mzs_windowed >= other_iso_mz - other_tol) & (spec_mzs_windowed <= other_iso_mz + other_tol)
                                if np.any(other_mask):
                                    other_ints = spec_ints_windowed[other_mask]
                                    if max_spec_intensity > 0 and np.max(other_ints) > max_spec_intensity * 0.01:
                                        ax_spec.vlines(other_iso_mz, 0, y_max_spec, colors='#808080',
                                                     linestyles='--', linewidths=2.5, alpha=0.85, zorder=1)
                                        # Add label with peptide sequence and charge (truncate sequence if too long)
                                        seq_part = other_peptide_seq[:12] + '...' if len(other_peptide_seq) > 12 else other_peptide_seq
                                        charge_part = f" (+{other_charge_val})" if other_charge_val else ""
                                        label_text = seq_part + charge_part
                                        ax_spec.text(other_iso_mz, y_max_spec * 0.95, label_text,
                                                   fontsize=9, ha='center', va='bottom', fontfamily='serif',
                                                   bbox=dict(boxstyle='round,pad=0.25', facecolor='lightgray', alpha=0.7, edgecolor='gray', linewidth=0.6),
                                                   zorder=3)
                    
                            # Now draw colored lines for current peptide's isotopes (on top of gray lines)
                            for iso_idx, iso_mz in enumerate(isotope_mzs):
                                if iso_idx < len(iso_colors):
                                    # Find peaks near this isotope m/z in the summed spectrum
                                    iso_tolerance = iso_mz * 5e-6  # 5 ppm tolerance for isotope matching
                                    iso_mask = (spec_mzs_windowed >= iso_mz - iso_tolerance) & (spec_mzs_windowed <= iso_mz + iso_tolerance)
                            
                                    if np.any(iso_mask):
                                        iso_intensities = spec_ints_windowed[iso_mask]
                                        iso_mzs_found = spec_mzs_windowed[iso_mask]
                                        # Plot vertical line at isotope m/z, extending to max intensity in spectrum
                                        max_iso_intensity = np.max(iso_intensities) if len(iso_intensities) > 0 else np.max(spec_ints_windowed) if len(spec_ints_windowed) > 0 else 0
                                        # Always use normal isotope colors; interference is indicated by black dashed lines
                                        line_color = iso_colors[iso_idx]
                                        line_width = 3.5  # Thicker to reflect ±5 ppm tolerance window
                                        alpha_val = 1.0
                                        ax_spec.vlines(iso_mz, 0, max_iso_intensity, colors=line_color, 
                                                     linestyles='--', linewidths=line_width, alpha=alpha_val, zorder=2,
                                                     label=f'M+{iso_idx}')
                                    # Only draw lines where there is actual signal matching within tolerance
                    
                            # Set y-axis limits
                            y_min_spec = 0
                            ax_spec.set_ylim([y_min_spec, y_max_spec])
                    
                            # Add annotation if there are overlapping peptides
                            if overlapping_peptides_iso or overlapping_peptides_rt or overlapping_iso_mzs:
                                overlap_text = []
                                if overlapping_peptides_iso or overlapping_iso_mzs:
                                    n_overlapping_iso = len(overlapping_iso_mzs) if overlapping_iso_mzs else 0
                                    overlap_text.append(f"{len(overlapping_peptides_iso)} peptide(s) with overlapping isotopic peaks ({n_overlapping_iso} m/z overlaps)")
                                if overlapping_peptides_rt or overlapping_rt_values:
                                    n_overlapping_rt = len(overlapping_rt_values) if overlapping_rt_values else 0
                                    overlap_text.append(f"{len(overlapping_peptides_rt)} peptide(s) with overlapping RT ({n_overlapping_rt} RT overlaps)")
                                x_lim_spec = ax_spec.get_xlim()
                                y_lim_spec = ax_spec.get_ylim()
                                overlap_label_x = x_lim_spec[0] + (x_lim_spec[1] - x_lim_spec[0]) * 0.02
                                overlap_label_y = y_lim_spec[0] + (y_lim_spec[1] - y_lim_spec[0]) * 0.85
                                ax_spec.text(overlap_label_x, overlap_label_y, '[!] ' + '; '.join(overlap_text), 
                                           fontsize=7, ha='left', va='top', fontfamily='serif',
                                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8),
                                           zorder=25, clip_on=True)
                        
                                # Debug output for first few peptides
                                if idx < 3:
                                    print(f"[DEBUG]   Overlaps detected for {peptide[:30]}: {len(overlapping_iso_mzs)} iso m/z overlaps, {len(overlapping_rt_values)} RT overlaps")
                    
                            # Format spectrum plot (black background: light text)
                            ax_spec.set_xlabel('m/z', fontsize=15, fontfamily='serif', color='0.85')
                            ax_spec.set_ylabel('Intensity', fontsize=15, fontfamily='serif', color='0.85')
                            # Update title to reflect summed spectrum across window (Freestyle-style)
                            # Indicate if using anchor RT window
                            title_rt_range = f'{spectrum_rt_min:.1f}-{spectrum_rt_max:.1f}s'
                            if using_anchor_window:
                                title_rt_range += ' (anchor RT window)'
                            ax_spec.set_title(f'Summed MS1 Spectrum (RT={title_rt_range})', fontsize=15, fontweight='bold', fontfamily='serif', color='0.9')
                            ax_spec.tick_params(labelsize=14, colors='0.85')
                            for label in ax_spec.get_xticklabels() + ax_spec.get_yticklabels():
                                label.set_fontfamily('serif')
                                label.set_color('0.85')
                            for spine in ax_spec.spines.values():
                                spine.set_color('0.6')
                            ax_spec.grid(False)  # Remove grid bars
                            # Position legend above the spectrum plot (not overlapping)
                            # Filter legend to only include isotopes that have matching signals
                            # Clear any existing legend first
                            if ax_spec.get_legend():
                                ax_spec.get_legend().remove()
                            # Get all handles and labels from the axes
                            handles, labels = ax_spec.get_legend_handles_labels()
                            # Filter to only include labels that correspond to matched isotopes (only isotopes with signal)
                            filtered_handles = []
                            filtered_labels = []
                            for handle, label in zip(handles, labels):
                                # Only include if it's an isotope label (starts with 'M+') and corresponds to a matched isotope
                                if label.startswith('M+'):
                                    try:
                                        # Extract number after M+ (handles both 'M+0' and 'M+0 (OVERLAP)')
                                        iso_idx_str = label.split('+')[1].split()[0]  # Gets '0' from 'M+0' or 'M+0 (OVERLAP)'
                                        iso_idx = int(iso_idx_str)
                                        # Only include if this isotope has signal (is in matched_iso_indices)
                                        if iso_idx in matched_iso_indices:
                                            filtered_handles.append(handle)
                                            filtered_labels.append(label)
                                    except (ValueError, IndexError):
                                        # If parsing fails, skip it (shouldn't happen with proper labels)
                                        pass
                            # Only create legend if there are filtered entries (isotopes with signal)
                            if filtered_handles and filtered_labels:
                                leg = ax_spec.legend(filtered_handles, filtered_labels, loc='upper right', 
                                              fontsize=11, prop={'family': 'serif'}, facecolor='#1a1a1a', edgecolor='0.6', labelcolor='0.9')
                            ax_spec.set_xlim([mz_min_window, mz_max_window])
                    
                            # Add label showing isotope matching tolerance
                            x_lim_spec = ax_spec.get_xlim()
                            y_lim_spec = ax_spec.get_ylim()
                            tolerance_label_x = x_lim_spec[0] + (x_lim_spec[1] - x_lim_spec[0]) * 0.02  # Left side
                            tolerance_label_y = y_lim_spec[0] + (y_lim_spec[1] - y_lim_spec[0]) * 0.95  # Near top
                            ax_spec.text(tolerance_label_x, tolerance_label_y, 'Isotope matching: ±5 ppm', 
                                       fontsize=12, ha='left', va='top', fontfamily='serif',
                                       bbox=dict(boxstyle='round,pad=0.35', facecolor='lightblue', alpha=0.8, edgecolor='black', linewidth=0.8),
                                       zorder=25, clip_on=True)
                    
                    else:
                        # Try fallback: single spectrum at apex RT if averaging failed
                        if apex_rt:
                            spec_mzs, spec_ints = get_ms1_spectrum_at_rt(raw_file, apex_rt, rt_tolerance_sec=2.0)
                            if spec_mzs is not None and len(spec_mzs) > 0:
                                mask = (spec_mzs >= mz_min_window) & (spec_mzs <= mz_max_window)
                                spec_mzs_windowed = spec_mzs[mask]
                                spec_ints_windowed = spec_ints[mask]
                                if len(spec_mzs_windowed) > 0:
                                    markerline, stemlines, baseline = ax_spec.stem(spec_mzs_windowed, spec_ints_windowed, 
                                                                                   basefmt=' ', linefmt='#888888', markerfmt=' ')
                                    plt.setp(stemlines, linewidth=1.0)
                                    plt.setp(markerline, markersize=2.5)
                                    # Highlight isotopes - thinner, more saturated, darker colors
                                    iso_colors = ['#0033CC', '#6600CC', '#CC0099', '#CC0066']  # More saturated and darker
                                    # Collect maximum intensities of matched isotopic peaks for y-axis scaling
                                    matched_iso_intensities_fallback = []
                                    matched_iso_indices_fallback = []  # Track which isotopes have matches for legend filtering
                                    for iso_idx, iso_mz in enumerate(isotope_mzs):
                                        if iso_idx < len(iso_colors):
                                            iso_tolerance = iso_mz * 5e-6  # 5 ppm tolerance
                                            iso_mask = (spec_mzs_windowed >= iso_mz - iso_tolerance) & (spec_mzs_windowed <= iso_mz + iso_tolerance)
                                            if np.any(iso_mask):
                                                iso_intensities = spec_ints_windowed[iso_mask]
                                                max_iso_intensity = np.max(iso_intensities) if len(iso_intensities) > 0 else 0
                                                matched_iso_intensities_fallback.append(max_iso_intensity)
                                                matched_iso_indices_fallback.append(iso_idx)  # Track that this isotope has a match
                                                ax_spec.vlines(iso_mz, 0, max_iso_intensity, colors=iso_colors[iso_idx], 
                                                             linestyles='--', linewidths=3.0, alpha=1.0,
                                                             label=f'M+{iso_idx}')  # Add label for legend filtering
                            
                                    # Scale y-axis based on maximum matched isotopic peak intensity
                                    if len(matched_iso_intensities_fallback) > 0:
                                        max_matched_intensity_fallback = np.max(matched_iso_intensities_fallback)
                                        y_max_spec_fallback = max_matched_intensity_fallback * 1.2
                                        y_min_spec_fallback = 0
                                        ax_spec.set_ylim([y_min_spec_fallback, y_max_spec_fallback])
                                    else:
                                        # Fallback: if no matched peaks, use spectrum maximum
                                        if len(spec_ints_windowed) > 0:
                                            max_spec_intensity_fallback = np.max(spec_ints_windowed)
                                            y_max_spec_fallback = max_spec_intensity_fallback * 1.1
                                            y_min_spec_fallback = 0
                                            ax_spec.set_ylim([y_min_spec_fallback, y_max_spec_fallback])
                            
                                    ax_spec.set_xlabel('m/z', fontsize=15, fontfamily='serif', color='0.85')
                                    ax_spec.set_ylabel('Intensity', fontsize=15, fontfamily='serif', color='0.85')
                                    ax_spec.set_title(f'MS1 Spectrum at Peak Apex (RT={apex_rt:.1f}s)', fontsize=15, fontweight='bold', fontfamily='serif', color='0.9')
                                    ax_spec.tick_params(labelsize=14)
                                    for label in ax_spec.get_xticklabels() + ax_spec.get_yticklabels():
                                        label.set_fontfamily('serif')
                                    ax_spec.grid(False)  # Remove grid bars
                                    ax_spec.set_xlim([mz_min_window, mz_max_window])
                            
                                    # Filter legend to only include isotopes that have matching signals (fallback case)
                                    # Clear any existing legend first
                                    if ax_spec.get_legend():
                                        ax_spec.get_legend().remove()
                                    handles_fallback, labels_fallback = ax_spec.get_legend_handles_labels()
                                    filtered_handles_fallback = []
                                    filtered_labels_fallback = []
                                    for handle, label in zip(handles_fallback, labels_fallback):
                                        if label.startswith('M+'):
                                            try:
                                                # Extract number after M+ (handles both 'M+0' and 'M+0 (OVERLAP)')
                                                iso_idx_str = label.split('+')[1].split()[0]
                                                iso_idx = int(iso_idx_str)
                                                # Only include if this isotope has signal (is in matched_iso_indices_fallback)
                                                if iso_idx in matched_iso_indices_fallback:
                                                    filtered_handles_fallback.append(handle)
                                                    filtered_labels_fallback.append(label)
                                            except (ValueError, IndexError):
                                                # If parsing fails, skip it
                                                pass
                                    # Only create legend if there are filtered entries (isotopes with signal)
                                    if filtered_handles_fallback and filtered_labels_fallback:
                                        ax_spec.legend(filtered_handles_fallback, filtered_labels_fallback, 
                                                      loc='upper right', 
                                                      fontsize=13, prop={'family': 'serif'})
                            
                                    # Add label showing isotope matching tolerance (fallback case)
                                    x_lim_spec_fallback = ax_spec.get_xlim()
                                    y_lim_spec_fallback = ax_spec.get_ylim()
                                    tolerance_label_x_fallback = x_lim_spec_fallback[0] + (x_lim_spec_fallback[1] - x_lim_spec_fallback[0]) * 0.02
                                    tolerance_label_y_fallback = y_lim_spec_fallback[0] + (y_lim_spec_fallback[1] - y_lim_spec_fallback[0]) * 0.95
                                    ax_spec.text(tolerance_label_x_fallback, tolerance_label_y_fallback, 'Isotope matching: ±5 ppm', 
                                               fontsize=12, ha='left', va='top', fontfamily='serif',
                                               bbox=dict(boxstyle='round,pad=0.35', facecolor='lightblue', alpha=0.8, edgecolor='black', linewidth=0.8),
                                               zorder=25, clip_on=True)
                                else:
                                    ax_spec.axis('off')
                            else:
                                ax_spec.axis('off')
                        else:
                            ax_spec.axis('off')
                else:
                    # No apex RT or peak window - hide the subplot
                    ax_spec.axis('off')
        
                # Ions that get a yellow box on MS2 (5 ppm + 0.5% intensity); sequence panel and bubbles use only these.
                placed_ion_labels = None
                overhang_positions_from_placed = None  # When set, red squares and red bubble show only overhangs from placed ions
                # MS2 spectrum summed across peak integration window (same window as summed MS1)
                if raw_file and min_rt is not None and max_rt is not None:
                    ms2_mzs, ms2_ints = get_averaged_ms2_spectrum_in_window(raw_file, min_rt, max_rt, ppm_tolerance=20.0)
                    if ms2_mzs is not None and len(ms2_mzs) > 0 and ms2_ints is not None and len(ms2_ints) > 0:
                        ax_spec_ms2.stem(ms2_mzs, ms2_ints, basefmt=' ', linefmt='#888888', markerfmt=' ')
                        ax_spec_ms2.set_xlabel('m/z', fontsize=16, fontfamily='serif', color='0.85')
                        ax_spec_ms2.set_ylabel('Intensity', fontsize=16, fontfamily='serif', color='0.85')
                        min_rt_min = min_rt / 60.0
                        max_rt_min = max_rt / 60.0
                        ax_spec_ms2.set_title(f'Summed MS2 (RT={min_rt_min:.2f}-{max_rt_min:.2f} min)', fontsize=14, fontweight='bold', fontfamily='serif', color='0.9')
                        ax_spec_ms2.tick_params(labelsize=13, colors='0.85')
                        for label in ax_spec_ms2.get_xticklabels() + ax_spec_ms2.get_yticklabels():
                            label.set_fontfamily('serif')
                            label.set_color('0.85')
                        for spine in ax_spec_ms2.spines.values():
                            spine.set_color('0.6')
                        y_max_ms2 = np.max(ms2_ints) * 1.05 if len(ms2_ints) > 0 else 1.0
                        ax_spec_ms2.set_ylim([0, y_max_ms2])
                        x_max_ms2 = max(ms2_mzs) * 1.02 if len(ms2_mzs) > 0 else 500
                        ax_spec_ms2.set_xlim([0, x_max_ms2])
                        ax_spec_ms2.grid(True, alpha=0.25)
                        # Use ALL theoretical c, z, z+1 ions so every significant peak (5 ppm + 0.5%) gets a label on the MS2 plot
                        clean_peptide_for_mz = re.sub(r'\[.*?\]', '', peptide) if peptide else ''
                        if clean_peptide_for_mz:
                            c_list, z_list, z1_list = theoretical_c_z_z1_ion_mz(clean_peptide_for_mz, fragment_charge=1)
                            ion_mz_labels = (c_list or []) + (z_list or []) + (z1_list or [])
                        else:
                            ion_mz_labels = get_comet_c_z_z1_ion_mz_from_csv(group)
                            if ion_mz_labels is None:
                                comet_c_z_z1 = parse_comet_c_z_z1_matched_ions(group)
                                ion_mz_labels = comet_matched_ions_to_theoretical_mz(comet_c_z_z1, peptide, fragment_charge=1) or []
                        mz_max_plot = max(ms2_mzs) * 1.02 if len(ms2_mzs) > 0 else 500
                        ion_mz_labels = [(mz, lbl) for mz, lbl in (ion_mz_labels or []) if 0 < mz <= mz_max_plot]
                        # Per-peptide: red = significant single-AA overhangs from significant pairs from significant fragments (no carryover from previous peptide)
                        overhang_positions_from_placed = None
                        significant_pairs_for_display = []
                        # Non-overlapping label positions (Stage 2 significant: 5 ppm + % max intensity)
                        label_placements = _ms2_label_positions_no_overlap(
                            ion_mz_labels, ms2_mzs, ms2_ints, y_max_ms2, ppm_ms2=SIGNIFICANT_FRAGMENT_PPM_MS2, label_offset_frac=0.14,
                            mz_near=80.0, min_y_sep_frac=0.12, min_intensity_frac=(0.0 if skip_significance_intensity_filter else SIGNIFICANT_FRAGMENT_MIN_INTENSITY_FRAC), horizontal_offset_mz=18.0)
                        placed_ion_labels = {ion_label for (_x, _y, ion_label, _xy) in label_placements}
                        clean_peptide_ms2 = re.sub(r'\[.*?\]', '', peptide) if peptide else ''
                        row0 = group.iloc[0]
                        pairs_col = 'single_aa_overhang_fragment_pairs'
                        # Chain: SIGNIFICANT fragments (placed_ion_labels = 5 ppm + >0.5% intensity) -> significant pairs -> significant single-AA overhangs (red)
                        overhang_positions_from_placed = set()  # always set when we have MS2 so filename uses significant count (0 if no labels)
                        significant_pairs_for_display = []
                        ions_in_pair = set()  # MS2 bubbles: yellow only if part of significant pair; else light gray (c) / dark gray (z)
                        if clean_peptide_ms2 and placed_ion_labels is not None and len(placed_ion_labels) > 0:
                            L_ms2 = len(clean_peptide_ms2)
                            c_list = sorted([int(lbl[1:]) for lbl in placed_ion_labels if lbl.startswith('c') and lbl[1:].isdigit()])
                            z_list = sorted([int(lbl[1:]) for lbl in placed_ion_labels if re.match(r'^z\d+$', lbl)])
                            z1_list = sorted([int(re.match(r'^z(\d+)\+1$', lbl).group(1)) for lbl in placed_ion_labels if re.match(r'^z\d+\+1$', lbl)])
                            c_list = [n for n in c_list if 1 <= n <= L_ms2]
                            z_list = [n for n in z_list if 1 <= n <= L_ms2]
                            z1_list = [n for n in z1_list if 1 <= n <= L_ms2]
                            c_set = set(c_list)
                            z_set = set(z_list)
                            z1_set = set(z1_list)
                            # Significant single-AA overhangs = positions from consecutive significant fragment pairs only (both ions in pair must be in placed_ion_labels)
                            for n in sorted(c_set):
                                if (n + 1) in c_set:
                                    overhang_positions_from_placed.add(n + 1)
                            for n in sorted(z_set):
                                if (n + 1) in z_set:
                                    overhang_positions_from_placed.add(L_ms2 - n)
                            for n in sorted(z1_set):
                                if (n + 1) in z1_set:
                                    overhang_positions_from_placed.add(L_ms2 - n)
                            # C-terminal overhang: position L from c_{L-1} or from z_1 / z_1+1 (last residue is single-AA fragment)
                            if L_ms2 >= 2 and (L_ms2 - 1) in c_set:
                                overhang_positions_from_placed.add(L_ms2)
                            if 1 in z_set:
                                overhang_positions_from_placed.add(L_ms2)
                            if 1 in z1_set:
                                overhang_positions_from_placed.add(L_ms2)
                            overhang_positions_from_placed.discard(1)  # N-term always black (back exchange)
                            # Require >= 1 overhang from placed ions so we never accept without showing a red square (same definition of "significant" everywhere)
                            n_significant_overhangs = len(overhang_positions_from_placed)
                            if n_significant_overhangs < 1:
                                is_rejected = True
                                rejection_reason = "No significant single-AA overhang (5 ppm, >0.5% intensity)"
                                print(f"[DEBUG]   Rejecting peptide {peptide[:30]}: {rejection_reason}", file=sys.stderr)
                            # Significant pairs for display (consecutive both in placed_ion_labels)
                            significant_pairs_for_display = [f'c{n}|c{n+1}' for n in c_list if (n + 1) in c_set]
                            significant_pairs_for_display += [f'z{n}|z{n+1}' for n in z_list if (n + 1) in z_set]
                            significant_pairs_for_display += [f'z{n}+1|z{n+1}+1' for n in z1_list if (n + 1) in z1_set]
                            # Ions that are part of a significant pair (single-AA overhang): yellow on MS2; others = light gray (c) / dark gray (z)
                            ions_in_pair = set()
                            for n in c_set:
                                if (n - 1) in c_set or n == L_ms2:
                                    ions_in_pair.add(f'c{n}')
                            for n in z_set:
                                if (n - 1) in z_set:
                                    ions_in_pair.add(f'z{n}')
                            for n in z1_set:
                                if (n - 1) in z1_set:
                                    ions_in_pair.add(f'z{n}+1')
                        # MS2 labels: only significant fragments (Comet + 5 ppm + noise %)
                        if label_placements:
                            y_max_label = max(y for (_x, y, _lbl, _xy) in label_placements)
                            y_max_plot = max(y_max_ms2, y_max_label * 1.08)
                            ax_spec_ms2.set_ylim([0, y_max_plot])
                            x_max_label = max(x for (x, _y, _lbl, _xy) in label_placements)
                            ax_spec_ms2.set_xlim([0, max(x_max_ms2, x_max_label * 1.06)])
                        for x_text, y_text, ion_label, xy_peak in label_placements:
                            if ion_label in ions_in_pair:
                                bubble_face = OVERHANG_YELLOW
                                bubble_text_color = 'black'
                                arrow_color = OVERHANG_YELLOW
                            elif ion_label.startswith('c'):
                                bubble_face = C_FRAGMENT_COVERAGE_COLOR
                                bubble_text_color = 'white'
                                arrow_color = C_FRAGMENT_COVERAGE_COLOR
                            else:
                                bubble_face = Z_FRAGMENT_COVERAGE_COLOR
                                bubble_text_color = 'white'
                                arrow_color = Z_FRAGMENT_COVERAGE_COLOR
                            ax_spec_ms2.annotate(ion_label, xy_peak, xytext=(x_text, y_text), fontsize=14, fontfamily='serif', fontweight='bold',
                                ha='center', va='bottom', rotation=0, textcoords='data', color=bubble_text_color,
                                bbox=dict(boxstyle='round,pad=0.45', facecolor=bubble_face, alpha=0.95, edgecolor='black', linewidth=1.0),
                                arrowprops=dict(arrowstyle='-', lw=1.2, color=arrow_color))
                        # Top-right: only significant ion pairs (both fragments pass 5 ppm + noise %)
                        if pairs_col in group.columns:
                            if significant_pairs_for_display:
                                fragment_pairs_text = '\n'.join(p.replace('|', ' | ') for p in significant_pairs_for_display)
                            else:
                                fragment_pairs_text = '—'
                            ax_spec_ms2.text(0.98, 0.98, fragment_pairs_text, transform=ax_spec_ms2.transAxes,
                                fontsize=8, fontfamily='serif', va='top', ha='right', wrap=True,
                                bbox=dict(boxstyle='round,pad=0.25', facecolor=OVERHANG_YELLOW, alpha=0.9, edgecolor='black', linewidth=0.5),
                                zorder=20)
                        # Red bubble: only overhang positions from placed ions (pos + AA)
                        if overhang_positions_from_placed is not None and clean_peptide_ms2 and len(overhang_positions_from_placed) > 0:
                            seq_start = row0.get('sequence_start_pos') if hasattr(row0, 'get') else None
                            try:
                                start = int(seq_start) if pd.notna(seq_start) and seq_start is not None else None
                            except (TypeError, ValueError):
                                start = None
                            if (start is None or start <= 0) and 'sequence_positions' in group.columns:
                                sp = row0.get('sequence_positions')
                                if pd.notna(sp) and str(sp).strip() and '-' in str(sp):
                                    try:
                                        start = int(str(sp).strip().split('-')[0])
                                    except (ValueError, TypeError):
                                        pass
                            pos_aa_parts = []
                            for pos in sorted(overhang_positions_from_placed):
                                if 1 <= pos <= len(clean_peptide_ms2):
                                    aa = clean_peptide_ms2[pos - 1]
                                    protein_pos = (start + pos - 1) if start and start > 0 else pos
                                    pos_aa_parts.append(f'{protein_pos}{aa}')
                            pos_aa_text = ', '.join(pos_aa_parts) if pos_aa_parts else ''
                            if pos_aa_text:
                                ax_spec_ms2.text(0.98, 0.75, pos_aa_text, transform=ax_spec_ms2.transAxes,
                                    fontsize=16, fontfamily='serif', fontweight='bold', va='top', ha='right', wrap=True, color='white',
                                    bbox=dict(boxstyle='round,pad=0.5', facecolor=OVERHANG_RED, alpha=0.95, edgecolor=OVERHANG_RED, linewidth=0.8),
                                    zorder=20)
                        # Store for output CSV: significant fragments, pairs, overhangs (peptide + protein positions); run when we have MS2 + placed ions
                        if placed_ion_labels is not None and peptide is not None and clean_peptide_ms2:
                            _sig_start = None
                            try:
                                _ss = row0.get('sequence_start_pos')
                                if pd.notna(_ss) and _ss is not None:
                                    _sig_start = int(_ss)
                            except (TypeError, ValueError):
                                pass
                            if (_sig_start is None or _sig_start <= 0) and 'sequence_positions' in group.columns and pd.notna(row0.get('sequence_positions')) and '-' in str(row0.get('sequence_positions')):
                                try:
                                    _sig_start = int(str(row0.get('sequence_positions')).strip().split('-')[0])
                                except (ValueError, TypeError):
                                    pass
                            sig_key = (str(peptide).strip(), int(charge), _norm_mods_sig(row0.get('modifications', '-')))
                            sig_ions_str = ', '.join(sorted(placed_ion_labels))
                            sig_pairs_str = ', '.join(significant_pairs_for_display) if significant_pairs_for_display else ''
                            _oph = overhang_positions_from_placed if overhang_positions_from_placed is not None else set()
                            sig_overhang_protein = ', '.join(f'{_sig_start + pos - 1}{clean_peptide_ms2[pos-1]}' if _sig_start and _sig_start > 0 else f'{pos}{clean_peptide_ms2[pos-1]}' for pos in sorted(_oph) if 1 <= pos <= len(clean_peptide_ms2))
                            significant_data[sig_key] = {
                                'significant_fragment_ions': sig_ions_str,
                                'significant_fragment_pairs': sig_pairs_str,
                                # Canonical single-AA overhang mapping column (protein positions).
                                'single_aa_overhangs_protein_positions': sig_overhang_protein,
                            }
                    else:
                        ax_spec_ms2.text(0.5, 0.5, 'No MS2 spectra\nin integration window', ha='center', va='center', fontsize=11, fontfamily='serif', transform=ax_spec_ms2.transAxes)
                        ax_spec_ms2.set_xlim([0, 1])
                        ax_spec_ms2.set_ylim([0, 1])
                        ax_spec_ms2.axis('off')
                else:
                    ax_spec_ms2.axis('off')
                # Reject when we have a peak window but no MS2 or no significant fragments (pipeline uses ONLY significant frags/overhangs/pairs)
                if (raw_file and min_rt is not None and max_rt is not None) and placed_ion_labels is None and not is_rejected:
                    is_rejected = True
                    rejection_reason = "No MS2 in integration window or no significant fragments (5 ppm, >0.5%)"
                    print(f"[DEBUG]   Rejecting peptide {peptide[:30]}: {rejection_reason}", file=sys.stderr)
        
                # Lower right: peptide sequence panel — main row (ALKRINKEL in black squares); above it one row per
                # matched c-ion (fragment sequence in black squares); below it one row per matched z / z+1 (same style).
                clean_peptide = re.sub(r'\[.*?\]', '', peptide) if peptide else ''
                if clean_peptide:
                    L = len(clean_peptide)
                    # Significant fragments from MS2 only (placed_ion_labels = 5 ppm + >0.5% intensity); no CSV fallback.
                    if placed_ion_labels is not None and len(placed_ion_labels) > 0:
                        c_list = sorted([int(lbl[1:]) for lbl in placed_ion_labels if lbl.startswith('c') and lbl[1:].isdigit()])
                        z_list = sorted([int(lbl[1:]) for lbl in placed_ion_labels if re.match(r'^z\d+$', lbl)])
                        z1_list = sorted([int(re.match(r'^z(\d+)\+1$', lbl).group(1)) for lbl in placed_ion_labels if re.match(r'^z\d+\+1$', lbl)])
                        c_list = [n for n in c_list if 1 <= n <= L]
                        z_list = [n for n in z_list if 1 <= n <= L]
                        z1_list = [n for n in z1_list if 1 <= n <= L]
                        c_set = set(c_list)
                        z_set = set(z_list)
                        z1_set = set(z1_list)
                        c_contributing = {n for n in c_list if (n + 1) in c_set or (n - 1) in c_set or (L >= 2 and n == L - 1)}
                        z_contributing = {n for n in z_list if (n + 1) in z_set or (n - 1) in z_set}
                        z1_contributing = {n for n in z1_list if (n + 1) in z1_set or (n - 1) in z1_set}
                        # Red on seq row = positions from consecutive placed pairs; C-term from c_{L-1}, z_1, or z_1+1
                        fragment_overhang_positions = set()
                        for n in c_set:
                            if 1 <= n <= L and ((n - 1) in c_set or n == L):
                                fragment_overhang_positions.add(n)
                            if n == L - 1:
                                fragment_overhang_positions.add(L)  # c_{L-1} → overhang at position L
                        for n in z_set:
                            if 1 <= n <= L and (n - 1) in z_set:
                                fragment_overhang_positions.add(L - n + 1)
                        for n in z1_set:
                            if 1 <= n <= L and (n - 1) in z1_set:
                                fragment_overhang_positions.add(L - n + 1)
                        if L >= 2 and (L - 1) in c_set:
                            fragment_overhang_positions.add(L)  # ensure last residue red when c_{L-1} present
                        if 1 in z_set:
                            fragment_overhang_positions.add(L)  # z_1 = single-AA C-term overhang
                        if 1 in z1_set:
                            fragment_overhang_positions.add(L)  # z_1+1 = single-AA C-term overhang
                        fragment_overhang_positions.discard(1)  # N-term always black (back exchange)
                        # Red = placed ions only (no CSV merge). So "significant" means the same everywhere: we only accept when >=1 overhang from placed (see rejection below), and we only show red for those.
                        overhang_positions = fragment_overhang_positions
                    elif placed_ion_labels is not None:
                        # Had MS2 but no significant labels (empty set): show no red squares; do NOT fall back to CSV so panel matches filename (000_000)
                        c_list, z_list, z1_list = [], [], []
                        c_set = z_set = z1_set = set()
                        c_contributing = z_contributing = z1_contributing = set()
                        overhang_positions = set()
                    else:
                        # No MS2 spectrum: no significant data — show no red squares and no fragment rows (pipeline uses ONLY significant)
                        c_list, z_list, z1_list = [], [], []
                        c_set = z_set = z1_set = set()
                        c_contributing = z_contributing = z1_contributing = set()
                        overhang_positions = set()
                    # Row order (bottom to top in y): z+1 rows, then z rows, then main sequence, then c rows
                    n_z1 = len(z1_list)
                    n_z = len(z_list)
                    n_c = len(c_list)
                    n_rows = n_z1 + n_z + 1 + n_c
                    if n_rows == 0:
                        n_rows = 1
                    label_w = 0.9
                    intensity_col_w = 4.0  # space for intensity (e.g. "1.2e6") to the right of grid
                    ax_seq.set_ylim(0, n_rows + 0.4)
                    ax_seq.set_xlim(-label_w, L + intensity_col_w)
                    ax_seq.set_aspect('auto')  # Fill lower-right corner; equal aspect shrinks long-peptide grids
                    ax_seq.axis('off')
                    # Title: peptide sequence, charge, human-readable mods, percent of total area; then color legend
                    def _human_readable_mods(mods):
                        if pd.isna(mods) or mods == '-' or str(mods).strip() == '':
                            return 'none'
                        mod_map = {'15.9949': 'Ox', '57.0215': 'Carbamidomethyl', '0.9840': 'Deamidated'}
                        mods_str = str(mods)
                        mod_parts = []
                        for mod_entry in mods_str.split(','):
                            mod_entry = mod_entry.strip()
                            if '_' in mod_entry:
                                parts = mod_entry.split('_')
                                if len(parts) >= 3:
                                    pos = parts[0]
                                    mass = parts[2].split(')')[0].strip()
                                    mod_name = mod_map.get(mass, f'+{mass}')
                                    mod_parts.append(f'{mod_name}@{pos}')
                        if not mod_parts:
                            return mods_str[:25] + ('...' if len(mods_str) > 25 else '')
                        return ', '.join(mod_parts[:4]) + (f' (+{len(mod_parts)-4} more)' if len(mod_parts) > 4 else '')
                    _mods_hr = _human_readable_mods(group['modifications'].iloc[0] if 'modifications' in group.columns else '-')
                    _ch = int(charge) if charge is not None and not (isinstance(charge, float) and np.isnan(charge)) else 0
                    try:
                        _ta = all_peptide_scatter_data.get(peptide_key, {}).get('total_area')
                    except NameError:
                        _ta = None
                    if _ta is None or (isinstance(_ta, float) and (np.isnan(_ta) or _ta <= 0)):
                        _ta = overlay_by_key.get(peptide_key)
                    _sum_all = sum(overlay_by_key.values()) if overlay_by_key else None
                    _pct = (100.0 * float(_ta) / float(_sum_all)) if (_ta is not None and _sum_all and float(_sum_all) > 0 and float(_ta) >= 0) else None
                    _pct_str = f'  {_pct:.2f}% of total area' if _pct is not None else ''
                    _title_line1 = f'{clean_peptide}   +{_ch}   {_mods_hr}{_pct_str}'
                    ax_seq.set_title(f'{_title_line1}\n(red = overhang, blue-black = coverage, yellow = fragment overhang)', fontsize=10, fontweight='bold', fontfamily='serif', color='0.9')
                    # Column header for intensity (5 ppm match from summed MS2)
                    if n_z1 or n_z or n_c:
                        ax_seq.text(L + intensity_col_w / 2, n_rows + 0.15, 'Intensity\n(5 ppm)', fontsize=8, ha='center', va='bottom', color='0.85', fontfamily='serif')
                    cell_w = 1.0
                    # Build fragment -> intensity map from summed MS2 (within 5 ppm of theoretical m/z)
                    frag_intensity = {}
                    if (n_z1 or n_z or n_c) and raw_file and min_rt is not None and max_rt is not None:
                        try:
                            c_theo, z_theo, z1_theo = theoretical_c_z_z1_ion_mz(clean_peptide, fragment_charge=1)
                            for n in z1_list:
                                if 1 <= n <= len(z1_theo):
                                    mz = z1_theo[n - 1][0]
                                    frag_intensity[f'z{n}+1'] = _ms2_intensity_within_ppm(ms2_mzs, ms2_ints, mz, ppm=5.0)
                            for n in z_list:
                                if 1 <= n <= len(z_theo):
                                    mz = z_theo[n - 1][0]
                                    frag_intensity[f'z{n}'] = _ms2_intensity_within_ppm(ms2_mzs, ms2_ints, mz, ppm=5.0)
                            for n in c_list:
                                if 1 <= n <= len(c_theo):
                                    mz = c_theo[n - 1][0]
                                    frag_intensity[f'c{n}'] = _ms2_intensity_within_ppm(ms2_mzs, ms2_ints, mz, ppm=5.0)
                        except NameError:
                            pass  # ms2_mzs/ms2_ints not in scope (no MS2 block)
                    row = 0
                    # z+1 rows: shortest at bottom, longest adjacent to seq (ascending n)
                    for n in z1_list:
                        if n < 1 or n > L:
                            continue
                        ax_seq.text(-label_w / 2, row + 0.5, f'z{n}+1', fontsize=14, ha='center', va='center', color='0.9', fontweight='bold', fontfamily='serif')
                        for i in range(L - n, L):
                            is_overhang = (i == L - n and (n - 1) in z1_set)
                            fill_color = OVERHANG_YELLOW if is_overhang else Z_FRAGMENT_COVERAGE_COLOR
                            text_color = 'black' if fill_color == OVERHANG_YELLOW else 'white'
                            rect = Rectangle((i, row), cell_w, 1, facecolor=fill_color, edgecolor='0.5', linewidth=0.5)
                            ax_seq.add_patch(rect)
                            ax_seq.text(i + 0.5, row + 0.5, clean_peptide[i], fontsize=14, ha='center', va='center', color=text_color, fontweight='bold', fontfamily='monospace')
                        for i in range(0, L - n):
                            rect = Rectangle((i, row), cell_w, 1, facecolor='#1a1a1a', edgecolor='0.5', linewidth=0.5)
                            ax_seq.add_patch(rect)
                        int_val = frag_intensity.get(f'z{n}+1', 0.0)
                        int_str = f'{int_val:.2e}' if int_val > 0 else '—'
                        ax_seq.text(L + intensity_col_w / 2, row + 0.5, int_str, fontsize=9, ha='center', va='center', color='0.9', fontfamily='serif')
                        row += 1
                    # z rows: shortest at bottom, longest adjacent to seq (ascending n)
                    for n in z_list:
                        if n < 1 or n > L:
                            continue
                        ax_seq.text(-label_w / 2, row + 0.5, f'z{n}', fontsize=14, ha='center', va='center', color='0.9', fontweight='bold', fontfamily='serif')
                        for i in range(L - n, L):
                            is_overhang = (i == L - n and (n - 1) in z_set)
                            fill_color = OVERHANG_YELLOW if is_overhang else Z_FRAGMENT_COVERAGE_COLOR
                            text_color = 'black' if fill_color == OVERHANG_YELLOW else 'white'
                            rect = Rectangle((i, row), cell_w, 1, facecolor=fill_color, edgecolor='0.5', linewidth=0.5)
                            ax_seq.add_patch(rect)
                            ax_seq.text(i + 0.5, row + 0.5, clean_peptide[i], fontsize=14, ha='center', va='center', color=text_color, fontweight='bold', fontfamily='monospace')
                        for i in range(0, L - n):
                            rect = Rectangle((i, row), cell_w, 1, facecolor='#1a1a1a', edgecolor='0.5', linewidth=0.5)
                            ax_seq.add_patch(rect)
                        int_val = frag_intensity.get(f'z{n}', 0.0)
                        int_str = f'{int_val:.2e}' if int_val > 0 else '—'
                        ax_seq.text(L + intensity_col_w / 2, row + 0.5, int_str, fontsize=9, ha='center', va='center', color='0.9', fontfamily='serif')
                        row += 1
                    # Main sequence row (ALKRINKEL in black/red squares). Position 1 (N-term) always black (back exchange).
                    ax_seq.text(-label_w / 2, row + 0.5, 'seq', fontsize=14, ha='center', va='center', color='0.9', fontweight='bold', fontfamily='serif')
                    for i in range(L):
                        pos_1based = i + 1
                        face = OVERHANG_RED if (pos_1based != 1 and pos_1based in overhang_positions) else (*BLUE_BLACK, 0.7)
                        rect = Rectangle((i, row), cell_w, 1, facecolor=face, edgecolor='0.5', linewidth=0.5)
                        ax_seq.add_patch(rect)
                        ax_seq.text(i + 0.5, row + 0.5, clean_peptide[i], fontsize=18, ha='center', va='center',
                                    color='white', fontweight='bold', fontfamily='monospace')
                    row += 1
                    # c rows: longest closest to main, shortest farthest above (reversed)
                    for n in reversed(c_list):
                        if n < 1 or n > L:
                            continue
                        ax_seq.text(-label_w / 2, row + 0.5, f'c{n}', fontsize=14, ha='center', va='center', color='0.9', fontweight='bold', fontfamily='serif')
                        for i in range(n):
                            is_overhang = (i == n - 1 and (n - 1) in c_set) or (i == L - 1 and n == L - 1)
                            fill_color = OVERHANG_YELLOW if is_overhang else C_FRAGMENT_COVERAGE_COLOR
                            text_color = 'black' if fill_color == OVERHANG_YELLOW else 'white'
                            rect = Rectangle((i, row), cell_w, 1, facecolor=fill_color, edgecolor='0.5', linewidth=0.5)
                            ax_seq.add_patch(rect)
                            ax_seq.text(i + 0.5, row + 0.5, clean_peptide[i], fontsize=14, ha='center', va='center', color=text_color, fontweight='bold', fontfamily='monospace')
                        for i in range(n, L):
                            rect = Rectangle((i, row), cell_w, 1, facecolor='#1a1a1a', edgecolor='0.5', linewidth=0.5)
                            ax_seq.add_patch(rect)
                        int_val = frag_intensity.get(f'c{n}', 0.0)
                        int_str = f'{int_val:.2e}' if int_val > 0 else '—'
                        ax_seq.text(L + intensity_col_w / 2, row + 0.5, int_str, fontsize=9, ha='center', va='center', color='0.9', fontfamily='serif')
                        row += 1
                else:
                    ax_seq.axis('off')
        
                # Caption removed per user request
        
                # Draw to combined figure only for non-rejected peptides (slot = picked_idx; no gaps)
                if not is_rejected:
                    # Add annotations to combined figure chromatogram (mark as combined for smaller labels)
                    ax_combined_chrom._is_combined = True
                    add_peak_annotations(ax_combined_chrom, scored_candidates, best_peak, min_rt_min, max_rt_min, 
                               apex_rt_min, rts, rts_min, total_intensities, min_rt, max_rt,
                               rt_anchor=rt_anchor, anchor_evalue=anchor_evalue,
                               evalue_cmap=evalue_cmap, evalue_norm=evalue_norm,
                               peptide_crossref=peptide_crossref, peptide_key=peptide_key,
                               rt_to_frag_count=rt_to_frag_count, ms1_rts_table=ms1_rts_table,
                               collection_min_rt_min=collection_min_rt_min, collection_max_rt_min=collection_max_rt_min,
                               significant_frag_count=significant_frag_count,
                               drift_min_rt_min=drift_min_rt_min, drift_max_rt_min=drift_max_rt_min,
                               alternative_reasons=alternative_reasons)
        
                def format_title(peptide, charge, mods):
                    """Format peptide title with modifications"""
                    if pd.isna(mods) or mods == '-':
                        return f"{peptide} (+{charge})"
                    else:
                        mods_str = str(mods)
                        mod_map = {
                            '15.9949': 'Ox',
                            '57.0215': 'Carbamidomethyl',
                            '0.9840': 'Deamidated',
                        }
                        mod_parts = []
                        for mod_entry in mods_str.split(','):
                            if '_' in mod_entry:
                                parts = mod_entry.split('_')
                                if len(parts) >= 3:
                                    pos = parts[0]
                                    mod_type = parts[1]
                                    mass = parts[2].split(')')[0]
                                    mod_name = mod_map.get(mass, f"+{mass}")
                                    mod_parts.append(f"{mod_name}@{pos}")
                        if mod_parts:
                            mods_display = ','.join(mod_parts[:3])
                            if len(mod_parts) > 3:
                                mods_display += f" (+{len(mod_parts)-3} more)"
                            return f"{peptide} (+{charge}, {mods_display})"
                        else:
                            return f"{peptide} (+{charge}, {mods_str[:30]})"
        
                seq_start_pos = 999999
                if 'sequence_positions' in group.columns:
                    seq_pos = group['sequence_positions'].iloc[0]
                    if pd.notna(seq_pos) and seq_pos != '':
                        try:
                            if '-' in str(seq_pos):
                                seq_start_pos = int(str(seq_pos).split('-')[0])
                        except:
                            pass
        
                mods = group['modifications'].iloc[0] if 'modifications' in group.columns else '-'
                title = format_title(peptide, charge, mods)
        
                # Detailed figure-level title: identity (sequence position, peptide, charge, mods, m/z), apex, windows, metrics
                mods_display = (str(mods) if pd.notna(mods) and mods != '-' else 'none')
                if len(mods_display) > 40:
                    mods_display = mods_display[:37] + '...'
                integration_str = f'{min_rt_min:.2f}-{max_rt_min:.2f} min' if (min_rt_min is not None and max_rt_min is not None) else 'N/A'
                collection_str = f'{collection_min_rt_min:.2f}-{collection_max_rt_min:.2f} min' if (collection_min_rt_min is not None and collection_max_rt_min is not None) else 'N/A'
                apex_str = f'{apex_rt_min:.2f} min' if apex_rt_min is not None else 'N/A'
                pos_str = f'pos {seq_start_pos}' if seq_start_pos != 999999 else '—'
                n_psms = len(group)
                try:
                    ta = total_area
                    total_area_val = ta if (ta is not None and not (isinstance(ta, float) and (np.isnan(ta) or ta < 0))) else None
                except NameError:
                    total_area_val = None
                area_str = f'  total_area: {total_area_val:.2e}' if total_area_val is not None else ''
                apex_intensity_val = best_peak.get('apex_intensity')
                if apex_intensity_val is not None and not (isinstance(apex_intensity_val, float) and (np.isnan(apex_intensity_val) or apex_intensity_val < 0)):
                    apex_peak_str = f'  apex_peak: {apex_intensity_val:.2e}'
                else:
                    apex_peak_str = ''
                apex_rt_str = f'  apex_rt: {apex_str}' if apex_str != 'N/A' else ''
                big_title_lines = [
                f'Sequence position: {pos_str}   |   Peptide: {peptide}   |   Charge: z{int(charge)}   |   Mods: {mods_display}   |   m/z: {precursor_mz:.4f}',
                f'Apex RT: {apex_str}   |   Integration window: {integration_str}   |   Collection window (grey): {collection_str}',
                f'1 unique peptide (seq+charge+mods), {n_psms} spectra{apex_rt_str}{apex_peak_str}{area_str}',
                f'Filtering: {FILTERING_CRITERIA}'
                ]
                fig.suptitle('\n'.join(big_title_lines), fontsize=24, fontweight='bold', fontfamily='serif', y=0.98, va='top', color='0.9')
        
                # Get peptide-level scores and q-values for this unique peptide (sequence+charge+mods)
                # Use spectrum (row) closest to detected peak apex RT; fallback to best q-value, then best e-value
                best_psm = None
                best_psm_idx = None
        
                if apex_rt and 'MS1_retention_time_sec' in group.columns:
                    group_with_rt = group[group['MS1_retention_time_sec'].notna()]
                    if len(group_with_rt) > 0:
                        rt_diffs = abs(group_with_rt['MS1_retention_time_sec'] - apex_rt)
                        closest_idx = rt_diffs.idxmin()
                        best_psm_idx = closest_idx
                        best_psm = group.loc[closest_idx]
        
                if best_psm is None:
                    if 'percolator_qvalue' in group.columns:
                        qvalue_mask = group['percolator_qvalue'].notna()
                        if qvalue_mask.any():
                            best_psm_idx = group[qvalue_mask]['percolator_qvalue'].idxmin()
                        else:
                            evals = pd.to_numeric(group['e-value'], errors='coerce').dropna() if 'e-value' in group.columns else pd.Series(dtype=float)
                            best_psm_idx = evals.idxmin() if len(evals) > 0 and np.isfinite(evals).any() else group.index[0]
                    else:
                        evals = pd.to_numeric(group['e-value'], errors='coerce').dropna() if 'e-value' in group.columns else pd.Series(dtype=float)
                        best_psm_idx = evals.idxmin() if len(evals) > 0 and np.isfinite(evals).any() else group.index[0]
                    best_psm = group.loc[best_psm_idx]
        
                def _score_from_psm(psm, col, *fallback_keys):
                    if col and col in psm.index:
                        v = psm[col]
                        if pd.notna(v) and v != '':
                            try:
                                return float(v)
                            except (TypeError, ValueError):
                                pass
                    for k in fallback_keys:
                        if k in psm.index:
                            v = psm[k]
                            if pd.notna(v) and v != '':
                                try:
                                    return float(v)
                                except (TypeError, ValueError):
                                    pass
                    return None
                e_value = _score_from_psm(best_psm, evalue_col, 'e-value', 'e_value', 'E-value')
                qvalue = _score_from_psm(best_psm, qvalue_col, 'percolator_qvalue', 'q-value', 'qvalue', 'Q-value')
                pep = _score_from_psm(best_psm, pep_col, 'percolator_PEP', 'pep', 'PEP')
        
                # Format scores for display with quality indicators (no Xcorr, no Sp)
                score_lines = []
                score_quality = []  # Track quality for each score
        
                if e_value is not None and pd.notna(e_value):
                    if e_value < 1e-5:
                        quality = "good"
                    elif e_value > 0.01:
                        quality = "poor"
                    else:
                        quality = "fair"
                    score_lines.append(f'E-value: {e_value:.2e}')
                    score_quality.append(quality)
        
                if qvalue is not None and pd.notna(qvalue):
                    if qvalue < 0.01:
                        quality = "good"
                    elif qvalue > 0.05:
                        quality = "poor"
                    else:
                        quality = "fair"
                    score_lines.append(f'Q-value: {qvalue:.2e}')
                    score_quality.append(quality)
        
                if pep is not None and pd.notna(pep):
                    if pep < 0.01:
                        quality = "good"
                    elif pep > 0.05:
                        quality = "poor"
                    else:
                        quality = "fair"
                    score_lines.append(f'PEP: {pep:.2e}')
                    score_quality.append(quality)
        
                ratio_stability_display = _fmt_ratio_stability_plot(ratio_stability)
                qc_text = f'Best Peak:\nCoelution Variance: {coelution_var:.2f}s\nShape Correlation: {shape_corr:.3f}\nRatio Stability: {ratio_stability_display}\nRT Offset: {rt_offset:.1f}s\nPeak Score: {best_score:.3f}'
                if n_candidates > 1:
                    qc_text += f'\n\n({n_candidates} peaks found)'
                if second_best_peak:
                    qc_text += f'\n\nRunner-up Score: {second_best_score:.3f}'
                    qc_text += f'\nΔ Score: {delta_score:.3f}'
                    if peak2_loss_reasons:
                        qc_text += f'\n\nPeak 2 lost on:\n' + '\n'.join(peak2_loss_reasons[:3])
                if interference_flag:
                    qc_text += '\n[!] INTERFERENCE/AMBIGUITY'
                if rt_deviation_flag:
                    qc_text += f'\n[FLAG] Large RT deviation ({rt_offset:.1f}s)'
                    if not interference_flag:
                        qc_text += '\n(but isotope evidence strong)'
        
                # Add scores section with quality indicators
                if score_lines:
                    qc_text += '\n\nPeptide Scores:'
                    for score_line, quality in zip(score_lines, score_quality):
                        quality_label = "[OK] good" if quality == "good" else ("[X] poor" if quality == "poor" else "~ fair")
                        qc_text += f'\n{score_line} ({quality_label})'
        
                # Add QC text box inside the plot area (upper left corner)
                # Position it using axes coordinates so it stays within plot boundaries
                ax.text(0.02, 0.98, qc_text, transform=ax.transAxes,
                   fontsize=13, verticalalignment='top', horizontalalignment='left', fontfamily='serif',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5, pad=0.35),
                   wrap=True,  # Enable text wrapping
                   clip_on=True)  # Clip to axes boundaries
        
                # Add scores to combined figure (smaller, top-right) - only for first plot of each file to avoid clutter
                # file_idx, local_idx are from picked_idx when not rejected (set above)
                if not is_rejected and score_lines and local_idx < 3:
                    score_text_combined_lines = []
                    for score_line, quality in zip(score_lines[:3], score_quality[:3]):
                        quality_label = "[OK]" if quality == "good" else ("[X]" if quality == "poor" else "~")
                        score_text_combined_lines.append(f'{score_line} {quality_label}')
                    score_text_combined = '\n'.join(score_text_combined_lines)
                    fontsize_score = 9 if num_peptides > 100 else 10
                    ax_combined_chrom.text(0.98, 0.98, score_text_combined, transform=ax_combined_chrom.transAxes,
                               fontsize=fontsize_score, verticalalignment='top', horizontalalignment='right',
                               fontfamily='serif',
                               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5),
                               clip_on=True)  # Clip to axes boundaries
        
                # Format individual plot (Times New Roman font) - axis title omitted; large suptitle above has full details
                # Black background: light tick/label colors for visibility
                ax.set_xlabel('MS1 Retention Time (min)', fontsize=16, fontfamily='serif', color='0.85')
                ax.set_ylabel('Intensity', fontsize=16, fontfamily='serif', color='0.85')
                ax.tick_params(labelsize=14, colors='0.85')
                for label in ax.get_xticklabels() + ax.get_yticklabels():
                    label.set_fontfamily('serif')
                    label.set_color('0.85')
                ax.grid(False)
                for spine in ax.spines.values():
                    spine.set_color('0.6')
        
                # Add secondary x-axis on top showing seconds
                def min_to_sec(x):
                    return x * 60.0
                def sec_to_min(x):
                    return x / 60.0
                ax_top = ax.secondary_xaxis('top', functions=(min_to_sec, sec_to_min))
                ax_top.set_xlabel('MS1 Retention Time (s)', fontsize=16, fontfamily='serif', color='0.85')
                ax_top.tick_params(labelsize=14, colors='0.85')
                for label in ax_top.get_xticklabels():
                    label.set_fontfamily('serif')
                    label.set_color('0.85')
        
                if not is_rejected:
                    title_combined = f"[{seq_start_pos}] {title}" if seq_start_pos != 999999 else title
                    ax_combined_chrom.set_title(title_combined, fontsize=11, fontweight='bold', pad=4, fontfamily='serif', color='0.9')
                    ax_combined_chrom.set_xlabel('MS1 RT (min)', fontsize=10, fontfamily='serif', color='0.85')
                    ax_combined_chrom.set_ylabel('Intensity', fontsize=10, fontfamily='serif', color='0.85')
                    ax_combined_chrom.tick_params(labelsize=10, colors='0.85')
                    for label in ax_combined_chrom.get_xticklabels() + ax_combined_chrom.get_yticklabels():
                        label.set_fontfamily('serif')
                        label.set_color('0.85')
                    ax_combined_chrom.grid(False)
                    for spine in ax_combined_chrom.spines.values():
                        spine.set_color('0.6')
                    ax_combined_top = ax_combined_chrom.secondary_xaxis('top', functions=(min_to_sec, sec_to_min))
                    ax_combined_top.set_xlabel('MS1 RT (s)', fontsize=10, fontfamily='serif', color='0.85')
                    ax_combined_top.tick_params(labelsize=10, colors='0.85')
                    for label in ax_combined_top.get_xticklabels():
                        label.set_fontfamily('serif')
                        label.set_color('0.85')
        
                # Format y-axis (offset text for scientific notation: light color on black)
                ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0), useMathText=True)
                try:
                    ax.yaxis.get_offset_text().set_color('0.85')
                except Exception:
                    pass
        
                # Set x-axis limits to span ALL table RTs for this peptide (so both clusters visible on one plot)
                # Use full range with padding; extend beyond collection window so we can see context.
                # Peak drift window is defined as collection window ± PEAK_DRIFT_BUFFER_SEC (see regenerate_rt_windows_from_csv._compute_drift_limits_per_channel).
                # Always derive the drawn drift from THIS run's collection so the grey collection and the drift band stay aligned.
                drift_min_rt = None
                drift_max_rt = None
                if collection_min_rt is not None and collection_max_rt is not None:
                    drift_min_rt = max(0.0, float(collection_min_rt) - PEAK_DRIFT_BUFFER_SEC)
                    drift_max_rt = float(collection_max_rt) + PEAK_DRIFT_BUFFER_SEC
                # Fallback: if no collection yet (shouldn't happen here), use filter CSV drift when provided
                if (drift_min_rt is None or drift_max_rt is None) and 'drift_min_rt' in group.columns and 'drift_max_rt' in group.columns:
                    vmin = group['drift_min_rt'].dropna()
                    vmax = group['drift_max_rt'].dropna()
                    if len(vmin) > 0 and len(vmax) > 0:
                        try:
                            drift_min_rt = float(vmin.iloc[0])
                            drift_max_rt = float(vmax.iloc[0])
                        except (TypeError, ValueError):
                            pass
                global_rt_min_sec = min(ms1_rts_table) if ms1_rts_table else min_rt
                global_rt_max_sec = max(ms1_rts_table) if ms1_rts_table else max_rt
                span_sec = global_rt_max_sec - global_rt_min_sec
                collection_span_min = (max_rt - min_rt) / 60.0 if (min_rt and max_rt) else span_sec / 60.0
                buffer_sec = max(90.0, span_sec * 0.25, collection_span_min * 60.0 * 0.5)  # at least 1.5 min, or 25% of span, or 50% of collection window
                buffer_min = buffer_sec / 60.0
                x_min = max(0, global_rt_min_sec / 60.0 - buffer_min)
                x_max = global_rt_max_sec / 60.0 + buffer_min

                # Ensure matched peak window and apex are included; extend well beyond collection window for context
                if min_rt_min < x_min:
                    x_min = max(0, min_rt_min - buffer_min * 1.5)
                if max_rt_min > x_max:
                    x_max = max_rt_min + buffer_min * 1.5
                if rt_anchor:
                    anchor_rt_min = rt_anchor / 60.0
                    x_min = min(x_min, anchor_rt_min - buffer_min * 1.5)
                    x_max = max(x_max, anchor_rt_min + buffer_min * 1.5)
                if apex_rt_min:
                    x_min = min(x_min, apex_rt_min - buffer_min * 1.5)
                    x_max = max(x_max, apex_rt_min + buffer_min * 1.5)
                if x_max <= x_min:
                    x_min = max(0, min_rt_min - buffer_min)
                    x_max = max_rt_min + buffer_min
                # Ensure chromatogram data is visible: if no trace points fall in [x_min, x_max], expand to include data range
                if len(rts_min) > 0:
                    data_min = float(np.min(rts_min))
                    data_max = float(np.max(rts_min))
                    if data_max < x_min or data_min > x_max:
                        x_min = max(0, data_min - buffer_min)
                        x_max = data_max + buffer_min
                # Ensure collection window is always visible on the time-window plot (avoid clipping when close to RT range)
                margin_coll_min = max(0.008, (max_rt_min - min_rt_min) * 0.02) if (min_rt_min is not None and max_rt_min is not None) else 0.01
                if collection_min_rt_min is not None and collection_min_rt_min < x_min:
                    x_min = max(0, collection_min_rt_min - margin_coll_min)
                if collection_max_rt_min is not None and collection_max_rt_min > x_max:
                    x_max = collection_max_rt_min + margin_coll_min
                # Extend x-axis to accommodate drift window limits when provided (e.g. from RT-window filter CSV)
                if drift_min_rt is not None and drift_max_rt is not None and drift_max_rt > drift_min_rt:
                    drift_min_min = drift_min_rt / 60.0
                    drift_max_min = drift_max_rt / 60.0
                    x_min = min(x_min, max(0, drift_min_min - buffer_min))
                    x_max = max(x_max, drift_max_min + buffer_min)
                ax.set_xlim([x_min, x_max])
                if not is_rejected:
                    ax_combined_chrom.set_xlim([x_min, x_max])
                # Overlay panels: same x duration as top chromatogram (sync now and again after drawing)
                ax_overlay.set_xlim(x_min, x_max)
                ax_overlay_log.set_xlim(x_min, x_max)
                # Layer order: black only for peak drift buffer (inside drift window, outside collection); areas outside drift stay white
                if drift_min_rt is not None and drift_max_rt is not None and drift_max_rt > drift_min_rt:
                    drift_min_min = drift_min_rt / 60.0
                    drift_max_min = drift_max_rt / 60.0
                    z_drift = -8   # peak drift buffer (black) = inside drift, outside collection
                    coll_left = collection_min_rt_min if collection_min_rt_min is not None else min_rt_min
                    coll_right = collection_max_rt_min if collection_max_rt_min is not None else max_rt_min
                    if coll_left is None:
                        coll_left = min_rt_min
                    if coll_right is None:
                        coll_right = max_rt_min
                    # Left buffer: only inside drift window [drift_min_min, coll_left]; draw only when collection starts inside drift (coll_left > drift_min_min)
                    left_buf_end = min(coll_left, drift_max_min) if coll_left is not None else drift_max_min
                    if left_buf_end > drift_min_min and (coll_left is None or coll_left > drift_min_min):
                        ax.axvspan(drift_min_min, left_buf_end, facecolor='black', alpha=1.0, zorder=z_drift)
                        if not is_rejected:
                            ax_combined_chrom.axvspan(drift_min_min, left_buf_end, facecolor='black', alpha=1.0, zorder=z_drift)
                    # Right buffer: only inside drift window [coll_right, drift_max_min]; draw only when collection ends inside drift (coll_right < drift_max_min)
                    right_buf_start = max(coll_right, drift_min_min) if coll_right is not None else drift_min_min
                    if drift_max_min > right_buf_start and (coll_right is None or coll_right < drift_max_min):
                        ax.axvspan(right_buf_start, drift_max_min, facecolor='black', alpha=1.0, zorder=z_drift)
                        if not is_rejected:
                            ax_combined_chrom.axvspan(right_buf_start, drift_max_min, facecolor='black', alpha=1.0, zorder=z_drift)
        
                # Verify anchor RT is visible (debug for first few peptides)
                if idx < 3 and rt_anchor:
                    anchor_rt_min_check = rt_anchor / 60.0
                    x_lim_actual = ax.get_xlim()
                    if not (x_lim_actual[0] <= anchor_rt_min_check <= x_lim_actual[1]):
                        print(f"[DEBUG]   ERROR: Anchor RT {anchor_rt_min_check:.3f} min still outside x-axis range [{x_lim_actual[0]:.3f}, {x_lim_actual[1]:.3f}] after adjustment")
        
                if len(total_intensities) > 0:
                    # rts_min is in minutes (same units as x_min, x_max)
                    window_mask = (rts_min >= x_min) & (rts_min <= x_max)
                    window_intensities = total_intensities[window_mask] if np.any(window_mask) else total_intensities
                    # Use window max for y when we have signal in view; else use full trace so peak is never flattened
                    if len(window_intensities) > 0 and np.max(window_intensities) > 0:
                        y_max = np.max(window_intensities) * 1.1
                        y_min = 0
                        ax.set_ylim([y_min, y_max])
                        if not is_rejected:
                            ax_combined_chrom.set_ylim([y_min, y_max])
                            drew_combined_spec = True
                    elif len(total_intensities) > 0 and np.nanmax(total_intensities) > 0:
                        # Window has no signal but full trace does: scale y from full data so peak is visible
                        y_max = float(np.nanmax(total_intensities)) * 1.1
                        y_min = 0
                        ax.set_ylim([y_min, y_max])
                        if not is_rejected:
                            ax_combined_chrom.set_ylim([y_min, y_max])
                            drew_combined_spec = True
                    else:
                        ax.set_ylim([-0.1, 0.1])
                        if not is_rejected:
                            ax_combined_chrom.set_ylim([-0.1, 0.1])
                            drew_combined_spec = True
        
                # Window bar below summed spectrum: same x-axis (time). Layer order: peak drift (bottom) -> collection -> integration (top)
                ax_windows.set_ylim(0, 1)
                ax_windows.set_yticks([])
                ax_windows.set_ylabel('')
                ax_windows.set_xlabel('')
                # Outline around the window bar (light gray on black background)
                for spine in ax_windows.spines.values():
                    spine.set_visible(True)
                    spine.set_edgecolor('0.6')
                    spine.set_linewidth(1.2)
                # Same as main chromatogram: black only for peak drift buffer (inside drift, outside collection); outside drift stays white
                if drift_min_rt is not None and drift_max_rt is not None and drift_max_rt > drift_min_rt:
                    drift_min_min_w = drift_min_rt / 60.0
                    drift_max_min_w = drift_max_rt / 60.0
                    z_drift_w = -8
                    coll_left_w = collection_min_rt_min if collection_min_rt_min is not None else min_rt_min
                    coll_right_w = collection_max_rt_min if collection_max_rt_min is not None else max_rt_min
                    if coll_left_w is None:
                        coll_left_w = min_rt_min
                    if coll_right_w is None:
                        coll_right_w = max_rt_min
                    left_buf_end_w = min(coll_left_w, drift_max_min_w) if coll_left_w is not None else drift_max_min_w
                    if left_buf_end_w > drift_min_min_w and (coll_left_w is None or coll_left_w > drift_min_min_w):
                        ax_windows.axvspan(drift_min_min_w, left_buf_end_w, facecolor='black', alpha=1.0, zorder=z_drift_w)
                    right_buf_start_w = max(coll_right_w, drift_min_min_w) if coll_right_w is not None else drift_min_min_w
                    if drift_max_min_w > right_buf_start_w and (coll_right_w is None or coll_right_w < drift_max_min_w):
                        ax_windows.axvspan(right_buf_start_w, drift_max_min_w, facecolor='black', alpha=1.0, zorder=z_drift_w)
                # Collection window always includes integration and more; draw one opaque gray band then integration on top
                coll_left = collection_min_rt_min if collection_min_rt_min is not None else min_rt_min
                coll_right = collection_max_rt_min if collection_max_rt_min is not None else max_rt_min
                if coll_left is None:
                    coll_left = min_rt_min
                if coll_right is None:
                    coll_right = max_rt_min
                if min_rt_min is not None and coll_left is not None and coll_left > min_rt_min:
                    coll_left = min_rt_min
                if max_rt_min is not None and coll_right is not None and coll_right < max_rt_min:
                    coll_right = max_rt_min
                if min_rt_min is not None and max_rt_min is not None and max_rt_min > min_rt_min:
                    window_bar_margin = max(0.008, (max_rt_min - min_rt_min) * 0.02)
                    draw_left = min(coll_left, min_rt_min - window_bar_margin) if coll_left is not None else (min_rt_min - window_bar_margin)
                    draw_right = max(coll_right, max_rt_min + window_bar_margin) if coll_right is not None else (max_rt_min + window_bar_margin)
                else:
                    draw_left = coll_left
                    draw_right = coll_right
                from matplotlib.colors import Normalize
                cmap_ta = _total_area_colormap_early()
                ta_val = None
                if total_area is not None and not (isinstance(total_area, float) and (np.isnan(total_area) or total_area < 0)):
                    try:
                        ta_val = float(total_area)
                    except (TypeError, ValueError):
                        pass
                # Use same norm as overlay (norm_overlay/area_norm_global) so window bar and overlay colors match
                if norm_overlay is not None and ta_val is not None:
                    nval = np.clip(norm_overlay(ta_val), 0, 1)
                    bar_color = cmap_ta(nval)
                elif area_vmax_global > area_vmin_global:
                    area_norm_bar = Normalize(vmin=area_vmin_global, vmax=area_vmax_global)
                    nval = np.clip(area_norm_bar(ta_val), 0, 1) if ta_val is not None else 0.5
                    bar_color = cmap_ta(nval)
                else:
                    area_norm_bar = Normalize(vmin=0, vmax=max(ta_val or 1, 1))
                    nval = np.clip(area_norm_bar(ta_val), 0, 1) if ta_val is not None else 0.5
                    bar_color = cmap_ta(nval)
                # Time window plot: collection (gray) then integration window (colored by total_area) — same min_rt/max_rt as main chrom green band
                if draw_left is not None and draw_right is not None and draw_right > draw_left:
                    ax_windows.axvspan(draw_left, draw_right, alpha=COLLECTION_WINDOW_ALPHA, color=COLLECTION_WINDOW_COLOR, zorder=0)
                if min_rt_min is not None and max_rt_min is not None:
                    ax_windows.axvspan(min_rt_min, max_rt_min, alpha=INTEGRATION_WINDOW_ALPHA, color='#00FF7F', zorder=1)  # integration window (green, matches main chromatogram)
                ax_windows.tick_params(axis='x', labelbottom=False)
        
                # Bottom: overlay of ALL peptides' total peaks (pre-computed); this peptide on top with black outline (colored by total_area, same scale as window bar)
                from matplotlib.colors import Normalize
                cmap_overlay = _total_area_colormap_early()
                # norm_overlay is set at start of pass 1 to area_norm_global (so colors match window bar/overview/zoom); use total_area for color when that norm is in use
                try:
                    use_total_area_for_overlay = (norm_overlay is area_norm_global)
                except NameError:
                    use_total_area_for_overlay = False
                # Collection window (lighter gray) then peak drift buffer (black): same as main chromatogram
                coll_left_o = collection_min_rt_min if collection_min_rt_min is not None else min_rt_min
                coll_right_o = collection_max_rt_min if collection_max_rt_min is not None else max_rt_min
                if coll_left_o is None:
                    coll_left_o = min_rt_min
                if coll_right_o is None:
                    coll_right_o = max_rt_min
                if coll_left_o is not None and coll_right_o is not None and coll_right_o > coll_left_o:
                    ax_overlay.axvspan(coll_left_o, coll_right_o, alpha=COLLECTION_WINDOW_ALPHA, color=COLLECTION_WINDOW_COLOR, zorder=-10)
                if min_rt_min is not None and max_rt_min is not None and max_rt_min > min_rt_min:
                    ax_overlay.axvspan(min_rt_min, max_rt_min, alpha=INTEGRATION_WINDOW_ALPHA, color='#00FF7F', zorder=-9)  # integration window (green)
                # Black only for peak drift buffer (inside drift, outside collection)
                if drift_min_rt is not None and drift_max_rt is not None and drift_max_rt > drift_min_rt:
                    drift_min_min_o = drift_min_rt / 60.0
                    drift_max_min_o = drift_max_rt / 60.0
                    z_drift_o = -8
                    left_buf_end_o = min(coll_left_o, drift_max_min_o) if coll_left_o is not None else drift_max_min_o
                    if left_buf_end_o > drift_min_min_o and (coll_left_o is None or coll_left_o > drift_min_min_o):
                        ax_overlay.axvspan(drift_min_min_o, left_buf_end_o, facecolor='black', alpha=1.0, zorder=z_drift_o)
                    right_buf_start_o = max(coll_right_o, drift_min_min_o) if coll_right_o is not None else drift_min_min_o
                    if drift_max_min_o > right_buf_start_o and (coll_right_o is None or coll_right_o < drift_max_min_o):
                        ax_overlay.axvspan(right_buf_start_o, drift_max_min_o, facecolor='black', alpha=1.0, zorder=z_drift_o)
                # All peptides overlay: plot extracted chromatogram lines (raw, no gap-breaking so lines always draw)
                for item in all_peptide_overlay_data:
                    rts_sec = item['rts']  # seconds from extraction
                    rts_min_ov = np.asarray(rts_sec, dtype=float) / 60.0
                    total_ints_ov = np.asarray(item['total_intensities'], dtype=float)
                    if len(rts_min_ov) < 2 or len(total_ints_ov) < 2:
                        continue
                    if use_total_area_for_overlay:
                        val_ov = all_peptide_scatter_data.get(item['peptide_key'], {}).get('total_area')
                    else:
                        val_ov = item.get('proxy_area')
                    if val_ov is not None and np.isfinite(val_ov) and val_ov >= 0 and norm_overlay is not None:
                        try:
                            c = cmap_overlay(norm_overlay(np.clip(float(val_ov), norm_overlay.vmin, norm_overlay.vmax)))
                        except Exception:
                            c = (0.6, 0.6, 0.6, 0.8)
                    else:
                        c = (0.6, 0.6, 0.6, 0.8)
                    ax_overlay.fill_between(rts_min_ov, 0, total_ints_ov, color=c, alpha=0.25, zorder=0)
                    ax_overlay.plot(rts_min_ov, total_ints_ov, color=c, alpha=0.7, linewidth=1.5, zorder=1)
                # This peptide: shaded fill then line on top with black outline (raw so line always draws)
                val_cur = (total_area if (total_area is not None and not (isinstance(total_area, float) and (np.isnan(total_area) or total_area < 0))) else None) if use_total_area_for_overlay else (float(np.sum(total_intensities)) if len(total_intensities) > 0 else None)
                if norm_overlay is not None and val_cur is not None:
                    try:
                        c_cur = cmap_overlay(norm_overlay(np.clip(float(val_cur), norm_overlay.vmin, norm_overlay.vmax)))
                    except Exception:
                        c_cur = (0.2, 0.4, 0.8, 0.9)
                else:
                    c_cur = (0.2, 0.4, 0.8, 0.9)
                rts_min_cur = np.asarray(rts_min, dtype=float)
                total_intensities_cur = np.asarray(total_intensities, dtype=float)
                if len(rts_min_cur) >= 2 and len(total_intensities_cur) >= 2:
                    ax_overlay.fill_between(rts_min_cur, 0, total_intensities_cur, color=c_cur, alpha=0.35, zorder=9)
                    from matplotlib.patheffects import withStroke
                    ax_overlay.plot(rts_min_cur, total_intensities_cur, color=c_cur, alpha=0.95, linewidth=2.5, zorder=11,
                                   path_effects=[withStroke(linewidth=7.0, foreground='black')])
                # Linear overlay y-axis based on peak of interest only (not dominated by larger peaks)
                apex_rt_min = (best_peak['apex_rt'] / 60.0) if best_peak and best_peak.get('apex_rt') is not None else None
                peak_win_min = min_rt_min if min_rt_min is not None else (apex_rt_min - 0.5 if apex_rt_min is not None else x_min)
                peak_win_max = max_rt_min if max_rt_min is not None else (apex_rt_min + 0.5 if apex_rt_min is not None else x_max)
                overlay_ymax_apex = 0.0
                if len(rts_min_cur) > 0 and len(total_intensities_cur) > 0:
                    in_peak = (rts_min_cur >= peak_win_min) & (rts_min_cur <= peak_win_max)
                    if np.any(in_peak):
                        overlay_ymax_apex = float(np.nanmax(np.asarray(total_intensities_cur, dtype=float)[in_peak]))
                if overlay_ymax_apex > 0:
                    ax_overlay.set_ylim([0, overlay_ymax_apex * 1.15])
                else:
                    ax_overlay.set_ylim([0, overlay_ymax_global * 1.1])
                ax_overlay.set_xlim(x_min, x_max)
                ax_overlay.set_ylabel('Intensity', fontsize=11, fontfamily='serif', color='0.85')
                ax_overlay.set_xlabel('MS1 Retention Time (min)', fontsize=11, fontfamily='serif', color='0.85')
                ax_overlay.tick_params(labelsize=10, colors='0.85')
                for spine in ax_overlay.spines.values():
                    spine.set_visible(True)
                    spine.set_color('0.6')
                for label in ax_overlay.get_xticklabels() + ax_overlay.get_yticklabels():
                    label.set_fontfamily('serif')
                    label.set_color('0.85')
                ax_overlay.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0), useMathText=True)
                ax_overlay.set_title('All peptides overlay (same time window; this peptide outlined in black)', fontsize=11, fontfamily='serif', color='0.9')
                # Total area colorbar so overlay colors are interpretable (same scale as overlay)
                if norm_overlay is not None and cmap_overlay is not None:
                    from matplotlib.cm import ScalarMappable
                    sm_overlay = ScalarMappable(norm=norm_overlay, cmap=cmap_overlay)
                    sm_overlay.set_array([])
                    cbar_overlay = fig.colorbar(sm_overlay, ax=ax_overlay, orientation='horizontal', pad=0.18, shrink=0.7, aspect=35, label='Total area')
                    cbar_overlay.ax.set_facecolor('black')
                    cbar_overlay.ax.tick_params(labelsize=9, colors='0.85')
                    cbar_overlay.ax.xaxis.label.set_color('0.85')
                    for label in cbar_overlay.ax.get_xticklabels():
                        label.set_fontfamily('serif')
                        label.set_color('0.85')
                # Log-scale copy of all-peptides overlay below (second subplot)
                _floor = 1e-10
                if coll_left_o is not None and coll_right_o is not None and coll_right_o > coll_left_o:
                    ax_overlay_log.axvspan(coll_left_o, coll_right_o, alpha=COLLECTION_WINDOW_ALPHA, color=COLLECTION_WINDOW_COLOR, zorder=-10)
                if min_rt_min is not None and max_rt_min is not None and max_rt_min > min_rt_min:
                    ax_overlay_log.axvspan(min_rt_min, max_rt_min, alpha=INTEGRATION_WINDOW_ALPHA, color='#00FF7F', zorder=-9)  # integration window (green)
                if drift_min_rt is not None and drift_max_rt is not None and drift_max_rt > drift_min_rt:
                    drift_min_min_o = drift_min_rt / 60.0
                    drift_max_min_o = drift_max_rt / 60.0
                    z_drift_o = -8
                    left_buf_end_o = min(coll_left_o, drift_max_min_o) if coll_left_o is not None else drift_max_min_o
                    if left_buf_end_o > drift_min_min_o and (coll_left_o is None or coll_left_o > drift_min_min_o):
                        ax_overlay_log.axvspan(drift_min_min_o, left_buf_end_o, facecolor='black', alpha=1.0, zorder=z_drift_o)
                    right_buf_start_o = max(coll_right_o, drift_min_min_o) if coll_right_o is not None else drift_min_min_o
                    if drift_max_min_o > right_buf_start_o and (coll_right_o is None or coll_right_o < drift_max_min_o):
                        ax_overlay_log.axvspan(right_buf_start_o, drift_max_min_o, facecolor='black', alpha=1.0, zorder=z_drift_o)
                INTENSITY_THRESH_LOG = 1e5
                for item in all_peptide_overlay_data:
                    rts_sec = item['rts']
                    rts_min_ov = np.asarray(rts_sec, dtype=float) / 60.0
                    total_ints_ov = np.asarray(item['total_intensities'], dtype=float)
                    total_ints_ov = np.where(np.isfinite(total_ints_ov), np.maximum(total_ints_ov, _floor), _floor)
                    if len(rts_min_ov) < 2 or len(total_ints_ov) < 2:
                        continue
                    if use_total_area_for_overlay:
                        val_ov = all_peptide_scatter_data.get(item['peptide_key'], {}).get('total_area')
                    else:
                        val_ov = item.get('proxy_area')
                    if val_ov is not None and np.isfinite(val_ov) and val_ov >= 0 and norm_overlay is not None:
                        try:
                            c = cmap_overlay(norm_overlay(np.clip(float(val_ov), norm_overlay.vmin, norm_overlay.vmax)))
                        except Exception:
                            c = (0.6, 0.6, 0.6, 0.8)
                    else:
                        c = (0.6, 0.6, 0.6, 0.8)
                    c_rgb = c[:3] if len(c) >= 3 else c
                    low_ov = np.where(total_ints_ov < INTENSITY_THRESH_LOG, total_ints_ov, _floor)
                    high_ov = np.where(total_ints_ov >= INTENSITY_THRESH_LOG, total_ints_ov, _floor)
                    ax_overlay_log.fill_between(rts_min_ov, _floor, low_ov, color=c_rgb, alpha=0.08, zorder=0)
                    ax_overlay_log.fill_between(rts_min_ov, _floor, high_ov, color=c_rgb, alpha=0.25, zorder=0)
                    low_plot_ov = np.where(total_ints_ov < INTENSITY_THRESH_LOG, total_ints_ov, np.nan)
                    high_plot_ov = np.where(total_ints_ov >= INTENSITY_THRESH_LOG, total_ints_ov, np.nan)
                    ax_overlay_log.plot(rts_min_ov, low_plot_ov, color=c_rgb, alpha=0.15, linewidth=1.0, zorder=1)
                    ax_overlay_log.plot(rts_min_ov, high_plot_ov, color=c_rgb, alpha=0.85, linewidth=1.5, zorder=1)
                total_ints_cur = np.asarray(total_intensities, dtype=float)
                total_ints_cur = np.where(np.isfinite(total_ints_cur), np.maximum(total_ints_cur, _floor), _floor)
                rts_min_cur = np.asarray(rts_min, dtype=float)
                if len(rts_min_cur) >= 2 and len(total_ints_cur) >= 2:
                    low_cur = np.where(total_ints_cur < INTENSITY_THRESH_LOG, total_ints_cur, _floor)
                    high_cur = np.where(total_ints_cur >= INTENSITY_THRESH_LOG, total_ints_cur, _floor)
                    ax_overlay_log.fill_between(rts_min_cur, _floor, low_cur, color=c_cur, alpha=0.12, zorder=9)
                    ax_overlay_log.fill_between(rts_min_cur, _floor, high_cur, color=c_cur, alpha=0.35, zorder=9)
                    low_plot_cur = np.where(total_ints_cur < INTENSITY_THRESH_LOG, total_ints_cur, np.nan)
                    high_plot_cur = np.where(total_ints_cur >= INTENSITY_THRESH_LOG, total_ints_cur, np.nan)
                    ax_overlay_log.plot(rts_min_cur, low_plot_cur, color=c_cur, alpha=0.2, linewidth=1.0, zorder=11,
                                       path_effects=[withStroke(linewidth=4.0, foreground='black')])
                    ax_overlay_log.plot(rts_min_cur, high_plot_cur, color=c_cur, alpha=0.95, linewidth=2.5, zorder=11,
                                       path_effects=[withStroke(linewidth=7.0, foreground='black')])
                ax_overlay_log.set_xlim(x_min, x_max)
                y_log_max = float(np.max(total_ints_cur)) if len(total_ints_cur) > 0 else _floor
                for item in all_peptide_overlay_data:
                    arr = np.asarray(item['total_intensities'], dtype=float)
                    arr = np.where(np.isfinite(arr), np.maximum(arr, _floor), _floor)
                    if len(arr) > 0:
                        y_log_max = max(y_log_max, float(np.max(arr)))
                if not np.isfinite(y_log_max) or y_log_max <= 0:
                    y_log_max = _floor
                y_top = max(y_log_max * 2.0, _floor * 10)
                if not np.isfinite(y_top) or y_top <= _floor:
                    y_top = _floor * 10
                ax_overlay_log.set_ylim(_floor, y_top)
                ax_overlay_log.set_yscale('log')
                ax_overlay_log.set_ylabel('Intensity (log)', fontsize=11, fontfamily='serif', color='0.85')
                ax_overlay_log.set_xlabel('MS1 Retention Time (min)', fontsize=11, fontfamily='serif', color='0.85')
                ax_overlay_log.tick_params(labelsize=10, colors='0.85')
                for spine in ax_overlay_log.spines.values():
                    spine.set_visible(True)
                    spine.set_color('0.6')
                for label in ax_overlay_log.get_xticklabels() + ax_overlay_log.get_yticklabels():
                    label.set_fontfamily('serif')
                    label.set_color('0.85')
                ax_overlay_log.set_title('All peptides overlay (log scale; this peptide outlined in black)', fontsize=11, fontfamily='serif', color='0.9')
        
                # Score scatter plots: small dots for others, peptide of interest on top with black outline; square aspect
                def _safe_log10(x, floor=1e-10):
                    """Return log10(x) with floor for non-positive values."""
                    if x is None or (isinstance(x, float) and (np.isnan(x) or x <= 0)):
                        return np.log10(floor)
                    try:
                        v = float(x)
                        return np.log10(max(v, floor))
                    except (TypeError, ValueError):
                        return np.log10(floor)
                def _draw_score_scatter(ax_scatter, x_other, y_other, c_other, x_cur, y_cur, c_cur, x_thresh, y_thresh, xlabel, ylabel, title, logx=False, logy=False, y_lims=None, gray_y_above_zero=False, gray_y_above_value=None, x_lims=None, y_lims_data=None, highlight_black_only=False):
                    """Plot others (zorder=2) and peptide of interest (zorder=5), all with black outline.
                    If highlight_black_only: peptide of interest is same size as others, black fill. Else: larger with black outline.
                    y_lims: optional (ymin, ymax). gray_y_above_value: in data coords. x_lims, y_lims_data: (min,max) for data-based axes."""
                    if logx:
                        x_other = [_safe_log10(v) for v in x_other]
                        if x_cur is not None:
                            x_cur = _safe_log10(x_cur)
                        if x_thresh:
                            x_thresh = {k: _safe_log10(v) if v is not None else None for k, v in x_thresh.items()}
                    if logy:
                        y_other = [_safe_log10(v) for v in y_other]
                        if y_cur is not None:
                            y_cur = _safe_log10(y_cur)
                        if y_thresh:
                            y_thresh = {k: _safe_log10(v) if v is not None else None for k, v in y_thresh.items()}
                    if len(x_other) > 0:
                        ax_scatter.scatter(x_other, y_other, c=c_other, s=18, alpha=0.8, edgecolors='black', linewidths=0.5, zorder=2)
                    if x_cur is not None and y_cur is not None:
                        if highlight_black_only:
                            ax_scatter.scatter([x_cur], [y_cur], c='black', s=18, alpha=0.95, edgecolors='black', linewidths=0.5, zorder=5)
                        else:
                            ax_scatter.scatter([x_cur], [y_cur], c=[c_cur], s=100, alpha=0.95, edgecolors='black', linewidths=2.5, zorder=5)
                    _xt = x_thresh or {}
                    _yt = y_thresh or {}
                    # One line per threshold: use primary (green) only
                    _xv = _xt.get('green')
                    if _xv is not None:
                        ax_scatter.axvline(x=_xv, color='green', linestyle='--', linewidth=1.2, alpha=0.9, zorder=0)
                    _yv = _yt.get('green')
                    if _yv is not None:
                        ax_scatter.axhline(y=_yv, color='green', linestyle='--', linewidth=1.2, alpha=0.9, zorder=0)
                    if logx:
                        ax_scatter.set_xscale('log')
                    if logy:
                        ax_scatter.set_yscale('log')
                    ax_scatter.set_xlabel(xlabel, fontsize=9, fontfamily='serif')
                    ax_scatter.set_ylabel(ylabel, fontsize=9, fontfamily='serif')
                    ax_scatter.set_title(title, fontsize=9, fontfamily='serif')
                    ax_scatter.tick_params(labelsize=8)
                    for spine in ax_scatter.spines.values():
                        spine.set_visible(True)
                    for label in ax_scatter.get_xticklabels() + ax_scatter.get_yticklabels():
                        label.set_fontfamily('serif')
                    ax_scatter.set_aspect('equal', adjustable='datalim')
                    if not logx and not logy:
                        if y_lims is not None:
                            ax_scatter.set_xlim(0, 1)
                            ax_scatter.set_ylim(y_lims[0], y_lims[1])
                        elif x_lims is not None and y_lims_data is not None:
                            ax_scatter.set_xlim(x_lims[0], x_lims[1])
                            ax_scatter.set_ylim(y_lims_data[0], y_lims_data[1])
                        else:
                            ax_scatter.set_xlim(0, 1)
                            ax_scatter.set_ylim(0, 1)
        
                # Build point lists: one dot per peptide (all unique_peptide_keys) for coelution, shape, ratio (no peak score)
                _sentinel = -0.05  # off-range for coelution/shape/ratio so unprocessed peptides still show a dot
                x1_other, y1_other, c1_other = [], [], []
                # For "Shape correlation vs ratio stability" plot: one point per other peptide (include all; use sentinel when ratio missing)
                shape_corr_other_peptides = []
                ratio_stability_other_peptides = []
                colors_other_peptides_shape_ratio = []
                def _ratio_stability_computed(rat):
                    """True if ratio stability was computed (not None/NaN/0). 0 means insufficient data -> N/A."""
                    if rat is None or (isinstance(rat, float) and np.isnan(rat)):
                        return False
                    if isinstance(rat, (int, float)) and rat == 0:
                        return False
                    return True
                for key in unique_peptide_keys:
                    if key == peptide_key:
                        continue  # current peptide drawn separately with highlight
                    d = all_peptide_scatter_data.get(key, {})
                    csc = d.get('coelution_score')
                    shp = d.get('shape_corr')
                    rat = d.get('ratio_stability')
                    ta = d.get('total_area')
                    csc = float(csc) if csc is not None and not (isinstance(csc, float) and np.isnan(csc)) else _sentinel
                    shp = float(shp) if shp is not None and not (isinstance(shp, float) and np.isnan(shp)) else _sentinel
                    rat_val = float(rat) if _ratio_stability_computed(rat) else _sentinel
                    x1_other.append(csc)
                    y1_other.append(shp)
                    shape_corr_other_peptides.append(shp)
                    ratio_stability_other_peptides.append(rat_val)
                    if norm_overlay and ta is not None and not (isinstance(ta, float) and (np.isnan(ta) or ta < 0)):
                        try:
                            col = cmap_overlay(norm_overlay(np.clip(float(ta), norm_overlay.vmin, norm_overlay.vmax)))
                        except Exception:
                            col = (0.6, 0.6, 0.6, 0.8)
                    else:
                        col = (0.6, 0.6, 0.6, 0.8)
                    c1_other.append(col)
                    colors_other_peptides_shape_ratio.append(col)
                c_cur = (0.2, 0.4, 0.8, 0.9)
                ta_cur = total_area if (total_area is not None and not (isinstance(total_area, float) and (np.isnan(total_area) or total_area < 0))) else None
                if norm_overlay and ta_cur is not None:
                    try:
                        c_cur = cmap_overlay(norm_overlay(np.clip(float(ta_cur), norm_overlay.vmin, norm_overlay.vmax)))
                    except Exception:
                        pass
                x1_cur = coelution_score if coelution_score is not None and not (isinstance(coelution_score, float) and np.isnan(coelution_score)) else _sentinel
                y1_cur = shape_corr if shape_corr is not None and not (isinstance(shape_corr, float) and np.isnan(shape_corr)) else _sentinel
                shape_cur = y1_cur
                ratio_cur = float(ratio_stability) if _ratio_stability_computed(ratio_stability) else _sentinel
                thresh_shape = {'green': 0.85, 'yellow': 0.5, 'red': 0.35}
                thresh_q_linear = {'green': 0.01, 'yellow': 0.05, 'red': 0.1}  # Q-value thresholds (linear): good ≤0.01, fair ≤0.05
        
                # Peptide score scatter plots: E vs PEP, Q vs coelution, shape vs ratio
                def _safe_float(v, default_for_nan=0.0):
                    """Convert to float; use default_for_nan for NaN/None/invalid (so we plot all peptides)."""
                    if v is None or (isinstance(v, float) and np.isnan(v)):
                        return default_for_nan
                    try:
                        f = float(v)
                        return f if np.isfinite(f) else default_for_nan
                    except (TypeError, ValueError):
                        return default_for_nan
                def _safe_log10(v, sentinel=-12.0):
                    """Convert to log10 for axis; use sentinel for invalid/zero/negative."""
                    f = _safe_float(v, None)
                    if f is None or f <= 0 or not np.isfinite(f):
                        return sentinel
                    return float(np.log10(max(f, 1e-15)))
                def _draw_psm_scatter(ax_psm, x_all, y_all, c_all, x_cur_list, y_cur_list, xlabel, ylabel, title, logx=False, logy=False, logc=False, draw_q_reference=False, use_log10_coords=False, ref_lines=None, colors_all=None, cur_psm_fails=None):
                    """Plot all PSMs color-coded; current peptide's PSMs in black on top.
                    If cur_psm_fails[i] is True, draw a bright red X over that PSM dot.
                    If colors_all is provided (list of RGBA), use total_area coloring to match other scatter plots.
                    Else use c_all with viridis. use_log10_coords=True: plot log10(values). ref_lines for threshold lines."""
                    n_cur = 0
                    x_sentinel = 1e-15 if logx else 0.0
                    y_sentinel = 1e-15 if logy else 0.0
                    c_sentinel = 1e-15 if logc else 0.0
                    def _safe_for_log(v, sentinel):
                        f = _safe_float(v, sentinel)
                        if logx or logy or logc:
                            if f is None or f <= 0 or not np.isfinite(f):
                                return sentinel
                            return max(f, sentinel)
                        return f if f is not None and np.isfinite(f) else sentinel
                    if use_log10_coords:
                        x_plot = np.asarray([_safe_log10(v) for v in x_all])
                        y_plot = np.asarray([_safe_log10(v) for v in y_all])
                        c_plot = np.asarray([_safe_log10(v, sentinel=-12.0) for v in c_all] if logc else [_safe_float(v, 0.0) for v in c_all])
                    else:
                        x_plot = np.asarray([_safe_for_log(v, x_sentinel) for v in x_all])
                        y_plot = np.asarray([_safe_for_log(v, y_sentinel) for v in y_all])
                        c_plot = np.asarray([_safe_for_log(v, c_sentinel) for v in c_all] if logc else [_safe_float(v, 0.0) for v in c_all])
                    n_plot = min(len(x_plot), len(y_plot), len(c_plot) if colors_all is None else len(colors_all))
                    if n_plot > 0:
                        x_plot = x_plot[:n_plot].astype(float)
                        y_plot = y_plot[:n_plot].astype(float)
                        x_plot[~np.isfinite(x_plot)] = (-12.0 if use_log10_coords else x_sentinel)
                        y_plot[~np.isfinite(y_plot)] = (-12.0 if use_log10_coords else y_sentinel)
                        if colors_all is not None and len(colors_all) >= n_plot:
                            # Color by total_area (same as coelution/shape and ratio/peak scatter)
                            c_plot = [colors_all[i] for i in range(n_plot)]
                            ax_psm.scatter(x_plot, y_plot, c=c_plot, alpha=0.5, s=20, edgecolors='black', linewidths=0.5, zorder=2)
                        else:
                            c_plot = np.asarray(c_plot[:n_plot], dtype=float)
                            c_finite = np.isfinite(c_plot)
                            if not np.any(c_finite):
                                c_plot[:] = 0.0
                            else:
                                c_plot[~c_finite] = np.nanmin(c_plot[c_finite])
                            ax_psm.scatter(x_plot, y_plot, c=c_plot, cmap='viridis', alpha=0.5, s=20, edgecolors='black', linewidths=0.5, zorder=2)
                    # Current peptide's PSMs in black, plotted last (on top)
                    cx_plot = np.array([])
                    cy_plot = np.array([])
                    if x_cur_list and y_cur_list:
                        if use_log10_coords:
                            cx_plot = np.asarray([_safe_log10(v) for v in x_cur_list], dtype=float)
                            cy_plot = np.asarray([_safe_log10(v) for v in y_cur_list], dtype=float)
                        else:
                            cx_plot = np.asarray([_safe_for_log(v, x_sentinel) for v in x_cur_list], dtype=float)
                            cy_plot = np.asarray([_safe_for_log(v, y_sentinel) for v in y_cur_list], dtype=float)
                        n_cur = min(len(cx_plot), len(cy_plot))
                        if n_cur > 0:
                            cx_plot, cy_plot = cx_plot[:n_cur], cy_plot[:n_cur]
                            if use_log10_coords:
                                cx_plot[~np.isfinite(cx_plot)] = -12.0
                                cy_plot[~np.isfinite(cy_plot)] = -12.0
                            else:
                                cx_plot[~np.isfinite(cx_plot)] = x_sentinel
                                cy_plot[~np.isfinite(cy_plot)] = y_sentinel
                            # Draw red X for failing PSMs first (zorder=4) so black dots (zorder=5) are on top and good ones don't look crossed out
                            if cur_psm_fails is not None and len(cur_psm_fails) >= n_cur:
                                if use_log10_coords:
                                    x_size, y_size = 0.5, 0.5
                                else:
                                    x_range = float(np.nanmax(x_plot)) - float(np.nanmin(x_plot)) if len(x_plot) > 0 else 1.0
                                    y_range = float(np.nanmax(y_plot)) - float(np.nanmin(y_plot)) if len(y_plot) > 0 else 1.0
                                    x_size = 0.02 * max(x_range, 1e-12)
                                    y_size = 0.02 * max(y_range, 1e-12)
                                for i in range(n_cur):
                                    if cur_psm_fails[i]:
                                        x, y = float(cx_plot[i]), float(cy_plot[i])
                                        ax_psm.plot([x - x_size, x + x_size], [y - y_size, y + y_size], color='#E60000', linewidth=2.5, zorder=4, solid_capstyle='round')
                                        ax_psm.plot([x - x_size, x + x_size], [y + y_size, y - y_size], color='#E60000', linewidth=2.5, zorder=4, solid_capstyle='round')
                            ax_psm.scatter(cx_plot, cy_plot, c='black', s=40, alpha=0.9, edgecolors='black', linewidths=0.5, zorder=5)
                    if use_log10_coords:
                        ax_psm.set_xscale('linear')
                        ax_psm.set_yscale('linear')
                        x_default_lo, x_default_hi = (-12, 0)
                        y_default_lo, y_default_hi = (-12, 0)
                    else:
                        if logx:
                            ax_psm.set_xscale('log')
                        if logy:
                            ax_psm.set_yscale('log')
                        x_default_lo, x_default_hi = (1e-12, 1.0) if logx else (0, 1)
                        y_default_lo, y_default_hi = (1e-12, 1.0) if logy else (0, 1)
                    x_min = np.nanmin(x_plot) if len(x_plot) > 0 else x_default_lo
                    x_max = np.nanmax(x_plot) if len(x_plot) > 0 else x_default_hi
                    y_min = np.nanmin(y_plot) if len(y_plot) > 0 else y_default_lo
                    y_max = np.nanmax(y_plot) if len(y_plot) > 0 else y_default_hi
                    if n_cur > 0 and len(cx_plot) > 0 and len(cy_plot) > 0:
                        x_min = min(x_min, float(np.nanmin(cx_plot)))
                        x_max = max(x_max, float(np.nanmax(cx_plot)))
                        y_min = min(y_min, float(np.nanmin(cy_plot)))
                        y_max = max(y_max, float(np.nanmax(cy_plot)))
                    if not np.isfinite(x_min) or not np.isfinite(x_max):
                        x_min, x_max = x_default_lo, x_default_hi
                    if not np.isfinite(y_min) or not np.isfinite(y_max):
                        y_min, y_max = y_default_lo, y_default_hi
                    if use_log10_coords:
                        # E-value vs PEP (log10): both axes from data min/max + padding so all points are visible
                        x_range = x_max - x_min
                        y_range = y_max - y_min
                        pad_x = max(x_range * 0.05, 0.5) if x_range > 0 else 0.5
                        pad_y = max(y_range * 0.05, 0.5) if y_range > 0 else 0.5
                        x_lo = x_min - pad_x
                        x_hi = x_max + pad_x
                        y_lo = y_min - pad_y
                        y_hi = y_max + pad_y
                        # Extend PEP (y) high end so "bad" region (PEP>1) remains visible when all data is good
                        y_hi = max(y_hi, 0.5)
                        ax_psm.set_xlim(x_lo, x_hi)
                        ax_psm.set_ylim(y_lo, y_hi)
                    else:
                        if x_max <= x_min:
                            x_min, x_max = max(1e-15, x_min / 10), x_min * 10
                        if y_max <= y_min:
                            y_min, y_max = max(1e-15, y_min / 10), y_min * 10
                        pad_x = (np.log10(x_max) - np.log10(x_min)) * 0.05 if logx and x_min > 0 and x_max > 0 else 0.01
                        pad_y = (np.log10(y_max) - np.log10(y_min)) * 0.05 if logy and y_min > 0 and y_max > 0 else 0.01
                        if logx and x_min > 0 and x_max > 0:
                            ax_psm.set_xlim(x_min / (10 ** pad_x), x_max * (10 ** pad_x))
                        else:
                            ax_psm.set_xlim(x_min - pad_x, x_max + pad_x)
                        if logy and y_min > 0 and y_max > 0:
                            ax_psm.set_ylim(y_min / (10 ** pad_y), y_max * (10 ** pad_y))
                        else:
                            ax_psm.set_ylim(y_min - pad_y, y_max + pad_y)
                    ax_psm.set_xlabel(xlabel, fontsize=9, fontfamily='serif')
                    ax_psm.set_ylabel(ylabel, fontsize=9, fontfamily='serif')
                    ax_psm.set_title(title, fontsize=9, fontfamily='serif')
                    ax_psm.tick_params(labelsize=8)
                    for spine in ax_psm.spines.values():
                        spine.set_visible(True)
                    for label in ax_psm.get_xticklabels() + ax_psm.get_yticklabels():
                        label.set_fontfamily('serif')
                    ax_psm.set_aspect('equal', adjustable='datalim')
                    if use_log10_coords and ref_lines:
                        x_lo, x_hi = ax_psm.get_xlim()
                        y_lo, y_hi = ax_psm.get_ylim()
                        v_list = ref_lines.get('v', [])
                        h_list = ref_lines.get('h', [])
                        if len(v_list) > 0:
                            x_pos, label = v_list[-1]
                            ax_psm.axvline(x_pos, color='green', linestyle='--', linewidth=1.2, alpha=0.9, zorder=1)
                            ax_psm.text(x_pos, y_max, ' ' + label, fontsize=6, fontfamily='serif', va='bottom', color='green')
                        if len(h_list) > 0:
                            y_pos, label = h_list[-1]
                            ax_psm.axhline(y_pos, color='green', linestyle='--', linewidth=1.2, alpha=0.9, zorder=1)
                            ax_psm.text(x_min, y_pos, ' ' + label, fontsize=6, fontfamily='serif', va='bottom', color='green')
                        if ref_lines.get('text'):
                            ax_psm.text(0.02, 0.98, ref_lines['text'], transform=ax_psm.transAxes, fontsize=6, fontfamily='serif',
                                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85))
                    elif draw_q_reference and logy and not use_log10_coords:
                        ax_psm.axhline(0.01, color='green', linestyle='--', linewidth=1, alpha=0.8, zorder=1, label='good: q ≤ 0.01 (log₁₀(q) ≤ −2)')
                        ax_psm.axhline(0.001, color='darkgreen', linestyle='--', linewidth=1, alpha=0.8, zorder=1, label='excellent: q ≤ 0.001 (log₁₀(q) ≤ −3)')
        
                # Plot summed spectrum in combined figure spectrum subplot (only if not rejected)
                # Rejected peptides will be plotted in separate combined figures
                spec_mzs_windowed_combined = None
                spec_ints_windowed_combined = None
                if not is_rejected and apex_rt and min_rt and max_rt:
                    if isotope_mzs and len(isotope_mzs) > 0:
                        isotope_mz_min = min(isotope_mzs)
                        isotope_mz_max = max(isotope_mzs)
                        mz_buffer = 0.5
                        mz_min_window = isotope_mz_min - mz_buffer
                        mz_max_window = isotope_mz_max + mz_buffer
                    else:
                        mz_buffer = 2.0
                        mz_min_window = precursor_mz - mz_buffer
                        mz_max_window = precursor_mz + mz_buffer
                    try:
                        spectrum_rt_min_combined = spectrum_rt_min
                        spectrum_rt_max_combined = spectrum_rt_max
                    except NameError:
                        spectrum_rt_min_combined = min_rt
                        spectrum_rt_max_combined = max_rt
                    spec_mzs_combined, spec_ints_combined = get_averaged_ms1_spectrum_in_window(
                        raw_file, spectrum_rt_min_combined, spectrum_rt_max_combined, 
                        mz_min=mz_min_window, mz_max=mz_max_window, 
                        ppm_tolerance=7.0
                    )
                    if spec_mzs_combined is not None and len(spec_mzs_combined) > 0:
                        mask_combined = (spec_mzs_combined >= mz_min_window) & (spec_mzs_combined <= mz_max_window)
                        spec_mzs_windowed_combined = spec_mzs_combined[mask_combined]
                        spec_ints_windowed_combined = spec_ints_combined[mask_combined]
                
                        # Count matches in combined figure (should match individual figure since we use same window)
                        matched_iso_indices_combined = []
                        if len(spec_mzs_windowed_combined) > 0 and isotope_mzs and len(isotope_mzs) > 0:
                            for iso_idx, iso_mz in enumerate(isotope_mzs):
                                iso_tolerance = iso_mz * 5e-6  # 5 ppm tolerance
                                iso_mask = (spec_mzs_windowed_combined >= iso_mz - iso_tolerance) & (spec_mzs_windowed_combined <= iso_mz + iso_tolerance)
                                if np.any(iso_mask):
                                    iso_intensities = spec_ints_windowed_combined[iso_mask]
                                    max_iso_intensity = np.max(iso_intensities) if len(iso_intensities) > 0 else 0
                                    # Check if signal is significant (>1% of max spectrum intensity)
                                    max_spec_intensity = np.max(spec_ints_windowed_combined) if len(spec_ints_windowed_combined) > 0 else 0
                                    if max_spec_intensity > 0 and max_iso_intensity > max_spec_intensity * 0.01:
                                        matched_iso_indices_combined.append(iso_idx)
                            n_matches_combined = len(matched_iso_indices_combined)
                        else:
                            # No spectrum data or isotope m/z values available
                            n_matches_combined = 0
                            matched_iso_indices_combined = []
                
                        # Filter: Require M0, M+1, and M+2 in the MS1 spectrum (for combined figure; same as acceptance)
                        combined_passed = all(i in matched_iso_indices_combined for i in REQUIRED_ISOTOPE_INDICES)
                
                        if not combined_passed:
                            # Clear and hide this plot in combined figure (chromatogram was already plotted, so clear it)
                            axes_combined_chrom_list[local_idx].clear()
                            axes_combined_chrom_list[local_idx].axis('off')
                            axes_combined_spec_list[local_idx].clear()
                            axes_combined_spec_list[local_idx].axis('off')
                            drew_combined_spec = False  # Don't increment; next picked peptide will reuse this slot
                            # Skip plotting spectrum for combined figure
                            spec_mzs_windowed_combined = None
                            spec_ints_windowed_combined = None
                
                    # Note: Overlap filtering for combined figure happens after overlap detection (see below)
                
                # Only proceed with plotting if we have spectrum data and sufficient matches
                if spec_mzs_windowed_combined is not None and len(spec_mzs_windowed_combined) > 0:
                    # Plot spectrum in combined figure (smaller, simpler)
                    markerline_combined, stemlines_combined, baseline_combined = ax_combined_spec.stem(
                        spec_mzs_windowed_combined, spec_ints_windowed_combined, 
                        basefmt=' ', linefmt='#666666', markerfmt=' ')
                    plt.setp(stemlines_combined, linewidth=0.6)
                    plt.setp(markerline_combined, markersize=2)
                
                    # Cross-reference: Check if other peptides have overlapping isotopic peaks or RT values WITHIN THE WINDOW (same as individual figure)
                    # Use full_crossref (all non-decoy peptides from CSV) for comprehensive interference detection
                    overlapping_peptides_iso_combined = []
                    overlapping_peptides_rt_combined = []
                    overlapping_iso_mzs_combined = set()  # Set of current peptide's iso_mz values that overlap with other peptides
                    other_peptides_iso_mzs_combined = {}  # Dict mapping OTHER peptides' isotopic m/z values to peptide sequence (for gray dashed lines with labels)
                
                    for other_key, other_data in full_crossref.items():
                        if other_key == peptide_key:
                            continue
                    
                        # Get peptide sequence and charge from group data for labeling
                        other_group = df[df['peptide_key'] == other_key] if 'peptide_key' in df.columns else None
                        other_peptide_seq = None
                        other_charge = None
                        if other_group is not None and len(other_group) > 0:
                            other_peptide_seq = other_group['plain_peptide'].iloc[0] if 'plain_peptide' in other_group.columns else None
                            other_charge = other_group['charge'].iloc[0] if 'charge' in other_group.columns else None
                        if other_peptide_seq is None:
                            # Fallback: try to extract from key (less reliable)
                            parts = other_key.split('_')
                            if len(parts) >= 4:
                                # Assume peptide is the second part (after mz), charge is second-to-last
                                other_peptide_seq = parts[1] if len(parts) > 1 else other_key[:20]
                                try:
                                    other_charge = int(parts[-2]) if len(parts) >= 3 else None
                                except:
                                    other_charge = None
                    
                        # Check isotopic peak overlap (within 5 ppm tolerance) AND within m/z window
                        for iso_mz in isotope_mzs:
                            # Only check if this iso_mz is within the spectrum window
                            if mz_min_window <= iso_mz <= mz_max_window:
                                iso_tolerance = iso_mz * 5e-6  # 5 ppm tolerance
                                for other_iso_mz in other_data['isotope_mzs']:
                                    # Check if other peptide's isotopic peak is within window AND overlaps
                                    if mz_min_window <= other_iso_mz <= mz_max_window:
                                        if abs(iso_mz - other_iso_mz) <= iso_tolerance:
                                            overlapping_iso_mzs_combined.add(iso_mz)
                                            if other_key not in overlapping_peptides_iso_combined:
                                                overlapping_peptides_iso_combined.append(other_key)
                                            break
                    
                        # RT overlap: track for combined figure
                        other_rt_anchor = other_data.get('rt_anchor', None)
                        other_rt_min, other_rt_max = other_data['rt_range']
                        if other_rt_anchor is not None:
                            if min_rt <= other_rt_anchor <= max_rt:
                                if other_key not in overlapping_peptides_rt_combined:
                                    overlapping_peptides_rt_combined.append(other_key)
                        if not (other_rt_max < min_rt or other_rt_min > max_rt):
                            if other_key not in overlapping_peptides_rt_combined:
                                overlapping_peptides_rt_combined.append(other_key)
                    
                        # Collect other peptides' isotopic m/z in window (gray dashed = interfering positions)
                        # Store mapping of m/z to (peptide sequence, charge) tuple for labeling
                        for other_iso_mz in other_data['isotope_mzs']:
                            if mz_min_window <= other_iso_mz <= mz_max_window:
                                if other_iso_mz not in other_peptides_iso_mzs_combined:
                                    other_peptides_iso_mzs_combined[other_iso_mz] = (
                                        other_peptide_seq if other_peptide_seq else other_key[:20],
                                        other_charge if other_charge else other_data.get('charge', None)
                                    )
                
                    # Note: We no longer filter out peptides with overlapping isotopic peaks
                    # Instead, we visualize them with red lines in the spectrum
                    # (This check was removed to allow visualization of interference)
                
                    # Highlight isotopes (simplified for combined figure) - thinner, more saturated, darker colors
                    iso_colors = ['#0033CC', '#6600CC', '#CC0099', '#CC0066']  # More saturated and darker
                
                    # Collect maximum intensities of matched isotopic peaks for y-axis scaling
                    matched_iso_intensities_combined = []
                
                    # First pass: collect matched intensities to compute y_max
                    for iso_idx, iso_mz in enumerate(isotope_mzs):
                        if iso_idx < len(iso_colors):
                            iso_tolerance = iso_mz * 5e-6  # 5 ppm tolerance
                            iso_mask = (spec_mzs_windowed_combined >= iso_mz - iso_tolerance) & (spec_mzs_windowed_combined <= iso_mz + iso_tolerance)
                            if np.any(iso_mask):
                                iso_intensities = spec_ints_windowed_combined[iso_mask]
                                max_iso_intensity = np.max(iso_intensities) if len(iso_intensities) > 0 else np.max(spec_ints_windowed_combined) if len(spec_ints_windowed_combined) > 0 else 0
                                matched_iso_intensities_combined.append(max_iso_intensity)
                
                    # Compute y-axis max for drawing off-target lines
                    if len(matched_iso_intensities_combined) > 0:
                        max_matched_intensity_combined = np.max(matched_iso_intensities_combined)
                        y_max_spec_combined = max_matched_intensity_combined * 1.2
                    else:
                        y_max_spec_combined = np.max(spec_ints_windowed_combined) * 1.1 if len(spec_ints_windowed_combined) > 0 else 1.0
                
                    # Draw gray dashed lines only where there is signal within 5 ppm of the expected (other) isotopic m/z
                    max_spec_intensity_combined = np.max(spec_ints_windowed_combined) if len(spec_ints_windowed_combined) > 0 else 0
                    for other_iso_mz, (other_peptide_seq, other_charge_val) in other_peptides_iso_mzs_combined.items():
                        other_tol = other_iso_mz * 5e-6  # 5 ppm
                        other_mask = (spec_mzs_windowed_combined >= other_iso_mz - other_tol) & (spec_mzs_windowed_combined <= other_iso_mz + other_tol)
                        if np.any(other_mask):
                            other_ints = spec_ints_windowed_combined[other_mask]
                            if max_spec_intensity_combined > 0 and np.max(other_ints) > max_spec_intensity_combined * 0.01:
                                ax_combined_spec.vlines(other_iso_mz, 0, y_max_spec_combined, colors='#808080',
                                                     linestyles='--', linewidths=2.5, alpha=0.85, zorder=1)
                                # Add label with peptide sequence and charge (truncate sequence if too long, smaller font for combined figure)
                                seq_part = other_peptide_seq[:8] + '...' if len(other_peptide_seq) > 8 else other_peptide_seq
                                charge_part = f" (+{other_charge_val})" if other_charge_val else ""
                                label_text = seq_part + charge_part
                                ax_combined_spec.text(other_iso_mz, y_max_spec_combined * 0.95, label_text,
                                                     fontsize=6, ha='center', va='bottom', fontfamily='serif',
                                                     bbox=dict(boxstyle='round,pad=0.2', facecolor='lightgray', alpha=0.7, edgecolor='gray', linewidth=0.5),
                                                     zorder=3)
                
                    # Now draw colored lines for current peptide's isotopes (on top of gray lines)
                    for iso_idx, iso_mz in enumerate(isotope_mzs):
                        if iso_idx < len(iso_colors):
                            iso_tolerance = iso_mz * 5e-6  # 5 ppm tolerance
                            iso_mask = (spec_mzs_windowed_combined >= iso_mz - iso_tolerance) & (spec_mzs_windowed_combined <= iso_mz + iso_tolerance)
                        
                            if np.any(iso_mask):
                                iso_intensities = spec_ints_windowed_combined[iso_mask]
                                max_iso_intensity = np.max(iso_intensities) if len(iso_intensities) > 0 else np.max(spec_ints_windowed_combined) if len(spec_ints_windowed_combined) > 0 else 0
                                # Always use normal isotope colors; interference is indicated by black dashed lines
                                line_color = iso_colors[iso_idx]
                                line_width = 3.0  # Thicker to reflect ±5 ppm tolerance window
                                ax_combined_spec.vlines(iso_mz, 0, max_iso_intensity, colors=line_color, 
                                                     linestyles='--', linewidths=line_width, alpha=1.0, zorder=2)
                
                    # Set y-axis limits
                    y_min_spec_combined = 0
                    ax_combined_spec.set_ylim([y_min_spec_combined, y_max_spec_combined])
                
                    # Format combined spectrum subplot
                    ax_combined_spec.set_xlabel('m/z', fontsize=5, fontfamily='serif', color='0.85')
                    ax_combined_spec.set_ylabel('Intensity', fontsize=5, fontfamily='serif', color='0.85')
                    ax_combined_spec.tick_params(labelsize=4, colors='0.85')
                    for label in ax_combined_spec.get_xticklabels() + ax_combined_spec.get_yticklabels():
                        label.set_fontfamily('serif')
                        label.set_color('0.85')
                    for spine in ax_combined_spec.spines.values():
                        spine.set_color('0.6')
                    ax_combined_spec.grid(False)  # Remove grid bars
                    ax_combined_spec.set_xlim([mz_min_window, mz_max_window])
                
                    # Add label showing isotope matching tolerance (smaller for combined figure)
                    x_lim_spec_combined = ax_combined_spec.get_xlim()
                    y_lim_spec_combined = ax_combined_spec.get_ylim()
                    tolerance_label_x_combined = x_lim_spec_combined[0] + (x_lim_spec_combined[1] - x_lim_spec_combined[0]) * 0.02
                    tolerance_label_y_combined = y_lim_spec_combined[0] + (y_lim_spec_combined[1] - y_lim_spec_combined[0]) * 0.95
                    ax_combined_spec.text(tolerance_label_x_combined, tolerance_label_y_combined, '±5 ppm', 
                                        fontsize=6, ha='left', va='top', fontfamily='serif',
                                        bbox=dict(boxstyle='round,pad=0.25', facecolor='lightblue', alpha=0.8, edgecolor='black', linewidth=0.5),
                                        zorder=25, clip_on=True)
                
                    drew_combined_spec = True
                else:
                    if not is_rejected:
                        # Try fallback for combined figure (only if not rejected)
                        # Ensure m/z window is defined (should be from individual figure code, but check for safety)
                        if mz_min_window is None or mz_max_window is None:
                            if isotope_mzs and len(isotope_mzs) > 0:
                                isotope_mz_min = min(isotope_mzs)
                                isotope_mz_max = max(isotope_mzs)
                                mz_buffer = 0.5
                                mz_min_window = isotope_mz_min - mz_buffer
                                mz_max_window = isotope_mz_max + mz_buffer
                            elif precursor_mz and precursor_mz > 0:
                                mz_buffer = 2.0
                                mz_min_window = precursor_mz - mz_buffer
                                mz_max_window = precursor_mz + mz_buffer
                            else:
                                mz_min_window = None
                                mz_max_window = None
                
                        if apex_rt and mz_min_window is not None and mz_max_window is not None:
                            spec_mzs_fallback, spec_ints_fallback = get_ms1_spectrum_at_rt(raw_file, apex_rt, rt_tolerance_sec=2.0)
                            if spec_mzs_fallback is not None and len(spec_mzs_fallback) > 0:
                                mask_fallback = (spec_mzs_fallback >= mz_min_window) & (spec_mzs_fallback <= mz_max_window)
                                spec_mzs_windowed_fallback = spec_mzs_fallback[mask_fallback]
                                spec_ints_windowed_fallback = spec_ints_fallback[mask_fallback]
                                if len(spec_mzs_windowed_fallback) > 0:
                                    markerline_fallback, stemlines_fallback, baseline_fallback = ax_combined_spec.stem(
                                        spec_mzs_windowed_fallback, spec_ints_windowed_fallback, 
                                        basefmt=' ', linefmt='#888888', markerfmt=' ')
                                    plt.setp(stemlines_fallback, linewidth=0.5)  # Finer but visible lines for fallback
                                    plt.setp(markerline_fallback, markersize=2)
                                    ax_combined_spec.set_xlabel('m/z', fontsize=7, fontfamily='serif', color='0.85')
                                    ax_combined_spec.set_ylabel('Intensity', fontsize=7, fontfamily='serif', color='0.85')
                                    ax_combined_spec.tick_params(labelsize=6, colors='0.85')
                                    ax_combined_spec.grid(False)  # Remove grid bars
                                    ax_combined_spec.set_xlim([mz_min_window, mz_max_window])
                            
                                    # Add label showing isotope matching tolerance (fallback case for combined)
                                    x_lim_spec_fallback_combined = ax_combined_spec.get_xlim()
                                    y_lim_spec_fallback_combined = ax_combined_spec.get_ylim()
                                    tolerance_label_x_fallback_combined = x_lim_spec_fallback_combined[0] + (x_lim_spec_fallback_combined[1] - x_lim_spec_fallback_combined[0]) * 0.02
                                    tolerance_label_y_fallback_combined = y_lim_spec_fallback_combined[0] + (y_lim_spec_fallback_combined[1] - y_lim_spec_fallback_combined[0]) * 0.95
                                    ax_combined_spec.text(tolerance_label_x_fallback_combined, tolerance_label_y_fallback_combined, '±5 ppm', 
                                                         fontsize=6, ha='left', va='top', fontfamily='serif',
                                                         bbox=dict(boxstyle='round,pad=0.25', facecolor='lightblue', alpha=0.8, edgecolor='black', linewidth=0.5),
                                                         zorder=25, clip_on=True)
                            
                                    drew_combined_spec = True
                                else:
                                    ax_combined_spec.axis('off')
                            else:
                                ax_combined_spec.axis('off')
                        else:
                            ax_combined_spec.axis('off')
                    # When is_rejected, ax_combined_spec was never set; skip axis('off') for that case
        
                # Add legend inside the plot (upper right corner) - Times New Roman font, single column
                # Position legend in upper right, but adjust to avoid overlap with QC box (light text on black background)
                legend = ax.legend(loc='upper right', fontsize=14, framealpha=0.9, ncol=1, prop={'family': 'serif'},
                                  facecolor='#1a1a1a', edgecolor='0.6', labelcolor='0.9')
                for text in legend.get_texts():
                    text.set_fontfamily('serif')
                    text.set_color('0.9')
        
                # Add E-value colorbar for anchor RT line (individual figure only)
                if anchor_evalue is not None and pd.notna(anchor_evalue) and anchor_evalue > 0:
                    # Create a ScalarMappable for the colorbar
                    sm = ScalarMappable(cmap=evalue_cmap, norm=evalue_norm)
                    sm.set_array([])  # Empty array, we just want the colormap
                
                    # Add colorbar positioned on the left side of the plot (to avoid legend in upper right)
                    # Create axes for colorbar manually positioned on the left (larger for legibility)
                    cbar_ax = fig.add_axes([0.02, 0.65, 0.028, 0.26])  # [left, bottom, width, height] in figure coordinates
                    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
                    cbar.set_label('E-value\n(OpenMS RT)', fontsize=13, fontfamily='serif', rotation=0, labelpad=14)
                    cbar.ax.tick_params(labelsize=12)
                    for label in cbar.ax.get_yticklabels():
                        label.set_fontfamily('serif')
        
                # Add legend to combined figure (first subplot of each file only) - Times New Roman font, single column
                if not is_rejected and local_idx == 0:
                    # Position legend inside the combined plot (upper right corner)
                    legend_combined = ax_combined_chrom.legend(loc='upper right', 
                                                               fontsize=11, framealpha=0.9, ncol=1, prop={'family': 'serif'})
                    for text in legend_combined.get_texts():
                        text.set_fontfamily('serif')
        
                # Increment combined-figure slot only when we drew both chrom and spec (not when cleared for n_matches_combined < 2)
                if not is_rejected and drew_combined_spec:
                    total_area_val = total_area if total_area is not None and not (isinstance(total_area, float) and (np.isnan(total_area) or total_area < 0)) else None
                    seq_start_pos = 999999
                    if 'sequence_positions' in group.columns:
                        seq_pos = group['sequence_positions'].iloc[0]
                        if pd.notna(seq_pos) and seq_pos != '':
                            try:
                                seq_pos_str = str(seq_pos).strip()
                                if '-' in seq_pos_str:
                                    seq_start_pos = int(seq_pos_str.split('-')[0])
                                elif ',' in seq_pos_str:
                                    seq_start_pos = int(seq_pos_str.split(',')[0].strip())
                                else:
                                    seq_start_pos = int(seq_pos_str)
                            except (ValueError, IndexError):
                                pass
                    chrom_zoom_data.append({
                        'rts': np.asarray(rts, dtype=float).copy(),
                        'rts_min': np.asarray(rts / 60.0, dtype=float).copy(),
                        'total_intensities': np.asarray(total_intensities, dtype=float).copy(),
                        'min_rt': min_rt,
                        'max_rt': max_rt,
                        'collection_min_rt': collection_min_rt if collection_min_rt else None,
                        'collection_max_rt': collection_max_rt if collection_max_rt else None,
                        'label': str(title_combined),
                        'peptide_index_1based': picked_idx + 1,
                        'total_area': total_area_val
                    })
                    overlay_data.append({
                        'rts': np.asarray(rts, dtype=float).copy(),
                        'total_intensities': np.asarray(total_intensities, dtype=float).copy(),
                        'label': title_combined,
                        'total_area': total_area_val,
                        'sequence_start_pos': seq_start_pos,
                        'coelution_score': coelution_score,
                        'shape_corr': shape_corr,
                        'ratio_stability': ratio_stability,
                        'peak_quality_score': peak_quality_score,
                        'peak_score': best_score,
                        'min_rt': min_rt,
                        'max_rt': max_rt,
                        'collection_min_rt': collection_min_rt if collection_min_rt else None,
                        'collection_max_rt': collection_max_rt if collection_max_rt else None,
                    })
                    all_peptide_scatter_data[peptide_key] = {
                        'coelution_score': coelution_score,
                        'shape_corr': shape_corr,
                        'ratio_stability': ratio_stability,
                        'peak_score': best_score,
                        'total_area': total_area_val
                    }
                    picked_idx += 1
        
                # Save individual figure
                # Create safe filename starting with sequence position, then peptide sequence, m/z, modifications, and charge
                safe_peptide = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in peptide)
        
                # Count significant single-AA overhangs and significant fragments for filename prefix (must match panel: only MS2 5 ppm + noise %)
                n_sig_overhangs = 0
                n_sig_fragments = 0
                # Filename prefix: ONLY significant (MS2 5 ppm + >0.5%); no CSV fallback
                if placed_ion_labels is not None:
                    n_sig_fragments = len(placed_ion_labels)
                    n_sig_overhangs = len(overhang_positions_from_placed) if overhang_positions_from_placed is not None else 0
                else:
                    n_sig_overhangs = 0
                    n_sig_fragments = 0
        
                # Sequence start position (for peak_window_data CSV and rejected_peptides dict)
                seq_start_pos = 999999
                if 'sequence_positions' in group.columns:
                    seq_pos = group['sequence_positions'].iloc[0]
                    if pd.notna(seq_pos) and seq_pos != '':
                        try:
                            seq_pos_str = str(seq_pos).strip()
                            if '-' in seq_pos_str:
                                seq_start_pos = int(seq_pos_str.split('-')[0])
                            elif ',' in seq_pos_str:
                                seq_start_pos = int(seq_pos_str.split(',')[0].strip())
                            else:
                                seq_start_pos = int(seq_pos_str)
                        except (ValueError, IndexError):
                            pass
        
                # Get modifications for filename
                mods = group['modifications'].iloc[0] if 'modifications' in group.columns else '-'
                if pd.isna(mods) or mods == '' or mods == '-':
                    safe_mods = 'nomod'
                else:
                    # Format modifications compactly for filename (e.g., "Ox@1" instead of "1_V_15.994900")
                    mods_str = str(mods)
                    # Common modifications mapping (same as format_title)
                    mod_map = {
                        '15.9949': 'Ox',  # Oxidation
                        '15.994900': 'Ox',  # Oxidation (with trailing zeros)
                        '57.0215': 'Carb',  # Carbamidomethylation (abbreviated)
                        '57.021500': 'Carb',
                        '0.9840': 'Deam',  # Deamidation (abbreviated)
                        '0.984000': 'Deam',
                    }
                
                    # Parse modifications: format is "pos_type_mass" or "pos_type_mass,pos_type_mass"
                    mod_parts = []
                    for mod_entry in mods_str.split(','):
                        if '_' in mod_entry:
                            parts = mod_entry.split('_')
                            if len(parts) >= 3:
                                pos = parts[0]
                                mod_type = parts[1]
                                mass = parts[2].split(')')[0]  # Remove any trailing )
                                # Try exact match first, then try rounding
                                mod_name = mod_map.get(mass, None)
                                if mod_name is None:
                                    # Try rounding to 4 decimal places
                                    try:
                                        mass_rounded = f"{float(mass):.4f}"
                                        mod_name = mod_map.get(mass_rounded, f"M{mass}")
                                    except:
                                        mod_name = f"M{mass}"
                                # Format as "Mod@Pos" (e.g., "Ox@1")
                                mod_parts.append(f"{mod_name}@{pos}")
                
                    if mod_parts:
                        # Join modifications with comma, limit to reasonable length
                        safe_mods = ','.join(mod_parts[:5])  # Show up to 5 modifications
                        if len(mod_parts) > 5:
                            safe_mods += f"+{len(mod_parts)-5}"
                        # Sanitize: keep only alphanumeric, @, comma, and + for "more" indicator
                        safe_mods = "".join(c if c.isalnum() or c in ('@', ',', '+') else '_' for c in safe_mods)
                        # Limit total length
                        if len(safe_mods) > 40:
                            safe_mods = safe_mods[:40]
                    else:
                        # Fallback: sanitize original string if parsing fails
                        safe_mods = "".join(c if c.isalnum() or c in ('-', '_', '.') else '_' for c in mods_str)
                        if len(safe_mods) > 30:
                            safe_mods = safe_mods[:30]
        
                # Format m/z for filename (replace decimal point with 'p' to avoid filesystem issues)
                mz_str = f"{precursor_mz:.2f}".replace('.', 'p')  # e.g., 500.25 -> 500p25
        
                # Collection window in minutes for filename (e.g. rt12p5_15p2min)
                if collection_min_rt_min is not None and collection_max_rt_min is not None:
                    coll_str = f"rt{f'{collection_min_rt_min:.1f}'.replace('.', 'p')}_{f'{collection_max_rt_min:.1f}'.replace('.', 'p')}min"
                else:
                    coll_str = "rtNA"
        
                # Total area for filename (e.g. ta1p23e7)
                try:
                    ta_val = total_area
                    if ta_val is not None and not (isinstance(ta_val, float) and (np.isnan(ta_val) or ta_val < 0)):
                        total_str = "ta" + f"{ta_val:.2e}".replace('.', 'p').replace('+', '')
                    else:
                        total_str = "taNA"
                except NameError:
                    total_str = "taNA"
        
                # Format: noverhangs_nfrags_peptide_mz_mods_zcharge_rtX_Ymin_taZ.png
                # First two numbers = significant single-AA overhangs, significant fragments (sortable).
                filename = f"{n_sig_overhangs:03d}_{n_sig_fragments:03d}_{safe_peptide}_{mz_str}_{safe_mods}_z{charge}_{coll_str}_{total_str}.png"
        
                # Collect peak window data for CSV export (for both accepted and rejected peptides)
                mz_vals = group['mz'].dropna().unique().tolist() if 'mz' in group.columns else []
                rt_range = max(ms1_rts_table) - min(ms1_rts_table) if len(ms1_rts_table) > 1 else 0
        
                # Envelope metrics are computed above from strict summed-MS1
                # integration-window logic (highest peak within +/-5 ppm for M0/M+1/M+2).
                apex_intensity_val = best_peak.get('apex_intensity') if best_peak else None
        
                peak_window_data = {
                'peptide': peptide,
                'peptide_key': peptide_key,
                'charge': charge,
                'modifications': mods if pd.notna(mods) else '-',
                'has_required_isotopes': has_required_in_window,
                'status': 'rejected' if is_rejected else 'accepted',
                'rejection_reason': rejection_reason if is_rejected else '',
                'num_psms': len(group),
                'unique_mz': len(mz_vals),
                'ms1_rt_range': rt_range,
                'ms1_rt_min': min(ms1_rts_table) if len(ms1_rts_table) > 0 else None,
                'ms1_rt_max': max(ms1_rts_table) if len(ms1_rts_table) > 0 else None,
                'anchor_rt': rt_anchor,
                'anchor_evalue': anchor_evalue,
                'apex_rt': apex_rt if apex_rt else None,
                'apex_intensity': apex_intensity_val,
                'detected_peak_min_rt': min_rt if min_rt else None,
                'detected_peak_max_rt': max_rt if max_rt else None,
                'detected_peak_window_size': (max_rt - min_rt) if (min_rt and max_rt) else None,
                'collection_min_rt': collection_min_rt if collection_min_rt else None,
                'collection_max_rt': collection_max_rt if collection_max_rt else None,
                'collection_window_size': (collection_max_rt - collection_min_rt) if (collection_min_rt and collection_max_rt) else None,
                'spectrum_window_min_rt': spectrum_rt_min,
                'spectrum_window_max_rt': spectrum_rt_max,
                'spectrum_window_size': (spectrum_rt_max - spectrum_rt_min) if (spectrum_rt_min and spectrum_rt_max) else None,
                'using_anchor_window': using_anchor_window,
                'n_isotopic_matches': len(core_matched_indices),
                'matched_isotopes': ', '.join([f'M+{i}' for i in core_matched_indices]) if core_matched_indices else '',
                'm0_m1_gt_m2_m3': m0_m1_gt_m2_m3,
                'm0_intensity': I0 if has_required_in_window else 0.0,
                'm1_intensity': I1 if has_required_in_window else 0.0,
                'm2_intensity': I2 if has_required_in_window else 0.0,
                'm3_intensity': I3 if has_required_in_window else 0.0,
                'm0_ppm_error': core_peak_metrics.get(0, {}).get('ppm_error'),
                'm1_ppm_error': core_peak_metrics.get(1, {}).get('ppm_error'),
                'm2_ppm_error': core_peak_metrics.get(2, {}).get('ppm_error'),
                'precursor_mz': precursor_mz,
                'sequence_start_pos': seq_start_pos,
                'ms1_trace_ok': True,  # Had >=1 scan (within 5 ppm in pipeline); for backfill eligibility in regenerate_rt_windows
                'ppm_at_apex': ppm_at_apex
                }
        
                if pass_num == 1 and is_rejected and rejection_reason:
                    cat = _rejection_category(rejection_reason)
                    if cat:
                        rejection_counts[cat] += 1
        
                # Add QC metrics if available (these may not be set if peak detection failed)
                try:
                    if 'best_score' in locals() and best_score is not None:
                        peak_window_data['best_score'] = best_score
                except:
                    pass
                try:
                    if 'n_candidates' in locals() and n_candidates is not None:
                        peak_window_data['n_candidates'] = n_candidates
                except:
                    pass
                try:
                    if 'total_area' in locals() and total_area is not None:
                        peak_window_data['total_area'] = total_area
                except:
                    pass
                try:
                    if 'coelution_score' in locals() and coelution_score is not None:
                        peak_window_data['coelution_score'] = coelution_score
                except:
                    pass
                try:
                    if 'shape_corr' in locals() and shape_corr is not None:
                        peak_window_data['shape_corr'] = shape_corr
                except:
                    pass
        
                all_peak_windows.append(peak_window_data)

                # Collect traces: per-scan RT + per-scan intensities (never aggregates like total_area)
                rts_arr = np.asarray(rts, dtype=float).flatten().copy()
                if extract_only:
                    int_arr = np.asarray(intensity_matrix, dtype=float).copy()
                else:
                    total_1d = np.sum(intensity_matrix, axis=1) if intensity_matrix.ndim == 2 else np.asarray(intensity_matrix, dtype=float)
                    int_arr = np.asarray(total_1d, dtype=float).flatten().copy()
                if len(rts_arr) != (int_arr.shape[0] if int_arr.ndim >= 1 else len(int_arr)):
                    n = min(len(rts_arr), int_arr.shape[0] if int_arr.ndim >= 1 else len(int_arr))
                    rts_arr = rts_arr[:n].copy()
                    int_arr = int_arr[:n] if int_arr.ndim == 1 else int_arr[:n, :].copy()
                if len(rts_arr) < 2:
                    continue
                traces_list.append((rts_arr, int_arr))

                if extract_only:
                    if current_peptide_1based % 10 == 0 or current_peptide_1based == num_peptides:
                        print(f"  Peptide {current_peptide_1based}/{num_peptides} (metrics+traces only, no PNG)")
                    continue
    
                # Save to rejected directory if rejected, otherwise to normal directory
                if is_rejected:
                    filepath = os.path.join(rejected_dir, filename)
                    # Add rejection reason to the figure suptitle
                    current_suptitle = fig._suptitle.get_text() if fig._suptitle else '\n'.join(big_title_lines)
                    fig.suptitle(current_suptitle + f'\n[REJECTED: {rejection_reason}]', fontsize=20, fontweight='bold', fontfamily='serif', y=0.94, va='top', color='red')
                
                    # Store rejected peptide data for combined figure (collect all necessary data)
                    rejected_peptides.append({
                        'peptide_key': peptide_key,
                        'peptide': peptide,
                        'charge': charge,
                        'mods': mods,
                        'title': title,
                        'rejection_reason': rejection_reason,
                        'rts': rts,
                        'rts_min': rts_min,
                        'intensity_matrix': intensity_matrix,
                        'total_intensities': total_intensities,
                        'scored_candidates': scored_candidates,
                        'best_peak': best_peak,
                        'min_rt': min_rt,
                        'max_rt': max_rt,
                        'min_rt_min': min_rt_min,
                        'max_rt_min': max_rt_min,
                        'collection_min_rt_min': collection_min_rt_min,
                        'collection_max_rt_min': collection_max_rt_min,
                        'apex_rt': apex_rt,
                        'apex_rt_min': apex_rt_min,
                        'rt_anchor': rt_anchor,
                        'anchor_evalue': anchor_evalue,
                        'rt_to_frag_count': rt_to_frag_count,
                        'spec_mzs_windowed': spec_mzs_windowed,
                        'spec_ints_windowed': spec_ints_windowed,
                        'isotope_mzs': isotope_mzs,
                        'measured_mz_matrix': measured_mz_matrix,
                        'start_idx': start_idx,
                        'end_idx': end_idx,
                        'mz_min_window': mz_min_window,
                        'mz_max_window': mz_max_window,
                        'spectrum_rt_min': spectrum_rt_min,
                        'spectrum_rt_max': spectrum_rt_max,
                        'using_anchor_window': using_anchor_window,
                        'n_matches_final': n_matches_final,
                        'matched_iso_indices_final': matched_iso_indices_final,
                        'peptide_crossref': peptide_crossref,
                        'evalue_cmap': evalue_cmap,
                        'evalue_norm': evalue_norm,
                        'seq_start_pos': seq_start_pos,
                        'precursor_mz': precursor_mz,
                        'ms1_rts_table': ms1_rts_table,
                        'ms1_rt_evalue_pairs': [(float(row['MS1_retention_time_sec']), row.get('e-value')) for _, row in group.iterrows() if pd.notna(row.get('MS1_retention_time_sec')) and row['MS1_retention_time_sec'] > 0],
                        'idx': len(rejected_peptides)  # Index in rejected list
                    })
                else:
                    filepath = os.path.join(output_dir, filename)
        
                # Skip tight_layout() as it can cause figure resizing - we're using fixed dimensions instead
        
                # Final lock: Explicitly set all axis limits one more time before saving to prevent any resizing
                # Get current limits (they should already be set correctly)
                x_lim_current = ax.get_xlim()
                y_lim_current = ax.get_ylim()
                x_lim_spec_current = ax_spec.get_xlim()
                y_lim_spec_current = ax_spec.get_ylim()
                x_lim_ms2_current = ax_spec_ms2.get_xlim()
                y_lim_ms2_current = ax_spec_ms2.get_ylim()
        
                # Re-apply limits explicitly to ensure they're locked (keeps MS1+MS2 stacked layout)
                ax.set_xlim(x_lim_current)
                ax.set_ylim(y_lim_current)
                ax_spec.set_xlim(x_lim_spec_current)
                ax_spec.set_ylim(y_lim_spec_current)
                ax_spec_ms2.set_xlim(x_lim_ms2_current)
                ax_spec_ms2.set_ylim(y_lim_ms2_current)
        
                # Lock figure size - ensure it hasn't changed (wide x for readable time axis; height includes taller overlay row)
                fig.set_size_inches(16, 28)
        
                # Use fixed bbox instead of 'tight' to prevent figure resizing
                # bbox_inches=None uses the figure size as-is without padding adjustments
                abspath = os.path.abspath(filepath)
                print(f"[DEBUG] CHROMATOGRAMS_PEPTIDES: Saving PNG -> {abspath}")
                print(f"[DEBUG] Saving figure to: {abspath}")
                try:
                    os.makedirs(rejected_dir if is_rejected else output_dir, exist_ok=True)
                    plt.savefig(filepath, dpi=200, bbox_inches=None, facecolor='black', pad_inches=0.1)
                    if is_rejected:
                        saved_rejected_paths.append(abspath)
                    else:
                        saved_accepted_paths.append(abspath)
                    decision = f"rejected ({rejection_reason})" if is_rejected else "accepted"
                    print(f"[DEBUG] CHROMATOGRAMS_PEPTIDES: saved -> {abspath} [{decision}]")
                    print(f"[DEBUG] Peptide {current_peptide_1based}/{num_peptides}: {peptide_key} -> {decision}")
                    print(f"[DEBUG] Saved {'rejected' if is_rejected else 'accepted'} individual plot: {abspath}")
                except Exception as e:
                    print(f"[ERROR] Failed to save peptide figure: {abspath}: {e}", file=sys.stderr)
                    print(f"[ERROR] Peptide {current_peptide_1based}/{num_peptides}: {peptide_key} ({'rejected' if is_rejected else 'accepted'})", file=sys.stderr)
        
                # Close figure (we've saved it and collected data for combined figure)
                plt.close(fig)
        
                if current_peptide_1based % 10 == 0:
                    print(f"[DEBUG] Saved {current_peptide_1based}/{num_peptides} peptide figures...")
        
                # Test mode: we process a fixed sample (no early stop)
        
                # Store peak window data
                mz_vals = group[mz_col].dropna().unique().tolist() if mz_col in group.columns else []
                rt_range = max(ms1_rts_table) - min(ms1_rts_table) if len(ms1_rts_table) > 1 else 0
        
                # Store data for best peak (Skyline-style metrics)
                peak_windows.append({
                'peptide': peptide,
                'peptide_key': peptide_key,
                'charge': charge,
                'modifications': mods,
                'num_psms': len(group),
                'unique_mz': len(mz_vals),
                'ms1_rt_range': rt_range,
                'ms1_rt_min': min(ms1_rts_table),
                'ms1_rt_max': max(ms1_rts_table),
                'n_candidates': n_candidates,
                'best_score': best_score,
                'second_best_score': second_best_score,
                'delta_score': delta_score,
                'anchor_rt': rt_anchor,
                'apex_rt': apex_rt if apex_rt else None,
                'min_rt': min_rt,
                'max_rt': max_rt,
                'window_size': max_rt - min_rt,
                'area_M': isotope_areas.get(0, 0.0),
                'area_M1': isotope_areas.get(1, 0.0),
                'area_M2': isotope_areas.get(2, 0.0),
                'area_M3': isotope_areas.get(3, 0.0),
                'total_area': total_area,
                'coelution_var': coelution_var,
                'coelution_score': coelution_score,
                'mean_shape_corr': shape_corr,
                'ratio_stability': ratio_stability,
                'rt_offset_sec': rt_offset,
                'rt_deviation_flag': rt_deviation_flag,
                'peak_quality_sn': peak_quality_raw,
                'peak_quality_score': peak_quality_score,
                'interference_flag': interference_flag,
                'peak2_loss_reasons': '; '.join(peak2_loss_reasons) if peak2_loss_reasons else '',
                'peak_score': best_score,
                'ppm_at_apex': ppm_at_apex
        })

    if extract_only and len(all_peak_windows) > 0 and traces_list is not None and len(traces_list) == len(all_peak_windows):
        extract_output_dir = getattr(args, 'extract_output_dir', None) or os.path.dirname(os.path.abspath(output_png))
        dataframes_dir = getattr(args, 'dataframes_dir', None) or extract_output_dir
        os.makedirs(extract_output_dir, exist_ok=True)
        df_extract = pd.DataFrame(all_peak_windows)
        df_extract['trace_index'] = range(len(df_extract))
        # Metrics CSV (used by plot-from-dataframe)
        metrics_path = os.path.join(extract_output_dir, 'chromatogram_metrics_all.csv')
        df_extract.to_csv(metrics_path, index=False)
        npz_path = os.path.join(extract_output_dir, 'chromatogram_traces.npz')
        save_dict = {}
        for i, (rts_arr, int_arr) in enumerate(traces_list):
            save_dict[f'trace_{i}_rts'] = np.asarray(rts_arr, dtype=float).flatten().copy()
            save_dict[f'trace_{i}_int'] = np.asarray(int_arr, dtype=float).copy()
        np.savez_compressed(npz_path, **save_dict)
        # Merge extraction metrics into input CSV (for downstream Envelope, Significance steps)
        os.makedirs(dataframes_dir, exist_ok=True)
        def _norm_mods_df(m):
            if m is None or (isinstance(m, float) and pd.isna(m)):
                return '-'
            s = str(m).strip()
            return s if s and s.lower() != 'nan' else '-'
        chrom_cols = [
            'status', 'rejection_reason', 'num_psms', 'unique_mz', 'ms1_rt_range', 'ms1_rt_min', 'ms1_rt_max',
            'anchor_rt', 'anchor_evalue', 'apex_rt', 'apex_intensity',
            'detected_peak_min_rt', 'detected_peak_max_rt', 'detected_peak_window_size',
            'collection_min_rt', 'collection_max_rt', 'collection_window_size',
            'spectrum_window_min_rt', 'spectrum_window_max_rt', 'spectrum_window_size',
            'using_anchor_window', 'n_isotopic_matches', 'matched_isotopes',
            'm0_m1_gt_m2_m3', 'm0_intensity', 'm1_intensity', 'm2_intensity', 'm3_intensity',
            'precursor_mz', 'sequence_start_pos', 'ppm_at_apex',
            'best_score', 'n_candidates', 'total_area', 'coelution_score', 'shape_corr'
        ]
        def _format_extraction_output_columns(df_out):
            """Apply extraction output naming + ordering conventions for UI clarity."""
            df_fmt = df_out.copy()

            # Use extracted precursor m/z as theoretical_mz in outputs.
            if 'precursor_mz' in df_fmt.columns:
                if 'theoretical_mz' in df_fmt.columns:
                    df_fmt['theoretical_mz'] = df_fmt['precursor_mz'].where(
                        pd.notna(df_fmt['precursor_mz']),
                        df_fmt['theoretical_mz']
                    )
                    df_fmt = df_fmt.drop(columns=['precursor_mz'])
                else:
                    df_fmt = df_fmt.rename(columns={'precursor_mz': 'theoretical_mz'})
            # Standardize legacy retention_time headers to RT headers.
            rt_name_map = {
                'MS1_retention_time_sec': 'MS1_RT_sec',
                'MS1_retention_time_min': 'MS1_RT_minutes',
                'MS1_retention_time_intensity': 'MS1_RT_intensity',
                'MS2_retention_time_sec': 'MS2_RT_sec',
                'MS2_retention_time_min': 'MS2_RT_minutes',
                'retention_time_sec': 'RT_sec',
                'retention_time_min': 'RT_minutes',
            }
            for old, new in rt_name_map.items():
                if old in df_fmt.columns:
                    if new in df_fmt.columns:
                        df_fmt[new] = df_fmt[old].where(pd.notna(df_fmt[old]), df_fmt[new])
                        df_fmt = df_fmt.drop(columns=[old])
                    else:
                        df_fmt = df_fmt.rename(columns={old: new})

            # Rename RT window columns (seconds)
            rt_rename = {
                'detected_peak_min_rt': 'MS1_RT_integration_start_sec',
                'detected_peak_max_rt': 'MS1_RT_integration_stop_sec',
                'collection_min_rt': 'MS1_RT_collection_start_sec',
                'collection_max_rt': 'MS1_RT_collection_stop_sec',
            }
            for old, new in rt_rename.items():
                if old in df_fmt.columns:
                    if new in df_fmt.columns:
                        df_fmt[new] = df_fmt[old].where(pd.notna(df_fmt[old]), df_fmt[new])
                        df_fmt = df_fmt.drop(columns=[old])
                    else:
                        df_fmt = df_fmt.rename(columns={old: new})

            # Add RT window columns (minutes)
            sec_to_min = [
                ('MS1_RT_integration_start_sec', 'MS1_RT_integration_start_minutes'),
                ('MS1_RT_integration_stop_sec', 'MS1_RT_integration_stop_minutes'),
                ('MS1_RT_collection_start_sec', 'MS1_RT_collection_start_minutes'),
                ('MS1_RT_collection_stop_sec', 'MS1_RT_collection_stop_minutes'),
            ]
            for sec_col, min_col in sec_to_min:
                if sec_col in df_fmt.columns and min_col not in df_fmt.columns:
                    df_fmt[min_col] = pd.to_numeric(df_fmt[sec_col], errors='coerce') / 60.0

            # Reorder block 1: immediately after observed_mz
            block_after_observed = [c for c in ['theoretical_mz', 'total_area', 'matched_isotopes'] if c in df_fmt.columns]
            if 'observed_mz' in df_fmt.columns and block_after_observed:
                cols = list(df_fmt.columns)
                rest = [c for c in cols if c not in block_after_observed]
                insert_idx = rest.index('observed_mz') + 1
                cols_new = rest[:insert_idx] + block_after_observed + rest[insert_idx:]
                df_fmt = df_fmt[cols_new]

            # Reorder block 2: immediately after all Q/PEP columns
            rt_block = [
                c for c in [
                    'MS1_RT_integration_start_sec', 'MS1_RT_integration_stop_sec',
                    'MS1_RT_collection_start_sec', 'MS1_RT_collection_stop_sec',
                    'MS1_RT_integration_start_minutes', 'MS1_RT_integration_stop_minutes',
                    'MS1_RT_collection_start_minutes', 'MS1_RT_collection_stop_minutes',
                ] if c in df_fmt.columns
            ]
            qpep_candidates = ['perc_qvalue', 'perc_PEP', 'qvalue', 'pep', 'percolator_qvalue', 'percolator_PEP', 'q-value', 'PEP']
            if rt_block:
                cols = list(df_fmt.columns)
                rest = [c for c in cols if c not in rt_block]
                qpep_idx = [i for i, c in enumerate(rest) if c in qpep_candidates]
                if qpep_idx:
                    insert_idx = max(qpep_idx) + 1
                    cols_new = rest[:insert_idx] + rt_block + rest[insert_idx:]
                    df_fmt = df_fmt[cols_new]

            frag_tail = [c for c in ['comet_matched_frags', 'comet_matched_frags_mz', 'comet_matched_frags_intensities', 'comet_matched_frags_quality_scores'] if c in df_fmt.columns]
            ms2_tail = [c for c in ['MS1_RT_sec', 'MS2_RT_sec', 'MS2_RT_minutes'] if c in df_fmt.columns]
            if 'MS1_mz_error' in df_fmt.columns and 'MS1_mz_error_ppm' not in df_fmt.columns:
                df_fmt = df_fmt.rename(columns={'MS1_mz_error': 'MS1_mz_error_ppm'})
            front_cols = [c for c in ['protein_position', 'peptide_sequence', 'charge', 'observed_mz', 'theoretical_mz', 'MS1_mz_error_ppm', 'MS1_RT_minutes'] if c in df_fmt.columns]
            ms1_cols = [c for c in ['MS1_RT_intensity'] if c in df_fmt.columns]
            if front_cols or ms1_cols or frag_tail or ms2_tail:
                lead_cols = [c for c in df_fmt.columns if c not in front_cols and c not in ms1_cols and c not in frag_tail and c not in ms2_tail]
                df_fmt = df_fmt[front_cols + ms1_cols + lead_cols + frag_tail + ms2_tail]

            # Place ppm_at_apex immediately after MS1_mz_error_ppm for extraction readability.
            if 'MS1_mz_error_ppm' in df_fmt.columns and 'ppm_at_apex' in df_fmt.columns:
                cols = list(df_fmt.columns)
                cols.remove('ppm_at_apex')
                insert_idx = cols.index('MS1_mz_error_ppm') + 1
                cols = cols[:insert_idx] + ['ppm_at_apex'] + cols[insert_idx:]
                df_fmt = df_fmt[cols]

            # Final ordering rule: RT columns explicitly in seconds always go last.
            rt_sec_cols = [
                c for c in df_fmt.columns
                if (('RT' in c or 'retention_time' in c) and c.endswith('_sec'))
            ]
            if rt_sec_cols:
                non_rt_sec_cols = [c for c in df_fmt.columns if c not in rt_sec_cols]
                df_fmt = df_fmt[non_rt_sec_cols + rt_sec_cols]

            return df_fmt
        windows_lookup = {}
        for w in all_peak_windows:
            pep = str(w.get('peptide', '')).strip()
            ch = int(w.get('charge', 0)) if w.get('charge') is not None else 0
            mod = _norm_mods_df(w.get('modifications', '-'))
            windows_lookup[(pep, ch, mod)] = {c: w.get(c) for c in chrom_cols if c in w}
        try:
            with open(comet_csv, 'r') as f:
                first_line = f.readline()
            skip_rows = 1 if 'CometVersion' in first_line else 0
        except Exception:
            skip_rows = 0
        df_base = pd.read_csv(comet_csv, sep=',', skiprows=skip_rows, engine='python', quotechar='"', on_bad_lines='warn')
        base_name = os.path.splitext(os.path.basename(comet_csv))[0]
        # In test mode, keep only rows that were actually processed so extraction-test
        # outputs do not appear mostly empty after merge-back.
        if getattr(args, 'test', False):
            def _row_key_for_filter(row_obj):
                pep = ''
                for col in ['peptide_sequence', 'plain_peptide', 'sequence']:
                    v = row_obj.get(col, '')
                    if pd.notna(v) and str(v).strip():
                        pep = str(v).strip()
                        break
                ch = int(row_obj.get('charge', 0)) if pd.notna(row_obj.get('charge')) else 0
                mod = _norm_mods_df(row_obj.get('modifications', '-'))
                return (pep, ch, mod)
            keep_mask = df_base.apply(lambda r: _row_key_for_filter(r) in windows_lookup, axis=1)
            n_before = len(df_base)
            df_base = df_base.loc[keep_mask].copy()
            print(f"[DEBUG] Test mode merge: keeping {len(df_base)} of {n_before} rows with extracted metrics")

        for c in chrom_cols:
            if c not in df_base.columns:
                df_base[c] = None
        for c in ['significant_fragment_ions', 'significant_fragment_pairs', 'single_aa_overhangs_protein_positions']:
            if c not in df_base.columns:
                df_base[c] = None
        # Reset canonical significant columns so stale legacy values are not carried forward.
        for c in ['significant_fragment_ions', 'significant_fragment_pairs', 'single_aa_overhangs_protein_positions']:
            df_base[c] = None
        def _row_peptide_key_val(row_obj):
            # Canonical name first; fallback to legacy names.
            for col in ['peptide_sequence', 'plain_peptide', 'sequence']:
                v = row_obj.get(col, '')
                if pd.notna(v) and str(v).strip():
                    return str(v).strip()
            return ''

        for idx, row in df_base.iterrows():
            key = (
                _row_peptide_key_val(row),
                int(row.get('charge', 0)) if pd.notna(row.get('charge')) else 0,
                _norm_mods_df(row.get('modifications', '-'))
            )
            if key in windows_lookup:
                for col, val in windows_lookup[key].items():
                    if val is not None:
                        df_base.at[idx, col] = val
            if key in significant_data:
                for col, val in significant_data[key].items():
                    if val is not None and col in df_base.columns:
                        df_base.at[idx, col] = val
        for legacy_col in ['significant_single_aa_overhangs', 'significant_single_aa_overhangs_protein_positions']:
            if legacy_col in df_base.columns:
                df_base = df_base.drop(columns=[legacy_col])
        merged_csv = os.path.join(dataframes_dir, f'{base_name}_with_chromatogram_metrics.csv')
        _format_extraction_output_columns(df_base).to_csv(merged_csv, index=False)
        print(f"[DEBUG] Merged extraction CSV saved to: {os.path.abspath(merged_csv)}")
        # Human-readable summary
        out_abs = os.path.abspath(extract_output_dir)
        print("")
        print("  Chromatogram extraction complete (extract-only, no PNGs).")
        print(f"  Output folder: {out_abs}")
        print("  Files written:")
        print(f"    · chromatogram_metrics_all.csv   — metrics for all peptides")
        print(f"    · chromatogram_traces.npz        — RT/intensity traces for plotting")
        print(f"    · {base_name}_with_chromatogram_metrics.csv — merged CSV for downstream steps")
        print("  To generate plots later, run:")
        print(f"    plot-from-dataframe --plot-only --chromatogram-metrics-csv \"{out_abs}/chromatogram_metrics_all.csv\" --chromatogram-traces \"{out_abs}/chromatogram_traces.npz\"")
        print("")
        return

    # Debug summary: where did each peptide figure go?
    print(f"[DEBUG] ========== CHROMATOGRAMS_PEPTIDES: SAVE SUMMARY ==========")
    print(f"[DEBUG] === SAVE SUMMARY ===")
    print(f"[DEBUG] Chromatogram peptide plots (one PNG per peptide):")
    print(f"[DEBUG]   Accepted: {output_dir_abs}  ({len(saved_accepted_paths)} files)")
    print(f"[DEBUG]   Rejected: {rejected_dir_abs}  ({len(saved_rejected_paths)} + {len(early_rejected_paths)} early-rejected)")
    print(f"[DEBUG] Early-rejected (minimal): {len(early_rejected_paths)} -> {rejected_dir_abs}")
    print(f"[DEBUG] Total peptides: {num_peptides} (accepted + rejected + early-rejected = {len(saved_accepted_paths) + len(saved_rejected_paths) + len(early_rejected_paths)})")
    if saved_accepted_paths:
        print(f"[DEBUG] CHROMATOGRAMS_PEPTIDES: First 3 accepted paths: {saved_accepted_paths[:3]}")
    if saved_rejected_paths:
        print(f"[DEBUG] CHROMATOGRAMS_PEPTIDES: First 3 rejected paths: {saved_rejected_paths[:3]}")
    if early_rejected_paths:
        print(f"[DEBUG] CHROMATOGRAMS_PEPTIDES: First 3 early-rejected paths: {early_rejected_paths[:3]}")
    if (len(saved_accepted_paths) + len(saved_rejected_paths) + len(early_rejected_paths)) == 0:
        print(f"[DEBUG] CHROMATOGRAMS_PEPTIDES: WARNING: No figures were saved!")
        print(f"[DEBUG] WARNING: No figures were saved! Check: num_peptides={num_peptides}, output_dir={output_dir!r}, rejected_dir={rejected_dir!r}")
    print(f"[DEBUG] === END SAVE SUMMARY ===")

    # Custom total_area colormap: lowest scores fade lavender -> purple -> dark red; rest = Spectral (red, orange, yellow, green, blue)
    def _total_area_colormap():
        from matplotlib.colors import ListedColormap
        spectral = plt.get_cmap('Spectral')
        lavender = np.array([0.90, 0.90, 0.98, 1.0])
        purple = np.array([0.58, 0.44, 0.86, 1.0])
        dark_red = np.array([0.55, 0.0, 0.0, 1.0])
        n_low = 64   # norm 0 to 0.25: lavender -> purple -> dark red
        n_high = 192 # norm 0.25 to 1.0: Spectral
        colors = []
        for i in range(n_low):
            t = i / max(n_low - 1, 1)
            if t < 0.33:
                c = lavender + (purple - lavender) * (t / 0.33)
            elif t < 0.66:
                c = purple + (dark_red - purple) * ((t - 0.33) / 0.33)
            else:
                c = dark_red + (np.array(spectral(0)[:4]) - dark_red) * ((t - 0.66) / 0.34)
            colors.append(c)
        # Cap high end at blue (spectral ~0.85) so top doesn't go into purple
        SPECTRAL_V_MAX = 0.85
        for i in range(n_high):
            v = SPECTRAL_V_MAX * (i / max(n_high - 1, 1))
            colors.append(spectral(v))
        return ListedColormap(colors, name='total_area')
    
    # Overview plot: accepted peptides only – Total peaks on one x-axis (MS1 RT), colored by total_area (same scale as RT windows by peptide)
    if len(overlay_data) > 0:
        # Order peptides by sequence position (y-axis: top = N-term, bottom = C-term)
        overlay_data = sorted(overlay_data, key=lambda item: (item.get('sequence_start_pos', 999999), str(item.get('label', ''))))
        # Filter for >1% relative area (for _1pct overlay variants)
        _sum_ta_overlay = sum(
            float(e.get('total_area') or 0) for e in overlay_data
            if e.get('total_area') is not None and not (isinstance(e.get('total_area'), float) and (np.isnan(e.get('total_area')) or e.get('total_area') < 0))
        )
        overlay_data_1pct = [
            e for e in overlay_data
            if _sum_ta_overlay > 0 and (float(e.get('total_area') or 0) / _sum_ta_overlay) > 0.01
        ]
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
        try:
            from matplotlib.patheffects import withStroke
            overlay_path_effects = [withStroke(linewidth=1.5, foreground='black')]
        except Exception:
            overlay_path_effects = []
        # Use same global total_area range as individual figures and window bar for consistent colors
        if area_vmax_global > area_vmin_global:
            area_norm_overview = Normalize(vmin=area_vmin_global, vmax=area_vmax_global)
            cmap_overview = _total_area_colormap_early()
        else:
            valid_areas = [item['total_area'] for item in overlay_data if item.get('total_area') is not None and not (isinstance(item['total_area'], float) and (np.isnan(item['total_area']) or item['total_area'] < 0))]
            if valid_areas:
                area_vmin = min(valid_areas)
                area_vmax = max(valid_areas)
                if area_vmax <= area_vmin:
                    area_vmax = area_vmin + 1.0
                area_norm_overview = Normalize(vmin=area_vmin, vmax=area_vmax)
            else:
                area_norm_overview = None
            cmap_overview = _total_area_colormap_early()
        # Overlay (linear + log) and tracks: give overlay panels much taller y-axes; height_ratios [5, 5, 6] = overlay ~31% each, tracks 37%
        from matplotlib.gridspec import GridSpec
        from matplotlib.ticker import MultipleLocator as _ML
        fig_overview = plt.figure(figsize=(22, 72))
        fig_overview.patch.set_facecolor('black')
        gs_overview = GridSpec(3, 1, figure=fig_overview, height_ratios=[5, 5, 6], hspace=0.35)
        ax_overview = fig_overview.add_subplot(gs_overview[0])
        ax_overview_log = fig_overview.add_subplot(gs_overview[1], sharex=ax_overview)
        ax_overview_tracks = fig_overview.add_subplot(gs_overview[2])
        for ax in (ax_overview, ax_overview_log, ax_overview_tracks):
            ax.set_facecolor('black')
        # Full RT range from data (rts are in seconds); ensure overlay spans actual run, not squashed into first minutes
        if overlay_data:
            all_rts_sec = np.concatenate([np.asarray(item['rts'], dtype=float).ravel() for item in overlay_data])
            rt_min_sec = float(np.min(all_rts_sec))
            rt_max_sec = float(np.max(all_rts_sec))
            if rt_max_sec <= rt_min_sec:
                rt_max_sec = rt_min_sec + 60.0
            buffer_min = max(0.5, (rt_max_sec - rt_min_sec) / 60.0 * 0.1)
            x_min_ov = max(0.0, rt_min_sec / 60.0 - buffer_min)
            x_max_ov = rt_max_sec / 60.0 + buffer_min
            # Tracks plot: zoom to integration windows span so peaks aren't squashed in x
            wins_min = [item['min_rt'] for item in overlay_data if item.get('min_rt') is not None]
            wins_max = [item['max_rt'] for item in overlay_data if item.get('max_rt') is not None]
            if wins_min and wins_max:
                active_min_sec = float(min(wins_min))
                active_max_sec = float(max(wins_max))
                if active_max_sec > active_min_sec:
                    buf_tracks = max(0.5, (active_max_sec - active_min_sec) / 60.0 * 0.15)
                    x_min_tracks = max(0.0, active_min_sec / 60.0 - buf_tracks)
                    x_max_tracks = active_max_sec / 60.0 + buf_tracks
                else:
                    x_min_tracks, x_max_tracks = x_min_ov, x_max_ov
            else:
                x_min_tracks, x_max_tracks = x_min_ov, x_max_ov
        else:
            x_min_ov = 0.0
            x_max_ov = 1.0
            x_min_tracks = 0.0
            x_max_tracks = 1.0
        y_max_ov = 0.0
        for item in overlay_data:
            rts_min_ov = np.asarray(item['rts'], dtype=float) / 60.0
            total_ints_ov = np.asarray(item['total_intensities'], dtype=float)
            mask = (rts_min_ov >= x_min_ov) & (rts_min_ov <= x_max_ov)
            if np.any(mask) and len(total_ints_ov) == len(rts_min_ov):
                m = float(np.max(total_ints_ov[mask]))
                if np.isfinite(m):
                    y_max_ov = max(y_max_ov, m)
        # Top: linear scale (raw intensity, no gap-breaking so lines always draw)
        x_max_ov_plot = x_max_ov if x_max_ov > x_min_ov else x_min_ov + 1.0
        # Designated windows (integration = green, collection = gray) — draw first so behind traces
        for item in overlay_data:
            min_rt_m = item.get('min_rt')
            max_rt_m = item.get('max_rt')
            cmin = item.get('collection_min_rt')
            cmax = item.get('collection_max_rt')
            if min_rt_m is not None and max_rt_m is not None:
                ax_overview.axvspan(float(min_rt_m) / 60.0, float(max_rt_m) / 60.0, alpha=0.06, color='green', zorder=0)
            if cmin is not None and cmax is not None:
                ax_overview.axvspan(float(cmin) / 60.0, float(cmax) / 60.0, alpha=0.04, color=COLLECTION_WINDOW_COLOR, zorder=0)
        for i, item in enumerate(overlay_data):
            rts_sec = item['rts']
            rts_min_overlay = np.asarray(rts_sec, dtype=float) / 60.0
            total_ints = np.asarray(item['total_intensities'], dtype=float)
            if len(rts_min_overlay) < 2 or len(total_ints) < 2:
                continue
            ta = item.get('total_area')
            if area_norm_overview is not None and ta is not None and not (isinstance(ta, float) and (np.isnan(ta) or ta < 0)):
                color = cmap_overview(area_norm_overview(ta))
            else:
                color = (0.6, 0.6, 0.6, 0.8)
            color_rgb = color[:3] if len(color) >= 3 else color
            # Fade below 10^5 (reduce alpha / increase transparency)
            low_ints = np.where(total_ints < 1e5, total_ints, 0)
            high_ints = np.where(total_ints >= 1e5, total_ints, 0)
            ax_overview.fill_between(rts_min_overlay, 0, low_ints, color=color_rgb, alpha=0.12, zorder=5)
            ax_overview.fill_between(rts_min_overlay, 0, high_ints, color=color_rgb, alpha=0.35, zorder=5)
            low_plot = np.where(total_ints < 1e5, total_ints, np.nan)
            high_plot = np.where(total_ints >= 1e5, total_ints, np.nan)
            ax_overview.plot(rts_min_overlay, low_plot, color=color_rgb, alpha=0.25, linewidth=1.0, path_effects=overlay_path_effects)
            ax_overview.plot(rts_min_overlay, high_plot, color=color_rgb, alpha=0.95, linewidth=2.5, label=item['label'], path_effects=overlay_path_effects)
        ax_overview.set_ylabel('Intensity', fontsize=13, fontfamily='serif', color='0.85')
        ax_overview.set_title('Accepted Peptides – Raw Intensity (green = integration window, gray = collection)', fontsize=14, fontweight='bold', fontfamily='serif', color='0.9')
        ax_overview.set_xlim(left=x_min_ov, right=x_max_ov_plot)
        ax_overview.xaxis.set_major_locator(_ML(max(0.5, (x_max_ov_plot - x_min_ov) / 10)))
        ax_overview.xaxis.set_minor_locator(_ML(max(0.1, (x_max_ov_plot - x_min_ov) / 30)))
        if y_max_ov <= 0:
            for item in overlay_data:
                total_ints_ov = np.asarray(item['total_intensities'], dtype=float)
                if len(total_ints_ov) > 0:
                    m = float(np.nanmax(total_ints_ov))
                    if np.isfinite(m) and m > 0:
                        y_max_ov = max(y_max_ov, m)
        if y_max_ov > 0:
            ax_overview.set_ylim(0, y_max_ov * 1.1)
        ax_overview.tick_params(labelsize=11, colors='0.85')
        for label in ax_overview.get_xticklabels() + ax_overview.get_yticklabels():
            label.set_fontfamily('serif')
        ax_overview.grid(True, alpha=0.25)
        for spine in ax_overview.spines.values():
            spine.set_color('0.6')
        ax_overview.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8, prop={'family': 'serif'}, ncol=1, labelcolor='0.9')
        # Bottom: log scale (reveals tails and coelution; diagnostic)
        y_floor = 1.0  # avoid log(0)
        for item in overlay_data:
            min_rt_m = item.get('min_rt')
            max_rt_m = item.get('max_rt')
            cmin = item.get('collection_min_rt')
            cmax = item.get('collection_max_rt')
            if min_rt_m is not None and max_rt_m is not None:
                ax_overview_log.axvspan(float(min_rt_m) / 60.0, float(max_rt_m) / 60.0, alpha=0.06, color='green', zorder=0)
            if cmin is not None and cmax is not None:
                ax_overview_log.axvspan(float(cmin) / 60.0, float(cmax) / 60.0, alpha=0.04, color=COLLECTION_WINDOW_COLOR, zorder=0)
        for i, item in enumerate(overlay_data):
            rts_sec = item['rts']
            rts_min_overlay = np.asarray(rts_sec, dtype=float) / 60.0
            total_ints = np.asarray(item['total_intensities'], dtype=float)
            total_ints_log = np.maximum(total_ints, y_floor)
            if len(rts_min_overlay) < 2 or len(total_ints_log) < 2:
                continue
            ta = item.get('total_area')
            if area_norm_overview is not None and ta is not None and not (isinstance(ta, float) and (np.isnan(ta) or ta < 0)):
                color = cmap_overview(area_norm_overview(ta))
            else:
                color = (0.6, 0.6, 0.6, 0.8)
            color_rgb = color[:3] if len(color) >= 3 else color
            # Fade below 10^5 (reduce alpha / increase transparency)
            low_log = np.where(total_ints_log < 1e5, total_ints_log, y_floor)
            high_log = np.where(total_ints_log >= 1e5, total_ints_log, y_floor)
            ax_overview_log.fill_between(rts_min_overlay, y_floor, low_log, color=color_rgb, alpha=0.12, zorder=5)
            ax_overview_log.fill_between(rts_min_overlay, y_floor, high_log, color=color_rgb, alpha=0.35, zorder=5)
            low_plot_log = np.where(total_ints_log < 1e5, total_ints_log, np.nan)
            high_plot_log = np.where(total_ints_log >= 1e5, total_ints_log, np.nan)
            ax_overview_log.plot(rts_min_overlay, low_plot_log, color=color_rgb, alpha=0.25, linewidth=1.0, path_effects=overlay_path_effects)
            ax_overview_log.plot(rts_min_overlay, high_plot_log, color=color_rgb, alpha=0.95, linewidth=2.5, label=item['label'], path_effects=overlay_path_effects)
        ax_overview_log.set_ylabel('Intensity (log)', fontsize=13, fontfamily='serif', color='0.85')
        ax_overview_log.set_yscale('log')
        ax_overview_log.set_title('Accepted Peptides – Raw Intensity (log; green = integration, gray = collection)', fontsize=14, fontweight='bold', fontfamily='serif', color='0.9')
        ax_overview_log.set_xlim(left=x_min_ov, right=x_max_ov_plot)
        ax_overview_log.xaxis.set_major_locator(_ML(max(0.5, (x_max_ov_plot - x_min_ov) / 10)))
        ax_overview_log.xaxis.set_minor_locator(_ML(max(0.1, (x_max_ov_plot - x_min_ov) / 30)))
        if y_max_ov > 0:
            ax_overview_log.set_ylim(y_floor, y_max_ov * 2.0)
        ax_overview_log.tick_params(labelsize=11, colors='0.85')
        for label in ax_overview_log.get_xticklabels() + ax_overview_log.get_yticklabels():
            label.set_fontfamily('serif')
        ax_overview_log.grid(True, alpha=0.25, which='both')
        for spine in ax_overview_log.spines.values():
            spine.set_color('0.6')
        ax_overview_log.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8, prop={'family': 'serif'}, ncol=1, labelcolor='0.9')
        # Third panel: track-offset (ridgeline) — per-peptide normalized to 0–1 for shape comparison (raw intensity in linear/log above)
        track_height = 2.0
        track_gap = 0.25
        n_tracks = len(overlay_data)
        label_pad_min = 1.2  # minutes left of x_min_ov for row labels
        for i, item in enumerate(overlay_data):
            rts_sec = item['rts']
            rts_min = np.asarray(rts_sec, dtype=float) / 60.0
            total_ints = np.asarray(item['total_intensities'], dtype=float)
            if len(rts_min) < 2 or len(total_ints) < 2:
                continue
            imax = float(np.nanmax(total_ints)) if len(total_ints) > 0 else 1.0
            if imax <= 0 or np.isnan(imax):
                imax = 1.0
            I_norm = total_ints / imax
            y_offset = i * (track_height + track_gap)
            y_plot = I_norm * track_height + y_offset
            ta = item.get('total_area')
            if area_norm_overview is not None and ta is not None and not (isinstance(ta, float) and (np.isnan(ta) or ta < 0)):
                color = cmap_overview(area_norm_overview(ta))
            else:
                color = (0.5, 0.5, 0.5, 0.9)
            ax_overview_tracks.fill_between(rts_min, y_offset, y_plot, color=color, alpha=0.95)
            ax_overview_tracks.plot(rts_min, y_plot, color=color, linewidth=1.5, alpha=1.0)
            # Row label (no legend): peptide identity by vertical position
            lab = item.get('label') or f'#{i+1}'
            if len(str(lab)) > 22:
                lab = str(lab)[:19] + '…'
            ax_overview_tracks.text(x_min_tracks - label_pad_min, y_offset + track_height * 0.5, lab,
                                    va='center', ha='right', fontsize=7, fontfamily='serif')
        # RT windows on tracks panel (green/gray)
        for item in overlay_data:
            min_rt_m = item.get('min_rt')
            max_rt_m = item.get('max_rt')
            cmin = item.get('collection_min_rt')
            cmax = item.get('collection_max_rt')
            if min_rt_m is not None and max_rt_m is not None:
                ax_overview_tracks.axvspan(float(min_rt_m) / 60.0, float(max_rt_m) / 60.0, alpha=0.06, color='green', zorder=0)
            if cmin is not None and cmax is not None:
                ax_overview_tracks.axvspan(float(cmin) / 60.0, float(cmax) / 60.0, alpha=0.04, color=COLLECTION_WINDOW_COLOR, zorder=0)
        ax_overview_tracks.set_xlabel('MS1 Retention Time (min)', fontsize=13, fontfamily='serif', color='0.85')
        ax_overview_tracks.set_ylabel('Track (shape norm.)', fontsize=11, fontfamily='serif', color='0.85')
        ax_overview_tracks.set_title('Accepted Peptides – Tracks (shape only: each trace max=1; x zoomed to integration windows)', fontsize=14, fontweight='bold', fontfamily='serif', color='0.9')
        ax_overview_tracks.set_xlim(left=x_min_tracks - label_pad_min - 0.5, right=x_max_tracks + 0.3)
        _span_t = (x_max_tracks + 0.3) - (x_min_tracks - label_pad_min - 0.5)
        ax_overview_tracks.xaxis.set_major_locator(_ML(max(0.5, _span_t / 10)))
        ax_overview_tracks.xaxis.set_minor_locator(_ML(max(0.1, _span_t / 30)))
        ax_overview_tracks.set_ylim(-0.3, n_tracks * (track_height + track_gap) - track_gap + 0.3)
        ax_overview_tracks.tick_params(labelsize=11, colors='0.85')
        for label in ax_overview_tracks.get_xticklabels() + ax_overview_tracks.get_yticklabels():
            label.set_fontfamily('serif')
        ax_overview_tracks.set_yticks([])
        ax_overview_tracks.tick_params(axis='y', left=False)
        ax_overview_tracks.grid(True, alpha=0.25, axis='x')
        for spine in ax_overview_tracks.spines.values():
            spine.set_color('0.6')
        # Track row labels need light color on black
        for txt in ax_overview_tracks.texts:
            txt.set_color('0.9')
        if area_norm_overview is not None:
            sm_overview = ScalarMappable(norm=area_norm_overview, cmap=cmap_overview)
            sm_overview.set_array([])
            cbar_overview = fig_overview.colorbar(sm_overview, ax=[ax_overview, ax_overview_log, ax_overview_tracks], shrink=0.5, aspect=25, pad=0.08)
            cbar_overview.ax.set_facecolor('black')
            cbar_overview.set_label('total_area (MS1 peak)', fontsize=12, fontweight='bold', fontfamily='serif', color='0.9')
            cbar_overview.ax.tick_params(labelsize=10, colors='0.85')
            for label in cbar_overview.ax.get_xticklabels() + cbar_overview.ax.get_yticklabels():
                label.set_fontfamily('serif')
                label.set_color('0.85')
        fig_overview.tight_layout(rect=[0, 0, 0.82, 1], pad=1.2)
        overview_path = os.path.join(output_dir, 'all_peptides_overlay.png')
        fig_overview.savefig(overview_path, dpi=150, pad_inches=0.25, facecolor='black')
        plt.close(fig_overview)
        print(f"[DEBUG] Overview overlay plot (linear + log + track subplots) saved to: {os.path.abspath(overview_path)}")

        # Standalone A: big tracks-only plot (full-page, legible row labels) — always square, much taller
        track_h = 2.0
        track_gap = 0.25
        n_tr = len(overlay_data)
        fig_h_tracks = max(28, 1.0 * n_tr)
        sq = max(28, fig_h_tracks)
        fig_tracks, ax_tracks = plt.subplots(figsize=(sq, sq))
        fig_tracks.patch.set_facecolor('black')
        ax_tracks.set_facecolor('black')
        label_pad = 1.2
        # RT windows (green = integration, gray = collection) — draw first
        for item in overlay_data:
            min_rt_m = item.get('min_rt')
            max_rt_m = item.get('max_rt')
            cmin = item.get('collection_min_rt')
            cmax = item.get('collection_max_rt')
            if min_rt_m is not None and max_rt_m is not None:
                ax_tracks.axvspan(float(min_rt_m) / 60.0, float(max_rt_m) / 60.0, alpha=0.06, color='green', zorder=0)
            if cmin is not None and cmax is not None:
                ax_tracks.axvspan(float(cmin) / 60.0, float(cmax) / 60.0, alpha=0.04, color=COLLECTION_WINDOW_COLOR, zorder=0)
        for i, item in enumerate(overlay_data):
            rts_min = np.asarray(item['rts'], dtype=float) / 60.0
            I = np.asarray(item['total_intensities'], dtype=float)
            # No gap-breaking: keep x-axis aligned with actual retention time (same as overlay and skyline)
            I = np.clip(I, 0, None)
            imax = float(np.nanmax(I)) if len(I) > 0 else 1.0
            if imax <= 0 or np.isnan(imax):
                imax = 1.0
            In = I / imax
            y0 = i * (track_h + track_gap)
            ta = item.get('total_area')
            if area_norm_overview is not None and ta is not None and not (isinstance(ta, float) and (np.isnan(ta) or ta < 0)):
                color = cmap_overview(area_norm_overview(ta))
            else:
                color = (0.5, 0.5, 0.5, 0.9)
            ax_tracks.fill_between(rts_min, y0, In * track_h + y0, color=color, alpha=0.95)
            ax_tracks.plot(rts_min, In * track_h + y0, color=color, linewidth=1.5, alpha=1.0)
            lab = item.get('label') or f'#{i+1}'
            if len(str(lab)) > 28:
                lab = str(lab)[:25] + '…'
            ax_tracks.text(x_min_tracks - label_pad, y0 + track_h * 0.5, lab, va='center', ha='right', fontsize=8, fontfamily='serif', color='0.9')
        ax_tracks.set_xlabel('MS1 Retention Time (min)', fontsize=13, fontfamily='serif', color='0.85')
        ax_tracks.set_ylabel('Track (shape norm.)', fontsize=11, fontfamily='serif', color='0.85')
        ax_tracks.set_title('Accepted Peptides – Tracks (per-peptide normalized; x zoomed to integration windows)', fontsize=14, fontweight='bold', fontfamily='serif', color='0.9')
        ax_tracks.set_xlim(x_min_tracks - label_pad - 0.5, x_max_tracks + 0.3)
        ax_tracks.set_ylim(-0.3, n_tr * (track_h + track_gap) - track_gap + 0.3)
        ax_tracks.set_yticks([])
        ax_tracks.tick_params(axis='y', left=False)
        ax_tracks.tick_params(axis='x', labelsize=11, colors='0.85')
        for label in ax_tracks.get_xticklabels():
            label.set_fontfamily('serif')
        ax_tracks.grid(True, alpha=0.25, axis='x')
        for spine in ax_tracks.spines.values():
            spine.set_color('0.6')
        fig_tracks.tight_layout()
        tracks_only_path = os.path.join(output_dir, 'all_peptides_overlay_tracks_only.png')
        fig_tracks.savefig(tracks_only_path, dpi=150, bbox_inches='tight', facecolor='black')
        plt.close(fig_tracks)
        print(f"[DEBUG] Standalone tracks plot saved to: {os.path.abspath(tracks_only_path)}")

        # Standalone B: Raw + log intensity overlay with designated windows, colored by total area
        # Black background; peaks (axvspan) very low opacity; lines less opaque below 10^5, thicker/more opaque above 10^5
        INTENSITY_HIGH_THRESH = 1e5
        fig_sky, (ax_sky_lin, ax_sky_log) = plt.subplots(2, 1, figsize=(20, 40), sharex=True)
        fig_sky.patch.set_facecolor('black')
        ax_sky_lin.set_facecolor('black')
        ax_sky_log.set_facecolor('black')
        y_floor_log = 1.0  # avoid log(0)
        y_max_sky = 0.0
        for item in overlay_data:
            rts_min = np.asarray(item['rts'], dtype=float) / 60.0
            I = np.asarray(item['total_intensities'], dtype=float)
            I = np.clip(I, 0, None)
            if len(I) > 0 and np.nanmax(I) > 0:
                y_max_sky = max(y_max_sky, float(np.nanmax(I)))
            if len(rts_min) < 2 or len(I) < 2:
                continue
            ta = item.get('total_area')
            if area_norm_overview is not None and ta is not None and not (isinstance(ta, float) and (np.isnan(ta) or ta < 0)):
                color = cmap_overview(area_norm_overview(ta))
            else:
                color = (0.5, 0.5, 0.5, 0.7)
            color_rgb = color[:3] if len(color) >= 3 else color
            # Designated windows (integration = green, collection = gray) — very low opacity so peaks are less prominent than lines
            min_rt_min = item.get('min_rt')
            max_rt_min = item.get('max_rt')
            coll_min = item.get('collection_min_rt')
            coll_max = item.get('collection_max_rt')
            if min_rt_min is not None and max_rt_min is not None:
                min_rt_min = float(min_rt_min) / 60.0
                max_rt_min = float(max_rt_min) / 60.0
                ax_sky_lin.axvspan(min_rt_min, max_rt_min, alpha=0.06, color='green', zorder=0)
                ax_sky_log.axvspan(min_rt_min, max_rt_min, alpha=0.06, color='green', zorder=0)
            if coll_min is not None and coll_max is not None:
                coll_min = float(coll_min) / 60.0
                coll_max = float(coll_max) / 60.0
                ax_sky_lin.axvspan(coll_min, coll_max, alpha=0.04, color=COLLECTION_WINDOW_COLOR, zorder=0)
                ax_sky_log.axvspan(coll_min, coll_max, alpha=0.04, color=COLLECTION_WINDOW_COLOR, zorder=0)
            # Semi-opaque shaded fill under traces; fade below 10^5
            low_I = np.where(I < 1e5, I, 0)
            high_I = np.where(I >= 1e5, I, 0)
            ax_sky_lin.fill_between(rts_min, 0, low_I, color=color_rgb, alpha=0.12, zorder=5)
            ax_sky_lin.fill_between(rts_min, 0, high_I, color=color_rgb, alpha=0.35, zorder=5)
            I_log = np.maximum(I, y_floor_log)
            low_I_log = np.where(I_log < 1e5, I_log, y_floor_log)
            high_I_log = np.where(I_log >= 1e5, I_log, y_floor_log)
            ax_sky_log.fill_between(rts_min, y_floor_log, low_I_log, color=color_rgb, alpha=0.12, zorder=5)
            ax_sky_log.fill_between(rts_min, y_floor_log, high_I_log, color=color_rgb, alpha=0.35, zorder=5)
            # Full trace: low opacity, thin (emphasizes region below 10^5 as faint)
            ax_sky_lin.plot(rts_min, I, alpha=0.12, linewidth=1.0, color=color_rgb, zorder=6)
            ax_sky_log.plot(rts_min, I_log, alpha=0.12, linewidth=1.0, color=color_rgb, zorder=6)
            # Contiguous segments where I >= 10^5: thicker and more opaque
            above = (I >= INTENSITY_HIGH_THRESH).astype(np.int8)
            if np.any(above):
                edges = np.diff(np.concatenate([[0], above, [0]]))
                starts = np.where(edges == 1)[0]
                ends = np.where(edges == -1)[0]
                for s, e in zip(starts, ends):
                    if e > s:
                        sl = slice(s, e)  # e is first index after run in diff, so run is [s, e-1]
                        ax_sky_lin.plot(rts_min[sl], I[sl], alpha=0.75, linewidth=2.5, color=color_rgb, zorder=6)
                        ax_sky_log.plot(rts_min[sl], I_log[sl], alpha=0.75, linewidth=2.5, color=color_rgb, zorder=6)
        ax_sky_lin.set_ylabel('Intensity', fontsize=13, fontfamily='serif', color='0.85')
        ax_sky_lin.set_title('Accepted Peptides – Raw Intensity Overlay (green = integration window, gray = collection)', fontsize=14, fontweight='bold', fontfamily='serif', color='0.9')
        ax_sky_lin.set_xlim(x_min_ov, x_max_ov_plot)
        if y_max_sky > 0:
            ax_sky_lin.set_ylim(0, y_max_sky * 1.1)
        ax_sky_lin.grid(True, alpha=0.25)
        ax_sky_lin.tick_params(labelsize=11, colors='0.85')
        ax_sky_lin.spines['bottom'].set_color('0.6')
        ax_sky_lin.spines['top'].set_color('0.6')
        ax_sky_lin.spines['left'].set_color('0.6')
        ax_sky_lin.spines['right'].set_color('0.6')
        ax_sky_log.set_yscale('log')
        ax_sky_log.set_ylabel('Intensity (log)', fontsize=13, fontfamily='serif', color='0.85')
        ax_sky_log.set_xlabel('MS1 Retention Time (min)', fontsize=13, fontfamily='serif', color='0.85')
        ax_sky_log.set_title('Accepted Peptides – Raw Intensity Overlay (log scale)', fontsize=14, fontweight='bold', fontfamily='serif', color='0.9')
        ax_sky_log.set_xlim(x_min_ov, x_max_ov_plot)
        if y_max_sky > 0:
            ax_sky_log.set_ylim(y_floor_log, y_max_sky * 2.0)
        ax_sky_log.grid(True, alpha=0.25, which='both')
        ax_sky_log.tick_params(labelsize=11, colors='0.85')
        ax_sky_log.spines['bottom'].set_color('0.6')
        ax_sky_log.spines['top'].set_color('0.6')
        ax_sky_log.spines['left'].set_color('0.6')
        ax_sky_log.spines['right'].set_color('0.6')
        for ax in (ax_sky_lin, ax_sky_log):
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontfamily('serif')
        if area_norm_overview is not None:
            sm_sky = ScalarMappable(norm=area_norm_overview, cmap=cmap_overview)
            sm_sky.set_array([])
            cbar_sky = fig_sky.colorbar(sm_sky, ax=[ax_sky_lin, ax_sky_log], shrink=0.6, aspect=25, pad=0.08)
            cbar_sky.ax.set_facecolor('black')
            cbar_sky.set_label('total_area (MS1 peak)', fontsize=12, fontweight='bold', fontfamily='serif', color='0.9')
            cbar_sky.ax.tick_params(labelsize=10, colors='0.85')
            for label in cbar_sky.ax.get_xticklabels() + cbar_sky.ax.get_yticklabels():
                label.set_color('0.85')
        fig_sky.tight_layout()
        skyline_path = os.path.join(output_dir, 'all_peptides_overlay_skyline_style.png')
        fig_sky.savefig(skyline_path, dpi=150, bbox_inches='tight', facecolor='black')
        plt.close(fig_sky)
        print(f"[DEBUG] Raw + log overlay with windows (colored by total area) saved to: {os.path.abspath(skyline_path)}")

        # Standalone A/B/C: same plots but only peptides with >1% relative area
        if len(overlay_data_1pct) > 0:
            # Track bounds for 1pct subset (zoom to their windows)
            wins_min_1pct = [e['min_rt'] for e in overlay_data_1pct if e.get('min_rt') is not None]
            wins_max_1pct = [e['max_rt'] for e in overlay_data_1pct if e.get('max_rt') is not None]
            if wins_min_1pct and wins_max_1pct:
                active_min_1pct = float(min(wins_min_1pct))
                active_max_1pct = float(max(wins_max_1pct))
                buf_1pct = max(0.5, (active_max_1pct - active_min_1pct) / 60.0 * 0.15)
                x_min_tracks_1pct = max(0.0, active_min_1pct / 60.0 - buf_1pct)
                x_max_tracks_1pct = active_max_1pct / 60.0 + buf_1pct
            else:
                x_min_tracks_1pct, x_max_tracks_1pct = x_min_tracks, x_max_tracks

            n_tr_1pct = len(overlay_data_1pct)
            fig_h_tracks_1pct = max(28, 1.0 * n_tr_1pct)
            sq_1pct = max(28, fig_h_tracks_1pct)
            fig_tracks_1pct, ax_tracks_1pct = plt.subplots(figsize=(sq_1pct, sq_1pct))
            fig_tracks_1pct.patch.set_facecolor('black')
            ax_tracks_1pct.set_facecolor('black')
            for item in overlay_data_1pct:
                min_rt_m = item.get('min_rt')
                max_rt_m = item.get('max_rt')
                cmin = item.get('collection_min_rt')
                cmax = item.get('collection_max_rt')
                if min_rt_m is not None and max_rt_m is not None:
                    ax_tracks_1pct.axvspan(float(min_rt_m) / 60.0, float(max_rt_m) / 60.0, alpha=0.06, color='green', zorder=0)
                if cmin is not None and cmax is not None:
                    ax_tracks_1pct.axvspan(float(cmin) / 60.0, float(cmax) / 60.0, alpha=0.04, color=COLLECTION_WINDOW_COLOR, zorder=0)
            for i, item in enumerate(overlay_data_1pct):
                rts_min = np.asarray(item['rts'], dtype=float) / 60.0
                I = np.asarray(item['total_intensities'], dtype=float)
                I = np.clip(I, 0, None)
                imax = float(np.nanmax(I)) if len(I) > 0 else 1.0
                if imax <= 0 or np.isnan(imax):
                    imax = 1.0
                In = I / imax
                y0 = i * (track_h + track_gap)
                ta = item.get('total_area')
                if area_norm_overview is not None and ta is not None and not (isinstance(ta, float) and (np.isnan(ta) or ta < 0)):
                    color = cmap_overview(area_norm_overview(ta))
                else:
                    color = (0.5, 0.5, 0.5, 0.9)
                ax_tracks_1pct.fill_between(rts_min, y0, In * track_h + y0, color=color, alpha=0.95)
                ax_tracks_1pct.plot(rts_min, In * track_h + y0, color=color, linewidth=1.5, alpha=1.0)
                lab = item.get('label') or f'#{i+1}'
                if len(str(lab)) > 28:
                    lab = str(lab)[:25] + '…'
                ax_tracks_1pct.text(x_min_tracks_1pct - label_pad, y0 + track_h * 0.5, lab, va='center', ha='right', fontsize=8, fontfamily='serif', color='0.9')
            ax_tracks_1pct.set_xlabel('MS1 Retention Time (min)', fontsize=13, fontfamily='serif', color='0.85')
            ax_tracks_1pct.set_ylabel('Track (shape norm.)', fontsize=11, fontfamily='serif', color='0.85')
            ax_tracks_1pct.set_title(f'Peptides >1% relative area (n={n_tr_1pct}) – Tracks', fontsize=14, fontweight='bold', fontfamily='serif', color='0.9')
            ax_tracks_1pct.set_xlim(x_min_tracks_1pct - label_pad - 0.5, x_max_tracks_1pct + 0.3)
            ax_tracks_1pct.set_ylim(-0.3, n_tr_1pct * (track_h + track_gap) - track_gap + 0.3)
            ax_tracks_1pct.set_yticks([])
            ax_tracks_1pct.tick_params(axis='y', left=False)
            ax_tracks_1pct.tick_params(axis='x', labelsize=11, colors='0.85')
            for label in ax_tracks_1pct.get_xticklabels():
                label.set_fontfamily('serif')
            ax_tracks_1pct.grid(True, alpha=0.25, axis='x')
            for spine in ax_tracks_1pct.spines.values():
                spine.set_color('0.6')
            fig_tracks_1pct.tight_layout()
            tracks_only_1pct_path = os.path.join(output_dir, 'all_peptides_overlay_tracks_only_1pct.png')
            fig_tracks_1pct.savefig(tracks_only_1pct_path, dpi=150, bbox_inches='tight', facecolor='black')
            plt.close(fig_tracks_1pct)
            print(f"[DEBUG] Tracks ≥1% rel area saved to: {os.path.abspath(tracks_only_1pct_path)}")

    # Zoom chromatogram figures: one figure per RT segment (split at coverage gaps), all traces overlaid, colored by total_area (same scale as RT windows)
    # Skip when skip_zoom_segments=True (e.g. plot regeneration script: use individual chromatograms only, new layout)
    skip_zoom_segments = getattr(args, 'skip_zoom_segments', False)
    if len(chrom_zoom_data) > 0 and not skip_zoom_segments:
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
        # Use same global total_area range as individual figures for consistent colors
        if area_vmax_global > area_vmin_global:
            area_norm_zoom = Normalize(vmin=area_vmin_global, vmax=area_vmax_global)
        else:
            valid_areas_zoom = [e['total_area'] for e in chrom_zoom_data if e.get('total_area') is not None and not (isinstance(e['total_area'], float) and (np.isnan(e['total_area']) or e['total_area'] < 0))]
            if valid_areas_zoom:
                area_vmin_z = min(valid_areas_zoom)
                area_vmax_z = max(valid_areas_zoom)
                if area_vmax_z <= area_vmin_z:
                    area_vmax_z = area_vmin_z + 1.0
                area_norm_zoom = Normalize(vmin=area_vmin_z, vmax=area_vmax_z)
            else:
                area_norm_zoom = None
        cmap_zoom = _total_area_colormap_early()
        try:
            from matplotlib.patheffects import withStroke
            zoom_path_effects = [withStroke(linewidth=1.5, foreground='black')]
        except Exception:
            zoom_path_effects = []
        base_zoom = output_png.replace('.png', '')
        results_dir = os.path.dirname(os.path.abspath(output_png))
        # Sort by min_rt (ascending) so segments match RT windows plot
        chrom_zoom_data = sorted(chrom_zoom_data, key=lambda e: e['min_rt'])
        for i, entry in enumerate(chrom_zoom_data):
            entry['peptide_index_1based'] = i + 1
        # Split at RT coverage gaps (same logic as combined): use actual integration + collection extent per entry
        def _extent_lo(e):
            lo = e['min_rt']
            if e.get('collection_min_rt') is not None:
                lo = min(lo, e['collection_min_rt'])
            return lo
        def _extent_hi(e):
            hi = e['max_rt']
            if e.get('collection_max_rt') is not None:
                hi = max(hi, e['collection_max_rt'])
            return hi
        intervals_sorted = sorted([(_extent_lo(e), _extent_hi(e)) for e in chrom_zoom_data], key=lambda x: x[0])
        merged = []
        for a, b in intervals_sorted:
            if merged and a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        def _assign_segment(e):
            lo, hi = _extent_lo(e), _extent_hi(e)
            for seg_idx, (m_lo, m_hi) in enumerate(merged):
                if lo <= m_hi and hi >= m_lo:
                    return seg_idx
            return 0
        segment_lists = {}
        for i, entry in enumerate(chrom_zoom_data):
            s = _assign_segment(entry)
            if s not in segment_lists:
                segment_lists[s] = []
            segment_lists[s].append(entry)
        zoom_chunks = [segment_lists[k] for k in sorted(segment_lists.keys())]
        # Discover RT windows zoom images by segment (combined saves as zoom_peptides_segment_N_RT_*s-*s.png)
        import glob
        rt_windows_zoom_by_segment = {}
        zoom_glob = os.path.join(results_dir, 'filtered_rt_windows_by_peptide_zoom_peptides_segment_*.png')

        def _segment_num(path):
            m = re.search(r'segment_(\d+)', os.path.basename(path))
            return int(m.group(1)) if m else 0

        for p in sorted(glob.glob(zoom_glob), key=_segment_num):
            m = re.search(r'segment_(\d+)', os.path.basename(p))
            if m:
                rt_windows_zoom_by_segment[int(m.group(1))] = p
        for chunk_idx, chunk in enumerate(zoom_chunks):
            n_sub = len(chunk)
            # Use actual integration window (min_rt, max_rt) and extend to collection window so zoom matches RT windows plot
            seg_start_sec = min(e['min_rt'] for e in chunk)
            seg_end_sec = max(e['max_rt'] for e in chunk)
            for e in chunk:
                if e.get('collection_min_rt') is not None:
                    seg_start_sec = min(seg_start_sec, e['collection_min_rt'])
                if e.get('collection_max_rt') is not None:
                    seg_end_sec = max(seg_end_sec, e['collection_max_rt'])
            rt_span_sec = seg_end_sec - seg_start_sec
            margin_sec = max(30.0, rt_span_sec * 0.1)
            x_min_min = 0.0 if chunk_idx == 0 else (seg_start_sec - margin_sec) / 60.0
            x_max_min = (seg_end_sec + margin_sec) / 60.0
            pep_first = chunk[0]['peptide_index_1based']
            pep_last = chunk[-1]['peptide_index_1based']
            seg_num = chunk_idx + 1
            rt_windows_zoom_path = rt_windows_zoom_by_segment.get(seg_num)
            has_rt_windows = rt_windows_zoom_path is not None and os.path.isfile(rt_windows_zoom_path)
            # Use same x range and units as RT windows plot (seconds) when we have the image, so axes match
            if has_rt_windows:
                left_margin_sec = max(10.0, rt_span_sec * 0.02)
                label_margin_sec = max(90.0, rt_span_sec * 0.5)
                x_min_sec = seg_start_sec - left_margin_sec
                x_max_sec = seg_end_sec + label_margin_sec
            if has_rt_windows:
                fig_zoom, (ax_rt_windows, ax_zoom) = plt.subplots(2, 1, figsize=(14, 20), height_ratios=[1, 2], sharex=True)
                fig_zoom.patch.set_facecolor('black')
                ax_rt_windows.set_facecolor('black')
                ax_zoom.set_facecolor('black')
                # RT windows (green = integration, gray = collection) — draw first, match overlay style
                for entry in chunk:
                    min_rt_m = entry.get('min_rt')
                    max_rt_m = entry.get('max_rt')
                    cmin = entry.get('collection_min_rt')
                    cmax = entry.get('collection_max_rt')
                    if min_rt_m is not None and max_rt_m is not None:
                        ax_zoom.axvspan(float(min_rt_m), float(max_rt_m), alpha=0.06, color='green', zorder=0)
                    if cmin is not None and cmax is not None:
                        ax_zoom.axvspan(float(cmin), float(cmax), alpha=0.04, color=COLLECTION_WINDOW_COLOR, zorder=0)
                try:
                    from matplotlib import image as mpl_image
                    img_rt = mpl_image.imread(rt_windows_zoom_path)
                    # RT windows PNG was drawn with x in seconds (x_min_sec to x_max_sec); use same range so image aligns
                    ax_rt_windows.imshow(img_rt, aspect='auto', extent=[x_min_sec, x_max_sec, 0, 1], origin='upper')
                    ax_rt_windows.set_ylim(0, 1)
                    ax_rt_windows.set_yticks([])
                    ax_rt_windows.set_title('RT Integration Windows', fontsize=10, fontfamily='serif')
                    ax_rt_windows.tick_params(axis='x', labelbottom=False)
                    for spine in ax_rt_windows.spines.values():
                        spine.set_visible(False)
                except Exception:
                    ax_rt_windows.axis('off')
                    ax_rt_windows.text(0.5, 0.5, 'RT windows image not available', ha='center', va='center', transform=ax_rt_windows.transAxes)
            else:
                fig_zoom, ax_zoom = plt.subplots(1, 1, figsize=(14, 12))
                fig_zoom.patch.set_facecolor('black')
                ax_zoom.set_facecolor('black')
                # RT windows (green = integration, gray = collection) — draw first, match overlay style
                for entry in chunk:
                    min_rt_m = entry.get('min_rt')
                    max_rt_m = entry.get('max_rt')
                    cmin = entry.get('collection_min_rt')
                    cmax = entry.get('collection_max_rt')
                    if min_rt_m is not None and max_rt_m is not None:
                        ax_zoom.axvspan(float(min_rt_m) / 60.0, float(max_rt_m) / 60.0, alpha=0.06, color='green', zorder=0)
                    if cmin is not None and cmax is not None:
                        ax_zoom.axvspan(float(cmin) / 60.0, float(cmax) / 60.0, alpha=0.04, color=COLLECTION_WINDOW_COLOR, zorder=0)
            for entry in chunk:
                rts_sec = np.asarray(entry['rts'], dtype=float)
                rts_min_z = np.asarray(entry['rts_min'], dtype=float)
                total_z = np.asarray(entry['total_intensities'], dtype=float)
                # Use raw RT/intensity (no gap-breaking) so x-axis stays aligned with actual retention time
                if has_rt_windows:
                    rt_axis = rts_sec
                    mask = (rt_axis >= x_min_sec) & (rt_axis <= x_max_sec)
                else:
                    rt_axis = rts_min_z
                    mask = (rt_axis >= x_min_min) & (rt_axis <= x_max_min)
                x_plot = rt_axis[mask]
                ta = entry.get('total_area')
                if area_norm_zoom is not None and ta is not None and not (isinstance(ta, float) and (np.isnan(ta) or ta < 0)):
                    color = cmap_zoom(area_norm_zoom(ta))
                else:
                    color = (0.3, 0.3, 0.3, 0.9)
                zoom_legend_label = f"Peptide {entry['peptide_index_1based']}: {entry['label']}"
                total_masked = total_z[mask]
                # Fade below 10^5 (reduce alpha / increase transparency)
                low_z = np.where(total_masked < 1e5, total_masked, np.nan)
                high_z = np.where(total_masked >= 1e5, total_masked, np.nan)
                ax_zoom.plot(x_plot, low_z, color=color, linewidth=1.0, alpha=0.35, path_effects=zoom_path_effects)
                ax_zoom.plot(x_plot, high_z, color=color, linewidth=2.5, alpha=0.95, label=zoom_legend_label, path_effects=zoom_path_effects)
            if has_rt_windows:
                ax_zoom.set_xlim(x_min_sec, x_max_sec)
                ax_rt_windows.set_xlim(x_min_sec, x_max_sec)
                ax_zoom.set_xlabel('Retention Time (s)', fontsize=13, fontfamily='serif', color='0.85')
                ax_zoom_min = ax_zoom.secondary_xaxis('top', functions=(lambda s: s / 60.0, lambda m: m * 60.0))
                ax_zoom_min.set_xlabel('Retention Time (min)', fontsize=12, fontfamily='serif', color='0.85')
                ax_zoom_min.tick_params(colors='0.85')
            else:
                ax_zoom.set_xlim(x_min_min, x_max_min)
                ax_zoom.set_xlabel('MS1 Retention Time (min)', fontsize=13, fontfamily='serif', color='0.85')
            ax_zoom.set_ylabel('Intensity', fontsize=13, fontfamily='serif', color='0.85')
            ax_zoom.tick_params(labelsize=11, colors='0.85')
            for label in ax_zoom.get_xticklabels() + ax_zoom.get_yticklabels():
                label.set_fontfamily('serif')
            ax_zoom.grid(True, alpha=0.25)
            ax_zoom.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=8, prop={'family': 'serif'}, ncol=1, labelcolor='0.9')
            for spine in ax_zoom.spines.values():
                spine.set_color('0.6')
            if area_norm_zoom is not None:
                sm_zoom = ScalarMappable(norm=area_norm_zoom, cmap=cmap_zoom)
                sm_zoom.set_array([])
                cbar_zoom = fig_zoom.colorbar(sm_zoom, ax=ax_zoom, shrink=0.65, aspect=18, pad=0.12)
                cbar_zoom.ax.set_facecolor('black')
                cbar_zoom.set_label('total_area (MS1 peak)', fontsize=12, fontweight='bold', fontfamily='serif', color='0.9')
                cbar_zoom.ax.tick_params(labelsize=10, colors='0.85')
                for label in cbar_zoom.ax.get_xticklabels() + cbar_zoom.ax.get_yticklabels():
                    label.set_color('0.85')
            suptitle_y = 0.98 if has_rt_windows else 1.02
            fig_zoom.suptitle(f'MS1 Chromatograms – Segment {seg_num} (RT {seg_start_sec/60:.2f}-{seg_end_sec/60:.2f} min, peptides {pep_first}-{pep_last})', fontsize=14, fontweight='bold', fontfamily='serif', y=suptitle_y, color='0.9')
            plt.tight_layout(rect=[0, 0, 0.82, 0.96], pad=1.2)
            zoom_path = base_zoom + f'_zoom_peptides_segment_{seg_num}_RT_{seg_start_sec:.0f}s-{seg_end_sec:.0f}s.png'
            plt.savefig(zoom_path, dpi=150, bbox_inches='tight', facecolor='black', pad_inches=0.25)
            plt.close(fig_zoom)
            print(f"[DEBUG] Chromatogram zoom (segment {seg_num}, RT {seg_start_sec:.0f}-{seg_end_sec:.0f} s, peptides {pep_first}-{pep_last}) saved to: {os.path.abspath(zoom_path)}")

        # One non-segmented zoom: all peptides in one figure, three panels (linear, log, tracks)
        all_zoom = chrom_zoom_data  # already sorted by min_rt
        if len(all_zoom) > 0:
            seg_start_all = min(e['min_rt'] for e in all_zoom)
            seg_end_all = max(e['max_rt'] for e in all_zoom)
            for e in all_zoom:
                if e.get('collection_min_rt') is not None:
                    seg_start_all = min(seg_start_all, e['collection_min_rt'])
                if e.get('collection_max_rt') is not None:
                    seg_end_all = max(seg_end_all, e['collection_max_rt'])
            rt_span_all = seg_end_all - seg_start_all
            margin_all = max(30.0, rt_span_all * 0.1)
            x_min_all_min = max(0.0, (seg_start_all - margin_all) / 60.0)
            x_max_all_min = (seg_end_all + margin_all) / 60.0
            from matplotlib.gridspec import GridSpec
            from matplotlib.ticker import MultipleLocator
            fig_zoom_all = plt.figure(figsize=(14, 32))
            fig_zoom_all.patch.set_facecolor('black')
            gs_za = GridSpec(3, 1, figure=fig_zoom_all, height_ratios=[1, 1, 4], hspace=0.3)
            ax_z_lin = fig_zoom_all.add_subplot(gs_za[0])
            ax_z_log = fig_zoom_all.add_subplot(gs_za[1], sharex=ax_z_lin)
            ax_z_tr = fig_zoom_all.add_subplot(gs_za[2], sharex=ax_z_lin)
            for ax in (ax_z_lin, ax_z_log, ax_z_tr):
                ax.set_facecolor('black')
            # RT windows (green = integration, gray = collection) — draw first
            for entry in all_zoom:
                min_rt_m = entry.get('min_rt')
                max_rt_m = entry.get('max_rt')
                cmin = entry.get('collection_min_rt')
                cmax = entry.get('collection_max_rt')
                if min_rt_m is not None and max_rt_m is not None:
                    min_m, max_m = float(min_rt_m) / 60.0, float(max_rt_m) / 60.0
                    ax_z_lin.axvspan(min_m, max_m, alpha=0.06, color='green', zorder=0)
                    ax_z_log.axvspan(min_m, max_m, alpha=0.06, color='green', zorder=0)
                    ax_z_tr.axvspan(min_m, max_m, alpha=0.06, color='green', zorder=0)
                if cmin is not None and cmax is not None:
                    cm, cx = float(cmin) / 60.0, float(cmax) / 60.0
                    ax_z_lin.axvspan(cm, cx, alpha=0.04, color=COLLECTION_WINDOW_COLOR, zorder=0)
                    ax_z_log.axvspan(cm, cx, alpha=0.04, color=COLLECTION_WINDOW_COLOR, zorder=0)
                    ax_z_tr.axvspan(cm, cx, alpha=0.04, color=COLLECTION_WINDOW_COLOR, zorder=0)
            y_max_zoom = 0.0
            for entry in all_zoom:
                rts_min_z = np.asarray(entry['rts_min'], dtype=float)
                total_z = np.asarray(entry['total_intensities'], dtype=float)
                # No gap-breaking: keep RT axis aligned with actual retention time
                mask = (rts_min_z >= x_min_all_min) & (rts_min_z <= x_max_all_min)
                if np.any(mask) and len(total_z) == len(rts_min_z):
                    y_max_zoom = max(y_max_zoom, float(np.nanmax(total_z[mask])))
                ta = entry.get('total_area')
                if area_norm_zoom is not None and ta is not None and not (isinstance(ta, float) and (np.isnan(ta) or ta < 0)):
                    color = cmap_zoom(area_norm_zoom(ta))
                else:
                    color = (0.3, 0.3, 0.3, 0.9)
                color_rgb = color[:3] if len(color) >= 3 else color
                # Lines: lower opacity below 10^5, thicker and more opaque above
                low_z = np.where(total_z < 1e5, total_z, np.nan)
                high_z = np.where(total_z >= 1e5, total_z, np.nan)
                ax_z_lin.plot(rts_min_z, low_z, color=color_rgb, linewidth=1.0, alpha=0.35, zorder=5)
                ax_z_lin.plot(rts_min_z, high_z, color=color_rgb, linewidth=2.5, alpha=0.95, zorder=5)
                total_log = np.maximum(total_z, 1.0)
                low_log = np.where(total_log < 1e5, total_log, np.nan)
                high_log = np.where(total_log >= 1e5, total_log, np.nan)
                ax_z_log.plot(rts_min_z, low_log, color=color_rgb, linewidth=1.0, alpha=0.35, zorder=5)
                ax_z_log.plot(rts_min_z, high_log, color=color_rgb, linewidth=2.5, alpha=0.95, zorder=5)
            ax_z_lin.set_xlim(x_min_all_min, x_max_all_min)
            ax_z_lin.xaxis.set_major_locator(MultipleLocator(max(0.5, (x_max_all_min - x_min_all_min) / 10)))
            ax_z_lin.xaxis.set_minor_locator(MultipleLocator(max(0.1, (x_max_all_min - x_min_all_min) / 30)))
            ax_z_lin.set_ylabel('Intensity', fontsize=11, fontfamily='serif', color='0.85')
            ax_z_lin.set_title('All peptides (linear)', fontsize=12, fontweight='bold', fontfamily='serif', color='0.9')
            if y_max_zoom > 0:
                ax_z_lin.set_ylim(0, y_max_zoom * 1.1)
            ax_z_lin.grid(True, alpha=0.25, which='both')
            ax_z_lin.tick_params(colors='0.85')
            for spine in ax_z_lin.spines.values():
                spine.set_color('0.6')
            ax_z_log.set_yscale('log')
            ax_z_log.set_xlim(x_min_all_min, x_max_all_min)
            ax_z_log.set_ylabel('Intensity (log)', fontsize=11, fontfamily='serif', color='0.85')
            ax_z_log.set_title('All peptides (log scale)', fontsize=12, fontweight='bold', fontfamily='serif', color='0.9')
            if y_max_zoom > 0:
                ax_z_log.set_ylim(1.0, y_max_zoom * 2.0)
            ax_z_log.grid(True, alpha=0.25, which='both')
            ax_z_log.tick_params(colors='0.85')
            for spine in ax_z_log.spines.values():
                spine.set_color('0.6')
            track_h_z = 1.0
            track_gap_z = 0.12
            for i, entry in enumerate(all_zoom):
                rts_min_z = np.asarray(entry['rts_min'], dtype=float)
                total_z = np.asarray(entry['total_intensities'], dtype=float)
                # No gap-breaking: keep RT axis aligned with actual retention time
                imax_z = float(np.nanmax(total_z)) if len(total_z) > 0 else 1.0
                if imax_z <= 0 or np.isnan(imax_z):
                    imax_z = 1.0
                In_z = total_z / imax_z
                y0_z = i * (track_h_z + track_gap_z)
                ta = entry.get('total_area')
                if area_norm_zoom is not None and ta is not None and not (isinstance(ta, float) and (np.isnan(ta) or ta < 0)):
                    color = cmap_zoom(area_norm_zoom(ta))
                else:
                    color = (0.4, 0.4, 0.4, 0.9)
                ax_z_tr.plot(rts_min_z, In_z * track_h_z + y0_z, color=color, linewidth=1.2, alpha=0.9)
                lab_z = entry.get('label') or f'#{i+1}'
                if len(str(lab_z)) > 24:
                    lab_z = str(lab_z)[:21] + '…'
                ax_z_tr.text(x_min_all_min, y0_z + track_h_z * 0.5, lab_z, va='center', ha='right', fontsize=7, fontfamily='serif', color='0.9')
            # Tracks share x with linear/log (same time range in min); labels sit at left edge of data range
            ax_z_tr.set_ylim(-0.2, len(all_zoom) * (track_h_z + track_gap_z) - track_gap_z + 0.2)
            ax_z_tr.set_xlabel('MS1 Retention Time (min)', fontsize=11, fontfamily='serif', color='0.85')
            ax_z_tr.set_ylabel('Track (shape norm.)', fontsize=10, fontfamily='serif', color='0.85')
            ax_z_tr.set_title('All peptides (per-peptide normalized)', fontsize=12, fontweight='bold', fontfamily='serif', color='0.9')
            ax_z_tr.set_yticks([])
            ax_z_tr.tick_params(axis='y', left=False)
            ax_z_tr.tick_params(axis='x', colors='0.85')
            ax_z_tr.grid(True, alpha=0.25, axis='x')
            for spine in ax_z_tr.spines.values():
                spine.set_color('0.6')
            ax_z_lin.tick_params(axis='x', labelbottom=False)
            ax_z_log.tick_params(axis='x', labelbottom=False)
            for ax in (ax_z_lin, ax_z_log, ax_z_tr):
                for label in ax.get_xticklabels() + ax.get_yticklabels():
                    label.set_fontfamily('serif')
            fig_zoom_all.suptitle(f'MS1 Chromatograms – All peptides (no segment split)  RT {seg_start_all/60:.2f}-{seg_end_all/60:.2f} min', fontsize=13, fontweight='bold', fontfamily='serif', y=0.995, color='0.9')
            plt.tight_layout(rect=[0, 0, 1, 0.97], pad=1.0)
            zoom_all_path = base_zoom + '_zoom_peptides_all_linear_log_tracks.png'
            fig_zoom_all.savefig(zoom_all_path, dpi=150, bbox_inches='tight', facecolor='black', pad_inches=0.25)
            plt.close(fig_zoom_all)
            print(f"[DEBUG] Zoom (all peptides, linear + log + tracks) saved to: {os.path.abspath(zoom_all_path)}")

            # Individual high-res large plots for each panel (in addition to the 3x1)
            _dpi_single = 300
            _fs_title = 16
            _fs_axis = 14
            _fs_ticks = 12
            _lw_single = 2.0
            # 1) Linear only
            fig_lin = plt.figure(figsize=(24, 24))
            fig_lin.patch.set_facecolor('black')
            ax_lin = fig_lin.add_subplot(1, 1, 1)
            ax_lin.set_facecolor('black')
            for entry in all_zoom:
                min_rt_m = entry.get('min_rt')
                max_rt_m = entry.get('max_rt')
                cmin = entry.get('collection_min_rt')
                cmax = entry.get('collection_max_rt')
                if min_rt_m is not None and max_rt_m is not None:
                    ax_lin.axvspan(float(min_rt_m) / 60.0, float(max_rt_m) / 60.0, alpha=0.06, color='green', zorder=0)
                if cmin is not None and cmax is not None:
                    ax_lin.axvspan(float(cmin) / 60.0, float(cmax) / 60.0, alpha=0.04, color=COLLECTION_WINDOW_COLOR, zorder=0)
            for entry in all_zoom:
                rts_min_z = np.asarray(entry['rts_min'], dtype=float)
                total_z = np.asarray(entry['total_intensities'], dtype=float)
                # No gap-breaking: keep RT axis aligned with actual retention time
                ta = entry.get('total_area')
                if area_norm_zoom is not None and ta is not None and not (isinstance(ta, float) and (np.isnan(ta) or ta < 0)):
                    color = cmap_zoom(area_norm_zoom(ta))
                else:
                    color = (0.3, 0.3, 0.3, 0.9)
                color_rgb = color[:3] if len(color) >= 3 else color
                low_z = np.where(total_z < 1e5, total_z, np.nan)
                high_z = np.where(total_z >= 1e5, total_z, np.nan)
                ax_lin.plot(rts_min_z, low_z, color=color_rgb, linewidth=1.0, alpha=0.35, zorder=5)
                ax_lin.plot(rts_min_z, high_z, color=color_rgb, linewidth=2.5, alpha=0.95, zorder=5)
            ax_lin.set_xlim(x_min_all_min, x_max_all_min)
            ax_lin.xaxis.set_major_locator(MultipleLocator(max(0.5, (x_max_all_min - x_min_all_min) / 10)))
            ax_lin.xaxis.set_minor_locator(MultipleLocator(max(0.1, (x_max_all_min - x_min_all_min) / 30)))
            ax_lin.set_ylabel('Intensity', fontsize=_fs_axis, fontfamily='serif', color='0.85')
            ax_lin.set_title(f'All peptides (linear)  RT {seg_start_all/60:.2f}-{seg_end_all/60:.2f} min', fontsize=_fs_title, fontweight='bold', fontfamily='serif', color='0.9')
            if y_max_zoom > 0:
                ax_lin.set_ylim(0, y_max_zoom * 1.1)
            ax_lin.grid(True, alpha=0.25)
            ax_lin.tick_params(axis='both', labelsize=_fs_ticks, colors='0.85')
            ax_lin.set_xlabel('MS1 Retention Time (min)', fontsize=_fs_axis, fontfamily='serif', color='0.85')
            for label in ax_lin.get_xticklabels() + ax_lin.get_yticklabels():
                label.set_fontfamily('serif')
            for spine in ax_lin.spines.values():
                spine.set_color('0.6')
            plt.tight_layout(pad=1.2)
            path_lin = base_zoom + '_zoom_peptides_all_linear.png'
            fig_lin.savefig(path_lin, dpi=_dpi_single, bbox_inches='tight', facecolor='black', pad_inches=0.25)
            plt.close(fig_lin)
            print(f"[DEBUG] Zoom all peptides (linear, high-res) saved to: {os.path.abspath(path_lin)}")
            # 2) Log only
            fig_log = plt.figure(figsize=(24, 24))
            fig_log.patch.set_facecolor('black')
            ax_log = fig_log.add_subplot(1, 1, 1)
            ax_log.set_facecolor('black')
            for entry in all_zoom:
                min_rt_m = entry.get('min_rt')
                max_rt_m = entry.get('max_rt')
                cmin = entry.get('collection_min_rt')
                cmax = entry.get('collection_max_rt')
                if min_rt_m is not None and max_rt_m is not None:
                    ax_log.axvspan(float(min_rt_m) / 60.0, float(max_rt_m) / 60.0, alpha=0.06, color='green', zorder=0)
                if cmin is not None and cmax is not None:
                    ax_log.axvspan(float(cmin) / 60.0, float(cmax) / 60.0, alpha=0.04, color=COLLECTION_WINDOW_COLOR, zorder=0)
            for entry in all_zoom:
                rts_min_z = np.asarray(entry['rts_min'], dtype=float)
                total_z = np.asarray(entry['total_intensities'], dtype=float)
                # No gap-breaking: keep RT axis aligned with actual retention time
                total_log = np.maximum(total_z, 1.0)
                ta = entry.get('total_area')
                if area_norm_zoom is not None and ta is not None and not (isinstance(ta, float) and (np.isnan(ta) or ta < 0)):
                    color = cmap_zoom(area_norm_zoom(ta))
                else:
                    color = (0.3, 0.3, 0.3, 0.9)
                color_rgb = color[:3] if len(color) >= 3 else color
                low_log = np.where(total_log < 1e5, total_log, np.nan)
                high_log = np.where(total_log >= 1e5, total_log, np.nan)
                ax_log.plot(rts_min_z, low_log, color=color_rgb, linewidth=1.0, alpha=0.35, zorder=5)
                ax_log.plot(rts_min_z, high_log, color=color_rgb, linewidth=2.5, alpha=0.95, zorder=5)
            ax_log.set_yscale('log')
            ax_log.set_xlim(x_min_all_min, x_max_all_min)
            ax_log.xaxis.set_major_locator(MultipleLocator(max(0.5, (x_max_all_min - x_min_all_min) / 10)))
            ax_log.xaxis.set_minor_locator(MultipleLocator(max(0.1, (x_max_all_min - x_min_all_min) / 30)))
            ax_log.set_ylabel('Intensity (log)', fontsize=_fs_axis, fontfamily='serif', color='0.85')
            ax_log.set_title(f'All peptides (log scale)  RT {seg_start_all/60:.2f}-{seg_end_all/60:.2f} min', fontsize=_fs_title, fontweight='bold', fontfamily='serif', color='0.9')
            if y_max_zoom > 0:
                ax_log.set_ylim(1.0, y_max_zoom * 2.0)
            ax_log.grid(True, alpha=0.25, which='both')
            ax_log.tick_params(axis='both', labelsize=_fs_ticks, colors='0.85')
            ax_log.set_xlabel('MS1 Retention Time (min)', fontsize=_fs_axis, fontfamily='serif', color='0.85')
            for label in ax_log.get_xticklabels() + ax_log.get_yticklabels():
                label.set_fontfamily('serif')
            for spine in ax_log.spines.values():
                spine.set_color('0.6')
            plt.tight_layout(pad=1.2)
            path_log = base_zoom + '_zoom_peptides_all_log.png'
            fig_log.savefig(path_log, dpi=_dpi_single, bbox_inches='tight', facecolor='black', pad_inches=0.25)
            plt.close(fig_log)
            print(f"[DEBUG] Zoom all peptides (log, high-res) saved to: {os.path.abspath(path_log)}")
            # 3) Tracks only (height scales with number of peptides)
            track_h_z = 2.0
            track_gap_z = 0.25
            label_pad_z = (x_max_all_min - x_min_all_min) * 0.08
            n_tr = len(all_zoom)
            fig_tr = plt.figure(figsize=(24, max(28, n_tr * 1.2)))
            fig_tr.patch.set_facecolor('black')
            ax_tr = fig_tr.add_subplot(1, 1, 1)
            ax_tr.set_facecolor('black')
            for entry in all_zoom:
                min_rt_m = entry.get('min_rt')
                max_rt_m = entry.get('max_rt')
                cmin = entry.get('collection_min_rt')
                cmax = entry.get('collection_max_rt')
                if min_rt_m is not None and max_rt_m is not None:
                    ax_tr.axvspan(float(min_rt_m) / 60.0, float(max_rt_m) / 60.0, alpha=0.06, color='green', zorder=0)
                if cmin is not None and cmax is not None:
                    ax_tr.axvspan(float(cmin) / 60.0, float(cmax) / 60.0, alpha=0.04, color=COLLECTION_WINDOW_COLOR, zorder=0)
            for i, entry in enumerate(all_zoom):
                rts_min_z = np.asarray(entry['rts_min'], dtype=float)
                total_z = np.asarray(entry['total_intensities'], dtype=float)
                # No gap-breaking: keep RT axis aligned with actual retention time
                imax_z = float(np.nanmax(total_z)) if len(total_z) > 0 else 1.0
                if imax_z <= 0 or np.isnan(imax_z):
                    imax_z = 1.0
                In_z = total_z / imax_z
                y0_z = i * (track_h_z + track_gap_z)
                ta = entry.get('total_area')
                if area_norm_zoom is not None and ta is not None and not (isinstance(ta, float) and (np.isnan(ta) or ta < 0)):
                    color = cmap_zoom(area_norm_zoom(ta))
                else:
                    color = (0.4, 0.4, 0.4, 0.9)
                ax_tr.plot(rts_min_z, In_z * track_h_z + y0_z, color=color, linewidth=1.2, alpha=0.9)
                lab_z = entry.get('label') or f'#{i+1}'
                if len(str(lab_z)) > 32:
                    lab_z = str(lab_z)[:29] + '…'
                ax_tr.text(x_min_all_min - label_pad_z, y0_z + track_h_z * 0.5, lab_z, va='center', ha='right', fontsize=10, fontfamily='serif', color='0.9')
            ax_tr.set_xlim(x_min_all_min - label_pad_z - 0.2, x_max_all_min + 0.2)
            ax_tr.xaxis.set_major_locator(MultipleLocator(max(0.5, (x_max_all_min - x_min_all_min) / 10)))
            ax_tr.xaxis.set_minor_locator(MultipleLocator(max(0.1, (x_max_all_min - x_min_all_min) / 30)))
            ax_tr.set_ylim(-0.2, n_tr * (track_h_z + track_gap_z) - track_gap_z + 0.2)
            ax_tr.set_xlabel('MS1 Retention Time (min)', fontsize=_fs_axis, fontfamily='serif', color='0.85')
            ax_tr.set_ylabel('Track (shape norm.)', fontsize=_fs_axis, fontfamily='serif', color='0.85')
            ax_tr.set_title(f'All peptides (per-peptide normalized)  RT {seg_start_all/60:.2f}-{seg_end_all/60:.2f} min', fontsize=_fs_title, fontweight='bold', fontfamily='serif', color='0.9')
            ax_tr.set_yticks([])
            ax_tr.tick_params(axis='x', labelsize=_fs_ticks, colors='0.85')
            ax_tr.grid(True, alpha=0.25, axis='x')
            for label in ax_tr.get_xticklabels():
                label.set_fontfamily('serif')
            for spine in ax_tr.spines.values():
                spine.set_color('0.6')
            plt.tight_layout(pad=1.2)
            path_tr = base_zoom + '_zoom_peptides_all_tracks.png'
            fig_tr.savefig(path_tr, dpi=_dpi_single, bbox_inches='tight', facecolor='black', pad_inches=0.25)
            plt.close(fig_tr)
            print(f"[DEBUG] Zoom all peptides (tracks, high-res) saved to: {os.path.abspath(path_tr)}")

            # Tracks only: peaks with >=1% relative area
            _sum_ta = sum(
                float(e.get('total_area') or 0) for e in all_zoom
                if e.get('total_area') is not None and not (isinstance(e.get('total_area'), float) and (np.isnan(e.get('total_area')) or e.get('total_area') < 0))
            )
            all_zoom_1pct = [
                e for e in all_zoom
                if _sum_ta > 0 and (float(e.get('total_area') or 0) / _sum_ta) >= 0.01
            ]
            if len(all_zoom_1pct) > 0:
                n_tr_1pct = len(all_zoom_1pct)
                fig_tr_1pct = plt.figure(figsize=(24, max(28, n_tr_1pct * 1.2)))
                fig_tr_1pct.patch.set_facecolor('black')
                ax_tr_1pct = fig_tr_1pct.add_subplot(1, 1, 1)
                ax_tr_1pct.set_facecolor('black')
                for entry in all_zoom_1pct:
                    min_rt_m = entry.get('min_rt')
                    max_rt_m = entry.get('max_rt')
                    cmin = entry.get('collection_min_rt')
                    cmax = entry.get('collection_max_rt')
                    if min_rt_m is not None and max_rt_m is not None:
                        ax_tr_1pct.axvspan(float(min_rt_m) / 60.0, float(max_rt_m) / 60.0, alpha=0.06, color='green', zorder=0)
                    if cmin is not None and cmax is not None:
                        ax_tr_1pct.axvspan(float(cmin) / 60.0, float(cmax) / 60.0, alpha=0.04, color=COLLECTION_WINDOW_COLOR, zorder=0)
                for i, entry in enumerate(all_zoom_1pct):
                    rts_min_z = np.asarray(entry['rts_min'], dtype=float)
                    total_z = np.asarray(entry['total_intensities'], dtype=float)
                    imax_z = float(np.nanmax(total_z)) if len(total_z) > 0 else 1.0
                    if imax_z <= 0 or np.isnan(imax_z):
                        imax_z = 1.0
                    In_z = total_z / imax_z
                    y0_z = i * (track_h_z + track_gap_z)
                    ta = entry.get('total_area')
                    if area_norm_zoom is not None and ta is not None and not (isinstance(ta, float) and (np.isnan(ta) or ta < 0)):
                        color = cmap_zoom(area_norm_zoom(ta))
                    else:
                        color = (0.4, 0.4, 0.4, 0.9)
                    ax_tr_1pct.plot(rts_min_z, In_z * track_h_z + y0_z, color=color, linewidth=1.2, alpha=0.9)
                    lab_z = entry.get('label') or f'#{i+1}'
                    if len(str(lab_z)) > 32:
                        lab_z = str(lab_z)[:29] + '…'
                    ax_tr_1pct.text(x_min_all_min - label_pad_z, y0_z + track_h_z * 0.5, lab_z, va='center', ha='right', fontsize=10, fontfamily='serif', color='0.9')
                ax_tr_1pct.set_xlim(x_min_all_min - label_pad_z - 0.2, x_max_all_min + 0.2)
                ax_tr_1pct.xaxis.set_major_locator(MultipleLocator(max(0.5, (x_max_all_min - x_min_all_min) / 10)))
                ax_tr_1pct.xaxis.set_minor_locator(MultipleLocator(max(0.1, (x_max_all_min - x_min_all_min) / 30)))
                ax_tr_1pct.set_ylim(-0.2, n_tr_1pct * (track_h_z + track_gap_z) - track_gap_z + 0.2)
                ax_tr_1pct.set_xlabel('MS1 Retention Time (min)', fontsize=_fs_axis, fontfamily='serif', color='0.85')
                ax_tr_1pct.set_ylabel('Track (shape norm.)', fontsize=_fs_axis, fontfamily='serif', color='0.85')
                ax_tr_1pct.set_title(f'Peptides ≥1% relative area (n={n_tr_1pct})  RT {seg_start_all/60:.2f}-{seg_end_all/60:.2f} min', fontsize=_fs_title, fontweight='bold', fontfamily='serif', color='0.9')
                ax_tr_1pct.set_yticks([])
                ax_tr_1pct.tick_params(axis='x', labelsize=_fs_ticks, colors='0.85')
                ax_tr_1pct.grid(True, alpha=0.25, axis='x')
                for label in ax_tr_1pct.get_xticklabels():
                    label.set_fontfamily('serif')
                for spine in ax_tr_1pct.spines.values():
                    spine.set_color('0.6')
                plt.tight_layout(pad=1.2)
                path_tr_1pct = base_zoom + '_zoom_peptides_all_tracks_1pct.png'
                fig_tr_1pct.savefig(path_tr_1pct, dpi=_dpi_single, bbox_inches='tight', facecolor='black', pad_inches=0.25)
                plt.close(fig_tr_1pct)
                print(f"[DEBUG] Zoom peptides ≥1% rel area (tracks) saved to: {os.path.abspath(path_tr_1pct)}")

    # Hide unused subplots (slots from picked_idx to end; only picked peptides get panels, no gaps)
    for file_idx, fig_combined in enumerate(combined_figures):
        axes_combined_chrom_list, axes_combined_spec_list = combined_axes_list[file_idx]
        for local_idx in range(len(axes_combined_chrom_list)):
            global_slot = file_idx * plots_per_file + local_idx
            if global_slot >= picked_idx:
                axes_combined_chrom_list[local_idx].axis('off')
                axes_combined_spec_list[local_idx].axis('off')
        
        # Add E-value colorbar to combined figure (shared across all subplots)
        # Use the same colormap and normalization as individual figures
        from matplotlib.cm import ScalarMappable
        sm_combined = ScalarMappable(cmap=evalue_cmap, norm=evalue_norm)
        sm_combined.set_array([])  # Empty array, we just want the colormap
        
        # Add colorbar positioned on the left side of the figure (larger for legibility)
        # Position it so it doesn't overlap with subplots (adjust based on figure layout)
        cbar_ax_combined = fig_combined.add_axes([0.01, 0.65, 0.018, 0.26])  # [left, bottom, width, height] in figure coordinates
        cbar_ax_combined.set_facecolor('black')
        cbar_combined = fig_combined.colorbar(sm_combined, cax=cbar_ax_combined, orientation='vertical')
        cbar_combined.set_label('E-value\n(OpenMS RT)', fontsize=10, fontfamily='serif', rotation=0, labelpad=10, color='0.85')
        cbar_combined.ax.tick_params(labelsize=9, colors='0.85')
        for label in cbar_combined.ax.get_yticklabels():
            label.set_fontfamily('serif')
            label.set_color('0.85')
        
        # Generate output filename
        if num_files > 1:
            base_name = output_png.replace('.png', '')
            output_png_file = f"{base_name}_part{file_idx + 1}.png"
        else:
            output_png_file = output_png
        
        # Save combined figure (fixed dimensions: no tight_layout/bbox_inches so layout matches figsize)
        plt.figure(fig_combined.number)
        plt.savefig(output_png_file, dpi=150, bbox_inches=None, facecolor='black')
        plt.close(fig_combined)
        output_png_file_abs = os.path.abspath(output_png_file)
        print(f"[DEBUG] Saved combined figure: {output_png_file_abs}")
    print(f"[DEBUG] Combined figure saved to: {output_png_abs}")
    
    # Create combined figures for rejected peptides
    if len(rejected_peptides) > 0:
        # Sort rejected peptides by apex RT (rt_anchor fallback for edge cases)
        def _rejected_rt_sort_key(rd):
            rt = rd.get('apex_rt') or rd.get('rt_anchor')
            return (float(rt) if rt is not None and not (isinstance(rt, float) and np.isnan(rt)) else 999999.0, rd.get('seq_start_pos', 999999))
        rejected_peptides = sorted(rejected_peptides, key=_rejected_rt_sort_key)
        print(f"[DEBUG] Creating combined figures for {len(rejected_peptides)} rejected peptides (sorted by apex RT)...")
        
        # Calculate how many files we need for rejected peptides
        num_rejected_files = int(np.ceil(len(rejected_peptides) / plots_per_file))
        
        for file_idx in range(num_rejected_files):
            start_idx = file_idx * plots_per_file
            end_idx = min(start_idx + plots_per_file, len(rejected_peptides))
            peptides_in_file = end_idx - start_idx
            
            if peptides_in_file == 0:
                continue
            
            # Calculate rows needed for this file
            rows_this_file = int(np.ceil(peptides_in_file / cols_per_file))
            
            # Create figure for rejected peptides (black background to match overlay)
            fig_rejected = plt.figure(figsize=(fig_width, rows_this_file * subplot_height * 2.0))
            fig_rejected.patch.set_facecolor('black')
            file_title = f'REJECTED Peptides (Sorted by Apex RT) - {len(rejected_peptides)} total'
            if num_rejected_files > 1:
                file_title += f' - Part {file_idx + 1}/{num_rejected_files}'
            fig_rejected.suptitle(file_title, fontsize=16, y=0.995, fontfamily='serif', color='red')
            
            # Create gridspec
            gs_rejected = fig_rejected.add_gridspec(rows_this_file * 2, cols_per_file,
                                                    hspace=0.4, wspace=wspace,
                                                    height_ratios=[2.5, 1] * rows_this_file)
            
            # Create axes arrays
            axes_rejected_chrom = []
            axes_rejected_spec = []
            
            for peptide_idx in range(peptides_in_file):
                row_idx = peptide_idx // cols_per_file
                col_idx = peptide_idx % cols_per_file
                
                ax_chrom = fig_rejected.add_subplot(gs_rejected[row_idx * 2, col_idx])
                ax_chrom.set_facecolor('black')
                axes_rejected_chrom.append(ax_chrom)
                
                ax_spec = fig_rejected.add_subplot(gs_rejected[row_idx * 2 + 1, col_idx])
                ax_spec.set_facecolor('black')
                axes_rejected_spec.append(ax_spec)
            
            rejected_figures.append(fig_rejected)
            rejected_axes_list.append((axes_rejected_chrom, axes_rejected_spec))
        
        # Now plot each rejected peptide to its combined figure
        for rejected_idx, rejected_data in enumerate(rejected_peptides):
            file_idx = rejected_idx // plots_per_file
            local_idx = rejected_idx % plots_per_file
            
            if file_idx >= len(rejected_axes_list):
                continue
            
            axes_rejected_chrom_list, axes_rejected_spec_list = rejected_axes_list[file_idx]
            ax_rejected_chrom = axes_rejected_chrom_list[local_idx]
            ax_rejected_spec = axes_rejected_spec_list[local_idx]
            
            # Extract data from rejected_data dictionary
            rts = rejected_data['rts']
            rts_min = rejected_data['rts_min']
            intensity_matrix = rejected_data['intensity_matrix']
            total_intensities = rejected_data['total_intensities']
            scored_candidates = rejected_data['scored_candidates']
            best_peak = rejected_data['best_peak']
            min_rt = rejected_data['min_rt']
            max_rt = rejected_data['max_rt']
            min_rt_min = rejected_data['min_rt_min']
            max_rt_min = rejected_data['max_rt_min']
            apex_rt = rejected_data['apex_rt']
            apex_rt_min = rejected_data['apex_rt_min']
            rt_anchor = rejected_data['rt_anchor']
            anchor_evalue = rejected_data['anchor_evalue']
            rt_to_frag_count = rejected_data['rt_to_frag_count']
            spec_mzs_windowed = rejected_data['spec_mzs_windowed']
            spec_ints_windowed = rejected_data['spec_ints_windowed']
            isotope_mzs = rejected_data['isotope_mzs']
            mz_min_window = rejected_data['mz_min_window']
            mz_max_window = rejected_data['mz_max_window']
            spectrum_rt_min = rejected_data['spectrum_rt_min']
            spectrum_rt_max = rejected_data['spectrum_rt_max']
            using_anchor_window = rejected_data['using_anchor_window']
            peptide_key = rejected_data['peptide_key']
            peptide = rejected_data['peptide']
            charge = rejected_data['charge']
            mods = rejected_data['mods']
            title = rejected_data['title']
            rejection_reason = rejected_data['rejection_reason']
            seq_start_pos = rejected_data['seq_start_pos']
            peptide_crossref = rejected_data['peptide_crossref']
            evalue_cmap = rejected_data['evalue_cmap']
            evalue_norm = rejected_data['evalue_norm']
            
            # Plot isotope traces on rejected combined chromatogram
            plot_isotope_traces(ax_rejected_chrom, rts_min, intensity_matrix, total_intensities)
            measured_mz_matrix_rej = rejected_data.get('measured_mz_matrix')
            start_idx_rej = rejected_data.get('start_idx', 0)
            end_idx_rej = rejected_data.get('end_idx', len(rts_min) - 1)
            add_ppm_labels_above_peaks(ax_rejected_chrom, rts_min, intensity_matrix, total_intensities, measured_mz_matrix_rej, isotope_mzs, start_idx_rej, end_idx_rej)
            
            # Add peak annotations
            ax_rejected_chrom._is_combined = True
            ms1_rts_rej = rejected_data.get('ms1_rts_table')
            ms1_rt_evalue_rej = rejected_data.get('ms1_rt_evalue_pairs')
            coll_min_rej = rejected_data.get('collection_min_rt_min')
            coll_max_rej = rejected_data.get('collection_max_rt_min')
            add_peak_annotations(ax_rejected_chrom, scored_candidates, best_peak, min_rt_min, max_rt_min,
                               apex_rt_min, rts, rts_min, total_intensities, min_rt, max_rt,
                               rt_anchor=rt_anchor, anchor_evalue=anchor_evalue,
                               evalue_cmap=evalue_cmap, evalue_norm=evalue_norm,
                               peptide_crossref=peptide_crossref, peptide_key=peptide_key,
                               rt_to_frag_count=rt_to_frag_count, ms1_rts_table=ms1_rts_rej,
                               ms1_rt_evalue_pairs=ms1_rt_evalue_rej,
                               collection_min_rt_min=coll_min_rej, collection_max_rt_min=coll_max_rej,
                               measured_mz_matrix_ann=measured_mz_matrix_rej, isotope_mzs_ann=isotope_mzs,
                               intensity_matrix_ann=intensity_matrix, total_intensities_ann=total_intensities)
            
            # Format rejected chromatogram (include reason for rejection; light text on black background)
            reason_short = (rejection_reason or '')[:60] + ('…' if len(rejection_reason or '') > 60 else '')
            title_rejected = f"[{seq_start_pos}] {title} [REJECTED: {reason_short}]"
            ax_rejected_chrom.set_title(title_rejected, fontsize=8, fontweight='bold', pad=3, fontfamily='serif', color='red')
            ax_rejected_chrom.set_xlabel('MS1 RT (min)', fontsize=8, fontfamily='serif', color='0.85')
            ax_rejected_chrom.set_ylabel('Intensity', fontsize=8, fontfamily='serif', color='0.85')
            ax_rejected_chrom.tick_params(labelsize=7, colors='0.85')
            for label in ax_rejected_chrom.get_xticklabels() + ax_rejected_chrom.get_yticklabels():
                label.set_fontfamily('serif')
                label.set_color('0.85')
            for spine in ax_rejected_chrom.spines.values():
                spine.set_color('0.6')
            ax_rejected_chrom.grid(False)
            
            # Set x-axis to span all table RTs for this peptide; extend beyond collection window for context
            rt_table = ms1_rts_rej if ms1_rts_rej else []
            if rt_table:
                global_min_sec = min(rt_table)
                global_max_sec = max(rt_table)
                span_sec = global_max_sec - global_min_sec
                collection_span_min_rej = (max_rt - min_rt) / 60.0 if (min_rt and max_rt) else span_sec / 60.0
                buffer_sec = max(90.0, span_sec * 0.25, collection_span_min_rej * 60.0 * 0.5)
                buffer_min = buffer_sec / 60.0
                x_min = max(0, global_min_sec / 60.0 - buffer_min)
                x_max = global_max_sec / 60.0 + buffer_min
            else:
                buffer_min = max(1.5, (max_rt - min_rt) / 60.0 * 0.5)
                x_min = max(0, min_rt_min - buffer_min)
                x_max = max_rt_min + buffer_min
            ax_rejected_chrom.set_xlim([x_min, x_max])
            
            # Set y-axis limits
            if len(total_intensities) > 0:
                window_mask = (rts_min >= x_min) & (rts_min <= x_max)
                window_intensities = total_intensities[window_mask] if np.any(window_mask) else total_intensities
                if len(window_intensities) > 0 and np.max(window_intensities) > 0:
                    y_max = np.max(window_intensities) * 1.1
                    y_min = 0
                    ax_rejected_chrom.set_ylim([y_min, y_max])
            
            # Plot spectrum if available
            if spec_mzs_windowed is not None and len(spec_mzs_windowed) > 0:
                # Plot spectrum
                markerline_rejected, stemlines_rejected, baseline_rejected = ax_rejected_spec.stem(
                    spec_mzs_windowed, spec_ints_windowed,
                    basefmt=' ', linefmt='#888888', markerfmt=' ')
                plt.setp(stemlines_rejected, linewidth=0.3)
                plt.setp(markerline_rejected, markersize=1)
                
                # Highlight isotopes
                iso_colors = ['#0033CC', '#6600CC', '#CC0099', '#CC0066']
                matched_iso_intensities_rejected = []
                
                matched_iso_indices_rejected = []
                for iso_idx, iso_mz in enumerate(isotope_mzs):
                    if iso_idx < len(iso_colors):
                        iso_tolerance = iso_mz * 5e-6  # 5 ppm tolerance
                        iso_mask = (spec_mzs_windowed >= iso_mz - iso_tolerance) & (spec_mzs_windowed <= iso_mz + iso_tolerance)
                        if np.any(iso_mask):
                            iso_intensities = spec_ints_windowed[iso_mask]
                            max_iso_intensity = np.max(iso_intensities) if len(iso_intensities) > 0 else 0
                            matched_iso_intensities_rejected.append(max_iso_intensity)
                            matched_iso_indices_rejected.append(iso_idx)
                            ax_rejected_spec.vlines(iso_mz, 0, max_iso_intensity, colors=iso_colors[iso_idx],
                                                  linestyles='--', linewidths=2.5, alpha=1.0,
                                                  label=f'M+{iso_idx}')
                
                # Scale y-axis based on matched peaks
                if len(matched_iso_intensities_rejected) > 0:
                    max_matched_intensity = np.max(matched_iso_intensities_rejected)
                    y_max_spec = max_matched_intensity * 1.2
                    ax_rejected_spec.set_ylim([0, y_max_spec])
                
                # Format spectrum
                title_rt_range = f'{spectrum_rt_min:.1f}-{spectrum_rt_max:.1f}s' if spectrum_rt_min and spectrum_rt_max else 'N/A'
                if using_anchor_window:
                    title_rt_range += ' (anchor RT window)'
                ax_rejected_spec.set_title(f'Summed MS1 Spectrum (RT={title_rt_range})', fontsize=8, fontweight='bold', fontfamily='serif', color='0.9')
                ax_rejected_spec.set_xlabel('m/z', fontsize=7, fontfamily='serif', color='0.85')
                ax_rejected_spec.set_ylabel('Intensity', fontsize=7, fontfamily='serif', color='0.85')
                ax_rejected_spec.tick_params(labelsize=7, colors='0.85')
                for label in ax_rejected_spec.get_xticklabels() + ax_rejected_spec.get_yticklabels():
                    label.set_color('0.85')
                    label.set_fontfamily('serif')
                for spine in ax_rejected_spec.spines.values():
                    spine.set_color('0.6')
                ax_rejected_spec.grid(False)
                if mz_min_window and mz_max_window:
                    ax_rejected_spec.set_xlim([mz_min_window, mz_max_window])
                # Spectrum legend: M+0, M+1, M+2, M+3 for isotopes that were drawn
                handles_rej, labels_rej = ax_rejected_spec.get_legend_handles_labels()
                if not handles_rej and matched_iso_indices_rejected:
                    # vlines may not expose label to legend; use proxy Line2D
                    from matplotlib.lines import Line2D
                    handles_rej = [Line2D([0], [0], color=iso_colors[i], linestyle='--', linewidth=2.5, label=f'M+{i}')
                                  for i in matched_iso_indices_rejected if i < len(iso_colors)]
                    labels_rej = [f'M+{i}' for i in matched_iso_indices_rejected if i < len(iso_colors)]
                if handles_rej and labels_rej:
                    ax_rejected_spec.legend(handles_rej, labels_rej, loc='upper right',
                                           fontsize=5, prop={'family': 'serif'}, facecolor='#1a1a1a', edgecolor='0.6', labelcolor='0.9')
                # ±5 ppm tolerance label
                if mz_min_window and mz_max_window:
                    x_lim_r = ax_rejected_spec.get_xlim()
                    y_lim_r = ax_rejected_spec.get_ylim()
                    tol_x = x_lim_r[0] + (x_lim_r[1] - x_lim_r[0]) * 0.02
                    tol_y = y_lim_r[0] + (y_lim_r[1] - y_lim_r[0]) * 0.95
                    ax_rejected_spec.text(tol_x, tol_y, '±5 ppm',
                                          fontsize=6, ha='left', va='top', fontfamily='serif',
                                          bbox=dict(boxstyle='round,pad=0.25', facecolor='lightblue', alpha=0.8, edgecolor='black', linewidth=0.5),
                                          zorder=25, clip_on=True)
            else:
                ax_rejected_spec.axis('off')
            
            # Individual figure was already closed when saved
        
        # Save rejected combined figures
        rejected_output_base = output_png.replace('.png', '_rejected.png')
        for file_idx, fig_rejected in enumerate(rejected_figures):
            if num_rejected_files > 1:
                rejected_output_file = rejected_output_base.replace('.png', f'_part{file_idx+1}.png')
            else:
                rejected_output_file = rejected_output_base
            
            axes_rejected_chrom_list, axes_rejected_spec_list = rejected_axes_list[file_idx]
            start_idx = file_idx * plots_per_file
            end_idx = min(start_idx + plots_per_file, len(rejected_peptides))
            peptides_in_file = end_idx - start_idx
            
            # Hide unused subplots
            for local_idx in range(peptides_in_file, len(axes_rejected_chrom_list)):
                axes_rejected_chrom_list[local_idx].axis('off')
                axes_rejected_spec_list[local_idx].axis('off')
            
            # Add E-value colorbar
            sm_rejected = ScalarMappable(cmap=evalue_cmap, norm=evalue_norm)
            sm_rejected.set_array([])
            cbar_ax_rejected = fig_rejected.add_axes([0.01, 0.65, 0.018, 0.26])
            cbar_rejected = fig_rejected.colorbar(sm_rejected, cax=cbar_ax_rejected, orientation='vertical')
            cbar_rejected.set_label('E-value\n(OpenMS RT)', fontsize=10, fontfamily='serif', rotation=0, labelpad=10)
            cbar_rejected.ax.tick_params(labelsize=9)
            
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=UserWarning, message='.*tight_layout.*')
                    plt.tight_layout()
            except Exception:
                pass
            
            try:
                out_dir = os.path.dirname(rejected_output_file)
                if out_dir:
                    os.makedirs(out_dir, exist_ok=True)
                fig_rejected.savefig(rejected_output_file, dpi=150, bbox_inches='tight', facecolor='black')
                print(f"[DEBUG] Saved rejected combined figure {file_idx + 1}/{num_rejected_files}: {os.path.abspath(rejected_output_file)}")
            except Exception as e:
                print(f"[ERROR] Failed to save rejected combined figure: {os.path.abspath(rejected_output_file)}: {e}", file=sys.stderr)
            finally:
                plt.close(fig_rejected)
        
        print(f"[DEBUG] All rejected combined figures saved")
    else:
        print(f"[DEBUG] No rejected peptides to plot")
    
    # Print peak window summary
    print("[DEBUG] ========================================")
    print("[DEBUG] Peak Window Summary")
    print("[DEBUG] ========================================")
    if all_peak_windows:
        # Separate accepted and rejected for summary
        accepted_windows = [w for w in all_peak_windows if w['status'] == 'accepted']
        rejected_windows = [w for w in all_peak_windows if w['status'] == 'rejected']
        
        print(f"[DEBUG] Found peak windows for {len(all_peak_windows)}/{num_peptides} peptides")
        print(f"[DEBUG]   Accepted: {len(accepted_windows)}")
        print(f"[DEBUG]   Rejected: {len(rejected_windows)}")
        
        if len(accepted_windows) > 0:
            accepted_window_sizes = [w['detected_peak_window_size'] for w in accepted_windows if w['detected_peak_window_size'] is not None]
            if accepted_window_sizes:
                avg_window = np.mean(accepted_window_sizes)
                print(f"[DEBUG] Average detected peak window size (accepted): {avg_window:.2f} seconds")
                print(f"[DEBUG] Min window size: {min(accepted_window_sizes):.2f} seconds")
                print(f"[DEBUG] Max window size: {max(accepted_window_sizes):.2f} seconds")
        
        if len(rejected_windows) > 0:
            rejection_reasons = {}
            for w in rejected_windows:
                reason = w.get('rejection_reason', 'Unknown')
                rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
            print(f"[DEBUG] Rejection reasons:")
            for reason, count in rejection_reasons.items():
                print(f"[DEBUG]   {reason}: {count}")
        
        # Save ALL peak windows (accepted and rejected) to CSV in the output directory
        # Merge significant_data (fragments, pairs, single-AA overhang positions) into each window for CSV columns
        def _norm_mods_w(m):
            if m is None or (isinstance(m, float) and pd.isna(m)):
                return '-'
            s = str(m).strip()
            return s if s and s.lower() != 'nan' else '-'
        for w in all_peak_windows:
            key = (str(w.get('peptide', '')).strip(), int(w.get('charge', 0)) if pd.notna(w.get('charge')) else 0, _norm_mods_w(w.get('modifications', '-')))
            if key in significant_data:
                for col, val in significant_data[key].items():
                    if val is not None:
                        w[col] = val
        dataframes_dir = getattr(args, 'dataframes_dir', None)
        if dataframes_dir and os.path.isdir(dataframes_dir):
            # Save chromatogram traces (same order as all_peak_windows) so RT windows / by-channel can show real peaks without re-extraction
            if traces_list and len(traces_list) == len(all_peak_windows):
                npz_path = os.path.join(dataframes_dir, 'chromatogram_traces.npz')
                save_dict = {}
                for i, (rts_arr, int_arr) in enumerate(traces_list):
                    save_dict[f'trace_{i}_rts'] = np.asarray(rts_arr, dtype=float).flatten().copy()
                    save_dict[f'trace_{i}_int'] = np.asarray(int_arr, dtype=float).copy()
                np.savez_compressed(npz_path, **save_dict)
                print(f"[DEBUG] chromatogram_traces.npz saved to: {os.path.abspath(npz_path)} (for by-channel/overlay)")
                # Metrics CSV in same row order as npz (trace_i = row i) so downstream can match by index or (peptide,charge,mods)
                metrics_path = os.path.join(dataframes_dir, 'chromatogram_metrics_all.csv')
                for i, w in enumerate(all_peak_windows):
                    w['trace_index'] = i
                pd.DataFrame(all_peak_windows).to_csv(metrics_path, index=False)
                print(f"[DEBUG] chromatogram_metrics_all.csv saved to: {os.path.abspath(metrics_path)}")
        dataframes_accepted_dir = getattr(args, 'dataframes_accepted_dir', None)
        dataframes_rejected_dir = getattr(args, 'dataframes_rejected_dir', None)

        # If dataframes_dir provided (e.g. by run_visualization), write filtered dataset + chromatogram metrics (accepted only / rejected separate)
        if dataframes_dir and len(all_peak_windows) > 0:
            os.makedirs(dataframes_dir, exist_ok=True)
            def _norm_mods_df(m):
                if m is None or (isinstance(m, float) and pd.isna(m)):
                    return '-'
                s = str(m).strip()
                return s if s and s.lower() != 'nan' else '-'
            windows_lookup = {}
            chrom_cols = [
                'status', 'rejection_reason', 'num_psms', 'unique_mz', 'ms1_rt_range', 'ms1_rt_min', 'ms1_rt_max',
                'anchor_rt', 'anchor_evalue', 'apex_rt', 'apex_intensity',
                'detected_peak_min_rt', 'detected_peak_max_rt', 'detected_peak_window_size',
                'collection_min_rt', 'collection_max_rt', 'collection_window_size',
                'spectrum_window_min_rt', 'spectrum_window_max_rt', 'spectrum_window_size',
                'using_anchor_window', 'n_isotopic_matches', 'matched_isotopes',
                'm0_m1_gt_m2_m3', 'm0_intensity', 'm1_intensity', 'm2_intensity', 'm3_intensity',
                'precursor_mz', 'sequence_start_pos', 'ppm_at_apex',
                'best_score', 'n_candidates', 'total_area', 'coelution_score'
            ]
            for w in all_peak_windows:
                pep = str(w.get('peptide', '')).strip()
                ch = int(w.get('charge', 0)) if w.get('charge') is not None else 0
                mod = _norm_mods_df(w.get('modifications', '-'))
                windows_lookup[(pep, ch, mod)] = {c: w.get(c) for c in chrom_cols if c in w}
            # Base = filtered dataset when filter_csv is set, else full input CSV
            if filter_csv and os.path.exists(filter_csv):
                try:
                    with open(filter_csv, 'r') as f:
                        first_line = f.readline()
                    skip_rows = 1 if 'CometVersion' in first_line else 0
                except Exception:
                    skip_rows = 0
                df_base = pd.read_csv(filter_csv, sep=',', skiprows=skip_rows, engine='python', quotechar='"', on_bad_lines='warn')
                base_name = os.path.splitext(os.path.basename(filter_csv))[0]
            else:
                try:
                    with open(comet_csv, 'r') as f:
                        first_line = f.readline()
                    skip_rows = 1 if 'CometVersion' in first_line else 0
                except Exception:
                    skip_rows = 0
                df_base = pd.read_csv(comet_csv, sep=',', skiprows=skip_rows, engine='python', quotechar='"', on_bad_lines='warn')
                base_name = os.path.splitext(os.path.basename(comet_csv))[0]
            # Validate base dataframe before merging and writing
            try:
                from visualization.csv_validation import validate_and_log
                validate_and_log(
                    df_base,
                    source_name=base_name + '_with_chromatogram_metrics',
                    required_columns=('plain_peptide', 'charge'),
                    validate_identifiers=True,
                    validate_numeric=True,
                )
            except Exception as e:
                print(f"[DEBUG] Validation warning: {e}")
            # Ensure sequence_start_pos is present; fill from sequence_positions when missing (same as main extraction path)
            if 'sequence_positions' in df_base.columns:
                if 'sequence_start_pos' not in df_base.columns:
                    df_base['sequence_start_pos'] = np.nan
                missing_seq_start = df_base['sequence_start_pos'].isna()
                if missing_seq_start.any():
                    def _parse_start(sp):
                        if pd.isna(sp) or sp is None or str(sp).strip() == '':
                            return np.nan
                        s = str(sp).strip()
                        if '-' in s:
                            try:
                                return int(s.split('-')[0])
                            except (ValueError, TypeError):
                                return np.nan
                        if ',' in s:
                            try:
                                return int(s.split(',')[0].strip())
                            except (ValueError, TypeError):
                                return np.nan
                        try:
                            return int(s)
                        except (ValueError, TypeError):
                            return np.nan
                    parsed = df_base.loc[missing_seq_start, 'sequence_positions'].map(_parse_start)
                    df_base.loc[missing_seq_start, 'sequence_start_pos'] = parsed
            # In test mode, keep only rows that were actually processed so extraction-test
            # outputs do not appear mostly empty after merge-back.
            if getattr(args, 'test', False):
                def _row_key_for_filter(row_obj):
                    pep = ''
                    for col in ['peptide_sequence', 'plain_peptide', 'sequence']:
                        v = row_obj.get(col, '')
                        if pd.notna(v) and str(v).strip():
                            pep = str(v).strip()
                            break
                    ch = int(row_obj.get('charge', 0)) if pd.notna(row_obj.get('charge')) else 0
                    mod = _norm_mods_df(row_obj.get('modifications', '-'))
                    return (pep, ch, mod)
                keep_mask = df_base.apply(lambda r: _row_key_for_filter(r) in windows_lookup, axis=1)
                n_before = len(df_base)
                df_base = df_base.loc[keep_mask].copy()
                print(f"[DEBUG] Test mode merge: keeping {len(df_base)} of {n_before} rows with extracted metrics")

            # Significant = Comet fragments passing 5 ppm + noise %. Columns: fragments, pairs, and canonical protein-position overhang mapping.
            sig_cols = ['significant_fragment_ions', 'significant_fragment_pairs', 'single_aa_overhangs_protein_positions']
            for c in chrom_cols:
                if c not in df_base.columns:
                    df_base[c] = None
            for c in sig_cols:
                if c not in df_base.columns:
                    df_base[c] = None
            for c in sig_cols:
                df_base[c] = None
            def _row_peptide_key_val(row_obj):
                # Canonical name first; fallback to legacy names.
                for col in ['peptide_sequence', 'plain_peptide', 'sequence']:
                    v = row_obj.get(col, '')
                    if pd.notna(v) and str(v).strip():
                        return str(v).strip()
                return ''

            for idx, row in df_base.iterrows():
                key = (
                    _row_peptide_key_val(row),
                    int(row.get('charge', 0)) if pd.notna(row.get('charge')) else 0,
                    _norm_mods_df(row.get('modifications', '-'))
                )
                if key in windows_lookup:
                    for col, val in windows_lookup[key].items():
                        if val is not None:
                            df_base.at[idx, col] = val
                if key in significant_data:
                    for col, val in significant_data[key].items():
                        if val is not None and col in df_base.columns:
                            df_base.at[idx, col] = val
            # Remove legacy duplicate columns that caused off-by-one confusion in downstream views.
            for legacy_col in ['significant_single_aa_overhangs', 'significant_single_aa_overhangs_protein_positions']:
                if legacy_col in df_base.columns:
                    df_base = df_base.drop(columns=[legacy_col])
            if '_format_extraction_output_columns' not in locals():
                def _format_extraction_output_columns(df_out):
                    df_fmt = df_out.copy()
                    if 'precursor_mz' in df_fmt.columns:
                        if 'theoretical_mz' in df_fmt.columns:
                            df_fmt['theoretical_mz'] = df_fmt['precursor_mz'].where(
                                pd.notna(df_fmt['precursor_mz']),
                                df_fmt['theoretical_mz']
                            )
                            df_fmt = df_fmt.drop(columns=['precursor_mz'])
                        else:
                            df_fmt = df_fmt.rename(columns={'precursor_mz': 'theoretical_mz'})
                    rt_name_map = {
                        'MS1_retention_time_sec': 'MS1_RT_sec',
                        'MS1_retention_time_min': 'MS1_RT_minutes',
                        'MS1_retention_time_intensity': 'MS1_RT_intensity',
                        'MS2_retention_time_sec': 'MS2_RT_sec',
                        'MS2_retention_time_min': 'MS2_RT_minutes',
                        'retention_time_sec': 'RT_sec',
                        'retention_time_min': 'RT_minutes',
                    }
                    for old, new in rt_name_map.items():
                        if old in df_fmt.columns:
                            if new in df_fmt.columns:
                                df_fmt[new] = df_fmt[old].where(pd.notna(df_fmt[old]), df_fmt[new])
                                df_fmt = df_fmt.drop(columns=[old])
                            else:
                                df_fmt = df_fmt.rename(columns={old: new})
                    rt_rename = {
                        'detected_peak_min_rt': 'MS1_RT_integration_start_sec',
                        'detected_peak_max_rt': 'MS1_RT_integration_stop_sec',
                        'collection_min_rt': 'MS1_RT_collection_start_sec',
                        'collection_max_rt': 'MS1_RT_collection_stop_sec',
                    }
                    for old, new in rt_rename.items():
                        if old in df_fmt.columns:
                            if new in df_fmt.columns:
                                df_fmt[new] = df_fmt[old].where(pd.notna(df_fmt[old]), df_fmt[new])
                                df_fmt = df_fmt.drop(columns=[old])
                            else:
                                df_fmt = df_fmt.rename(columns={old: new})
                    for sec_col, min_col in [
                        ('MS1_RT_integration_start_sec', 'MS1_RT_integration_start_minutes'),
                        ('MS1_RT_integration_stop_sec', 'MS1_RT_integration_stop_minutes'),
                        ('MS1_RT_collection_start_sec', 'MS1_RT_collection_start_minutes'),
                        ('MS1_RT_collection_stop_sec', 'MS1_RT_collection_stop_minutes'),
                    ]:
                        if sec_col in df_fmt.columns and min_col not in df_fmt.columns:
                            df_fmt[min_col] = pd.to_numeric(df_fmt[sec_col], errors='coerce') / 60.0
                    block_after_observed = [c for c in ['theoretical_mz', 'total_area', 'matched_isotopes'] if c in df_fmt.columns]
                    if 'observed_mz' in df_fmt.columns and block_after_observed:
                        cols = list(df_fmt.columns)
                        rest = [c for c in cols if c not in block_after_observed]
                        insert_idx = rest.index('observed_mz') + 1
                        df_fmt = df_fmt[rest[:insert_idx] + block_after_observed + rest[insert_idx:]]
                    rt_block = [
                        c for c in [
                            'MS1_RT_integration_start_sec', 'MS1_RT_integration_stop_sec',
                            'MS1_RT_collection_start_sec', 'MS1_RT_collection_stop_sec',
                            'MS1_RT_integration_start_minutes', 'MS1_RT_integration_stop_minutes',
                            'MS1_RT_collection_start_minutes', 'MS1_RT_collection_stop_minutes',
                        ] if c in df_fmt.columns
                    ]
                    qpep_candidates = ['perc_qvalue', 'perc_PEP', 'qvalue', 'pep', 'percolator_qvalue', 'percolator_PEP', 'q-value', 'PEP']
                    if rt_block:
                        cols = list(df_fmt.columns)
                        rest = [c for c in cols if c not in rt_block]
                        qpep_idx = [i for i, c in enumerate(rest) if c in qpep_candidates]
                        if qpep_idx:
                            insert_idx = max(qpep_idx) + 1
                            df_fmt = df_fmt[rest[:insert_idx] + rt_block + rest[insert_idx:]]
                    frag_tail = [c for c in ['comet_matched_frags', 'comet_matched_frags_mz', 'comet_matched_frags_intensities', 'comet_matched_frags_quality_scores'] if c in df_fmt.columns]
                    ms2_tail = [c for c in ['MS1_RT_sec', 'MS2_RT_sec', 'MS2_RT_minutes'] if c in df_fmt.columns]
                    if 'MS1_mz_error' in df_fmt.columns and 'MS1_mz_error_ppm' not in df_fmt.columns:
                        df_fmt = df_fmt.rename(columns={'MS1_mz_error': 'MS1_mz_error_ppm'})
                    front_cols = [c for c in ['protein_position', 'peptide_sequence', 'charge', 'observed_mz', 'theoretical_mz', 'MS1_mz_error_ppm', 'MS1_RT_minutes'] if c in df_fmt.columns]
                    ms1_cols = [c for c in ['MS1_RT_intensity'] if c in df_fmt.columns]
                    if front_cols or ms1_cols or frag_tail or ms2_tail:
                        lead_cols = [c for c in df_fmt.columns if c not in front_cols and c not in ms1_cols and c not in frag_tail and c not in ms2_tail]
                        df_fmt = df_fmt[front_cols + ms1_cols + lead_cols + frag_tail + ms2_tail]
                    return df_fmt
            # Full (filtered set + metrics) to dataframes_dir root
            out_csv = os.path.join(dataframes_dir, f'{base_name}_with_chromatogram_metrics.csv')
            os.makedirs(os.path.dirname(out_csv), exist_ok=True)
            _format_extraction_output_columns(df_base).to_csv(out_csv, index=False)
            print(f"[DEBUG] Filtered dataset + chromatogram metrics saved to: {os.path.abspath(out_csv)}")

            # Diagnostic: coelution vs shape scatter (rejected marked)
            try:
                coel_vals = []
                shape_vals = []
                status_vals = []
                for w in all_peak_windows:
                    c = w.get('coelution_score')
                    s = w.get('shape_corr') or w.get('mean_shape_corr')
                    if c is not None and s is not None and pd.notna(c) and pd.notna(s):
                        coel_vals.append(float(c))
                        shape_vals.append(float(s))
                        status_vals.append(w.get('status', 'accepted'))
                if coel_vals and shape_vals:
                    diagnostics_dir = os.path.join(os.path.dirname(dataframes_dir), 'diagnostics')
                    os.makedirs(diagnostics_dir, exist_ok=True)
                    import matplotlib
                    matplotlib.use('Agg')
                    # plt already imported at module level; local import caused UnboundLocalError
                    coel_arr = np.array(coel_vals)
                    shape_arr = np.array(shape_vals)
                    accepted_mask = np.array([st == 'accepted' for st in status_vals])
                    rejected_mask = ~accepted_mask
                    fig, ax = plt.subplots(figsize=(10, 8))
                    fig.patch.set_facecolor('black')
                    ax.set_facecolor('black')
                    if np.any(accepted_mask):
                        ax.scatter(coel_arr[accepted_mask], shape_arr[accepted_mask], c='lavender', alpha=0.6, s=25, label=f'Accepted (n={np.sum(accepted_mask)})')
                    if np.any(rejected_mask):
                        ax.scatter(coel_arr[rejected_mask], shape_arr[rejected_mask], c='orange', alpha=0.6, s=25, label=f'Rejected (n={np.sum(rejected_mask)})')
                    ax.set_xlabel('Coelution score', fontsize=12, color='0.85')
                    ax.set_ylabel('Shape correlation', fontsize=12, color='0.85')
                    ax.set_title('Extraction: coelution vs shape (rejected marked)', fontsize=14, color='0.9')
                    ax.tick_params(colors='0.85')
                    for label in ax.get_xticklabels() + ax.get_yticklabels():
                        label.set_color('0.85')
                    for spine in ax.spines.values():
                        spine.set_color('0.6')
                    ax.legend(loc='lower left', fontsize=10, facecolor='#1a1a1a', edgecolor='0.6', labelcolor='0.9')
                    ax.grid(True, alpha=0.25)
                    plt.tight_layout()
                    diag_path = os.path.join(diagnostics_dir, 'extraction_coelution_vs_shape.png')
                    fig.savefig(diag_path, dpi=150, bbox_inches='tight', facecolor='black')
                    plt.close(fig)
                    print(f"[DEBUG] Diagnostic scatter saved: {diag_path}")
            except Exception as e:
                print(f"[DEBUG] Could not create coelution vs shape diagnostic: {e}")

            # Diagnostic: rejection counts by category (proline, envelope, shape)
            diagnostics_dir = os.path.join(os.path.dirname(dataframes_dir), 'diagnostics')
            os.makedirs(diagnostics_dir, exist_ok=True)
            try:
                import matplotlib
                matplotlib.use('Agg')

                # 1. Proline rejection: bar chart
                n_proline = rejection_counts.get('Starts/ends with proline', 0)
                n_processed = num_peptides - n_proline if num_peptides else len(all_peak_windows)
                if num_peptides > 0 or n_proline > 0:
                    fig_p, ax_p = plt.subplots(figsize=(6, 5))
                    fig_p.patch.set_facecolor('black')
                    ax_p.set_facecolor('black')
                    labels = ['Rejected\n(proline)', 'Processed\n(not proline)']
                    vals = [n_proline, max(0, n_processed)]
                    colors = ['orange', 'lavender']
                    bars = ax_p.bar(labels, vals, color=colors, alpha=0.8, edgecolor='0.6')
                    ax_p.set_ylabel('Count', fontsize=12, color='0.85')
                    ax_p.set_title(f'Rejected for starting/ending with proline: {n_proline}', fontsize=14, color='0.9')
                    ax_p.tick_params(colors='0.85')
                    for label in ax_p.get_xticklabels() + ax_p.get_yticklabels():
                        label.set_color('0.85')
                    for spine in ax_p.spines.values():
                        spine.set_color('0.6')
                    for b in bars:
                        if b.get_height() > 0:
                            ax_p.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5, str(int(b.get_height())), ha='center', va='bottom', fontsize=11, color='0.9')
                    plt.tight_layout()
                    fig_p.savefig(os.path.join(diagnostics_dir, 'extraction_rejection_proline.png'), dpi=150, bbox_inches='tight', facecolor='black')
                    plt.close(fig_p)
                    print(f"[DEBUG] Diagnostic proline rejection saved: {os.path.join(diagnostics_dir, 'extraction_rejection_proline.png')}")
            except Exception as e:
                print(f"[DEBUG] Could not create proline rejection diagnostic: {e}")

            # 2. Envelope filter: scatter coelution vs shape (lavender=passed, orange=rejected for envelope)
            try:
                coel_env, shape_env, env_rej = [], [], []
                for w in all_peak_windows:
                    c = w.get('coelution_score')
                    s = w.get('shape_corr') or w.get('mean_shape_corr')
                    if c is not None and s is not None and pd.notna(c) and pd.notna(s):
                        coel_env.append(float(c))
                        shape_env.append(float(s))
                        rr = str(w.get('rejection_reason', ''))
                        env_rej.append(w.get('status') == 'rejected' and 'Isotopic envelope' in rr)
                if coel_env and shape_env:
                    coel_a = np.array(coel_env)
                    shape_a = np.array(shape_env)
                    env_rej_a = np.array(env_rej)
                    passed = ~env_rej_a
                    fig_e, ax_e = plt.subplots(figsize=(10, 8))
                    fig_e.patch.set_facecolor('black')
                    ax_e.set_facecolor('black')
                    if np.any(passed):
                        ax_e.scatter(coel_a[passed], shape_a[passed], c='lavender', alpha=0.6, s=25, label=f'Passed envelope (n={np.sum(passed)})')
                    if np.any(env_rej_a):
                        ax_e.scatter(coel_a[env_rej_a], shape_a[env_rej_a], c='orange', alpha=0.6, s=25, label=f'Rejected envelope (n={np.sum(env_rej_a)})')
                    ax_e.set_xlabel('Coelution score', fontsize=12, color='0.85')
                    ax_e.set_ylabel('Shape correlation', fontsize=12, color='0.85')
                    ax_e.set_title('Extraction: coelution vs shape (envelope filter)', fontsize=14, color='0.9')
                    ax_e.tick_params(colors='0.85')
                    for label in ax_e.get_xticklabels() + ax_e.get_yticklabels():
                        label.set_color('0.85')
                    for spine in ax_e.spines.values():
                        spine.set_color('0.6')
                    ax_e.legend(loc='lower left', fontsize=10, facecolor='#1a1a1a', edgecolor='0.6', labelcolor='0.9')
                    ax_e.grid(True, alpha=0.25)
                    plt.tight_layout()
                    fig_e.savefig(os.path.join(diagnostics_dir, 'extraction_rejection_envelope.png'), dpi=150, bbox_inches='tight', facecolor='black')
                    plt.close(fig_e)
                    print(f"[DEBUG] Diagnostic envelope filter saved: {os.path.join(diagnostics_dir, 'extraction_rejection_envelope.png')}")
            except Exception as e:
                print(f"[DEBUG] Could not create envelope filter diagnostic: {e}")

            # 3. Shape correlation: scatter coelution vs shape (lavender=passed, orange=rejected for shape)
            try:
                coel_sh, shape_sh, sh_rej = [], [], []
                for w in all_peak_windows:
                    c = w.get('coelution_score')
                    s = w.get('shape_corr') or w.get('mean_shape_corr')
                    if c is not None and s is not None and pd.notna(c) and pd.notna(s):
                        coel_sh.append(float(c))
                        shape_sh.append(float(s))
                        rr = str(w.get('rejection_reason', '')).lower()
                        sh_rej.append(w.get('status') == 'rejected' and ('shape correlation' in rr or 'shape_corr' in rr))
                if coel_sh and shape_sh:
                    coel_a = np.array(coel_sh)
                    shape_a = np.array(shape_sh)
                    sh_rej_a = np.array(sh_rej)
                    passed = ~sh_rej_a
                    fig_s, ax_s = plt.subplots(figsize=(10, 8))
                    fig_s.patch.set_facecolor('black')
                    ax_s.set_facecolor('black')
                    if np.any(passed):
                        ax_s.scatter(coel_a[passed], shape_a[passed], c='lavender', alpha=0.6, s=25, label=f'Passed shape (n={np.sum(passed)})')
                    if np.any(sh_rej_a):
                        ax_s.scatter(coel_a[sh_rej_a], shape_a[sh_rej_a], c='orange', alpha=0.6, s=25, label=f'Rejected shape (n={np.sum(sh_rej_a)})')
                    ax_s.set_xlabel('Coelution score', fontsize=12, color='0.85')
                    ax_s.set_ylabel('Shape correlation', fontsize=12, color='0.85')
                    ax_s.set_title('Extraction: coelution vs shape (shape correlation filter)', fontsize=14, color='0.9')
                    ax_s.tick_params(colors='0.85')
                    for label in ax_s.get_xticklabels() + ax_s.get_yticklabels():
                        label.set_color('0.85')
                    for spine in ax_s.spines.values():
                        spine.set_color('0.6')
                    ax_s.legend(loc='lower left', fontsize=10, facecolor='#1a1a1a', edgecolor='0.6', labelcolor='0.9')
                    ax_s.grid(True, alpha=0.25)
                    plt.tight_layout()
                    fig_s.savefig(os.path.join(diagnostics_dir, 'extraction_rejection_shape.png'), dpi=150, bbox_inches='tight', facecolor='black')
                    plt.close(fig_s)
                    print(f"[DEBUG] Diagnostic shape filter saved: {os.path.join(diagnostics_dir, 'extraction_rejection_shape.png')}")
            except Exception as e:
                print(f"[DEBUG] Could not create shape filter diagnostic: {e}")

            # 4. Modifications filter: scatter coelution vs shape (lavender=passed, orange=rejected for modifications)
            try:
                coel_mod, shape_mod, mod_rej = [], [], []
                for w in all_peak_windows:
                    c = w.get('coelution_score')
                    s = w.get('shape_corr') or w.get('mean_shape_corr')
                    if c is not None and s is not None and pd.notna(c) and pd.notna(s):
                        coel_mod.append(float(c))
                        shape_mod.append(float(s))
                        rr = str(w.get('rejection_reason', '')).lower()
                        mod_rej.append(w.get('status') == 'rejected' and (('excluded' in rr and 'modification' in rr) or 'has mods' in rr or 'excluded mods' in rr))
                if coel_mod and shape_mod:
                    coel_a = np.array(coel_mod)
                    shape_a = np.array(shape_mod)
                    mod_rej_a = np.array(mod_rej)
                    passed = ~mod_rej_a
                    fig_m, ax_m = plt.subplots(figsize=(10, 8))
                    fig_m.patch.set_facecolor('black')
                    ax_m.set_facecolor('black')
                    if np.any(passed):
                        ax_m.scatter(coel_a[passed], shape_a[passed], c='lavender', alpha=0.6, s=25, label=f'Passed mods (n={np.sum(passed)})')
                    if np.any(mod_rej_a):
                        ax_m.scatter(coel_a[mod_rej_a], shape_a[mod_rej_a], c='orange', alpha=0.6, s=25, label=f'Rejected mods (n={np.sum(mod_rej_a)})')
                    ax_m.set_xlabel('Coelution score', fontsize=12, color='0.85')
                    ax_m.set_ylabel('Shape correlation', fontsize=12, color='0.85')
                    ax_m.set_title('Extraction: coelution vs shape (modifications filter)', fontsize=14, color='0.9')
                    ax_m.tick_params(colors='0.85')
                    for label in ax_m.get_xticklabels() + ax_m.get_yticklabels():
                        label.set_color('0.85')
                    for spine in ax_m.spines.values():
                        spine.set_color('0.6')
                    ax_m.legend(loc='lower left', fontsize=10, facecolor='#1a1a1a', edgecolor='0.6', labelcolor='0.9')
                    ax_m.grid(True, alpha=0.25)
                    plt.tight_layout()
                    fig_m.savefig(os.path.join(diagnostics_dir, 'extraction_rejection_modifications.png'), dpi=150, bbox_inches='tight', facecolor='black')
                    plt.close(fig_m)
                    print(f"[DEBUG] Diagnostic modifications filter saved: {os.path.join(diagnostics_dir, 'extraction_rejection_modifications.png')}")
            except Exception as e:
                print(f"[DEBUG] Could not create modifications filter diagnostic: {e}")

            # Accepted only -> accepted/ (exclude rejected)
            if dataframes_accepted_dir and os.path.isdir(dataframes_accepted_dir) and 'status' in df_base.columns:
                df_acc = df_base[df_base['status'] == 'accepted']
                if len(df_acc) > 0:
                    out_acc = os.path.join(dataframes_accepted_dir, f'{base_name}_chromatograms_accepted.csv')
                    _format_extraction_output_columns(df_acc).to_csv(out_acc, index=False)
                    print(f"[DEBUG] Accepted only (chromatograms) saved to: {os.path.abspath(out_acc)}")
            # Rejected only -> rejected/
            if dataframes_rejected_dir and os.path.isdir(dataframes_rejected_dir) and 'status' in df_base.columns:
                df_rej = df_base[df_base['status'] == 'rejected']
                if len(df_rej) > 0:
                    out_rej = os.path.join(dataframes_rejected_dir, f'{base_name}_chromatograms_rejected.csv')
                    _format_extraction_output_columns(df_rej).to_csv(out_rej, index=False)
                    print(f"[DEBUG] Rejected only (chromatograms) saved to: {os.path.abspath(out_rej)}")

            # Windows integration CSVs: one row per unique (plain_peptide, charge, modifications), identifiers first then integration/collection/drift windows then all other run_visualization stats
            def _build_windows_integration_df(df_subset):
                if df_subset is None or len(df_subset) == 0:
                    return None
                df = df_subset.copy()
                # theoretical_mz: use precursor m/z from Comet (mz column)
                if 'mz' in df.columns and 'theoretical_mz' not in df.columns:
                    df['theoretical_mz'] = df['mz']
                # Drift window = collection ± PEAK_DRIFT_BUFFER_SEC
                if 'collection_min_rt' in df.columns and 'collection_max_rt' in df.columns:
                    df['drift_min_rt'] = df['collection_min_rt'].apply(
                        lambda x: max(0.0, float(x) - PEAK_DRIFT_BUFFER_SEC) if pd.notna(x) else None
                    )
                    df['drift_max_rt'] = df['collection_max_rt'].apply(
                        lambda x: float(x) + PEAK_DRIFT_BUFFER_SEC if pd.notna(x) else None
                    )
                else:
                    df['drift_min_rt'] = None
                    df['drift_max_rt'] = None
                # Deduplicate by (plain_peptide, charge, modifications): keep row with max total_area per group, else first
                key_cols = ['plain_peptide', 'charge', 'modifications']
                if all(c in df.columns for c in key_cols):
                    if 'total_area' in df.columns:
                        df = df.loc[df.groupby(key_cols, dropna=False)['total_area'].idxmax()].reset_index(drop=True)
                    else:
                        df = df.groupby(key_cols, dropna=False).first().reset_index()
                # Column order: identifiers, integration_window, collection_window, drift_window, then rest
                id_cols = ['plain_peptide', 'charge', 'modifications']
                if 'theoretical_mz' in df.columns:
                    id_cols.append('theoretical_mz')
                int_cols = ['detected_peak_min_rt', 'detected_peak_max_rt', 'detected_peak_window_size']
                coll_cols = ['collection_min_rt', 'collection_max_rt', 'collection_window_size']
                drift_cols = ['drift_min_rt', 'drift_max_rt']
                rest = [c for c in df.columns if c not in id_cols + int_cols + coll_cols + drift_cols]
                order = [c for c in id_cols + int_cols + coll_cols + drift_cols + rest if c in df.columns]
                df = df[[c for c in order if c in df.columns]]
                return df

            if dataframes_dir and 'status' in df_base.columns:
                df_acc_wi = _build_windows_integration_df(df_base[df_base['status'] == 'accepted'] if 'status' in df_base.columns else None)
                if df_acc_wi is not None and len(df_acc_wi) > 0:
                    try:
                        from visualization.csv_validation import validate_and_log
                        validate_and_log(
                            df_acc_wi,
                            source_name=base_name + '_accepted_windows_integration',
                            required_columns=('plain_peptide', 'charge'),
                            validate_identifiers=True,
                            validate_numeric=True,
                            validate_windows=True,
                        )
                    except Exception as e:
                        print(f"[DEBUG] Validation warning: {e}")
                    out_acc_wi = os.path.join(dataframes_dir, f'{base_name}_accepted_windows_integration.csv')
                    _format_extraction_output_columns(df_acc_wi).to_csv(out_acc_wi, index=False)
                    print(f"[DEBUG] Accepted windows integration (one row per peptide) saved to: {os.path.abspath(out_acc_wi)}")
                    if dataframes_accepted_dir and os.path.isdir(dataframes_accepted_dir):
                        out_acc_wi_acc = os.path.join(dataframes_accepted_dir, f'{base_name}_accepted_windows_integration.csv')
                        _format_extraction_output_columns(df_acc_wi).to_csv(out_acc_wi_acc, index=False)
                df_rej_wi = _build_windows_integration_df(df_base[df_base['status'] == 'rejected'] if 'status' in df_base.columns else None)
                if df_rej_wi is not None and len(df_rej_wi) > 0:
                    try:
                        from visualization.csv_validation import validate_and_log
                        validate_and_log(
                            df_rej_wi,
                            source_name=base_name + '_rejected_windows_integration',
                            required_columns=('plain_peptide', 'charge'),
                            validate_identifiers=True,
                            validate_numeric=True,
                            validate_windows=True,
                        )
                    except Exception as e:
                        print(f"[DEBUG] Validation warning: {e}")
                    out_rej_wi = os.path.join(dataframes_dir, f'{base_name}_rejected_windows_integration.csv')
                    _format_extraction_output_columns(df_rej_wi).to_csv(out_rej_wi, index=False)
                    print(f"[DEBUG] Rejected windows integration (one row per peptide) saved to: {os.path.abspath(out_rej_wi)}")
                    if dataframes_rejected_dir and os.path.isdir(dataframes_rejected_dir):
                        out_rej_wi_rej = os.path.join(dataframes_rejected_dir, f'{base_name}_rejected_windows_integration.csv')
                        _format_extraction_output_columns(df_rej_wi).to_csv(out_rej_wi_rej, index=False)

        # If a filter CSV was used, merge quantification and collection window columns into it and save
        if filter_csv and len(all_peak_windows) > 0:
            def _norm_mods_merge(m):
                if m is None or (isinstance(m, float) and pd.isna(m)):
                    return '-'
                s = str(m).strip()
                return s if s and s.lower() != 'nan' else '-'
            try:
                with open(filter_csv, 'r') as f:
                    filter_first = f.readline()
                filter_skip = 1 if 'CometVersion' in filter_first else 0
            except Exception:
                filter_skip = 0
            df_filter_read = pd.read_csv(filter_csv, sep=',', skiprows=filter_skip, engine='python', quotechar='"', on_bad_lines='warn')
            if 'plain_peptide' in df_filter_read.columns:
                # Build lookup: (peptide, charge, mods) -> window row
                windows_lookup = {}
                for w in all_peak_windows:
                    pep = str(w.get('peptide', '')).strip()
                    ch = int(w.get('charge', 0)) if w.get('charge') is not None else 0
                    mod = _norm_mods_merge(w.get('modifications', '-'))
                    windows_lookup[(pep, ch, mod)] = w
                # Add window columns to filter CSV (total_area for RT windows color-coding; apex_intensity for labeling)
                window_cols = [
                    'detected_peak_min_rt', 'detected_peak_max_rt', 'detected_peak_window_size',
                    'collection_min_rt', 'collection_max_rt', 'collection_window_size',
                    'apex_intensity', 'total_area'
                ]
                for c in window_cols:
                    if c not in df_filter_read.columns:
                        df_filter_read[c] = None
                for idx, row in df_filter_read.iterrows():
                    key = (
                        str(row.get('plain_peptide', '')).strip(),
                        int(row.get('charge', 0)) if pd.notna(row.get('charge')) else 0,
                        _norm_mods_merge(row.get('modifications', '-'))
                    )
                    if key in windows_lookup:
                        w = windows_lookup[key]
                        for c in window_cols:
                            if c in w and w[c] is not None:
                                df_filter_read.at[idx, c] = w[c]
                # Write back to the existing filter CSV (add/update window columns in place)
                _format_extraction_output_columns(df_filter_read).to_csv(filter_csv, index=False)
                print(f"[DEBUG] Quantification and collection windows added to: {os.path.abspath(filter_csv)}")
            else:
                print(f"[DEBUG] Filter CSV missing 'plain_peptide'; skipping merge into filtered_peptides.csv")
    else:
        print(f"[DEBUG] No peak windows found")
    
    print("[DEBUG] ========================================")
    print(f"[DEBUG] Done! Individual peptide figures:")
    print(f"[DEBUG]   accepted/ -> {output_dir_abs}")
    print(f"[DEBUG]   rejected/ -> {rejected_dir_abs}")
    print(f"[DEBUG] Combined figure saved to: {output_png_abs}")
    accepted_count = len([w for w in all_peak_windows if w['status'] == 'accepted']) if all_peak_windows else 0
    rejected_count = len([w for w in all_peak_windows if w['status'] == 'rejected']) if all_peak_windows else 0
    print(f"[DEBUG] Total individual figures created: {len(all_peak_windows)} ({accepted_count} accepted, {rejected_count} rejected)")
    print("[DEBUG] ========================================")


def _parse_chromatogram_args():
    parser = argparse.ArgumentParser(
        description='Extract MS1 chromatograms for each unique peptide and determine peak windows. '
                    'Two-phase: --extract-only writes metrics + traces; --plot-only reads them and writes accepted/rejected dataframes and plots.'
    )
    parser.add_argument('comet_csv', nargs='?', default=None,
                        help='Input CSV file from Comet with MS1 retention times (full data source). Required unless --plot-only.')
    parser.add_argument('raw_file', nargs='?', default=None, help='Input mzML file. Required unless --plot-only.')
    parser.add_argument('output_png', nargs='?', help='Output PNG file (default: adds _chromatograms.png suffix)')
    parser.add_argument('--filter-csv', dest='filter_csv', default=None,
                        help='Optional: CSV listing peptides to include (e.g. filtered_peptides.csv). Only rows in comet_csv whose (plain_peptide, charge, mods) match this list are processed.')
    parser.add_argument('--exclude-mods', dest='exclude_mods', action='store_true',
                        help='Exclude peptides with modifications: save their chromatograms to chromatograms_peptides_rejected and do not include them in overlay plots.')
    parser.add_argument('--no-psm-filter', dest='no_psm_filter', action='store_true',
                        help='No filter: do not reject for PSM scores (Q/PEP/E) or isotope envelope (M0>M+1>M+2); all peptides go to accepted (unless --exclude-mods).')
    parser.add_argument('--test', action='store_true', help='Test mode: randomly sample 30 peptides (faster)')
    # Two-phase: extract -> dataframe + traces; plot -> from dataframe
    parser.add_argument('--extract-only', dest='extract_only', action='store_true',
                        help='Only extract chromatograms and save chromatogram_metrics_all.csv + chromatogram_traces.npz (no plots).')
    parser.add_argument('--plot-only', dest='plot_only', action='store_true',
                        help='Only create plots from existing chromatogram_metrics_all.csv and chromatogram_traces.npz; apply filtering and write accepted/rejected dataframes and plots.')
    parser.add_argument('--chromatogram-metrics-csv', dest='chromatogram_metrics_csv', default=None,
                        help='Path to chromatogram_metrics_all.csv (from --extract-only). Required for --plot-only.')
    parser.add_argument('--chromatogram-traces', dest='chromatogram_traces', default=None,
                        help='Path to chromatogram_traces.npz (from --extract-only). Required for --plot-only.')
    parser.add_argument('--extract-output-dir', dest='extract_output_dir', default=None,
                        help='Directory for chromatogram_metrics_all.csv and chromatogram_traces.npz when using --extract-only (default: same dir as output_png).')
    return parser.parse_args()


if __name__ == '__main__':
    run_chromatograms(_parse_chromatogram_args())
