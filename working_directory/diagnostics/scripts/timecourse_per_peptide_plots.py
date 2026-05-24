"""Per-peptide HDX timecourse diagnostic plots.

For each unique (target_mz, charge) in the UN type5 matrix, render a row per
HDX exposure with six columns, plus one fragment-zoom PNG per registry precursor (MS2|MS3 pairs):
  1) MS1 XIC over the instrument MS2 isolation window (±Da from mzML precursor)
     plus registry peptide M0 traces at ±ppm tolerance from step 1
  2) MS2 fragment-sum chromatogram with the continuity-tracked integration window
  3) Registry peptide isotope XICs (M0–M+4 + sum) zoomed to the blue integration window
     (UN: ±ppm at theoretical isotopes; HDX: +Da shift toward target m/z with wider ±ppm)
  4) Integrated MS1 spectrum with registry isotope peaks color-coded (col3 palette)
  5) Integrated MS2 spectrum (e.g. ETD) with matched fragment labels
  6) Integrated MS3 spectrum (e.g. scramHCD_ETD) with matched fragment labels

Separate PNG per registry precursor: rows = UN → HDX timepoints; one column per fragment
with MS2 (grey raw / blue matched / dark-blue fit) and MS3 (brown raw / orange matched /
dark-orange fit) overlaid on the same subplot (MS3 under MS2), ordered N-terminal c ions
short→long then C-terminal z ions long→short.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.transforms import Bbox

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from targeted_hxc_core import (  # noqa: E402
    REL_H,
    build_row_labels,
    collect_candidates_for_peptide,
    detect_candidates,
    integrate_ms1,
    integrate_msx,
    ms_frag_chrom,
    nearest_keys,
    select_with_continuity,
    xic_ms1_isowin,
    xic_ms1_ppm,
)
from identify_peptide_seq_and_m0_from_isolation_window_mz import (  # noqa: E402
    GET_NA_ISOTOPE,
    parse_comet_params,
    predict_theoretical_fragments,
)
from fragment_hdx_fits import (  # noqa: E402
    exposure_label_to_fit_time,
    fit_cache_dir,
    fragment_has_proline,
    maybe_run_dual_pop_fragment_fit,
    per_scan_spectra_in_window,
    predicted_fit_envelope_peaks,
    run_cached_fragment_hdx_fit,
    run_cached_precursor_ms1_hdx_fit,
    scale_fit_envelope_to_observed,
)

COMET_ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
if COMET_ROOT not in sys.path:
    sys.path.insert(0, COMET_ROOT)
from visualization.chromatograms import matches_b_or_y_ion  # noqa: E402

try:
    from brainpy import isotopic_variants  # noqa: E402
except ImportError:
    isotopic_variants = None  # type: ignore[misc, assignment]
try:
    from pyteomics import mass as _pyteomics_mass  # noqa: E402
except ImportError:
    _pyteomics_mass = None  # type: ignore[misc, assignment]
try:
    from pyHXExpress.pyhxexpress.hxex import count_amides as _pyhx_count_amides  # noqa: E402
except ImportError:
    try:
        from pyhxexpress.hxex import count_amides as _pyhx_count_amides  # noqa: E402
    except ImportError:
        _pyhx_count_amides = None  # type: ignore[misc, assignment]

DEFAULT_M0_PPM_TOL = 5.0
# UN col1: show registry M0 overlays only when the trace co-elutes with the
# integrated peak and has real signal in the collection RT window (not flat noise).
MIN_UN_COL1_PRESENT_PEAK_FRAC = 0.10
MIN_UN_COL1_PRESENT_ABS_INTENSITY = 1e4
MIN_UN_COL1_PRESENT_MIN_NONZERO_FRAC = 0.08
MIN_UN_COL1_SHOULDER_MIN_NONZERO_FRAC = 0.04
# Max |M0 apex − integration apex| for shoulder peptides still shown on col1.
UN_COL1_SHOULDER_APEX_SLACK_SEC = 20.0
UN_COL1_SHOULDER_APEX_SLACK_INT_WIDTH_MULT = 1.5
# UN col3: carry a registry group into HDX rows only if its isotope-sum XIC peak
# in the integration window reaches this fraction of the strongest group (and abs floor).
MIN_UN_COL3_PRESENT_PEAK_FRAC = 0.05
MIN_UN_COL3_PRESENT_ABS_INTENSITY = 1e4
MIN_UN_COL3_PRESENT_MIN_NONZERO_FRAC = 0.12

# Match hdx_filter_app.py theme (banner blue + cream panels).
UI_BANNER_BLUE = '#62c1e5'
UI_INTEGRATION_FILL = '#b8e8f8'  # lighter powder-blue hue for integration window
UI_INTEGRATION_ALPHA = 0.55
UI_COLLECTION_FILL = '#f6f0df'
UI_COLLECTION_BORDER = '#d8ccb3'
UI_INK = '#1f2933'
TARGET_ISO_TRACE = UI_INK
SPECTRUM_TRACE_COLOR = '#000000'
CHROM_TRACE_LW = 1.85
SPECTRUM_TRACE_LW = 1.85

# Sorbet-sunset palette for registry peptide traces (high contrast on white/cream).
# Order matters: alternate warm/cool so co-plotted peptides never pair pink+orange.
PEPTIDE_GROUP_COLORS = (
    '#f472b6',  # pink (warm)
    '#9333ea',  # purple (cool)
    '#f97316',  # orange (warm)
    '#4ade80',  # green (cool)
    '#d946ef',  # magenta (warm)
    '#0891b2',  # turquoise (cool)
    '#fb7185',  # coral (warm)
    '#7c3aed',  # violet (cool)
)
M0_TRACE_COLORS = PEPTIDE_GROUP_COLORS  # backward-compatible alias
# Fully saturated opaque colors for col1 registry M0 overlays (visible on cream background).
M0_TRACE_COLORS_SATURATED = (
    '#be185d',  # deep pink
    '#6b21a8',  # deep purple
    '#c2410c',  # deep orange
    '#15803d',  # deep green
    '#a21caf',  # deep magenta
    '#0e7490',  # deep cyan
    '#e11d48',  # deep coral
    '#5b21b6',  # deep violet
)
M0_TRACE_LW = 2.4
M0_TRACE_LW_UN = 4.4  # thicker dashed M0 overlays on UN col1
M0_TRACE_LSTYLES = ('--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 2, 1, 2)))

BASE = '/Users/rebeccaskotheim/Comet/working_directory'
DEFAULT_MZML_DIR = f'{BASE}/data/mzml/ube2d3_may19'
DEFAULT_TYPE1 = f'{BASE}/terminology/ube2d3_may19/file_name_variables_matrix_type1_hdx_exposures.csv'
DEFAULT_TYPE5 = f'{BASE}/terminology/ube2d3_may19/file_name_variables_matrix_type5.csv'
DEFAULT_OUT_DIR = f'{BASE}/extraction/ube2d3_may19/diagnostics/timecourse_per_peptide'
DEFAULT_REGISTRY = f'{BASE}/targeted_etd/ube2d3_may19/ube2d3_may19_coisolated_peptides_for_hdx_fits.csv'
DEFAULT_HDX_MANIFEST = (
    f'{BASE}/targeted_etd/ube2d3_may19/build_tables/ube2d3_may19_hxexpress_input_manifest.csv'
)
DEFAULT_COMET_PARAMS = f'{BASE}/../comet.params.new'
DEFAULT_FRAG_PPM_TOL = 5.0
MIN_FRAG_ISOTOPE_PEAKS = 2
FRAG_ISOTOPE_MAX_K = 5
FRAG_ISOTOPE_MIN_FRAC = 0.02
# MS1 precursor matching: keep weak isotope satellites that are still above noise.
MS1_ISOTOPE_MIN_FRAC = 0.001
MS1_M0_REFINE_PPM_MULT = 4.0
MS1_M0_REFINE_MIN_DA = 0.025
MS1_ISOTOPE_PPM_K_SCALE = 0.12  # widen match tolerance for higher isotope index k
FRAG_SPECTRUM_PLOT_BIN_SIZE = 0.05  # coarse bins for MS2/MS3 display only
FRAG_ZOOM_BIN_SIZE = 0.01
FRAG_ZOOM_PAD_DA = 12.0
PLOT_DPI = 180
ZOOM_SUBPLOT_CELL_IN = 2.25
ZOOM_SUBPLOT_WIDTH_MULT = 2.25
TIMECOURSE_SUBPLOT_CELL_IN = 3.75
TIMECOURSE_SUBPLOT_WIDTH_MULT = 2.25
TIMECOURSE_N_COLS = 6
SAVE_PAD_INCHES = 0.18
MS1_ISOTOPE_MATCH_LW = 3.8
FRAG_MATCH_LW = 3.2
ZOOM_COLORED_STEM_Z = 2
ZOOM_BLACK_TRACE_Z = 5
ZOOM_M0_REF_Z = 1
ZOOM_LABEL_Z = 8
ZOOM_OVERLAY_STEM_ALPHA_UN = 0.38
ZOOM_OVERLAY_STEM_ALPHA_HDX = 0.30
# Combined MS2+MS3 zoom: MS3 orange on top of MS2 blue — separate matched-stem opacity.
ZOOM_MS2_MATCH_STEM_ALPHA_UN = 0.50
ZOOM_MS2_MATCH_STEM_ALPHA_HDX = 0.40
ZOOM_MS3_MATCH_STEM_ALPHA_UN = 0.96
ZOOM_MS3_MATCH_STEM_ALPHA_HDX = 0.90
ZOOM_MS2_MATCH_STEM_ALPHA_OVERLAY = 0.32
ZOOM_MS3_MATCH_STEM_ALPHA_OVERLAY = 0.82
ZOOM_OVERLAY_M0_ALPHA = 0.40
ZOOM_OVERLAY_LABEL_BBOX_ALPHA = 0.72
ZOOM_FIT_STEM_COLOR = '#555555'
ZOOM_FIT_STEM_LW = 1.6
ZOOM_FIT_STEM_ALPHA = 0.55
ZOOM_FIT_LABEL_Z = 9
# Dabs_std_1 error bars: normalize within each trace, then scale to this fraction of y-span.
DABS_STD_VISIBLE_FRAC = 0.12
# Top D-uptake timecourse row height vs fragment spectrum rows in zoom figures.
DABS_TIMECOURSE_ROW_HEIGHT_MULT = 4.0
ZOOM_GRID_WSPACE = 0.08
ZOOM_GRID_HSPACE = 0.10
ZOOM_SAVE_PAD_INCHES = 0.08
# Combined MS2+MS3 zoom overlay (one column per fragment).
ZOOM_MS2_RAW_COLOR = '#9ca3af'
ZOOM_MS3_RAW_COLOR = '#a16207'
ZOOM_MS2_MATCH_COLORS = ('#1d4ed8', '#2563eb', '#3b82f6', '#60a5fa', '#93c5fd')
ZOOM_MS2_FIT_COLORS = ('#1e3a8a', '#1e40af', '#1d4ed8', '#2563eb', '#3b82f6')
ZOOM_MS3_MATCH_COLORS = ('#c2410c', '#ea580c', '#f97316', '#fb923c', '#fdba74')
ZOOM_MS3_FIT_COLORS = ('#7c2d12', '#9a3412', '#c2410c', '#ea580c', '#f97316')
ZOOM_MS3_RAW_Z = 1
ZOOM_MS3_STEM_Z = 2
ZOOM_MS3_FIT_Z = 3
ZOOM_MS3_M0_REF_Z = 4
ZOOM_MS2_RAW_Z = 5
ZOOM_MS2_STEM_Z = 6
ZOOM_MS2_FIT_Z = 7
ZOOM_MS2_M0_REF_Z = 8
BIMODAL_BADGE_COLOR = '#b45309'
BIMODAL_POP_LINESTYLES = ('--', ':')
DEFAULT_FRAGMENT_FITS_ROOT = f'{BASE}/hdx_fits/ube2d3_may19/fragment_fits'

C13 = 1.0033548378
DEUT_MASS_DA = 1.006277  # H→D mass increase per exchanged amide
# ExMS cumulative deuteration shifts (pyHXExpress peak_picker); used for HDX envelope centers.
EXMS_DELTA_M = (1.003355, 1.003355, 1.004816, 1.004816, 1.006277)
N_PEPTIDE_ISOTOPES = 5  # M0 through M+4 (display labels only; matching uses full NA envelope)
ISO_XIC_LABELS = ('M0', 'M+1', 'M+2', 'M+3', 'M+4')
# Lightness steps for isotopes of a single peptide (avoid rainbow pinks on one trace).
ISO_XIC_COLORS_SINGLE = ('#9d174d', '#db2777', '#f472b6', '#f9a8d4', '#fce7f3')
MS1_HDX_DEUT_COLORS = (
    '#dbeafe', '#93c5fd', '#60a5fa', '#3b82f6', '#2563eb',
    '#1d4ed8', '#1e40af', '#1e3a8a', '#172554', '#0f172a',
)
PEPTIDE_SUM_LSTYLES = ('-', '--', '-.', ':', (0, (5, 2, 1, 2)), (0, (3, 1, 1, 1, 1, 1)))
PEPTIDE_SUM_LWS = (2.2, 2.9, 3.6, 4.3, 5.0)
PEPTIDE_SUM_ALPHAS = (0.95, 0.88, 0.82, 0.76, 0.70)
PEPTIDE_ISO_LWS = (0.85, 1.0, 1.15, 1.3, 1.45)


def _peptide_group_color(gi: int) -> str:
    return PEPTIDE_GROUP_COLORS[gi % len(PEPTIDE_GROUP_COLORS)]


def _m0_trace_color(gi: int) -> str:
    """Registry peptide color — matches col3 isotope XIC palette."""
    return _peptide_group_color(gi)


def _blend_hex_colors(hex_a: str, hex_b: str, t: float) -> str:
    """Linear blend between two #RRGGBB colors (t=0 → a, t=1 → b)."""
    t = float(np.clip(t, 0.0, 1.0))
    a = str(hex_a or '').strip().lstrip('#')
    b = str(hex_b or '').strip().lstrip('#')
    if len(a) != 6 or len(b) != 6:
        return _peptide_group_color(0)
    out: list[str] = []
    for i in range(0, 6, 2):
        av = int(a[i:i + 2], 16)
        bv = int(b[i:i + 2], 16)
        out.append(f'{int(round(av + (bv - av) * t)):02x}')
    return f'#{"".join(out)}'


def _peptide_isotope_trace_color(gi: int, iso_index: int) -> str:
    """Isotope XIC shade for peptide gi (M0 darkest → M+4 lightest), matches col1 group hue."""
    base = _peptide_group_color(gi)
    k = max(0, int(iso_index))
    if k == 0:
        return _darken_color(base, 0.82)
    return _blend_hex_colors(base, '#ffffff', min(0.70, 0.14 * k))


def _peptide_group_linestyle(gi: int) -> str | tuple:
    return M0_TRACE_LSTYLES[gi % len(M0_TRACE_LSTYLES)]


def _zoom_subplot_cell_in(n_cols: int, n_plot_rows: int) -> float:
    """Scale zoom panel height down slightly for very wide/tall grids."""
    cell = float(ZOOM_SUBPLOT_CELL_IN)
    if n_cols > 12:
        cell *= 0.84
    elif n_cols > 8:
        cell *= 0.90
    elif n_cols > 5:
        cell *= 0.95
    if n_plot_rows > 16:
        cell *= 0.88
    elif n_plot_rows > 12:
        cell *= 0.94
    return cell


def _timecourse_subplot_cell_in(n_rows: int) -> float:
    cell = float(TIMECOURSE_SUBPLOT_CELL_IN)
    if n_rows > 14:
        cell *= 0.90
    elif n_rows > 10:
        cell *= 0.95
    return cell


def _gridspec_spacing(n_rows: int, n_cols: int) -> dict[str, float]:
    """Extra wspace/hspace so axis labels and legends do not collide."""
    return {
        'wspace': min(0.55, 0.22 + 0.025 * n_cols),
        'hspace': min(0.65, 0.28 + 0.018 * n_rows),
    }


def _zoom_gridspec_spacing(n_rows: int, n_cols: int) -> dict[str, float]:
    """Tighter wspace/hspace for fragment zoom grids."""
    return {
        'wspace': min(0.22, ZOOM_GRID_WSPACE + 0.006 * n_cols),
        'hspace': min(0.28, ZOOM_GRID_HSPACE + 0.004 * n_rows),
    }


def _darken_color(hex_color: str, factor: float = 0.58) -> str:
    """Return a darker shade of a #RRGGBB color for readable labels."""
    raw = str(hex_color or '').strip().lstrip('#')
    if len(raw) != 6:
        return TARGET_ISO_TRACE
    r = int(raw[0:2], 16)
    g = int(raw[2:4], 16)
    b = int(raw[4:6], 16)
    f = float(factor)
    return f'#{int(r * f):02x}{int(g * f):02x}{int(b * f):02x}'


def _shade_integration_window(ax, x_lo: float, x_hi: float, *, zorder: float = 0.2) -> None:
    """Apply the same integration-window blue used in chromatogram columns."""
    ax.axvspan(
        x_lo, x_hi,
        facecolor=UI_INTEGRATION_FILL,
        alpha=UI_INTEGRATION_ALPHA,
        zorder=zorder,
        linewidth=0,
    )


def _shade_integration_panel(ax, x_lo: float, x_hi: float) -> None:
    """Full-panel integration blue for integrated MS1/MS2/MS3 spectrum columns."""
    ax.set_facecolor('white')
    _shade_integration_window(ax, x_lo, x_hi, zorder=0)


# --- Fragment matching for col3 registry peptides only (MS2/MS3: ±5ppm + b/y rejection + ≥2 isotopes) ---

def _collect_msx_raw_peaks(
    ms_by,
    target_mz: float,
    level: int,
    rt_lo: float,
    rt_hi: float,
    mz_lo: float,
    mz_hi: float,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Concatenate centroid m/z and intensities across scans in an RT window."""
    keys = nearest_keys(ms_by, target_mz, level)
    allm, alli = [], []
    n = 0
    for k in keys:
        for rt, mzs, intensities, _pmz in ms_by.get((k, level), []):
            if rt_lo <= rt <= rt_hi:
                m = (mzs >= mz_lo) & (mzs <= mz_hi)
                if np.any(m):
                    allm.append(mzs[m])
                    alli.append(intensities[m])
                    n += 1
    if not allm:
        return n, np.array([]), np.array([])
    return n, np.concatenate(allm), np.concatenate(alli)


def _sum_peaks_by_exact_mz(raw_m: np.ndarray, raw_i: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sum repeated centroid m/z values into one intensity per exact m/z."""
    if raw_m.size == 0:
        return raw_m, raw_i
    uniq_m, inv = np.unique(raw_m, return_inverse=True)
    uniq_i = np.zeros(uniq_m.size, dtype=float)
    np.add.at(uniq_i, inv, raw_i)
    return uniq_m, uniq_i

def _comet_ion_name(label: str) -> str:
    """Convert a theoretical fragment label to Comet ion name for b/y overlap checks."""
    raw = str(label).strip()
    m_paren = re.match(r'^\(([abcxyz])([+-]\d+)\)(\d+)', raw)
    if m_paren:
        base_type = m_paren.group(1)
        variant = m_paren.group(2)
        ordinal = m_paren.group(3)
        if base_type == 'z' and variant == '+1':
            return f'z1_{ordinal}'
        return f'{base_type}{ordinal}'
    raw = re.sub(r'[\s_]\d+\+$', '', raw)
    raw = re.sub(r'-(?:H2O|NH3|CO|H3PO4|HPO3)$', '', raw, flags=re.IGNORECASE)
    if raw.endswith('+1') and raw.startswith('z') and raw[1:-2].isdigit():
        return f'z1_{raw[1:-2]}'
    return raw


def _display_fragment_label(label: str) -> str:
    raw = str(label).strip()
    raw = re.sub(r'[\s_]\d+\+$', '', raw)
    return raw


def _col3_registry_peptides(
    m0_groups: list[dict[str, object]],
    un_col3_group_indices: set[int] | None = None,
) -> list[dict[str, object]]:
    """Unique peptides plotted in column 3, preserving group color index."""
    out: list[dict[str, object]] = []
    seen: set[str] = set()
    for gi, grp in enumerate(m0_groups):
        if un_col3_group_indices is not None and int(gi) not in un_col3_group_indices:
            continue
        clr = _peptide_group_color(gi)
        for row in grp.get('peptides', []):
            pep = str(row.get('peptide_sequence', '') or '').strip()
            if not pep or pep in seen:
                continue
            seen.add(pep)
            out.append({'gi': gi, 'peptide': pep, 'color': clr})
    return out


def _cached_theoretical_fragments(
    peptide: str,
    precursor_charge: int,
    comet_cfg: dict,
    cache: dict[tuple[str, int], list],
) -> list:
    key = (str(peptide), int(precursor_charge))
    if key not in cache:
        cache[key] = predict_theoretical_fragments(str(peptide), int(precursor_charge), comet_cfg)
    return cache[key]


def _count_fragment_isotope_peaks(
    mc: np.ndarray,
    hc: np.ndarray,
    center_mz: float,
    charge: int,
    ppm_tol: float = DEFAULT_FRAG_PPM_TOL,
    min_peaks: int = MIN_FRAG_ISOTOPE_PEAKS,
    max_k: int = FRAG_ISOTOPE_MAX_K,
    min_frac: float = FRAG_ISOTOPE_MIN_FRAC,
) -> int:
    """Count isotopic peaks M0..M+max_k within ±ppm of center_mz + k*C13/z."""
    if mc.size == 0 or hc.size == 0 or charge <= 0 or center_mz <= 0:
        return 0
    spacing = C13 / int(charge)
    ppm_tol = float(ppm_tol)
    env: list[float] = []
    for k in range(int(max_k) + 1):
        target = float(center_mz) + (k * spacing)
        tol = target * ppm_tol * 1e-6
        m = np.abs(mc - target) <= tol
        env.append(float(hc[m].max()) if np.any(m) else 0.0)
    peak_int = max(env) if env else 0.0
    if peak_int <= 0:
        return 0
    floor = peak_int * float(min_frac)
    return sum(1 for intensity in env if intensity >= floor)


def _fragment_isotope_peak_list(
    mc: np.ndarray,
    hc: np.ndarray,
    center_mz: float,
    charge: int,
    ppm_tol: float = DEFAULT_FRAG_PPM_TOL,
    max_k: int = FRAG_ISOTOPE_MAX_K,
    min_frac: float = FRAG_ISOTOPE_MIN_FRAC,
) -> list[dict[str, float]]:
    """Observed isotope peaks M0..M+max_k for one matched fragment envelope."""
    if mc.size == 0 or hc.size == 0 or charge <= 0 or center_mz <= 0:
        return []
    spacing = C13 / int(charge)
    ppm_tol = float(ppm_tol)
    raw: list[dict[str, float]] = []
    for k in range(int(max_k) + 1):
        target = float(center_mz) + (k * spacing)
        tol = target * ppm_tol * 1e-6
        m = np.abs(mc - target) <= tol
        if not np.any(m):
            continue
        idxs = np.where(m)[0]
        best = int(idxs[int(np.argmax(hc[idxs]))])
        raw.append(
            {
                'k': float(k),
                'obs_mz': float(mc[best]),
                'obs_int': float(hc[best]),
            }
        )
    if not raw:
        return []
    peak_int = max(pk['obs_int'] for pk in raw)
    floor = peak_int * float(min_frac)
    return [pk for pk in raw if pk['obs_int'] >= floor]


def _fragment_na_isotope_npeaks(frag_seq: str, charge: int) -> int:
    """Number of NA isotope components from pyHXExpress for a fragment sequence."""
    _offsets, intensities = _na_isotope_envelope_offsets(frag_seq, charge)
    if intensities.size:
        return int(intensities.size)
    return FRAG_ISOTOPE_MAX_K + 1


def _peptide_isotope_composition(peptide: str, *, d: int = 0) -> dict[str, int]:
    """Element composition for brainpy isotopic_variants (pyHXExpress get_na_isotope convention).

    d>0 adds that many deuterons (exchangeable amide H→D) for d×NA isotope convolution.
    """
    seq = str(peptide or '').strip().upper()
    if not seq:
        return {}
    if _pyteomics_mass is not None:
        comp = dict(_pyteomics_mass.Composition(seq))
    else:
        comp: dict[str, int] = {}
    if _pyhx_count_amides is not None:
        n_hex = int(_pyhx_count_amides(seq, 0.0))
    else:
        n_hex = _exchangeable_amide_count(seq)
    if comp:
        comp['H'] = int(comp.get('H', 0)) - int(n_hex)
        d = max(0, int(d))
        if d > 0:
            comp['D'] = int(comp.get('D', 0)) + d
        return {k: int(v) for k, v in comp.items() if int(v) > 0}
    return {}


def _exms_deut_mz_shift(d: int, charge: int) -> float:
    """Cumulative m/z shift for deuteration level d (matches pyHXExpress peak_picker)."""
    if int(d) <= 0:
        return 0.0
    z = max(1, int(charge))
    total = 0.0
    for nd in range(int(d)):
        dm = EXMS_DELTA_M[nd] if nd < len(EXMS_DELTA_M) else EXMS_DELTA_M[-1]
        total += float(dm) / z
    return total


def _na_isotope_npeaks_for_peptide(seq: str, charge: int, *, d: int = 0) -> int:
    """Peak count for d×NA envelope width (undeuterated conv width + extra for deuteration)."""
    pep = str(seq or '').strip().upper()
    z = max(1, int(charge))
    d = max(0, int(d))
    base = N_PEPTIDE_ISOTOPES
    if pep:
        offsets, _ints = _na_isotope_envelope_offsets(pep, z, d=0, npeaks=None)
        if offsets.size:
            base = int(offsets.size)
    if d <= 0:
        return base
    return base + d + (base // 2)


def _na_isotope_envelope_offsets(
    seq: str,
    charge: int,
    *,
    d: int = 0,
    npeaks: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (m/z offsets from M0, relative intensities) for NA isotope convolution.

    Uses brainpy isotopic_variants full convolved m/z positions (not uniform C13/z).
    For d×NA precursor matching, pass d=0 here and apply deuteration via EXMS M0 shift.
    """
    pep = str(seq or '').strip().upper()
    z = max(1, int(charge))
    d = max(0, int(d))
    spacing = C13 / z
    if not pep:
        k = max(1, int(npeaks or N_PEPTIDE_ISOTOPES))
        idx = np.arange(k, dtype=float)
        return idx * spacing, np.ones(k, dtype=float)

    comp = _peptide_isotope_composition(pep, d=d)
    if isotopic_variants is not None and comp:
        try:
            cluster = list(isotopic_variants(comp, npeaks=npeaks, charge=z))
        except Exception:
            cluster = []
        if cluster:
            mzs: list[float] = []
            ints: list[float] = []
            for peak in cluster:
                mz = getattr(peak, 'mz', None)
                try:
                    mzs.append(float(mz))
                except (TypeError, ValueError):
                    mzs.append(float(len(mzs)) * spacing)
                ints.append(float(getattr(peak, 'intensity', 0.0) or 0.0))
            mzs_arr = np.asarray(mzs, dtype=float)
            ints_arr = np.asarray(ints, dtype=float)
            if mzs_arr.size and np.isfinite(mzs_arr[0]):
                offsets = mzs_arr - float(mzs_arr[0])
                return offsets, ints_arr

    k = max(1, int(npeaks or N_PEPTIDE_ISOTOPES))
    idx = np.arange(k, dtype=float)
    return idx * spacing, np.ones(idx.size, dtype=float)


def _equiv_deut_from_mz_shift(shift_da: float, charge: int) -> int:
    """Approximate deuteration level from cumulative m/z shift (for envelope width)."""
    if shift_da <= 0:
        return 0
    z = max(1, int(charge))
    return max(0, int(round(float(shift_da) * z / DEUT_MASS_DA)))


def _refine_ms1_m0_anchor(
    mc: np.ndarray,
    hc: np.ndarray,
    m0_theory: float,
    charge: int,
    seq: str,
    ppm_tol: float,
    *,
    d: int = 0,
    npeaks: int | None = None,
    search_half_width_da: float | None = None,
) -> float:
    """Shift theoretical M0 to the best-fit observed d×NA isotope envelope anchor."""
    if mc.size == 0 or hc.size == 0 or not np.isfinite(float(m0_theory)):
        return float(m0_theory)
    d = max(0, int(d))
    if npeaks is None:
        npeaks = _na_isotope_npeaks_for_peptide(seq, charge, d=d)
    offsets, _na_ints = _na_isotope_envelope_offsets(seq, charge, d=d, npeaks=npeaks)
    if offsets.size == 0:
        return float(m0_theory)

    ppm_tol = float(ppm_tol)
    env_span = float(offsets[-1]) if offsets.size else 0.0
    search_da = max(
        float(m0_theory) * ppm_tol * MS1_M0_REFINE_PPM_MULT * 1e-6,
        MS1_M0_REFINE_MIN_DA,
        env_span * 0.20,
    )
    if search_half_width_da is not None:
        search_da = max(search_da, float(search_half_width_da))
    best_m0 = float(m0_theory)
    best_score = -1.0
    for delta in np.linspace(-search_da, search_da, 41):
        m0_trial = float(m0_theory) + float(delta)
        score = 0.0
        for k in range(int(offsets.size)):
            target = m0_trial + float(offsets[k])
            tol = max(
                target * ppm_tol * (1.0 + MS1_ISOTOPE_PPM_K_SCALE * k) * 1e-6,
                0.015,
            )
            m = np.abs(mc - target) <= tol
            if np.any(m):
                score += float(hc[m].max())
        if score > best_score:
            best_score = score
            best_m0 = m0_trial
    return best_m0


def _deuteration_peak_matches_from_envelope(
    mc: np.ndarray,
    hc: np.ndarray,
    m0_un: float,
    charge: int,
    seq: str,
    ppm_tol: float,
    *,
    max_deut: int,
    min_frac: float,
    npeaks: int | None = None,
    refine_m0: bool = False,
    refine_search_da: float | None = None,
) -> list[dict[str, float]]:
    """Match observed peaks to d-shifted full get_na_isotope convolved envelopes."""
    if mc.size == 0 or hc.size == 0 or charge <= 0 or m0_un <= 0:
        return []

    ppm_tol = float(ppm_tol)
    matched: dict[int, dict[str, float]] = {}

    npeaks_un = (
        int(npeaks)
        if npeaks is not None
        else _na_isotope_npeaks_for_peptide(seq, charge, d=0)
    )
    m0_anchor_un = float(m0_un)
    if refine_m0:
        # Refine undeuterated M0 once; do not re-refine per d (that collapses every
        # d×NA envelope onto the same brightest peak and leaves only one colored stem).
        m0_anchor_un = _refine_ms1_m0_anchor(
            mc, hc, m0_anchor_un, charge, seq, ppm_tol,
            d=0, npeaks=npeaks_un, search_half_width_da=refine_search_da,
        )

    for d in range(max(0, int(max_deut)) + 1):
        npeaks_d = npeaks_un if int(d) == 0 else _na_isotope_npeaks_for_peptide(seq, charge, d=d)
        # d×NA: EXMS shift + undeuterated get_na_isotope m/z positions (not D-mod + shift).
        offsets, _na_ints = _na_isotope_envelope_offsets(seq, charge, d=0, npeaks=npeaks_d)
        if offsets.size == 0:
            continue

        m0_anchor_d = m0_anchor_un + _exms_deut_mz_shift(d, charge)

        env_span = float(offsets[-1]) if offsets.size else 0.0
        for k in range(int(offsets.size)):
            target = m0_anchor_d + float(offsets[k])
            tol = max(
                target * ppm_tol * (1.0 + MS1_ISOTOPE_PPM_K_SCALE * k + 0.04 * d) * 1e-6,
                0.015 + 0.004 * d,
                env_span * 0.05,
            )
            m = np.abs(mc - target) <= tol
            if not np.any(m):
                continue
            idxs = np.where(m)[0]
            best = int(idxs[int(np.argmax(hc[idxs]))])
            obs_mz = float(mc[best])
            obs_int = float(hc[best])
            dist = abs(obs_mz - target)
            prev = matched.get(best)
            if prev is None or dist < abs(float(prev['obs_mz']) - float(prev['theo_mz'])):
                matched[best] = {
                    'obs_mz': obs_mz,
                    'obs_int': obs_int,
                    'd': float(d),
                    'k': float(k),
                    'anchor_mz': float(m0_anchor_d),
                    'theo_mz': float(target),
                }

    peaks = list(matched.values())
    if not peaks:
        return []
    if float(min_frac) <= 0:
        return peaks
    # Apply intensity floor per deuteration level so weak isotope satellites within
    # each d×NA envelope are kept even when another d level has a much taller M0.
    kept: list[dict[str, float]] = []
    by_d: dict[int, list[dict[str, float]]] = {}
    for pk in peaks:
        by_d.setdefault(int(pk['d']), []).append(pk)
    for group in by_d.values():
        peak_int = max(p['obs_int'] for p in group)
        floor = peak_int * float(min_frac)
        kept.extend(p for p in group if p['obs_int'] >= floor)
    return kept


def _fragment_deuteration_peak_matches(
    mc: np.ndarray,
    hc: np.ndarray,
    spec: dict[str, object],
    ppm_tol: float = DEFAULT_FRAG_PPM_TOL,
    *,
    max_deut: int | None = None,
    min_frac: float = FRAG_ISOTOPE_MIN_FRAC,
) -> list[dict[str, float]]:
    """Match observed peaks to any deuteration level 0..n_ex with NA isotope sub-peaks.

    Each deuteration level d shifts the undeuterated M0 by d*DEUT_MASS_DA/charge and
    places a get_na_isotope envelope on that shifted monoisotopic position.
    """
    m0_un = float(spec['m0_mz'])
    charge = int(spec['charge'])
    n_ex = int(spec.get('n_ex', 0) or 0)
    if max_deut is None:
        max_deut = n_ex
    max_deut = max(0, min(int(max_deut), n_ex))

    if mc.size == 0 or hc.size == 0 or charge <= 0 or m0_un <= 0:
        return []

    frag_seq = str(spec.get('frag_seq', '') or '').strip().upper()
    return _deuteration_peak_matches_from_envelope(
        mc,
        hc,
        m0_un,
        charge,
        frag_seq,
        float(ppm_tol),
        max_deut=max_deut,
        min_frac=float(min_frac),
        npeaks=max(n_ex + 1, FRAG_ISOTOPE_MAX_K + 1),
    )


def _ms1_precursor_na_isotope_npeaks(peptide_seq: str, charge: int, *, d: int = 0) -> int:
    return _na_isotope_npeaks_for_peptide(peptide_seq, charge, d=d)


def _ms1_d_na_theoretical_envelopes(
    m0_un: float,
    charge: int,
    seq: str,
    d: int,
    ppm_tol: float,
    npeaks: int | None = None,
) -> tuple[list[dict[str, float]], float, float]:
    """Full d×get_na_isotope convolved theoretical peaks and envelope m/z bounds.

    Deuteration level d is applied only via EXMS M0 shift; isotope m/z positions come from
    the undeuterated get_na_isotope convolution (full convolved positions, not C13 spacing).
    """
    pep = str(seq or '').strip().upper()
    z = max(1, int(charge))
    d = max(0, int(d))
    if npeaks is None:
        npeaks = _na_isotope_npeaks_for_peptide(pep, z, d=d)
    offsets, _ = _na_isotope_envelope_offsets(pep, z, d=0, npeaks=npeaks)
    m0_d = float(m0_un) + _exms_deut_mz_shift(d, z)
    if offsets.size == 0:
        return [], m0_d, m0_d

    theos: list[dict[str, float]] = []
    edge_tol = 0.0
    for k in range(int(offsets.size)):
        theo = m0_d + float(offsets[k])
        tol = max(
            theo * float(ppm_tol) * (1.0 + MS1_ISOTOPE_PPM_K_SCALE * k + 0.04 * d) * 1e-6,
            0.015 + 0.004 * d,
        )
        edge_tol = max(edge_tol, tol)
        theos.append(
            {
                'd': float(d),
                'k': float(k),
                'theo_mz': float(theo),
                'tol': float(tol),
                'anchor_mz': float(m0_d),
            }
        )
    mz_vals = [float(t['theo_mz']) for t in theos]
    span_pad = max(edge_tol, C13 / z * 0.35)
    return theos, min(mz_vals) - edge_tol - span_pad, max(mz_vals) + edge_tol + span_pad


def _ms1_precursor_deuteration_peak_matches(
    mc: np.ndarray,
    hc: np.ndarray,
    peptide_seq: str,
    mono_mz: float,
    charge: int,
    ppm_tol: float = DEFAULT_M0_PPM_TOL,
    *,
    max_deut: int | None = None,
    min_frac: float = MS1_ISOTOPE_MIN_FRAC,
    npeaks: int | None = None,
    target_mz: float | None = None,
    lbl: str = 'UN',
) -> list[dict[str, float]]:
    """Match integrated MS1 peaks to d×get_na_isotope convolved precursor envelopes.

    Uses registry mono_m/z only (no M0 refinement). UN: d=0 only, ±ppm to each convolved peak.
    HDX: color every observed peak whose m/z falls inside any d×NA envelope span (d=0..n_ex).
    """
    pep = str(peptide_seq or '').strip().upper()
    m0_un = float(mono_mz)
    charge = max(1, int(charge))
    is_un_row = str(lbl).strip().upper() == 'UN'

    if mc.size == 0 or hc.size == 0 or m0_un <= 0:
        return []

    n_ex = _exchangeable_amide_count(pep) if pep else 0
    if max_deut is None:
        max_deut = 0 if is_un_row else n_ex
    else:
        max_deut = max(0, min(int(max_deut), n_ex if not is_un_row else 0))

    ppm_tol = float(ppm_tol)
    npeaks_un = (
        npeaks if npeaks is not None else _na_isotope_npeaks_for_peptide(pep, charge, d=0)
    )

    d_envelopes: list[tuple[int, float, float, list[dict[str, float]]]] = []
    for d in range(int(max_deut) + 1):
        npeaks_d = npeaks_un if d == 0 else _na_isotope_npeaks_for_peptide(pep, charge, d=d)
        theos, lo, hi = _ms1_d_na_theoretical_envelopes(
            m0_un, charge, pep, d, ppm_tol, npeaks_d,
        )
        if theos:
            d_envelopes.append((d, lo, hi, theos))

    matched: dict[int, dict[str, float]] = {}
    for j in range(mc.size):
        if hc[j] <= 0:
            continue
        obs_mz = float(mc[j])
        best_entry: dict[str, float] | None = None
        best_dist = float('inf')

        if is_un_row:
            for _d, _lo, _hi, theos in d_envelopes:
                if int(_d) != 0:
                    continue
                for te in theos:
                    theo = float(te['theo_mz'])
                    tol = float(te['tol'])
                    dist = abs(obs_mz - theo)
                    if dist <= tol and dist < best_dist:
                        best_entry = te
                        best_dist = dist
        else:
            for _d, lo, hi, theos in d_envelopes:
                if obs_mz < lo or obs_mz > hi:
                    continue
                for te in theos:
                    dist = abs(obs_mz - float(te['theo_mz']))
                    if dist < best_dist:
                        best_entry = te
                        best_dist = dist

        if best_entry is None:
            continue
        matched[j] = {
            'obs_mz': obs_mz,
            'obs_int': float(hc[j]),
            'd': float(best_entry['d']),
            'k': float(best_entry['k']),
            'anchor_mz': float(best_entry['anchor_mz']),
            'theo_mz': float(best_entry['theo_mz']),
        }

    return list(matched.values())


def _ms1_deut_stem_color(d: float, k: float, *, is_un_row: bool, gi: int = 0) -> str:
    """Matched MS1 stems: peptide group hue only (purple/pink/…); never HDX blue or washed-out cyan."""
    base = _peptide_group_color(gi)
    if is_un_row or int(d) <= 0:
        return base
    # HDX: keep peptide hue; only a slight lighten with deuteration level.
    t = min(0.22, 0.04 * int(d))
    return _blend_hex_colors(base, '#ffffff', t)


def _parse_ion_ordinal(ion_label: str) -> tuple[str, int] | None:
    """Return (series, ordinal) for c/z fragment labels, e.g. (c+1)11 -> ('c', 11)."""
    raw = _display_fragment_label(ion_label)
    m = re.match(r'^\(([cz])([+-]\d+)\)(\d+)$', raw, re.I)
    if m:
        return m.group(1).lower(), int(m.group(3))
    m = re.match(r'^z(\d+)\+(\d+)$', raw, re.I)
    if m:
        return 'z', int(m.group(1))
    m = re.match(r'^z(\d+)$', raw, re.I)
    if m:
        return 'z', int(m.group(1))
    m = re.match(r'^c(\d+)$', raw, re.I)
    if m:
        return 'c', int(m.group(1))
    return None


def _fragment_residue_sequence(peptide: str, ion_label: str) -> str:
    """Residue substring represented by a matched c/z fragment ion."""
    pep = str(peptide or '').strip().upper()
    parsed = _parse_ion_ordinal(ion_label)
    if not parsed or not pep:
        return pep
    series, n = parsed
    n = min(max(1, int(n)), len(pep))
    if series == 'c':
        return pep[:n]
    return pep[-n:]


def _exchangeable_amide_count(peptide: str) -> int:
    """Exchangeable backbone amides ≈ sequence length minus prolines."""
    seq = str(peptide or '').strip().upper()
    return max(0, len(seq) - seq.count('P'))


def _exchangeable_amide_count_fragment(peptide: str, ion_label: str) -> int:
    """Exchangeable amides within the matched fragment sequence only."""
    return _exchangeable_amide_count(_fragment_residue_sequence(peptide, ion_label))


def _max_deut_mz_shift(n_ex: int, charge: int) -> float:
    """Upper m/z span for full deuteration: n_ex × (H→D mass) / fragment charge."""
    return float(max(0, int(n_ex))) * DEUT_MASS_DA / max(1, int(charge))


def _fragment_hdx_zoom_xlim(
    peptide: str,
    ion_label: str,
    m0_mz: float,
    charge: int,
    *,
    m0_mz_hi: float | None = None,
) -> tuple[float, float]:
    """Fixed zoom window: UN M0 through M0 + charge-scaled max D shift (+ isotope pad)."""
    z = max(1, int(charge))
    spacing = C13 / z
    n_ex = _exchangeable_amide_count_fragment(peptide, ion_label)
    m0_lo = float(m0_mz)
    m0_hi = float(m0_mz_hi) if m0_mz_hi is not None else m0_lo
    max_shift_mz = _max_deut_mz_shift(n_ex, z)
    mz_lo = m0_lo - (spacing * 0.85) - 1.5
    mz_hi = m0_hi + max_shift_mz + (FRAG_ISOTOPE_MAX_K * spacing) + 1.5
    pad = max(2.0, (mz_hi - mz_lo) * 0.04)
    return mz_lo - pad, mz_hi + pad


def _build_fragment_zoom_spec(hit: dict[str, object]) -> dict[str, object]:
    peptide = str(hit.get('peptide', '') or '')
    ion = str(hit.get('ion', '') or '')
    m0_mz = float(hit['obs_mz'])
    charge = int(hit['charge'])
    frag_seq = _fragment_residue_sequence(peptide, ion)
    n_ex = _exchangeable_amide_count_fragment(peptide, ion)
    xlo, xhi = _fragment_hdx_zoom_xlim(peptide, ion, m0_mz, charge)
    return {
        'peptide': peptide,
        'ion': ion,
        'charge': charge,
        'color': str(hit.get('color', TARGET_ISO_TRACE)),
        'm0_mz': m0_mz,
        'frag_seq': frag_seq,
        'n_ex': int(n_ex),
        'xlo': float(xlo),
        'xhi': float(xhi),
    }


def _is_radical_variant_ion(ion_label: str) -> bool:
    """True for (c±N)n / (z±N)n style radical fragment labels."""
    raw = _display_fragment_label(ion_label)
    return bool(re.match(r'^\([cz][+-]\d+\)\d+', raw, re.I))


def _fragment_variant_cluster_key(spec: dict[str, object]) -> tuple[str, str, int]:
    """Group radical/plain variants sharing residue sequence and charge."""
    return (
        str(spec.get('peptide', '') or ''),
        str(spec.get('frag_seq', '') or ''),
        int(spec.get('charge', 0) or 0),
    )


def _fragment_column_sort_key(
    ion: str,
    *,
    frag_seq: str = '',
    m0_mz: float = 0.0,
) -> tuple[int, int, float, str]:
    """Sort key: N-terminal c (short→long), then C-terminal z (long→short)."""
    parsed = _parse_ion_ordinal(ion)
    seq_len = len(str(frag_seq or '').strip())
    if parsed:
        series, n = parsed
        if series == 'c':
            return (0, int(n), float(m0_mz), str(ion))
        if series == 'z':
            return (1, -int(n), float(m0_mz), str(ion))
    return (2, seq_len, float(m0_mz), str(ion))


def _fragment_cluster_column_sort_key(cluster: dict[str, object]) -> tuple[int, int, float, str]:
    variants = list(cluster.get('variants', []) or [])
    if not variants:
        return (3, 0, 0.0, '')
    ms2_spec, _ms3_spec, ion, _sort_mz = variants[0]
    return _fragment_column_sort_key(
        str(ion),
        frag_seq=str(ms2_spec.get('frag_seq', '') or ''),
        m0_mz=float(cluster.get('sort_mz', ms2_spec.get('m0_mz', 0.0))),
    )


def _cluster_radical_variant_zoom_pairs(
    pairs: list[tuple[dict[str, object], dict[str, object], str, float]],
) -> list[dict[str, object]]:
    """Merge radical variants with the same frag_seq onto one MS2|MS3 column pair."""
    by_cluster: dict[tuple[str, str, int], list[tuple[dict[str, object], dict[str, object], str, float]]] = {}
    for pair in pairs:
        key = _fragment_variant_cluster_key(pair[0])
        by_cluster.setdefault(key, []).append(pair)

    clusters: list[dict[str, object]] = []
    for cluster_pairs in by_cluster.values():
        cluster_pairs = sorted(cluster_pairs, key=lambda t: float(t[3]))
        ions = [str(t[2]) for t in cluster_pairs]
        overlay = len(set(ions)) > 1 and any(_is_radical_variant_ion(ion) for ion in ions)
        clusters.append(
            {
                'variants': cluster_pairs,
                'sort_mz': float(cluster_pairs[0][3]),
                'overlay': overlay,
            }
        )
    clusters.sort(key=_fragment_cluster_column_sort_key)
    return clusters


def _variant_overlay_colors(n: int, base_clr: str) -> list[str]:
    """Distinct colors for overlaid radical variants on one axis."""
    if n <= 1:
        return [str(base_clr)]
    return [_peptide_group_color(i) for i in range(n)]


def _zoom_ms2_match_colors(n: int) -> list[str]:
    return [ZOOM_MS2_MATCH_COLORS[i % len(ZOOM_MS2_MATCH_COLORS)] for i in range(max(1, n))]


def _zoom_ms2_fit_colors(n: int) -> list[str]:
    return [ZOOM_MS2_FIT_COLORS[i % len(ZOOM_MS2_FIT_COLORS)] for i in range(max(1, n))]


def _zoom_ms3_match_colors(n: int) -> list[str]:
    return [ZOOM_MS3_MATCH_COLORS[i % len(ZOOM_MS3_MATCH_COLORS)] for i in range(max(1, n))]


def _zoom_ms3_fit_colors(n: int) -> list[str]:
    return [ZOOM_MS3_FIT_COLORS[i % len(ZOOM_MS3_FIT_COLORS)] for i in range(max(1, n))]


def _order_fragment_zoom_columns(specs: list[dict[str, object]]) -> list[dict[str, object]]:
    """One column per unique matched fragment, N-term c short→long then C-term z long→short."""
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, object]] = []
    for spec in sorted(
        specs,
        key=lambda s: _fragment_column_sort_key(
            str(s.get('ion', '') or ''),
            frag_seq=str(s.get('frag_seq', '') or ''),
            m0_mz=float(s.get('m0_mz', 0.0)),
        ),
    ):
        key = (str(spec.get('peptide', '') or ''), str(spec.get('ion', '') or ''))
        if key in seen:
            continue
        seen.add(key)
        out.append(spec)
    return out


def _zoom_specs_by_fragment_key(
    specs: list[dict[str, object]],
) -> dict[tuple[str, str], dict[str, object]]:
    out: dict[tuple[str, str], dict[str, object]] = {}
    for spec in specs:
        key = (str(spec.get('peptide', '') or ''), str(spec.get('ion', '') or ''))
        if key not in out:
            out[key] = spec
    return out


def _registry_peptide_rank_map(m0_groups: list[dict[str, object]]) -> dict[str, int]:
    ranks: dict[str, int] = {}
    for grp in m0_groups:
        for row in grp.get('peptides', []):
            pep = str(row.get('peptide_sequence', '') or '').strip()
            if not pep:
                continue
            rank = int(row.get('rank', 999999) or 999999)
            ranks[pep] = min(rank, ranks.get(pep, rank))
    return ranks


def _mono_mz_for_peptide(
    m0_groups: list[dict[str, object]],
    peptide: str,
) -> float | None:
    pep_u = str(peptide or '').strip().upper()
    if not pep_u:
        return None
    for grp in m0_groups:
        for row in grp.get('peptides', []):
            if str(row.get('peptide_sequence', '') or '').strip().upper() == pep_u:
                mono = float(grp.get('mono_mz', np.nan))
                if np.isfinite(mono):
                    return mono
    return None


def _fit_envelope_peak_rows(
    spec: dict[str, object],
    fit_dabs: float | None,
    match_m: np.ndarray,
    match_i: np.ndarray,
    *,
    frag_ppm_tol: float,
    pop_frac: float = 1.0,
    dual_pop_index: int | None = None,
) -> list[dict[str, object]]:
    if fit_dabs is None or not np.isfinite(float(fit_dabs)):
        return []
    xlo = float(spec['xlo'])
    xhi = float(spec['xhi'])
    pred = predicted_fit_envelope_peaks(spec, float(fit_dabs))
    scaled = scale_fit_envelope_to_observed(
        pred, match_m, match_i, xlo, xhi, ppm_tol=frag_ppm_tol,
    )
    rows: list[dict[str, object]] = []
    for mz_p, height, d_level, _obs_at in scaled:
        if mz_p < xlo or mz_p > xhi or height <= 0:
            continue
        rows.append({
            'mz': float(mz_p),
            'intensity': float(height) * float(pop_frac),
            'deut_level': float(d_level),
            'fit_dabs': float(fit_dabs),
            'dual_pop_index': dual_pop_index,
            'dual_pop_fraction': float(pop_frac) if dual_pop_index is not None else np.nan,
        })
    return rows


def _build_zoom_fragment_windows_df(
    plot_clusters: list[dict[str, object]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for col_idx, cluster in enumerate(plot_clusters):
        variants = list(cluster.get('variants', []) or [])
        if not variants:
            continue
        col_xlo = min(min(float(v[0]['xlo']), float(v[1]['xlo'])) for v in variants)
        col_xhi = max(max(float(v[0]['xhi']), float(v[1]['xhi'])) for v in variants)
        for vi, (ms2_spec, ms3_spec, ion, _sort_mz) in enumerate(variants):
            rows.append({
                'column_index': int(col_idx),
                'variant_index': int(vi),
                'fragment_ion': str(ion),
                'fragment_seq': str(ms2_spec.get('frag_seq', '') or ''),
                'column_mz_lo': float(col_xlo),
                'column_mz_hi': float(col_xhi),
                'ms2_mz_lo': float(ms2_spec['xlo']),
                'ms2_mz_hi': float(ms2_spec['xhi']),
                'ms3_mz_lo': float(ms3_spec['xlo']),
                'ms3_mz_hi': float(ms3_spec['xhi']),
                'ms2_m0_mz': float(ms2_spec['m0_mz']),
                'ms3_m0_mz': float(ms3_spec['m0_mz']),
                'charge': int(ms2_spec.get('charge', 0) or 0),
                'n_exchangeable': int(ms2_spec.get('n_ex', 0) or 0),
                'radical_overlay': bool(cluster.get('overlay')),
            })
    return pd.DataFrame(rows)


def _build_zoom_dabs_timecourse_df(
    zoom_rows: list[dict[str, object]],
    plot_clusters: list[dict[str, object]],
    peptide: str,
    fit_dabs_cache_ms2: dict[tuple[str, str], dict[str, float]],
    fit_dabs_cache_ms3: dict[tuple[str, str], dict[str, float]],
    fit_qc_cache_ms2: dict[tuple[str, str], dict[str, dict[str, object]]],
    fit_qc_cache_ms3: dict[tuple[str, str], dict[str, dict[str, object]]],
    dual_pop_cache_ms2: dict[tuple[str, str], dict[str, list[tuple[float, float]]]],
    dual_pop_cache_ms3: dict[tuple[str, str], dict[str, list[tuple[float, float]]]],
    *,
    shared_dabs_ylim: tuple[float, float] | None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    y_span = float(shared_dabs_ylim[1]) if shared_dabs_ylim is not None else np.nan

    for col_idx, cluster in enumerate(plot_clusters):
        for _ms2_spec, _ms3_spec, ion, _sort_mz in cluster.get('variants', []) or []:
            key = (peptide, str(ion))
            for ms_level, dabs_map, qc_map, dual_map in (
                (2, fit_dabs_cache_ms2.get(key, {}), fit_qc_cache_ms2.get(key, {}), dual_pop_cache_ms2.get(key, {})),
                (3, fit_dabs_cache_ms3.get(key, {}), fit_qc_cache_ms3.get(key, {}), dual_pop_cache_ms3.get(key, {})),
            ):
                raw_stds: list[float] = []
                pending: list[tuple[str, float | None, float]] = []
                for row in zoom_rows:
                    lbl = str(row['lbl'])
                    t = _dabs_timecourse_x_seconds(lbl)
                    if t is None:
                        continue
                    dabs = dabs_map.get(lbl)
                    if dabs is None or not np.isfinite(float(dabs)):
                        continue
                    std_raw = _dabs_std_for_timepoint(qc_map, lbl)
                    raw_stds.append(std_raw)
                    pending.append((lbl, t, float(dabs)))
                std_display = _visible_dabs_std_errors(
                    raw_stds,
                    y_span=y_span if np.isfinite(y_span) and y_span > 0 else max([p[2] for p in pending] + [1.0]),
                )
                for i, (lbl, t, dabs) in enumerate(pending):
                    rows.append({
                        'column_index': int(col_idx),
                        'fragment_ion': str(ion),
                        'ms_level': int(ms_level),
                        'trace_type': 'fitted_dabs',
                        'exposure_label': lbl,
                        'exposure_s': float(t),
                        'dabs': float(dabs),
                        'dabs_std_raw': float(raw_stds[i]) if i < len(raw_stds) else np.nan,
                        'dabs_std_display': float(std_display[i]) if i < len(std_display) else np.nan,
                    })
                if dual_map:
                    for pi in range(2):
                        for row in zoom_rows:
                            lbl = str(row['lbl'])
                            t = _dabs_timecourse_x_seconds(lbl)
                            comps = dual_map.get(lbl)
                            if t is None or not comps or len(comps) <= pi:
                                continue
                            dabs_p, pop = comps[pi]
                            rows.append({
                                'column_index': int(col_idx),
                                'fragment_ion': str(ion),
                                'ms_level': int(ms_level),
                                'trace_type': f'dual_pop_p{pi + 1}',
                                'exposure_label': lbl,
                                'exposure_s': float(t),
                                'dabs': float(dabs_p),
                                'dual_pop_fraction': float(pop),
                                'dabs_std_raw': np.nan,
                                'dabs_std_display': np.nan,
                            })
    return pd.DataFrame(rows)


def _peaks_in_mz_window(
    match_m: np.ndarray,
    match_i: np.ndarray,
    xlo: float,
    xhi: float,
) -> list[tuple[float, float]]:
    if match_m.size == 0:
        return []
    mask = (match_m >= float(xlo)) & (match_m <= float(xhi))
    if not np.any(mask):
        return []
    return [(float(m), float(i)) for m, i in zip(match_m[mask], match_i[mask])]


def _build_zoom_spectra_df(
    zoom_rows: list[dict[str, object]],
    plot_clusters: list[dict[str, object]],
    peptide: str,
    target_mz: float,
    precursor_charge: int,
    m0_groups: list[dict[str, object]],
    comet_cfg: dict,
    theo_frag_cache: dict[tuple[str, int], list],
    frag_ppm_tol: float,
    un_frag_label_keys: dict[int, set[tuple[str, str]]],
    fit_dabs_cache_ms2: dict[tuple[str, str], dict[str, float]],
    fit_dabs_cache_ms3: dict[tuple[str, str], dict[str, float]],
    dual_pop_cache_ms2: dict[tuple[str, str], dict[str, list[tuple[float, float]]]],
    dual_pop_cache_ms3: dict[tuple[str, str], dict[str, list[tuple[float, float]]]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for row in zoom_rows:
        lbl = str(row['lbl'])
        is_un_row = lbl.strip().upper() == 'UN'
        rt_lo = float(row['rt_lo'])
        rt_hi = float(row['rt_hi'])
        ms_by = row['ms_by']
        match_m2, match_i2, _lk2 = _msx_row_fragment_context(
            ms_by, target_mz, 2, rt_lo, rt_hi,
            m0_groups, precursor_charge, comet_cfg, theo_frag_cache, frag_ppm_tol,
            lbl, un_frag_label_keys,
        )
        match_m3, match_i3, _lk3 = _msx_row_fragment_context(
            ms_by, target_mz, 3, rt_lo, rt_hi,
            m0_groups, precursor_charge, comet_cfg, theo_frag_cache, frag_ppm_tol,
            lbl, un_frag_label_keys,
        )

        for col_idx, cluster in enumerate(plot_clusters):
            variants = list(cluster.get('variants', []) or [])
            if not variants:
                continue
            col_xlo = min(min(float(v[0]['xlo']), float(v[1]['xlo'])) for v in variants)
            col_xhi = max(max(float(v[0]['xhi']), float(v[1]['xhi'])) for v in variants)

            if not is_un_row:
                for ms_level, match_m, match_i in ((2, match_m2, match_i2), (3, match_m3, match_i3)):
                    for mz_p, inten in _peaks_in_mz_window(match_m, match_i, col_xlo, col_xhi):
                        rows.append({
                            'timepoint': lbl,
                            'integration_rt_lo_s': rt_lo,
                            'integration_rt_hi_s': rt_hi,
                            'column_index': int(col_idx),
                            'column_mz_lo': float(col_xlo),
                            'column_mz_hi': float(col_xhi),
                            'fragment_ion': '',
                            'ms_level': int(ms_level),
                            'peak_type': 'raw',
                            'mz': float(mz_p),
                            'intensity': float(inten),
                        })

            for vi, (ms2_spec, ms3_spec, ion, _sort_mz) in enumerate(variants):
                key = (peptide, str(ion))
                for ms_level, spec, match_m, match_i in (
                    (2, ms2_spec, match_m2, match_i2),
                    (3, ms3_spec, match_m3, match_i3),
                ):
                    frag_xlo = float(spec['xlo'])
                    frag_xhi = float(spec['xhi'])
                    max_deut = 0 if is_un_row else int(spec.get('n_ex', 0) or 0)
                    for pk in _fragment_deuteration_peak_matches(
                        match_m, match_i, spec, ppm_tol=frag_ppm_tol, max_deut=max_deut,
                    ):
                        obs_mz = float(pk['obs_mz'])
                        if obs_mz < col_xlo or obs_mz > col_xhi:
                            continue
                        rows.append({
                            'timepoint': lbl,
                            'integration_rt_lo_s': rt_lo,
                            'integration_rt_hi_s': rt_hi,
                            'column_index': int(col_idx),
                            'column_mz_lo': float(col_xlo),
                            'column_mz_hi': float(col_xhi),
                            'fragment_ion': str(ion),
                            'fragment_mz_lo': float(frag_xlo),
                            'fragment_mz_hi': float(frag_xhi),
                            'ms_level': int(ms_level),
                            'peak_type': 'matched',
                            'mz': obs_mz,
                            'intensity': float(pk['obs_int']),
                            'deut_level': float(pk.get('d', np.nan)),
                            'isotope_index': float(pk.get('k', np.nan)),
                        })

                if not is_un_row:
                    fit_d2 = fit_dabs_cache_ms2.get(key, {}).get(lbl)
                    fit_d3 = fit_dabs_cache_ms3.get(key, {}).get(lbl)
                    dual2 = dual_pop_cache_ms2.get(key, {}).get(lbl)
                    dual3 = dual_pop_cache_ms3.get(key, {}).get(lbl)
                    if dual2 and len(dual2) >= 2:
                        for pi, (dabs_p, pop) in enumerate(dual2[:2]):
                            for pk in _fit_envelope_peak_rows(
                                ms2_spec, dabs_p, match_m2, match_i2,
                                frag_ppm_tol=frag_ppm_tol, pop_frac=float(pop), dual_pop_index=pi,
                            ):
                                if float(pk['mz']) < col_xlo or float(pk['mz']) > col_xhi:
                                    continue
                                rows.append({
                                    'timepoint': lbl,
                                    'integration_rt_lo_s': rt_lo,
                                    'integration_rt_hi_s': rt_hi,
                                    'column_index': int(col_idx),
                                    'column_mz_lo': float(col_xlo),
                                    'column_mz_hi': float(col_xhi),
                                    'fragment_ion': str(ion),
                                    'fragment_mz_lo': float(ms2_spec['xlo']),
                                    'fragment_mz_hi': float(ms2_spec['xhi']),
                                    'ms_level': 2,
                                    'peak_type': 'fit_dual_pop',
                                    'mz': float(pk['mz']),
                                    'intensity': float(pk['intensity']),
                                    'deut_level': float(pk['deut_level']),
                                    'fit_dabs': float(pk['fit_dabs']),
                                    'dual_pop_index': int(pi),
                                    'dual_pop_fraction': float(pop),
                                })
                    elif fit_d2 is not None and np.isfinite(float(fit_d2)):
                        for pk in _fit_envelope_peak_rows(
                            ms2_spec, fit_d2, match_m2, match_i2, frag_ppm_tol=frag_ppm_tol,
                        ):
                            if float(pk['mz']) < col_xlo or float(pk['mz']) > col_xhi:
                                continue
                            rows.append({
                                'timepoint': lbl,
                                'integration_rt_lo_s': rt_lo,
                                'integration_rt_hi_s': rt_hi,
                                'column_index': int(col_idx),
                                'column_mz_lo': float(col_xlo),
                                'column_mz_hi': float(col_xhi),
                                'fragment_ion': str(ion),
                                'fragment_mz_lo': float(ms2_spec['xlo']),
                                'fragment_mz_hi': float(ms2_spec['xhi']),
                                'ms_level': 2,
                                'peak_type': 'fit',
                                'mz': float(pk['mz']),
                                'intensity': float(pk['intensity']),
                                'deut_level': float(pk['deut_level']),
                                'fit_dabs': float(pk['fit_dabs']),
                            })
                    if dual3 and len(dual3) >= 2:
                        for pi, (dabs_p, pop) in enumerate(dual3[:2]):
                            for pk in _fit_envelope_peak_rows(
                                ms3_spec, dabs_p, match_m3, match_i3,
                                frag_ppm_tol=frag_ppm_tol, pop_frac=float(pop), dual_pop_index=pi,
                            ):
                                if float(pk['mz']) < col_xlo or float(pk['mz']) > col_xhi:
                                    continue
                                rows.append({
                                    'timepoint': lbl,
                                    'integration_rt_lo_s': rt_lo,
                                    'integration_rt_hi_s': rt_hi,
                                    'column_index': int(col_idx),
                                    'column_mz_lo': float(col_xlo),
                                    'column_mz_hi': float(col_xhi),
                                    'fragment_ion': str(ion),
                                    'fragment_mz_lo': float(ms3_spec['xlo']),
                                    'fragment_mz_hi': float(ms3_spec['xhi']),
                                    'ms_level': 3,
                                    'peak_type': 'fit_dual_pop',
                                    'mz': float(pk['mz']),
                                    'intensity': float(pk['intensity']),
                                    'deut_level': float(pk['deut_level']),
                                    'fit_dabs': float(pk['fit_dabs']),
                                    'dual_pop_index': int(pi),
                                    'dual_pop_fraction': float(pop),
                                })
                    elif fit_d3 is not None and np.isfinite(float(fit_d3)):
                        for pk in _fit_envelope_peak_rows(
                            ms3_spec, fit_d3, match_m3, match_i3, frag_ppm_tol=frag_ppm_tol,
                        ):
                            if float(pk['mz']) < col_xlo or float(pk['mz']) > col_xhi:
                                continue
                            rows.append({
                                'timepoint': lbl,
                                'integration_rt_lo_s': rt_lo,
                                'integration_rt_hi_s': rt_hi,
                                'column_index': int(col_idx),
                                'column_mz_lo': float(col_xlo),
                                'column_mz_hi': float(col_xhi),
                                'fragment_ion': str(ion),
                                'fragment_mz_lo': float(ms3_spec['xlo']),
                                'fragment_mz_hi': float(ms3_spec['xhi']),
                                'ms_level': 3,
                                'peak_type': 'fit',
                                'mz': float(pk['mz']),
                                'intensity': float(pk['intensity']),
                                'deut_level': float(pk['deut_level']),
                                'fit_dabs': float(pk['fit_dabs']),
                            })
    return pd.DataFrame(rows)


def _save_zoom_excel_companion(
    png_path: str,
    *,
    target_mz: float,
    precursor_charge: int,
    peptide: str,
    rank: int,
    zoom_rows: list[dict[str, object]],
    plot_clusters: list[dict[str, object]],
    m0_groups: list[dict[str, object]],
    comet_cfg: dict,
    theo_frag_cache: dict[tuple[str, int], list],
    frag_ppm_tol: float,
    un_frag_label_keys: dict[int, set[tuple[str, str]]],
    fit_dabs_cache_ms2: dict[tuple[str, str], dict[str, float]],
    fit_dabs_cache_ms3: dict[tuple[str, str], dict[str, float]],
    fit_qc_cache_ms2: dict[tuple[str, str], dict[str, dict[str, object]]],
    fit_qc_cache_ms3: dict[tuple[str, str], dict[str, dict[str, object]]],
    dual_pop_cache_ms2: dict[tuple[str, str], dict[str, list[tuple[float, float]]]],
    dual_pop_cache_ms3: dict[tuple[str, str], dict[str, list[tuple[float, float]]]],
    shared_dabs_ylim: tuple[float, float] | None,
    timecourse_sheets: dict[str, pd.DataFrame] | None = None,
) -> str | None:
    xlsx_path = f'{os.path.splitext(png_path)[0]}.xlsx'
    fragment_windows = _build_zoom_fragment_windows_df(plot_clusters)
    dabs_timecourse = _build_zoom_dabs_timecourse_df(
        zoom_rows, plot_clusters, peptide,
        fit_dabs_cache_ms2, fit_dabs_cache_ms3,
        fit_qc_cache_ms2, fit_qc_cache_ms3,
        dual_pop_cache_ms2, dual_pop_cache_ms3,
        shared_dabs_ylim=shared_dabs_ylim,
    )
    spectra = _build_zoom_spectra_df(
        zoom_rows, plot_clusters, peptide, target_mz, precursor_charge,
        m0_groups, comet_cfg, theo_frag_cache, frag_ppm_tol, un_frag_label_keys,
        fit_dabs_cache_ms2, fit_dabs_cache_ms3,
        dual_pop_cache_ms2, dual_pop_cache_ms3,
    )
    summary = pd.DataFrame([{
        'target_mz': float(target_mz),
        'precursor_charge': int(precursor_charge),
        'plot_type': 'fragment_zoom',
        'registry_rank': int(rank),
        'peptide_sequence': str(peptide),
        'n_timepoints': int(len(zoom_rows)),
        'n_fragment_columns': int(len(plot_clusters)),
        'n_spectrum_rows': int(len(spectra)),
        'n_dabs_rows': int(len(dabs_timecourse)),
    }])
    sheets: dict[str, pd.DataFrame] = {'summary': summary}
    if timecourse_sheets:
        sheets.update(timecourse_sheets)
    sheets.update({
        'fragment_windows': fragment_windows,
        'dabs_timecourse': dabs_timecourse,
        'spectra': spectra,
    })
    _write_excel_workbook(xlsx_path, sheets)
    print(
        f'Saved {xlsx_path}  (timecourse={len(timecourse_sheets or {})}, '
        f'windows={len(fragment_windows)}, dabs={len(dabs_timecourse)}, spectra={len(spectra)})'
    )
    return xlsx_path


def _safe_peptide_filename_stem(peptide: str, rank: int | None = None) -> str:
    pep = re.sub(r'[^A-Za-z0-9]+', '', str(peptide or '').strip().upper())
    pep = pep[:28] if pep else 'peptide'
    if rank is not None:
        return f'r{int(rank)}_{pep}'
    return pep


def _safe_fragment_ion_token(ion: str) -> str:
    tok = re.sub(r'[^A-Za-z0-9+\-()]+', '', str(ion or '').strip())
    return tok or 'ion'


def _zoom_precursor_filename_stem(
    peptide: str,
    rank: int,
    plot_clusters: list[dict[str, object]],
) -> str:
    """Filename stem: rank + precursor + all fragment ion labels in this zoom figure."""
    pep = re.sub(r'[^A-Za-z0-9]+', '', str(peptide or '').strip().upper())
    pep = pep[:20] if pep else 'peptide'
    ions: list[str] = []
    seen: set[str] = set()
    for cluster in plot_clusters:
        for pair in list(cluster.get('variants', []) or []):
            ion = str(pair[2])
            if ion in seen:
                continue
            seen.add(ion)
            ions.append(_safe_fragment_ion_token(ion))
    ion_part = '_'.join(ions) if ions else 'nofrag'
    stem = f'r{int(rank)}_{pep}_{ion_part}'
    if len(stem) > 220:
        stem = f'r{int(rank)}_{pep}_{len(ions)}ions'
    return stem


def _group_ms2_ms3_zoom_pairs_by_precursor(
    ms2_specs: list[dict[str, object]],
    ms3_specs: list[dict[str, object]],
    m0_groups: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Group fragments by registry precursor; require both MS2 and MS3 UN matches."""
    ms2_by = _zoom_specs_by_fragment_key(ms2_specs)
    ms3_by = _zoom_specs_by_fragment_key(ms3_specs)
    rank_by_pep = _registry_peptide_rank_map(m0_groups)

    by_pep: dict[str, list[tuple[dict[str, object], dict[str, object], str, float]]] = {}
    colors: dict[str, str] = {}
    for key in sorted(set(ms2_by) & set(ms3_by)):
        pep, ion = key
        ms2_spec = ms2_by[key]
        ms3_spec = ms3_by[key]
        sort_mz = float(ms2_spec['m0_mz'])
        by_pep.setdefault(pep, []).append((ms2_spec, ms3_spec, ion, sort_mz))
        colors[pep] = str(ms2_spec.get('color', TARGET_ISO_TRACE))

    groups: list[dict[str, object]] = []
    for pep, pairs in by_pep.items():
        pairs_sorted = sorted(pairs, key=lambda t: float(t[3]))
        groups.append(
            {
                'peptide': pep,
                'rank': int(rank_by_pep.get(pep, 999999)),
                'color': colors.get(pep, TARGET_ISO_TRACE),
                'pairs': pairs_sorted,
            }
        )
    groups.sort(key=lambda g: (int(g['rank']), str(g['peptide'])))
    return groups


def _deuteration_level_peaks_in_window(
    match_m: np.ndarray,
    match_i: np.ndarray,
    spec: dict[str, object],
    frag_ppm_tol: float,
) -> dict[int, list[dict[str, float]]]:
    """Group matched convolved peaks by deuteration level within the zoom m/z window."""
    xlo = float(spec['xlo'])
    xhi = float(spec['xhi'])
    peaks = _fragment_deuteration_peak_matches(
        match_m, match_i, spec, ppm_tol=frag_ppm_tol,
        max_deut=int(spec.get('n_ex', 0) or 0),
    )
    by_d: dict[int, list[dict[str, float]]] = {}
    for pk in peaks:
        d = int(float(pk.get('d', 0.0)))
        if d <= 0:
            continue
        obs_mz = float(pk['obs_mz'])
        if obs_mz < xlo or obs_mz > xhi:
            continue
        by_d.setdefault(d, []).append(pk)
    return by_d


def _fragment_hdx_deuteration_levels_in_window(
    match_m: np.ndarray,
    match_i: np.ndarray,
    spec: dict[str, object],
    frag_ppm_tol: float,
    *,
    min_isotope_peaks: int = 1,
) -> set[int]:
    """Deuteration levels d>0 whose shifted M0 has an isotope envelope in the zoom window.

    Uses C13-spaced isotope counting at m0 + d*DEUT/charge (same as fragment matching),
    not the colored-stem deuteration matcher (which assigns each m/z to one d only).
    """
    xlo = float(spec['xlo'])
    xhi = float(spec['xhi'])
    m0 = float(spec['m0_mz'])
    charge = int(spec['charge'])
    n_ex = int(spec.get('n_ex', 0) or 0)
    min_iso = int(min_isotope_peaks)
    levels: set[int] = set()
    for d in range(1, n_ex + 1):
        center = m0 + (d * DEUT_MASS_DA / charge)
        if center < xlo or center > xhi:
            continue
        n_iso = _count_fragment_isotope_peaks(
            match_m, match_i, center, charge,
            ppm_tol=frag_ppm_tol, min_peaks=min_iso,
        )
        if n_iso >= min_iso:
            levels.add(d)
    return levels


def _msx_has_any_hdx_deuteration_signal(
    hdx_row_cache: list[dict[str, object]],
    level_key_m: str,
    level_key_i: str,
    spec: dict[str, object],
    frag_ppm_tol: float,
    *,
    min_isotope_peaks: int = 1,
) -> bool:
    """True if any HDX row shows d>0 convolved matches at this MS level."""
    for row in hdx_row_cache:
        levels = _fragment_hdx_deuteration_levels_in_window(
            row[level_key_m], row[level_key_i], spec, frag_ppm_tol,
            min_isotope_peaks=min_isotope_peaks,
        )
        if levels:
            return True
    return False


def _precompute_zoom_row_msx_spectra(
    zoom_rows: list[dict[str, object]],
    target_mz: float,
    m0_groups: list[dict[str, object]],
    precursor_charge: int,
    comet_cfg: dict,
    theo_frag_cache: dict[tuple[str, int], list],
    frag_ppm_tol: float,
    un_frag_label_keys: dict[int, set[tuple[str, str]]],
) -> list[dict[str, object]]:
    """Cache integrated MS2/MS3 spectra per zoom row (skip UN for HDX checks)."""
    cache: list[dict[str, object]] = []
    for row in zoom_rows:
        lbl = str(row['lbl'])
        if lbl.strip().upper() == 'UN':
            continue
        match_m2, match_i2, _ = _msx_row_fragment_context(
            row['ms_by'], target_mz, 2,
            float(row['rt_lo']), float(row['rt_hi']),
            m0_groups, precursor_charge, comet_cfg, theo_frag_cache, frag_ppm_tol,
            lbl, un_frag_label_keys,
        )
        match_m3, match_i3, _ = _msx_row_fragment_context(
            row['ms_by'], target_mz, 3,
            float(row['rt_lo']), float(row['rt_hi']),
            m0_groups, precursor_charge, comet_cfg, theo_frag_cache, frag_ppm_tol,
            lbl, un_frag_label_keys,
        )
        cache.append(
            {
                'lbl': lbl,
                'match_m2': match_m2,
                'match_i2': match_i2,
                'match_m3': match_m3,
                'match_i3': match_i3,
            }
        )
    return cache


def _fragment_pair_has_hdx_deuteration_evidence(
    ms2_spec: dict[str, object],
    ms3_spec: dict[str, object],
    hdx_row_cache: list[dict[str, object]],
    frag_ppm_tol: float,
) -> bool:
    """MS2 and MS3 each need HDX d>0 signal; D level need not match between levels.

    ETD MS2 and scramHCD MS3 yield different Dabs, so we never require the same
    deuteration level on both. MS2 requires a convolved envelope (>=2 isotope
    sub-peaks at some d>0); MS3 only needs any d>0 match (scrambled envelope).
    """
    ms2_ok = _msx_has_any_hdx_deuteration_signal(
        hdx_row_cache, 'match_m2', 'match_i2', ms2_spec, frag_ppm_tol,
        min_isotope_peaks=MIN_FRAG_ISOTOPE_PEAKS,
    )
    ms3_ok = _msx_has_any_hdx_deuteration_signal(
        hdx_row_cache, 'match_m3', 'match_i3', ms3_spec, frag_ppm_tol,
        min_isotope_peaks=1,
    )
    return ms2_ok and ms3_ok


def _filter_zoom_groups_by_hdx_deuteration_evidence(
    groups: list[dict[str, object]],
    zoom_rows: list[dict[str, object]],
    target_mz: float,
    m0_groups: list[dict[str, object]],
    precursor_charge: int,
    comet_cfg: dict,
    theo_frag_cache: dict[tuple[str, int], list],
    frag_ppm_tol: float,
    un_frag_label_keys: dict[int, set[tuple[str, str]]],
    hdx_fit_peptides: set[str] | None = None,
) -> list[dict[str, object]]:
    """Drop pairs unless MS2 and MS3 each show independent HDX d>0 signal.

    HDX fit-target peptides (from manifest) are always kept with all MS2|MS3 UN
    pairs so zoom figures and fragment Dabs can be generated for quant targets
    even when MS2 convolved d>0 evidence is weak.
    """
    hdx_row_cache = _precompute_zoom_row_msx_spectra(
        zoom_rows, target_mz, m0_groups, precursor_charge,
        comet_cfg, theo_frag_cache, frag_ppm_tol, un_frag_label_keys,
    )
    if not hdx_row_cache:
        return []

    force_peps = {str(p).strip().upper() for p in (hdx_fit_peptides or set()) if str(p).strip()}
    filtered: list[dict[str, object]] = []
    for grp in groups:
        pairs = list(grp.get('pairs', []) or [])
        kept = [
            pair for pair in pairs
            if _fragment_pair_has_hdx_deuteration_evidence(
                pair[0], pair[1], hdx_row_cache, frag_ppm_tol,
            )
        ]
        pep = str(grp.get('peptide', '') or '').strip().upper()
        if not kept and pep in force_peps:
            kept = pairs
        if kept:
            filtered.append({**grp, 'pairs': kept})
    return filtered


def _plot_empty_zoom_cell(ax, message: str, *, xlo: float | None = None, xhi: float | None = None) -> None:
    if xlo is not None and xhi is not None and xhi > xlo:
        ax.set_xlim(float(xlo), float(xhi))
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.22, lw=0.4)
    ax.text(
        0.5, 0.5, message, transform=ax.transAxes,
        ha='center', va='center', color='gray', fontsize=7,
    )
    ax.tick_params(labelleft=False, labelbottom=False)


def _fragment_envelope_ymax_in_window(
    match_m: np.ndarray,
    match_i: np.ndarray,
    xlo: float,
    xhi: float,
) -> float:
    in_win = (match_m >= xlo) & (match_m <= xhi)
    if not np.any(in_win):
        return 0.0
    return float(match_i[in_win].max())


def _populate_un_frag_zoom_specs(
    ms_by,
    target_mz: float,
    rt_lo: float,
    rt_hi: float,
    m0_groups: list[dict[str, object]],
    precursor_charge: int,
    comet_cfg: dict,
    theo_cache: dict[tuple[str, int], list],
    frag_ppm_tol: float,
    un_frag_zoom_specs: dict[int, list[dict[str, object]]],
    un_frag_label_keys: dict[int, set[tuple[str, str]]],
    un_col3_group_indices: set[int] | None = None,
) -> None:
    """Freeze UN fragment zoom windows + label keys for reuse on all HDX rows."""
    for level in (2, 3):
        _, raw_m, raw_i = _collect_msx_raw_peaks(
            ms_by, target_mz, level, rt_lo, rt_hi, 100, 2000,
        )
        match_m, match_i = _sum_peaks_by_exact_mz(raw_m, raw_i)
        frags = (
            _collect_col3_fragment_matches(
                match_m, match_i, m0_groups, precursor_charge,
                comet_cfg, theo_cache, frag_ppm_tol,
                un_col3_group_indices=un_col3_group_indices,
            )
            if m0_groups and match_m.size else []
        )
        un_frag_zoom_specs[level] = [_build_fragment_zoom_spec(hit) for hit in frags]
        un_frag_label_keys[level] = _fragment_label_keys(frags)


def _fragment_envelope_ymax(
    match_m: np.ndarray,
    match_i: np.ndarray,
    hit: dict[str, object],
    ppm_tol: float,
) -> float:
    iso_peaks = _fragment_isotope_peak_list(
        match_m, match_i, float(hit['obs_mz']), int(hit['charge']), ppm_tol=ppm_tol,
    )
    if iso_peaks:
        return max(float(pk['obs_int']) for pk in iso_peaks)
    return float(hit.get('obs_int', 0.0))


def _matched_fragments_for_peptide(
    mc: np.ndarray,
    hc: np.ndarray,
    peptide: str,
    precursor_charge: int,
    comet_cfg: dict,
    theo_cache: dict[tuple[str, int], list],
    ppm_tol: float = DEFAULT_FRAG_PPM_TOL,
) -> list[dict[str, object]]:
    """Match col3 peptide c/z ions: ±ppm, reject b/y overlaps, require ≥2 isotope peaks."""
    peptide = str(peptide or '').strip()
    if mc.size == 0 or hc.size == 0 or not peptide:
        return []
    ppm_tol = float(ppm_tol)
    matched: list[dict[str, object]] = []
    for frag in _cached_theoretical_fragments(peptide, precursor_charge, comet_cfg, theo_cache):
        theo_mz = float(frag.mz)
        if theo_mz <= 0:
            continue
        hit = _peak_in_ppm_window(mc, hc, theo_mz, ppm_tol, min_tol_da=0.0)
        if hit is None:
            continue
        obs_mz, obs_int = hit
        comet_name = _comet_ion_name(str(frag.ion))
        if matches_b_or_y_ion(comet_name, peptide, obs_mz, ppm_threshold=ppm_tol):
            continue
        n_iso = _count_fragment_isotope_peaks(
            mc, hc, float(obs_mz), int(frag.charge), ppm_tol=ppm_tol,
        )
        if n_iso < MIN_FRAG_ISOTOPE_PEAKS:
            continue
        matched.append(
            {
                'ion': _display_fragment_label(str(frag.ion)),
                'theo_mz': theo_mz,
                'obs_mz': float(obs_mz),
                'obs_int': float(obs_int),
                'charge': int(frag.charge),
                'peptide': peptide,
                'n_isotope_peaks': int(n_iso),
            }
        )
    return matched


def _collect_col3_fragment_matches(
    mc: np.ndarray,
    hc: np.ndarray,
    m0_groups: list[dict[str, object]],
    precursor_charge: int,
    comet_cfg: dict,
    theo_cache: dict[tuple[str, int], list],
    ppm_tol: float = DEFAULT_FRAG_PPM_TOL,
    un_col3_group_indices: set[int] | None = None,
) -> list[dict[str, object]]:
    """Matched fragments for col3 registry peptides only; dedupe by observed m/z."""
    if not m0_groups:
        return []
    all_matches: list[dict[str, object]] = []
    for entry in _col3_registry_peptides(m0_groups, un_col3_group_indices):
        pep = str(entry['peptide'])
        clr = str(entry['color'])
        for hit in _matched_fragments_for_peptide(
            mc, hc, pep, precursor_charge, comet_cfg, theo_cache, ppm_tol,
        ):
            hit = dict(hit)
            hit['color'] = clr
            hit['group_idx'] = int(entry['gi'])
            all_matches.append(hit)
    if not all_matches:
        return []
    all_matches.sort(key=lambda m: float(m['obs_int']), reverse=True)
    kept: list[dict[str, object]] = []
    for hit in all_matches:
        obs_mz = float(hit['obs_mz'])
        dedupe_tol = max(
            0.002,
            obs_mz * float(ppm_tol) * 1e-6,
        )
        if any(abs(obs_mz - float(k['obs_mz'])) <= dedupe_tol for k in kept):
            continue
        kept.append(hit)
    kept.sort(key=lambda m: float(m['obs_mz']))
    return kept


def _fragment_label_key(hit: dict[str, object]) -> tuple[str, str]:
    """Stable identity for cross-row fragment comparison (peptide + ion name)."""
    return (str(hit.get('peptide', '') or ''), str(hit.get('ion', '') or ''))


def _fragment_label_keys(matches: list[dict[str, object]]) -> set[tuple[str, str]]:
    return {_fragment_label_key(hit) for hit in matches}


def _filter_fragments_to_un_control(
    matches: list[dict[str, object]],
    un_keys: set[tuple[str, str]],
) -> list[dict[str, object]]:
    """Keep only fragments also matched in the UN row for the same MS level."""
    if not matches or not un_keys:
        return []
    return [hit for hit in matches if _fragment_label_key(hit) in un_keys]


def _nearest_peak_in_ppm_window(
    mc: np.ndarray,
    hc: np.ndarray,
    target_mz: float,
    ppm_tol: float,
    min_tol_da: float = 0.015,
) -> tuple[float, float] | None:
    """Nearest observed peak to a theoretical m/z (matches significant-frags summing logic)."""
    if mc.size == 0 or hc.size == 0 or not np.isfinite(float(target_mz)):
        return None
    tol = max(float(target_mz) * float(ppm_tol) * 1e-6, float(min_tol_da))
    idx = int(np.argmin(np.abs(mc - float(target_mz))))
    obs_mz = float(mc[idx])
    if abs(obs_mz - float(target_mz)) > tol:
        return None
    obs_int = float(hc[idx])
    if obs_int <= 0:
        return None
    return obs_mz, obs_int


def _peak_in_ppm_window(
    mc: np.ndarray,
    hc: np.ndarray,
    target_mz: float,
    ppm_tol: float,
    min_tol_da: float = 0.015,
) -> tuple[float, float] | None:
    if mc.size == 0 or hc.size == 0 or not np.isfinite(float(target_mz)):
        return None
    tol = max(float(target_mz) * float(ppm_tol) * 1e-6, float(min_tol_da))
    m = np.abs(mc - float(target_mz)) <= tol
    if not np.any(m):
        return None
    idx_local = int(np.argmax(hc[m]))
    idx = int(np.where(m)[0][idx_local])
    obs_int = float(hc[idx])
    if obs_int <= 0:
        return None
    return float(mc[idx]), obs_int


def _plot_matched_fragment_stems(
    ax,
    matches: list[dict[str, object]],
) -> None:
    for hit in matches:
        ax.vlines(
            float(hit['obs_mz']), 0.0, float(hit['obs_int']),
            color=str(hit.get('color', TARGET_ISO_TRACE)),
            lw=FRAG_MATCH_LW, alpha=1.0, zorder=3,
        )


def _matched_fragment_ymax(matches: list[dict[str, object]]) -> float:
    if not matches:
        return 0.0
    return max(float(hit['obs_int']) for hit in matches)


def _fragment_zoom_mz_limits(
    matches: list[dict[str, object]],
    pad_da: float = FRAG_ZOOM_PAD_DA,
) -> tuple[float, float]:
    """m/z window spanning matched fragments and their isotope envelopes."""
    if not matches:
        return 100.0, 2000.0
    lo = float('inf')
    hi = float('-inf')
    for hit in matches:
        center = float(hit['obs_mz'])
        charge = max(1, int(hit['charge']))
        spacing = C13 / charge
        lo = min(lo, center - spacing)
        hi = max(hi, center + (FRAG_ISOTOPE_MAX_K * spacing))
    span = max(1.0, hi - lo)
    pad = max(float(pad_da), span * 0.18)
    return lo - pad, hi + pad


def _label_box_overlaps(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
    pad_x: float = 3.0,
    pad_y: float = 3.0,
) -> bool:
    ax0, ax1, ay0, ay1 = a
    bx0, bx1, by0, by1 = b
    return not (
        (ax1 + pad_x) < bx0
        or (bx1 + pad_x) < ax0
        or (ay1 + pad_y) < by0
        or (by1 + pad_y) < ay0
    )


def _label_width(text: str, x_span: float) -> float:
    return max(42.0, len(text) * x_span * 0.0068) * 1.18


def _label_height(ymax: float) -> float:
    return ymax * 0.065


def _estimate_horizontal_label_box(
    x_center: float,
    y_base: float,
    text: str,
    x_span: float,
    ymax: float,
) -> tuple[float, float, float, float]:
    width = _label_width(text, x_span)
    height = _label_height(ymax)
    return (x_center - (0.5 * width), x_center + (0.5 * width), y_base, y_base + height)


def _intervals_overlap(a0: float, a1: float, b0: float, b1: float, pad: float = 0.0) -> bool:
    return not ((a1 + pad) < b0 or (b1 + pad) < a0)


def _peak_band_halfwidths(
    matches: list[dict[str, object]],
    x_span: float,
) -> dict[float, float]:
    """Horizontal keep-out half-width around each matched peak m/z."""
    xs = sorted({float(hit['obs_mz']) for hit in matches})
    bands: dict[float, float] = {}
    default_hw = max(10.0, x_span * 0.011)
    for x_peak in xs:
        neighbor_dists = [abs(x - x_peak) for x in xs if abs(x - x_peak) > 1e-6]
        if neighbor_dists:
            nearest = min(neighbor_dists)
            bands[x_peak] = max(8.0, min(nearest * 0.42, x_span * 0.028))
        else:
            bands[x_peak] = default_hw
    return bands


def _signal_exclusion_boxes(
    matches: list[dict[str, object]],
    bands: dict[float, float],
    ymax: float,
) -> list[tuple[float, float, float, float]]:
    """Vertical signal columns for every matched peak (own + neighbors)."""
    pad = ymax * 0.03
    return [
        (x_peak - bands[x_peak], x_peak + bands[x_peak], 0.0, float(hit['obs_int']) + pad)
        for hit in matches
        for x_peak in [float(hit['obs_mz'])]
    ]


def _label_overlaps_foreign_peak_band(
    label_box: tuple[float, float, float, float],
    x_peak: float,
    bands: dict[float, float],
) -> bool:
    lx0, lx1, _, _ = label_box
    for x_other, half_w in bands.items():
        if abs(x_other - x_peak) < 1e-6:
            continue
        if _intervals_overlap(lx0, lx1, x_other - half_w, x_other + half_w, pad=1.0):
            return True
    return False


def _label_overlaps_signal(
    label_box: tuple[float, float, float, float],
    signal_boxes: list[tuple[float, float, float, float]],
) -> bool:
    return any(_label_box_overlaps(label_box, sig, pad_x=2.0, pad_y=2.0) for sig in signal_boxes)


def _label_overlaps_other_labels(
    label_box: tuple[float, float, float, float],
    placed_boxes: list[tuple[float, float, float, float]],
    pad_x: float,
    pad_y: float,
) -> bool:
    return any(
        _label_box_overlaps(label_box, prev, pad_x=pad_x, pad_y=pad_y)
        for prev in placed_boxes
    )


def _label_placement_valid(
    label_box: tuple[float, float, float, float],
    x_peak: float,
    bands: dict[float, float],
    signal_boxes: list[tuple[float, float, float, float]],
    placed_boxes: list[tuple[float, float, float, float]],
) -> bool:
    if _label_overlaps_foreign_peak_band(label_box, x_peak, bands):
        return False
    if _label_overlaps_signal(label_box, signal_boxes):
        return False
    return True


def _fragment_label_x_candidates(
    x_peak: float,
    x_nudge: float,
    max_cols: int,
) -> list[float]:
    out = [x_peak]
    for col in range(1, max_cols + 1):
        delta = col * x_nudge
        out.extend([x_peak + delta, x_peak - delta])
    return out


def _fragment_label_placements_no_overlap(
    matches: list[dict[str, object]],
    ymax: float,
    x_lo: float,
    x_hi: float,
) -> list[tuple[float, float, float, float, dict[str, object]]]:
    """Place offset labels above a clear margin; elbow leaders avoid peak columns."""
    if not matches or ymax <= 0:
        return []
    x_span = max(1.0, float(x_hi) - float(x_lo))
    bands = _peak_band_halfwidths(matches, x_span)
    signal_boxes = _signal_exclusion_boxes(matches, bands, ymax)
    y_floor = ymax * 1.30
    y_step = ymax * 0.10
    x_nudge = max(16.0, x_span * 0.012)
    pad_x = max(12.0, x_span * 0.0055)
    pad_y = ymax * 0.028

    ordered = sorted(matches, key=lambda m: float(m['obs_mz']))
    placed_boxes: list[tuple[float, float, float, float]] = []
    out: list[tuple[float, float, float, float, dict[str, object]]] = []
    max_rows = max(18, len(matches) + 8)
    max_cols = max(10, len(matches) // 2 + 3)
    for hit in ordered:
        x_peak = float(hit['obs_mz'])
        y_peak = float(hit['obs_int'])
        ion = str(hit.get('ion', '') or '')
        placed = False
        for row in range(max_rows):
            y_text = y_floor + (row * y_step)
            for x_text in _fragment_label_x_candidates(x_peak, x_nudge, max_cols):
                box = _estimate_horizontal_label_box(x_text, y_text, ion, x_span, ymax)
                if not _label_placement_valid(box, x_peak, bands, signal_boxes, placed_boxes):
                    continue
                if _label_overlaps_other_labels(box, placed_boxes, pad_x, pad_y):
                    continue
                placed_boxes.append(box)
                out.append((x_text, y_text, x_peak, y_peak, hit))
                placed = True
                break
            if placed:
                break
        if not placed:
            y_text = max((b[3] for b in placed_boxes), default=y_floor) + y_step
            while True:
                box = _estimate_horizontal_label_box(x_peak, y_text, ion, x_span, ymax)
                if not _label_overlaps_other_labels(box, placed_boxes, pad_x, pad_y):
                    placed_boxes.append(box)
                    out.append((x_peak, y_text, x_peak, y_peak, hit))
                    break
                y_text += y_step
    return out


def _draw_fragment_leader(
    ax,
    x_text: float,
    y_text: float,
    x_peak: float,
    y_peak: float,
    clr: str,
) -> None:
    """Draw an elbow leader from label to peak without FancyArrow layout issues."""
    if abs(x_text - x_peak) <= 0.5:
        xs = [x_peak, x_peak]
        ys = [y_text, y_peak]
    else:
        xs = [x_text, x_peak, x_peak]
        ys = [y_text, y_text, y_peak]
    ax.plot(xs, ys, color=clr, lw=0.7, alpha=0.85, solid_capstyle='round', zorder=7)
    ax.plot(
        [x_peak], [y_peak], marker='>', color=clr, markersize=4,
        alpha=0.85, zorder=7, clip_on=False,
    )


def _annotate_matched_fragments(
    ax,
    matches: list[dict[str, object]],
    hc: np.ndarray,
    y_ref_max: float | None = None,
) -> None:
    if not matches or hc.size == 0 or not np.any(hc > 0):
        return
    ymax = float(y_ref_max) if y_ref_max is not None and y_ref_max > 0 else float(hc.max())
    x_lo, x_hi = ax.get_xlim()
    x_span = max(1.0, float(x_hi) - float(x_lo))
    placements = _fragment_label_placements_no_overlap(matches, ymax, x_lo, x_hi)
    label_top = ymax * 1.20
    for x_text, y_text, x_peak, y_peak, hit in placements:
        base_clr = str(hit.get('color', TARGET_ISO_TRACE))
        clr = _darken_color(base_clr)
        ion = str(hit.get('ion', '') or '')
        _draw_fragment_leader(ax, x_text, y_text, x_peak, y_peak, clr)
        ax.text(
            x_text, y_text, ion,
            fontsize=6.5,
            ha='center',
            va='bottom',
            color=clr,
            clip_on=False,
            zorder=8,
            bbox=dict(facecolor='white', edgecolor=clr, alpha=0.94, pad=0.55, linewidth=0.75),
        )
        _, _, _, y1 = _estimate_horizontal_label_box(x_text, y_text, ion, x_span, ymax)
        label_top = max(label_top, y1)
    if placements:
        ax.set_ylim(0, label_top * 1.08)


def _msx_row_fragment_context(
    ms_by,
    target_mz: float,
    level: int,
    rt_lo: float,
    rt_hi: float,
    m0_groups: list[dict[str, object]],
    precursor_charge: int,
    comet_cfg: dict,
    theo_frag_cache: dict[tuple[str, int], list],
    frag_ppm_tol: float,
    lbl: str,
    un_frag_label_keys: dict[int, set[tuple[str, str]]],
    un_col3_group_indices: set[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    _, raw_m, raw_i = _collect_msx_raw_peaks(ms_by, target_mz, level, rt_lo, rt_hi, 100, 2000)
    match_m, match_i = _sum_peaks_by_exact_mz(raw_m, raw_i)
    frag_matches = (
        _collect_col3_fragment_matches(
            match_m, match_i, m0_groups, precursor_charge, comet_cfg, theo_frag_cache, frag_ppm_tol,
            un_col3_group_indices=un_col3_group_indices,
        )
        if m0_groups and match_m.size else []
    )
    if str(lbl).strip().upper() == 'UN':
        label_matches = frag_matches
    else:
        label_matches = _filter_fragments_to_un_control(frag_matches, un_frag_label_keys[level])
    return match_m, match_i, label_matches


def _timepoint_qc_pass(qc_map: dict[str, dict[str, object]], lbl: str) -> bool:
    """Always include fitted Dabs in timecourse plots (QC flags kept for reference only)."""
    return True


def _dabs_std_for_timepoint(qc_map: dict[str, dict[str, object]], lbl: str) -> float:
    """Raw pyHXExpress Dabs_std_1 for one timepoint."""
    qc = qc_map.get(str(lbl))
    if not qc:
        return 0.0
    std = qc.get('dabs_std', np.nan)
    if std is None or not np.isfinite(float(std)) or float(std) < 0:
        return 0.0
    return float(std)


def _visible_dabs_std_errors(raw_stds: list[float], *, y_span: float) -> list[float]:
    """Normalize Dabs_std_1 within a trace, then scale for visible error bars."""
    if not raw_stds or y_span <= 0:
        return [0.0] * len(raw_stds)
    pos = [
        float(s) for s in raw_stds
        if s is not None and np.isfinite(float(s)) and float(s) > 0
    ]
    if not pos:
        return [0.0] * len(raw_stds)
    scale = (float(y_span) * DABS_STD_VISIBLE_FRAC) / max(pos)
    out: list[float] = []
    for s in raw_stds:
        if s is None or not np.isfinite(float(s)) or float(s) <= 0:
            out.append(0.0)
        else:
            out.append(float(s) * scale)
    return out


def _build_fragment_fit_dabs_cache(
    zoom_rows: list[dict[str, object]],
    target_mz: float,
    precursor_charge: int,
    pairs: list[tuple[dict[str, object], dict[str, object], str, float]],
    *,
    act2_label: str,
    act3_label: str,
    fits_root: str = DEFAULT_FRAGMENT_FITS_ROOT,
    frag_ppm_tol: float = DEFAULT_FRAG_PPM_TOL,
) -> tuple[
    dict[tuple[str, str], dict[str, float]],
    dict[tuple[str, str], dict[str, float]],
    dict[tuple[str, str], dict[str, dict[str, object]]],
    dict[tuple[str, str], dict[str, dict[str, object]]],
    dict[tuple[str, str], dict[str, list[tuple[float, float]]]],
    dict[tuple[str, str], dict[str, list[tuple[float, float]]]],
    set[tuple[str, str]],
    dict[tuple[str, str], set[str]],
]:
    """Run/cache 1-pop fragment fits; auto-refit bimodal ions with 2 populations."""
    ms2_cache: dict[tuple[str, str], dict[str, float]] = {}
    ms3_cache: dict[tuple[str, str], dict[str, float]] = {}
    ms2_qc_cache: dict[tuple[str, str], dict[str, dict[str, object]]] = {}
    ms3_qc_cache: dict[tuple[str, str], dict[str, dict[str, object]]] = {}
    ms2_dual_cache: dict[tuple[str, str], dict[str, list[tuple[float, float]]]] = {}
    ms3_dual_cache: dict[tuple[str, str], dict[str, list[tuple[float, float]]]] = {}
    bimodal_keys: set[tuple[str, str]] = set()
    bimodal_labels_cache: dict[tuple[str, str], set[str]] = {}
    seen: set[tuple[str, str]] = set()
    for ms2_spec, ms3_spec, _ion, _sort_mz in pairs:
        pep = str(ms2_spec.get('peptide', '') or '')
        ion = str(ms2_spec.get('ion', '') or '')
        key = (pep, ion)
        if key in seen:
            continue
        seen.add(key)
        ms2_map, ms2_qc = run_cached_fragment_hdx_fit(
            zoom_rows, target_mz, precursor_charge, pep, ms2_spec,
            fits_root=fits_root, ms_level=2, act_label=act2_label,
            frag_ppm_tol=frag_ppm_tol,
        )
        if ms2_map:
            ms2_cache[key] = ms2_map
        if ms2_qc:
            ms2_qc_cache[key] = ms2_qc
        cache_dir_ms2 = fit_cache_dir(
            fits_root, target_mz, precursor_charge, pep, ms2_spec, ms_level=2,
        )
        is_bim_ms2, dual_ms2, _ = maybe_run_dual_pop_fragment_fit(
            zoom_rows, target_mz, precursor_charge, pep, ms2_spec, cache_dir_ms2,
            fits_root=fits_root, ms_level=2, act_label=act2_label,
        )
        if is_bim_ms2 and dual_ms2:
            ms2_dual_cache[key] = dual_ms2
            bimodal_keys.add(key)
            bimodal_labels_cache[key] = set(dual_ms2.keys())

        ms3_map, ms3_qc = run_cached_fragment_hdx_fit(
            zoom_rows, target_mz, precursor_charge, pep, ms3_spec,
            fits_root=fits_root, ms_level=3, act_label=act3_label,
            frag_ppm_tol=frag_ppm_tol,
        )
        if ms3_map:
            ms3_cache[key] = ms3_map
        if ms3_qc:
            ms3_qc_cache[key] = ms3_qc
        cache_dir_ms3 = fit_cache_dir(
            fits_root, target_mz, precursor_charge, pep, ms3_spec, ms_level=3,
        )
        is_bim_ms3, dual_ms3, _ = maybe_run_dual_pop_fragment_fit(
            zoom_rows, target_mz, precursor_charge, pep, ms3_spec, cache_dir_ms3,
            fits_root=fits_root, ms_level=3, act_label=act3_label,
        )
        if is_bim_ms3 and dual_ms3:
            ms3_dual_cache[key] = dual_ms3
            bimodal_keys.add(key)
            if key in bimodal_labels_cache:
                bimodal_labels_cache[key] |= set(dual_ms3.keys())
            else:
                bimodal_labels_cache[key] = set(dual_ms3.keys())
    return (
        ms2_cache, ms3_cache, ms2_qc_cache, ms3_qc_cache,
        ms2_dual_cache, ms3_dual_cache, bimodal_keys, bimodal_labels_cache,
    )


def _overlay_fit_envelope_on_zoom_ax(
    ax,
    spec: dict[str, object],
    fit_dabs: float | None,
    match_m: np.ndarray,
    match_i: np.ndarray,
    *,
    frag_ppm_tol: float = DEFAULT_FRAG_PPM_TOL,
    fit_color: str = ZOOM_FIT_STEM_COLOR,
    zorder: float = ZOOM_FIT_LABEL_Z - 1,
) -> None:
    """Overlay single-pop fit envelope stems on a zoom axis."""
    if fit_dabs is None or not np.isfinite(float(fit_dabs)):
        return
    xlo = float(spec['xlo'])
    xhi = float(spec['xhi'])
    pred = predicted_fit_envelope_peaks(spec, float(fit_dabs))
    scaled = scale_fit_envelope_to_observed(
        pred, match_m, match_i, xlo, xhi, ppm_tol=frag_ppm_tol,
    )
    for mz_p, height, _d_level, _obs_at in scaled:
        if mz_p < xlo or mz_p > xhi or height <= 0:
            continue
        ax.vlines(
            mz_p, 0.0, height,
            color=fit_color, lw=ZOOM_FIT_STEM_LW,
            alpha=ZOOM_FIT_STEM_ALPHA, linestyles='--', zorder=zorder,
        )


def _overlay_multi_pop_fit_envelopes_on_zoom_ax(
    ax,
    spec: dict[str, object],
    components: list[tuple[float, float]] | None,
    match_m: np.ndarray,
    match_i: np.ndarray,
    *,
    frag_ppm_tol: float = DEFAULT_FRAG_PPM_TOL,
    fit_color: str = ZOOM_FIT_STEM_COLOR,
    zorder: float = ZOOM_FIT_LABEL_Z - 1,
) -> None:
    """Overlay 2-population fit envelopes (each scaled by population fraction)."""
    if not components or len(components) < 2:
        return
    xlo = float(spec['xlo'])
    xhi = float(spec['xhi'])
    for pi, (dabs, pop_frac) in enumerate(components[:2]):
        if dabs is None or not np.isfinite(float(dabs)) or float(pop_frac) <= 0:
            continue
        pred = predicted_fit_envelope_peaks(spec, float(dabs))
        scaled = scale_fit_envelope_to_observed(
            pred, match_m, match_i, xlo, xhi, ppm_tol=frag_ppm_tol,
        )
        ls = BIMODAL_POP_LINESTYLES[pi % len(BIMODAL_POP_LINESTYLES)]
        alpha = 0.75 if pi == 0 else 0.95
        lw = ZOOM_FIT_STEM_LW if pi == 0 else (ZOOM_FIT_STEM_LW + 0.2)
        for mz_p, height, _d_level, _obs_at in scaled:
            if mz_p < xlo or mz_p > xhi or height <= 0:
                continue
            ax.vlines(
                mz_p, 0.0, float(height) * float(pop_frac),
                color=fit_color, lw=lw,
                alpha=alpha, linestyles=ls, zorder=zorder + (pi * 0.01),
            )


def _bimodal_column_title_suffix(frag_seq: str) -> str:
    if fragment_has_proline(frag_seq):
        return '\n[bimodal, Pro-rich; 2-pop fit]'
    return '\n[bimodal; 2-pop fit]'


def _format_dual_pop_label(components: list[tuple[float, float]]) -> str:
    parts = []
    for i, (dabs, pop) in enumerate(components[:2], start=1):
        parts.append(f'p{i} D={float(dabs):.2f} ({100.0 * float(pop):.0f}%)')
    return ' | '.join(parts)


def _overlay_fit_dabs_label_on_zoom_ax(
    ax,
    fit_dabs: float | None,
    *,
    fit_color: str = ZOOM_FIT_STEM_COLOR,
    label_x: float = 0.98,
    label_ha: str = 'right',
    label_y: float = 0.96,
    label_suffix: str = '',
    ms_level_prefix: str = '',
    fontsize: float = 6.0,
) -> None:
    """Place a fit Dabs label in a zoom axis corner."""
    if fit_dabs is None or not np.isfinite(float(fit_dabs)):
        return
    label = f'fit D={float(fit_dabs):.2f}'
    if label_suffix:
        label = f'{label_suffix} {label}'
    if ms_level_prefix:
        label = f'{ms_level_prefix} {label}'
    ax.text(
        label_x, label_y, label,
        transform=ax.transAxes, ha=label_ha, va='top',
        fontsize=fontsize, color=fit_color, clip_on=False, zorder=ZOOM_FIT_LABEL_Z,
        bbox=dict(
            facecolor='white', edgecolor=fit_color,
            alpha=0.82, pad=0.35, linewidth=0.6,
        ),
    )


def _stack_corner_fit_dabs_labels(
    ax,
    entries: list[tuple[float | None, str, str]],
    *,
    corner: str = 'left',
    ms_level_prefix: str = '',
    n_var: int = 1,
) -> int:
    """Stack fit-D labels vertically in a corner without overlap."""
    valid = [
        (float(dabs), str(ion), str(clr))
        for dabs, ion, clr in entries
        if dabs is not None and np.isfinite(float(dabs))
    ]
    if not valid:
        return 0
    n = len(valid)
    fs = 5.2 if n >= 4 else (5.5 if n >= 3 else (5.8 if n >= 2 else 6.0))
    step = min(0.135, 0.90 / max(n, 1))
    top = 0.985
    label_x = 0.02 if corner == 'left' else 0.98
    ha = 'left' if corner == 'left' else 'right'
    for i, (dabs, ion, clr) in enumerate(valid):
        if n_var > 1:
            label = f'{ion} D={dabs:.2f}'
            if i == 0 and ms_level_prefix:
                label = f'{ms_level_prefix} {label}'
        else:
            label = (
                f'{ms_level_prefix} fit D={dabs:.2f}'
                if ms_level_prefix else f'fit D={dabs:.2f}'
            )
        ax.text(
            label_x, top - (i * step), label,
            transform=ax.transAxes, ha=ha, va='top',
            fontsize=fs, color=clr, clip_on=False, zorder=ZOOM_FIT_LABEL_Z,
            bbox=dict(
                facecolor='white', edgecolor=clr,
                alpha=0.88, pad=0.28, linewidth=0.55,
            ),
        )
    return n


def _overlay_fit_dabs_on_zoom_ax(
    ax,
    spec: dict[str, object],
    fit_dabs: float | None,
    match_m: np.ndarray,
    match_i: np.ndarray,
    *,
    frag_ppm_tol: float = DEFAULT_FRAG_PPM_TOL,
    label_y: float = 0.96,
    label_suffix: str = '',
    fit_color: str = ZOOM_FIT_STEM_COLOR,
    label_x: float = 0.98,
    label_ha: str = 'right',
) -> None:
    """Overlay single-pop fit envelope and Dabs label on an MS2/MS3 zoom axis."""
    _overlay_fit_envelope_on_zoom_ax(
        ax, spec, fit_dabs, match_m, match_i,
        frag_ppm_tol=frag_ppm_tol, fit_color=fit_color,
    )
    _overlay_fit_dabs_label_on_zoom_ax(
        ax, fit_dabs, fit_color=fit_color, label_x=label_x, label_ha=label_ha,
        label_y=label_y, label_suffix=label_suffix,
    )


def _plot_fragment_zoom_ms2_ms3_combined_ax(
    ax,
    variants: list[tuple[dict[str, object], dict[str, object], str, float]],
    match_m2: np.ndarray,
    match_i2: np.ndarray,
    label_keys2: set[tuple[str, str]],
    match_m3: np.ndarray,
    match_i3: np.ndarray,
    label_keys3: set[tuple[str, str]],
    frag_ppm_tol: float,
    *,
    is_un_row: bool = False,
    fit_by_ion_ms2: dict[str, float] | None = None,
    fit_by_ion_ms3: dict[str, float] | None = None,
    dual_pop_by_ion_ms2: dict[str, list[tuple[float, float]]] | None = None,
    dual_pop_by_ion_ms3: dict[str, list[tuple[float, float]]] | None = None,
    bimodal_timepoint: bool = False,
    qc_by_ion_ms2: dict[str, dict[str, object]] | None = None,
    qc_by_ion_ms3: dict[str, dict[str, object]] | None = None,
    show_xlabel: bool = False,
    show_ylabel: bool = False,
    title_fs: float = 6.0,
    tick_fs: float = 5.0,
) -> None:
    """One column: MS3 (brown/orange, lower z) under MS2 (grey/blue) on the same axis."""
    if not variants:
        return

    n_var = len(variants)
    ms2_match_cols = _zoom_ms2_match_colors(n_var)
    ms2_fit_cols = _zoom_ms2_fit_colors(n_var)
    ms3_match_cols = _zoom_ms3_match_colors(n_var)
    ms3_fit_cols = _zoom_ms3_fit_colors(n_var)

    xlo = min(min(float(v[0]['xlo']), float(v[1]['xlo'])) for v in variants)
    xhi = max(max(float(v[0]['xhi']), float(v[1]['xhi'])) for v in variants)
    env_max = 0.0
    if n_var > 1:
        ms2_stem_alpha = ZOOM_MS2_MATCH_STEM_ALPHA_OVERLAY
        ms3_stem_alpha = ZOOM_MS3_MATCH_STEM_ALPHA_OVERLAY
    elif is_un_row:
        ms2_stem_alpha = ZOOM_MS2_MATCH_STEM_ALPHA_UN
        ms3_stem_alpha = ZOOM_MS3_MATCH_STEM_ALPHA_UN
    else:
        ms2_stem_alpha = ZOOM_MS2_MATCH_STEM_ALPHA_HDX
        ms3_stem_alpha = ZOOM_MS3_MATCH_STEM_ALPHA_HDX

    def _fit_dabs_for_display(ion: str, dabs: float | None, qc_by_ion: dict | None) -> float | None:
        if dabs is None or not np.isfinite(float(dabs)):
            return None
        return float(dabs)

    # MS3 underlay (lower z-order).
    if not is_un_row:
        in_win3 = (match_m3 >= xlo) & (match_m3 <= xhi)
        rm3, ri3 = match_m3[in_win3], match_i3[in_win3]
        if rm3.size:
            ax.vlines(
                rm3, 0, ri3, color=ZOOM_MS3_RAW_COLOR, lw=0.85,
                alpha=0.50, zorder=ZOOM_MS3_RAW_Z,
            )
        env_max = max(env_max, _fragment_envelope_ymax_in_window(match_m3, match_i3, xlo, xhi))

    for vi, (ms2_spec, ms3_spec, ion, _sort_mz) in enumerate(variants):
        ms3_match = ms3_match_cols[vi]
        ms3_fit = ms3_fit_cols[vi]
        m0_mz3 = float(ms3_spec['m0_mz'])
        charge3 = int(ms3_spec['charge'])
        max_deut3 = 0 if is_un_row else int(ms3_spec.get('n_ex', 0) or 0)
        iso_peaks3 = _fragment_deuteration_peak_matches(
            match_m3, match_i3, ms3_spec, ppm_tol=frag_ppm_tol, max_deut=max_deut3,
        )
        for pk in iso_peaks3:
            ax.vlines(
                float(pk['obs_mz']), 0.0, float(pk['obs_int']),
                color=ms3_match, lw=FRAG_MATCH_LW, alpha=ms3_stem_alpha, zorder=ZOOM_MS3_STEM_Z,
            )
        if iso_peaks3:
            env_max = max(env_max, max(float(pk['obs_int']) for pk in iso_peaks3))
        ax.axvline(
            m0_mz3, color=_darken_color(ms3_match, 0.72), ls=':', lw=0.6,
            alpha=0.45, zorder=ZOOM_MS3_M0_REF_Z,
        )
        spec_key3 = (str(ms3_spec.get('peptide', '') or ''), ion)
        if spec_key3 in label_keys3 and is_un_row:
            ax.text(
                m0_mz3, env_max * 1.08 if env_max > 0 else 1.0, ion,
                fontsize=max(4.5, title_fs - 0.5), ha='center', va='bottom',
                color=_darken_color(ms3_match), clip_on=False, zorder=ZOOM_LABEL_Z,
                bbox=dict(
                    facecolor='white', edgecolor=_darken_color(ms3_match),
                    alpha=0.94, pad=0.35, linewidth=0.6,
                ),
            )
        if not is_un_row and fit_by_ion_ms3 is not None:
            dual3 = (dual_pop_by_ion_ms3 or {}).get(ion)
            fit_d3 = _fit_dabs_for_display(ion, fit_by_ion_ms3.get(ion), qc_by_ion_ms3)
            if dual3 and len(dual3) >= 2:
                _overlay_multi_pop_fit_envelopes_on_zoom_ax(
                    ax, ms3_spec, dual3, match_m3, match_i3,
                    frag_ppm_tol=frag_ppm_tol, fit_color=ms3_fit, zorder=ZOOM_MS3_FIT_Z,
                )
            else:
                _overlay_fit_envelope_on_zoom_ax(
                    ax, ms3_spec, fit_d3, match_m3, match_i3,
                    frag_ppm_tol=frag_ppm_tol, fit_color=ms3_fit, zorder=ZOOM_MS3_FIT_Z,
                )

    # MS2 overlay (higher z-order).
    if not is_un_row:
        in_win2 = (match_m2 >= xlo) & (match_m2 <= xhi)
        rm2, ri2 = match_m2[in_win2], match_i2[in_win2]
        if rm2.size:
            ax.vlines(
                rm2, 0, ri2, color=ZOOM_MS2_RAW_COLOR, lw=0.85,
                alpha=0.55, zorder=ZOOM_MS2_RAW_Z,
            )
        env_max = max(env_max, _fragment_envelope_ymax_in_window(match_m2, match_i2, xlo, xhi))

    for vi, (ms2_spec, ms3_spec, ion, _sort_mz) in enumerate(variants):
        ms2_match = ms2_match_cols[vi]
        ms2_fit = ms2_fit_cols[vi]
        m0_mz2 = float(ms2_spec['m0_mz'])
        charge2 = int(ms2_spec['charge'])
        max_deut2 = 0 if is_un_row else int(ms2_spec.get('n_ex', 0) or 0)
        iso_peaks2 = _fragment_deuteration_peak_matches(
            match_m2, match_i2, ms2_spec, ppm_tol=frag_ppm_tol, max_deut=max_deut2,
        )
        for pk in iso_peaks2:
            ax.vlines(
                float(pk['obs_mz']), 0.0, float(pk['obs_int']),
                color=ms2_match, lw=FRAG_MATCH_LW, alpha=ms2_stem_alpha, zorder=ZOOM_MS2_STEM_Z,
            )
        if iso_peaks2:
            env_max = max(env_max, max(float(pk['obs_int']) for pk in iso_peaks2))
        else:
            env_max = max(
                env_max,
                _fragment_envelope_ymax(
                    match_m2, match_i2,
                    {'obs_mz': m0_mz2, 'charge': charge2, 'obs_int': 0.0},
                    frag_ppm_tol,
                ),
            )
        ax.axvline(
            m0_mz2, color=_darken_color(ms2_match, 0.72), ls=':', lw=0.6,
            alpha=0.50, zorder=ZOOM_MS2_M0_REF_Z,
        )
        spec_key2 = (str(ms2_spec.get('peptide', '') or ''), ion)
        if spec_key2 in label_keys2 and is_un_row:
            ax.text(
                m0_mz2, env_max * 1.08 if env_max > 0 else 1.0, ion,
                fontsize=max(4.5, title_fs - 0.5), ha='center', va='bottom',
                color=_darken_color(ms2_match), clip_on=False, zorder=ZOOM_LABEL_Z,
                bbox=dict(
                    facecolor='white', edgecolor=_darken_color(ms2_match),
                    alpha=0.94, pad=0.35, linewidth=0.6,
                ),
            )
        if not is_un_row and fit_by_ion_ms2 is not None:
            dual2 = (dual_pop_by_ion_ms2 or {}).get(ion)
            fit_d2 = _fit_dabs_for_display(ion, fit_by_ion_ms2.get(ion), qc_by_ion_ms2)
            if dual2 and len(dual2) >= 2:
                _overlay_multi_pop_fit_envelopes_on_zoom_ax(
                    ax, ms2_spec, dual2, match_m2, match_i2,
                    frag_ppm_tol=frag_ppm_tol, fit_color=ms2_fit, zorder=ZOOM_MS2_FIT_Z,
                )
            else:
                _overlay_fit_envelope_on_zoom_ax(
                    ax, ms2_spec, fit_d2, match_m2, match_i2,
                    frag_ppm_tol=frag_ppm_tol, fit_color=ms2_fit, zorder=ZOOM_MS2_FIT_Z,
                )

    if env_max <= 0:
        env_max = 1.0
    label_rows = 1
    if not is_un_row and (fit_by_ion_ms2 or fit_by_ion_ms3):
        ms2_n = sum(
            1 for pair in variants
            if fit_by_ion_ms2 and fit_by_ion_ms2.get(str(pair[2])) is not None
            and np.isfinite(float(fit_by_ion_ms2.get(str(pair[2]))))
        )
        ms3_n = sum(
            1 for pair in variants
            if fit_by_ion_ms3 and fit_by_ion_ms3.get(str(pair[2])) is not None
            and np.isfinite(float(fit_by_ion_ms3.get(str(pair[2]))))
        )
        label_rows = max(ms2_n, ms3_n, 1)
    y_headroom = 1.38 + max(0, label_rows - 1) * 0.10
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(0, env_max * y_headroom)
    ax.grid(alpha=0.22, lw=0.4)
    ax.tick_params(labelsize=tick_fs)

    if show_ylabel:
        ax.set_ylabel('intensity', fontsize=tick_fs)
    else:
        ax.tick_params(labelleft=False)
    if show_xlabel:
        ax.set_xlabel('m/z', fontsize=tick_fs)
    else:
        ax.tick_params(labelbottom=False)

    if not is_un_row:
        ms2_entries = []
        ms3_entries = []
        for vi, (ms2_spec, _ms3_spec, ion, _sort_mz) in enumerate(variants):
            dual2 = (dual_pop_by_ion_ms2 or {}).get(ion)
            dual3 = (dual_pop_by_ion_ms3 or {}).get(ion)
            if dual2 and len(dual2) >= 2:
                ax.text(
                    0.02, 0.78 - (vi * 0.12),
                    f'MS2 {_format_dual_pop_label(dual2)}',
                    transform=ax.transAxes, ha='left', va='top',
                    fontsize=max(4.0, tick_fs - 1.2), color=ms2_fit_cols[vi],
                    clip_on=False, zorder=ZOOM_FIT_LABEL_Z,
                    bbox=dict(facecolor='white', edgecolor=ms2_fit_cols[vi], alpha=0.88, pad=0.25, linewidth=0.5),
                )
            else:
                ms2_entries.append((
                    _fit_dabs_for_display(ion, fit_by_ion_ms2.get(ion) if fit_by_ion_ms2 else None, qc_by_ion_ms2),
                    ion,
                    ms2_fit_cols[vi],
                ))
            if dual3 and len(dual3) >= 2:
                ax.text(
                    0.98, 0.78 - (vi * 0.12),
                    f'MS3 {_format_dual_pop_label(dual3)}',
                    transform=ax.transAxes, ha='right', va='top',
                    fontsize=max(4.0, tick_fs - 1.2), color=ms3_fit_cols[vi],
                    clip_on=False, zorder=ZOOM_FIT_LABEL_Z,
                    bbox=dict(facecolor='white', edgecolor=ms3_fit_cols[vi], alpha=0.88, pad=0.25, linewidth=0.5),
                )
            else:
                ms3_entries.append((
                    _fit_dabs_for_display(ion, fit_by_ion_ms3.get(ion) if fit_by_ion_ms3 else None, qc_by_ion_ms3),
                    ion,
                    ms3_fit_cols[vi],
                ))
        if ms2_entries:
            _stack_corner_fit_dabs_labels(
                ax, ms2_entries, corner='left', ms_level_prefix='MS2', n_var=n_var,
            )
        if ms3_entries:
            _stack_corner_fit_dabs_labels(
                ax, ms3_entries, corner='right', ms_level_prefix='MS3', n_var=n_var,
            )
        if bimodal_timepoint:
            ax.text(
                0.5, 0.02, '2-pop',
                transform=ax.transAxes, ha='center', va='bottom',
                fontsize=max(4.2, tick_fs - 0.8), color=BIMODAL_BADGE_COLOR,
                clip_on=False, zorder=ZOOM_FIT_LABEL_Z + 1,
                bbox=dict(facecolor='#fffbeb', edgecolor=BIMODAL_BADGE_COLOR, alpha=0.92, pad=0.25, linewidth=0.6),
            )


def _dabs_timecourse_x_seconds(label: str) -> float | None:
    """Map exposure label to seconds for Dabs timecourse x-axis (UN → 1 s for log scale)."""
    lbl = str(label).strip()
    if lbl.upper() == 'UN':
        return 1.0
    t = exposure_label_to_fit_time(lbl)
    if t is None or not np.isfinite(float(t)) or float(t) <= 0:
        return None
    return float(t)


def _dabs_timecourse_shared_ylim(
    zoom_rows: list[dict[str, object]],
    plot_clusters: list[dict[str, object]],
    peptide: str,
    fit_dabs_cache_ms2: dict[tuple[str, str], dict[str, float]],
    fit_dabs_cache_ms3: dict[tuple[str, str], dict[str, float]],
    fit_qc_cache_ms2: dict[tuple[str, str], dict[str, dict[str, object]]] | None = None,
    fit_qc_cache_ms3: dict[tuple[str, str], dict[str, dict[str, object]]] | None = None,
    *,
    pad_frac: float = 0.12,
) -> tuple[float, float] | None:
    """Collect max fitted Dabs (+ visible scaled Dabs_std_1) for shared top-row y-scale."""
    ymax = 0.0
    for cluster in plot_clusters:
        for _ms2_spec, _ms3_spec, ion, _sort_mz in cluster.get('variants', []) or []:
            key = (peptide, str(ion))
            qc2 = (fit_qc_cache_ms2 or {}).get(key, {})
            qc3 = (fit_qc_cache_ms3 or {}).get(key, {})
            for cache, qc_map in (
                (fit_dabs_cache_ms2.get(key, {}), qc2),
                (fit_dabs_cache_ms3.get(key, {}), qc3),
            ):
                for row in zoom_rows:
                    lbl = str(row['lbl'])
                    if not _timepoint_qc_pass(qc_map, lbl):
                        continue
                    val = cache.get(lbl)
                    if val is not None and np.isfinite(float(val)):
                        ymax = max(ymax, float(val))
    if ymax <= 0:
        return None
    y_top = ymax * (1.0 + float(pad_frac)) * (1.0 + DABS_STD_VISIBLE_FRAC)
    return (0.0, y_top)


def _plot_fragment_column_dabs_timecourse_ax(
    ax,
    zoom_rows: list[dict[str, object]],
    variants: list[tuple[dict[str, object], dict[str, object], str, float]],
    peptide: str,
    fit_dabs_cache_ms2: dict[tuple[str, str], dict[str, float]],
    fit_dabs_cache_ms3: dict[tuple[str, str], dict[str, float]],
    fit_qc_cache_ms2: dict[tuple[str, str], dict[str, dict[str, object]]] | None = None,
    fit_qc_cache_ms3: dict[tuple[str, str], dict[str, dict[str, object]]] | None = None,
    dual_pop_cache_ms2: dict[tuple[str, str], dict[str, list[tuple[float, float]]]] | None = None,
    dual_pop_cache_ms3: dict[tuple[str, str], dict[str, list[tuple[float, float]]]] | None = None,
    *,
    title: str = '',
    tick_fs: float = 5.0,
    ylim: tuple[float, float] | None = None,
) -> None:
    """Top-of-column summary: fitted Dabs vs HDX exposure time (MS2 blue, MS3 orange)."""
    n_var = len(variants)
    ms2_line_cols = _zoom_ms2_fit_colors(n_var)
    ms3_line_cols = _zoom_ms3_fit_colors(n_var)
    n_excluded = 0

    for vi, (_ms2_spec, _ms3_spec, ion, _sort_mz) in enumerate(variants):
        key = (peptide, str(ion))
        ms2_map = fit_dabs_cache_ms2.get(key, {})
        ms3_map = fit_dabs_cache_ms3.get(key, {})
        qc2 = (fit_qc_cache_ms2 or {}).get(key, {})
        qc3 = (fit_qc_cache_ms3 or {}).get(key, {})
        xs2: list[float] = []
        ys2: list[float] = []
        es2: list[float] = []
        xs2_bad: list[float] = []
        ys2_bad: list[float] = []
        xs3: list[float] = []
        ys3: list[float] = []
        es3: list[float] = []
        xs3_bad: list[float] = []
        ys3_bad: list[float] = []
        for row in zoom_rows:
            lbl = str(row['lbl'])
            t = _dabs_timecourse_x_seconds(lbl)
            if t is None:
                continue
            d2 = ms2_map.get(lbl)
            d3 = ms3_map.get(lbl)
            if d2 is not None and np.isfinite(float(d2)):
                if _timepoint_qc_pass(qc2, lbl):
                    xs2.append(t)
                    ys2.append(float(d2))
                    es2.append(_dabs_std_for_timepoint(qc2, lbl))
                else:
                    xs2_bad.append(t)
                    ys2_bad.append(float(d2))
                    n_excluded += 1
            if d3 is not None and np.isfinite(float(d3)):
                if _timepoint_qc_pass(qc3, lbl):
                    xs3.append(t)
                    ys3.append(float(d3))
                    es3.append(_dabs_std_for_timepoint(qc3, lbl))
                else:
                    xs3_bad.append(t)
                    ys3_bad.append(float(d3))
                    n_excluded += 1
        y_span = float(ylim[1]) if ylim is not None else max([0.0, *ys2, *ys3, 1.0])
        if y_span <= 0:
            y_span = 1.0
        es2 = _visible_dabs_std_errors(es2, y_span=y_span)
        es3 = _visible_dabs_std_errors(es3, y_span=y_span)
        ion_tag = f' {ion}' if n_var > 1 else ''
        if xs2:
            ax.errorbar(
                xs2, ys2, yerr=es2, fmt='o-',
                color=ms2_line_cols[vi], lw=1.4, ms=3.5,
                capsize=2.0, capthick=0.8, elinewidth=0.9,
                alpha=0.95, label=f'MS2{ion_tag}', zorder=3 + vi,
            )
        if xs2_bad:
            ax.plot(
                xs2_bad, ys2_bad, 'x', color=ms2_line_cols[vi], ms=4.5, mew=1.1,
                alpha=0.55, zorder=2 + vi,
            )
        if xs3:
            ax.errorbar(
                xs3, ys3, yerr=es3, fmt='s--',
                color=ms3_line_cols[vi], lw=1.3, ms=3.0,
                capsize=2.0, capthick=0.8, elinewidth=0.9,
                alpha=0.92, label=f'MS3{ion_tag}', zorder=2 + vi,
            )
        if xs3_bad:
            ax.plot(
                xs3_bad, ys3_bad, 'x', color=ms3_line_cols[vi], ms=4.0, mew=1.0,
                alpha=0.55, zorder=1 + vi,
            )
        dual2 = (dual_pop_cache_ms2 or {}).get(key, {})
        dual3 = (dual_pop_cache_ms3 or {}).get(key, {})
        if dual2:
            for pi, style in enumerate(BIMODAL_POP_LINESTYLES):
                xs_p: list[float] = []
                ys_p: list[float] = []
                for row in zoom_rows:
                    lbl = str(row['lbl'])
                    t = _dabs_timecourse_x_seconds(lbl)
                    comps = dual2.get(lbl)
                    if t is None or not comps or len(comps) <= pi:
                        continue
                    xs_p.append(t)
                    ys_p.append(float(comps[pi][0]))
                if xs_p:
                    ax.plot(
                        xs_p, ys_p, linestyle=style, marker='.',
                        color=ms2_line_cols[vi], lw=0.9, ms=2.5, alpha=0.45,
                        zorder=1 + vi,
                    )
        if dual3:
            for pi, style in enumerate(BIMODAL_POP_LINESTYLES):
                xs_p = []
                ys_p = []
                for row in zoom_rows:
                    lbl = str(row['lbl'])
                    t = _dabs_timecourse_x_seconds(lbl)
                    comps = dual3.get(lbl)
                    if t is None or not comps or len(comps) <= pi:
                        continue
                    xs_p.append(t)
                    ys_p.append(float(comps[pi][0]))
                if xs_p:
                    ax.plot(
                        xs_p, ys_p, linestyle=style, marker='.',
                        color=ms3_line_cols[vi], lw=0.9, ms=2.5, alpha=0.40,
                        zorder=0 + vi,
                    )

    ax.set_xscale('log')
    ax.set_ylabel('Dabs', fontsize=tick_fs)
    ax.grid(alpha=0.28, lw=0.4, which='both')
    ax.tick_params(labelsize=tick_fs)
    if title:
        ax.set_title(title, fontsize=6.0 if n_var > 1 else 6.2, pad=2.0)
    if ax.lines or ax.containers:
        ax.legend(
            loc='lower right', fontsize=max(4.5, tick_fs - 0.5),
            framealpha=0.9, handlelength=1.2, borderpad=0.35,
        )
    if n_excluded > 0:
        ax.text(
            0.02, 0.04, f'{n_excluded} QC-excl',
            transform=ax.transAxes, ha='left', va='bottom',
            fontsize=max(4.2, tick_fs - 0.8), color='#991b1b',
            bbox=dict(facecolor='white', edgecolor='#fecaca', alpha=0.88, pad=0.8),
            zorder=20,
        )
    if ylim is not None:
        ax.set_ylim(float(ylim[0]), float(ylim[1]))


def _plot_fragment_zoom_ax(
    ax,
    spec: dict[str, object],
    match_m: np.ndarray,
    match_i: np.ndarray,
    label_keys: set[tuple[str, str]],
    frag_ppm_tol: float,
    *,
    is_un_row: bool = False,
    fit_dabs: float | None = None,
    show_xlabel: bool = False,
    show_ylabel: bool = False,
    title_fs: float = 6.0,
    tick_fs: float = 5.0,
) -> None:
    """Draw one fragment zoom cell: colored isotope stems on UN, colored underlay + black window on HDX."""
    xlo = float(spec['xlo'])
    xhi = float(spec['xhi'])
    m0_mz = float(spec['m0_mz'])
    charge = int(spec['charge'])
    clr = str(spec.get('color', TARGET_ISO_TRACE))
    ion = str(spec.get('ion', '') or '')

    env_max = 0.0
    max_deut = 0 if is_un_row else int(spec.get('n_ex', 0) or 0)
    iso_peaks = _fragment_deuteration_peak_matches(
        match_m, match_i, spec, ppm_tol=frag_ppm_tol, max_deut=max_deut,
    )
    stem_alpha = 1.0 if is_un_row else 0.65
    for pk in iso_peaks:
        ax.vlines(
            float(pk['obs_mz']), 0.0, float(pk['obs_int']),
            color=clr, lw=FRAG_MATCH_LW, alpha=stem_alpha, zorder=ZOOM_COLORED_STEM_Z,
        )
    if iso_peaks:
        env_max = max(float(pk['obs_int']) for pk in iso_peaks)
    else:
        env_max = max(
            env_max,
            _fragment_envelope_ymax(
                match_m, match_i,
                {'obs_mz': m0_mz, 'charge': charge, 'obs_int': 0.0},
                frag_ppm_tol,
            ),
        )

    ax.axvline(m0_mz, color='gray', ls=':', lw=0.6, alpha=0.45, zorder=ZOOM_M0_REF_Z)

    spec_key = (str(spec.get('peptide', '') or ''), ion)
    if spec_key in label_keys and is_un_row:
        label_clr = _darken_color(clr)
        ax.text(
            m0_mz, env_max * 1.08 if env_max > 0 else 1.0, ion,
            fontsize=max(4.5, title_fs - 0.5), ha='center', va='bottom',
            color=label_clr, clip_on=False, zorder=ZOOM_LABEL_Z,
            bbox=dict(
                facecolor='white', edgecolor=label_clr,
                alpha=0.94, pad=0.35, linewidth=0.6,
            ),
        )

    if not is_un_row:
        in_win = (match_m >= xlo) & (match_m <= xhi)
        rm, ri = match_m[in_win], match_i[in_win]
        if rm.size:
            ax.vlines(
                rm, 0, ri, color=SPECTRUM_TRACE_COLOR, lw=0.85,
                alpha=0.88, zorder=ZOOM_BLACK_TRACE_Z,
            )
        env_max = max(env_max, _fragment_envelope_ymax_in_window(match_m, match_i, xlo, xhi))

    if env_max <= 0:
        env_max = 1.0
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(0, env_max * 1.30)
    ax.grid(alpha=0.22, lw=0.4)
    ax.tick_params(labelsize=tick_fs)

    if show_ylabel:
        ax.set_ylabel('intensity', fontsize=tick_fs)
    else:
        ax.tick_params(labelleft=False)
    if show_xlabel:
        ax.set_xlabel('m/z', fontsize=tick_fs)
    else:
        ax.tick_params(labelbottom=False)

    if not is_un_row:
        _overlay_fit_dabs_on_zoom_ax(
            ax, spec, fit_dabs, match_m, match_i, frag_ppm_tol=frag_ppm_tol,
        )


def _plot_fragment_zoom_variants_on_ax(
    ax,
    specs_with_colors: list[tuple[dict[str, object], str]],
    match_m: np.ndarray,
    match_i: np.ndarray,
    label_keys: set[tuple[str, str]],
    frag_ppm_tol: float,
    *,
    is_un_row: bool = False,
    fit_dabs_by_ion: dict[str, float] | None = None,
    show_xlabel: bool = False,
    show_ylabel: bool = False,
    title_fs: float = 6.0,
    tick_fs: float = 5.0,
) -> None:
    """Overlay multiple fragment variants (e.g. z9 and (z+1)9) on one zoom axis."""
    if not specs_with_colors:
        return

    xlo = min(float(spec['xlo']) for spec, _ in specs_with_colors)
    xhi = max(float(spec['xhi']) for spec, _ in specs_with_colors)
    env_max = 0.0
    n_var = len(specs_with_colors)
    if n_var > 1:
        stem_alpha = ZOOM_OVERLAY_STEM_ALPHA_UN if is_un_row else ZOOM_OVERLAY_STEM_ALPHA_HDX
        m0_alpha = ZOOM_OVERLAY_M0_ALPHA
        label_bbox_alpha = ZOOM_OVERLAY_LABEL_BBOX_ALPHA
    else:
        stem_alpha = 1.0 if is_un_row else 0.65
        m0_alpha = 0.55
        label_bbox_alpha = 0.94

    for vi, (spec, clr) in enumerate(specs_with_colors):
        m0_mz = float(spec['m0_mz'])
        charge = int(spec['charge'])
        ion = str(spec.get('ion', '') or '')
        max_deut = 0 if is_un_row else int(spec.get('n_ex', 0) or 0)
        iso_peaks = _fragment_deuteration_peak_matches(
            match_m, match_i, spec, ppm_tol=frag_ppm_tol, max_deut=max_deut,
        )
        for pk in iso_peaks:
            ax.vlines(
                float(pk['obs_mz']), 0.0, float(pk['obs_int']),
                color=clr, lw=FRAG_MATCH_LW, alpha=stem_alpha, zorder=ZOOM_COLORED_STEM_Z,
            )
        if iso_peaks:
            env_max = max(env_max, max(float(pk['obs_int']) for pk in iso_peaks))
        else:
            env_max = max(
                env_max,
                _fragment_envelope_ymax(
                    match_m, match_i,
                    {'obs_mz': m0_mz, 'charge': charge, 'obs_int': 0.0},
                    frag_ppm_tol,
                ),
            )

        label_clr = _darken_color(clr)
        ax.axvline(m0_mz, color=label_clr, ls=':', lw=0.6, alpha=m0_alpha, zorder=ZOOM_M0_REF_Z)

        spec_key = (str(spec.get('peptide', '') or ''), ion)
        if spec_key in label_keys and is_un_row:
            ax.text(
                m0_mz, env_max * 1.08 if env_max > 0 else 1.0, ion,
                fontsize=max(4.5, title_fs - 0.5), ha='center', va='bottom',
                color=label_clr, clip_on=False, zorder=ZOOM_LABEL_Z,
                bbox=dict(
                    facecolor='white', edgecolor=label_clr,
                    alpha=label_bbox_alpha, pad=0.35, linewidth=0.6,
                ),
            )

    if not is_un_row:
        in_win = (match_m >= xlo) & (match_m <= xhi)
        rm, ri = match_m[in_win], match_i[in_win]
        if rm.size:
            ax.vlines(
                rm, 0, ri, color=SPECTRUM_TRACE_COLOR, lw=0.85,
                alpha=0.88, zorder=ZOOM_BLACK_TRACE_Z,
            )
        env_max = max(env_max, _fragment_envelope_ymax_in_window(match_m, match_i, xlo, xhi))

    if env_max <= 0:
        env_max = 1.0
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(0, env_max * 1.30)
    ax.grid(alpha=0.22, lw=0.4)
    ax.tick_params(labelsize=tick_fs)

    if show_ylabel:
        ax.set_ylabel('intensity', fontsize=tick_fs)
    else:
        ax.tick_params(labelleft=False)
    if show_xlabel:
        ax.set_xlabel('m/z', fontsize=tick_fs)
    else:
        ax.tick_params(labelbottom=False)

    if not is_un_row and fit_dabs_by_ion is not None:
        for vi, (spec, _clr) in enumerate(specs_with_colors):
            ion = str(spec.get('ion', '') or '')
            _overlay_fit_dabs_on_zoom_ax(
                ax, spec, fit_dabs_by_ion.get(ion), match_m, match_i,
                frag_ppm_tol=frag_ppm_tol,
                label_y=max(0.72, 0.96 - (vi * 0.08)),
                label_suffix=ion if n_var > 1 else '',
            )


def _zoom_fragment_column_label(spec: dict[str, object]) -> str:
    ion = str(spec.get('ion', '') or '')
    frag_seq = str(spec.get('frag_seq', '') or '')
    n_ex = int(spec.get('n_ex', 0) or 0)
    m0_mz = float(spec['m0_mz'])
    xlo, xhi = float(spec['xlo']), float(spec['xhi'])
    return f"{ion}\n{frag_seq} n_ex={n_ex}\nm0={m0_mz:.2f}\n[{xlo:.1f}-{xhi:.1f}]"


def _apply_axes_box_aspect(axes: np.ndarray, *, height_over_width: float = 1.0) -> None:
    """Set each axes panel box aspect (matplotlib: height / width)."""
    for ax in np.atleast_1d(axes).flat:
        if ax is not None:
            ax.set_box_aspect(height_over_width)


def _apply_zoom_fragment_box_aspects(
    axes: np.ndarray,
    n_plot_rows: int,
    n_cols: int,
) -> None:
    """Spectrum rows keep square panels; D-uptake row (row 0) fills its taller grid cell."""
    spec_aspect = 1.0 / ZOOM_SUBPLOT_WIDTH_MULT
    dabs_aspect = spec_aspect * float(DABS_TIMECOURSE_ROW_HEIGHT_MULT)
    for ri in range(n_plot_rows):
        aspect = dabs_aspect if ri == 0 else spec_aspect
        for fi in range(n_cols):
            ax = axes[ri, fi]
            if ax is not None:
                ax.set_box_aspect(aspect)


def _add_fragment_group_boxes(
    fig: plt.Figure,
    axes: np.ndarray,
    n_plot_rows: int,
    n_frags: int,
    *,
    pad_pt: float = 2.5,
    lw: float = 1.0,
) -> None:
    """Draw a black rectangle around each fragment column (Dabs row + spectra)."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    to_fig = fig.transFigure.inverted() + fig.dpi_scale_trans.inverted()

    for fi in range(n_frags):
        group_axes = [axes[ri, fi] for ri in range(n_plot_rows)]
        bb_disp = Bbox.union([ax.get_tightbbox(renderer) for ax in group_axes])
        bb_disp = bb_disp.padded(pad_pt)
        bb_fig = bb_disp.transformed(to_fig)
        fig.add_artist(
            Rectangle(
                (bb_fig.x0, bb_fig.y0),
                bb_fig.width,
                bb_fig.height,
                transform=fig.transFigure,
                fill=False,
                edgecolor='black',
                linewidth=lw,
                clip_on=False,
                zorder=1000,
            )
        )


def _pick_representative_bimodal_timepoint(
    dual_map: dict[str, list[tuple[float, float]]],
) -> str | None:
    best_lbl: str | None = None
    best_score = -1.0
    for lbl, comps in dual_map.items():
        if str(lbl).strip().upper() == 'UN':
            continue
        if len(comps) < 2:
            continue
        p1 = float(comps[0][1])
        p2 = float(comps[1][1])
        if min(p1, p2) < 0.12:
            continue
        sep = abs(float(comps[1][0]) - float(comps[0][0]))
        score = sep * min(p1, p2)
        if score > best_score:
            best_score = score
            best_lbl = str(lbl)
    if best_lbl is not None:
        return best_lbl
    for lbl in dual_map:
        if str(lbl).strip().upper() != 'UN':
            return str(lbl)
    return None


def _save_bimodal_scan_diagnostic_figure(
    out_dir: str,
    safe_mz: str,
    target_mz: float,
    precursor_charge: int,
    peptide: str,
    rank: int,
    ion: str,
    ms2_spec: dict[str, object],
    zoom_rows: list[dict[str, object]],
    dual_map_ms2: dict[str, list[tuple[float, float]]],
    *,
    frag_ppm_tol: float = DEFAULT_FRAG_PPM_TOL,
    max_scans: int = 6,
) -> str | None:
    """Per-scan MS2 diagnostic for a bimodal fragment at one representative timepoint."""
    lbl = _pick_representative_bimodal_timepoint(dual_map_ms2)
    if lbl is None:
        return None
    row = next((r for r in zoom_rows if str(r['lbl']) == lbl), None)
    if row is None:
        return None
    xlo = float(ms2_spec['xlo'])
    xhi = float(ms2_spec['xhi'])
    scans = per_scan_spectra_in_window(
        row['ms_by'], target_mz, 2,
        float(row['rt_lo']), float(row['rt_hi']), xlo, xhi,
    )
    if not scans:
        return None
    scans = scans[: int(max_scans)]
    integ_m = np.array([])
    integ_i = np.array([])
    for scan in scans:
        integ_m = np.concatenate([integ_m, scan['mz']]) if integ_m.size else scan['mz']
        integ_i = np.concatenate([integ_i, scan['intensity']]) if integ_i.size else scan['intensity']
    if integ_m.size:
        integ_m, integ_i = _sum_peaks_by_exact_mz(integ_m, integ_i)

    n_panels = len(scans) + 1
    n_cols = min(4, n_panels)
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 2.4 * n_rows), squeeze=False)
    comps = dual_map_ms2.get(lbl, [])
    frag_seq = str(ms2_spec.get('frag_seq', '') or '')

    for si, scan in enumerate(scans):
        ax = axes[si // n_cols, si % n_cols]
        ax.vlines(scan['mz'], 0, scan['intensity'], color='#64748b', lw=0.8, alpha=0.85)
        ax.set_title(f'scan RT={scan["rt"]:.1f} min', fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_xlim(xlo, xhi)
        ax.grid(alpha=0.2, lw=0.3)

    ax_sum = axes[len(scans) // n_cols, len(scans) % n_cols]
    if integ_m.size:
        ax_sum.vlines(integ_m, 0, integ_i, color='#1d4ed8', lw=1.0, alpha=0.85)
        _overlay_multi_pop_fit_envelopes_on_zoom_ax(
            ax_sum, ms2_spec, comps, integ_m, integ_i,
            frag_ppm_tol=frag_ppm_tol, fit_color=ZOOM_MS2_FIT_COLORS[0],
        )
    ax_sum.set_title(f'integrated {lbl}', fontsize=7)
    ax_sum.tick_params(labelsize=6)
    ax_sum.set_xlim(xlo, xhi)
    ax_sum.grid(alpha=0.2, lw=0.3)
    if comps:
        ax_sum.text(
            0.02, 0.96, _format_dual_pop_label(comps),
            transform=ax_sum.transAxes, ha='left', va='top', fontsize=6,
            color=BIMODAL_BADGE_COLOR,
            bbox=dict(facecolor='white', edgecolor=BIMODAL_BADGE_COLOR, alpha=0.9, pad=0.3),
        )

    for idx in range(n_panels, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].axis('off')

    proline_note = ' (Pro-rich)' if fragment_has_proline(frag_seq) else ''
    fig.suptitle(
        f'Bimodal scan diagnostic: mz{target_mz:.4f} z{precursor_charge} | r{rank}:{peptide} {ion}{proline_note}\n'
        f'timepoint={lbl} | grey=individual MS2 scans, blue=RT-integrated + 2-pop fit',
        fontsize=9,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    ion_safe = re.sub(r'[^A-Za-z0-9+\-]+', '_', ion).strip('_')
    out_path = (
        f'{out_dir}/timecourse_{safe_mz}_z{precursor_charge}_bimodal_scans_'
        f'r{rank}_{peptide[:18]}__{ion_safe}__{lbl.replace(" ", "")}.png'
    )
    plt.savefig(out_path, dpi=PLOT_DPI, bbox_inches='tight', pad_inches=0.12, facecolor='white')
    plt.close(fig)
    print(f'Saved bimodal scan diagnostic {out_path}')
    return out_path


def _save_precursor_fragment_zoom_figures(
    out_dir: str,
    safe_mz: str,
    target_mz: float,
    precursor_charge: int,
    act2_label: str,
    act3_label: str,
    ms2_specs: list[dict[str, object]],
    ms3_specs: list[dict[str, object]],
    zoom_rows: list[dict[str, object]],
    m0_groups: list[dict[str, object]],
    comet_cfg: dict,
    theo_frag_cache: dict[tuple[str, int], list],
    frag_ppm_tol: float,
    un_frag_label_keys: dict[int, set[tuple[str, str]]],
    hdx_fit_peptides: set[str] | None = None,
    rank_only: int | None = None,
    timecourse_sheets: dict[str, pd.DataFrame] | None = None,
    skip_excel: bool = False,
) -> list[str]:
    """One figure per registry precursor: one MS2+MS3 overlay column per fragment."""
    groups = _group_ms2_ms3_zoom_pairs_by_precursor(ms2_specs, ms3_specs, m0_groups)
    groups = _filter_zoom_groups_by_hdx_deuteration_evidence(
        groups, zoom_rows, target_mz, m0_groups, precursor_charge,
        comet_cfg, theo_frag_cache, frag_ppm_tol, un_frag_label_keys,
        hdx_fit_peptides=hdx_fit_peptides,
    )
    if not groups or not zoom_rows:
        return []
    if rank_only is not None:
        groups = [g for g in groups if int(g.get('rank', 999999)) == int(rank_only)]
        if not groups:
            return []

    saved: list[str] = []
    n_time = len(zoom_rows)
    for grp in groups:
        peptide = str(grp['peptide'])
        rank = int(grp['rank'])
        clr = str(grp['color'])
        pairs = list(grp.get('pairs', []) or [])
        if not pairs:
            continue

        fit_dabs_cache_ms2, fit_dabs_cache_ms3, fit_qc_cache_ms2, fit_qc_cache_ms3, dual_pop_cache_ms2, dual_pop_cache_ms3, bimodal_keys, bimodal_labels_cache = _build_fragment_fit_dabs_cache(
            zoom_rows, target_mz, precursor_charge, pairs,
            act2_label=act2_label, act3_label=act3_label,
            frag_ppm_tol=frag_ppm_tol,
        )
        plot_clusters = _cluster_radical_variant_zoom_pairs(pairs)
        n_frags = len(plot_clusters)
        n_cols = n_frags
        shared_dabs_ylim = _dabs_timecourse_shared_ylim(
            zoom_rows, plot_clusters, peptide,
            fit_dabs_cache_ms2, fit_dabs_cache_ms3,
            fit_qc_cache_ms2, fit_qc_cache_ms3,
        )
        n_plot_rows = n_time + 1
        cell_in = _zoom_subplot_cell_in(n_cols, n_plot_rows)
        cell_w = cell_in * ZOOM_SUBPLOT_WIDTH_MULT
        fig_w = max(10.0, n_cols * cell_w + 0.6)
        dabs_row_units = float(DABS_TIMECOURSE_ROW_HEIGHT_MULT)
        spectrum_row_units = float(n_plot_rows - 1)
        fig_h = max(5.5, (dabs_row_units + spectrum_row_units) * cell_in + 0.5)
        gs_spacing = _zoom_gridspec_spacing(n_plot_rows, n_cols)
        height_ratios = [dabs_row_units] + [1.0] * (n_plot_rows - 1)
        fig, axes = plt.subplots(
            n_plot_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False,
            gridspec_kw={'height_ratios': height_ratios, **gs_spacing},
        )

        title_fs = 6.0 if n_cols <= 12 else (5.2 if n_cols <= 24 else 4.5)
        tick_fs = 5.0 if n_cols <= 12 else (4.2 if n_cols <= 24 else 3.6)
        variant_legend_handles: list[Line2D] = []

        for fi, cluster in enumerate(plot_clusters):
            variants = list(cluster.get('variants', []) or [])
            if not variants:
                continue
            ion_lines = [str(pair[2]) for pair in variants]
            frag_seq = str(variants[0][0].get('frag_seq', '') or '')
            m0_lines = [
                f'm0 MS2={float(pair[0]["m0_mz"]):.2f} MS3={float(pair[1]["m0_mz"]):.2f}'
                for pair in variants
            ]
            if len(variants) <= 2:
                col_title = (
                    f'{frag_seq}\n' + '\n'.join(ion_lines)
                    + f'\nMS2 ({act2_label}) + MS3 ({act3_label})\n' + '\n'.join(m0_lines)
                )
            else:
                col_title = (
                    f'{frag_seq}\n{", ".join(ion_lines[:2])}, +{len(variants) - 2} more\n'
                    f'MS2 ({act2_label}) + MS3 ({act3_label})'
                )
            ion_key = str(variants[0][2])
            if (peptide, ion_key) in bimodal_keys:
                col_title += _bimodal_column_title_suffix(frag_seq)
            ax_dabs = axes[0, fi]
            _plot_fragment_column_dabs_timecourse_ax(
                ax_dabs, zoom_rows, variants, peptide,
                fit_dabs_cache_ms2, fit_dabs_cache_ms3,
                fit_qc_cache_ms2, fit_qc_cache_ms3,
                dual_pop_cache_ms2, dual_pop_cache_ms3,
                title=col_title, tick_fs=tick_fs, ylim=shared_dabs_ylim,
            )
            if fi == 0:
                ax_dabs.set_ylabel('Dabs', fontsize=tick_fs)
            if fi == n_cols - 1:
                ax_dabs.set_xlabel('Exposure time (s)', fontsize=tick_fs)
            ax_dabs.tick_params(labelbottom=True)
            if len(variants) > 1:
                for vi, pair in enumerate(variants):
                    ion_lbl = str(pair[2])
                    if any(h.get_label() == ion_lbl for h in variant_legend_handles):
                        continue
                    variant_legend_handles.append(
                        Line2D(
                            [0], [0],
                            color=_zoom_ms2_match_colors(len(variants))[vi],
                            lw=FRAG_MATCH_LW,
                            label=f'{ion_lbl} (MS2 blue / MS3 orange)',
                        )
                    )

        for ti, row in enumerate(zoom_rows):
            lbl = str(row['lbl'])
            is_un_row = lbl.strip().upper() == 'UN'
            plot_row = ti + 1
            match_m2, match_i2, label_keys2 = _msx_row_fragment_context(
                row['ms_by'], target_mz, 2,
                float(row['rt_lo']), float(row['rt_hi']),
                m0_groups, precursor_charge, comet_cfg, theo_frag_cache, frag_ppm_tol,
                lbl, un_frag_label_keys,
            )
            match_m3, match_i3, label_keys3 = _msx_row_fragment_context(
                row['ms_by'], target_mz, 3,
                float(row['rt_lo']), float(row['rt_hi']),
                m0_groups, precursor_charge, comet_cfg, theo_frag_cache, frag_ppm_tol,
                lbl, un_frag_label_keys,
            )

            for fi, cluster in enumerate(plot_clusters):
                variants = list(cluster.get('variants', []) or [])
                if not variants:
                    continue
                ax = axes[plot_row, fi]

                fit_by_ion_ms2 = {
                    str(pair[2]): fit_dabs_cache_ms2.get((peptide, str(pair[2])), {}).get(lbl)
                    for pair in variants
                } if not is_un_row else None
                fit_by_ion_ms3 = {
                    str(pair[2]): fit_dabs_cache_ms3.get((peptide, str(pair[2])), {}).get(lbl)
                    for pair in variants
                } if not is_un_row else None
                qc_by_ion_ms2 = {
                    str(pair[2]): fit_qc_cache_ms2.get((peptide, str(pair[2])), {}).get(lbl, {})
                    for pair in variants
                } if not is_un_row else None
                qc_by_ion_ms3 = {
                    str(pair[2]): fit_qc_cache_ms3.get((peptide, str(pair[2])), {}).get(lbl, {})
                    for pair in variants
                } if not is_un_row else None
                dual_pop_by_ion_ms2 = {
                    str(pair[2]): dual_pop_cache_ms2.get((peptide, str(pair[2])), {}).get(lbl)
                    for pair in variants
                } if not is_un_row else None
                dual_pop_by_ion_ms3 = {
                    str(pair[2]): dual_pop_cache_ms3.get((peptide, str(pair[2])), {}).get(lbl)
                    for pair in variants
                } if not is_un_row else None
                bimodal_tp = any(
                    lbl in bimodal_labels_cache.get((peptide, str(pair[2])), set())
                    for pair in variants
                )

                _plot_fragment_zoom_ms2_ms3_combined_ax(
                    ax, variants,
                    match_m2, match_i2, label_keys2,
                    match_m3, match_i3, label_keys3,
                    frag_ppm_tol,
                    is_un_row=is_un_row,
                    fit_by_ion_ms2=fit_by_ion_ms2,
                    fit_by_ion_ms3=fit_by_ion_ms3,
                    dual_pop_by_ion_ms2=dual_pop_by_ion_ms2,
                    dual_pop_by_ion_ms3=dual_pop_by_ion_ms3,
                    bimodal_timepoint=bimodal_tp,
                    qc_by_ion_ms2=qc_by_ion_ms2,
                    qc_by_ion_ms3=qc_by_ion_ms3,
                    show_xlabel=(ti == n_time - 1),
                    show_ylabel=False,
                    title_fs=title_fs,
                    tick_fs=tick_fs,
                )

                if fi == 0:
                    ax.annotate(
                        lbl, xy=(0.0, 0.5), xycoords='axes fraction',
                        xytext=(-8, 0), textcoords='offset points',
                        ha='right', va='center', fontsize=7, clip_on=False,
                    )

        pep_stem = _zoom_precursor_filename_stem(peptide, rank, plot_clusters)
        out_path = (
            f'{out_dir}/timecourse_{safe_mz}_z{precursor_charge}_zoom_{pep_stem}.png'
        )
        n_variant_cols = sum(1 for c in plot_clusters if c.get('overlay'))
        fig.suptitle(
            f'Fragment zoom: mz{target_mz:.4f} z{precursor_charge} | r{rank}:{peptide}\n'
            f'top row: Dabs vs HDX time (± Dabs_std_1 normalized/scaled; MS2 blue, MS3 orange) | rows ↓ UN → HDX ({n_time} timepoints) | '
            f'{n_frags} fragment columns'
            f'{f" ({n_variant_cols} radical overlays)" if n_variant_cols else ""} | '
            f'spectra: MS3 brown/orange under MS2 grey/blue | fit D: MS2 top-left, MS3 top-right',
            fontsize=10, y=0.998,
        )
        fig.text(
            0.5, 0.012,
            '→ N-term c: short→long | C-term z: long→short | top: log exposure | '
            'spectra: grey/blue=MS2, brown/orange=MS3',
            ha='center', va='bottom', fontsize=8,
        )
        legend_handles = [
            Line2D([0], [0], color=ZOOM_MS2_RAW_COLOR, lw=1.0, label='MS2 raw (grey)'),
            Line2D([0], [0], color=ZOOM_MS2_MATCH_COLORS[0], lw=FRAG_MATCH_LW, label='MS2 matched (blue)'),
            Line2D([0], [0], color=ZOOM_MS2_FIT_COLORS[0], lw=ZOOM_FIT_STEM_LW, ls='--', label='MS2 fit (dark blue)'),
            Line2D([0], [0], color=ZOOM_MS3_RAW_COLOR, lw=1.0, label='MS3 raw (brown)'),
            Line2D([0], [0], color=ZOOM_MS3_MATCH_COLORS[0], lw=FRAG_MATCH_LW, label='MS3 matched (orange)'),
            Line2D([0], [0], color=ZOOM_MS3_FIT_COLORS[0], lw=ZOOM_FIT_STEM_LW, ls='--', label='MS3 fit (dark orange)'),
            Line2D([0], [0], color=clr, lw=FRAG_MATCH_LW, label=f'r{rank}:{peptide}'),
        ]
        legend_handles.extend(variant_legend_handles)
        fig.legend(
            handles=legend_handles,
            loc='upper left',
            bbox_to_anchor=(0.01, 0.965),
            fontsize=6.5,
            framealpha=0.92,
            handlelength=1.4,
        )
        zoom_left = 0.08 if n_cols <= 4 else (0.06 if n_cols <= 10 else 0.05)
        _apply_zoom_fragment_box_aspects(axes, n_plot_rows, n_cols)
        plt.tight_layout(rect=[zoom_left, 0.02, 1.0, 0.91])
        fig.canvas.draw()
        _add_fragment_group_boxes(fig, axes, n_plot_rows, n_frags)
        plt.savefig(
            out_path, dpi=PLOT_DPI, bbox_inches='tight',
            pad_inches=ZOOM_SAVE_PAD_INCHES, facecolor='white',
        )
        plt.close(fig)
        print(
            f'Saved {out_path}  ({n_plot_rows} rows × {n_cols} cols, '
            f'{n_frags} MS2+MS3 overlay columns incl. Dabs timecourse row, '
            f'{ZOOM_SUBPLOT_WIDTH_MULT:g}:1 panels)'
        )
        saved.append(out_path)

        if not skip_excel:
            try:
                xlsx_path = _save_zoom_excel_companion(
                    out_path,
                    target_mz=target_mz,
                    precursor_charge=precursor_charge,
                    peptide=peptide,
                    rank=rank,
                    zoom_rows=zoom_rows,
                    plot_clusters=plot_clusters,
                    m0_groups=m0_groups,
                    comet_cfg=comet_cfg,
                    theo_frag_cache=theo_frag_cache,
                    frag_ppm_tol=frag_ppm_tol,
                    un_frag_label_keys=un_frag_label_keys,
                    fit_dabs_cache_ms2=fit_dabs_cache_ms2,
                    fit_dabs_cache_ms3=fit_dabs_cache_ms3,
                    fit_qc_cache_ms2=fit_qc_cache_ms2,
                    fit_qc_cache_ms3=fit_qc_cache_ms3,
                    dual_pop_cache_ms2=dual_pop_cache_ms2,
                    dual_pop_cache_ms3=dual_pop_cache_ms3,
                    shared_dabs_ylim=shared_dabs_ylim,
                    timecourse_sheets=timecourse_sheets,
                )
                if xlsx_path:
                    saved.append(xlsx_path)
            except OSError as exc:
                print(f'WARNING: skipped zoom Excel for {out_path}: {exc}')

        for cluster in plot_clusters:
            variants = list(cluster.get('variants', []) or [])
            if not variants:
                continue
            ion = str(variants[0][2])
            key = (peptide, ion)
            if key not in bimodal_keys:
                continue
            dual_ms2 = dual_pop_cache_ms2.get(key, {})
            if not dual_ms2:
                continue
            diag_path = _save_bimodal_scan_diagnostic_figure(
                out_dir, safe_mz, target_mz, precursor_charge,
                peptide, rank, ion, variants[0][0], zoom_rows, dual_ms2,
                frag_ppm_tol=frag_ppm_tol,
            )
            if diag_path:
                saved.append(diag_path)
    return saved


def _plot_integrated_msx_spectrum_panel(
    ax,
    title: str,
    ms_by,
    target_mz: float,
    level: int,
    rt_lo: float,
    rt_hi: float,
    m0_groups: list[dict[str, object]],
    precursor_charge: int,
    comet_cfg: dict,
    theo_frag_cache: dict[tuple[str, int], list],
    frag_ppm_tol: float,
    lbl: str,
    un_frag_label_keys: dict[int, set[tuple[str, str]]],
    un_col3_group_indices: set[int] | None = None,
) -> None:
    _, raw_m, raw_i = _collect_msx_raw_peaks(ms_by, target_mz, level, rt_lo, rt_hi, 100, 2000)
    match_m, match_i = _sum_peaks_by_exact_mz(raw_m, raw_i)
    frag_matches = (
        _collect_col3_fragment_matches(
            match_m, match_i, m0_groups, precursor_charge, comet_cfg, theo_frag_cache, frag_ppm_tol,
            un_col3_group_indices=un_col3_group_indices,
        )
        if m0_groups and match_m.size else []
    )
    is_un_row = str(lbl).strip().upper() == 'UN'
    if is_un_row:
        label_matches = frag_matches
    else:
        label_matches = _filter_fragments_to_un_control(
            frag_matches, un_frag_label_keys[level],
        )

    mz_lo, mz_hi = 100.0, 2000.0
    nn, mc, hc = integrate_msx(
        ms_by, target_mz, level, rt_lo, rt_hi,
        mz_lo, mz_hi, bin_size=FRAG_SPECTRUM_PLOT_BIN_SIZE,
    )
    y_ref_max = None

    ax.set_title(f'{title} ∑[integration] n={nn}', fontsize=8)
    if mc.size and np.any(hc > 0):
        ax.set_xlim(mz_lo, mz_hi)
        _shade_integration_panel(ax, mz_lo, mz_hi)
        stem_matches = frag_matches if is_un_row else label_matches
        if stem_matches:
            _plot_matched_fragment_stems(ax, stem_matches)
        ax.vlines(
            mc, 0, hc, color=SPECTRUM_TRACE_COLOR, lw=SPECTRUM_TRACE_LW,
            alpha=0.92, zorder=5,
        )
        ax.set_ylim(0, float(hc.max()) * 1.18)
        if label_matches:
            _annotate_matched_fragments(ax, label_matches, hc)
    else:
        ax.text(
            0.5, 0.5, 'no scans in window', transform=ax.transAxes,
            ha='center', va='center', color='gray', fontsize=8,
        )
        ax.set_xlim(mz_lo, mz_hi)
        _shade_integration_panel(ax, mz_lo, mz_hi)


def _match_ms1_isotope_peaks_in_spectrum(
    mc: np.ndarray,
    hc: np.ndarray,
    mono_mz: float,
    charge: int,
    lbl: str,
    target_mz: float,
    base_ppm_tol: float,
    peptide_seq: str = '',
) -> list[dict[str, object]]:
    """Find integrated MS1 peaks using full get_na_isotope convolved envelope positions."""
    if mc.size == 0 or not np.isfinite(float(mono_mz)):
        return []
    iso_cfg = _isotope_xic_settings(lbl, mono_mz, target_mz, base_ppm_tol, charge)
    ppm_tol = float(iso_cfg['ppm_tol'])
    pep = str(peptide_seq or '').strip().upper()
    is_un_row = str(lbl).strip().upper() == 'UN'
    n_ex = _exchangeable_amide_count(pep) if pep else 0
    max_deut = 0 if is_un_row else n_ex
    peaks = _ms1_precursor_deuteration_peak_matches(
        mc, hc, pep, mono_mz, charge, ppm_tol,
        target_mz=target_mz, lbl=lbl,
    )
    out: list[dict[str, object]] = []
    for pk in peaks:
        k = int(pk.get('k', 0))
        iso_lbl = ISO_XIC_LABELS[k] if 0 <= k < len(ISO_XIC_LABELS) else f'M+{k}'
        out.append(
            {
                'iso_label': iso_lbl,
                'center_mz': float(pk.get('theo_mz', np.nan)),
                'obs_mz': float(pk['obs_mz']),
                'obs_int': float(pk['obs_int']),
                'deut_level': float(pk.get('d', 0.0)),
                'isotope_index': float(k),
                'anchor_mz': float(pk.get('anchor_mz', mono_mz)),
            }
        )
    return out


def _plot_ms1_registry_isotope_matches(
    ax,
    mc: np.ndarray,
    hc: np.ndarray,
    m0_groups: list[dict[str, object]],
    charge: int,
    target_mz: float,
    lbl: str,
    base_ppm_tol: float,
    un_col3_group_indices: set[int] | None = None,
) -> list[Line2D]:
    """Plot MS1 stems color-coded by d×get_na_isotope convolution matches."""
    is_un_row = str(lbl).strip().upper() == 'UN'
    legend_handles: list[Line2D] = []
    plot_groups = _m0_groups_for_col3_row(m0_groups, lbl, un_col3_group_indices)
    for gi, grp in enumerate(plot_groups):
        orig_gi = m0_groups.index(grp) if grp in m0_groups else gi
        mono_mz = float(grp['mono_mz'])
        iso_cfg = _isotope_xic_settings(lbl, mono_mz, target_mz, base_ppm_tol, charge)
        ppm_tol = float(iso_cfg['ppm_tol'])
        pep_seq = str(grp['peptides'][0]['peptide_sequence'] if grp.get('peptides') else '')
        n_ex = _exchangeable_amide_count(pep_seq)
        max_deut = 0 if is_un_row else n_ex
        peaks = _ms1_precursor_deuteration_peak_matches(
            mc, hc, pep_seq, mono_mz, charge, ppm_tol,
            target_mz=target_mz, lbl=lbl,
        )
        if not peaks:
            continue
        for pk in peaks:
            stem_clr = _ms1_deut_stem_color(
                float(pk['d']), float(pk['k']), is_un_row=is_un_row, gi=orig_gi,
            )
            ax.vlines(
                float(pk['obs_mz']), 0.0, float(pk['obs_int']),
                color=stem_clr, lw=MS1_ISOTOPE_MATCH_LW, alpha=0.95, zorder=4,
            )
        pep_labels = [f"r{int(r['rank'])}:{r['peptide_sequence']}" for r in grp['peptides']]
        mode = 'UN conv iso' if is_un_row else 'HDX d×iso'
        legend_handles.append(
            Line2D(
                [0], [0], color=_peptide_group_color(orig_gi), lw=MS1_ISOTOPE_MATCH_LW,
                label=f"{' / '.join(pep_labels)} {mode}",
            )
        )
    return legend_handles


def annotate_msx_horizontal(ax, mc, hc, top_n=8):
    if mc.size == 0 or not np.any(hc > 0):
        return
    ymax = float(hc.max())
    idx = np.argsort(hc)[::-1]
    keep = []
    min_dx = 35.0
    for j in idx:
        if hc[j] < ymax * 0.05:
            continue
        x = float(mc[j])
        if all(abs(x - kx) >= min_dx for kx, _ in keep):
            keep.append((x, float(hc[j])))
        if len(keep) >= top_n:
            break
    tiers = [1.02, 1.08, 1.14, 1.20]
    for i, (x, _y) in enumerate(sorted(keep, key=lambda t: t[0])):
        tier = tiers[i % len(tiers)]
        ax.text(
            x, ymax * tier, f'{x:.2f}',
            fontsize=6.5, rotation=0, ha='center', va='bottom', color='black',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.65, pad=1.0),
        )


def _load_hdx_fit_peptides_by_window(path: str) -> dict[tuple[float, int], set[str]]:
    """Peptide sequences slated for HDX quantification, keyed by (target_mz, charge)."""
    if not path or not os.path.isfile(path):
        return {}
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}
    if df.empty:
        return {}
    mz_col = 'target_peptide_mz' if 'target_peptide_mz' in df.columns else None
    z_col = 'peptide_charge' if 'peptide_charge' in df.columns else None
    pep_col = 'peptide_sequence' if 'peptide_sequence' in df.columns else (
        'peptide' if 'peptide' in df.columns else None
    )
    if not mz_col or not z_col or not pep_col:
        return {}
    out: dict[tuple[float, int], set[str]] = {}
    for _, row in df.iterrows():
        mz = pd.to_numeric(row.get(mz_col), errors='coerce')
        z = pd.to_numeric(row.get(z_col), errors='coerce')
        pep = str(row.get(pep_col, '') or '').strip().upper()
        if pd.isna(mz) or pd.isna(z) or not pep:
            continue
        out.setdefault((round(float(mz), 3), int(z)), set()).add(pep)
    return out


def _load_registry_by_window(path: str) -> dict[tuple[float, int], list[dict[str, object]]]:
    if not path or (not os.path.isfile(path)):
        return {}
    try:
        reg = pd.read_csv(path)
    except Exception:
        return {}
    if reg.empty:
        return {}
    if "target_peptide_mz" not in reg.columns or "peptide_charge" not in reg.columns:
        return {}
    pep_col = "peptide_sequence" if "peptide_sequence" in reg.columns else ("peptide" if "peptide" in reg.columns else "")
    if not pep_col:
        return {}
    reg["_mz"] = pd.to_numeric(reg["target_peptide_mz"], errors="coerce").round(3)
    reg["_z"] = pd.to_numeric(reg["peptide_charge"], errors="coerce")
    reg = reg[reg["_mz"].notna() & reg["_z"].notna()].copy()
    if reg.empty:
        return {}
    if "coisolation_rank" in reg.columns:
        reg["_rank"] = pd.to_numeric(reg["coisolation_rank"], errors="coerce").fillna(999999).astype(int)
    else:
        reg["_rank"] = 999999
    if "hybrid_relative_abundance_in_window" in reg.columns:
        reg["_hyb"] = pd.to_numeric(reg["hybrid_relative_abundance_in_window"], errors="coerce")
    else:
        reg["_hyb"] = np.nan
    if "mono_mz" in reg.columns:
        reg["_mono_mz"] = pd.to_numeric(reg["mono_mz"], errors="coerce")
    else:
        reg["_mono_mz"] = np.nan
    out: dict[tuple[float, int], list[dict[str, object]]] = {}
    reg = reg.sort_values(["_mz", "_z", "_rank", "_hyb"], ascending=[True, True, True, False])
    for _, r in reg.iterrows():
        key = (float(r["_mz"]), int(r["_z"]))
        mono_mz = float(r.get("_mono_mz")) if pd.notna(r.get("_mono_mz")) else np.nan
        out.setdefault(key, []).append(
            {
                "peptide_sequence": str(r.get(pep_col, "") or "").strip(),
                "rank": int(r.get("_rank", 999999) or 999999),
                "hybrid_rel": float(r.get("_hyb")) if pd.notna(r.get("_hyb")) else np.nan,
                "mono_mz": mono_mz,
            }
        )
    return out


def _registry_annotation(reg_rows: list[dict[str, object]], max_items: int = 6) -> str:
    if not reg_rows:
        return "Registry peptides: none"
    chunks: list[str] = []
    for rr in reg_rows[:max_items]:
        pep = str(rr.get("peptide_sequence", "") or "").strip()
        if not pep:
            continue
        rank = rr.get("rank", 999999)
        hyb = rr.get("hybrid_rel", np.nan)
        if np.isfinite(float(hyb)):
            chunks.append(f"r{int(rank)}:{pep} ({float(hyb):.1%})")
        else:
            chunks.append(f"r{int(rank)}:{pep}")
    if len(reg_rows) > max_items:
        chunks.append(f"... +{len(reg_rows) - max_items} more")
    return "Registry peptides (step 1): " + " | ".join(chunks)


def _registry_peptide_legend_handles(
    m0_groups: list[dict[str, object]],
    *,
    group_indices: set[int] | None = None,
    lw: float = FRAG_MATCH_LW,
) -> list[Line2D]:
    """Color-coded legend entries for each unique col3 registry peptide."""
    entries: list[tuple[int, str, str]] = []
    seen: set[str] = set()
    for gi, grp in enumerate(m0_groups):
        if group_indices is not None and int(gi) not in group_indices:
            continue
        clr = _peptide_group_color(gi)
        for row in grp.get('peptides', []):
            pep = str(row.get('peptide_sequence', '') or '').strip()
            if not pep or pep in seen:
                continue
            seen.add(pep)
            rank = int(row.get('rank', 999999) or 999999)
            entries.append((rank, pep, clr))
    entries.sort(key=lambda t: (t[0], t[1]))
    return [
        Line2D([0], [0], color=clr, lw=lw, label=f'r{rank}:{pep}')
        for rank, pep, clr in entries
    ]


def _registry_peptide_legend_ncol(n_peptides: int) -> int:
    if n_peptides <= 1:
        return 1
    if n_peptides <= 4:
        return n_peptides
    if n_peptides <= 8:
        return 4
    if n_peptides <= 15:
        return 5
    return 6


def _add_registry_peptide_figure_legend(
    fig,
    m0_groups: list[dict[str, object]],
    *,
    group_indices: set[int] | None = None,
    y_anchor: float = 0.965,
    fontsize: float = 7.0,
) -> float:
    """Draw a color-coded registry peptide key; return top rect for tight_layout."""
    handles = _registry_peptide_legend_handles(m0_groups, group_indices=group_indices)
    if not handles:
        return 0.94
    n_pep = len(handles)
    ncol = _registry_peptide_legend_ncol(n_pep)
    n_rows = (n_pep + ncol - 1) // ncol
    fs = fontsize if n_pep <= 10 else max(5.5, fontsize - 0.8)
    fig.legend(
        handles=handles,
        loc='upper left',
        bbox_to_anchor=(0.01, y_anchor),
        ncol=ncol,
        fontsize=fs,
        framealpha=0.92,
        handlelength=1.4,
        columnspacing=0.9,
        borderpad=0.35,
        labelspacing=0.30,
        title='Registry peptides',
        title_fontsize=fs + 0.3,
    )
    top_rect = y_anchor - 0.012 - (n_rows * 0.024)
    return max(0.76, top_rect)


def _build_m0_trace_groups(
    reg_rows: list[dict[str, object]], ppm_tol: float
) -> list[dict[str, object]]:
    """One plotted trace per unique mono_m/z; legend lists all co-isolated peptides at that mass."""
    candidates: list[dict[str, object]] = []
    seen_peptides: set[str] = set()
    for rr in reg_rows:
        mono_mz = rr.get("mono_mz", np.nan)
        pep = str(rr.get("peptide_sequence", "") or "").strip()
        if not pep or not np.isfinite(float(mono_mz)) or pep in seen_peptides:
            continue
        seen_peptides.add(pep)
        candidates.append(
            {
                "peptide_sequence": pep,
                "rank": int(rr.get("rank", 999999) or 999999),
                "mono_mz": float(mono_mz),
            }
        )
    if not candidates:
        return []
    by_mono: dict[float, list[dict[str, object]]] = {}
    for cand in candidates:
        mono_key = round(float(cand["mono_mz"]), 4)
        by_mono.setdefault(mono_key, []).append(cand)
    groups: list[dict[str, object]] = []
    for mono_key in sorted(by_mono.keys()):
        rows = sorted(by_mono[mono_key], key=lambda r: int(r["rank"]))
        mono_mz = float(rows[0]["mono_mz"])
        pep_labels = [f"r{int(r['rank'])}:{r['peptide_sequence']}" for r in rows]
        groups.append(
            {
                "mono_mz": mono_mz,
                "peptides": rows,
                "primary_rank": int(min(int(r["rank"]) for r in rows)),
                "legend": f"{' / '.join(pep_labels)} m0={mono_mz:.4f} ±{ppm_tol:g}ppm",
            }
        )
    return groups


def _isotope_xic_settings(
    lbl: str,
    mono_mz: float,
    target_mz: float,
    base_ppm: float,
    charge: int,
    n_isotopes: int = N_PEPTIDE_ISOTOPES,
) -> dict[str, float | str]:
    """Choose isotope centers/ppm for UN vs HDX rows (deuterated envelope shift)."""
    spacing = C13 / max(1, int(charge))
    if str(lbl).strip().upper() == 'UN':
        return {
            'shift_da': 0.0,
            'ppm_tol': float(base_ppm),
            'mode_label': f'UN theoretical ±{base_ppm:g}ppm',
        }
    shift_da = max(0.0, float(target_mz) - float(mono_mz))
    span_da = shift_da + max(0, int(n_isotopes) - 1) * spacing
    center = float(mono_mz) + (0.5 * span_da)
    span_ppm = (span_da / max(center, 1.0)) * 1e6
    ppm_tol = max(float(base_ppm), 0.40 * span_ppm, 25.0)
    return {
        'shift_da': float(shift_da),
        'ppm_tol': float(ppm_tol),
        'mode_label': f'HDX +{shift_da:.2f}Da shift ±{ppm_tol:.0f}ppm',
    }


def _peptide_isotope_xics_in_window(
    ms1_specs: list,
    mono_mz: float,
    charge: int,
    rt_lo: float,
    rt_hi: float,
    ppm_tol: float,
    center_shift_da: float = 0.0,
    n_isotopes: int = N_PEPTIDE_ISOTOPES,
    min_tol_da: float = 0.015,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    """MS1 XIC for each isotope (±ppm) and their sum, restricted to an RT window."""
    if not np.isfinite(float(mono_mz)):
        empty = np.array([], dtype=float)
        return empty, [empty.copy() for _ in range(int(n_isotopes))], empty.copy()
    spacing = C13 / max(1, int(charge))
    rts: list[float] = []
    per_iso: list[list[float]] = [[] for _ in range(int(n_isotopes))]
    sum_vals: list[float] = []
    for rt, mzs, intensities in ms1_specs:
        rt_f = float(rt)
        if rt_f < float(rt_lo) or rt_f > float(rt_hi):
            continue
        mzs = np.asarray(mzs, dtype=float)
        intensities = np.asarray(intensities, dtype=float)
        iso_at_rt = [0.0] * int(n_isotopes)
        for k in range(int(n_isotopes)):
            center = float(mono_mz) + float(center_shift_da) + (k * spacing)
            tol = max(center * float(ppm_tol) / 1e6, float(min_tol_da))
            m = np.abs(mzs - center) <= tol
            if np.any(m):
                iso_at_rt[k] = float(intensities[m].sum())
        rts.append(rt_f)
        for k in range(int(n_isotopes)):
            per_iso[k].append(iso_at_rt[k])
        sum_vals.append(float(sum(iso_at_rt)))
    if not rts:
        empty = np.array([], dtype=float)
        return empty, [empty.copy() for _ in range(int(n_isotopes))], empty.copy()
    return (
        np.asarray(rts, dtype=float),
        [np.asarray(v, dtype=float) for v in per_iso],
        np.asarray(sum_vals, dtype=float),
    )


def _populate_un_col1_present_groups(
    ms1_specs,
    m0_groups: list[dict[str, object]],
    rt_lo: float,
    rt_hi: float,
    m0_ppm_tol: float,
    integration_ch: dict[str, float] | None = None,
) -> set[int]:
    """m0_group indices with real M0 XIC signal co-eluting with the integrated peak."""
    out: set[int] = set()
    if not m0_groups or not np.isfinite(float(rt_lo)) or not np.isfinite(float(rt_hi)):
        return out
    rt_lo_f, rt_hi_f = float(rt_lo), float(rt_hi)
    int_apex = np.nan
    int_lo = int_hi = np.nan
    shoulder_slack = np.inf
    if integration_ch is not None:
        int_apex = float(integration_ch.get('apex_rt', np.nan))
        int_lo = float(integration_ch.get('rt_lo', np.nan))
        int_hi = float(integration_ch.get('rt_hi', np.nan))
        if np.isfinite(int_lo) and np.isfinite(int_hi) and int_hi > int_lo:
            shoulder_slack = max(
                float(UN_COL1_SHOULDER_APEX_SLACK_SEC),
                float(UN_COL1_SHOULDER_APEX_SLACK_INT_WIDTH_MULT) * (int_hi - int_lo),
            )
    peak_by_gi: dict[int, float] = {}
    apex_by_gi: dict[int, float] = {}
    for gi, grp in enumerate(m0_groups):
        mono_mz = float(grp['mono_mz'])
        rtx, ix = xic_ms1_ppm(ms1_specs, mono_mz, ppm_tol=m0_ppm_tol)
        if rtx.size == 0:
            continue
        apex_rt = float(rtx[int(np.argmax(ix))])
        if not (rt_lo_f <= apex_rt <= rt_hi_f):
            continue
        if integration_ch is not None and np.isfinite(int_apex):
            in_integration = (
                np.isfinite(int_lo) and np.isfinite(int_hi)
                and int_lo <= apex_rt <= int_hi
            )
            near_integration = abs(apex_rt - int_apex) <= shoulder_slack
            if not (in_integration or near_integration):
                continue
        else:
            in_integration = False
            near_integration = False
        m = (rtx >= rt_lo_f) & (rtx <= rt_hi_f)
        if not np.any(m):
            continue
        y = np.asarray(ix[m], dtype=float)
        if y.size == 0 or not np.any(y > 0):
            continue
        ymax = float(np.max(y))
        if ymax <= 0:
            continue
        nonzero_frac = float(np.mean(y >= (0.10 * ymax)))
        min_nz = (
            float(MIN_UN_COL1_SHOULDER_MIN_NONZERO_FRAC)
            if near_integration and not in_integration
            else float(MIN_UN_COL1_PRESENT_MIN_NONZERO_FRAC)
        )
        if nonzero_frac < min_nz:
            continue
        peak_by_gi[int(gi)] = ymax
        apex_by_gi[int(gi)] = apex_rt
    if not peak_by_gi:
        return out
    global_peak = max(peak_by_gi.values())
    threshold = max(
        float(MIN_UN_COL1_PRESENT_PEAK_FRAC) * global_peak,
        float(MIN_UN_COL1_PRESENT_ABS_INTENSITY),
    )
    for gi, peak in peak_by_gi.items():
        if peak >= threshold:
            out.add(int(gi))
    return out


def _populate_un_col3_present_groups(
    ms1_specs,
    m0_groups: list[dict[str, object]],
    charge: int,
    target_mz: float,
    integration_ch: dict[str, float] | None,
    base_ppm_tol: float,
) -> set[int]:
    """m0_group indices with meaningful isotope XIC sum in UN col3 integration window."""
    out: set[int] = set()
    if integration_ch is None or not m0_groups:
        return out
    rt_lo = float(integration_ch['rt_lo'])
    rt_hi = float(integration_ch['rt_hi'])
    peak_by_gi: dict[int, float] = {}
    for gi, grp in enumerate(m0_groups):
        mono_mz = float(grp['mono_mz'])
        rtx_m0, ix_m0 = xic_ms1_ppm(ms1_specs, mono_mz, ppm_tol=base_ppm_tol)
        if rtx_m0.size == 0:
            continue
        apex_rt = float(rtx_m0[int(np.argmax(ix_m0))])
        if not (rt_lo <= apex_rt <= rt_hi):
            continue
        iso_cfg = _isotope_xic_settings('UN', mono_mz, target_mz, base_ppm_tol, charge)
        ppm_tol = float(iso_cfg['ppm_tol'])
        shift_da = float(iso_cfg['shift_da'])
        rtx_i, _iso_traces, sum_trace = _peptide_isotope_xics_in_window(
            ms1_specs, mono_mz, charge, rt_lo, rt_hi, ppm_tol, center_shift_da=shift_da,
        )
        if rtx_i.size == 0 or not np.any(sum_trace > 0):
            continue
        ymax = float(np.max(sum_trace))
        if ymax <= 0:
            continue
        nonzero_frac = float(np.mean(sum_trace >= (0.10 * ymax)))
        if nonzero_frac < float(MIN_UN_COL3_PRESENT_MIN_NONZERO_FRAC):
            continue
        peak_by_gi[int(gi)] = ymax
    if not peak_by_gi:
        return out
    global_peak = max(peak_by_gi.values())
    threshold = max(
        float(MIN_UN_COL3_PRESENT_PEAK_FRAC) * global_peak,
        float(MIN_UN_COL3_PRESENT_ABS_INTENSITY),
    )
    for gi, peak in peak_by_gi.items():
        if peak >= threshold:
            out.add(int(gi))
    return out


def _m0_groups_for_col1_row(
    m0_groups: list[dict[str, object]],
    lbl: str,
    un_col1_group_indices: set[int] | None,
) -> list[dict[str, object]]:
    """UN col1: only registry groups with real M0 signal in the collection window."""
    if un_col1_group_indices is not None:
        return [grp for gi, grp in enumerate(m0_groups) if gi in un_col1_group_indices]
    if str(lbl).strip().upper() == 'UN':
        return m0_groups
    return []


def _m0_groups_for_col3_row(
    m0_groups: list[dict[str, object]],
    lbl: str,
    un_col3_group_indices: set[int] | None,
) -> list[dict[str, object]]:
    """All rows: only registry groups with meaningful UN col3 isotope signal."""
    if un_col3_group_indices is not None:
        return [grp for gi, grp in enumerate(m0_groups) if gi in un_col3_group_indices]
    if str(lbl).strip().upper() == 'UN':
        return m0_groups
    return []


def _plot_registry_isotope_xics(
    ax,
    ms1_specs: list,
    m0_groups: list[dict[str, object]],
    charge: int,
    target_mz: float,
    integration_ch: dict[str, float] | None,
    base_ppm_tol: float,
    lbl: str,
    un_col3_group_indices: set[int] | None = None,
) -> None:
    """Shared isotope XIC panel (M0–M+4 + sum) for all registry peptides in the integration window."""
    is_un = str(lbl).strip().upper() == 'UN'
    plot_groups = _m0_groups_for_col3_row(
        m0_groups, lbl, un_col3_group_indices,
    )
    if is_un:
        title_mode = f'UN theoretical ±{base_ppm_tol:g}ppm'
    else:
        title_mode = 'HDX shift-aware ±ppm (see legend)'
    ax.set_title(f'{lbl} | isotope XICs M0–M+4 [integration RT] | {title_mode}', fontsize=8)
    if integration_ch is None or not plot_groups:
        ax.text(
            0.5, 0.5, 'no registry / no integration window',
            transform=ax.transAxes, ha='center', va='center', color='gray', fontsize=8,
        )
        return

    rt_lo = float(integration_ch['rt_lo'])
    rt_hi = float(integration_ch['rt_hi'])
    pad = max(1.0, (rt_hi - rt_lo) * 0.05)
    ax.set_xlim(rt_lo - pad, rt_hi + pad)
    _shade_integration_window(ax, rt_lo, rt_hi, zorder=0.2)
    ax.axvline(float(integration_ch['apex_rt']), color='red', ls='--', lw=0.9, alpha=0.85, zorder=1)

    legend_handles: list[Line2D] = []
    ymax = 0.0
    any_data = False
    n_groups = len(plot_groups)
    multi_pep = n_groups > 1
    for gi, grp in enumerate(plot_groups):
        orig_gi = m0_groups.index(grp) if grp in m0_groups else gi
        mono_mz = float(grp["mono_mz"])
        pep_labels = [f"r{int(r['rank'])}:{r['peptide_sequence']}" for r in grp["peptides"]]
        pep_tag = " / ".join(pep_labels)
        grp_clr = _peptide_group_color(orig_gi)
        grp_ls = _peptide_group_linestyle(orig_gi)
        sum_ls = PEPTIDE_SUM_LSTYLES[orig_gi % len(PEPTIDE_SUM_LSTYLES)]
        sum_lw = PEPTIDE_SUM_LWS[orig_gi % len(PEPTIDE_SUM_LWS)]
        sum_alpha = PEPTIDE_SUM_ALPHAS[orig_gi % len(PEPTIDE_SUM_ALPHAS)] if multi_pep else 0.95
        iso_base_lw = PEPTIDE_ISO_LWS[orig_gi % len(PEPTIDE_ISO_LWS)]
        iso_cfg = _isotope_xic_settings(lbl, mono_mz, target_mz, base_ppm_tol, charge)
        ppm_tol = float(iso_cfg['ppm_tol'])
        shift_da = float(iso_cfg['shift_da'])
        mode_label = str(iso_cfg['mode_label'])
        rtx_i, iso_traces, sum_trace = _peptide_isotope_xics_in_window(
            ms1_specs, mono_mz, charge, rt_lo, rt_hi, ppm_tol, center_shift_da=shift_da,
        )
        if rtx_i.size == 0 or not np.any(sum_trace > 0):
            continue
        any_data = True
        for k, iso_lbl in enumerate(ISO_XIC_LABELS):
            y = iso_traces[k]
            if not np.any(y > 0):
                continue
            trace_clr = _peptide_isotope_trace_color(orig_gi, k)
            trace_ls = grp_ls
            trace_lw = (iso_base_lw + (0.08 * k)) if multi_pep else (1.0 + (0.12 * k))
            trace_alpha = 0.92 if multi_pep else 0.90
            ax.plot(
                rtx_i, y, color=trace_clr, lw=trace_lw, ls=trace_ls,
                alpha=trace_alpha, zorder=2 + orig_gi,
            )
            ymax = max(ymax, float(y.max()))
        ax.plot(
            rtx_i, sum_trace, color=grp_clr, lw=sum_lw, ls=sum_ls,
            alpha=sum_alpha, zorder=10 + orig_gi,
        )
        ymax = max(ymax, float(sum_trace.max()))
        legend_handles.append(
            Line2D(
                [0], [0], color=grp_clr, lw=sum_lw, ls=sum_ls, alpha=sum_alpha,
                label=f'{pep_tag} sum m0={mono_mz:.4f} ({mode_label})',
            )
        )

    if not any_data:
        ax.text(
            0.5, 0.5, 'no peptide isotope signal in window',
            transform=ax.transAxes, ha='center', va='center', color='gray', fontsize=8,
        )
        return
    if ymax > 0:
        ax.set_ylim(0, ymax * 1.22)
    if legend_handles:
        ax.legend(handles=legend_handles, fontsize=4.6, loc='upper right', framealpha=0.85)


def _trace_peak_within_window(
    rts: np.ndarray,
    ints: np.ndarray,
    rt_lo: float,
    rt_hi: float,
    expected_apex_rt: float | None = None,
) -> dict[str, float] | None:
    """Find a trace peak using the same candidate formula as the integration window."""
    if rts.size == 0 or ints.size == 0:
        return None
    cands = detect_candidates(rts, ints, float(rt_lo), float(rt_hi))
    if not cands:
        return None
    m_win = (rts >= float(rt_lo)) & (rts <= float(rt_hi))
    if not np.any(m_win):
        return None
    total_area = float(np.asarray(ints[m_win], dtype=float).sum())
    if total_area <= 0:
        return None
    best = None
    best_score = -1e18
    for cc in cands:
        cm = (rts >= float(cc["rt_lo"])) & (rts <= float(cc["rt_hi"]))
        area = float(np.asarray(ints[cm], dtype=float).sum()) if np.any(cm) else 0.0
        height = float(cc.get("height", 0.0))
        score = area + 0.20 * height
        if expected_apex_rt is not None:
            score -= 0.02 * abs(float(cc["apex_rt"]) - float(expected_apex_rt)) * max(height, 1.0)
        if score > best_score:
            best_score = score
            best = (cc, area / max(total_area, 1e-12))
    if best is None:
        return None
    c, area_frac = best
    return {
        "apex_rt": float(c["apex_rt"]),
        "rt_lo": float(c["rt_lo"]),
        "rt_hi": float(c["rt_hi"]),
        "height": float(c["height"]),
        "area_frac": float(area_frac),
    }


def _trace_similarity_in_window(
    rtx_a: np.ndarray,
    ix_a: np.ndarray,
    rtx_b: np.ndarray,
    ix_b: np.ndarray,
    rt_lo: float,
    rt_hi: float,
) -> float:
    """Correlation of two traces restricted to the same RT window."""
    if rtx_a.size == 0 or rtx_b.size == 0:
        return np.nan
    m = (rtx_a >= float(rt_lo)) & (rtx_a <= float(rt_hi))
    if np.sum(m) < 4:
        return np.nan
    x = np.asarray(ix_a[m], dtype=float)
    y = np.interp(np.asarray(rtx_a[m], dtype=float), rtx_b, ix_b, left=0.0, right=0.0)
    if not np.any(x > 0) or not np.any(y > 0):
        return np.nan
    x = x / max(float(np.max(x)), 1.0)
    y = y / max(float(np.max(y)), 1.0)
    if np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def _m0_trace_ok_for_refinement(
    rtx_m0: np.ndarray,
    ix_m0: np.ndarray,
    rtx_iso: np.ndarray,
    ix_iso: np.ndarray,
    rt_lo: float,
    rt_hi: float,
    min_rel_to_iso_peak: float = 0.05,
    min_nonzero_frac: float = 0.80,
) -> bool:
    """Skip M0-based re-integration when the peptide trace is too weak or sparse."""
    m0 = (rtx_m0 >= float(rt_lo)) & (rtx_m0 <= float(rt_hi))
    iso = (rtx_iso >= float(rt_lo)) & (rtx_iso <= float(rt_hi))
    if not np.any(m0) or not np.any(iso):
        return False
    y = np.asarray(ix_m0[m0], dtype=float)
    iso_y = np.asarray(ix_iso[iso], dtype=float)
    ymax = float(np.max(y)) if y.size else 0.0
    iso_max = float(np.max(iso_y)) if iso_y.size else 0.0
    if ymax <= 0.0 or iso_max <= 0.0:
        return False
    rel_peak = ymax / iso_max
    nonzero_frac = float(np.mean(y > 0.0)) if y.size else 0.0
    return rel_peak >= float(min_rel_to_iso_peak) and nonzero_frac >= float(min_nonzero_frac)


def _m0_refinement_peak_aligned(
    m0_pk: dict[str, float],
    iso_pk: dict[str, float],
    base_ch: dict[str, float],
    max_apex_delta_sec: float | None = None,
) -> bool:
    """Require the M0-derived peak to co-elute with the iso-window envelope peak."""
    if m0_pk is None or iso_pk is None or base_ch is None:
        return False
    window_w = float(base_ch["rt_hi"]) - float(base_ch["rt_lo"])
    if window_w <= 0:
        return False
    tol = float(max_apex_delta_sec) if max_apex_delta_sec is not None else max(4.0, 0.25 * window_w)
    return abs(float(m0_pk["apex_rt"]) - float(iso_pk["apex_rt"])) <= tol


def _shoulder_profile_from_refinement(
    original_ch: dict[str, float],
    refined_ch: dict[str, float],
) -> dict[str, float] | None:
    """Capture apex-relative integration bounds learned from UN M0 refinement."""
    old_w = float(original_ch["rt_hi"]) - float(original_ch["rt_lo"])
    new_w = float(refined_ch["rt_hi"]) - float(refined_ch["rt_lo"])
    if old_w <= 0 or new_w <= 0 or new_w >= (old_w * 0.85):
        return None
    apex = float(refined_ch["apex_rt"])
    return {
        "lo_from_apex": float(apex - float(refined_ch["rt_lo"])),
        "hi_from_apex": float(float(refined_ch["rt_hi"]) - apex),
        "refined_width": float(new_w),
    }


def _resolve_timecourse_integration_ch(
    lbl: str,
    ch: dict[str, float] | None,
    ms1_specs,
    mz: float,
    iso_lo: float,
    iso_hi: float,
    m0_groups: list,
    m0_ppm_tol: float,
    un_shoulder_profile: dict[str, float] | None,
) -> tuple[
    dict[str, float] | None,
    dict[str, float] | None,
    str,
    dict[str, float] | None,
]:
    """Resolve MS2-tracked integration window with optional UN M0 refinement."""
    integration_ch = ch
    ch_refine_note = ''
    if ch is None:
        return None, None, ch_refine_note, un_shoulder_profile

    if lbl == 'UN' and m0_groups:
        below_target = [g for g in m0_groups if float(g.get("mono_mz", np.inf)) < float(mz)]
        if below_target:
            grp_primary = min(
                below_target,
                key=lambda g: (
                    int(g.get("primary_rank", 999999)),
                    abs(float(g.get("mono_mz", np.inf)) - float(mz)),
                ),
            )
        else:
            grp_primary = min(m0_groups, key=lambda g: int(g.get("primary_rank", 999999)))
        mono_primary = float(grp_primary["mono_mz"])
        rtx, ix = xic_ms1_isowin(ms1_specs, mz, iso_lo, iso_hi)
        rtx_m0p, ix_m0p = xic_ms1_ppm(ms1_specs, mono_primary, ppm_tol=m0_ppm_tol)
        m0_trace_ok = _m0_trace_ok_for_refinement(
            rtx_m0p, ix_m0p, rtx, ix,
            float(ch['rt_lo']), float(ch['rt_hi']),
        )
        iso_pk = _trace_peak_within_window(
            rtx, ix,
            float(ch['rt_lo']), float(ch['rt_hi']),
            expected_apex_rt=float(ch['apex_rt']),
        ) if m0_trace_ok else None
        pk = _trace_peak_within_window(
            rtx_m0p, ix_m0p,
            float(ch['rt_lo']), float(ch['rt_hi']),
            expected_apex_rt=float(iso_pk['apex_rt']) if iso_pk is not None else float(ch['apex_rt']),
        ) if m0_trace_ok and iso_pk is not None else None
        if pk is not None and not _m0_refinement_peak_aligned(pk, iso_pk, ch):
            pk = None
        if pk is not None:
            old_w = float(ch['rt_hi']) - float(ch['rt_lo'])
            new_w = float(pk['rt_hi']) - float(pk['rt_lo'])
            sim = _trace_similarity_in_window(
                rtx, ix, rtx_m0p, ix_m0p,
                float(ch['rt_lo']), float(ch['rt_hi']),
            )
            area_frac = float(pk.get("area_frac", 0.0))
            if (
                new_w > 0
                and new_w < (old_w * 0.85)
                and area_frac >= 0.35
                and (not np.isfinite(sim) or sim < 0.985)
            ):
                integration_ch = pk
                ch_refine_note = (
                    f"refined by low-m/z M0={mono_primary:.4f} "
                    f"({old_w:.1f}s -> {new_w:.1f}s, area={area_frac:.2f})"
                )
                un_shoulder_profile = _shoulder_profile_from_refinement(ch, integration_ch)
    elif un_shoulder_profile is not None:
        trimmed = _apply_un_shoulder_profile(ch, un_shoulder_profile)
        if trimmed is not None:
            integration_ch = trimmed
            ch_refine_note = (
                f"UN shoulder trim applied "
                f"({float(un_shoulder_profile['refined_width']):.1f}s profile)"
            )

    return integration_ch, ch, ch_refine_note, un_shoulder_profile


def _build_ms1_dabs_cache_for_window(
    ms1_fit_rows: list[dict[str, object]],
    m0_groups: list,
    mz: float,
    z: int,
    first_iso_lo: float,
    first_iso_hi: float,
    un_col3_group_indices: set[int] | None = None,
) -> dict[str, dict[str, float]]:
    """Run/cache MS1 precursor HDX fits before opening the matplotlib figure."""
    ms1_dabs_cache: dict[str, dict[str, float]] = {}
    if not ms1_fit_rows or not m0_groups:
        return ms1_dabs_cache
    plot_groups = _m0_groups_for_col3_row(m0_groups, 'UN', un_col3_group_indices)
    if not plot_groups:
        return ms1_dabs_cache
    ms1_mz_lo = float(mz) - first_iso_lo
    ms1_mz_hi = float(mz) + first_iso_hi
    seen_peps: set[str] = set()
    for grp in plot_groups:
        mono_mz = float(grp['mono_mz'])
        for reg in grp.get('peptides', []) or []:
            pep = str(reg.get('peptide_sequence', '') or '').strip().upper()
            if not pep or pep in seen_peps:
                continue
            seen_peps.add(pep)
            dabs_map = run_cached_precursor_ms1_hdx_fit(
                ms1_fit_rows, mz, z, pep, mono_mz,
                ms1_mz_lo, ms1_mz_hi,
                fits_root=DEFAULT_FRAGMENT_FITS_ROOT,
            )
            if dabs_map:
                ms1_dabs_cache[pep] = dabs_map
    return ms1_dabs_cache


def _ms1_dabs_annotation_lines(
    lbl: str,
    m0_groups: list,
    ms1_dabs_cache: dict[str, dict[str, float]],
    un_col3_group_indices: set[int] | None = None,
) -> list[str]:
    ann_lines: list[str] = []
    plot_groups = _m0_groups_for_col3_row(m0_groups, lbl, un_col3_group_indices)
    for grp in plot_groups:
        for reg in grp.get('peptides', []) or []:
            pep = str(reg.get('peptide_sequence', '') or '').strip().upper()
            dabs_val = ms1_dabs_cache.get(pep, {}).get(lbl)
            if dabs_val is not None and np.isfinite(float(dabs_val)):
                ann_lines.append(
                    f"r{int(reg.get('rank', 0))} MS1 D={float(dabs_val):.2f}"
                )
    return ann_lines


def _build_timecourse_excel_sheets(
    rows_out: list[tuple],
    row_integration_plans: list[dict[str, object]],
    m0_groups: list[dict[str, object]],
    target_mz: float,
    precursor_charge: int,
    m0_ppm_tol: float,
    comet_cfg: dict,
    theo_frag_cache: dict[tuple[str, int], list],
    frag_ppm_tol: float,
    un_frag_label_keys: dict[int, set[tuple[str, str]]],
    ms1_dabs_cache: dict[str, dict[str, float]],
    un_col3_group_indices: set[int] | None = None,
    un_col1_group_indices: set[int] | None = None,
) -> dict[str, pd.DataFrame]:
    """Export data shown in the 6-column timecourse figure (non-zoom plot)."""
    integration_rows: list[dict[str, object]] = []
    col1_rows: list[dict[str, object]] = []
    col2_rows: list[dict[str, object]] = []
    col3_rows: list[dict[str, object]] = []
    col4_rows: list[dict[str, object]] = []
    col5_rows: list[dict[str, object]] = []
    col6_rows: list[dict[str, object]] = []
    ms1_dabs_rows: list[dict[str, object]] = []

    for i, (lbl, fname, run, s_lo, s_hi, iso_lo, iso_hi) in enumerate(rows_out):
        timepoint = str(lbl)
        ms1_specs = run['ms1_specs']
        ms_by = run['ms_by']
        plan = row_integration_plans[i]
        integration_ch = plan['integration_ch']
        ch = plan['ch']

        if integration_ch is not None:
            integration_rows.append({
                'timepoint': timepoint,
                'file_name': str(fname),
                'integration_rt_lo_s': float(integration_ch['rt_lo']),
                'integration_rt_hi_s': float(integration_ch['rt_hi']),
                'integration_apex_rt_s': float(integration_ch['apex_rt']),
                'tracked_rt_lo_s': float(ch['rt_lo']) if ch is not None else np.nan,
                'tracked_rt_hi_s': float(ch['rt_hi']) if ch is not None else np.nan,
                'schedule_rt_lo_s': float(s_lo) if s_lo is not None else np.nan,
                'schedule_rt_hi_s': float(s_hi) if s_hi is not None else np.nan,
                'refinement_note': str(plan.get('ch_refine_note', '') or ''),
            })

        rtx, ix = xic_ms1_isowin(ms1_specs, target_mz, iso_lo, iso_hi)
        for rt, inten in zip(rtx, ix):
            col1_rows.append({
                'timepoint': timepoint,
                'trace_type': 'target_isolation_window',
                'rt_s': float(rt),
                'intensity': float(inten),
                'isolation_da_lo': float(iso_lo),
                'isolation_da_hi': float(iso_hi),
            })
        if timepoint.strip().upper() == 'UN':
            col1_groups = _m0_groups_for_col1_row(m0_groups, timepoint, un_col1_group_indices)
            for gi, grp in enumerate(m0_groups):
                if grp not in col1_groups:
                    continue
                mono_mz = float(grp['mono_mz'])
                for reg in grp.get('peptides', []) or []:
                    pep = str(reg.get('peptide_sequence', '') or '').strip()
                    rank = int(reg.get('rank', 999999) or 999999)
                    rtx_m0, ix_m0 = xic_ms1_ppm(ms1_specs, mono_mz, ppm_tol=m0_ppm_tol)
                    for rt, inten in zip(rtx_m0, ix_m0):
                        col1_rows.append({
                            'timepoint': timepoint,
                            'trace_type': 'registry_peptide_m0_5ppm',
                            'rt_s': float(rt),
                            'intensity': float(inten),
                            'peptide_sequence': pep,
                            'registry_rank': rank,
                            'mono_mz': mono_mz,
                            'ppm_tol': float(m0_ppm_tol),
                            'group_index': int(gi),
                        })

        rtf, tf = ms_frag_chrom(ms_by, target_mz)
        for rt, inten in zip(rtf, tf):
            col2_rows.append({
                'timepoint': timepoint,
                'rt_s': float(rt),
                'ms2_fragment_sum_intensity': float(inten),
                'integration_rt_lo_s': float(integration_ch['rt_lo']) if integration_ch else np.nan,
                'integration_rt_hi_s': float(integration_ch['rt_hi']) if integration_ch else np.nan,
                'integration_apex_rt_s': float(integration_ch['apex_rt']) if integration_ch else np.nan,
            })

        if integration_ch is not None and m0_groups:
            rt_lo = float(integration_ch['rt_lo'])
            rt_hi = float(integration_ch['rt_hi'])
            col3_groups = _m0_groups_for_col3_row(
                m0_groups, timepoint, un_col3_group_indices,
            )
            for gi, grp in enumerate(col3_groups):
                orig_gi = m0_groups.index(grp) if grp in m0_groups else gi
                mono_mz = float(grp['mono_mz'])
                pep_labels = [f"r{int(r['rank'])}:{r['peptide_sequence']}" for r in grp['peptides']]
                iso_cfg = _isotope_xic_settings(timepoint, mono_mz, target_mz, m0_ppm_tol, precursor_charge)
                ppm_tol = float(iso_cfg['ppm_tol'])
                shift_da = float(iso_cfg['shift_da'])
                rtx_i, iso_traces, sum_trace = _peptide_isotope_xics_in_window(
                    ms1_specs, mono_mz, precursor_charge, rt_lo, rt_hi, ppm_tol,
                    center_shift_da=shift_da,
                )
                if rtx_i.size == 0:
                    continue
                for k, iso_lbl in enumerate(ISO_XIC_LABELS):
                    y = iso_traces[k]
                    for rt, inten in zip(rtx_i, y):
                        col3_rows.append({
                            'timepoint': timepoint,
                            'peptide_group': ' / '.join(pep_labels),
                            'registry_rank': int(grp['primary_rank']),
                            'mono_mz': mono_mz,
                            'isotope': iso_lbl,
                            'rt_s': float(rt),
                            'intensity': float(inten),
                            'ppm_tol': ppm_tol,
                            'hdx_shift_da': shift_da,
                            'group_index': int(orig_gi),
                        })
                for rt, inten in zip(rtx_i, sum_trace):
                    col3_rows.append({
                        'timepoint': timepoint,
                        'peptide_group': ' / '.join(pep_labels),
                        'registry_rank': int(grp['primary_rank']),
                        'mono_mz': mono_mz,
                        'isotope': 'sum',
                        'rt_s': float(rt),
                        'intensity': float(inten),
                        'ppm_tol': ppm_tol,
                        'hdx_shift_da': shift_da,
                        'group_index': int(orig_gi),
                    })

        mz_lo, mz_hi = float(target_mz) - float(iso_lo), float(target_mz) + float(iso_hi)
        if integration_ch is not None:
            rt_lo = float(integration_ch['rt_lo'])
            rt_hi = float(integration_ch['rt_hi'])
            _n1, mc, hc = integrate_ms1(ms1_specs, rt_lo, rt_hi, mz_lo, mz_hi)
            is_un_row = timepoint.strip().upper() == 'UN'
            col4_groups = _m0_groups_for_col3_row(
                m0_groups, timepoint, un_col3_group_indices,
            )
            for mz_p, inten in zip(mc, hc):
                if inten <= 0:
                    continue
                col4_rows.append({
                    'timepoint': timepoint,
                    'peak_type': 'binned_spectrum',
                    'mz': float(mz_p),
                    'intensity': float(inten),
                    'integration_rt_lo_s': rt_lo,
                    'integration_rt_hi_s': rt_hi,
                    'mz_lo': mz_lo,
                    'mz_hi': mz_hi,
                })
            for gi, grp in enumerate(col4_groups):
                orig_gi = m0_groups.index(grp) if grp in m0_groups else gi
                mono_mz = float(grp['mono_mz'])
                iso_cfg = _isotope_xic_settings(timepoint, mono_mz, target_mz, m0_ppm_tol, precursor_charge)
                ppm_tol = float(iso_cfg['ppm_tol'])
                pep_seq = str(grp['peptides'][0]['peptide_sequence'] if grp.get('peptides') else '')
                n_ex = _exchangeable_amide_count(pep_seq)
                max_deut = 0 if is_un_row else n_ex
                for pk in _ms1_precursor_deuteration_peak_matches(
                    mc, hc, pep_seq, mono_mz, precursor_charge, ppm_tol,
                    target_mz=target_mz, lbl=timepoint,
                ):
                    col4_rows.append({
                        'timepoint': timepoint,
                        'peak_type': 'matched_isotope',
                        'peptide_sequence': pep_seq,
                        'registry_rank': int(grp['peptides'][0].get('rank', 999999)),
                        'mono_mz': mono_mz,
                        'mz': float(pk['obs_mz']),
                        'intensity': float(pk['obs_int']),
                        'deut_level': float(pk.get('d', np.nan)),
                        'isotope_index': float(pk.get('k', np.nan)),
                        'group_index': int(orig_gi),
                    })
            for grp in col4_groups:
                for reg in grp.get('peptides', []) or []:
                    pep = str(reg.get('peptide_sequence', '') or '').strip().upper()
                    dabs_val = ms1_dabs_cache.get(pep, {}).get(timepoint)
                    if dabs_val is not None and np.isfinite(float(dabs_val)):
                        ms1_dabs_rows.append({
                            'timepoint': timepoint,
                            'peptide_sequence': pep,
                            'registry_rank': int(reg.get('rank', 999999) or 999999),
                            'mono_mz': float(grp['mono_mz']),
                            'ms1_dabs': float(dabs_val),
                        })

            for ms_level, col_rows in ((2, col5_rows), (3, col6_rows)):
                _nn, spec_mz, spec_i = integrate_msx(
                    ms_by, target_mz, ms_level, rt_lo, rt_hi,
                    100.0, 2000.0, bin_size=FRAG_SPECTRUM_PLOT_BIN_SIZE,
                )
                _, raw_m, raw_i = _collect_msx_raw_peaks(
                    ms_by, target_mz, ms_level, rt_lo, rt_hi, 100, 2000,
                )
                match_m, match_i = _sum_peaks_by_exact_mz(raw_m, raw_i)
                frag_matches = (
                    _collect_col3_fragment_matches(
                        match_m, match_i, m0_groups, precursor_charge,
                        comet_cfg, theo_frag_cache, frag_ppm_tol,
                        un_col3_group_indices=un_col3_group_indices,
                    )
                    if m0_groups and match_m.size else []
                )
                if is_un_row:
                    label_matches = frag_matches
                else:
                    label_matches = _filter_fragments_to_un_control(
                        frag_matches, un_frag_label_keys[ms_level],
                    )
                for mz_p, inten in zip(spec_mz, spec_i):
                    if inten <= 0:
                        continue
                    col_rows.append({
                        'timepoint': timepoint,
                        'peak_type': 'binned_spectrum',
                        'ms_level': int(ms_level),
                        'mz': float(mz_p),
                        'intensity': float(inten),
                        'integration_rt_lo_s': rt_lo,
                        'integration_rt_hi_s': rt_hi,
                    })
                for hit in label_matches:
                    col_rows.append({
                        'timepoint': timepoint,
                        'peak_type': 'matched_fragment',
                        'ms_level': int(ms_level),
                        'peptide_sequence': str(hit.get('peptide', '') or ''),
                        'fragment_ion': str(hit.get('ion', '') or ''),
                        'mz': float(hit['obs_mz']),
                        'intensity': float(hit['obs_int']),
                        'charge': int(hit.get('charge', 0) or 0),
                    })

    return {
        'tc_integration': pd.DataFrame(integration_rows),
        'tc_col1_ms1_xic': pd.DataFrame(col1_rows),
        'tc_col2_ms2_sum': pd.DataFrame(col2_rows),
        'tc_col3_isotope_xic': pd.DataFrame(col3_rows),
        'tc_col4_ms1': pd.DataFrame(col4_rows),
        'tc_col5_ms2': pd.DataFrame(col5_rows),
        'tc_col6_ms3': pd.DataFrame(col6_rows),
        'tc_ms1_dabs': pd.DataFrame(ms1_dabs_rows),
    }


def _write_excel_workbook(path: str, sheets: dict[str, pd.DataFrame]) -> None:
    with pd.ExcelWriter(path, engine='openpyxl') as writer:
        for sheet_name, df in sheets.items():
            safe_name = str(sheet_name)[:31]
            (df if not df.empty else pd.DataFrame({'note': ['no data']})).to_excel(
                writer, index=False, sheet_name=safe_name,
            )


def _save_timecourse_excel_companion(
    png_path: str,
    sheets: dict[str, pd.DataFrame],
    *,
    target_mz: float,
    precursor_charge: int,
) -> str | None:
    xlsx_path = f'{os.path.splitext(png_path)[0]}.xlsx'
    summary = pd.DataFrame([{
        'target_mz': float(target_mz),
        'precursor_charge': int(precursor_charge),
        'plot_type': 'timecourse_6col',
        'n_timepoints': int(len(sheets.get('tc_integration', pd.DataFrame()))),
    }])
    all_sheets = {'summary': summary, **sheets}
    _write_excel_workbook(xlsx_path, all_sheets)
    print(f'Saved {xlsx_path}  (timecourse sheets: {len(sheets)})')
    return xlsx_path


def _apply_un_shoulder_profile(
    ch: dict[str, float],
    profile: dict[str, float],
) -> dict[str, float] | None:
    """Apply UN-learned shoulder trim to a tracked peak in a later exposure row."""
    if ch is None or profile is None:
        return None
    apex = float(ch["apex_rt"])
    rt_lo = apex - float(profile["lo_from_apex"])
    rt_hi = apex + float(profile["hi_from_apex"])
    rt_lo = max(rt_lo, float(ch["rt_lo"]))
    rt_hi = min(rt_hi, float(ch["rt_hi"]))
    if rt_hi <= rt_lo:
        return None
    old_w = float(ch["rt_hi"]) - float(ch["rt_lo"])
    new_w = float(rt_hi - rt_lo)
    if new_w <= 0 or new_w >= (old_w * 0.85):
        return None
    return {
        "apex_rt": apex,
        "rt_lo": float(rt_lo),
        "rt_hi": float(rt_hi),
        "height": float(ch.get("height", 0.0)),
    }


def main():
    ap = argparse.ArgumentParser(description="Render per-peptide HDX timecourse diagnostics with optional registry annotation.")
    ap.add_argument("--type1", required=True, help="Type1 HDX exposure matrix CSV (all timepoints).")
    ap.add_argument("--type5", required=True, help="Type5 UN target matrix CSV (isolation windows).")
    ap.add_argument("--mzml-dir", default=DEFAULT_MZML_DIR, help="Directory containing mzML files for row labels.")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Output directory for per-peptide PNGs.")
    ap.add_argument("--registry", default=DEFAULT_REGISTRY, help="Optional co-isolation registry CSV from step 1.")
    ap.add_argument(
        "--hdx-manifest",
        default=DEFAULT_HDX_MANIFEST,
        help="HDX fit input manifest CSV; peptides listed here are always kept in zoom figures.",
    )
    ap.add_argument(
        "--m0-ppm-tol",
        type=float,
        default=DEFAULT_M0_PPM_TOL,
        help="ppm tolerance for registry peptide M0 MS1 XIC overlays (default 5).",
    )
    ap.add_argument(
        "--comet-params",
        default=DEFAULT_COMET_PARAMS,
        help="Comet params file for theoretical fragment prediction (default ../comet.params.new).",
    )
    ap.add_argument(
        "--frag-ppm-tol",
        type=float,
        default=DEFAULT_FRAG_PPM_TOL,
        help="ppm tolerance for MS2/MS3 fragment matching on col3 peptides (default 5).",
    )
    ap.add_argument("--target-mz", type=float, default=None, help="Optional: render only this target m/z.")
    ap.add_argument("--target-charge", type=int, default=None, help="Optional: render only this charge state.")
    ap.add_argument(
        "--zoom-rank-only",
        type=int,
        default=None,
        help="Optional: generate fragment zoom PNGs for this registry rank only (e.g. 1).",
    )
    ap.add_argument(
        "--skip-excel",
        action="store_true",
        help="Skip companion .xlsx export (PNG only).",
    )
    ap.add_argument(
        "--skip-zoom",
        action="store_true",
        help="Skip fragment zoom PNG generation (6-column timecourse only).",
    )
    ns = ap.parse_args()

    out_dir = str(ns.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    type1 = str(ns.type1)
    type5 = str(ns.type5)
    mzml_dir = str(ns.mzml_dir)
    if not os.path.isfile(type1):
        raise SystemExit(f"Type1 HDX exposures matrix not found: {type1}")
    if not os.path.isfile(type5):
        raise SystemExit(f"Type5 UN target matrix not found: {type5}")
    registry_by_window = _load_registry_by_window(str(ns.registry))
    hdx_fit_by_window = _load_hdx_fit_peptides_by_window(str(ns.hdx_manifest))
    m0_ppm_tol = float(ns.m0_ppm_tol)
    frag_ppm_tol = float(ns.frag_ppm_tol)
    comet_cfg = parse_comet_params(str(ns.comet_params) if ns.comet_params else None)

    row_labels, row_to_file = build_row_labels(type1, type5_csv=type5)

    df1 = pd.read_csv(type1)
    df5 = pd.read_csv(type5)
    df1['hdx_exposure_time'] = df1['hdx_exposure_time'].astype(str).str.strip()

    df5['_mz'] = df5['target_peptide_mz'].astype(float).round(3)
    df5['_z'] = df5['peptide_charge'].astype(int)
    df5['_lv'] = df5['ms_level'].astype(int)
    pep_rows = df5[['_mz', '_z']].drop_duplicates().sort_values(['_z', '_mz']).reset_index(drop=True)
    if ns.target_mz is not None:
        pep_rows = pep_rows[np.isclose(pep_rows['_mz'], float(ns.target_mz), rtol=0, atol=1e-3)]
    if ns.target_charge is not None:
        pep_rows = pep_rows[pep_rows['_z'] == int(ns.target_charge)]
    if pep_rows.empty:
        raise SystemExit('No targets matched the requested --target-mz / --target-charge filters.')

    act_un = {}
    for _, r in df5.iterrows():
        kk = (float(r['_mz']), int(r['_z']))
        act_un.setdefault(kk, {})[int(r['_lv'])] = str(r.get('activation_method', ''))

    df1['_mz'] = df1['target_peptide_mz'].astype(float).round(3)
    df1['_z'] = df1['peptide_charge'].astype(int)
    df1['_lv'] = df1['ms_level'].astype(int)
    act_exp = {}
    for _, r in df1.iterrows():
        kk = (str(r['hdx_exposure_time']).strip(), float(r['_mz']), int(r['_z']))
        act_exp.setdefault(kk, {})[int(r['_lv'])] = str(r.get('activation_method', ''))

    zoom_queue: list[dict[str, object]] = []
    for _, prow in pep_rows.iterrows():
        mz = float(prow['_mz'])
        z = int(prow['_z'])
        k = (mz, z)
        safe = f"mz{mz:.4f}".replace('.', 'p')
        out_path = f'{out_dir}/timecourse_{safe}_z{z}.png'
        zoom_path_glob = f'timecourse_{safe}_z{z}_zoom_r*.png'

        rows_out, cands_by, mets_by, is_un_by = collect_candidates_for_peptide(
            mz, z, row_labels, row_to_file, mzml_dir
        )
        if not rows_out:
            print(f'skip {mz} z{z}: no rows')
            continue
        path = select_with_continuity(cands_by, mets_by, is_un_by, mz)
        reg_rows = registry_by_window.get((round(mz, 3), int(z)), [])
        m0_groups = _build_m0_trace_groups(reg_rows, m0_ppm_tol)
        theo_frag_cache: dict[tuple[str, int], list] = {}

        n_rows = len(rows_out)
        first_iso_lo = float(rows_out[0][5]) if rows_out else 5.0
        first_iso_hi = float(rows_out[0][6]) if rows_out else 5.0

        row_integration_plans: list[dict[str, object]] = []
        un_shoulder_profile: dict[str, float] | None = None
        un_frag_label_keys: dict[int, set[tuple[str, str]]] = {2: set(), 3: set()}
        un_frag_zoom_specs: dict[int, list[dict[str, object]]] = {2: [], 3: []}
        un_col3_group_indices: set[int] | None = None
        un_col1_group_indices: set[int] | None = None
        zoom_rows: list[dict[str, object]] = []
        ms1_fit_rows: list[dict[str, object]] = []
        for i, (lbl, fname, run, s_lo, s_hi, iso_lo, iso_hi) in enumerate(rows_out):
            cands = cands_by[i]
            ch = cands[path[i]] if cands and path[i] < len(cands) and cands[path[i]] is not None else None
            integration_ch, ch, ch_refine_note, un_shoulder_profile = _resolve_timecourse_integration_ch(
                lbl, ch, run['ms1_specs'], mz, iso_lo, iso_hi,
                m0_groups, m0_ppm_tol, un_shoulder_profile,
            )
            row_integration_plans.append({
                'integration_ch': integration_ch,
                'ch': ch,
                'ch_refine_note': ch_refine_note,
            })
            if integration_ch is not None:
                rt_lo = float(integration_ch['rt_lo'])
                rt_hi = float(integration_ch['rt_hi'])
                ms1_fit_rows.append({
                    'lbl': lbl,
                    'ms1_specs': run['ms1_specs'],
                    'rt_lo': rt_lo,
                    'rt_hi': rt_hi,
                })
                if lbl == 'UN':
                    if s_lo is not None and s_hi is not None:
                        col1_pad = max(2.0, (float(s_hi) - float(s_lo)) * 0.05)
                        un_col1_group_indices = _populate_un_col1_present_groups(
                            run['ms1_specs'], m0_groups,
                            float(s_lo) - col1_pad, float(s_hi) + col1_pad,
                            m0_ppm_tol,
                            integration_ch=integration_ch,
                        )
                    un_col3_group_indices = _populate_un_col3_present_groups(
                        run['ms1_specs'], m0_groups, z, mz, integration_ch, m0_ppm_tol,
                    )
                    _populate_un_frag_zoom_specs(
                        run['ms_by'], mz, rt_lo, rt_hi,
                        m0_groups, z, comet_cfg, theo_frag_cache, frag_ppm_tol,
                        un_frag_zoom_specs, un_frag_label_keys,
                        un_col3_group_indices=un_col3_group_indices,
                    )
                zoom_rows.append({'lbl': lbl, 'ms_by': run['ms_by'], 'rt_lo': rt_lo, 'rt_hi': rt_hi})

        ms1_dabs_cache = _build_ms1_dabs_cache_for_window(
            ms1_fit_rows, m0_groups, mz, z, first_iso_lo, first_iso_hi,
            un_col3_group_indices=un_col3_group_indices,
        )
        timecourse_sheets = _build_timecourse_excel_sheets(
            rows_out, row_integration_plans, m0_groups, mz, z, m0_ppm_tol,
            comet_cfg, theo_frag_cache, frag_ppm_tol, un_frag_label_keys,
            ms1_dabs_cache, un_col3_group_indices=un_col3_group_indices,
            un_col1_group_indices=un_col1_group_indices,
        )

        tc_cell = _timecourse_subplot_cell_in(n_rows)
        tc_cell_w = tc_cell * TIMECOURSE_SUBPLOT_WIDTH_MULT
        tc_spacing = _gridspec_spacing(n_rows, TIMECOURSE_N_COLS)
        fig, axes = plt.subplots(
            n_rows, TIMECOURSE_N_COLS,
            figsize=(TIMECOURSE_N_COLS * tc_cell_w + 1.0, n_rows * tc_cell + 1.2),
            gridspec_kw=tc_spacing,
        )
        if n_rows == 1:
            axes = np.array([axes])
        fig.suptitle(
            f'Timecourse per peptide: mz{mz:.4f} z{z} | continuity-tracked peak branch (no hopping)\n'
            f'rows: UN → increasing HDX exposure | col1 MS1 XIC target iso[-{first_iso_lo:.1f},+{first_iso_hi:.1f}] Da + registry M0 ±{m0_ppm_tol:g}ppm (UN row only; real-signal peptides in collection window) | '
            f'col2 MS2 frag-sum + selected integration (rel_h={REL_H}) | col3 registry isotope XICs M0–M+4 + sum [integration RT; UN ±{m0_ppm_tol:g}ppm, HDX shift-aware ±ppm; only UN-present peptides in integration window (≥{MIN_UN_COL3_PRESENT_PEAK_FRAC:.0%} peak)] | '
            f'col4 integrated MS1 + d×iso colored stems + MS1 Dabs | col5/6 MS2/MS3 fragments (±{frag_ppm_tol:g}ppm, b/y rejected, ≥{MIN_FRAG_ISOTOPE_PEAKS} isotopes; HDX labels/stems UN-controlled) | '
            f'fragment zoom: one PNG per registry precursor ({zoom_path_glob}, MS2+MS3 overlay per fragment)',
            fontsize=10, y=0.998,
        )
        col3_m0_groups = _m0_groups_for_col3_row(m0_groups, 'UN', un_col3_group_indices)
        legend_top = _add_registry_peptide_figure_legend(
            fig, m0_groups, group_indices=un_col3_group_indices, y_anchor=0.965, fontsize=7.0,
        )

        act2_label = act_un.get(k, {}).get(2, 'ETD?')
        act3_label = act_un.get(k, {}).get(3, 'scram?')
        for i, (lbl, fname, run, s_lo, s_hi, iso_lo, iso_hi) in enumerate(rows_out):
            ms1_specs = run['ms1_specs']
            ms_by = run['ms_by']
            iso_by = run['iso_by']
            plan = row_integration_plans[i]
            integration_ch = plan['integration_ch']
            ch = plan['ch']
            ch_refine_note = str(plan['ch_refine_note'])

            rtx, ix = xic_ms1_isowin(ms1_specs, mz, iso_lo, iso_hi)
            rtf, tf = ms_frag_chrom(ms_by, mz)

            if lbl == 'UN':
                act2 = act_un.get(k, {}).get(2, 'ETD?')
                act3 = act_un.get(k, {}).get(3, 'scram?')
            else:
                act2 = act_exp.get((lbl, mz, z), {}).get(2, 'ETD?')
                act3 = act_exp.get((lbl, mz, z), {}).get(3, 'scram?')

            a1, a2, a3, a4, a5, a6 = (
                axes[i, 0], axes[i, 1], axes[i, 2], axes[i, 3],
                axes[i, 4], axes[i, 5],
            )
            for a in (a1, a2, a3):
                if s_lo is not None:
                    a.axvspan(
                        s_lo, s_hi,
                        facecolor=UI_COLLECTION_FILL, alpha=0.72, zorder=0,
                        linewidth=0.8, edgecolor=UI_COLLECTION_BORDER,
                    )
                a.grid(alpha=0.25, lw=0.5)
                a.tick_params(labelsize=8)
            for a in (a4, a5, a6):
                a.grid(alpha=0.25, lw=0.5)
                a.tick_params(labelsize=8)

            if s_lo is not None:
                pad = max(2.0, (s_hi - s_lo) * 0.05)
                zlo, zhi = s_lo - pad, s_hi + pad
            else:
                zlo = zhi = None

            show_m0 = (lbl == 'UN') and bool(m0_groups)
            col1_plot_groups = _m0_groups_for_col1_row(m0_groups, lbl, un_col1_group_indices)
            a1_m0 = a1.twinx() if show_m0 and col1_plot_groups else None
            if show_m0 and a1_m0 is not None:
                a1_m0.set_zorder(5)
                a1.set_zorder(2)
                a1.patch.set_visible(False)
                for gi, grp in enumerate(m0_groups):
                    if grp not in col1_plot_groups:
                        continue
                    mono_mz = float(grp["mono_mz"])
                    clr = _m0_trace_color(gi)
                    ls = _peptide_group_linestyle(gi)
                    rtx_m0, ix_m0 = xic_ms1_ppm(ms1_specs, mono_mz, ppm_tol=m0_ppm_tol)
                    a1_m0.plot(
                        rtx_m0, ix_m0, color=clr, lw=M0_TRACE_LW_UN, ls=ls,
                        alpha=1.0, zorder=5, solid_capstyle='round', label='_nolegend_',
                    )
                a1_m0.set_ylabel(f'M0 intensity (±{m0_ppm_tol:g}ppm)', fontsize=7, color='black')
                a1_m0.tick_params(axis='y', labelcolor='black', labelsize=7)

            a1.plot(
                rtx, ix, color=TARGET_ISO_TRACE, lw=CHROM_TRACE_LW,
                zorder=4, label='_nolegend_',
            )
            a1.set_ylabel('iso-window intensity', fontsize=7, color=TARGET_ISO_TRACE)
            a1.tick_params(axis='y', labelcolor=TARGET_ISO_TRACE, labelsize=7)
            if show_m0 and a1_m0 is not None:
                legend_handles = [
                    Line2D([0], [0], color=TARGET_ISO_TRACE, lw=CHROM_TRACE_LW, label=f'target {mz:.4f} ±iso'),
                ]
                for gi, grp in enumerate(m0_groups):
                    if grp not in col1_plot_groups:
                        continue
                    clr = _m0_trace_color(gi)
                    ls = _peptide_group_linestyle(gi)
                    legend_handles.append(
                        Line2D([0], [0], color=clr, lw=M0_TRACE_LW_UN, ls=ls, label=str(grp["legend"]))
                    )
                a1.legend(handles=legend_handles, fontsize=5.5, loc='upper right', framealpha=0.85)
            a1.set_title(
                f'{lbl} | {fname} | MS1 XIC target iso[-{iso_lo:.1f},+{iso_hi:.1f}]'
                + (' + M0 traces' if show_m0 else ''),
                fontsize=8,
            )
            if show_m0 and a1_m0 is not None:
                if zlo is not None:
                    a1.set_xlim(zlo, zhi)
                    m = (rtx >= zlo) & (rtx <= zhi)
                    if np.any(m):
                        ymax = float(ix[m].max())
                        if ymax > 0:
                            a1.set_ylim(0, ymax * 1.15)
                    m0_max = 0.0
                    for grp in col1_plot_groups:
                        rtx_m0, ix_m0 = xic_ms1_ppm(ms1_specs, float(grp["mono_mz"]), ppm_tol=m0_ppm_tol)
                        mm = (rtx_m0 >= zlo) & (rtx_m0 <= zhi)
                        if np.any(mm):
                            m0_max = max(m0_max, float(ix_m0[mm].max()))
                    if m0_max > 0:
                        a1_m0.set_ylim(0, m0_max * 1.15)
            elif zlo is not None:
                a1.set_xlim(zlo, zhi)
                m = (rtx >= zlo) & (rtx <= zhi)
                if np.any(m):
                    ymax = float(ix[m].max())
                    if ymax > 0:
                        a1.set_ylim(0, ymax * 1.15)

            a2.plot(rtf, tf, color=TARGET_ISO_TRACE, lw=CHROM_TRACE_LW, marker='o', ms=2.2)
            a2.set_title(f'{lbl} | MS2 frag-sum + tracked window', fontsize=8)
            if ch is not None:
                if integration_ch is not ch:
                    a2.axvspan(ch['rt_lo'], ch['rt_hi'], color='gray', alpha=0.20, zorder=0.25)
                _shade_integration_window(
                    a2, integration_ch['rt_lo'], integration_ch['rt_hi'], zorder=0.5,
                )
                a2.axvline(integration_ch['apex_rt'], color='red', ls='--', lw=0.9, alpha=0.85)
                yhi = a2.get_ylim()[1] if a2.get_ylim()[1] > 0 else (tf.max() if tf.size else 1)
                label_txt = f"apex={integration_ch['apex_rt']:.1f}s W={integration_ch['rt_hi'] - integration_ch['rt_lo']:.1f}s"
                if ch_refine_note:
                    label_txt += f"\n{ch_refine_note}"
                a2.text(
                    integration_ch['apex_rt'], yhi * 0.92,
                    label_txt,
                    fontsize=6.7, color='red', ha='left', va='top',
                    bbox=dict(facecolor='white', edgecolor='red', alpha=0.8, pad=1.5),
                )
            if zlo is not None:
                a2.set_xlim(zlo, zhi)

            _plot_registry_isotope_xics(
                a3, ms1_specs, m0_groups, z, mz, integration_ch, m0_ppm_tol, lbl,
                un_col3_group_indices=un_col3_group_indices,
            )

            mz_lo, mz_hi = mz - iso_lo, mz + iso_hi
            if integration_ch is not None:
                n, c, h = integrate_ms1(ms1_specs, integration_ch['rt_lo'], integration_ch['rt_hi'], mz_lo, mz_hi)
            else:
                n, c, h = (0, np.array([]), np.array([]))
            a4.set_title(f'{lbl} | Integrated MS1 (n={n})', fontsize=8)
            if c.size and np.any(h > 0):
                a4.set_xlim(mz_lo, mz_hi)
                _shade_integration_panel(a4, mz_lo, mz_hi)
                a4.vlines(
                    c, 0, h, color=SPECTRUM_TRACE_COLOR, lw=SPECTRUM_TRACE_LW,
                    alpha=0.92, zorder=2,
                )
                ms1_match_legend: list[Line2D] = []
                if m0_groups:
                    ms1_match_legend = _plot_ms1_registry_isotope_matches(
                        a4, c, h, m0_groups, z, mz, lbl, m0_ppm_tol,
                        un_col3_group_indices=un_col3_group_indices,
                    )
                ymax = float(h.max())
                a4.set_ylim(0, ymax * 1.18)
                spacing = C13 / max(1, z)
                ndown = int(np.floor((mz - mz_lo) / spacing)) + 1
                nup = int(np.floor((mz_hi - mz) / spacing)) + 1
                for kk in range(-ndown, nup + 1):
                    mk = mz + kk * spacing
                    if mz_lo <= mk <= mz_hi:
                        a4.axvline(mk, color='gray', ls=':', lw=0.6, alpha=0.35, zorder=6)
                a4.axvline(mz, color='red', ls='--', lw=1.0, alpha=0.9, zorder=6, label='PRM target')
                ms1_legend = [Line2D([0], [0], color='red', ls='--', lw=1.0, label='PRM target')]
                ms1_legend.extend(ms1_match_legend)
                if ms1_legend:
                    a4.legend(handles=ms1_legend, fontsize=5.5, loc='upper right', framealpha=0.85)
                ann_lines = _ms1_dabs_annotation_lines(
                    lbl, m0_groups, ms1_dabs_cache, un_col3_group_indices=un_col3_group_indices,
                )
                if ann_lines:
                    a4.text(
                        0.02, 0.98, '\n'.join(ann_lines),
                        transform=a4.transAxes,
                        ha='left', va='top', fontsize=6.2, color=TARGET_ISO_TRACE,
                        bbox=dict(facecolor='white', edgecolor='#cccccc', alpha=0.88, pad=1.2),
                        zorder=12,
                    )
            else:
                a4.text(
                    0.5, 0.5, 'no MS1 in window', transform=a4.transAxes,
                    ha='center', va='center', color='gray', fontsize=8,
                )
                a4.set_xlim(mz_lo, mz_hi)
                _shade_integration_panel(a4, mz_lo, mz_hi)

            if integration_ch is not None:
                rt_lo = float(integration_ch['rt_lo'])
                rt_hi = float(integration_ch['rt_hi'])
                _plot_integrated_msx_spectrum_panel(
                    a5, f'{lbl} | {act2} MS2', ms_by, mz, 2, rt_lo, rt_hi,
                    m0_groups, z, comet_cfg, theo_frag_cache, frag_ppm_tol, lbl, un_frag_label_keys,
                    un_col3_group_indices=un_col3_group_indices,
                )
                _plot_integrated_msx_spectrum_panel(
                    a6, f'{lbl} | {act3} MS3', ms_by, mz, 3, rt_lo, rt_hi,
                    m0_groups, z, comet_cfg, theo_frag_cache, frag_ppm_tol, lbl, un_frag_label_keys,
                    un_col3_group_indices=un_col3_group_indices,
                )
            else:
                for ax, title in [
                    (a5, f'{lbl} | {act2} MS2'),
                    (a6, f'{lbl} | {act3} MS3'),
                ]:
                    ax.set_title(title, fontsize=8)
                    ax.text(
                        0.5, 0.5, 'no integration window', transform=ax.transAxes,
                        ha='center', va='center', color='gray', fontsize=8,
                    )
                    ax.set_xlim(100, 2000)
                    _shade_integration_panel(ax, 100, 2000)

            a1.set_ylabel('intensity', fontsize=8)
            a3.set_ylabel('intensity', fontsize=8)
            if i == n_rows - 1:
                a1.set_xlabel('RT (s)', fontsize=9)
                a2.set_xlabel('RT (s)', fontsize=9)
                a3.set_xlabel('RT (s)', fontsize=9)
                a4.set_xlabel('m/z', fontsize=9)
                a5.set_xlabel('m/z', fontsize=9)
                a6.set_xlabel('m/z', fontsize=9)

        plt.tight_layout(rect=[0.03, 0.02, 1, legend_top])
        _apply_axes_box_aspect(axes, height_over_width=1.0 / TIMECOURSE_SUBPLOT_WIDTH_MULT)
        fig.canvas.draw()
        plt.savefig(
            out_path, dpi=PLOT_DPI, bbox_inches='tight',
            pad_inches=SAVE_PAD_INCHES, facecolor='white',
        )
        plt.close(fig)
        print('Saved', out_path)
        if not ns.skip_excel:
            try:
                _save_timecourse_excel_companion(
                    out_path, timecourse_sheets, target_mz=mz, precursor_charge=z,
                )
            except OSError as exc:
                print(f'WARNING: skipped timecourse Excel for {out_path}: {exc}')

        if zoom_rows:
            zoom_queue.append({
                'out_dir': out_dir,
                'safe': safe,
                'mz': mz,
                'z': z,
                'act2_label': act2_label,
                'act3_label': act3_label,
                'ms2_specs': list(un_frag_zoom_specs[2]),
                'ms3_specs': list(un_frag_zoom_specs[3]),
                'zoom_rows': list(zoom_rows),
                'm0_groups': col3_m0_groups,
                'theo_frag_cache': theo_frag_cache,
                'un_frag_label_keys': {
                    2: set(un_frag_label_keys[2]),
                    3: set(un_frag_label_keys[3]),
                },
                'hdx_fit_peptides': hdx_fit_by_window.get((round(mz, 3), int(z)), set()),
                'timecourse_sheets': timecourse_sheets,
            })

    zoom_ranks: set[int] = set()
    if not ns.skip_zoom:
        for ctx in zoom_queue:
            zoom_ranks.update(_registry_peptide_rank_map(ctx['m0_groups']).values())
        if ns.zoom_rank_only is not None:
            zoom_ranks = {int(ns.zoom_rank_only)}
        for rank in sorted(zoom_ranks):
            print(f'=== Fragment zoom rank r{rank} across {len(zoom_queue)} registry windows ===')
            for ctx in zoom_queue:
                _save_precursor_fragment_zoom_figures(
                    str(ctx['out_dir']), str(ctx['safe']), float(ctx['mz']), int(ctx['z']),
                    str(ctx['act2_label']), str(ctx['act3_label']),
                    list(ctx['ms2_specs']), list(ctx['ms3_specs']), list(ctx['zoom_rows']),
                    list(ctx['m0_groups']), comet_cfg, ctx['theo_frag_cache'],
                    frag_ppm_tol, ctx['un_frag_label_keys'],
                    hdx_fit_peptides=ctx['hdx_fit_peptides'],
                    rank_only=int(rank),
                    timecourse_sheets=ctx.get('timecourse_sheets'),
                    skip_excel=bool(ns.skip_excel),
                )

    print('Done generating continuity-tracked timecourse plots.')


if __name__ == '__main__':
    main()
