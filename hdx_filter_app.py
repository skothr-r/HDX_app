#!/usr/bin/env python3
"""
Interactive HDX filtering app – Streamlit interface for the Comet/HDX workflow.

Pipeline steps:
  Step 4: Prefilter — PEP, Q-value, prolines, mods, ppm (MS1 peptide), etc.
  Step 5: Significance — PPM of matched fragments (5 ppm), min significant fragment count
  Step 6: Extraction — envelope, apex >=10^5, coelution, shape correlation, S/N fragments 1% max

Run: streamlit run hdx_filter_app.py
"""

import gc
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from html import escape

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# For live streaming: child scripts must flush print() immediately
_PYTHONUNBUFFERED = '1'
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Default paths — data/data_hdx holds mzML + FASTA; pipeline CSVs saved there
DEFAULT_DATA_DIR = os.path.join(_SCRIPT_DIR, 'data', 'data_hdx')


def _upload_dir() -> str:
    """Directory for uploaded files (works on Streamlit Cloud where local data/ is empty)."""
    d = os.path.join(tempfile.gettempdir(), 'hdx_app_uploads')
    os.makedirs(d, exist_ok=True)
    return d

# Default Comet executable: project root (build with `make`)
def _default_comet_exe() -> str:
    # Prefer platform-specific binary (comet.linux.exe on Streamlit Cloud)
    candidates = ['comet.exe', 'comet']
    if sys.platform.startswith('linux'):
        candidates = ['comet.linux.exe', 'comet.exe', 'comet']
    for name in candidates:
        path = os.path.join(_SCRIPT_DIR, name)
        if os.path.exists(path) and os.access(path, os.X_OK):
            return path
    return os.path.join(_SCRIPT_DIR, 'comet.exe')  # fallback for user to correct

MZML_EXT = ('.mzml',)
FASTA_EXT = ('.fasta', '.fa', '.faa', '.fas')
CSV_EXT = ('.csv',)


def _clear_chromatograms_cache():
    """Clear mzML cache before run/generate so scripts load fresh data."""
    try:
        from visualization.chromatograms import clear_chromatograms_cache
        clear_chromatograms_cache()
    except ImportError:
        pass


def _find_files(data_dir: str, extensions: tuple) -> list[str]:
    """Find files with given extensions under data_dir (recursive). Returns full paths."""
    if not os.path.isdir(data_dir):
        return []
    exts = sorted(extensions, key=len, reverse=True)  # longer first (e.g. .fasta before .fa)
    out = []
    for root, _, files in os.walk(data_dir):
        for f in files:
            if any(f.lower().endswith(ext) for ext in exts):
                out.append(os.path.join(root, f))
    return sorted(out)


def _build_results_tree(root_dir: str) -> list[tuple[str, str]]:
    """Build list of (display_path, full_path) for images under root_dir."""
    if not root_dir or not os.path.isdir(root_dir):
        return []
    img_ext = ('.png', '.jpg', '.jpeg')
    out = []
    for dirpath, _, filenames in os.walk(root_dir):
        rel = os.path.relpath(dirpath, root_dir)
        prefix = rel + os.sep if rel != '.' else ''
        for f in sorted(filenames):
            if f.lower().endswith(img_ext):
                out.append((prefix + f, os.path.join(dirpath, f)))
    return sorted(out, key=lambda x: x[0])


def _list_chromatogram_plots(out_dir: str) -> tuple[list[str], str | None]:
    """
    List all individual chromatogram PNGs (exclude overlays) from extraction/ and rejected_extraction/.
    Returns (sorted list of paths, directory used).
    """
    accepted, rejected = _list_chromatogram_plots_by_status(out_dir, 'extraction')
    all_paths = sorted(set(accepted + rejected))
    return (all_paths, os.path.dirname(all_paths[0]) if all_paths else None)


def _list_chromatogram_plots_by_status(out_dir: str, step: str = 'extraction') -> tuple[list[str], list[str]]:
    """
    List chromatogram PNGs separately for accepted and rejected.
    step: 'extraction' -> extraction/ (accepted), rejected_extraction/ (rejected)
          'envelope' -> envelope/ (accepted), envelope_rejected/ (rejected)
    Returns (accepted_paths, rejected_paths).
    """
    def _delete_suffixed_extraction_plots(dirs: list[str]) -> None:
        """Delete known-bad suffixed extraction plots so they are never shown."""
        for d in dirs:
            if not os.path.isdir(d):
                continue
            for f in os.listdir(d):
                fl = f.lower()
                if not fl.endswith(('.png', '.jpg', '.jpeg')):
                    continue
                if '_accepted' in fl or '_rejected' in fl:
                    try:
                        os.remove(os.path.join(d, f))
                    except Exception:
                        # Best-effort cleanup; plotting should still continue.
                        pass

    def _extract_rt_sort_value(path: str) -> float | None:
        """
        Extract sortable RT (minutes) from extraction filenames like:
        ..._rt12p34_13p56min_...
        """
        name = os.path.basename(path).lower()
        m = re.search(r'_rt([0-9]+(?:p[0-9]+)?)_([0-9]+(?:p[0-9]+)?)min', name)
        if not m:
            return None
        try:
            return float(m.group(1).replace('p', '.'))
        except Exception:
            return None

    def _sort_extraction_paths_by_rt(paths: list[str]) -> list[str]:
        uniq = list(set(paths))
        return sorted(
            uniq,
            key=lambda p: (
                _extract_rt_sort_value(p) is None,
                _extract_rt_sort_value(p) if _extract_rt_sort_value(p) is not None else float('inf'),
                os.path.basename(p).lower(),
            ),
        )

    if step == 'extraction':
        accepted_dirs = [
            os.path.join(out_dir, 'extraction', 'accepted', 'chromatograms_peptides'),
            os.path.join(out_dir, 'extraction', 'chromatograms_peptides'),
            os.path.join(out_dir, 'extraction'),
            os.path.join(out_dir, 'accepted', 'chromatograms_peptides'),
        ]
        rejected_dirs = [os.path.join(out_dir, 'rejected_extraction')]
        _delete_suffixed_extraction_plots(accepted_dirs + rejected_dirs)
    else:  # envelope
        accepted_dirs = [os.path.join(out_dir, 'envelope')]
        rejected_dirs = [os.path.join(out_dir, 'envelope_rejected')]
    accepted_paths = []
    rejected_paths = []

    def _pngs_in_dir(d: str) -> list[str]:
        if not os.path.isdir(d):
            return []
        return [
            os.path.join(d, f) for f in sorted(os.listdir(d))
            if f.lower().endswith(('.png', '.jpg', '.jpeg')) and 'overlay' not in f.lower()
        ]
    for d in accepted_dirs:
        files = _pngs_in_dir(d)
        if not files:
            continue
        if step == 'extraction':
            # Prefer the combined-style peptide plots (no _accepted/_rejected suffix),
            # which correspond to the left-column chromatogram panels users want.
            preferred = [
                p for p in files
                if '_accepted' not in os.path.basename(p).lower()
                and '_rejected' not in os.path.basename(p).lower()
            ]
            if preferred:
                accepted_paths.extend(preferred)
            else:
                accepted_paths.extend([p for p in files if '_accepted' in os.path.basename(p).lower()])
        else:
            accepted_paths.extend(files)
    for d in rejected_dirs:
        files = _pngs_in_dir(d)
        if not files:
            continue
        if step == 'extraction':
            # Keep explicit rejected files for extraction rejected panel.
            tagged = [p for p in files if '_rejected' in os.path.basename(p).lower()]
            rejected_paths.extend(tagged if tagged else files)
        else:
            rejected_paths.extend(files)
    if step == 'extraction':
        return (_sort_extraction_paths_by_rt(accepted_paths), _sort_extraction_paths_by_rt(rejected_paths))
    return (sorted(set(accepted_paths)), sorted(set(rejected_paths)))


def _find_chromatogram_plots(out_dir: str) -> tuple[str | None, str | None]:
    """Find first chromatogram and MS1 overlay. Returns (first_chromatogram_path, ms1_overlay_path)."""
    chrom_paths, _ = _list_chromatogram_plots(out_dir)
    first_chrom = chrom_paths[0] if chrom_paths else None
    ms1_overlay = None
    ms1_overlay_preferred = None
    candidates = [
        os.path.join(out_dir, 'extraction', 'accepted', 'chromatograms_peptides'),
        os.path.join(out_dir, 'extraction', 'chromatograms_peptides'),
        os.path.join(out_dir, 'extraction'),
        os.path.join(out_dir, 'accepted', 'chromatograms_peptides'),
    ]
    for d in candidates:
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if not f.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            path = os.path.join(d, f)
            if 'overlay' in f.lower() and 'all_peptides_overlay' in f.lower():
                if f == 'all_peptides_overlay.png':
                    ms1_overlay_preferred = path
                elif ms1_overlay is None:
                    ms1_overlay = path
        if ms1_overlay_preferred or ms1_overlay:
            break
    return (first_chrom, ms1_overlay_preferred or ms1_overlay)


QVALUE_COLS = ['perc_qvalue', 'percolator_qvalue', 'q-value', 'qvalue', 'percolator_q-value', 'FDR', 'fdr']
PEP_COLS = ['perc_PEP', 'percolator_PEP', 'PEP', 'pep']
XCORR_COLS = ['xcorr', 'XCorr', 'xCorr']
SP_COLS = ['sp_score', 'sp', 'Sp']
DELTA_CN_COLS = ['delta_cn', 'deltacn', 'DeltaCn']
EVALUE_COLS = ['e-value', 'evalue', 'E-value']


def _resolve_col(df: pd.DataFrame, candidates: list[str]):
    for c in candidates:
        if c in df.columns:
            return c
    return None


# Column names: comet_matched_frags (raw), significant_frags (pass all 3: 5ppm, min sig frags, 1% max)
COMET_MATCHED_FRAGS_COL = 'comet_matched_frags'
COMET_MATCHED_FRAGS_MZ_COL = 'comet_matched_frags_mz'
COMET_MATCHED_FRAGS_INTENS_COL = 'comet_matched_frags_intensities'
COMET_MATCHED_FRAGS_QUALITY_COL = 'comet_matched_frags_quality_scores'
SIGNIFICANT_FRAGS_COL = 'significant_frags'


def _normalize_matched_frag_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename legacy columns for backward compatibility."""
    if 'matched fragment ions' in df.columns and COMET_MATCHED_FRAGS_COL not in df.columns:
        df = df.rename(columns={'matched fragment ions': COMET_MATCHED_FRAGS_COL})
    if 'matched fragment ion mz' in df.columns and COMET_MATCHED_FRAGS_MZ_COL not in df.columns:
        df = df.rename(columns={'matched fragment ion mz': COMET_MATCHED_FRAGS_MZ_COL})
    if 'matched fragment ion intensities' in df.columns and COMET_MATCHED_FRAGS_INTENS_COL not in df.columns:
        df = df.rename(columns={'matched fragment ion intensities': COMET_MATCHED_FRAGS_INTENS_COL})
    if 'comet_amtched_frags_intensities' in df.columns and COMET_MATCHED_FRAGS_INTENS_COL not in df.columns:
        df = df.rename(columns={'comet_amtched_frags_intensities': COMET_MATCHED_FRAGS_INTENS_COL})
    if 'matched fragment ion quality scores' in df.columns and COMET_MATCHED_FRAGS_QUALITY_COL not in df.columns:
        df = df.rename(columns={'matched fragment ion quality scores': COMET_MATCHED_FRAGS_QUALITY_COL})
    if 'significant_fragment_ions' in df.columns and SIGNIFICANT_FRAGS_COL not in df.columns:
        df = df.rename(columns={'significant_fragment_ions': SIGNIFICANT_FRAGS_COL})
    if 'significant_fragment_mz' in df.columns and 'significant_frags_mz' not in df.columns:
        df = df.rename(columns={'significant_fragment_mz': 'significant_frags_mz'})
    # Backward compat: old refined_significant_frags -> significant_frags when sig not present
    if 'refined_significant_frags' in df.columns and SIGNIFICANT_FRAGS_COL not in df.columns:
        df = df.rename(columns={'refined_significant_frags': SIGNIFICANT_FRAGS_COL})
    if 'refined_significant_frags_mz' in df.columns and 'significant_frags_mz' not in df.columns:
        df = df.rename(columns={'refined_significant_frags_mz': 'significant_frags_mz'})
    return df


def _add_internal_identifier_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """
    Internal compatibility aliases for workflow-wide column rename.
    Keep raw CSV headers untouched for previews/downloads.
    """
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
    if 'MS1_RT_intensity' in df.columns and 'MS1_retention_time_intensity' not in df.columns:
        df['MS1_retention_time_intensity'] = df['MS1_RT_intensity']
    if 'MS1_RT_min' in df.columns and 'MS1_RT_minutes' not in df.columns:
        df['MS1_RT_minutes'] = df['MS1_RT_min']
    if 'MS1_RT_integration_start_sec' in df.columns and 'detected_peak_min_rt' not in df.columns:
        df['detected_peak_min_rt'] = df['MS1_RT_integration_start_sec']
    if 'MS1_RT_integration_stop_sec' in df.columns and 'detected_peak_max_rt' not in df.columns:
        df['detected_peak_max_rt'] = df['MS1_RT_integration_stop_sec']
    if 'MS1_RT_collection_start_sec' in df.columns and 'collection_min_rt' not in df.columns:
        df['collection_min_rt'] = df['MS1_RT_collection_start_sec']
    if 'MS1_RT_collection_stop_sec' in df.columns and 'collection_max_rt' not in df.columns:
        df['collection_max_rt'] = df['MS1_RT_collection_stop_sec']
    if 'MS1_mz_error_ppm' in df.columns and 'MS1_mz_error' not in df.columns:
        df['MS1_mz_error'] = df['MS1_mz_error_ppm']
    if 'MS1_mz_error' in df.columns and 'MS1_mz_error_ppm' not in df.columns:
        df['MS1_mz_error_ppm'] = df['MS1_mz_error']
    return df


@st.cache_data(ttl=300, max_entries=5, show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    """Load CSV with optional Comet header skip. Cached to reduce memory (max 5 CSVs, 5 min TTL)."""
    try:
        with open(path, 'r') as f:
            first = f.readline()
        skip = 1 if 'CometVersion' in first else 0
    except Exception:
        skip = 0
    df = pd.read_csv(path, sep=',', skiprows=skip, engine='python', quotechar='"', on_bad_lines='warn')
    df = _normalize_matched_frag_columns(df)
    return _add_internal_identifier_aliases(df)


def _render_step_output_csv(out_path: str, key_prefix: str, expanded: bool = False) -> None:
    """Render output CSV block: status, path (clickable), expander with download + scrollable dataframe."""
    exists = os.path.exists(out_path)
    check = '✓' if exists else '○'
    color = '#84cc16' if exists else '#888'
    status_txt = 'Saved' if exists else 'Not saved'
    fname = os.path.basename(out_path)
    try:
        from pathlib import Path
        file_uri = Path(os.path.abspath(out_path)).as_uri() if exists else ''
    except Exception:
        file_uri = 'file://' + os.path.abspath(out_path) if exists else ''
    safe_path = escape(out_path) if out_path else ''
    link_html = f'<a href="{escape(file_uri)}" target="_blank" style="color:#6ea8fe; text-decoration:underline;">{safe_path}</a>' if exists else safe_path
    st.markdown(f"""
    <div style="max-height:80px; overflow-y:auto; overflow-x:auto; padding:8px; background:#111; border-radius:4px; font-family:'Times New Roman', Times, serif; font-size:0.85rem;">
        <span style="color:{color}; font-weight:bold;">{check} {status_txt}</span> — {link_html}
    </div>
    """, unsafe_allow_html=True)
    with st.expander('Output CSV preview', expanded=expanded, icon='▶'):
        st.code(out_path, language=None)
        if exists:
            try:
                mtime_txt = datetime.fromtimestamp(os.path.getmtime(out_path)).strftime('%Y-%m-%d %H:%M:%S')
            except Exception:
                mtime_txt = 'unknown'
            st.caption(f'File exists ({os.path.getsize(out_path):,} bytes) | modified: {mtime_txt}')
            with open(out_path, 'rb') as f:
                st.download_button('Download / open CSV', f.read(), file_name=fname, mime='text/csv', key=key_prefix + '_dl')
            try:
                # Preview should always reflect on-disk state immediately after a run.
                # Do an uncached read here (app-wide cached loader is still used elsewhere).
                try:
                    with open(out_path, 'r') as f:
                        first = f.readline()
                    skip = 1 if 'CometVersion' in first else 0
                except Exception:
                    skip = 0
                step_df = pd.read_csv(out_path, sep=',', skiprows=skip, engine='python', quotechar='"', on_bad_lines='warn')
                step_df = _normalize_matched_frag_columns(step_df)
                st.dataframe(step_df, use_container_width=True, height=400)
            except Exception as e:
                st.caption(f'Could not load preview: {e}')


def _resolve_qcol(df: pd.DataFrame):
    return _resolve_col(df, QVALUE_COLS)


def _resolve_pepcol(df: pd.DataFrame):
    return _resolve_col(df, PEP_COLS)


STEP5_REQUIRED_COLS = [COMET_MATCHED_FRAGS_COL, COMET_MATCHED_FRAGS_MZ_COL]


def _csv_has_step5_cols(path: str) -> bool:
    """Check if CSV has columns required for Step 5 (accuracy filter)."""
    try:
        df = load_csv(path)
        return all(c in df.columns for c in STEP5_REQUIRED_COLS)
    except Exception:
        return False


def _resolve_step_input(csv_path: str, out_dir: str, suffixes: list[str]) -> str:
    """Resolve pipeline input: prefer chained output (e.g. _confidence) if it exists."""
    base = os.path.splitext(os.path.basename(csv_path))[0]
    for suf in suffixes:
        cand = os.path.join(out_dir, base + suf + '.csv')
        if os.path.exists(cand):
            return cand
    return csv_path


def _resolve_step_input_from_conf_base(conf_base: str, out_dir: str, suffixes: list[str], fallback: str = None) -> str:
    """Resolve pipeline input using conf_base (root name before _prefilter/_confidence)."""
    for suf in suffixes:
        cand = os.path.join(out_dir, conf_base + suf + '.csv')
        if os.path.exists(cand):
            return cand
    return fallback or ''


def _latest_csv_with_suffix(out_dir: str, suffixes: list[str]) -> str:
    """Return most recently modified CSV in out_dir whose basename ends with any suffix."""
    if not out_dir or not os.path.isdir(out_dir):
        return ''
    matches = []
    try:
        for name in os.listdir(out_dir):
            if not name.lower().endswith('.csv'):
                continue
            base = name[:-4]
            if any(base.endswith(suf) for suf in suffixes):
                full = os.path.join(out_dir, name)
                try:
                    mtime = os.path.getmtime(full)
                except OSError:
                    continue
                matches.append((mtime, full))
    except OSError:
        return ''
    if not matches:
        return ''
    matches.sort(key=lambda x: x[0], reverse=True)
    return matches[0][1]


def _pass_confidence(row, df, qcol, pepcol, q_thresh, pep_thresh, enabled=True):
    if not enabled:
        return True
    q_ok = True
    if qcol and qcol in df.columns and pd.notna(row.get(qcol)):
        try:
            q_ok = float(row[qcol]) <= q_thresh
        except (TypeError, ValueError):
            q_ok = True
    pep_ok = True
    if pepcol and pepcol in df.columns and pd.notna(row.get(pepcol)):
        try:
            pep_ok = float(row[pepcol]) <= pep_thresh
        except (TypeError, ValueError):
            pep_ok = True
    return q_ok and pep_ok


def _to_bool(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return False
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    return s in ('true', '1', 'yes')


def _has_modifications(mods):
    """True if peptide has any modification (exclude). Unmodified only: empty, -, nan."""
    if mods is None or (isinstance(mods, float) and np.isnan(mods)):
        return False
    s = str(mods).strip()
    if not s or s == '-' or s.lower() in ('nan', 'none'):
        return False
    return True


def main():
    st.set_page_config(page_title='HDX Filtering', page_icon='🔬', layout='wide')
    # Load Material Icons so tabs, dropdowns, expanders render icons correctly
    st.markdown(
        '<link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">',
        unsafe_allow_html=True,
    )
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@24,400,0,0');
        /* Prevent overflow on media elements (avoids Shared Element Transitions API warning) */
        img, video, canvas { overflow: hidden !important; }
        /* Text elements: serif. Never use *, div, or body * — scope to specific elements. */
        body, p, span, label, h1, h2, h3, .stMarkdown, .stCaption, [data-testid="stMetric"] {
            font-family: "Times New Roman", Times, "DejaVu Serif", "Liberation Serif", "Apple Symbols", "Segoe UI Symbol", serif !important;
        }
        /* Terminal / code blocks — Times New Roman (no monospace) */
        .stCodeBlock, [data-testid="stCodeBlock"], .stCodeBlock pre, .stCodeBlock code,
        [data-testid="stCodeBlock"] pre, [data-testid="stCodeBlock"] code {
            font-family: "Times New Roman", Times, "DejaVu Serif", "Liberation Serif", serif !important;
        }
        /* Owned panels — wrap logical blocks with <div class="section"> */
        .section {
            background-color: #002755;
            border: 1px solid rgba(255,255,255,0.25);
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
        }
        /* Restore icon fonts — do not override with serif */
        .material-icons, .MaterialSymbolsRounded, [class*="material-icons"], [class*="MaterialSymbols"] {
            font-family: "Material Icons", "Material Symbols Rounded", "Segoe UI Symbol", sans-serif !important;
        }
        /* Sidebar collapse/expand button in banner — restore icon font (chevron/menu) */
        [data-testid="stSidebarCollapseButton"],
        [data-testid="stSidebarCollapseButton"] span,
        button[aria-label*="sidebar" i],
        button[aria-label*="collapse" i],
        button[aria-label*="expand" i],
        [aria-label*="sidebar" i] button,
        [aria-label*="sidebar" i] button span,
        /* Header first button = collapse when sidebar collapsed; BaseWeb icon containers */
        [data-testid="stHeader"] > div:first-child button,
        [data-testid="stHeader"] > div:first-child button span,
        [data-testid="stHeader"] [data-baseweb="button"]:first-of-type,
        [data-testid="stHeader"] [data-baseweb="button"]:first-of-type span,
        [data-testid="stHeader"] [data-baseweb="icon"],
        [data-testid="stToolbar"] [data-baseweb="icon"] {
            font-family: "Material Icons", "Material Symbols Rounded", "Segoe UI Symbol", sans-serif !important;
        }
        html, body, .stApp, [data-testid="stAppViewContainer"],
        [data-testid="stHeader"], .main .block-container,
        section[data-testid="stSidebar"], section[data-testid="stSidebar"] > div,
        [data-testid="stSidebarContent"], .stExpander, [data-testid="stExpander"],
        div[data-testid="stVerticalBlock"], [data-testid="stVerticalBlock"],
        .element-container, .stMarkdown, div[data-testid="column"],
        [data-testid="stMetric"], .stDataFrame, [data-testid="stDataFrame"],
        .stCodeBlock, [data-testid="stCodeBlock"], [data-testid="stTabs"],
        [data-baseweb="tab-list"], [data-baseweb="tab-panel"],
        [data-testid="stImage"], .stImage {
            background-color: #000000 !important;
        }
        /* Ensure text is visible on black background (Sequence tab and all main content) */
        .main .block-container, .main [data-testid="stVerticalBlock"],
        .main .stMarkdown, .main p, .main span, .main label,
        .main h1, .main h2, .main h3, .main .stCaptionContainer,
        [data-testid="stTabs"] [role="tabpanel"], .stSubheader, .stCaption {
            color: #FFF8E7 !important;
        }
        [data-baseweb="input"], [data-baseweb="select"],
        [role="listbox"] > div, [data-baseweb="popover"] {
            background-color: #000000 !important;
        }
        /* All buttons (Run, Generate, Download, etc.) — blue */
        .stButton > button, .stButton button,
        .stDownloadButton > button, .stDownloadButton button,
        .main .stButton button, .main .stDownloadButton button,
        [data-testid="stButton"] button, [data-baseweb="button"],
        button[kind="primary"], button[kind="secondary"], button[kind="tertiary"] {
            background-color: #002755 !important;
            border-color: #004a99 !important;
            color: #FFF8E7 !important;
        }
        /* Top header band (file change, rerun, deploy) — dark navy */
        [data-testid="stHeader"],
        [data-testid="stHeader"] > div,
        header[data-testid="stHeader"],
        [data-testid="stToolbar"] {
            background-color: #002d6b !important;
        }
        /* HDXapp branding on header — large title + subtitle lines, span across banner */
        [data-testid="stHeader"] {
            padding: 0.3rem 0.75rem !important;
            display: flex !important;
            align-items: center !important;
        }
        [data-testid="stHeader"]::before {
            content: "HDXapp - The Guttman Lab\\A Department of Medicinal Chemistry - School of Pharmacology - University of Washington - Seattle";
            white-space: pre-line;
            font-family: "Times New Roman", Times, serif !important;
            color: #FFF8E7 !important;
            font-size: 0.85rem !important;
            font-weight: 700 !important;
            line-height: 1.3 !important;
            display: block !important;
            flex: 1 1 auto !important;
            min-width: 0 !important;
        }
        /* Sidebar — dark navy to match header banner */
        section[data-testid="stSidebar"],
        section[data-testid="stSidebar"] > div,
        [data-testid="stSidebarContent"] {
            background-color: #002d6b !important;
            color: #FFF8E7 !important;
        }
        section[data-testid="stSidebar"] {
            padding: 0.15rem 0.35rem !important;
            font-size: 1rem !important;
            font-family: "Times New Roman", Times, "DejaVu Serif", serif !important;
        }
        [data-testid="stSidebarContent"] {
            overflow-y: scroll !important;
        }
        section[data-testid="stSidebar"]::-webkit-scrollbar,
        [data-testid="stSidebarContent"]::-webkit-scrollbar {
            width: 8px !important;
        }
        section[data-testid="stSidebar"]::-webkit-scrollbar-track,
        [data-testid="stSidebarContent"]::-webkit-scrollbar-track {
            background: #002755 !important;
        }
        section[data-testid="stSidebar"]::-webkit-scrollbar-thumb,
        [data-testid="stSidebarContent"]::-webkit-scrollbar-thumb {
            background: #4a7ab8 !important;
            border-radius: 4px;
        }
        section[data-testid="stSidebar"],
        [data-testid="stSidebarContent"] {
            scrollbar-width: thin !important;
            scrollbar-color: #4a7ab8 #002755 !important;
        }
        section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
            margin: 0.15rem 0 !important; font-size: 1rem !important; font-weight: 600;
            font-family: "Times New Roman", Times, "DejaVu Serif", serif !important;
        }
        section[data-testid="stSidebar"] .stMarkdown, section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] .stExpander summary, section[data-testid="stSidebar"] [data-testid="stExpander"] summary {
            margin: 0.15rem 0 !important; font-size: 1rem !important;
            font-family: "Times New Roman", Times, "DejaVu Serif", serif !important;
        }
        /* Sidebar: remove black blocks behind inputs; inputs sit on blue panel */
        section[data-testid="stSidebar"] .element-container,
        section[data-testid="stSidebar"] [data-testid="stVerticalBlock"],
        section[data-testid="stSidebar"] .stMarkdown,
        section[data-testid="stSidebar"] .stExpander .element-container,
        section[data-testid="stSidebar"] .stExpander [data-testid="stVerticalBlock"],
        section[data-testid="stSidebar"] .stExpander .stMarkdown,
        section[data-testid="stSidebar"] .stExpander [data-testid="stExpanderDetails"] .element-container,
        section[data-testid="stSidebar"] .stExpander [data-testid="stExpanderDetails"] [data-testid="stVerticalBlock"],
        section[data-testid="stSidebar"] .stExpander [data-testid="stExpanderDetails"] .stMarkdown {
            background-color: transparent !important;
        }
        /* Sidebar blocks and buttons — solid blue with black outline */
        section[data-testid="stSidebar"] .stExpander,
        section[data-testid="stSidebar"] [data-testid="stExpander"],
        section[data-testid="stSidebar"] .stExpander > details,
        section[data-testid="stSidebar"] [data-testid="stExpander"] details {
            background-color: #002755 !important;
            border: 1px solid #000000 !important;
            border-radius: 4px !important;
        }
        section[data-testid="stSidebar"] .stExpander summary,
        section[data-testid="stSidebar"] [data-testid="stExpander"] summary {
            background-color: #002755 !important;
            border: none !important;
            color: #FFF8E7 !important;
        }
        section[data-testid="stSidebar"] .stExpander [data-testid="stExpanderDetails"],
        section[data-testid="stSidebar"] [data-testid="stExpander"] > div {
            background-color: #002755 !important;
        }
        section[data-testid="stSidebar"] .stExpander [data-testid="stExpanderDetails"] .stMarkdown,
        section[data-testid="stSidebar"] .stExpander [data-testid="stExpanderDetails"] p,
        section[data-testid="stSidebar"] .stExpander [data-testid="stExpanderDetails"] label,
        section[data-testid="stSidebar"] .stExpander [data-testid="stExpanderDetails"] .element-container {
            font-size: 1rem !important;
            font-family: "Times New Roman", Times, "DejaVu Serif", serif !important;
        }
        /* Strong selectors for compact sidebar buttons (higher specificity) */
        body section[data-testid="stSidebar"] .stButton > button,
        body section[data-testid="stSidebar"] .stButton button,
        body section[data-testid="stSidebar"] div.stButton button,
        body section[data-testid="stSidebar"] [data-baseweb="button"],
        body section[data-testid="stSidebar"] [data-baseweb="base-button"],
        body section[data-testid="stSidebar"] button[kind="secondary"],
        body section[data-testid="stSidebar"] button[kind="tertiary"] {
            padding: 0.2rem 0.5rem !important;
            font-size: 1rem !important;
            min-height: unset !important;
            height: 1.8rem !important;
            line-height: 1.2 !important;
            box-sizing: border-box !important;
            background-color: #002755 !important;
            border: 1px solid #000000 !important;
            color: #FFF8E7 !important;
            font-family: "Times New Roman", Times, "DejaVu Serif", serif !important;
        }
        section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div { padding: 0.05rem 0 !important; }
        section[data-testid="stSidebar"] .stSlider,
        section[data-testid="stSidebar"] .stSlider label,
        section[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] {
            font-size: 1rem !important;
            font-family: "Times New Roman", Times, "DejaVu Serif", serif !important;
        }
        section[data-testid="stSidebar"] .stSlider { padding: 0.08rem 0 !important; }
        section[data-testid="stSidebar"] [data-baseweb="input"],
        section[data-testid="stSidebar"] [data-baseweb="input"] input {
            background-color: #002755 !important;
            color: #FFF8E7 !important;
            font-size: 1rem !important;
            font-family: "Times New Roman", Times, "DejaVu Serif", serif !important;
            border: 1px solid #000000 !important;
            border-radius: 4px !important;
        }
        section[data-testid="stSidebar"] [data-baseweb="select"],
        section[data-testid="stSidebar"] [data-baseweb="select"] > div,
        section[data-testid="stSidebar"] [data-baseweb="select"] input,
        section[data-testid="stSidebar"] [data-baseweb="select"] span {
            background-color: #002755 !important;
            color: #FFF8E7 !important;
            font-size: 1rem !important;
            font-family: "Times New Roman", Times, "DejaVu Serif", serif !important;
            border: 1px solid #000000 !important;
            border-radius: 4px !important;
        }
        section[data-testid="stSidebar"] [data-baseweb="input"] input { padding: 0.25rem 0.5rem !important; }
        section[data-testid="stSidebar"] [data-baseweb="select"] { min-height: 2.2rem !important; }
        section[data-testid="stSidebar"] [data-testid="stMetric"] { padding: 0.2rem 0 !important; }
        section[data-testid="stSidebar"] [data-testid="stMetric"] label,
        section[data-testid="stSidebar"] [data-testid="stMetric"] [data-testid="stMetricValue"],
        section[data-testid="stSidebar"] .stCaptionContainer {
            font-size: 1rem !important;
            font-family: "Times New Roman", Times, "DejaVu Serif", serif !important;
        }
        section[data-testid="stSidebar"] hr { margin: 0.2rem 0 !important; }
        /* Compact padding around Select Peptide Targets */
        #select-peptide-targets-section {
            margin: 0 !important;
            padding: 0 0 0.5rem 0 !important;
        }
        /* Reduce margins and padding globally */
        .main .block-container { padding: 0.5rem 1rem 1rem 1rem !important; max-width: 100% !important; }
        [data-testid="stVerticalBlock"] > div { padding: 0.1rem 0 !important; margin: 0 !important; }
        .element-container { margin: 0 !important; padding: 0 !important; }
        .stMarkdown { margin: 0 !important; padding: 0.1rem 0 !important; }
        [data-testid="stExpander"] { margin: 0.15rem 0 !important; }
        [data-testid="stExpander"] > details { padding: 0.25rem 0.5rem !important; }
        [data-testid="stTabs"] { margin: 0.2rem 0 !important; padding: 0 !important; }
        [data-testid="stTabs"] [role="tabpanel"] { padding: 0.25rem 0 !important; }
        [data-testid="stHorizontalBlock"] { gap: 0.5rem !important; }
        [data-testid="stVerticalBlock"] { gap: 0.15rem !important; }
        [data-testid="column"] { padding: 0.25rem !important; }
        [data-testid="stMetric"] { padding: 0.15rem 0 !important; }
        .stSlider { padding: 0.1rem 0 !important; }
        [data-baseweb="input"], [data-baseweb="select"] { padding: 0.15rem 0.3rem !important; }
        /* Dropdown list options — larger font */
        [role="listbox"] div, [data-baseweb="popover"] div, [data-baseweb="menu"] li,
        [data-baseweb="select"] [role="listbox"] div {
            font-size: 0.95rem !important;
        }
        .stButton > button { padding: 0.2rem 0.5rem !important; }
        hr, .stDivider { margin: 0.3rem 0 !important; }
        .stCaptionContainer { margin: 0.05rem 0 !important; padding: 0 !important; }
        /* Fix expander icon showing as "expand_arrow_right" text — restore Material Symbols font */
        [data-testid="stExpander"] summary > *:first-child,
        [data-testid="stExpander"] summary span[data-testid],
        [data-testid="stExpander"] summary [class*="Icon"],
        [data-testid="stExpander"] summary [class*="icon"] {
            font-family: 'Material Symbols Rounded', 'Material Icons', 'Segoe UI Symbol', sans-serif !important;
        }
        </style>
        """, unsafe_allow_html=True)
    if 'script_output' not in st.session_state:
        st.session_state.script_output = ''
        st.session_state.script_step = ''
    if 'pending_run' not in st.session_state:
        st.session_state.pending_run = None  # (cmd, step_name, out_suffix, needs_visualization, output_path)
    if 'last_step_output' not in st.session_state:
        st.session_state.last_step_output = None  # path to last successful step output CSV
    if 'last_step_name' not in st.session_state:
        st.session_state.last_step_name = None  # e.g. 'Extraction', 'Prefilter'
    if 'last_step_run_started_ts' not in st.session_state:
        st.session_state.last_step_run_started_ts = None
    try:
        import pyopenms  # noqa: F401
        pyopenms_ok = True
    except (ModuleNotFoundError, ImportError):
        pyopenms_ok = False
        st.warning('pyopenms not found. Steps 5, 6, and 8 require it. Use: conda activate hdx_app (or set DYLD_LIBRARY_PATH on macOS).')

    # Sidebar: directory + optional upload (when dir is empty)
    upload_dir_default = _upload_dir()
    default_data = os.path.abspath(DEFAULT_DATA_DIR)
    has_files_in_default = any(_find_files(default_data, MZML_EXT + FASTA_EXT + CSV_EXT))
    default_out = default_data if has_files_in_default else upload_dir_default
    # When default dir is empty (e.g. Streamlit Cloud), show upload path so users know where to add files
    data_dir_default = default_data if has_files_in_default else upload_dir_default
    with st.sidebar.expander('Input', expanded=True, icon='▶'):
        data_dir_input = st.text_input('Data directory', data_dir_default, autocomplete='off',
                                       help='Path to folder containing FASTA and mzML. When empty, use the upload section below.')
        output_dir = st.text_input('Out dir', default_out, autocomplete='off',
                                   help='Pipeline outputs go here.')

    # When dir is empty: offer upload to populate it (web app)
    # On Streamlit Cloud: write to disk and store path only (not bytes) to reduce memory
    # Streamlit Community Cloud has ~1–2.7GB RAM; mzML files can be 100MB+ and would double memory if kept in session
    _is_cloud = os.environ.get('STREAMLIT_SERVER_HEADLESS') == 'true'
    upload_d = _upload_dir()
    with st.sidebar.expander('Upload (when directory is empty)', expanded=not has_files_in_default):
        if not has_files_in_default:
            st.caption('Add FASTA and mzML here. Files are saved to the data directory above.')
        if uploaded_fasta := st.file_uploader('FASTA', type=['fasta', 'fa', 'faa', 'fas'], key='fasta_upload'):
            data = uploaded_fasta.getvalue()
            p = os.path.join(upload_d, os.path.basename(uploaded_fasta.name) or 'uploaded.fasta')
            with open(p, 'wb') as f:
                f.write(data)
            st.session_state.uploaded_fasta = (os.path.basename(uploaded_fasta.name) or 'uploaded.fasta', p)
            st.session_state.uploaded_data_dir = upload_d
        if uploaded_mzml := st.file_uploader('mzML', type=['mzml', 'mzML'], key='mzml_upload'):
            data = uploaded_mzml.getvalue()
            size_mb = len(data) / (1024 * 1024)
            if _is_cloud and size_mb > 150:
                st.warning(f'mzML is {size_mb:.0f} MB. Max 150 MB recommended on Streamlit Cloud to avoid memory limits. Consider using a smaller file or running locally.')
            p = os.path.join(upload_d, os.path.basename(uploaded_mzml.name) or 'uploaded.mzML')
            with open(p, 'wb') as f:
                f.write(data)
            st.session_state.uploaded_mzml = (os.path.basename(uploaded_mzml.name) or 'uploaded.mzML', p)
            st.session_state.uploaded_data_dir = upload_d
        if uploaded_csv := st.file_uploader('CSV (optional)', type=['csv'], key='csv_upload'):
            p = os.path.join(upload_d, os.path.basename(uploaded_csv.name) or 'uploaded.csv')
            with open(p, 'wb') as f:
                f.write(uploaded_csv.getvalue())
            st.session_state.uploaded_data_dir = upload_d
        # Clear uploads to free memory (useful when hitting resource limits)
        if st.session_state.get('uploaded_fasta') or st.session_state.get('uploaded_mzml'):
            if st.button('Clear uploads (free memory)', key='clear_uploads'):
                for k in ('uploaded_fasta', 'uploaded_mzml', 'uploaded_data_dir'):
                    st.session_state.pop(k, None)
                _clear_chromatograms_cache()
                gc.collect()
                st.rerun()
    # Re-write only when we have legacy bytes in session (upgrade to path-only)
    if up := st.session_state.get('uploaded_fasta'):
        name, val = up
        if isinstance(val, bytes):
            p = os.path.join(upload_d, name)
            with open(p, 'wb') as f:
                f.write(val)
            st.session_state.uploaded_fasta = (name, p)
        st.session_state.uploaded_data_dir = upload_d
    if up := st.session_state.get('uploaded_mzml'):
        name, val = up
        if isinstance(val, bytes):
            p = os.path.join(upload_d, name)
            with open(p, 'wb') as f:
                f.write(val)
            st.session_state.uploaded_mzml = (name, p)
        st.session_state.uploaded_data_dir = upload_d
    # Use upload dir when input dir is empty and we have uploads
    data_dir = (data_dir_input or default_data).strip()
    if not os.path.isdir(data_dir):
        data_dir = default_data
    upload_dir = st.session_state.get('uploaded_data_dir', '') or upload_dir_default
    if upload_dir and os.path.isdir(upload_dir) and any(_find_files(upload_dir, MZML_EXT + FASTA_EXT)):
        if not any(_find_files(data_dir, MZML_EXT + FASTA_EXT)):
            data_dir = upload_dir
    if not any(_find_files(data_dir, MZML_EXT + FASTA_EXT)):
        st.sidebar.caption('No FASTA or mzML in directory. Use the upload section below to add files.')
    elif st.session_state.get('uploaded_fasta') or st.session_state.get('uploaded_mzml'):
        st.sidebar.caption('FASTA and mzML loaded from upload.')

    def _rel_display(path: str) -> str:
        try:
            rel = os.path.relpath(path, data_dir)
            return rel if not rel.startswith('..') else path
        except ValueError:
            return path

    mzml_files = _find_files(data_dir, MZML_EXT)
    fasta_files = _find_files(data_dir, FASTA_EXT)
    csv_files = list(_find_files(data_dir, CSV_EXT))
    # Include output dir so pipeline outputs (Comet, Percolator, OpenMS) are discoverable
    output_dir_val = (output_dir.strip() if output_dir else '') or ''
    if output_dir_val and os.path.isdir(output_dir_val) and output_dir_val != data_dir:
        for p in _find_files(output_dir_val, CSV_EXT):
            if p not in csv_files:
                csv_files.append(p)
        csv_files = sorted(csv_files)
    mzml_path = mzml_files[0] if mzml_files else None
    fasta_path = fasta_files[0] if fasta_files else None
    # Override: when session has both uploads, use their paths directly (Streamlit Cloud)
    if st.session_state.get('uploaded_fasta') and st.session_state.get('uploaded_mzml'):
        ud = st.session_state.get('uploaded_data_dir', '') or upload_dir_default
        fname, fval = st.session_state.uploaded_fasta
        mname, mval = st.session_state.uploaded_mzml
        fp = fval if isinstance(fval, str) else os.path.join(ud, fname)
        mp = mval if isinstance(mval, str) else os.path.join(ud, mname)
        if ud and os.path.isdir(ud) and os.path.exists(fp) and os.path.exists(mp):
            fasta_path = fp
            mzml_path = mp
    csv_default_idx = 0
    last_out_preview = st.session_state.get('last_step_output')
    if csv_files:
        for i, p in enumerate(csv_files):
            if os.path.basename(p).endswith('_comet_perc_openMS.csv'):
                csv_default_idx = i
                break
        # Prefer last step output when it exists in the list (e.g. Percolator just ran)
        if last_out_preview and os.path.exists(last_out_preview):
            norm_last = os.path.normpath(os.path.abspath(last_out_preview))
            for i, p in enumerate(csv_files):
                if os.path.normpath(os.path.abspath(p)) == norm_last:
                    csv_default_idx = i
                    break
    csv_path = csv_files[csv_default_idx] if csv_files else None

    params_files = _find_files(data_dir, ('.params',))
    default_params = os.path.join(_SCRIPT_DIR, 'comet.params.new')
    params_path = default_params if os.path.exists(default_params) else (params_files[0] if params_files else None)

    with st.sidebar.expander('Advanced', expanded=False, icon='▶'):
        comet_exe_path = st.text_input('Comet exe', _default_comet_exe(),
                                       help='Path to comet.exe (build with `make` in project root)',
                                       autocomplete='off')

    # Run pipeline scripts (each outputs CSV + plots)
    def _build_env_and_cmd(cmd: list, needs_visualization: bool) -> tuple[dict, list]:
        """Build env and final cmd for subprocess. Returns (env, cmd)."""
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = _PYTHONUNBUFFERED
        env['PYTHONIOENCODING'] = 'utf-8'
        proj = os.path.abspath(_SCRIPT_DIR)
        env['PYTHONPATH'] = proj + (os.pathsep + env.get('PYTHONPATH', '')) if env.get('PYTHONPATH') else proj
        if sys.platform == 'darwin':
            conda_prefix = env.get('CONDA_PREFIX')
            if not conda_prefix and sys.executable:
                conda_prefix = os.path.dirname(os.path.dirname(os.path.abspath(sys.executable)))
            if conda_prefix:
                lib_dir = os.path.join(conda_prefix, 'lib')
                if os.path.isdir(lib_dir):
                    env['DYLD_LIBRARY_PATH'] = lib_dir
        if needs_visualization:
            script_path = os.path.abspath(cmd[1])
            args = cmd[2:]
            wrapper = (
                f"import sys\n"
                f"sys.path.insert(0, {repr(proj)})\n"
                f"sys.argv = {repr([script_path] + args)}\n"
                f"import runpy\n"
                f"runpy.run_path({repr(script_path)}, run_name='__main__')\n"
            )
            run_cmd = [sys.executable, '-u', '-c', wrapper]
        else:
            run_cmd = [sys.executable, '-u'] + cmd[1:]
        return env, run_cmd

    def _step_output_path(input_path: str, out_dir: str, out_suffix: str) -> str:
        """Compute expected output path for a pipeline step."""
        base = os.path.splitext(os.path.basename(input_path))[0]
        return os.path.join(out_dir, base + out_suffix)

    def _queue_run(cmd: list, step_name: str, out_suffix: str, needs_visualization: bool = False,
                   input_path: str = None, output_path: str = None, run_cwd: str = None,
                   comet_rename: tuple = None) -> None:
        """Queue a script run for execution at end of page (enables live streaming)."""
        _clear_chromatograms_cache()
        out_path = output_path or (input_path and _step_output_path(input_path, out_dir, out_suffix))
        st.session_state.pending_run = (cmd, step_name, out_suffix, needs_visualization, out_path, run_cwd, comet_rename)

    def _execute_pending_run(stream_container=None) -> bool:
        """Execute queued command now. Returns True when a run was executed."""
        pending = st.session_state.pending_run
        if not pending:
            return False
        parts = pending if isinstance(pending, (list, tuple)) else (pending + (None, None, None))
        cmd, step_name, out_suffix, needs_visualization = parts[:4]
        output_path = parts[4] if len(parts) > 4 else None
        run_cwd = parts[5] if len(parts) > 5 else None
        comet_rename = parts[6] if len(parts) > 6 else None
        st.session_state.pending_run = None
        env, run_cmd = _build_env_and_cmd(cmd, needs_visualization)
        proj = os.path.abspath(_SCRIPT_DIR)
        work_dir = run_cwd if run_cwd else proj
        run_started_ts = time.time()
        pre_output_mtime = os.path.getmtime(output_path) if output_path and os.path.exists(output_path) else None
        accumulated = []
        returncode = -1
        parent = stream_container if stream_container is not None else st.container()
        with parent:
            run_state_box = st.empty()
            run_state_box.info(f"Running **{step_name}**… Live output appears below.")
            with st.status(f"Running {step_name}…", expanded=True) as status:
                st.caption(f"Command: {' '.join(run_cmd)}")
                stream_box = st.empty()
                try:
                    proc = subprocess.Popen(
                        run_cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        encoding='utf-8',
                        errors='replace',
                        bufsize=1,
                        cwd=work_dir,
                        env=env,
                    )
                    for line in iter(proc.stdout.readline, ''):
                        ln = line.rstrip()
                        if ln:
                            accumulated.append(ln)
                            stream_box.code('\n'.join(accumulated), language=None)
                    returncode = proc.wait()
                except Exception as e:
                    accumulated.append(f"[Error running {step_name}: {e}]")
                out_text = '\n'.join(accumulated).strip()
                if not out_text and returncode != 0:
                    out_text = f"Process exited with code {returncode}. No output captured.\n\nTry running from terminal:\n  {' '.join(run_cmd)}"
                if not out_text:
                    out_text = "(No output)" if returncode == 0 else f"Exit code {returncode}"
                st.session_state.script_output = out_text
                st.session_state.script_step = step_name
                if returncode == 0:
                    status.update(label=f"{step_name} complete", state="complete")
                    run_state_box.success(f"✅ {step_name} complete. Output is shown in this tab terminal.")
                else:
                    status.update(label=f"{step_name} failed", state="error")
                    run_state_box.error(f"❌ {step_name} failed. See terminal output in this tab.")
        if returncode == 0:
            if comet_rename and step_name == 'Comet':
                mzml_base, fasta_base = comet_rename
                src_csv = os.path.join(work_dir, mzml_base + '.csv')
                src_pin = os.path.join(work_dir, mzml_base + '.pin')
                dst_csv = os.path.join(work_dir, fasta_base + '_comet.csv')
                dst_pin = os.path.join(work_dir, fasta_base + '_comet.pin')
                for src, dst in [(src_csv, dst_csv), (src_pin, dst_pin)]:
                    if os.path.exists(src):
                        try:
                            if os.path.exists(dst):
                                os.remove(dst)
                            os.rename(src, dst)
                        except Exception as e:
                            st.sidebar.warning(f'Could not rename {os.path.basename(src)}: {e}')
                output_path = dst_csv if os.path.exists(dst_csv) else output_path
            fresh_output_path = None
            # Prefer the expected output path when it was written in this run.
            if output_path and os.path.exists(output_path):
                try:
                    out_mtime = os.path.getmtime(output_path)
                    # Accept if mtime moved forward or if file timestamp is within this run window.
                    if (pre_output_mtime is not None and out_mtime > pre_output_mtime) or (out_mtime >= run_started_ts - 2):
                        fresh_output_path = output_path
                except Exception:
                    pass
            # Fallback: choose newest file in output dir that matches the step suffix and was written this run.
            if not fresh_output_path and out_suffix:
                search_dir = os.path.dirname(output_path) if output_path else work_dir
                try:
                    candidates = []
                    for fn in os.listdir(search_dir):
                        if fn.endswith(out_suffix):
                            fp = os.path.join(search_dir, fn)
                            if os.path.isfile(fp):
                                mt = os.path.getmtime(fp)
                                if mt >= run_started_ts - 2:
                                    candidates.append((mt, fp))
                    if candidates:
                        candidates.sort(key=lambda x: x[0], reverse=True)
                        fresh_output_path = candidates[0][1]
                except Exception:
                    pass

            if fresh_output_path and os.path.exists(fresh_output_path):
                st.session_state.last_step_output = fresh_output_path
                st.session_state.last_step_name = step_name
                st.session_state.last_step_run_started_ts = run_started_ts
            else:
                st.session_state.last_step_output = None
                st.session_state.last_step_name = step_name
                st.session_state.last_step_run_started_ts = run_started_ts
                st.sidebar.warning(
                    f'{step_name} completed, but no fresh output file timestamp was detected for {out_suffix}.'
                )
            load_csv.clear()
            _clear_chromatograms_cache()
            gc.collect()
            st.sidebar.success(f'{step_name} complete. Output: ...{out_suffix}')
            # Keep user on current tab; avoid full-page rerun that resets tabs.
        else:
            st.sidebar.error(f'{step_name} failed.')
        return True

    out_dir = (output_dir.strip() if output_dir else '') or (
        os.path.dirname(csv_path) if csv_path else
        (upload_dir if upload_dir and data_dir == upload_dir else data_dir)
    )
    os.makedirs(out_dir, exist_ok=True)
    for sub in ['diagnostics', 'extraction', 'rejected_extraction', 'envelope', 'envelope_rejected', 'sequence_coverage', 'channel_assignment', 'fragmentation_source']:
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

    # Resolve chained inputs. conf_base = root name (e.g. comet_frags_perc_openMS).
    base_for_resolve = os.path.splitext(os.path.basename(csv_path))[0] if csv_path else ''
    for strip in ['_prefilter_extraction_envelope_significance_sequence', '_prefilter_extraction_envelope_significance',
                  '_prefilter_extraction_envelope', '_prefilter_extraction', '_prefilter',
                  '_prefilter_extraction_test_envelope_significance_sequence', '_prefilter_extraction_test_envelope_significance',
                  '_prefilter_extraction_test_envelope', '_prefilter_extraction_test',
                  '_confidence_accuracy_extraction_envelope_significance', '_confidence_accuracy_extraction_envelope',
                  '_confidence_accuracy_extraction', '_confidence_accuracy', '_confidence']:
        if base_for_resolve.endswith(strip):
            conf_base = base_for_resolve[:-len(strip)]
            break
    else:
        conf_base = base_for_resolve
    extraction_test = st.session_state.get('extraction_test', True)
    step5_input = _resolve_step_input_from_conf_base(conf_base, out_dir, ['_prefilter', '_confidence'], csv_path) if csv_path else None
    step6_suffixes = (['_prefilter_extraction_test', '_confidence_accuracy_extraction_test', '_confidence_extraction_test'] +
                      ['_prefilter_extraction', '_confidence_accuracy_extraction', '_confidence_extraction']) if extraction_test else (['_prefilter_extraction', '_confidence_accuracy_extraction', '_confidence_extraction'] +
                      ['_prefilter_extraction_test', '_confidence_accuracy_extraction_test', '_confidence_extraction_test'])
    step6_input = _resolve_step_input_from_conf_base(conf_base, out_dir, step6_suffixes, csv_path) if csv_path else None
    step7_suffixes = (['_prefilter_extraction_test_envelope', '_confidence_accuracy_extraction_test_envelope', '_confidence_extraction_test_envelope'] +
                      ['_prefilter_extraction_envelope', '_confidence_accuracy_extraction_envelope', '_confidence_extraction_envelope']) if extraction_test else (['_prefilter_extraction_envelope', '_confidence_accuracy_extraction_envelope', '_confidence_extraction_envelope'] +
                      ['_prefilter_extraction_test_envelope', '_confidence_accuracy_extraction_test_envelope', '_confidence_extraction_test_envelope'])
    step7_input = _resolve_step_input_from_conf_base(conf_base, out_dir, step7_suffixes, csv_path) if csv_path else None
    # Significance runs on envelope or extraction (needs RT windows)
    sig_frag_input = (step7_input if step7_input and os.path.exists(step7_input) else step6_input) if csv_path else None

    st.sidebar.divider()

    # Prefer last step output (auto-load after pipeline run) over dropdown selection
    last_out = st.session_state.get('last_step_output')
    if last_out and os.path.exists(last_out):
        csv_path = last_out

    # Intentionally do not show sidebar "Loaded/Auto-loaded CSV" status;
    # inputs are FASTA + mzML and each step writes its own output CSV.
    pipeline_mode = (csv_path is None or not os.path.exists(csv_path)) and mzml_path and fasta_path
    if csv_path is None or not os.path.exists(csv_path):
        if not pipeline_mode:
            st.session_state.pending_run = None
            st.error('Upload **FASTA** and **mzML** in the sidebar to get started.')
            st.info('That\'s all you need. The pipeline (Comet → Percolator → OpenMS) will generate the rest.')
            # Still show tabs so user sees the structure
            df = pd.DataFrame()
            n_orig = 0
            n_final = 0
            base = ''
            step_conf_base = ''
            step_counts = []
            has_extraction = False
            has_envelope = False
            has_sig_frags = False
            has_coelution = False
            has_confidence = False
            qcol = None
            pepcol = None
            df_orig = pd.DataFrame()
        else:
            df = pd.DataFrame()
            n_orig = 0
            n_final = 0
            base = ''
            step_conf_base = ''
            step_counts = []
            has_extraction = False
            has_envelope = False
            has_sig_frags = False
            has_coelution = False
            has_confidence = False
            qcol = None
            pepcol = None
            df_orig = pd.DataFrame()
    else:
        df = load_csv(csv_path)
        # Optionally merge shape_corr from chromatogram_metrics if missing
        metrics_path = os.path.join(data_dir, 'dataframes', 'chromatogram_metrics_all.csv')
        if 'shape_corr' not in df.columns and os.path.exists(metrics_path):
            try:
                metrics = load_csv(metrics_path)
                if 'shape_corr' in metrics.columns and 'peptide_key' in metrics.columns and 'plain_peptide' in df.columns and 'charge' in df.columns:
                    def _key(r):
                        m = str(r.get('modifications', '-')).strip() if 'modifications' in df.columns else '-'
                        if not m or m.lower() == 'nan': m = '-'
                        return f"{r['plain_peptide']}_{r['charge']}_{m}"
                    df['_pk'] = df.apply(_key, axis=1)
                    msub = metrics[['peptide_key', 'shape_corr']].drop_duplicates('peptide_key')
                    df = df.merge(msub, left_on='_pk', right_on='peptide_key', how='left', suffixes=('', '_m'))
                    if 'shape_corr' not in df.columns:
                        df['shape_corr'] = df.get('shape_corr_m', pd.Series(dtype=float))
                    df = df.drop(columns=['_pk', 'peptide_key'], errors='ignore')
            except Exception as e:
                st.sidebar.warning(f'Could not merge chromatogram_metrics: {e}')
        n_orig = len(df)
        df_orig = df.copy()  # Keep for diagnostic tabs (before any filtering)
        base = os.path.splitext(os.path.basename(csv_path))[0]
        # Detect available data
        has_extraction = 'apex_intensity' in df.columns or 'total_area' in df.columns
        has_envelope = 'm0_gt_m1_gt_m2' in df.columns or 'envelope_ok' in df.columns
        has_sig_frags = (SIGNIFICANT_FRAGS_COL in df.columns)
        has_coelution = 'coelution_score' in df.columns
        qcol = _resolve_qcol(df)
        pepcol = _resolve_pepcol(df)
        has_confidence = bool(qcol or pepcol)

    # ─── Create tabs and run Metrics first (defines filter variables) ──
    st.markdown(
        '<div id="select-peptide-targets-section" style="margin:0;padding:0 0 0.5rem 0;font-size:1.25rem;font-weight:600">Select Peptide Targets</div>',
        unsafe_allow_html=True
    )
    # conf_base for step_outputs (root name)
    for strip in ['_prefilter_extraction_envelope_significance_sequence', '_prefilter_extraction_envelope_significance',
                  '_prefilter_extraction_envelope', '_prefilter_extraction', '_prefilter',
                  '_prefilter_extraction_test_envelope_significance_sequence', '_prefilter_extraction_test_envelope_significance',
                  '_prefilter_extraction_test_envelope', '_prefilter_extraction_test',
                  '_confidence_accuracy_extraction_envelope_significance', '_confidence_accuracy_extraction_envelope',
                  '_confidence_accuracy_extraction', '_confidence_accuracy', '_confidence',
                  '_extraction', '_extraction_test', '_channels', '_D_MZ', '_SF']:
        if base.endswith(strip):
            step_conf_base = base[:-len(strip)]
            break
    else:
        step_conf_base = base
    def _resolve_output(*candidates):
        for c in candidates:
            p = os.path.join(out_dir, step_conf_base + c + '.csv')
            if os.path.exists(p):
                return p
        return os.path.join(out_dir, step_conf_base + candidates[0] + '.csv')
    extraction_candidates = ('_prefilter_extraction_test', '_confidence_accuracy_extraction_test', '_confidence_extraction_test', '_extraction_test',
                            '_prefilter_extraction', '_confidence_accuracy_extraction', '_confidence_extraction', '_extraction') if extraction_test else (
                            '_prefilter_extraction', '_confidence_accuracy_extraction', '_confidence_extraction', '_extraction',
                            '_prefilter_extraction_test', '_confidence_accuracy_extraction_test', '_confidence_extraction_test', '_extraction_test')
    step_outputs = {
        'prefilter': _resolve_output('_prefilter', '_confidence'),
        'extraction': _resolve_output(*extraction_candidates),
        'envelope': _resolve_output('_prefilter_extraction_test_envelope', '_confidence_accuracy_extraction_test_envelope', '_confidence_extraction_test_envelope',
                                   '_prefilter_extraction_envelope', '_confidence_accuracy_extraction_envelope', '_confidence_extraction_envelope') if extraction_test else
                   _resolve_output('_prefilter_extraction_envelope', '_confidence_accuracy_extraction_envelope', '_confidence_extraction_envelope',
                                   '_prefilter_extraction_test_envelope', '_confidence_accuracy_extraction_test_envelope', '_confidence_extraction_test_envelope'),
        'significance': _resolve_output('_prefilter_extraction_test_envelope_significance', '_confidence_accuracy_extraction_test_envelope_significance',
                                        '_confidence_extraction_test_envelope_significance', '_confidence_extraction_test_significance', '_confidence_accuracy_extraction_test_significance',
                                        '_prefilter_extraction_envelope_significance', '_confidence_accuracy_extraction_envelope_significance',
                                        '_confidence_extraction_envelope_significance', '_confidence_extraction_significance', '_confidence_accuracy_extraction_significance') if extraction_test else
                       _resolve_output('_prefilter_extraction_envelope_significance', '_confidence_accuracy_extraction_envelope_significance',
                                        '_confidence_extraction_envelope_significance', '_confidence_extraction_significance', '_confidence_accuracy_extraction_significance',
                                        '_prefilter_extraction_test_envelope_significance', '_confidence_accuracy_extraction_test_envelope_significance',
                                        '_confidence_extraction_test_envelope_significance', '_confidence_extraction_test_significance', '_confidence_accuracy_extraction_test_significance'),
        'sequence': _resolve_output('_prefilter_extraction_test_envelope_significance_sequence', '_confidence_extraction_test_envelope_significance_sequence',
                                   '_prefilter_extraction_envelope_significance_sequence', '_confidence_extraction_envelope_significance_sequence', '_significance_sequence') if extraction_test else
                   _resolve_output('_prefilter_extraction_envelope_significance_sequence', '_confidence_extraction_envelope_significance_sequence', '_significance_sequence',
                                   '_prefilter_extraction_test_envelope_significance_sequence', '_confidence_extraction_test_envelope_significance_sequence'),
        'd_mz': _resolve_output('_D_MZ'),
        'sf': _resolve_output('_SF'),
        'channels': _resolve_output('_channels'),
    }
    def _resolve_preview_output(step_name: str, fallback_path: str):
        """
        Use the just-generated output for this step when available; otherwise fallback.
        Prevents stale previews after a run.
        """
        if st.session_state.get('last_step_name') == step_name:
            last_out = st.session_state.get('last_step_output')
            if last_out and os.path.exists(last_out):
                return last_out
            return None
        return fallback_path
    # Pipeline base for Comet/Percolator/OpenMS outputs (strip suffixes from loaded CSV, or use fasta when in pipeline mode)
    pipeline_base = base
    for strip in ['_comet_perc_openMS', '_comet_perc', '_comet']:
        if pipeline_base.endswith(strip):
            pipeline_base = pipeline_base[:-len(strip)]
            break
    if not pipeline_base and fasta_path:
        pipeline_base = os.path.splitext(os.path.basename(fasta_path))[0]
    pipeline_outputs = {
        'comet': os.path.join(out_dir, pipeline_base + '_comet.csv') if pipeline_base else '',
        'percolator': os.path.join(out_dir, pipeline_base + '_comet_perc.csv') if pipeline_base else '',
        'openms': os.path.join(out_dir, pipeline_base + '_comet_perc_openMS.csv') if pipeline_base else '',
    }
    green_tabs_css = []
    for name, path in [('comet', 2), ('percolator', 3), ('openms', 4), ('prefilter', 7), ('extraction', 8), ('envelope', 9), ('significance', 10), ('sequence', 11), ('d_mz', 12), ('sf', 13), ('channels', 14)]:
        out = pipeline_outputs.get(name) or step_outputs.get(name, '')
        if out and os.path.exists(out):
            green_tabs_css.append(f'[data-testid="stTabs"] [role="tab"]:nth-child({path}) {{ color: #84cc16 !important; }}')
    # Checkmarks in Metrics tab — blue to match app theme (same as buttons)
    # accent-color + span/svg: target both native checkbox and Streamlit's custom visual
    checkbox_css = [
        '[data-testid="stTabs"] [role="tabpanel"]:nth-of-type(5) input[type="checkbox"] { accent-color: #002755 !important; }',
        '[data-testid="stTabs"] [role="tabpanel"]:nth-of-type(5) [data-testid="stCheckbox"]:has(input:checked) span { background-color: #002755 !important; }',
        '[data-testid="stTabs"] [role="tabpanel"]:nth-of-type(5) [data-testid="stCheckbox"]:has(input:checked) svg, [data-testid="stTabs"] [role="tabpanel"]:nth-of-type(5) [data-testid="stCheckbox"]:has(input:checked) path { fill: #002755 !important; stroke: #002755 !important; }',
    ]
    all_css = green_tabs_css + checkbox_css
    if all_css:
        st.markdown(f'<style>\n' + '\n'.join(all_css) + '\n</style>', unsafe_allow_html=True)

    # Auto-load banner — visible above all tabs when CSV was loaded from previous step
    last_out = st.session_state.get('last_step_output')
    last_name = st.session_state.get('last_step_name') or 'previous step'
    _banner_match = last_out and csv_path and os.path.normpath(os.path.abspath(csv_path)) == os.path.normpath(os.path.abspath(last_out))
    if _banner_match and os.path.exists(csv_path):
        bc, bbtn = st.columns([6, 1])
        with bc:
            st.info(f"📁 **Auto-loaded:** {os.path.basename(csv_path)} from **{last_name}** step. Shown in all tabs below.")
        with bbtn:
            if st.button('Clear auto-load', key='clear_auto_load_banner', help='Revert to CSV from sidebar dropdown'):
                st.session_state.last_step_output = None
                st.session_state.last_step_name = None
                st.rerun()

    filter_steps = ['Summary', 'Comet', 'Percolator', 'OpenMS', 'Metrics', 'Compare', 'Prefilter', 'Extraction', 'Envelope', 'Significance', 'Sequence', 'D_MZ', 'SF', 'Channels', 'Method']
    filter_tabs = st.tabs(filter_steps)

    # Show banner when a pipeline step is queued (fallback runner executes at end of script)
    if st.session_state.pending_run:
        pending_step = st.session_state.pending_run[1] if isinstance(st.session_state.pending_run, (list, tuple)) else 'step'
        st.info(f"⏳ **Running {pending_step}…** Live output appears in the terminal near the run button.")

    def _render_terminal():
        term_header = f"Terminal — {st.session_state.script_step}" if st.session_state.script_step else "Terminal"
        with st.expander(term_header, expanded=bool(st.session_state.script_output), icon='▶'):
            if st.session_state.script_output:
                st.code(st.session_state.script_output, language=None)
            else:
                st.caption("Run a pipeline step (Step 1–8) to see the latest terminal output here.")

    # Metrics tab — run first so variables are defined before apply filters
    with filter_tabs[4]:
            # Pipeline order: Prefilter → Extraction → Significance. Compact layout, no expanders.
            # Prefilter (Step 4)
            st.caption('Prefilter')
            pf1, pf2, pf3, pf4, pf5 = st.columns(5)
            with pf1:
                enable_conf = st.checkbox('Prefilter (Proline, Mods, Q/PEP)', True, key='enable_conf')
            with pf2:
                use_qpep = st.checkbox('Use Q/PEP', False, key='use_qpep')
            with pf3:
                q_thresh = st.slider('Q-value', 0.001, 0.2, 0.05, 0.001, key='q_thresh', disabled=not use_qpep)
            with pf4:
                pep_thresh = st.slider('PEP', 0.001, 0.2, 0.05, 0.001, key='pep_thresh', disabled=not use_qpep)
            with pf5:
                reject_proline = st.checkbox('Proline', True, key='reject_pro')
            pf6a, pf6, pf7, pf8, pf9 = st.columns(5)
            with pf6a:
                reject_mods = st.checkbox('Modifications', True, key='reject_mods')
            with pf6:
                enable_xcorr_filter = st.checkbox('XCorr filter', False, key='enable_xcorr_filter')
            with pf7:
                xcorr_min = st.slider('Min XCorr', 0.0, 10.0, 2.0, 0.1, key='xcorr_min')
            with pf8:
                enable_sp_filter = st.checkbox('Sp filter', False, key='enable_sp_filter')
            with pf9:
                sp_min = st.number_input('Min Sp', 0.0, 1000.0, 50.0, 1.0, key='sp_min')
            # Extraction (Step 6)
            st.caption('Extraction')
            ex1, ex2, ex3, ex4 = st.columns(4)
            with ex1:
                enable_envelope = st.checkbox('Envelope (M0, M+1, M+2 present; M0>M+1 OR M+1>M+2)', True, key='enable_env')
            with ex2:
                enable_apex = st.checkbox('Apex', True, key='enable_apex')
                apex_min = st.number_input('Min apex', 1e4, 1e8, 1e5, 1e4, format='%.0e', key='apex_min')
            with ex3:
                enable_shape = st.checkbox('Shape', True, key='enable_shape')
                shape_min = st.slider('Min shape', 0.0, 1.0, 0.8, 0.05, key='shape_min')
            with ex4:
                enable_coelution = st.checkbox('Coelution', True, key='enable_coel')
                coelution_min = st.slider('Min coel', 0.0, 1.0, 0.8, 0.05, key='coel_min')
            # Significance (Step 5)
            st.caption('Significance')
            sf1, sf2, sf3, sf4 = st.columns(4)
            with sf1:
                enable_sig_frags = st.checkbox('Sig fragments (5ppm, min sig frags, 1% max)', True, key='enable_sig')
            with sf2:
                ppm_thresh = st.slider('5ppm', 1.0, 20.0, 5.0, 0.5, key='ppm_thresh', help='PPM tolerance for fragment matching')
            with sf3:
                min_sig_frag_count = st.number_input('Min sig frags', 0, 50, 3, 1, key='min_sig')
            with sf4:
                sig_frac_pct = st.slider('1% max peak', 0.1, 5.0, 1.0, 0.1, key='sig_frac', help='Min % of max MS2 for fragment (noise filter)')
            # Extraction run options
            st.caption('Extraction run')
            opt1, opt2, _ = st.columns(3)
            with opt1:
                extraction_test = st.checkbox('Test (30 pep)', True, key='extraction_test')
            with opt2:
                extraction_extract_only = st.checkbox('Extract only (no plots)', False, key='extraction_extract_only',
                                                     help='Skip individual chromatogram plots; CSV + traces saved for plotting later')

    # ─── Apply filters in pipeline order ────────────────────────────────────
    df_filtered = df.copy()
    n_prev = len(df_filtered)
    step_counts = []  # (step_name, passed, total) for summary

    # Step 4: Prefilter (Proline, Modifications, Q/PEP; PPM via Step 5 script)
    if enable_conf:
        # Proline: peptide starts/ends with P, or prev_aa/next_aa is P
        if reject_proline:
            mask_pro = pd.Series(True, index=df_filtered.index)
            if 'plain_peptide' in df_filtered.columns:
                def _peptide_has_proline_terminus(seq):
                    if pd.isna(seq) or not seq:
                        return True
                    s = str(seq).strip()
                    return s and (s[0].upper() == 'P' or s[-1].upper() == 'P')
                mask_pro = mask_pro & ~df_filtered['plain_peptide'].apply(_peptide_has_proline_terminus)
            if 'prev_aa' in df_filtered.columns and 'next_aa' in df_filtered.columns:
                def _flank_is_proline(row):
                    prev = str(row.get('prev_aa', '')).strip().upper()
                    next_ = str(row.get('next_aa', '')).strip().upper()
                    return prev == 'P' or next_ == 'P'
                mask_pro = mask_pro & ~df_filtered.apply(_flank_is_proline, axis=1)
            df_filtered = df_filtered[mask_pro].copy()
        # Modifications
        if reject_mods and 'modifications' in df_filtered.columns:
            mask_mods = ~df_filtered['modifications'].apply(_has_modifications)
            df_filtered = df_filtered[mask_mods].copy()
        # Q-value / PEP
        if use_qpep and has_confidence and (qcol or pepcol):
            mask_conf = df_filtered.apply(lambda r: _pass_confidence(r, df_filtered, qcol, pepcol, q_thresh, pep_thresh, enabled=use_qpep), axis=1)
            df_filtered = df_filtered[mask_conf].copy()
        # Optional XCorr / Sp filters
        xcorr_col_runtime = _resolve_col(df_filtered, XCORR_COLS)
        if enable_xcorr_filter and xcorr_col_runtime and xcorr_col_runtime in df_filtered.columns:
            mask_xcorr = pd.to_numeric(df_filtered[xcorr_col_runtime], errors='coerce') >= xcorr_min
            mask_xcorr = mask_xcorr | pd.isna(df_filtered[xcorr_col_runtime])
            df_filtered = df_filtered[mask_xcorr].copy()
        sp_col_runtime = _resolve_col(df_filtered, SP_COLS)
        if enable_sp_filter and sp_col_runtime and sp_col_runtime in df_filtered.columns:
            mask_sp = pd.to_numeric(df_filtered[sp_col_runtime], errors='coerce') >= sp_min
            mask_sp = mask_sp | pd.isna(df_filtered[sp_col_runtime])
            df_filtered = df_filtered[mask_sp].copy()
    step_counts.append(('Prefilter', len(df_filtered), n_prev))
    n_prev = len(df_filtered)

    # Step 6: Extraction (envelope, apex >=10^5, coelution, shape, S/N 1% max)
    if enable_envelope and has_envelope:
        def _pass_env(row):
            m0 = row.get('m0_gt_m1_gt_m2')
            env = row.get('envelope_ok')
            if pd.notna(m0):
                return _to_bool(m0)
            if pd.notna(env):
                return _to_bool(env)
            return False
        mask_env = df_filtered.apply(_pass_env, axis=1)
        df_filtered = df_filtered[mask_env].copy()
    if enable_apex and has_extraction and 'apex_intensity' in df_filtered.columns:
        mask_apex = pd.to_numeric(df_filtered['apex_intensity'], errors='coerce') >= apex_min
        mask_apex = mask_apex | pd.isna(df_filtered['apex_intensity'])
        df_filtered = df_filtered[mask_apex].copy()
    if enable_shape and 'shape_corr' in df_filtered.columns:
        mask_shape = pd.to_numeric(df_filtered['shape_corr'], errors='coerce') >= shape_min
        mask_shape = mask_shape | pd.isna(df_filtered['shape_corr'])
        df_filtered = df_filtered[mask_shape].copy()
    if enable_coelution and has_coelution:
        mask_coel = pd.to_numeric(df_filtered['coelution_score'], errors='coerce') >= coelution_min
        mask_coel = mask_coel | pd.isna(df_filtered['coelution_score'])
        df_filtered = df_filtered[mask_coel].copy()
    step_counts.append(('Extraction', len(df_filtered), n_prev))
    n_prev = len(df_filtered)

    # Step 5 (after extraction): Significant fragmentation (5ppm, min sig frags, 1% max peak)
    if enable_sig_frags and has_sig_frags:
        def _count_sig(row):
            # Use significant_frags (pass all 3 criteria) only.
            val = row.get(SIGNIFICANT_FRAGS_COL)
            if pd.isna(val) or val is None or str(val).strip() == '':
                return 0
            parts = [p.strip() for p in str(val).replace(',', ' ').split() if p.strip()]
            return len(parts)
        counts = df_filtered.apply(_count_sig, axis=1)
        effective_min = max(1, min_sig_frag_count)  # require at least 1 when enabled
        mask_sig = counts >= effective_min
        df_filtered = df_filtered[mask_sig].copy()
    step_counts.append(('Significance', len(df_filtered), n_prev))
    n_prev = len(df_filtered)

    n_final = len(df_filtered)
    df = df_filtered

    # Diagnostic helpers (use df_orig = data before any filtering)
    passed_all = set(df.index)  # rows that passed ALL filters — use for Pass/Fail color in all stage plots
    def _status_all(idx):
        return 'Pass' if idx in passed_all else 'Fail'
    def _pass_conf(r):
        return _pass_confidence(r, df_orig, qcol, pepcol, q_thresh, pep_thresh, enabled=use_qpep)
    def _pass_env(r):
        m0, env = r.get('m0_gt_m1_gt_m2'), r.get('envelope_ok')
        if pd.notna(m0): return _to_bool(m0)
        if pd.notna(env): return _to_bool(env)
        return False
    def _count_sig(r):
        val = r.get(SIGNIFICANT_FRAGS_COL)
        if pd.isna(val) or val is None or str(val).strip() == '': return 0
        return len([p for p in str(val).replace(',', ' ').split() if p.strip()])

    def _reorder_traces_smaller_on_top(fig, plot_df, color_col):
        """Put smaller category trace last so it's drawn on top."""
        if color_col not in plot_df.columns:
            return
        counts = plot_df[color_col].value_counts()
        if len(counts) >= 2:
            smaller, larger = counts.idxmin(), counts.idxmax()
            fig.data = sorted(fig.data, key=lambda t: (0 if getattr(t, 'name', None) == larger else 1))

    # Helper to make Plotly or matplotlib scatter (compact for grid layout)
    PLOT_HEIGHT = 280
    def _scatter_plotly(x, y, color=None, title='', xlabel='', ylabel='', log_x=False, log_y=False, hline=None, vline=None, height=None):
        try:
            import plotly.express as px
            h = height if height is not None else PLOT_HEIGHT
            d = {'x': x, 'y': y}
            if color is not None:
                d['color'] = color
            plot_df = pd.DataFrame(d)
            if color is not None:
                fig = px.scatter(plot_df, x='x', y='y', color='color', log_x=log_x, log_y=log_y,
                                 title=title, labels={'x': xlabel or 'x', 'y': ylabel or 'y', 'color': 'Status'},
                                 color_discrete_map={'Fail': 'red', 'Pass': 'dodgerblue'})
                _reorder_traces_smaller_on_top(fig, plot_df, 'color')
            else:
                fig = px.scatter(plot_df, x='x', y='y', log_x=log_x, log_y=log_y, title=title,
                                 labels={'x': xlabel or 'x', 'y': ylabel or 'y'})
            if hline is not None:
                fig.add_hline(y=hline, line_dash='dash', line_color='red')
            if vline is not None:
                fig.add_vline(x=vline, line_dash='dash', line_color='orange')
            fig.update_traces(marker=dict(opacity=0.5, line=dict(color='black', width=1)))
            fig.update_layout(template='plotly_dark', height=h, margin=dict(l=40, r=20, t=40, b=40))
            return fig
        except ImportError:
            return None

    def _plot_grid(items):
        """Render a list of (fig, key) in a 2-column grid. items: list of (fig or None, unique_key)."""
        for i in range(0, len(items), 2):
            pair = items[i:i+2]
            cols = st.columns(2)
            for j, (fig, _) in enumerate(pair):
                if fig is not None:
                    with cols[j]:
                        st.plotly_chart(fig, use_container_width=True)

    # ─── Rest of tabs (Summary, Prefilter, Extraction, etc.) ─────────────────
    with filter_tabs[0]:  # Summary
        if pipeline_mode:
            st.info('Run the **Comet** → **Percolator** → **OpenMS** pipeline (tabs 2–4) to create a CSV, then select it in the sidebar.')
        st.subheader('Filter summary')
        with st.expander('All tabs: pipeline steps and outputs', expanded=True, icon='▶'):
            # Build table of all tabs with status and description
            tab_info = [
                ('Summary', None, 'Overview and filter counts'),
                ('Comet', pipeline_outputs.get('comet', ''), 'Comet search on mzML + FASTA → _comet.csv'),
                ('Percolator', pipeline_outputs.get('percolator', ''), 'Q-values → _comet_perc.csv'),
                ('OpenMS', pipeline_outputs.get('openms', ''), 'MS1 RT windows → _comet_perc_openMS.csv'),
                ('Metrics', None, 'Filter thresholds (Q, PEP, apex, shape, coelution)'),
                ('Compare', None, 'Byonic vs Comet: overlap plots + byonic_only/comet_only CSVs'),
                ('Prefilter', step_outputs.get('prefilter', ''), 'PEP, Q, prolines, mods, ppm → _prefilter.csv'),
                ('Extraction', step_outputs.get('extraction', ''), 'Envelope, apex ≥10⁵, coelution, shape, S/N 1% max → _extraction.csv'),
                ('Envelope', step_outputs.get('envelope', ''), 'M0>M+1 OR M+1>M+2 → _envelope.csv'),
                ('Significance', step_outputs.get('significance', ''), '5ppm, min sig frags, 1% max peak → _significance.csv'),
                ('Sequence', step_outputs.get('sequence', ''), 'Unique peptides grid + protect_peptide → _sequence.csv'),
                ('D_MZ', step_outputs.get('d_mz', ''), 'Deuterated m/z target range (undeut + max deut) / 2 ± range → _D_MZ.csv'),
                ('SF', step_outputs.get('sf', ''), 'Source Fragmentation (y=mx+b) → _SF.csv'),
                ('Channels', step_outputs.get('channels', ''), 'Channel assignment → _channels.csv'),
                ('Method', None, 'Pipeline documentation'),
            ]
            step_rows = '| Tab | Status | Rows | Description |\n|-----|--------|------|-------------|\n'
            for tab_name, out_path, desc in tab_info:
                if out_path and os.path.exists(out_path):
                    try:
                        n = len(load_csv(out_path))
                        rows_str = str(n)
                    except Exception:
                        rows_str = '—'
                    status = '✓'
                else:
                    rows_str = '—'
                    status = '○' if out_path else '—'
                step_rows += f'| {tab_name} | {status} | {rows_str} | {desc} |\n'
            st.markdown(step_rows)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric('Original', n_orig)
        with col2:
            st.metric('After filters', n_final)
        with col3:
            pct = 100 * n_final / n_orig if n_orig > 0 else 0
            st.metric('Retention %', f'{pct:.1f}%')
        st.write('Columns:', ', '.join(df.columns[:20].tolist()) + ('...' if len(df.columns) > 20 else ''))

    with filter_tabs[1]:  # Comet
        st.subheader('Comet')
        st.caption('Run Comet search on mzML + FASTA. Outputs [fasta]_comet.csv and .pin for Percolator.')
        comet_run_live = st.container()
        run_comet = st.button('Run Comet', key='run_comet', help='Comet search only — outputs [fasta]_comet.csv and .pin')
        if run_comet and mzml_path and fasta_path and params_path and os.path.exists(params_path):
            fasta_base = os.path.splitext(os.path.basename(fasta_path))[0]
            comet_csv = os.path.join(out_dir, fasta_base + '_comet.csv')
            cmd = [sys.executable, os.path.join(_SCRIPT_DIR, 'run_comet_with_percolator.py'),
                   '--mzml', mzml_path, '--fasta', fasta_path, '--params', params_path,
                   '--skip-percolator', '--output-dir', out_dir, '--output-base', fasta_base]
            if comet_exe_path and comet_exe_path.strip():
                cmd.extend(['--comet-exe', comet_exe_path.strip()])
            _queue_run(cmd, 'Comet', '_comet.csv', run_cwd=out_dir, output_path=comet_csv)
            _execute_pending_run(stream_container=comet_run_live)
        elif run_comet and (not mzml_path or not fasta_path):
            st.warning('Select mzML and FASTA in the sidebar.')
        elif run_comet and (not params_path or not os.path.exists(params_path)):
            st.warning('Comet params file not found. Add comet.params.new to the project or a .params file to the data directory.')
        comet_csv_path = os.path.join(out_dir, os.path.splitext(os.path.basename(fasta_path))[0] + '_comet.csv') if fasta_path else None
        if st.session_state.get('last_step_name') == 'Comet':
            last_comet = st.session_state.get('last_step_output')
            if last_comet and os.path.exists(last_comet):
                comet_csv_path = last_comet
            else:
                comet_csv_path = None
                st.warning('No fresh Comet CSV detected from the latest run yet.')
        if comet_csv_path:
            _render_step_output_csv(comet_csv_path, 'comet_output', expanded=False)

    with filter_tabs[2]:  # Percolator
        st.subheader('Percolator')
        st.caption('Add Percolator q-values to [fasta]_comet.csv. Outputs [fasta]_comet_perc.csv.')
        perc_run_live = st.container()
        fasta_base = os.path.splitext(os.path.basename(fasta_path))[0] if fasta_path else ''
        comet_csv = os.path.join(out_dir, fasta_base + '_comet.csv') if fasta_base else None
        comet_pin = os.path.join(out_dir, fasta_base + '_comet.pin') if fasta_base else None
        perc_csv = os.path.join(out_dir, fasta_base + '_comet_perc.csv') if fasta_base else None
        has_comet_pair = comet_csv and comet_pin and os.path.exists(comet_csv) and os.path.exists(comet_pin)
        run_perc = st.button('Run Percolator (add q-values)', key='run_percolator', help='[fasta]_comet.csv → [fasta]_comet_perc.csv')
        if run_perc and has_comet_pair:
            cmd = [sys.executable, os.path.join(_SCRIPT_DIR, 'add_percolator_qvalues.py'),
                   '--csv', comet_csv, '--pin', comet_pin, '--output', perc_csv, '--fill-unmatched', '1.0']
            _queue_run(cmd, 'Percolator', '_comet_perc.csv', output_path=perc_csv)
            _execute_pending_run(stream_container=perc_run_live)
        elif run_perc and not has_comet_pair:
            st.warning('Run Comet first to generate [fasta]_comet.csv and .pin.')
        if perc_csv:
            if st.session_state.get('last_step_name') == 'Percolator':
                last_perc = st.session_state.get('last_step_output')
                if last_perc and os.path.exists(last_perc):
                    perc_csv = last_perc
                else:
                    perc_csv = None
                    st.warning('No fresh Percolator CSV detected from the latest run yet.')
        if perc_csv:
            _render_step_output_csv(perc_csv, 'perc_output', expanded=False)

    with filter_tabs[3]:  # OpenMS
        st.subheader('OpenMS')
        st.caption('Add MS1 retention times and intensities. Outputs [fasta]_comet_perc_openMS.csv.')
        openms_run_live = st.container()
        fasta_base = os.path.splitext(os.path.basename(fasta_path))[0] if fasta_path else ''
        comet_csv = os.path.join(out_dir, fasta_base + '_comet.csv') if fasta_base else None
        perc_csv = os.path.join(out_dir, fasta_base + '_comet_perc.csv') if fasta_base else None
        last_out = st.session_state.get('last_step_output') or ''
        openms_csv_in = None
        # Prefer the fresh Percolator output from this session when available.
        if st.session_state.get('last_step_name') == 'Percolator':
            last_perc = st.session_state.get('last_step_output')
            if last_perc and os.path.exists(last_perc):
                openms_csv_in = last_perc
        # Fallback to canonical percolator output path.
        if not openms_csv_in and perc_csv and os.path.exists(perc_csv):
            openms_csv_in = perc_csv
        # Safety check: do not run OpenMS with stale Percolator input older than Comet output.
        if openms_csv_in and comet_csv and os.path.exists(comet_csv):
            try:
                comet_mtime = os.path.getmtime(comet_csv)
                perc_mtime = os.path.getmtime(openms_csv_in)
                if perc_mtime + 1 < comet_mtime:
                    st.warning('Percolator CSV is older than the current Comet CSV. Re-run Percolator before OpenMS to avoid mixed-row outputs.')
                    openms_csv_in = None
            except Exception:
                pass
        openms_csv_out = (os.path.join(out_dir, fasta_base + '_comet_perc_openMS.csv') if fasta_base else
                          (os.path.join(out_dir, os.path.splitext(os.path.basename(openms_csv_in))[0] + '_openMS.csv')
                           if openms_csv_in else None))
        if openms_csv_in:
            st.caption(f'OpenMS input CSV: `{openms_csv_in}`')
        else:
            st.caption('OpenMS input CSV: (none detected)')
        run_openms = st.button('Run OpenMS (add MS1 data)', key='run_openms', help='[fasta]_comet_perc.csv → [fasta]_comet_perc_openMS.csv')
        run_seqpos = st.button('Add sequence positions', key='run_seqpos', help='add_sequence_positions.py')
        if run_openms and openms_csv_in and mzml_path and os.path.exists(mzml_path):
            cmd = [sys.executable, os.path.join(_SCRIPT_DIR, 'add_ms1_data_openms.py'), openms_csv_in, mzml_path, openms_csv_out]
            _queue_run(cmd, 'OpenMS MS1', '_comet_perc_openMS.csv', output_path=openms_csv_out)
            _execute_pending_run(stream_container=openms_run_live)
        elif run_openms and (not openms_csv_in or not mzml_path):
            if not openms_csv_in:
                st.warning('Run Comet and Percolator first.')
            if not mzml_path or not os.path.exists(mzml_path):
                st.warning('Select mzML in sidebar.')
        if run_seqpos and openms_csv_in and fasta_path and os.path.exists(fasta_path):
            seqpos_out = os.path.join(out_dir, os.path.splitext(os.path.basename(openms_csv_in))[0] + '_positions.csv')
            cmd = [sys.executable, os.path.join(_SCRIPT_DIR, 'add_sequence_positions.py'), openms_csv_in, fasta_path, seqpos_out]
            _queue_run(cmd, 'Sequence positions', '_positions.csv', output_path=seqpos_out)
            _execute_pending_run(stream_container=openms_run_live)
        elif run_seqpos and (not openms_csv_in or not fasta_path):
            st.warning('Run pipeline first. Select FASTA in sidebar.')
        if openms_csv_out:
            if st.session_state.get('last_step_name') == 'OpenMS MS1':
                last_openms = st.session_state.get('last_step_output')
                if last_openms and os.path.exists(last_openms):
                    openms_csv_out = last_openms
                else:
                    openms_csv_out = None
                    st.warning('No fresh OpenMS CSV detected from the latest run yet.')
        if openms_csv_out:
            _render_step_output_csv(openms_csv_out, 'openms_output', expanded=False)

    with filter_tabs[5]:  # Compare (Byonic vs Comet)
        st.subheader('Compare Byonic vs Comet')
        st.caption('Upload Byonic CSV(s) and cross-reference with the loaded Comet CSV. Output: overlap bar chart and two CSVs — Byonic-only (Comet missed) and Comet-only (Byonic missed).')
        byonic_upload = st.file_uploader('Byonic CSV(s)', type=['csv'], key='byonic_compare_upload', accept_multiple_files=True)
        compare_comet_path = csv_path if csv_path and os.path.exists(csv_path) else None
        run_compare = st.button('Run Compare', key='run_compare', help='Cross-reference Byonic and Comet peptides, generate plots and CSVs')
        if run_compare and byonic_upload and compare_comet_path:
            with st.status('Running Byonic vs Comet compare…', expanded=True) as cmp_status:
                try:
                    def _byonic_seq_to_plain(s):
                        if pd.isna(s) or s is None: return None
                        s = str(s).strip()
                        if not s: return None
                        if '.' in s:
                            parts = s.split('.')
                            if len(parts) >= 2: s = parts[1]
                            else: s = s.replace('.', '')
                        s = re.sub(r'\[\s*[+-]?\s*[\d.]+\s*\]', '', s)
                        return s if s else None
                    byonic_peptides = set()
                    for up in byonic_upload:
                        raw = up.getvalue().decode('utf-8', errors='replace')
                        first_nl = raw.find('\n')
                        header = raw[:first_nl].replace('\r\n', ' ').replace('\n', ' ')
                        rest = raw[first_nl + 1:].lstrip('\r\n')
                        try:
                            df_b = pd.read_csv(pd.io.common.StringIO(header + '\n' + rest), sep=',', engine='python', quotechar='"', on_bad_lines='warn')
                        except Exception:
                            df_b = pd.read_csv(pd.io.common.StringIO(raw), sep=',', engine='python', quotechar='"', on_bad_lines='warn')
                        seq_col = next((c for c in df_b.columns if 'sequence' in c.lower() and 'unformat' in c.lower()), None) or next((c for c in df_b.columns if 'sequence' in c.lower()), df_b.columns[2] if len(df_b.columns) > 2 else None)
                        z_col = next((c for c in df_b.columns if c.strip().lower() == 'z'), 'z' if 'z' in df_b.columns else None)
                        if seq_col is None or z_col is None: continue
                        for _, row in df_b.iterrows():
                            seq = _byonic_seq_to_plain(row.get(seq_col))
                            if not seq: continue
                            try: z = int(float(row[z_col]))
                            except (TypeError, ValueError): continue
                            byonic_peptides.add((seq, z))
                    comet_plain_charge = set()
                    comet_by_charge = {}
                    df_comet = load_csv(compare_comet_path)
                    if 'plain_peptide' not in df_comet.columns:
                        st.error('Comet CSV needs plain_peptide column.')
                    else:
                        for _, row in df_comet.iterrows():
                            p = row.get('plain_peptide')
                            if pd.isna(p) or not str(p).strip(): continue
                            p = str(p).strip()
                            try: z = int(float(row.get('charge', 1)))
                            except (TypeError, ValueError): continue
                            comet_plain_charge.add((p, z))
                            comet_by_charge.setdefault(z, set()).add(p)
                        byonic_only = [(s, z) for (s, z) in byonic_peptides if not any(
                            cs == s or s in cs or cs in s for cs in comet_by_charge.get(z, set()))]
                        comet_only = [(p, z) for (p, z) in comet_plain_charge if not any(
                            bs == p or p in bs or bs in p for (bs, bz) in byonic_peptides if bz == z)]
                        in_common = len(byonic_peptides) - len(byonic_only)
                        st.caption(f'Byonic: {len(byonic_peptides)} unique | Comet: {len(comet_plain_charge)} unique | In common: {in_common} | Byonic-only: {len(byonic_only)} | Comet-only: {len(comet_only)}')
                        import plotly.express as px
                        plot_df = pd.DataFrame({'Category': ['Byonic', 'Comet', 'In common', 'Byonic-only\n(Comet missed)', 'Comet-only\n(Byonic missed)'],
                                                'Count': [len(byonic_peptides), len(comet_plain_charge), in_common, len(byonic_only), len(comet_only)]})
                        fig = px.bar(plot_df, x='Category', y='Count', title='Byonic vs Comet peptide overlap')
                        fig.update_layout(template='plotly_dark', height=350, margin=dict(l=40, r=20, t=40, b=80), showlegend=False)
                        fig.update_xaxes(tickangle=-30)
                        st.plotly_chart(fig, use_container_width=True)
                        byonic_only_df = pd.DataFrame([{'plain_peptide': s, 'charge': z} for (s, z) in sorted(byonic_only)])
                        comet_only_df = pd.DataFrame([{'plain_peptide': p, 'charge': z} for (p, z) in sorted(comet_only)])
                        base = os.path.splitext(os.path.basename(compare_comet_path))[0]
                        byonic_out = os.path.join(out_dir, base + '_byonic_only_comet_missed.csv')
                        comet_out = os.path.join(out_dir, base + '_comet_only_byonic_missed.csv')
                        byonic_only_df.to_csv(byonic_out, index=False)
                        comet_only_df.to_csv(comet_out, index=False)
                        cmp_status.update(label='Compare complete', state='complete')
                        st.success(f'Saved: **{os.path.basename(byonic_out)}** ({len(byonic_only)} rows) and **{os.path.basename(comet_out)}** ({len(comet_only)} rows).')
                        st.download_button('Download Byonic-only (Comet missed)', byonic_only_df.to_csv(index=False), file_name=os.path.basename(byonic_out), mime='text/csv', key='dl_byonic_only')
                        st.download_button('Download Comet-only (Byonic missed)', comet_only_df.to_csv(index=False), file_name=os.path.basename(comet_out), mime='text/csv', key='dl_comet_only')
                except Exception as e:
                    cmp_status.update(label='Compare failed', state='error')
                    st.error(f'Compare error: {e}')
                    import traceback
                    st.code(traceback.format_exc())
        elif run_compare:
            if not byonic_upload:
                st.warning('Upload at least one Byonic CSV.')
            if not compare_comet_path:
                st.warning('Load a Comet CSV in the sidebar (or run pipeline first).')
        else:
            st.info('Upload Byonic CSV(s) and ensure a Comet CSV is loaded. Click **Run Compare** to cross-reference and generate overlap plots + CSVs.')

    with filter_tabs[6]:  # Prefilter (Proline, Mods, Q/PEP, PPM — all from CSV)
        prefilter_run_live = st.container()
        run_step4 = st.button('Run Prefilter', key='run_step4', help='Prefilter (Q, PEP, prolines, mods)')
        if run_step4 and csv_path and os.path.exists(csv_path):
            cmd = [sys.executable, os.path.join(_SCRIPT_DIR, 'workflow_scripts', 'filter_comet_frags_confidence.py'),
                   '--input', csv_path, '--output-dir', out_dir]
            if use_qpep:
                cmd.extend(['--use-qpep', '--q-threshold', str(q_thresh), '--pep-threshold', str(pep_thresh)])
            if enable_xcorr_filter:
                cmd.extend(['--use-xcorr', '--xcorr-min', str(xcorr_min)])
            if enable_sp_filter:
                cmd.extend(['--use-sp', '--sp-min', str(sp_min)])
            _queue_run(cmd, 'Prefilter', '_prefilter.csv', input_path=csv_path)
            _execute_pending_run(stream_container=prefilter_run_live)
        st.subheader('Prefilter')
        st.caption('PEP, Q-value, prolines, mods, ppm (MS1 peptide) — run Significant fragmentation for PPM of matched fragments (5ppm)')
        st.caption('Pass/Fail = passed all filters (see how failing rows distribute across metrics)')
        prefilter_preview = _resolve_preview_output('Prefilter', step_outputs['prefilter'])
        if prefilter_preview:
            _render_step_output_csv(prefilter_preview, 'step4_prefilter', expanded=(csv_path == prefilter_preview))
            prefilter_rejected = (
                prefilter_preview.replace('_prefilter.csv', '_prefilter_rejected.csv')
                if prefilter_preview.endswith('_prefilter.csv')
                else prefilter_preview.replace('.csv', '_rejected.csv')
            )
            if os.path.exists(prefilter_rejected):
                st.caption('Rejected rows (Prefilter)')
                _render_step_output_csv(prefilter_rejected, 'step4_prefilter_rejected', expanded=False)
        gen_prefilter = st.button('Generate plots', key='gen_prefilter_plots', help='Build diagnostic plots (Q-PEP, XCorr, Proline, Mods, PPM)')
        if gen_prefilter:
            _clear_chromatograms_cache()
            st.session_state['show_prefilter_plots'] = True
        if st.session_state.get('show_prefilter_plots', False):
            import plotly.express as px
            prefilter_plots = []
            # Q-value vs PEP
            if qcol and pepcol and qcol in df_orig.columns and pepcol in df_orig.columns:
                q_vals = pd.to_numeric(df_orig[qcol], errors='coerce')
                pep_vals = pd.to_numeric(df_orig[pepcol], errors='coerce')
                valid = q_vals.notna() & pep_vals.notna()
                if valid.sum() > 0:
                    x = np.clip(q_vals[valid].values, 1e-10, 1.0)
                    y = np.clip(pep_vals[valid].values, 1e-10, 1.0)
                    st_ = [_status_all(i) for i in df_orig[valid].index]
                    fig = _scatter_plotly(x, y, color=st_, title=f'Q-value vs PEP (n={len(x)})',
                                          xlabel='Q-value', ylabel='PEP', log_x=True, log_y=True,
                                          hline=pep_thresh, vline=q_thresh)
                    if fig:
                        prefilter_plots.append((fig, 'qpep'))
            # Xcorr vs Sp
            xcorr_col = _resolve_col(df_orig, XCORR_COLS)
            sp_col = _resolve_col(df_orig, SP_COLS)
            if xcorr_col and sp_col:
                xv = pd.to_numeric(df_orig[xcorr_col], errors='coerce')
                sv = pd.to_numeric(df_orig[sp_col], errors='coerce')
                v = xv.notna() & sv.notna()
                if v.sum() > 0:
                    fig = _scatter_plotly(xv[v].values, sv[v].values, color=[_status_all(i) for i in df_orig[v].index],
                                          title=f'XCorr vs Sp (n={v.sum()})', xlabel='XCorr', ylabel='Sp')
                    if fig:
                        prefilter_plots.append((fig, 'xcorr'))
            # DeltaCn vs E-value
            dc_col = _resolve_col(df_orig, DELTA_CN_COLS)
            ev_col = _resolve_col(df_orig, EVALUE_COLS)
            if dc_col and ev_col:
                dv = pd.to_numeric(df_orig[dc_col], errors='coerce')
                ev = pd.to_numeric(df_orig[ev_col], errors='coerce')
                v = dv.notna() & ev.notna() & (ev > 0)
                if v.sum() > 0:
                    fig = _scatter_plotly(dv[v].values, np.clip(ev[v].values, 1e-20, 1),
                                          color=[_status_all(i) for i in df_orig[v].index],
                                          title=f'DeltaCn vs E-value (n={v.sum()})', xlabel='DeltaCn', ylabel='E-value', log_y=True)
                    if fig:
                        prefilter_plots.append((fig, 'dcneval'))
            # Proline
            if 'plain_peptide' in df_orig.columns:
                def _is_pro(seq):
                    if pd.isna(seq) or not str(seq).strip(): return 'empty'
                    s = str(seq).strip()
                    n, c = s[0].upper() == 'P', s[-1].upper() == 'P'
                    if n and c: return 'N&C term P'
                    if n: return 'N-term P'
                    if c: return 'C-term P'
                    return 'No P'
                pro_status = df_orig['plain_peptide'].apply(_is_pro)
                all_status = [_status_all(idx) for idx in df_orig.index]
                plot_df = pd.DataFrame({'Proline': pro_status, 'All filters': all_status})
                fig = px.histogram(plot_df, x='Proline', color='All filters', barmode='stack', title='Proline - Pass all filters by category',
                                   color_discrete_map={'Fail': 'red', 'Pass': 'dodgerblue'})
                _reorder_traces_smaller_on_top(fig, plot_df, 'All filters')
                fig.update_layout(template='plotly_dark', height=PLOT_HEIGHT, margin=dict(l=40, r=20, t=40, b=40))
                prefilter_plots.append((fig, 'proline'))
            # Modifications
            if 'modifications' in df_orig.columns:
                has_mod = df_orig['modifications'].apply(_has_modifications)
                st_ = ['Modified' if h else 'Unmodified' for h in has_mod]
                all_status = [_status_all(idx) for idx in df_orig.index]
                plot_df = pd.DataFrame({'Modifications': st_, 'All filters': all_status})
                fig = px.histogram(plot_df, x='Modifications', color='All filters', barmode='stack',
                                   title='Modifications - Pass all filters by category',
                                   color_discrete_map={'Fail': 'red', 'Pass': 'dodgerblue'})
                _reorder_traces_smaller_on_top(fig, plot_df, 'All filters')
                fig.update_layout(template='plotly_dark', height=PLOT_HEIGHT, margin=dict(l=40, r=20, t=40, b=40), showlegend=False)
                prefilter_plots.append((fig, 'mods'))
            # PPM (observed/theoretical)
            if COMET_MATCHED_FRAGS_MZ_COL in df_orig.columns and 'plain_peptide' in df_orig.columns:
                try:
                    from visualization.chromatograms import (
                        get_best_theoretical_match,
                        matches_b_or_y_ion,
                        create_ms2_spectrum_figure,
                    )
                    col_ions, col_mz = COMET_MATCHED_FRAGS_COL, COMET_MATCHED_FRAGS_MZ_COL
                    scatter_data = []
                    for idx in df_orig.index:
                        row = df_orig.loc[idx]
                        peptide = row.get('plain_peptide') or ''
                        ions_val = row.get(col_ions, ''); mz_val = row.get(col_mz, '')
                        if pd.isna(ions_val) or pd.isna(mz_val) or not str(ions_val).strip(): continue
                        ions_list = [p.strip() for p in str(ions_val).split(',') if p.strip()]
                        mz_list = [p.strip() for p in str(mz_val).split(',') if p.strip()]
                        if len(ions_list) != len(mz_list): continue
                        for name, mz_s in zip(ions_list, mz_list):
                            if not (name.startswith('c') and name[1:].isdigit()) and not (name.startswith('z') and name[1:].isdigit()): continue
                            try:
                                obs_mz = float(mz_s)
                            except ValueError: continue
                            if obs_mz <= 0: continue
                            match = get_best_theoretical_match(name, peptide, obs_mz)
                            if match is None: continue
                            theo_mz, _, ppm_diff = match
                            scatter_data.append((theo_mz, obs_mz / theo_mz if theo_mz > 0 else np.nan, _status_all(idx)))
                    if scatter_data:
                        exp_arr = np.array([d[0] for d in scatter_data])
                        ratio_arr = np.array([d[1] for d in scatter_data])
                        status = [d[2] for d in scatter_data]
                        valid = np.isfinite(ratio_arr)
                        if valid.any():
                            fig = _scatter_plotly(exp_arr[valid], ratio_arr[valid], color=[status[i] for i in np.where(valid)[0]],
                                                  title=f'PPM: Observed/Theoretical m/z (ppm ≤ {ppm_thresh})', xlabel='Theoretical m/z',
                                                  ylabel='Observed / Theoretical', hline=1.0)
                            if fig:
                                prefilter_plots.append((fig, 'ppm'))
                except ImportError:
                    st.caption('PPM scatter requires visualization.chromatograms. Run Significant fragmentation for full PPM diagnostics.')
            _plot_grid(prefilter_plots)
        else:
            st.caption("Click **Generate plots** to load diagnostic plots.")

    with filter_tabs[7]:  # Extraction (Step 6)
        extraction_run_live = st.container()
        run_step6 = st.button('Run Extraction', key='run_step6', help='Extraction (envelope, apex, coelution, shape)')
        if run_step6:
            extraction_input = (step5_input if step5_input and os.path.exists(step5_input) else step6_input) or csv_path
            if not mzml_path:
                st.error('Select mzML.')
            elif not extraction_input or not os.path.exists(extraction_input):
                st.error('Run Prefilter first, or select prefilter CSV in the sidebar.')
            elif not os.path.exists(mzml_path):
                st.error('mzML not found.')
            else:
                try:
                    import pyopenms  # noqa: F401
                except (ModuleNotFoundError, ImportError):
                    st.error('pyopenms required for Extraction.')
                    with st.expander('Fix pyopenms', icon='▶'):
                        st.markdown("""
**Conda (recommended):**
```bash
conda activate <your_env>   # env where you ran: conda install -c bioconda pyopenms
streamlit run hdx_filter_app.py
```

**macOS – set library path:**
```bash
export DYLD_LIBRARY_PATH=$(brew --prefix openms)/lib:$DYLD_LIBRARY_PATH
# or conda: export DYLD_LIBRARY_PATH=$CONDA_PREFIX/lib:$DYLD_LIBRARY_PATH
```
                        """)
                else:
                    cmd = [sys.executable, os.path.join(_SCRIPT_DIR, 'workflow_scripts', 'extract_chromatograms_from_comet_frags_csv_mzml.py'),
                           '--mzml', os.path.abspath(mzml_path), '--csv', os.path.abspath(extraction_input),
                           '--output-dir', os.path.abspath(out_dir), '--use-nested-dirs']
                    if extraction_test:
                        cmd.append('--test')
                    if extraction_extract_only:
                        cmd.append('--extract-only')
                    out_suffix = '_extraction_test.csv' if extraction_test else '_extraction.csv'
                    # When test and input ends with _extraction, script outputs x_prefilter_extraction_test (not x_prefilter_extraction_extraction_test)
                    ext_output_path = None
                    if extraction_test and extraction_input:
                        ext_base = os.path.splitext(os.path.basename(extraction_input))[0]
                        out_base = ext_base[:-len('_extraction')] if ext_base.endswith('_extraction') else ext_base
                        ext_output_path = os.path.join(out_dir, out_base + '_extraction_test.csv')
                    _queue_run(cmd, 'Extraction', out_suffix, needs_visualization=True, input_path=extraction_input, output_path=ext_output_path)
                    _execute_pending_run(stream_container=extraction_run_live)
        st.subheader('Extraction')
        st.caption('Envelope, apex ≥10⁵, coelution, shape correlation, S/N fragments 1% max.')
        st.caption('Pass/Fail = passed all filters')
        extraction_preview = _resolve_preview_output('Extraction', step_outputs['extraction'])
        if extraction_preview:
            _render_step_output_csv(extraction_preview, 'step6_extraction', expanded=(csv_path == extraction_preview))

        dataframes_dir = os.path.join(out_dir, 'dataframes')
        metrics_csv = os.path.join(dataframes_dir, 'chromatogram_metrics_all.csv')
        traces_path = os.path.join(dataframes_dir, 'chromatogram_traces.npz')
        has_metrics_traces = has_extraction and os.path.exists(metrics_csv) and os.path.exists(traces_path)
        chrom_paths, _ = _list_chromatogram_plots(out_dir) if has_extraction else ([], None)

        # Chromatogram plots — run plot script to generate PNGs, display accepted/rejected separately
        if has_extraction:
            ext_accepted, ext_rejected = _list_chromatogram_plots_by_status(out_dir, 'extraction') if has_extraction else ([], [])
            # Auto-show plots when they exist (user wants them displayed)
            for key in ['show_extraction_accepted', 'show_extraction_rejected']:
                if key not in st.session_state:
                    st.session_state[key] = (key == 'show_extraction_accepted' and bool(ext_accepted)) or (key == 'show_extraction_rejected' and bool(ext_rejected))
            col_acc, col_rej, _ = st.columns([1, 1, 3])
            with col_acc:
                if st.button('Generate plots (accepted)', key='gen_extraction_accepted',
                             help='Run plot script and show accepted peptide chromatograms'):
                    st.session_state.show_extraction_accepted = True
                    if not ext_accepted and has_metrics_traces:
                        plot_cmd = [sys.executable, os.path.join(_SCRIPT_DIR, 'workflow_scripts', 'plot_chromatograms_from_extraction.py'),
                                    '--output-dir', os.path.abspath(out_dir)]
                        if step_outputs.get('extraction') and os.path.exists(step_outputs['extraction']):
                            plot_cmd.extend(['--filter-csv', step_outputs['extraction']])
                        _queue_run(plot_cmd, 'Chromatogram plots', '_chromatograms', needs_visualization=True)
            with col_rej:
                if st.button('Generate plots (rejected)', key='gen_extraction_rejected',
                             help='Run plot script and show rejected peptide chromatograms'):
                    st.session_state.show_extraction_rejected = True
                    if not ext_rejected and has_metrics_traces:
                        plot_cmd = [sys.executable, os.path.join(_SCRIPT_DIR, 'workflow_scripts', 'plot_chromatograms_from_extraction.py'),
                                    '--output-dir', os.path.abspath(out_dir)]
                        if step_outputs.get('extraction') and os.path.exists(step_outputs['extraction']):
                            plot_cmd.extend(['--filter-csv', step_outputs['extraction']])
                        _queue_run(plot_cmd, 'Chromatogram plots', '_chromatograms', needs_visualization=True)
            if st.button('Hide chromatogram plots', key='hide_extraction_chromatograms', help='Stop showing chromatogram plots.'):
                st.session_state.show_extraction_accepted = False
                st.session_state.show_extraction_rejected = False
            ext_accepted, ext_rejected = _list_chromatogram_plots_by_status(out_dir, 'extraction') if has_extraction else ([], [])
            plots_per_row, img_width = 2, 550
            if st.session_state.show_extraction_accepted:
                st.markdown('**Accepted** (extraction/)')
                if ext_accepted:
                    st.caption('Integrated isotopes, integration window green, collection window gray.')
                    for i in range(0, len(ext_accepted), plots_per_row):
                        chunk = ext_accepted[i:i + plots_per_row]
                        cols = st.columns(plots_per_row)
                        for j, p in enumerate(chunk):
                            if os.path.exists(p):
                                with cols[j]:
                                    st.image(p, width=img_width, caption=os.path.basename(p))
                elif has_metrics_traces:
                    st.info('Click **Generate plots (accepted)** to run the plot script. PNGs will appear after it completes.')
                else:
                    st.info('Run Extraction to generate chromatogram_metrics_all.csv and chromatogram_traces.npz.')
            if st.session_state.show_extraction_rejected:
                st.markdown('**Rejected** (rejected_extraction/)')
                if ext_rejected:
                    st.caption('Integrated isotopes, integration window green, collection window gray.')
                    for i in range(0, len(ext_rejected), plots_per_row):
                        chunk = ext_rejected[i:i + plots_per_row]
                        cols = st.columns(plots_per_row)
                        for j, p in enumerate(chunk):
                            if os.path.exists(p):
                                with cols[j]:
                                    st.image(p, width=img_width, caption=os.path.basename(p))
                elif has_metrics_traces:
                    st.info('Click **Generate plots (rejected)** to run the plot script. PNGs will appear after it completes.')

        # Diagnostic plots (coelution, apex, shape)
        if has_coelution or has_extraction:
            if st.session_state.get('show_extraction_plots', False) or st.session_state.get('show_extraction_accepted', False) or st.session_state.get('show_extraction_rejected', False):
                extraction_plots = []
                # Coelution vs shape (single plot: shape pass/fail with threshold)
                if has_coelution and 'shape_corr' in df_orig.columns:
                    valid = df_orig['coelution_score'].notna() & df_orig['shape_corr'].notna()
                    if valid.sum() > 0:
                        x = df_orig.loc[valid, 'coelution_score'].values
                        y = df_orig.loc[valid, 'shape_corr'].values
                        status = [_status_all(idx) for idx in df_orig[valid].index]
                        fig = _scatter_plotly(x, y, color=status, title=f'Coelution vs shape (min coel={coelution_min}, min shape={shape_min}) — Pass all filters',
                                              xlabel='Coelution score', ylabel='Shape correlation', hline=shape_min, vline=coelution_min)
                        if fig:
                            extraction_plots.append((fig, 'coel_shape'))
                elif has_coelution:
                    valid = df_orig['coelution_score'].notna()
                    if valid.sum() > 0:
                        import plotly.express as px
                        plot_df = pd.DataFrame({'Coelution score': df_orig.loc[valid, 'coelution_score'].values})
                        fig = px.histogram(plot_df, x='Coelution score', title='Coelution score distribution (extraction)')
                        fig.update_layout(template='plotly_dark', height=PLOT_HEIGHT, margin=dict(l=40, r=20, t=40, b=40))
                        extraction_plots.append((fig, 'coel_hist'))
                # Apex pass/fail
                if has_extraction and 'apex_intensity' in df_orig.columns:
                    valid = df_orig['apex_intensity'].notna()
                    if valid.sum() > 0:
                        import plotly.express as px
                        apex_vals = pd.to_numeric(df_orig.loc[valid, 'apex_intensity'], errors='coerce')
                        status = [_status_all(idx) for idx in df_orig[valid].index]
                        plot_df = pd.DataFrame({'Apex intensity': apex_vals.values, 'All filters': status})
                        fig = px.histogram(plot_df, x='Apex intensity', color='All filters', barmode='stack', log_y=True,
                                           title=f'Apex intensity (min={apex_min:.0e}) — Pass all filters',
                                           color_discrete_map={'Fail': 'red', 'Pass': 'dodgerblue'})
                        _reorder_traces_smaller_on_top(fig, plot_df, 'All filters')
                        fig.update_layout(
                            template='plotly_dark', height=PLOT_HEIGHT, margin=dict(l=40, r=20, t=40, b=40),
                            xaxis_title='MS1 apex intensity (instrument counts)'
                        )
                        fig.add_vline(x=apex_min, line_dash='dash', line_color='red')
                        extraction_plots.append((fig, 'apex'))
                # Coelution pass/fail (histogram)
                if has_coelution:
                    valid = df_orig['coelution_score'].notna()
                    if valid.sum() > 0:
                        import plotly.express as px
                        x = df_orig.loc[valid, 'coelution_score'].values
                        status = [_status_all(idx) for idx in df_orig[valid].index]
                        plot_df = pd.DataFrame({'Coelution': x, 'All filters': status})
                        fig = px.histogram(plot_df, x='Coelution', color='All filters', barmode='stack',
                                           title=f'Coelution (min={coelution_min}) — Pass all filters',
                                           color_discrete_map={'Fail': 'red', 'Pass': 'dodgerblue'})
                        _reorder_traces_smaller_on_top(fig, plot_df, 'All filters')
                        fig.update_layout(template='plotly_dark', height=PLOT_HEIGHT, margin=dict(l=40, r=20, t=40, b=40))
                        fig.add_vline(x=coelution_min, line_dash='dash', line_color='red')
                        extraction_plots.append((fig, 'coel'))
                if extraction_plots:
                    _plot_grid(extraction_plots)
        else:
            st.info('Load extraction CSV (Step 6 output) for chromatogram extraction diagnostics.')

    with filter_tabs[8]:  # Envelope (Step 7)
        run_step7 = st.button('Run Envelope', key='run_step7', help='Envelope filter')
        if run_step7:
            # Envelope runs on the most recent extraction CSV in output dir.
            env_input = _latest_csv_with_suffix(out_dir, ['_extraction', '_extraction_test']) or step6_input
            if not env_input or not os.path.exists(env_input):
                st.error('Run Extraction (Step 6) first. Envelope needs the extraction CSV with envelope_ok / m0_gt_m1_gt_m2 columns.')
            else:
                # Verify input has envelope columns (avoid passing OpenMS/prefilter CSV by mistake)
                try:
                    with open(env_input, 'r') as f:
                        hdr = f.readline()
                        if 'CometVersion' in hdr:
                            hdr = f.readline()
                    if 'envelope_ok' not in hdr and 'm0_gt_m1_gt_m2' not in hdr:
                        st.error('Input CSV has no envelope_ok or m0_gt_m1_gt_m2 columns. Run Extraction (Step 6) first.')
                    else:
                        cmd = [sys.executable, os.path.join(_SCRIPT_DIR, 'workflow_scripts', 'filter_envelope.py'),
                               '--input', env_input, '--output-dir', out_dir, '--no-plots']
                        _queue_run(cmd, 'Envelope', '_envelope.csv', needs_visualization=True, input_path=env_input)
                except Exception as e:
                    st.error(f'Could not verify input: {e}')
        _render_terminal()
        st.subheader('Envelope')
        st.caption('Require M0, M+1, M+2 present and M0>M+1 OR M+1>M+2. Output: ..._envelope.csv. Plot buttons below generate per-peptide summed MS1 spectra for accepted/rejected sets.')
        envelope_preview = _resolve_preview_output('Envelope', step_outputs['envelope'])
        if envelope_preview:
            _render_step_output_csv(envelope_preview, 'step7_envelope', expanded=(csv_path == envelope_preview))

        gen_envelope = st.button('Generate plots', key='gen_envelope_plots', help='Build envelope pass/fail diagnostic summary')
        if gen_envelope:
            _clear_chromatograms_cache()
            st.session_state['show_envelope_plots'] = True
        if st.session_state.get('show_envelope_plots', False):
            import plotly.express as px
            envelope_plots = []
            if has_envelope:
                env_col = 'm0_gt_m1_gt_m2' if 'm0_gt_m1_gt_m2' in df_orig.columns else ('envelope_ok' if 'envelope_ok' in df_orig.columns else None)
                if env_col:
                    def _env_status(row):
                        v = row.get(env_col)
                        if pd.isna(v): return 'Missing'
                        if _to_bool(v): return 'Pass (M0>M+1 or M+1>M+2)'
                        return 'Fail'
                    env_status = df_orig.apply(_env_status, axis=1)
                    all_status = [_status_all(idx) for idx in df_orig.index]
                    plot_df = pd.DataFrame({'Envelope': env_status, 'All filters': all_status})
                    fig = px.histogram(plot_df, x='Envelope', color='All filters', barmode='stack',
                                       title='Envelope (M0>M+1 OR M+1>M+2) — Pass all filters by category',
                                       color_discrete_map={'Fail': 'red', 'Pass': 'dodgerblue'})
                    _reorder_traces_smaller_on_top(fig, plot_df, 'All filters')
                    fig.update_layout(template='plotly_dark', height=PLOT_HEIGHT, margin=dict(l=40, r=20, t=40, b=40))
                    envelope_plots.append((fig, 'envelope'))
                if envelope_plots:
                    _plot_grid(envelope_plots)
            else:
                st.caption('Load extraction CSV (with envelope_ok / m0_gt_m1_gt_m2) for envelope diagnostics.')
        else:
            st.caption("Click **Generate plots** to load envelope diagnostic plots.")

        # Summed MS1 spectrum — same as individual extraction subplot
        if has_envelope and mzml_path and os.path.exists(mzml_path):
            st.markdown('**Summed MS1 spectrum** (integration window)')
            try:
                def _peptide_key_env(r):
                    m = str(r.get('modifications', '-')).strip() if 'modifications' in r.index else '-'
                    if not m or m.lower() == 'nan': m = '-'
                    return f"{str(r.get('plain_peptide', r.get('peptide', ''))).strip()}_{int(r.get('charge', 0) or 0)}_{m}"

                df_env = df_orig.copy()
                if 'peptide_key' not in df_env.columns:
                    df_env['peptide_key'] = df_env.apply(_peptide_key_env, axis=1)
                env_peptides = df_env['peptide_key'].drop_duplicates().tolist()
                env_options = ['— Select peptide —'] + env_peptides
                sel_env_pep = st.selectbox('Select peptide for summed MS1 spectrum', env_options, key='envelope_ms1_sel')
                if sel_env_pep and sel_env_pep != '— Select peptide —':
                    match = df_env[df_env['peptide_key'] == sel_env_pep].iloc[0]
                    from visualization.chromatograms import create_summed_ms1_spectrum_figure
                    fig = create_summed_ms1_spectrum_figure(match, mzml_path, peptide_key=sel_env_pep)
                    if fig is not None:
                        st.pyplot(fig)
                        plt.close(fig)
                    else:
                        st.caption('Could not generate spectrum (missing RT windows or m/z).')
            except ImportError as e:
                st.caption(f'Summed MS1 spectrum requires visualization.chromatograms: {e}')
            except Exception as e:
                st.caption(f'Summed MS1 spectrum error: {e}')
        elif has_envelope:
            st.info('Select mzML in sidebar to view summed MS1 spectra.')

        # Envelope summed MS1 spectrum plots — accepted vs rejected
        env_accepted, env_rejected = _list_chromatogram_plots_by_status(out_dir, 'envelope')
        env_plot_filter_input = _latest_csv_with_suffix(out_dir, ['_extraction', '_extraction_test']) or step6_input
        can_generate_env_summed = bool(env_plot_filter_input and os.path.exists(env_plot_filter_input) and mzml_path and os.path.exists(mzml_path))

        def _run_envelope_summed_ms1_plots(mode: str, target_dir: str) -> bool:
            """Run summed MS1 plot generation inline and stream newly saved PNGs live."""
            if mode not in ('accepted', 'rejected'):
                st.error(f'Unknown mode: {mode}')
                return False
            if not can_generate_env_summed:
                st.error('Need latest extraction CSV and mzML to generate summed MS1 envelope plots.')
                return False
            plot_cmd = [sys.executable, os.path.join(_SCRIPT_DIR, 'workflow_scripts', 'generate_envelope_summed_ms1_plots.py'),
                        '--input', env_plot_filter_input,
                        '--mzml', mzml_path,
                        '--mode', mode,
                        '--output-dir', target_dir,
                        '--clear-output']
            env, run_cmd = _build_env_and_cmd(plot_cmd, needs_visualization=True)
            work_dir = os.path.abspath(_SCRIPT_DIR)
            terminal_lines = []
            returncode = -1
            status_label = f"Envelope summed MS1 {mode} plots"
            with st.status(f'Running {status_label}…', expanded=True) as status:
                st.caption(f"Command: {' '.join(run_cmd)}")
                stream_box = st.empty()
                progress_box = st.empty()
                gallery_box = st.empty()
                try:
                    proc = subprocess.Popen(
                        run_cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        encoding='utf-8',
                        errors='replace',
                        bufsize=1,
                        cwd=work_dir,
                        env=env,
                    )
                    while True:
                        line = proc.stdout.readline()
                        if line:
                            ln = line.rstrip()
                            if ln:
                                terminal_lines.append(ln)
                                if len(terminal_lines) > 250:
                                    terminal_lines = terminal_lines[-250:]
                                stream_box.code('\n'.join(terminal_lines), language=None)
                        pngs = []
                        if os.path.isdir(target_dir):
                            try:
                                pngs = [os.path.join(target_dir, f) for f in os.listdir(target_dir) if f.lower().endswith('.png')]
                            except OSError:
                                pngs = []
                        progress_box.caption(f"Saved {len(pngs)} {mode} summed MS1 plots so far…")
                        if pngs:
                            recent = sorted(pngs, key=lambda p: os.path.getmtime(p))[-4:]
                            with gallery_box.container():
                                st.markdown('**Latest saved plots**')
                                cols = st.columns(2)
                                for i, img in enumerate(recent):
                                    with cols[i % 2]:
                                        st.image(img, width=450, caption=os.path.basename(img))
                        if not line and proc.poll() is not None:
                            break
                    returncode = proc.wait()
                except Exception as e:
                    terminal_lines.append(f"[Error running {status_label}: {e}]")
                    returncode = 1
                out_text = '\n'.join(terminal_lines).strip() or '(No output)'
                st.session_state.script_output = out_text
                st.session_state.script_step = status_label
                if returncode == 0:
                    status.update(label=f'{status_label} complete', state='complete')
                    return True
                status.update(label=f'{status_label} failed', state='error')
                return False

        for key in ['show_envelope_accepted', 'show_envelope_rejected']:
            if key not in st.session_state:
                st.session_state[key] = False
        col_env_acc, col_env_rej, _ = st.columns([1, 1, 3])
        with col_env_acc:
            if st.button('Generate Plots Accepted Envelopes', key='gen_envelope_accepted',
                         help='Generate summed MS1 spectrum plots for peptides that pass Envelope criteria'):
                st.session_state.show_envelope_accepted = True
                _run_envelope_summed_ms1_plots('accepted', os.path.join(out_dir, 'envelope'))
        with col_env_rej:
            if st.button('Generate Plots Rejected Envelopes', key='gen_envelope_rejected',
                         help='Generate summed MS1 spectrum plots for peptides rejected by Envelope criteria'):
                st.session_state.show_envelope_rejected = True
                _run_envelope_summed_ms1_plots('rejected', os.path.join(out_dir, 'envelope_rejected'))
        env_accepted, env_rejected = _list_chromatogram_plots_by_status(out_dir, 'envelope')
        plots_per_row, img_width = 2, 550
        if st.session_state.show_envelope_accepted:
            st.markdown('**Accepted** (envelope/)')
            if env_accepted:
                st.caption('Envelope-passed summed MS1 plots from the current Envelope step input.')
                for i in range(0, len(env_accepted), plots_per_row):
                    chunk = env_accepted[i:i + plots_per_row]
                    cols = st.columns(plots_per_row)
                    for j, p in enumerate(chunk):
                        if os.path.exists(p):
                            with cols[j]:
                                st.image(p, width=img_width, caption=os.path.basename(p))
            elif can_generate_env_summed:
                st.info('Click **Generate Plots Accepted Envelopes** to generate summed MS1 envelope plots.')
        if st.session_state.show_envelope_rejected:
            st.markdown('**Rejected** (envelope_rejected/)')
            if env_rejected:
                st.caption('Envelope-rejected summed MS1 plots from the current Envelope step input.')
                for i in range(0, len(env_rejected), plots_per_row):
                    chunk = env_rejected[i:i + plots_per_row]
                    cols = st.columns(plots_per_row)
                    for j, p in enumerate(chunk):
                        if os.path.exists(p):
                            with cols[j]:
                                st.image(p, width=img_width, caption=os.path.basename(p))
            elif can_generate_env_summed:
                st.info('Click **Generate Plots Rejected Envelopes** to generate summed MS1 envelope plots.')

    with filter_tabs[9]:  # Significance (combined: 5ppm + min sig frags + 1% max, all from summed MS2)
        run_sig_frag = st.button('Run Significant fragmentation', key='run_sig_frag',
                                 help='5 ppm + min sig frags + 1% max (all from summed extracted MS2)')
        if run_sig_frag:
            if not mzml_path or not os.path.exists(mzml_path):
                st.error('Select mzML.')
            elif not sig_frag_input or not os.path.exists(sig_frag_input):
                st.error('Run Extraction first (needs RT windows from extraction).')
            else:
                try:
                    import pyopenms  # noqa: F401
                except (ModuleNotFoundError, ImportError):
                    st.error('pyopenms required for Significant fragmentation.')
                    with st.expander('Fix pyopenms', icon='▶'):
                        st.markdown("""
**Use the conda environment:**
```bash
conda env create -f environment_hdx_app.yml
conda activate hdx_app
streamlit run hdx_filter_app.py
```

**macOS – set library path:**
```bash
export DYLD_LIBRARY_PATH=$CONDA_PREFIX/lib:$DYLD_LIBRARY_PATH
```
                        """)
                else:
                    min_frac = sig_frac_pct / 100.0
                    cmd = [sys.executable, os.path.join(_SCRIPT_DIR, 'workflow_scripts', 'filter_significant_frags_summed_ms2.py'),
                           '--mzml', mzml_path, '--input', sig_frag_input, '--output-dir', out_dir,
                           '--ppm', str(ppm_thresh), '--min-frac', str(min_frac), '--min-sig-frags', str(min_sig_frag_count)]
                    _queue_run(cmd, 'Significant fragmentation', '_significance.csv', needs_visualization=True, input_path=sig_frag_input)
        _render_terminal()
        st.subheader('Significant fragmentation')
        st.caption('5 ppm + min sig frags + 1% max peak (all from summed extracted MS2). Run Extraction first.')
        sig_preview = _resolve_preview_output('Significant fragmentation', step_outputs['significance'])
        if sig_preview:
            _render_step_output_csv(sig_preview, 'sig_frag_output', expanded=(csv_path == sig_preview))
        if has_sig_frags:
            sigfrag_plots = []
            counts = df_orig.apply(_count_sig, axis=1)
            effective_min = max(1, min_sig_frag_count) if enable_sig_frags else min_sig_frag_count
            if has_extraction and 'total_area' in df_orig.columns:
                valid = df_orig['total_area'].notna() & counts.notna()
                if valid.sum() > 0:
                    x = df_orig.loc[valid, 'total_area'].values
                    y = counts[valid].values
                    status = [_status_all(idx) for idx in df_orig[valid].index]
                    fig = _scatter_plotly(x, y, color=status, title=f'Sig fragment count vs total area (min={effective_min}) — Pass all filters',
                                          xlabel='Total area', ylabel='Sig fragment count', log_x=True)
                    if fig:
                        sigfrag_plots.append((fig, 'sig_area'))
            import plotly.express as px
            plot_df = pd.DataFrame({'Count': counts, 'All filters': [_status_all(idx) for idx in df_orig.index]})
            fig2 = px.histogram(plot_df, x='Count', color='All filters', barmode='stack',
                                title='Sig fragment count — Pass all filters',
                                   color_discrete_map={'Fail': 'red', 'Pass': 'dodgerblue'})
            _reorder_traces_smaller_on_top(fig2, plot_df, 'All filters')
            fig2.update_layout(template='plotly_dark', height=PLOT_HEIGHT, margin=dict(l=40, r=20, t=40, b=40))
            sigfrag_plots.append((fig2, 'sig_hist'))
            _plot_grid(sigfrag_plots)

            # MS2 spectrum plots — button-triggered so Significance tab loads fast and Sequence tab is not blocked
            st.subheader('MS2 spectra')
            can_show_ms2 = bool(mzml_path and os.path.exists(mzml_path) and
                              (SIGNIFICANT_FRAGS_COL in df_orig.columns))
            if not mzml_path or not os.path.exists(mzml_path):
                st.warning('Select an mzML file in the sidebar (Files) to view MS2 spectra.')
            elif not (SIGNIFICANT_FRAGS_COL in df_orig.columns):
                st.warning('CSV needs significant_frags column. Run Significant fragmentation first.')
            else:
                if 'show_ms2_spectra' not in st.session_state:
                    st.session_state.show_ms2_spectra = False
                col_btn, _ = st.columns([1, 4])
                with col_btn:
                    if st.button('Generate MS2 spectra', key='gen_ms2_spectra', help='Load mzML and render spectra for accepted + rejected peptides. May take a while.'):
                        _clear_chromatograms_cache()
                        st.session_state.show_ms2_spectra = True
                    if st.button('Hide MS2 spectra', key='hide_ms2_spectra', help='Stop showing MS2 plots to speed up tab loading.'):
                        st.session_state.show_ms2_spectra = False
                if st.session_state.show_ms2_spectra and can_show_ms2:
                    # Accepted peptides — one plot per unique peptide, summed MS2 in integration window
                    if passed_all and len(passed_all) > 0:
                        st.markdown('**Accepted peptides**')
                        accepted_rows = df_orig.loc[list(passed_all)]
                        rt_cols = ['detected_peak_min_rt', 'detected_peak_max_rt', 'collection_min_rt', 'collection_max_rt', 'anchor_rt', 'MS1_retention_time_sec']
                        has_rt = any(c in accepted_rows.columns for c in rt_cols)
                        if not has_rt:
                            st.warning('CSV needs RT columns (detected_peak_min_rt, collection_min_rt, or MS1_retention_time_sec). Load the extraction or significance CSV.')
                        else:
                            try:
                                from visualization.chromatograms import create_ms2_spectrum_figure
                                st.caption('One plot per unique peptide. Summed MS2 spectra within integration window. Significant fragments labeled.')
                                # Group by unique peptide (plain_peptide, charge, modifications); pick best row per peptide for integration window
                                def _pep_key(r):
                                    pep = str(r.get('plain_peptide', '')).strip() or str(r.get('sequence', '')).strip()
                                    ch = int(r.get('charge', 1)) if pd.notna(r.get('charge')) else 1
                                    m = str(r.get('modifications', '-')).strip() if pd.notna(r.get('modifications')) else '-'
                                    if not m or m.lower() == 'nan': m = '-'
                                    return (pep, ch, m)
                                qcol = _resolve_col(accepted_rows, QVALUE_COLS)
                                best_rows = []
                                for _, group in accepted_rows.groupby(accepted_rows.apply(_pep_key, axis=1)):
                                    if qcol and qcol in group.columns:
                                        g = group.copy()
                                        g['_qv'] = pd.to_numeric(g[qcol], errors='coerce')
                                        best_idx = g['_qv'].idxmin()
                                        best_rows.append(g.loc[best_idx])
                                    else:
                                        best_rows.append(group.iloc[0])
                                unique_peptides_df = pd.DataFrame(best_rows) if best_rows else pd.DataFrame()
                                rows_data = list(unique_peptides_df.iterrows()) if len(unique_peptides_df) > 0 else []
                                plots_per_row = 4
                                n_shown = 0
                                for i in range(0, len(rows_data), plots_per_row):
                                    chunk = rows_data[i : i + plots_per_row]
                                    cols = st.columns(plots_per_row)
                                    for j, (idx, row) in enumerate(chunk):
                                        with cols[j]:
                                            pep = str(row.get('plain_peptide', '')).strip() or str(row.get('sequence', '')).strip()
                                            ch = int(row.get('charge', 1)) if pd.notna(row.get('charge')) else 1
                                            label = f"{pep[:25]}{'...' if len(pep) > 25 else ''} +{ch}"
                                            fig = create_ms2_spectrum_figure(row, mzml_path, figsize=(6, 3))
                                            if fig is not None:
                                                st.pyplot(fig)
                                                plt.close(fig)
                                                n_shown += 1
                                                with st.expander(f'Expand {label}', expanded=False, icon='▶'):
                                                    fig_large = create_ms2_spectrum_figure(row, mzml_path, figsize=(14, 7))
                                                    if fig_large is not None:
                                                        st.pyplot(fig_large)
                                                        plt.close(fig_large)
                                            else:
                                                st.caption(f'{label} — no MS2')
                                if n_shown == 0 and len(rows_data) > 0:
                                    st.caption('No MS2 spectra could be generated. Check that mzML matches the experiment and RT windows are valid.')
                            except ImportError as e:
                                st.caption(f'MS2 plots require visualization.chromatograms: {e}')
                            except Exception as e:
                                st.error(f'MS2 spectrum plot error: {e}')
                    elif not passed_all or len(passed_all) == 0:
                        st.info('No peptides passed all filters. Adjust thresholds or load a CSV with accepted peptides.')

                    # Rejected peptides — one plot per unique peptide, summed MS2 in integration window
                    rejected_indices = set(df_orig.index) - passed_all
                    if rejected_indices:
                        st.markdown('**Rejected peptides**')
                        rejected_rows = df_orig.loc[list(rejected_indices)]
                        rt_cols = ['detected_peak_min_rt', 'detected_peak_max_rt', 'collection_min_rt', 'collection_max_rt', 'anchor_rt', 'MS1_retention_time_sec']
                        has_rt_rej = any(c in rejected_rows.columns for c in rt_cols)
                        if has_rt_rej:
                            try:
                                from visualization.chromatograms import create_ms2_spectrum_figure
                                st.caption('One plot per unique peptide. Summed MS2 in integration window. Uses significant_frags labels.')
                                def _pep_key_rej(r):
                                    pep = str(r.get('plain_peptide', '')).strip() or str(r.get('sequence', '')).strip()
                                    ch = int(r.get('charge', 1)) if pd.notna(r.get('charge')) else 1
                                    m = str(r.get('modifications', '-')).strip() if pd.notna(r.get('modifications')) else '-'
                                    if not m or m.lower() == 'nan': m = '-'
                                    return (pep, ch, m)
                                qcol_rej = _resolve_col(rejected_rows, QVALUE_COLS)
                                best_rows_rej = []
                                for _, group in rejected_rows.groupby(rejected_rows.apply(_pep_key_rej, axis=1)):
                                    if qcol_rej and qcol_rej in group.columns:
                                        g = group.copy()
                                        g['_qv'] = pd.to_numeric(g[qcol_rej], errors='coerce')
                                        best_idx = g['_qv'].idxmin()
                                        best_rows_rej.append(g.loc[best_idx])
                                    else:
                                        best_rows_rej.append(group.iloc[0])
                                unique_rej_df = pd.DataFrame(best_rows_rej) if best_rows_rej else pd.DataFrame()
                                rows_data_rej = list(unique_rej_df.iterrows()) if len(unique_rej_df) > 0 else []
                                plots_per_row = 4
                                for i in range(0, len(rows_data_rej), plots_per_row):
                                    chunk = rows_data_rej[i : i + plots_per_row]
                                    cols = st.columns(plots_per_row)
                                    for j, (idx, row) in enumerate(chunk):
                                        with cols[j]:
                                            pep = str(row.get('plain_peptide', '')).strip() or str(row.get('sequence', '')).strip()
                                            ch = int(row.get('charge', 1)) if pd.notna(row.get('charge')) else 1
                                            label = f"{pep[:25]}{'...' if len(pep) > 25 else ''} +{ch}"
                                            fig = create_ms2_spectrum_figure(row, mzml_path, figsize=(6, 3))
                                            if fig is not None:
                                                st.pyplot(fig)
                                                plt.close(fig)
                                                with st.expander(f'Expand {label}', expanded=False, icon='▶'):
                                                    fig_large = create_ms2_spectrum_figure(row, mzml_path, figsize=(14, 7))
                                                    if fig_large is not None:
                                                        st.pyplot(fig_large)
                                                        plt.close(fig_large)
                                            else:
                                                st.caption(f'{label} — no MS2')
                            except ImportError:
                                pass
                            except Exception as e:
                                st.error(f'Rejected MS2 plot error: {e}')
        else:
            st.info('No significant_frags column. Run Significant fragmentation.')

    with filter_tabs[10]:  # Sequence
        st.subheader('Sequence')
        st.caption('Unique peptides grid plots + protect_peptide column. Output: ..._prefilter_extraction_envelope_significance_sequence.csv')
        try:
            seq_cov_input = None
            for candidate in [step_outputs.get('significance'), step_outputs.get('envelope'), step_outputs.get('extraction')]:
                if candidate and os.path.exists(candidate):
                    seq_cov_input = candidate
                    break
            run_seq_cov = st.button('Run Sequence coverage', key='run_seq_cov',
                                    help='Unique peptides grid plots + protect_peptide column. Needs FASTA.')
            if run_seq_cov:
                if not seq_cov_input or not os.path.exists(seq_cov_input):
                    st.error('Run Significance (or Extraction/Envelope) first to create input CSV.')
                elif not fasta_path or not os.path.exists(fasta_path):
                    st.error('Select a FASTA file in the sidebar (Files).')
                else:
                    seq_cov_dir = os.path.join(out_dir, 'sequence_coverage')
                    os.makedirs(seq_cov_dir, exist_ok=True)
                    cmd = [sys.executable, os.path.join(_SCRIPT_DIR, 'workflow_scripts', 'compare_unique_peptides_sequence.py'),
                           '--input', seq_cov_input, '--output-dir', out_dir,
                           '--sequence-coverage-dir', seq_cov_dir, '--fasta', fasta_path]
                    _queue_run(cmd, 'Sequence coverage', '_sequence.csv', needs_visualization=True, input_path=seq_cov_input)
            _render_terminal()
            seq_output_path = _resolve_preview_output('Sequence coverage', step_outputs.get('sequence'))
            if seq_output_path and os.path.exists(seq_output_path):
                _render_step_output_csv(seq_output_path, 'seq_cov_sequence', expanded=False)
            seq_cov_dir = os.path.join(out_dir, 'sequence_coverage')
            seq_cov_files = _build_results_tree(seq_cov_dir) if os.path.isdir(seq_cov_dir) else []
            unique_pep_plots = [(d, p) for d, p in seq_cov_files if 'unique_peptides' in p.lower()]
            other_plots = [(d, p) for d, p in seq_cov_files if 'unique_peptides' not in p.lower()]
            if unique_pep_plots:
                st.markdown('**Unique peptides plots**')
                for disp, fp in unique_pep_plots:
                    st.image(fp, caption=disp, use_container_width=True)
            if other_plots:
                st.markdown('**Other sequence coverage plots**')
                path_map = {disp: fp for disp, fp in other_plots}
                options = ['— Select a plot —'] + [disp for disp, _ in other_plots]
                sel = st.selectbox('Plots', options, key='seq_cov_plot_sel')
                if sel and sel != '— Select a plot —':
                    st.image(path_map[sel], use_container_width=True)
            if not seq_cov_files:
                if not seq_cov_input or not os.path.exists(seq_cov_input):
                    st.info('Run Significance first, then click "Run Sequence coverage" to generate unique peptides plots. Select FASTA in the sidebar.')
                else:
                    st.info('Click "Run Sequence coverage" to generate unique peptides plots (requires FASTA in sidebar).')
        except Exception as e:
            st.error(f'Sequence tab error: {e}')
            import traceback
            st.code(traceback.format_exc())

        _render_terminal()

    with filter_tabs[11]:  # D_MZ (Estimated Deuterated MZ target range)
        st.subheader('Estimated Deuterated MZ target range calculation')
        st.caption('Max deuteration = peptide length − 2 (termini lack amide) − prolines (no amide bond). Target m/z = (undeuterated + deuterated) / 2; ± defines range from min to max.')
        d_mz_input = None
        for candidate in [step_outputs.get('sequence'), step_outputs.get('significance'), step_outputs.get('envelope'), step_outputs.get('extraction')]:
            if candidate and os.path.exists(candidate):
                d_mz_input = candidate
                break
        d_mz_input = d_mz_input or (csv_path if csv_path and os.path.exists(csv_path) else None)

        DEUTERIUM_MASS_DIFF = 1.00628  # Da per D (D − H)

        def _get_undeuterated_mz(row):
            for col in ('theoretical_mz', 'mz_theoretical', 'mz', 'exp_mz'):
                if col in row.index:
                    v = row.get(col)
                    if pd.notna(v) and v != '' and float(v) > 0:
                        return float(v)
            if 'calc_neutral_mass' in row.index and 'charge' in row.index:
                cm = row.get('calc_neutral_mass')
                z = int(row.get('charge', 1)) if pd.notna(row.get('charge')) else 1
                if pd.notna(cm) and cm > 0 and z > 0:
                    return (float(cm) + z * 1.00727646688) / z
            return np.nan

        def _max_deuteration(peptide):
            if pd.isna(peptide) or not peptide:
                return np.nan
            seq = str(peptide).strip().upper()
            if not seq:
                return np.nan
            n_pro = seq.count('P')
            return max(0, len(seq) - 2 - n_pro)

        if d_mz_input and os.path.exists(d_mz_input):
            run_d_mz = st.button('Run D_MZ calculation', key='run_d_mz', help='Compute deuterated m/z target and ± range for mass spectrometer')
            if run_d_mz:
                try:
                    df_dmz = load_csv(d_mz_input)
                    pep_col = 'plain_peptide' if 'plain_peptide' in df_dmz.columns else ('peptide' if 'peptide' in df_dmz.columns else 'sequence')
                    if pep_col not in df_dmz.columns:
                        st.error(f'CSV needs {pep_col} (or peptide/sequence) column.')
                    else:
                        mz_vals = df_dmz.apply(_get_undeuterated_mz, axis=1)
                        valid = mz_vals.notna() & (mz_vals > 0)
                        if valid.sum() == 0:
                            st.error('No valid undeuterated m/z found. CSV needs theoretical_mz, mz, or calc_neutral_mass + charge.')
                        else:
                            df_dmz['d_mz_max_deuteration'] = df_dmz[pep_col].apply(_max_deuteration)
                            z_vals = df_dmz['charge'].apply(lambda x: int(x) if pd.notna(x) and x != '' else 1)
                            df_dmz['d_mz_undeuterated'] = np.nan
                            df_dmz.loc[valid, 'd_mz_undeuterated'] = mz_vals[valid]
                            df_dmz['d_mz_deuterated'] = np.nan
                            df_dmz.loc[valid, 'd_mz_deuterated'] = (
                                df_dmz.loc[valid, 'd_mz_undeuterated'] +
                                df_dmz.loc[valid, 'd_mz_max_deuteration'] * DEUTERIUM_MASS_DIFF / z_vals[valid]
                            )
                            df_dmz['d_mz_target'] = np.nan
                            df_dmz.loc[valid, 'd_mz_target'] = (
                                (df_dmz.loc[valid, 'd_mz_undeuterated'] + df_dmz.loc[valid, 'd_mz_deuterated']) / 2
                            )
                            df_dmz['d_mz_range_pm'] = np.nan
                            df_dmz.loc[valid, 'd_mz_range_pm'] = (
                                (df_dmz.loc[valid, 'd_mz_deuterated'] - df_dmz.loc[valid, 'd_mz_undeuterated']) / 2
                            )
                            df_dmz['d_mz_min'] = df_dmz['d_mz_undeuterated']
                            df_dmz['d_mz_max'] = df_dmz['d_mz_deuterated']
                            base = os.path.splitext(os.path.basename(d_mz_input))[0]
                            dmz_out = os.path.join(out_dir, base + '_D_MZ.csv')
                            df_dmz.to_csv(dmz_out, index=False)
                            st.session_state.last_step_output = dmz_out
                            st.session_state.last_step_name = 'D_MZ'
                            st.success(f'Saved {os.path.basename(dmz_out)} with d_mz_target, d_mz_range_pm, d_mz_min, d_mz_max columns.')
                except Exception as e:
                    st.error(f'D_MZ calculation error: {e}')
                    import traceback
                    st.code(traceback.format_exc())
            dmz_out_path = os.path.join(out_dir, os.path.splitext(os.path.basename(d_mz_input))[0] + '_D_MZ.csv') if d_mz_input else None
            if dmz_out_path and os.path.exists(dmz_out_path):
                dmz_preview = _resolve_preview_output('D_MZ', dmz_out_path)
                if dmz_preview:
                    _render_step_output_csv(dmz_preview, 'dmz_output', expanded=(csv_path == dmz_preview))
        else:
            st.info('Load a CSV with plain_peptide (or peptide/sequence), charge, and theoretical_mz to run D_MZ calculation.')
        _render_terminal()

    with filter_tabs[12]:  # SF (Source Fragmentation)
        st.subheader('Estimated Source Fragmentation Calculation')
        st.caption('User-configurable y = mx + b (SF = m × theoretical_m/z + b). Group peptides with SF within N units; assign group_SF = midpoint of each bin (e.g. 30–40 → 35, 40–50 → 45).')
        sf_input = None
        for candidate in [step_outputs.get('sequence'), step_outputs.get('significance'), step_outputs.get('envelope'), step_outputs.get('extraction')]:
            if candidate and os.path.exists(candidate):
                sf_input = candidate
                break
        sf_input = sf_input or (csv_path if csv_path and os.path.exists(csv_path) else None)

        def _get_theoretical_mz(row):
            for col in ('theoretical_mz', 'mz_theoretical', 'mz', 'exp_mz'):
                if col in row.index:
                    v = row.get(col)
                    if pd.notna(v) and v != '' and float(v) > 0:
                        return float(v)
            if 'calc_neutral_mass' in row.index and 'charge' in row.index:
                cm = row.get('calc_neutral_mass')
                z = int(row.get('charge', 1)) if pd.notna(row.get('charge')) else 1
                if pd.notna(cm) and cm > 0 and z > 0:
                    return (float(cm) + z * 1.00727646688) / z
            return np.nan

        if sf_input and os.path.exists(sf_input):
            sf_m = st.number_input('m (slope)', value=1.38, min_value=0.01, max_value=10.0, step=0.01, key='sf_m',
                                   help='SF = m × theoretical_m/z + b')
            sf_b = st.number_input('b (intercept)', value=3.0, min_value=-100.0, max_value=100.0, step=0.1, key='sf_b')
            sf_bin_width = st.number_input('Bin width (units)', value=10.0, min_value=0.1, max_value=100.0, step=0.5, key='sf_bin_width',
                                           help='Group peptides with SF within this many units. E.g. 10 → 30–40 assigned 35, 40–50 assigned 45.')
            run_sf = st.button('Run SF grouping', key='run_sf', help='Compute SF, group by bin width, add group_SF = midpoint of each bin')
            if run_sf:
                try:
                    df_sf = load_csv(sf_input)
                    mz_vals = df_sf.apply(_get_theoretical_mz, axis=1)
                    valid = mz_vals.notna() & (mz_vals > 0)
                    if valid.sum() == 0:
                        st.error('No valid theoretical m/z found. CSV needs theoretical_mz, mz, or calc_neutral_mass + charge.')
                    else:
                        df_sf['SF'] = np.nan
                        df_sf.loc[valid, 'SF'] = sf_m * mz_vals[valid] + sf_b
                        sf_valid = df_sf['SF'].notna()
                        # Group by bin width: SF within N units → assign midpoint (e.g. 30–40 → 35, 40–50 → 45)
                        df_sf['group_SF'] = np.nan
                        if sf_valid.sum() > 0 and sf_bin_width > 0:
                            bin_start = np.floor(df_sf.loc[sf_valid, 'SF'].values / sf_bin_width) * sf_bin_width
                            df_sf.loc[sf_valid, 'group_SF'] = bin_start + sf_bin_width / 2
                        base = os.path.splitext(os.path.basename(sf_input))[0]
                        sf_out = os.path.join(out_dir, base + '_SF.csv')
                        df_sf.to_csv(sf_out, index=False)
                        st.session_state.last_step_output = sf_out
                        st.session_state.last_step_name = 'SF'
                        st.success(f'Saved {os.path.basename(sf_out)} with SF and group_SF columns.')
                except Exception as e:
                    st.error(f'SF grouping error: {e}')
                    import traceback
                    st.code(traceback.format_exc())
            sf_out_path = os.path.join(out_dir, os.path.splitext(os.path.basename(sf_input))[0] + '_SF.csv') if sf_input else None
            if sf_out_path and os.path.exists(sf_out_path):
                sf_preview = _resolve_preview_output('SF', sf_out_path)
                if sf_preview:
                    _render_step_output_csv(sf_preview, 'sf_output', expanded=(csv_path == sf_preview))
        else:
            st.info('Load a CSV with theoretical_mz (or mz, calc_neutral_mass+charge) to run SF grouping.')
        _render_terminal()

    with filter_tabs[13]:  # Channels
        st.subheader('Channels')
        st.caption('Assign each SF group to minimum number of channels based on non-overlapping RT. Peptides in same channel have non-overlapping retention times.')
        ch_input = None
        if csv_path and os.path.exists(csv_path):
            try:
                cols = load_csv(csv_path).columns
                if 'group_SF' in cols:
                    ch_input = csv_path
            except Exception:
                pass
        if not ch_input and os.path.isdir(out_dir):
            for f in os.listdir(out_dir):
                if f.endswith('_SF.csv'):
                    cand = os.path.join(out_dir, f)
                    if os.path.isfile(cand):
                        ch_input = cand
                        break

        def _get_rt_window(row):
            min_rt = row.get('detected_peak_min_rt') or row.get('collection_min_rt')
            max_rt = row.get('detected_peak_max_rt') or row.get('collection_max_rt')
            if pd.notna(min_rt) and pd.notna(max_rt) and float(min_rt) < float(max_rt):
                return (float(min_rt), float(max_rt))
            rt = row.get('MS1_retention_time_sec') or row.get('anchor_rt')
            if pd.notna(rt):
                t = float(rt)
                return (t - 30, t + 30)  # 30 s window around single RT
            return None

        def _assign_channels(intervals):
            """Greedy interval coloring: assign each to min channel with no overlap."""
            if not intervals:
                return []
            idx_start = sorted(range(len(intervals)), key=lambda i: intervals[i][0])
            channels = [-1] * len(intervals)
            for i in idx_start:
                s, e = intervals[i]
                k = 0
                while True:
                    overlap = False
                    for j in range(len(intervals)):
                        if channels[j] == k and j != i:
                            sj, ej = intervals[j]
                            if s < ej and sj < e:
                                overlap = True
                                break
                    if not overlap:
                        channels[i] = k
                        break
                    k += 1
            return channels

        if ch_input and os.path.exists(ch_input):
            df_ch = load_csv(ch_input)
            if 'group_SF' not in df_ch.columns:
                st.info('Run SF grouping first. Input needs group_SF column.')
            else:
                run_ch = st.button('Run Channel assignment', key='run_channels',
                                   help='Assign peptides to channels by non-overlapping RT within each SF group')
                if run_ch:
                    try:
                        df_ch = df_ch.copy()
                        rt_windows = df_ch.apply(_get_rt_window, axis=1)
                        valid_rt = rt_windows.notna()
                        if valid_rt.sum() == 0:
                            st.error('No valid RT found. CSV needs detected_peak_min_rt/max_rt, collection_min_rt/max_rt, or MS1_retention_time_sec.')
                        else:
                            df_ch['channel'] = np.nan
                            valid_both = valid_rt & df_ch['group_SF'].notna()
                            for gval, grp in df_ch.loc[valid_both].groupby('group_SF'):
                                idx = grp.index.tolist()
                                intervals = [rt_windows[i] for i in idx]
                                ch_assign = _assign_channels(intervals)
                                for j, i in enumerate(idx):
                                    df_ch.at[i, 'channel'] = ch_assign[j] + 1
                            base = os.path.splitext(os.path.basename(ch_input))[0]
                            ch_base = base.replace('_SF', '') if base.endswith('_SF') else base
                            ch_out = os.path.join(out_dir, ch_base + '_channels.csv')
                            df_ch.to_csv(ch_out, index=False)
                            st.session_state.last_step_output = ch_out
                            st.session_state.last_step_name = 'Channels'
                            st.success(f'Saved {os.path.basename(ch_out)} with channel column.')
                    except Exception as e:
                        st.error(f'Channel assignment error: {e}')
                        import traceback
                        st.code(traceback.format_exc())
                ch_base = os.path.splitext(os.path.basename(ch_input))[0].replace('_SF', '') if ch_input and '_SF' in os.path.basename(ch_input) else (os.path.splitext(os.path.basename(ch_input))[0] if ch_input else '')
                ch_out_path = os.path.join(out_dir, ch_base + '_channels.csv') if ch_base else None
                if ch_out_path and os.path.exists(ch_out_path):
                    channels_preview = _resolve_preview_output('Channels', ch_out_path)
                    if channels_preview:
                        _render_step_output_csv(channels_preview, 'channels_output', expanded=(csv_path == channels_preview))
        else:
            st.info('Run SF grouping first, then load the SF output CSV to assign channels.')
        _render_terminal()

    with filter_tabs[14]:  # Method
        st.subheader('Method')
        st.caption('Method parameters and documentation.')

    # Fallback: execute pending run at end for steps not started inline in their tab
    _execute_pending_run()

if __name__ == '__main__':
    main()
