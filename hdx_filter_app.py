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
import hashlib
import io
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
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
import streamlit.components.v1 as components
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Local app files can include very large generated PNGs. We downscale for display,
# but disable Pillow's decompression-bomb guard for trusted local content.
try:
    from PIL import Image as _PILImage

    _PILImage.MAX_IMAGE_PIXELS = None
    warnings.filterwarnings('ignore', category=getattr(__import__('PIL.Image', fromlist=['DecompressionBombWarning']), 'DecompressionBombWarning'))
except Exception:
    pass

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


@st.cache_data(show_spinner=False)
def _read_image_bytes_cached(path: str, mtime: float) -> bytes:
    """Read image bytes with mtime-keyed cache (refreshes when file changes)."""
    with open(path, 'rb') as f:
        return f.read()


@st.cache_data(show_spinner=False)
def _read_thumbnail_bytes_cached(path: str, mtime: float, max_width: int = 320) -> bytes:
    """
    Build and cache a lightweight thumbnail for faster gallery rendering.
    Falls back to original bytes if Pillow is unavailable.
    """
    try:
        from PIL import Image  # type: ignore

        with Image.open(path) as img:
            img = img.convert('RGB')
            w, h = img.size
            if w > max_width:
                new_h = max(1, int(h * (max_width / float(w))))
                img = img.resize((max_width, new_h), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format='JPEG', quality=82, optimize=True)
            return buf.getvalue()
    except Exception:
        return _read_image_bytes_cached(path, mtime)


@st.cache_data(show_spinner=False)
def _read_display_image_bytes_cached(path: str, mtime: float, max_width: int = 3600) -> bytes:
    """
    Read an image and downscale for display reliability.
    Keeps quality high but avoids enormous in-memory images in Streamlit/Pillow.
    """
    try:
        from PIL import Image  # type: ignore

        with Image.open(path) as img:
            img = img.convert('RGB')
            w, h = img.size
            if w > max_width:
                new_h = max(1, int(h * (max_width / float(w))))
                img = img.resize((max_width, new_h), Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format='JPEG', quality=90, optimize=True)
            return buf.getvalue()
    except Exception:
        return _read_image_bytes_cached(path, mtime)


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

    plot_base_dir = out_dir
    try:
        version_key = f'plot_version_{step}'
        plot_version = st.session_state.get(version_key)
        if plot_version:
            candidate = os.path.join(out_dir, 'plots_versions', step, str(plot_version))
            if os.path.isdir(candidate):
                plot_base_dir = candidate
            else:
                # Backward compatibility with earlier flat version layout.
                legacy_candidate = os.path.join(out_dir, 'plots_versions', str(plot_version))
                if os.path.isdir(legacy_candidate):
                    plot_base_dir = legacy_candidate
    except Exception:
        pass

    if step == 'extraction':
        accepted_dirs = [
            os.path.join(plot_base_dir, 'extraction', 'accepted', 'chromatograms_peptides'),
            os.path.join(plot_base_dir, 'extraction', 'chromatograms_peptides'),
            os.path.join(plot_base_dir, 'extraction'),
            os.path.join(plot_base_dir, 'accepted', 'chromatograms_peptides'),
        ]
        rejected_dirs = [os.path.join(plot_base_dir, 'rejected_extraction')]
        _delete_suffixed_extraction_plots(accepted_dirs + rejected_dirs)
    else:  # envelope
        accepted_dirs = [os.path.join(plot_base_dir, 'envelope')]
        rejected_dirs = [os.path.join(plot_base_dir, 'envelope_rejected')]
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
    plot_base_dir = out_dir
    try:
        plot_version = st.session_state.get('plot_version_extraction')
        if plot_version:
            candidate = os.path.join(out_dir, 'plots_versions', 'extraction', str(plot_version))
            if os.path.isdir(candidate):
                plot_base_dir = candidate
            else:
                legacy_candidate = os.path.join(out_dir, 'plots_versions', str(plot_version))
                if os.path.isdir(legacy_candidate):
                    plot_base_dir = legacy_candidate
    except Exception:
        pass
    candidates = [
        os.path.join(plot_base_dir, 'extraction', 'accepted', 'chromatograms_peptides'),
        os.path.join(plot_base_dir, 'extraction', 'chromatograms_peptides'),
        os.path.join(plot_base_dir, 'extraction'),
        os.path.join(plot_base_dir, 'accepted', 'chromatograms_peptides'),
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


def _list_chromatogram_overview_plots(out_dir: str) -> list[str]:
    """List overview/zoom chromatogram plots emitted by visualization.chromatograms."""
    plot_base_dir = out_dir
    try:
        plot_version = st.session_state.get('plot_version_extraction')
        if plot_version:
            candidate = os.path.join(out_dir, 'plots_versions', 'extraction', str(plot_version))
            if os.path.isdir(candidate):
                plot_base_dir = candidate
            else:
                legacy_candidate = os.path.join(out_dir, 'plots_versions', str(plot_version))
                if os.path.isdir(legacy_candidate):
                    plot_base_dir = legacy_candidate
    except Exception:
        pass

    if not plot_base_dir or not os.path.isdir(plot_base_dir):
        return []

    exts = ('.png', '.jpg', '.jpeg')
    paths = []
    for f in sorted(os.listdir(plot_base_dir)):
        fl = f.lower()
        if not fl.endswith(exts):
            continue
        if (
            'all_peptides_overlay' in fl
            or 'zoom_peptides' in fl
            or fl.endswith('_chromatograms.png')
        ):
            paths.append(os.path.join(plot_base_dir, f))

    rank_tokens = [
        'all_peptides_overlay.png',
        'all_peptides_overlay_tracks_only.png',
        'all_peptides_overlay_tracks_only_1pct.png',
        'all_peptides_overlay_skyline_style.png',
        'zoom_peptides_all_linear_log_tracks',
        'zoom_peptides_all_tracks_1pct',
        'zoom_peptides_all_tracks',
        'zoom_peptides_all_linear',
        'zoom_peptides_all_log',
        'zoom_peptides_segment_',
    ]

    def _rank(path: str) -> tuple[int, str]:
        name = os.path.basename(path).lower()
        for i, token in enumerate(rank_tokens):
            if token in name:
                return (i, name)
        return (len(rank_tokens), name)

    return sorted(set(paths), key=_rank)


def _collect_sequence_filter_keys(filter_csv: str) -> set[tuple[str, int, str]]:
    """Return normalized (peptide, charge, mods) keys from a sequence filter CSV."""
    if not filter_csv or not os.path.exists(filter_csv):
        return set()
    try:
        df = load_csv(filter_csv)
    except Exception:
        return set()
    pep_cols = ['plain_peptide', 'peptide_sequence', 'peptide', 'sequence']
    pep_col = next((c for c in pep_cols if c in df.columns), None)
    if pep_col is None:
        return set()
    mods_col = 'modifications' if 'modifications' in df.columns else ('mods' if 'mods' in df.columns else None)
    keys: set[tuple[str, int, str]] = set()
    for _, r in df.iterrows():
        pep = str(r.get(pep_col, '')).strip()
        if not pep:
            continue
        pep = re.sub(r'\[.*?\]', '', pep).strip().upper()
        if not pep:
            continue
        ch = int(pd.to_numeric(r.get('charge', 0), errors='coerce') or 0)
        mods_raw = str(r.get(mods_col, '-')).strip() if mods_col else '-'
        if not mods_raw or mods_raw.lower() in ('nan', 'none'):
            mods_raw = '-'
        mods_token = re.sub(r'[^a-z0-9]+', '_', mods_raw.lower()).strip('_') or '-'
        keys.add((pep, ch, mods_token))
    return keys


def _copy_sequence_matched_extraction_plots(filter_csv: str, out_dir: str, seq_acc_dir: str) -> int:
    """
    Copy accepted extraction peptide plots that match sequence filter peptides.
    Matches by normalized peptide token + charge token + mods token in filename.
    Ambiguous/no-mod matches are skipped to avoid pulling the wrong variant.
    """
    keys = _collect_sequence_filter_keys(filter_csv)
    if not keys:
        return 0
    accepted_dirs = [
        os.path.join(out_dir, 'extraction', 'accepted', 'chromatograms_peptides'),
        os.path.join(out_dir, 'extraction', 'chromatograms_peptides'),
        os.path.join(out_dir, 'extraction'),
        os.path.join(out_dir, 'accepted', 'chromatograms_peptides'),
    ]
    copied = 0
    os.makedirs(seq_acc_dir, exist_ok=True)
    for d in accepted_dirs:
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            nl = name.lower()
            if not nl.endswith(('.png', '.jpg', '.jpeg')):
                continue
            if 'overlay' in nl or '_rejected' in nl:
                continue
            src = os.path.join(d, name)
            base = os.path.basename(src).lower()
            matched = False
            for pep, charge, mods_token in keys:
                pep_token = re.sub(r'[^a-z0-9]+', '_', pep.lower()).strip('_')
                if not pep_token:
                    continue
                charge_ok = (charge <= 0) or (f'_z{charge}' in base or f'z{charge}_' in base or base.endswith(f'z{charge}.png'))
                if not charge_ok:
                    continue
                # Require mods token when available; if mods are '-' skip copy to avoid ambiguous matches.
                if mods_token == '-':
                    continue
                mods_ok = (mods_token in base)
                if pep_token in base and mods_ok:
                    matched = True
                    break
            if not matched:
                continue
            dst = os.path.join(seq_acc_dir, os.path.basename(src))
            try:
                shutil.copy(src, dst)
                os.utime(dst, None)
                copied += 1
            except Exception:
                pass
    return copied


def _summary_counts_from_csv(path: str) -> tuple[str, str, str, str]:
    """Return (rows, psms, unique_peptides_seq_charge_mods, unique_sequences)."""
    try:
        df = load_csv(path)
    except Exception:
        return ('—', '—', '—', '—')
    if df is None or len(df) == 0:
        return ('0', '0', '0', '0')

    rows = str(len(df))
    pep_col = next((c for c in ['plain_peptide', 'peptide_sequence', 'peptide', 'sequence'] if c in df.columns), None)
    if pep_col is None:
        return (rows, rows, rows, rows)

    def _norm_seq(v) -> str:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return ''
        # Remove bracketed mass deltas if present; Summary unique sequence should be AA sequence.
        return re.sub(r'\[.*?\]', '', str(v)).strip().upper()

    def _norm_mods(v) -> str:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return '-'
        s = str(v).strip()
        return s if s and s.lower() != 'nan' else '-'

    seq = df[pep_col].apply(_norm_seq)
    charge = pd.to_numeric(df['charge'], errors='coerce').fillna(0).astype(int) if 'charge' in df.columns else pd.Series([0] * len(df), index=df.index)
    mods_col = 'modifications' if 'modifications' in df.columns else ('mods' if 'mods' in df.columns else None)
    mods = df[mods_col].apply(_norm_mods) if mods_col else pd.Series(['-'] * len(df), index=df.index)

    unique_sequences = int(pd.Series(seq).replace('', np.nan).dropna().nunique())
    unique_peptides = int(pd.DataFrame({'seq': seq, 'charge': charge, 'mods': mods}).drop_duplicates().shape[0])

    # PSMs: prefer scan-level identity when scan-like column exists; fallback to rows.
    scan_col = next((c for c in ['scan', 'scan_num', 'scan_number', 'spectrum_scan', 'scan_id'] if c in df.columns), None)
    if scan_col:
        scan = pd.to_numeric(df[scan_col], errors='coerce').fillna(-1).astype(int)
        psms = int(pd.DataFrame({'seq': seq, 'charge': charge, 'mods': mods, 'scan': scan}).drop_duplicates().shape[0])
    else:
        psms = len(df)

    return (rows, str(psms), str(unique_peptides), str(unique_sequences))


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
                st.dataframe(step_df, width='stretch', height=400)
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


def _pass_confidence(row, df, qcol, pepcol, q_thresh, pep_thresh, use_q=False, use_pep=False):
    if not use_q and not use_pep:
        return True
    q_ok = True
    if use_q and qcol and qcol in df.columns and pd.notna(row.get(qcol)):
        try:
            q_ok = float(row[qcol]) <= q_thresh
        except (TypeError, ValueError):
            q_ok = True
    pep_ok = True
    if use_pep and pepcol and pepcol in df.columns and pd.notna(row.get(pepcol)):
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
    if 'pending_run_armed' not in st.session_state:
        st.session_state.pending_run_armed = False
    if 'last_step_output' not in st.session_state:
        st.session_state.last_step_output = None  # path to last successful step output CSV
    if 'last_step_name' not in st.session_state:
        st.session_state.last_step_name = None  # e.g. 'Extraction', 'Prefilter'
    if 'last_step_run_started_ts' not in st.session_state:
        st.session_state.last_step_run_started_ts = None
    if 'tab_focus_request' not in st.session_state:
        st.session_state.tab_focus_request = None
    if 'tab_focus_ttl' not in st.session_state:
        st.session_state.tab_focus_ttl = 0
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
    # Override uploaded paths independently so mixed sources work:
    # e.g. FASTA from data dir + mzML uploaded in this session.
    ud = st.session_state.get('uploaded_data_dir', '') or upload_dir_default
    up_fasta = st.session_state.get('uploaded_fasta')
    up_mzml = st.session_state.get('uploaded_mzml')
    if ud and os.path.isdir(ud):
        if up_fasta:
            fname, fval = up_fasta
            fp = fval if isinstance(fval, str) else os.path.join(ud, fname)
            if os.path.exists(fp):
                fasta_path = fp
        if up_mzml:
            mname, mval = up_mzml
            mp = mval if isinstance(mval, str) else os.path.join(ud, mname)
            if os.path.exists(mp):
                mzml_path = mp
    # Robust fallback: if session paths are stale after rerun/restart, rediscover uploads on disk.
    upload_fasta_files = _find_files(upload_dir_default, FASTA_EXT) if os.path.isdir(upload_dir_default) else []
    upload_mzml_files = _find_files(upload_dir_default, MZML_EXT) if os.path.isdir(upload_dir_default) else []
    if (not fasta_path or not os.path.exists(fasta_path)) and upload_fasta_files:
        fasta_path = max(upload_fasta_files, key=lambda p: os.path.getmtime(p))
    if (not mzml_path or not os.path.exists(mzml_path)) and upload_mzml_files:
        mzml_path = max(upload_mzml_files, key=lambda p: os.path.getmtime(p))
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
    pipeline_inputs_ready = bool(
        mzml_path and os.path.exists(mzml_path) and fasta_path and os.path.exists(fasta_path)
    )
    # Pipeline-first mode: when FASTA+mzML are available, prefer running from inputs.
    # But if a saved pipeline CSV exists, keep it for state/preview recovery.
    if pipeline_inputs_ready:
        recovery_suffixes = [
            '_sequence', '_significance', '_envelope',
            '_prefilter_extraction_test', '_confidence_accuracy_extraction_test', '_confidence_extraction_test', '_extraction_test',
            '_prefilter_extraction', '_confidence_accuracy_extraction', '_confidence_extraction', '_extraction',
            '_prefilter', '_confidence', '_comet_perc_openMS', '_comet_perc', '_comet'
        ]
        search_dir = output_dir_val if output_dir_val and os.path.isdir(output_dir_val) else data_dir
        recovered_csv = _latest_csv_with_suffix(search_dir, recovery_suffixes)
        if recovered_csv and os.path.exists(recovered_csv):
            csv_path = recovered_csv
        else:
            csv_path = None

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
        step_l = str(step_name or '').strip().lower()
        tab_name = None
        if step_l.startswith('comet'):
            tab_name = 'Comet'
        elif step_l.startswith('percolator'):
            tab_name = 'Percolator'
        elif step_l.startswith('openms'):
            tab_name = 'OpenMS'
        elif step_l.startswith('prefilter'):
            tab_name = 'Prefilter'
        elif step_l.startswith('extraction') or step_l.startswith('chromatogram plots'):
            tab_name = 'Extraction'
        elif step_l.startswith('envelope'):
            tab_name = 'Envelope'
        elif step_l.startswith('significance'):
            tab_name = 'Significance'
        elif step_l.startswith('sequence'):
            tab_name = 'Sequence'
        elif step_l.startswith('d_mz'):
            tab_name = 'D_MZ'
        elif step_l == 'sf':
            tab_name = 'SF'
        elif step_l.startswith('channels'):
            tab_name = 'Channels'
        if tab_name:
            st.session_state.tab_focus_request = tab_name
            # One-shot focus request; avoid stale requests pulling users back on later reruns.
            st.session_state.tab_focus_ttl = 1
        out_path = output_path or (input_path and _step_output_path(input_path, out_dir, out_suffix))
        st.session_state.pending_run = (cmd, step_name, out_suffix, needs_visualization, out_path, run_cwd, comet_rename)
        st.session_state.pending_run_armed = True

    def _request_tab_focus(tab_name: str, ttl: int = 1) -> None:
        """Request focus on a tab on next rerun (one-shot by default)."""
        st.session_state.tab_focus_request = tab_name
        st.session_state.tab_focus_ttl = max(1, int(ttl))

    def _execute_pending_run(stream_container=None) -> bool:
        """Execute queued command now. Returns True when a run was executed."""
        pending = st.session_state.pending_run
        pending_armed = bool(st.session_state.get('pending_run_armed', False))
        if pending and not pending_armed:
            st.session_state.pending_run = None
            return False
        if not pending:
            return False
        parts = pending if isinstance(pending, (list, tuple)) else (pending + (None, None, None))
        cmd, step_name, out_suffix, needs_visualization = parts[:4]
        output_path = parts[4] if len(parts) > 4 else None
        run_cwd = parts[5] if len(parts) > 5 else None
        comet_rename = parts[6] if len(parts) > 6 else None
        st.session_state.pending_run = None
        st.session_state.pending_run_armed = False
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

            # Only set last_step_output to a file path (CSV); never to a directory (e.g. plot dirs).
            if fresh_output_path and os.path.exists(fresh_output_path) and os.path.isfile(fresh_output_path):
                st.session_state.last_step_output = fresh_output_path
                st.session_state.last_step_name = step_name
                st.session_state.last_step_run_started_ts = run_started_ts
            else:
                # Don't "forget" prior successful step outputs when timestamp detection is noisy.
                fallback_output = ''
                if output_path and os.path.exists(output_path) and os.path.isfile(output_path):
                    fallback_output = output_path
                elif out_suffix:
                    try:
                        for fn in os.listdir(work_dir):
                            if fn.endswith(out_suffix):
                                fp = os.path.join(work_dir, fn)
                                if os.path.isfile(fp):
                                    fallback_output = fp
                                    break
                    except Exception:
                        fallback_output = ''
                if fallback_output and os.path.exists(fallback_output) and os.path.isfile(fallback_output):
                    st.session_state.last_step_output = fallback_output
                    st.session_state.last_step_name = step_name
                    st.session_state.last_step_run_started_ts = run_started_ts
                else:
                    # Keep prior last_step_output untouched to avoid forcing users to rerun steps.
                    st.session_state.last_step_name = step_name
                    st.session_state.last_step_run_started_ts = run_started_ts
                    st.sidebar.warning(
                        f'{step_name} completed, but no output file was auto-detected for {out_suffix}.'
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
    plots_root_dir = os.path.join(out_dir, 'plots_versions')

    def _plot_version_dir(step_name: str, version: str | None = None) -> str:
        step = str(step_name).strip().lower()
        key = f'plot_version_{step}'
        ver = version or str(st.session_state.get(key) or datetime.now().strftime('%Y%m%d_%H%M%S'))
        return os.path.join(plots_root_dir, step, ver)

    def _ensure_plot_version_dirs(step_name: str, version: str | None = None) -> str:
        base = _plot_version_dir(step_name, version)
        os.makedirs(base, exist_ok=True)
        step = str(step_name).strip().lower()
        if step == 'extraction':
            subdirs = ['diagnostics', 'extraction', 'rejected_extraction']
        elif step == 'envelope':
            subdirs = ['envelope', 'envelope_rejected']
        elif step == 'sequence':
            subdirs = ['sequence_coverage', 'sequence_unique_chromatograms']
        else:
            subdirs = []
        for sub in subdirs:
            os.makedirs(os.path.join(base, sub), exist_ok=True)
        return base

    def _rotate_plot_version(step_name: str) -> str:
        step = str(step_name).strip().lower()
        key = f'plot_version_{step}'
        st.session_state[key] = datetime.now().strftime('%Y%m%d_%H%M%S')
        return _ensure_plot_version_dirs(step, st.session_state[key])

    # Ensure each plot-producing section has an active version folder.
    for _step in ['extraction', 'envelope', 'sequence']:
        _key = f'plot_version_{_step}'
        if _key not in st.session_state:
            st.session_state[_key] = datetime.now().strftime('%Y%m%d_%H%M%S')
        _ensure_plot_version_dirs(_step, str(st.session_state[_key]))

    for sub in ['diagnostics', 'extraction', 'rejected_extraction', 'envelope', 'envelope_rejected', 'sequence_coverage', 'channel_assignment', 'fragmentation_source']:
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

    # Recover last successful step output from disk when session state is lost/stale.
    last_out_state = st.session_state.get('last_step_output')
    if not last_out_state or not os.path.exists(last_out_state):
        recovery_order = [
            ('Sequence coverage', ['_sequence']),
            ('Significant fragmentation', ['_significance']),
            ('Envelope', ['_envelope']),
            ('Extraction', ['_prefilter_extraction_test', '_confidence_accuracy_extraction_test', '_confidence_extraction_test', '_extraction_test',
                            '_prefilter_extraction', '_confidence_accuracy_extraction', '_confidence_extraction', '_extraction']),
            ('Prefilter', ['_prefilter', '_confidence']),
            ('OpenMS MS1', ['_comet_perc_openMS']),
            ('Percolator', ['_comet_perc']),
            ('Comet', ['_comet']),
        ]
        recovered = ''
        recovered_name = None
        for step_name, suffixes in recovery_order:
            p = _latest_csv_with_suffix(out_dir, suffixes)
            if p and os.path.exists(p):
                recovered = p
                recovered_name = step_name
                break
        if recovered:
            st.session_state.last_step_output = recovered
            st.session_state.last_step_name = recovered_name

    # Resolve chained inputs. conf_base = root name (e.g. comet_frags_perc_openMS).
    # Prefer last successful step output as chain source when available.
    last_out_for_chain = st.session_state.get('last_step_output')
    chain_csv_path = last_out_for_chain if (last_out_for_chain and os.path.exists(last_out_for_chain) and os.path.isfile(last_out_for_chain)) else csv_path
    base_for_resolve = os.path.splitext(os.path.basename(chain_csv_path))[0] if chain_csv_path else ''
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
    step5_input = _resolve_step_input_from_conf_base(conf_base, out_dir, ['_prefilter', '_confidence'], chain_csv_path) if chain_csv_path else None
    step6_suffixes = (['_prefilter_extraction_test', '_confidence_accuracy_extraction_test', '_confidence_extraction_test'] +
                      ['_prefilter_extraction', '_confidence_accuracy_extraction', '_confidence_extraction']) if extraction_test else (['_prefilter_extraction', '_confidence_accuracy_extraction', '_confidence_extraction'] +
                      ['_prefilter_extraction_test', '_confidence_accuracy_extraction_test', '_confidence_extraction_test'])
    step6_input = _resolve_step_input_from_conf_base(conf_base, out_dir, step6_suffixes, chain_csv_path) if chain_csv_path else None
    step7_suffixes = (['_prefilter_extraction_test_envelope', '_confidence_accuracy_extraction_test_envelope', '_confidence_extraction_test_envelope'] +
                      ['_prefilter_extraction_envelope', '_confidence_accuracy_extraction_envelope', '_confidence_extraction_envelope']) if extraction_test else (['_prefilter_extraction_envelope', '_confidence_accuracy_extraction_envelope', '_confidence_extraction_envelope'] +
                      ['_prefilter_extraction_test_envelope', '_confidence_accuracy_extraction_test_envelope', '_confidence_extraction_test_envelope'])
    step7_input = _resolve_step_input_from_conf_base(conf_base, out_dir, step7_suffixes, chain_csv_path) if chain_csv_path else None
    # Significance runs on envelope or extraction (needs RT windows)
    sig_frag_input = (step7_input if step7_input and os.path.exists(step7_input) else step6_input) if chain_csv_path else None

    st.sidebar.divider()

    # Prefer last step output (auto-load after pipeline run) over dropdown selection.
    # Only use it when it's a file (never a directory, e.g. plot output dir).
    last_out = st.session_state.get('last_step_output')
    if last_out and os.path.exists(last_out):
        if os.path.isfile(last_out):
            csv_path = last_out
        else:
            # Clear stale directory path (e.g. from envelope plot run) so load_csv never sees it.
            st.session_state.last_step_output = None
            st.session_state.last_step_name = None

    # Intentionally do not show sidebar "Loaded/Auto-loaded CSV" status;
    # inputs are FASTA + mzML and each step writes its own output CSV.
    pipeline_mode = (csv_path is None or not os.path.exists(csv_path)) and pipeline_inputs_ready
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
        has_envelope = (
            ('m0_m1_gt_m2_m3' in df.columns)
            or all(c in df.columns for c in ('m0_intensity', 'm1_intensity', 'm2_intensity', 'm3_intensity'))
            or ('m0_gt_m1_gt_m2' in df.columns)
            or ('envelope_ok' in df.columns)
        )
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
        'windows_accepted': _resolve_output('_windows_channels_accepted'),
        'windows_rejected': _resolve_output('_windows_channels_rejected'),
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
    tab_done_suffixes = {
        'comet': ['_comet'],
        'percolator': ['_comet_perc'],
        'openms': ['_comet_perc_openMS'],
        'prefilter': ['_prefilter', '_confidence'],
        'extraction': ['_prefilter_extraction_test', '_confidence_accuracy_extraction_test', '_confidence_extraction_test', '_extraction_test',
                       '_prefilter_extraction', '_confidence_accuracy_extraction', '_confidence_extraction', '_extraction'],
        'envelope': ['_prefilter_extraction_test_envelope', '_confidence_accuracy_extraction_test_envelope', '_confidence_extraction_test_envelope',
                     '_prefilter_extraction_envelope', '_confidence_accuracy_extraction_envelope', '_confidence_extraction_envelope', '_envelope'],
        'significance': ['_prefilter_extraction_test_envelope_significance', '_confidence_accuracy_extraction_test_envelope_significance',
                         '_confidence_extraction_test_envelope_significance', '_confidence_extraction_test_significance', '_confidence_accuracy_extraction_test_significance',
                         '_prefilter_extraction_envelope_significance', '_confidence_accuracy_extraction_envelope_significance',
                         '_confidence_extraction_envelope_significance', '_confidence_extraction_significance', '_confidence_accuracy_extraction_significance', '_significance'],
        'sequence': ['_prefilter_extraction_test_envelope_significance_sequence', '_confidence_extraction_test_envelope_significance_sequence',
                     '_prefilter_extraction_envelope_significance_sequence', '_confidence_extraction_envelope_significance_sequence', '_sequence', '_significance_sequence'],
        'd_mz': ['_D_MZ'],
        'sf': ['_SF'],
        'channels': ['_channels'],
        'windows': ['_windows_channels_accepted', '_windows_channels_rejected'],
    }

    def _tab_is_done(step_name: str) -> bool:
        # Prefer explicit expected outputs first.
        out = pipeline_outputs.get(step_name) or step_outputs.get(step_name, '')
        if out and os.path.exists(out):
            return True
        # In pipeline-first mode, csv_path can be intentionally None; fallback to suffix scan in out_dir.
        suffixes = tab_done_suffixes.get(step_name, [])
        if suffixes:
            latest = _latest_csv_with_suffix(out_dir, suffixes)
            if latest and os.path.exists(latest):
                return True
        return False

    green_tabs_css = []
    for name, path in [('comet', 2), ('percolator', 3), ('openms', 4), ('prefilter', 7), ('extraction', 8), ('envelope', 9), ('significance', 10), ('sequence', 11), ('windows', 12), ('channels', 13), ('d_mz', 15), ('sf', 16)]:
        if _tab_is_done(name):
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

    filter_steps = ['Summary', 'Comet', 'Percolator', 'OpenMS', 'Metrics', 'Compare', 'Prefilter', 'Extraction', 'Envelope', 'Significance', 'Sequence', 'Windows', 'Channels', 'Inspect', 'D_MZ', 'SF', 'Method']
    filter_tabs = st.tabs(filter_steps)
    # Temporary stability mode: disable JS-driven tab focus switching.
    # We only consume focus requests to avoid frontend hangs/spinners.
    req_ttl = int(st.session_state.get('tab_focus_ttl', 0) or 0)
    if req_ttl > 0:
        st.session_state.tab_focus_request = None
        st.session_state.tab_focus_ttl = 0
    else:
        st.session_state.tab_focus_request = None

    # Clear stale queued runs: all run buttons execute inline in their tab now.
    if st.session_state.pending_run:
        st.session_state.pending_run = None

    def _render_terminal(panel_key: str = 'default'):
        term_header = f"Terminal — {st.session_state.script_step}" if st.session_state.script_step else "Terminal"
        with st.expander(term_header, expanded=bool(st.session_state.script_output), icon='▶'):
            if st.session_state.script_output:
                # Use a fixed-height read-only text area so users can scroll long logs.
                st.text_area(
                    'Run output',
                    value=st.session_state.script_output,
                    height=420,
                    key=f'terminal_scrollable_output_{panel_key}',
                    disabled=True,
                    label_visibility='collapsed',
                )
            else:
                st.caption("Run a pipeline step (Step 1–8) to see the latest terminal output here.")

    # Metric defaults must exist even when Metrics is not the active section.
    enable_conf = bool(st.session_state.get('enable_conf', True))
    use_q_filter = bool(st.session_state.get('use_q_filter', False))
    q_thresh = float(st.session_state.get('q_thresh', 0.01))
    use_pep_filter = bool(st.session_state.get('use_pep_filter', False))
    pep_thresh = float(st.session_state.get('pep_thresh', 0.05))
    reject_proline = bool(st.session_state.get('reject_pro', True))
    reject_mods = bool(st.session_state.get('reject_mods', True))
    enable_xcorr_filter = bool(st.session_state.get('enable_xcorr_filter', False))
    xcorr_min = float(st.session_state.get('xcorr_min', 1.5))
    enable_sp_filter = bool(st.session_state.get('enable_sp_filter', False))
    sp_min = float(st.session_state.get('sp_min', 50.0))
    enable_ms1_mz_error_filter = bool(st.session_state.get('enable_ms1_mz_error', True))
    enable_envelope = bool(st.session_state.get('enable_env', True))
    envelope_rule_label = str(
        st.session_state.get(
            'envelope_rule',
            'Optional M+3: require M0/M+1/M+2 and M0 > M+2; if M+3 signal exists, require M0 > M+3 and M+2 > M+3',
        )
    )
    envelope_mode = 'm3_optional' if envelope_rule_label.startswith('Optional M+3') else 'strict_m3_required'
    enable_apex = bool(st.session_state.get('enable_apex', True))
    apex_min = float(st.session_state.get('apex_min', 1e6))
    enable_shape = bool(st.session_state.get('enable_shape', True))
    shape_min = float(st.session_state.get('shape_min', 0.8))
    enable_coelution = bool(st.session_state.get('enable_coel', True))
    coelution_min = float(st.session_state.get('coel_min', 0.8))
    enable_sig_frags = bool(st.session_state.get('enable_sig', True))
    min_sig_frag_count = int(st.session_state.get('min_sig', 5))
    sig_frac_pct = float(st.session_state.get('sig_frac', 1.0))
    ppm_thresh = float(st.session_state.get('ppm_thresh', 5.0))
    ms1_mz_error_ppm_thresh = float(st.session_state.get('ms1_mz_error_ppm_thresh', 6.0))
    peak_matching_ppm_threshold = float(st.session_state.get('peak_matching_ppm_threshold', 6.0))
    apex_ppm_reject_threshold = float(st.session_state.get('apex_ppm_reject_threshold', 6.0))
    extraction_test = bool(st.session_state.get('extraction_test', False))
    extraction_extract_only = bool(st.session_state.get('extraction_extract_only', False))
    channels_count = int(st.session_state.get('channels_count', 3))

    # Metrics section — run first so UI can override defaults before apply filters
    with filter_tabs[4]:
            # Pipeline order: Prefilter → Extraction → Significance. Compact layout, no expanders.
            # Prefilter (Step 4)
            st.caption('Prefilter')
            pf1, pf2, pf3, pf4, pf5 = st.columns(5)
            with pf1:
                enable_conf = st.checkbox('Prefilter (Proline, Mods, Confidence)', True, key='enable_conf')
            with pf2:
                use_q_filter = st.checkbox('Use Q-value', False, key='use_q_filter')
            with pf3:
                q_thresh = st.slider('Q-value', 0.001, 0.2, 0.01, 0.001, key='q_thresh', disabled=not use_q_filter)
            with pf4:
                use_pep_filter = st.checkbox('Use PEP', False, key='use_pep_filter')
            with pf5:
                pep_thresh = st.slider('PEP', 0.001, 0.2, 0.05, 0.001, key='pep_thresh', disabled=not use_pep_filter)
            pf6a, pf6, pf7, pf8, pf9, pf10 = st.columns(6)
            with pf6a:
                reject_proline = st.checkbox('Proline', True, key='reject_pro')
            with pf6:
                reject_mods = st.checkbox('Modifications', True, key='reject_mods')
            with pf7:
                enable_xcorr_filter = st.checkbox('XCorr filter', False, key='enable_xcorr_filter')
            with pf8:
                xcorr_min = st.slider('Min XCorr', 0.0, 10.0, 1.5, 0.1, key='xcorr_min', disabled=not enable_xcorr_filter)
            with pf9:
                enable_sp_filter = st.checkbox('Sp filter', False, key='enable_sp_filter')
            with pf10:
                sp_min = st.number_input('Min Sp', 0.0, 1000.0, 50.0, 1.0, key='sp_min', disabled=not enable_sp_filter)
            enable_ms1_mz_error_filter = st.checkbox('MS1 m/z error (ppm)', True, key='enable_ms1_mz_error')
            # Extraction (Step 6)
            st.caption('Extraction')
            ex1, ex2, ex3, ex4 = st.columns(4)
            with ex1:
                enable_envelope = st.checkbox('Envelope filter', True, key='enable_env')
                envelope_rule_label = st.selectbox(
                    'Envelope rule',
                    [
                        'Strict: require M0/M+1/M+2/M+3 and M0 > M+2 > M+3 (M+2 may exceed M+1)',
                        'Optional M+3: require M0/M+1/M+2 and M0 > M+2; if M+3 signal exists, require M0 > M+3 and M+2 > M+3',
                    ],
                    index=1,
                    key='envelope_rule',
                    help='Choose strict M+3-required filtering or an M+3-optional rule.',
                )
                envelope_mode = (
                    'm3_optional'
                    if envelope_rule_label.startswith('Optional M+3')
                    else 'strict_m3_required'
                )
            with ex2:
                enable_apex = st.checkbox('Apex', True, key='enable_apex')
                apex_options = [1e3, 1e4, 1e5, 1e6, 1e7, 1e8]
                apex_min = st.selectbox(
                    'Min apex',
                    options=apex_options,
                    index=3,  # default: 1e+6
                    key='apex_min',
                    format_func=lambda v: f"1e+{int(np.log10(v))}",
                )
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
                st.caption('PPM tolerance set below')
            with sf3:
                min_sig_frag_count = st.number_input('Min sig frags', 0, 50, 5, 1, key='min_sig')
            with sf4:
                sig_frac_pct = st.slider('1% max peak', 0.1, 5.0, 1.0, 0.1, key='sig_frac', help='Min % of max MS2 for fragment (noise filter)')
            st.caption('PPM settings')
            ppm_options = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0, 20.0]
            ppm1, ppm2, ppm3, ppm4 = st.columns(4)
            with ppm1:
                ppm_thresh = st.selectbox(
                    'Significance ppm (MS2)',
                    options=ppm_options,
                    index=3,  # default 5 ppm
                    key='ppm_thresh',
                    help='PPM tolerance for fragment matching in Significant fragmentation.',
                )
            with ppm2:
                ms1_mz_error_ppm_thresh = st.selectbox(
                    'Max |MS1 ppm|',
                    options=ppm_options,
                    index=4,  # default 6 ppm
                    key='ms1_mz_error_ppm_thresh',
                    disabled=not enable_ms1_mz_error_filter,
                    help='MS1 precursor m/z error tolerance used in Prefilter.',
                )
            with ppm3:
                peak_matching_ppm_threshold = st.selectbox(
                    'Peak matching ppm',
                    options=ppm_options,
                    index=4,  # default 6 ppm
                    key='peak_matching_ppm_threshold',
                    help='Extraction peak-matching stringency for resolving multiple candidate peaks.',
                )
            with ppm4:
                apex_ppm_reject_threshold = st.selectbox(
                    'Max |apex MS1 ppm|',
                    options=ppm_options,
                    index=4,  # default 6 ppm
                    key='apex_ppm_reject_threshold',
                    help='Extraction reject threshold for apex m/z error relative to theoretical precursor m/z.',
                )
            # Extraction run options
            st.caption('Extraction run')
            opt1, opt2, _ = st.columns(3)
            with opt1:
                extraction_test = st.checkbox('Test (30 pep)', False, key='extraction_test')
            with opt2:
                extraction_extract_only = st.checkbox('Extract only (no plots)', False, key='extraction_extract_only',
                                                     help='Skip individual chromatogram plots; CSV + traces saved for plotting later')
            # Channels metric
            st.caption('Channels')
            channels_count = st.selectbox(
                'Number of channels',
                options=[1, 2, 3, 4, 5],
                index=2,
                key='channels_count',
            )

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
        # Q-value / PEP (independent toggles)
        if (use_q_filter or use_pep_filter) and has_confidence and (qcol or pepcol):
            mask_conf = df_filtered.apply(
                lambda r: _pass_confidence(
                    r, df_filtered, qcol, pepcol, q_thresh, pep_thresh,
                    use_q=use_q_filter, use_pep=use_pep_filter
                ),
                axis=1
            )
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
        # Optional MS1 precursor m/z error filter (ppm)
        mzerr_col_runtime = 'MS1_mz_error_ppm' if 'MS1_mz_error_ppm' in df_filtered.columns else ('MS1_mz_error' if 'MS1_mz_error' in df_filtered.columns else None)
        if enable_ms1_mz_error_filter and mzerr_col_runtime:
            mzerr_vals = pd.to_numeric(df_filtered[mzerr_col_runtime], errors='coerce')
            mask_mzerr = mzerr_vals.abs() <= float(ms1_mz_error_ppm_thresh)
            mask_mzerr = mask_mzerr | mzerr_vals.isna()
            df_filtered = df_filtered[mask_mzerr].copy()
    step_counts.append(('Prefilter', len(df_filtered), n_prev))
    n_prev = len(df_filtered)

    # Step 6: Extraction (envelope, apex >=10^5, coelution, shape, S/N 1% max)
    if enable_envelope and has_envelope:
        def _pass_env(row):
            m_new = row.get('m0_m1_gt_m2_m3')
            m0 = row.get('m0_gt_m1_gt_m2')
            env = row.get('envelope_ok')
            i0 = pd.to_numeric(pd.Series([row.get('m0_intensity')]), errors='coerce').iloc[0]
            i1 = pd.to_numeric(pd.Series([row.get('m1_intensity')]), errors='coerce').iloc[0]
            i2 = pd.to_numeric(pd.Series([row.get('m2_intensity')]), errors='coerce').iloc[0]
            i3 = pd.to_numeric(pd.Series([row.get('m3_intensity')]), errors='coerce').iloc[0]
            if pd.notna(i0) and pd.notna(i1) and pd.notna(i2) and pd.notna(i3):
                return bool((float(i0) > float(i2)) and (float(i1) > float(i2)) and (float(i2) > float(i3)))
            if pd.notna(m_new):
                return _to_bool(m_new)
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
        def _count_overhangs(row):
            val = row.get('single_aa_overhangs_protein_positions')
            if pd.isna(val) or val is None or str(val).strip() == '':
                return 0
            return len([p.strip() for p in str(val).split(',') if p.strip()])
        counts = df_filtered.apply(_count_sig, axis=1)
        overhang_counts = df_filtered.apply(_count_overhangs, axis=1)
        effective_min = max(1, min_sig_frag_count)  # require at least 1 when enabled
        mask_sig = (counts >= effective_min) & (overhang_counts >= 2)
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
        return _pass_confidence(r, df_orig, qcol, pepcol, q_thresh, pep_thresh, use_q=use_q_filter, use_pep=use_pep_filter)
    def _pass_env(r):
        m_new, m0, env = r.get('m0_m1_gt_m2_m3'), r.get('m0_gt_m1_gt_m2'), r.get('envelope_ok')
        i0 = pd.to_numeric(pd.Series([r.get('m0_intensity')]), errors='coerce').iloc[0]
        i1 = pd.to_numeric(pd.Series([r.get('m1_intensity')]), errors='coerce').iloc[0]
        i2 = pd.to_numeric(pd.Series([r.get('m2_intensity')]), errors='coerce').iloc[0]
        i3 = pd.to_numeric(pd.Series([r.get('m3_intensity')]), errors='coerce').iloc[0]
        if pd.notna(i0) and pd.notna(i1) and pd.notna(i2) and pd.notna(i3):
            return bool((float(i0) > float(i2)) and (float(i1) > float(i2)) and (float(i2) > float(i3)))
        if pd.notna(m_new):
            return _to_bool(m_new)
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
                        st.plotly_chart(fig, width='stretch')

    # ─── Rest of tabs (Summary, Prefilter, Extraction, etc.) ─────────────────
    with filter_tabs[0]:  # Summary
        if pipeline_mode:
            st.info('Run the **Comet** → **Percolator** → **OpenMS** pipeline (tabs 2–4). The latest output CSV will auto-load for downstream steps.')
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
                ('Envelope', step_outputs.get('envelope', ''), 'M0/M+1 > M+2 > M+3 (requires M+3 present) → _envelope.csv'),
                ('Significance', step_outputs.get('significance', ''), '5ppm, min sig frags, 1% max peak → _significance.csv'),
                ('Sequence', step_outputs.get('sequence', ''), 'Unique peptides grid + protect_peptide → _sequence.csv'),
                ('Windows', None, 'Sequence-passed peptide RT windows + collection-window overlays'),
                ('D_MZ', step_outputs.get('d_mz', ''), 'Deuterated m/z target range (undeut + max deut) / 2 ± range → _D_MZ.csv'),
                ('SF', step_outputs.get('sf', ''), 'Source Fragmentation (y=mx+b) → _SF.csv'),
                ('Channels', step_outputs.get('channels', ''), 'Channel assignment → _channels.csv'),
                ('Method', None, 'Pipeline documentation'),
            ]
            summary_suffixes = {
                'Comet': ['_comet'],
                'Percolator': ['_comet_perc'],
                'OpenMS': ['_comet_perc_openMS'],
                'Prefilter': ['_prefilter', '_confidence'],
                'Extraction': ['_prefilter_extraction_test', '_confidence_accuracy_extraction_test', '_confidence_extraction_test', '_extraction_test',
                               '_prefilter_extraction', '_confidence_accuracy_extraction', '_confidence_extraction', '_extraction'],
                'Envelope': ['_prefilter_extraction_test_envelope', '_confidence_accuracy_extraction_test_envelope', '_confidence_extraction_test_envelope',
                             '_prefilter_extraction_envelope', '_confidence_accuracy_extraction_envelope', '_confidence_extraction_envelope', '_envelope'],
                'Significance': ['_prefilter_extraction_test_envelope_significance', '_confidence_accuracy_extraction_test_envelope_significance',
                                 '_confidence_extraction_test_envelope_significance', '_confidence_extraction_test_significance', '_confidence_accuracy_extraction_test_significance',
                                 '_prefilter_extraction_envelope_significance', '_confidence_accuracy_extraction_envelope_significance',
                                 '_confidence_extraction_envelope_significance', '_confidence_extraction_significance', '_confidence_accuracy_extraction_significance', '_significance'],
                'Sequence': ['_prefilter_extraction_test_envelope_significance_sequence', '_confidence_extraction_test_envelope_significance_sequence',
                             '_prefilter_extraction_envelope_significance_sequence', '_confidence_extraction_envelope_significance_sequence', '_sequence', '_significance_sequence'],
                'D_MZ': ['_D_MZ'],
                'SF': ['_SF'],
                'Channels': ['_channels'],
            }
            # Resolve outputs in dependency order so Summary does not mix stale files
            # from different runs (e.g. old _sequence with new _significance).
            resolved_by_tab: dict[str, str] = {}
            for tab_name, out_path, _ in tab_info:
                resolved_out = out_path
                if (not resolved_out or not os.path.exists(resolved_out)) and tab_name in summary_suffixes:
                    resolved_out = _latest_csv_with_suffix(out_dir, summary_suffixes[tab_name])
                resolved_by_tab[tab_name] = resolved_out if (resolved_out and os.path.exists(resolved_out)) else ''

            # Chain dependent steps to the same base when available.
            extraction_out = resolved_by_tab.get('Extraction', '')
            envelope_out = resolved_by_tab.get('Envelope', '')
            significance_out = resolved_by_tab.get('Significance', '')

            if extraction_out:
                env_candidate = os.path.splitext(extraction_out)[0] + '_envelope.csv'
                if os.path.exists(env_candidate):
                    envelope_out = env_candidate
                    resolved_by_tab['Envelope'] = env_candidate

            if envelope_out:
                sig_candidate = os.path.splitext(envelope_out)[0] + '_significance.csv'
                if os.path.exists(sig_candidate):
                    significance_out = sig_candidate
                    resolved_by_tab['Significance'] = sig_candidate

            if significance_out:
                seq_candidate = os.path.splitext(significance_out)[0] + '_sequence.csv'
                if os.path.exists(seq_candidate):
                    resolved_by_tab['Sequence'] = seq_candidate

            step_rows = '| Tab | Status | Rows | PSMs | Unique peptides (seq+charge+mods) | Unique sequences | Description |\n|-----|--------|------|------|-------------------------------|------------------|-------------|\n'
            prev_rows_num = None
            for tab_name, out_path, desc in tab_info:
                resolved_out = resolved_by_tab.get(tab_name, '') or out_path
                if resolved_out and os.path.exists(resolved_out):
                    rows_str, psm_str, uniq_pep_str, uniq_seq_str = _summary_counts_from_csv(resolved_out)
                    status = '✓'
                else:
                    rows_str, psm_str, uniq_pep_str, uniq_seq_str = ('—', '—', '—', '—')
                    status = '○' if out_path else '—'
                # Guardrail: row count increasing in downstream steps usually means mixed/stale outputs.
                try:
                    cur_rows_num = int(rows_str) if rows_str not in ('—', '') else None
                except Exception:
                    cur_rows_num = None
                if cur_rows_num is not None and prev_rows_num is not None and cur_rows_num > prev_rows_num:
                    status = '⚠'
                if cur_rows_num is not None:
                    prev_rows_num = cur_rows_num
                step_rows += f'| {tab_name} | {status} | {rows_str} | {psm_str} | {uniq_pep_str} | {uniq_seq_str} | {desc} |\n'
            st.markdown(step_rows)
            st.caption("Status `⚠` means downstream rows increased vs previous stage (likely mixed/stale outputs from different runs).")
            with st.expander('Detailed pipeline stages, active thresholds, and commands', expanded=False, icon='▶'):
                def _p(path: str, label: str) -> str:
                    return os.path.abspath(path) if path and os.path.exists(path) else f"<{label}>"

                def _cmd(cmd: list[str]) -> str:
                    return shlex.join([str(x) for x in cmd])

                sum_fasta_base = os.path.splitext(os.path.basename(fasta_path))[0] if fasta_path else 'fasta_base'
                sum_comet_csv = os.path.join(out_dir, f'{sum_fasta_base}_comet.csv')
                sum_comet_pin = os.path.join(out_dir, f'{sum_fasta_base}_comet.pin')
                sum_perc_csv = os.path.join(out_dir, f'{sum_fasta_base}_comet_perc.csv')
                sum_openms_csv = os.path.join(out_dir, f'{sum_fasta_base}_comet_perc_openMS.csv')

                sum_prefilter_input = ''
                for candidate in [step_outputs.get('openms', ''), _latest_csv_with_suffix(out_dir, ['_comet_perc_openMS']), csv_path]:
                    if candidate and os.path.exists(candidate):
                        sum_prefilter_input = candidate
                        break

                sum_extraction_input = ''
                for candidate in [
                    (step5_input if step5_input and os.path.exists(step5_input) else None),
                    _latest_csv_with_suffix(out_dir, ['_prefilter', '_confidence']),
                    (step6_input if step6_input and os.path.exists(step6_input) else None),
                    csv_path,
                ]:
                    if candidate and os.path.exists(candidate):
                        sum_extraction_input = candidate
                        break

                sum_envelope_input = _latest_csv_with_suffix(out_dir, ['_extraction', '_extraction_test']) or step6_input or ''
                sum_sig_input = (
                    _latest_csv_with_suffix(out_dir, ['_prefilter_extraction_test_envelope', '_prefilter_extraction_envelope', '_envelope'])
                    or step_outputs.get('envelope', '')
                    or step_outputs.get('extraction', '')
                    or ''
                )
                sum_seq_input = ''
                for candidate in [step_outputs.get('significance', ''), step_outputs.get('envelope', ''), step_outputs.get('extraction', '')]:
                    if candidate and os.path.exists(candidate):
                        sum_seq_input = candidate
                        break

                comet_cmd = [
                    sys.executable, os.path.join(_SCRIPT_DIR, 'run_comet_with_percolator.py'),
                    '--mzml', _p(mzml_path, 'mzML'),
                    '--fasta', _p(fasta_path, 'FASTA'),
                    '--params', _p(params_path, 'comet.params.new'),
                    '--skip-percolator',
                    '--output-dir', _p(out_dir, 'output_dir'),
                    '--output-base', sum_fasta_base,
                ]
                if comet_exe_path and str(comet_exe_path).strip():
                    comet_cmd += ['--comet-exe', comet_exe_path.strip()]

                percolator_cmd = [
                    sys.executable, os.path.join(_SCRIPT_DIR, 'add_percolator_qvalues.py'),
                    '--csv', _p(sum_comet_csv, 'comet_csv'),
                    '--pin', _p(sum_comet_pin, 'comet_pin'),
                    '--output', _p(sum_perc_csv, 'comet_perc_csv'),
                    '--fill-unmatched', '1.0',
                ]

                openms_cmd = [
                    sys.executable, os.path.join(_SCRIPT_DIR, 'add_ms1_data_openms.py'),
                    _p(sum_perc_csv, 'comet_perc_csv'),
                    _p(mzml_path, 'mzML'),
                    _p(sum_openms_csv, 'comet_perc_openMS_csv'),
                ]

                prefilter_cmd = [
                    sys.executable, os.path.join(_SCRIPT_DIR, 'workflow_scripts', 'filter_comet_frags_confidence.py'),
                    '--input', _p(sum_prefilter_input, 'openMS_csv'),
                    '--output-dir', _p(out_dir, 'output_dir'),
                    '--q-threshold', str(q_thresh),
                    '--pep-threshold', str(pep_thresh),
                    '--xcorr-min', str(xcorr_min),
                    '--sp-min', str(sp_min),
                ]
                if use_q_filter:
                    prefilter_cmd += ['--use-q']
                if use_pep_filter:
                    prefilter_cmd += ['--use-pep']
                if enable_xcorr_filter:
                    prefilter_cmd += ['--use-xcorr']
                if enable_sp_filter:
                    prefilter_cmd += ['--use-sp']
                if enable_ms1_mz_error_filter:
                    prefilter_cmd += ['--ms1-mz-error-ppm', str(ms1_mz_error_ppm_thresh)]
                else:
                    prefilter_cmd += ['--disable-ms1-mz-error']

                pipeline_extraction_plot_dir = _rotate_plot_version('extraction')
                pipeline_sequence_plot_dir = _rotate_plot_version('sequence')
                extraction_cmd = [
                    sys.executable, os.path.join(_SCRIPT_DIR, 'workflow_scripts', 'extract_chromatograms_from_comet_frags_csv_mzml.py'),
                    '--mzml', _p(mzml_path, 'mzML'),
                    '--csv', _p(sum_extraction_input, 'prefilter_csv'),
                    '--output-dir', _p(out_dir, 'output_dir'),
                    '--plots-output-dir', _p(pipeline_extraction_plot_dir, 'plots_output_dir'),
                    '--peak-matching-ppm-threshold', str(peak_matching_ppm_threshold),
                    '--apex-ppm-reject-threshold', str(apex_ppm_reject_threshold),
                    '--min-apex-intensity', str(apex_min if enable_apex else 0.0),
                    '--shape-corr-min', str(shape_min if enable_shape else 0.0),
                    '--use-nested-dirs',
                ]
                if extraction_test:
                    extraction_cmd.append('--test')
                if extraction_extract_only:
                    extraction_cmd.append('--extract-only')

                envelope_cmd = [
                    sys.executable, os.path.join(_SCRIPT_DIR, 'workflow_scripts', 'filter_envelope.py'),
                    '--input', _p(sum_envelope_input, 'extraction_csv'),
                    '--output-dir', _p(out_dir, 'output_dir'),
                    '--no-plots',
                ]

                sig_cmd = [
                    sys.executable, os.path.join(_SCRIPT_DIR, 'workflow_scripts', 'filter_significant_frags_summed_ms2.py'),
                    '--mzml', _p(mzml_path, 'mzML'),
                    '--input', _p(sum_sig_input, 'envelope_or_extraction_csv'),
                    '--output-dir', _p(out_dir, 'output_dir'),
                    '--ppm', str(ppm_thresh),
                    '--min-frac', str(sig_frac_pct / 100.0),
                    '--min-sig-frags', str(int(min_sig_frag_count)),
                    '--min-overhangs', '2',
                ]

                sequence_cmd = [
                    sys.executable, os.path.join(_SCRIPT_DIR, 'workflow_scripts', 'compare_unique_peptides_sequence.py'),
                    '--input', _p(sum_seq_input, 'significance_or_envelope_csv'),
                    '--output-dir', _p(out_dir, 'output_dir'),
                    '--sequence-coverage-dir', _p(os.path.join(pipeline_sequence_plot_dir, 'sequence_coverage'), 'sequence_coverage_dir'),
                    '--fasta', _p(fasta_path, 'FASTA'),
                ]

                detail_md = [
                    "### Stage Details",
                    f"- **Comet**: search mzML vs FASTA with Comet params. Parameters: `mzML`, `FASTA`, `params`, `comet_exe(optional)`, `output_base`. Command: `{_cmd(comet_cmd)}`",
                    f"- **Percolator**: adds q-values from `.pin` to Comet CSV. Parameters: `fill_unmatched=1.0`. Command: `{_cmd(percolator_cmd)}`",
                    f"- **OpenMS MS1**: annotates MS1 RT/intensity windows. Parameters: `input=comet_perc.csv`, `mzML`, `output=comet_perc_openMS.csv`. Command: `{_cmd(openms_cmd)}`",
                    f"- **Prefilter**: confidence and quality filters. Active thresholds: `use_q={use_q_filter}`, `q_threshold={q_thresh:.3f}`, `use_pep={use_pep_filter}`, `pep_threshold={pep_thresh:.3f}`, `use_xcorr={enable_xcorr_filter}`, `xcorr_min={xcorr_min:.2f}`, `use_sp={enable_sp_filter}`, `sp_min={sp_min:.2f}`, `use_ms1_mz_error={enable_ms1_mz_error_filter}`, `ms1_mz_error_ppm<={ms1_mz_error_ppm_thresh:.1f}`, plus proline/modification logic from prefilter script. Command: `{_cmd(prefilter_cmd)}`",
                    f"- **Extraction**: chromatogram extraction and summed MS1/MS2 features. Active options: `test_mode={extraction_test}`, `extract_only={extraction_extract_only}`, `peak_matching_ppm_threshold={peak_matching_ppm_threshold:.1f}`, `apex_ppm_reject_threshold={apex_ppm_reject_threshold:.1f}`, `min_apex_intensity={float(apex_min if enable_apex else 0.0):.0f}`, `shape_corr_min={float(shape_min if enable_shape else 0.0):.2f}`; extraction script computes envelope/apex/coelution/shape features used downstream. Command: `{_cmd(extraction_cmd)}`",
                    f"- **Envelope**: strict isotope-envelope filter. Criteria: requires `M0,M+1,M+2,M+3` and `M0/M+1 > M+2 > M+3`. Command: `{_cmd(envelope_cmd)}`",
                    f"- **Significance**: summed MS2 fragment significance filter. Active thresholds: `ppm={ppm_thresh:.1f}`, `min_sig_frags={int(min_sig_frag_count)}`, `min_overhangs=2`, `min_fragment_fraction={sig_frac_pct:.1f}%` of max summed MS2 peak. Command: `{_cmd(sig_cmd)}`",
                    f"- **Sequence**: sequence coverage/family/valuable-sequence computation and unique peptide outputs. Inputs: significance/envelope/extraction CSV + FASTA. Command: `{_cmd(sequence_cmd)}`",
                ]
                st.markdown('\n'.join(detail_md))
                st.markdown(
                    "\n".join([
                        "### Count Definitions",
                        "- **Rows**: total number of CSV rows at that stage (each row is one record in the table).",
                        "- **PSMs**: peptide-spectrum matches; when a scan column is present (`scan`, `scan_num`, `scan_number`, `spectrum_scan`, `scan_id`), this is counted as unique `(sequence, charge, modifications, scan)`; otherwise it falls back to row count.",
                        "- **Unique peptides (seq+charge+mods)**: unique peptide entities at precursor level, counted as unique `(sequence, charge, modifications)` regardless of how many scans/rows support them.",
                        "- **Unique sequences**: unique amino-acid sequences only (charge/modifications ignored; bracketed mass annotations removed before counting).",
                    ])
                )
        # Top summary metrics should reflect pipeline endpoints, not currently loaded intermediate CSV.
        comet_rows_num = None
        comet_unique_pep_num = None
        final_rows_num = None
        final_unique_pep_num = None
        try:
            comet_out = resolved_by_tab.get('Comet', '')
            if comet_out and os.path.exists(comet_out):
                c_rows, _, c_uniq_pep, _ = _summary_counts_from_csv(comet_out)
                comet_rows_num = int(c_rows) if c_rows not in ('—', '') else None
                comet_unique_pep_num = int(c_uniq_pep) if c_uniq_pep not in ('—', '') else None
        except Exception:
            comet_rows_num = None
            comet_unique_pep_num = None
        try:
            # Prefer Sequence unique peptides, then Significance/Envelope/Extraction unique peptides as fallback.
            for tname in ['Sequence', 'Significance', 'Envelope', 'Extraction', 'Prefilter']:
                t_out = resolved_by_tab.get(tname, '')
                if t_out and os.path.exists(t_out):
                    f_rows, _, u_pep, _ = _summary_counts_from_csv(t_out)
                    if f_rows not in ('—', ''):
                        final_rows_num = int(f_rows)
                    if u_pep not in ('—', ''):
                        final_unique_pep_num = int(u_pep)
                        break
        except Exception:
            final_rows_num = None
            final_unique_pep_num = None

        metric_original_rows = comet_rows_num if comet_rows_num is not None else n_orig
        metric_final_rows = final_rows_num if final_rows_num is not None else n_final
        metric_original_unique = comet_unique_pep_num if comet_unique_pep_num is not None else n_orig
        metric_final_unique = final_unique_pep_num if final_unique_pep_num is not None else n_final
        metric_retention_unique = (100.0 * metric_final_unique / metric_original_unique) if metric_original_unique and metric_original_unique > 0 else 0.0

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric('Original rows', metric_original_rows)
        with col2:
            st.metric('Final rows', metric_final_rows)
        with col3:
            st.metric('Original unique peptides', metric_original_unique)
        with col4:
            st.metric('Final unique peptides', metric_final_unique)
        with col5:
            st.metric('Unique peptide retention %', f'{metric_retention_unique:.1f}%')
        st.caption('Retention is computed using unique peptides (sequence+charge+mods): final / original.')
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
        if run_openms and openms_csv_in and mzml_path and os.path.exists(mzml_path):
            cmd = [sys.executable, os.path.join(_SCRIPT_DIR, 'add_ms1_data_openms.py'), openms_csv_in, mzml_path, openms_csv_out]
            _queue_run(cmd, 'OpenMS MS1', '_comet_perc_openMS.csv', output_path=openms_csv_out)
            _execute_pending_run(stream_container=openms_run_live)
        elif run_openms and (not openms_csv_in or not mzml_path):
            if not openms_csv_in:
                st.warning('Run Comet and Percolator first.')
            if not mzml_path or not os.path.exists(mzml_path):
                st.warning('Select mzML in sidebar.')
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
                        st.plotly_chart(fig, width='stretch')
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
        prefilter_input = ''
        _last_out = st.session_state.get('last_step_output', '') or ''
        _last_out_is_openms = bool(_last_out and os.path.exists(_last_out) and _last_out.endswith('_comet_perc_openMS.csv'))
        for candidate in [
            step_outputs.get('openms', ''),
            (_last_out if _last_out_is_openms else ''),
            _latest_csv_with_suffix(out_dir, ['_comet_perc_openMS']),
            csv_path,
        ]:
            if candidate and os.path.exists(candidate):
                prefilter_input = candidate
                break
        if run_step4 and prefilter_input:
            cmd = [sys.executable, os.path.join(_SCRIPT_DIR, 'workflow_scripts', 'filter_comet_frags_confidence.py'),
                   '--input', prefilter_input, '--output-dir', out_dir,
                   '--q-threshold', str(q_thresh), '--pep-threshold', str(pep_thresh),
                   '--xcorr-min', str(xcorr_min), '--sp-min', str(sp_min)]
            if use_q_filter:
                cmd.extend(['--use-q'])
            if use_pep_filter:
                cmd.extend(['--use-pep'])
            if enable_xcorr_filter:
                cmd.extend(['--use-xcorr'])
            if enable_sp_filter:
                cmd.extend(['--use-sp'])
            if enable_ms1_mz_error_filter:
                cmd.extend(['--ms1-mz-error-ppm', str(ms1_mz_error_ppm_thresh)])
            else:
                cmd.extend(['--disable-ms1-mz-error'])
            _queue_run(cmd, 'Prefilter', '_prefilter.csv', input_path=prefilter_input)
            _execute_pending_run(stream_container=prefilter_run_live)
        elif run_step4:
            st.error('Run OpenMS first (or load an OpenMS CSV output) before Prefilter.')
        st.subheader('Prefilter')
        st.caption('PEP, Q-value, prolines, mods, ppm (MS1 peptide) — run Significant fragmentation for PPM of matched fragments (5ppm)')
        st.caption('Pass/Fail = passed all filters (see how failing rows distribute across metrics)')
        prefilter_preview = _resolve_preview_output('Prefilter', step_outputs['prefilter'])
        prefilter_detected = prefilter_preview or _latest_csv_with_suffix(out_dir, ['_prefilter', '_confidence'])
        if prefilter_detected and os.path.exists(prefilter_detected):
            st.caption(f"Active Prefilter output: `{prefilter_detected}`")
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
                                          hline=(pep_thresh if use_pep_filter else None),
                                          vline=(q_thresh if use_q_filter else None))
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
            extraction_input = (
                (step5_input if step5_input and os.path.exists(step5_input) else None)
                or _latest_csv_with_suffix(out_dir, ['_prefilter', '_confidence'])
                or (step6_input if step6_input and os.path.exists(step6_input) else None)
                or csv_path
            )
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
                    plot_output_dir = _rotate_plot_version('extraction')
                    cmd = [sys.executable, os.path.join(_SCRIPT_DIR, 'workflow_scripts', 'extract_chromatograms_from_comet_frags_csv_mzml.py'),
                           '--mzml', os.path.abspath(mzml_path), '--csv', os.path.abspath(extraction_input),
                           '--output-dir', os.path.abspath(out_dir),
                           '--plots-output-dir', os.path.abspath(plot_output_dir),
                           '--peak-matching-ppm-threshold', str(peak_matching_ppm_threshold),
                           '--apex-ppm-reject-threshold', str(apex_ppm_reject_threshold),
                           '--min-apex-intensity', str(apex_min if enable_apex else 0.0),
                           '--shape-corr-min', str(shape_min if enable_shape else 0.0),
                           '--use-nested-dirs']
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
        st.caption('Extraction quality gates: apex intensity, shape correlation, and apex MS1 ppm. Envelope filtering is handled in Envelope step.')
        st.caption('Pass/Fail = passed all filters')
        extraction_preview = _resolve_preview_output('Extraction', step_outputs['extraction'])
        if extraction_preview:
            _render_step_output_csv(extraction_preview, 'step6_extraction', expanded=(csv_path == extraction_preview))

        dataframes_dir = os.path.join(out_dir, 'dataframes')
        metrics_csv = os.path.join(dataframes_dir, 'chromatogram_metrics_all.csv')
        traces_path = os.path.join(dataframes_dir, 'chromatogram_traces.npz')
        has_metrics_traces = os.path.exists(metrics_csv) and os.path.exists(traces_path)
        chrom_paths, _ = _list_chromatogram_plots(out_dir)
        ext_accepted, ext_rejected = _list_chromatogram_plots_by_status(out_dir, 'extraction')
        has_extraction_artifacts = has_metrics_traces or bool(ext_accepted) or bool(ext_rejected)

        # Chromatogram plots — run plot script to generate PNGs, display accepted/rejected separately
        if has_extraction_artifacts:
            if 'extraction_hires_plot' not in st.session_state:
                st.session_state.extraction_hires_plot = None
            if 'extraction_hires_label' not in st.session_state:
                st.session_state.extraction_hires_label = ''

            def _render_extraction_thumb_gallery(paths: list[str], gallery_key: str, thumbs_per_row: int = 3, thumb_width: int = 240) -> None:
                """Render small thumbnails with click-to-expand high-resolution viewer."""
                for i in range(0, len(paths), thumbs_per_row):
                    chunk = paths[i:i + thumbs_per_row]
                    cols = st.columns(thumbs_per_row)
                    for j, p in enumerate(chunk):
                        if not os.path.exists(p):
                            continue
                        try:
                            mt = os.path.getmtime(p)
                            thumb_bytes = _read_thumbnail_bytes_cached(p, mt, max_width=360)
                        except Exception:
                            thumb_bytes = p
                        with cols[j]:
                            st.image(thumb_bytes, width=thumb_width, caption=os.path.basename(p))
                            if st.button('Expand', key=f'{gallery_key}_expand_{i}_{j}_{os.path.basename(p)}'):
                                st.session_state.extraction_hires_plot = p
                                st.session_state.extraction_hires_label = os.path.basename(p)

            # Keep extraction plot rendering opt-in to avoid expensive auto-loading.
            for key in ['show_extraction_accepted', 'show_extraction_rejected']:
                if key not in st.session_state:
                    st.session_state[key] = False
            col_acc, col_rej, _ = st.columns([1, 1, 3])
            with col_acc:
                if st.button('Generate plots (accepted)', key='gen_extraction_accepted',
                             help='Run plot script and show accepted peptide chromatograms'):
                    plot_output_dir = _rotate_plot_version('extraction')
                    st.session_state.show_extraction_accepted = True
                    st.session_state.extraction_hires_plot = None
                    st.session_state.extraction_hires_label = ''
                    if has_metrics_traces:
                        plot_cmd = [sys.executable, os.path.join(_SCRIPT_DIR, 'workflow_scripts', 'plot_chromatograms_from_extraction.py'),
                                    '--output-dir', os.path.abspath(plot_output_dir),
                                    '--chromatogram-metrics-csv', os.path.abspath(metrics_csv),
                                    '--chromatogram-traces', os.path.abspath(traces_path),
                                    '--clear-output']
                        if step_outputs.get('extraction') and os.path.exists(step_outputs['extraction']):
                            plot_cmd.extend(['--filter-csv', step_outputs['extraction']])
                        _queue_run(plot_cmd, 'Chromatogram plots', '_chromatograms', needs_visualization=True)
            with col_rej:
                if st.button('Generate plots (rejected)', key='gen_extraction_rejected',
                             help='Run plot script and show rejected peptide chromatograms'):
                    plot_output_dir = _rotate_plot_version('extraction')
                    st.session_state.show_extraction_rejected = True
                    st.session_state.extraction_hires_plot = None
                    st.session_state.extraction_hires_label = ''
                    if has_metrics_traces:
                        plot_cmd = [sys.executable, os.path.join(_SCRIPT_DIR, 'workflow_scripts', 'plot_chromatograms_from_extraction.py'),
                                    '--output-dir', os.path.abspath(plot_output_dir),
                                    '--chromatogram-metrics-csv', os.path.abspath(metrics_csv),
                                    '--chromatogram-traces', os.path.abspath(traces_path),
                                    '--clear-output']
                        if step_outputs.get('extraction') and os.path.exists(step_outputs['extraction']):
                            plot_cmd.extend(['--filter-csv', step_outputs['extraction']])
                        _queue_run(plot_cmd, 'Chromatogram plots', '_chromatograms', needs_visualization=True)
            if st.button('Hide chromatogram plots', key='hide_extraction_chromatograms', help='Stop showing chromatogram plots.'):
                st.session_state.show_extraction_accepted = False
                st.session_state.show_extraction_rejected = False
            ext_accepted, ext_rejected = _list_chromatogram_plots_by_status(out_dir, 'extraction')
            if st.session_state.show_extraction_accepted:
                st.markdown('**Accepted** (extraction/)')
                if ext_accepted:
                    st.caption('Integrated isotopes, integration window green, collection window gray.')
                    _render_extraction_thumb_gallery(ext_accepted, 'ext_accepted')
                elif has_metrics_traces:
                    st.info('Click **Generate plots (accepted)** to run the plot script. PNGs will appear after it completes.')
                else:
                    st.info('Run Extraction to generate chromatogram_metrics_all.csv and chromatogram_traces.npz.')
            if st.session_state.show_extraction_rejected:
                st.markdown('**Rejected** (rejected_extraction/)')
                if ext_rejected:
                    st.caption('Integrated isotopes, integration window green, collection window gray.')
                    _render_extraction_thumb_gallery(ext_rejected, 'ext_rejected')
                elif has_metrics_traces:
                    st.info('Click **Generate plots (rejected)** to run the plot script. PNGs will appear after it completes.')

            show_any_extraction_plots = bool(
                st.session_state.get('show_extraction_accepted', False)
                or st.session_state.get('show_extraction_rejected', False)
            )
            if show_any_extraction_plots:
                overview_plots = _list_chromatogram_overview_plots(out_dir)
                if overview_plots:
                    st.markdown('**Chromatogram overview/zoom plots**')
                    _render_extraction_thumb_gallery(overview_plots, 'ext_overview')

                selected_plot = st.session_state.get('extraction_hires_plot')
                if selected_plot and os.path.exists(selected_plot):
                    plot_label = st.session_state.get('extraction_hires_label') or os.path.basename(selected_plot)
                    st.markdown('**High-resolution plot viewer**')
                    # Prominent Close button at top so user can always exit
                    if st.button('← Back to gallery', key='close_extraction_hires', type='primary', use_container_width=True):
                        st.session_state.extraction_hires_plot = None
                        st.session_state.extraction_hires_label = ''
                        st.rerun()
                    st.caption(plot_label)
                    # Full resolution: load original PNG bytes (no downscale) for expanded view
                    try:
                        original_bytes = _read_image_bytes_cached(selected_plot, os.path.getmtime(selected_plot))
                    except Exception:
                        original_bytes = None
                    if original_bytes:
                        st.image(original_bytes, use_container_width=True)
                        st.download_button(
                            'Download original PNG',
                            data=original_bytes,
                            file_name=os.path.basename(selected_plot),
                            mime='image/png',
                            key=f'dl_extraction_hires_{os.path.basename(selected_plot)}'
                        )
                    else:
                        st.image(selected_plot, use_container_width=True)
                    # Second Close button below image so it's visible after scrolling
                    if st.button('← Back to gallery', key='close_extraction_hires_bottom', type='primary', use_container_width=True):
                        st.session_state.extraction_hires_plot = None
                        st.session_state.extraction_hires_label = ''
                        st.rerun()
        elif has_extraction:
            st.info('Run Extraction to generate chromatogram_metrics_all.csv and chromatogram_traces.npz.')

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
        envelope_run_live = st.container()
        run_step7 = st.button('Run Envelope', key='run_step7', help='Envelope filter')
        if run_step7:
            _request_tab_focus('Envelope', ttl=1)
            # Envelope runs on the most recent extraction CSV in output dir.
            env_input = _latest_csv_with_suffix(out_dir, ['_extraction', '_extraction_test']) or step6_input
            if not env_input or not os.path.exists(env_input):
                st.error('Run Extraction (Step 6) first. Envelope needs extraction summed-MS1 intensity columns.')
            else:
                # Verify input has envelope columns (avoid passing OpenMS/prefilter CSV by mistake)
                try:
                    with open(env_input, 'r') as f:
                        hdr = f.readline()
                        if 'CometVersion' in hdr:
                            hdr = f.readline()
                    has_intensity_triplet = ('m0_intensity' in hdr and 'm1_intensity' in hdr and 'm2_intensity' in hdr and 'm3_intensity' in hdr)
                    has_any_env_metric = ('m0_m1_gt_m2_m3' in hdr or 'envelope_ok' in hdr or 'm0_gt_m1_gt_m2' in hdr)
                    if not has_intensity_triplet and not has_any_env_metric:
                        st.error('Input CSV has no envelope intensity columns (m0/m1/m2/m3). Run Extraction (Step 6) first.')
                    else:
                        cmd = [sys.executable, os.path.join(_SCRIPT_DIR, 'workflow_scripts', 'filter_envelope.py'),
                               '--input', env_input, '--output-dir', out_dir, '--no-plots',
                               '--envelope-mode', envelope_mode]
                        _queue_run(cmd, 'Envelope', '_envelope.csv', needs_visualization=True, input_path=env_input)
                        _execute_pending_run(stream_container=envelope_run_live)
                except Exception as e:
                    st.error(f'Could not verify input: {e}')
        st.subheader('Envelope')
        st.caption('Require M0, M+1, M+2, M+3 present and M0/M+1 > M+2 > M+3. Output: ..._envelope.csv. Plot buttons below generate per-peptide summed MS1 spectra for accepted/rejected sets.')
        envelope_preview = _resolve_preview_output('Envelope', step_outputs['envelope'])
        if envelope_preview:
            _render_step_output_csv(envelope_preview, 'step7_envelope', expanded=(csv_path == envelope_preview))
        if 'show_envelope_accepted' not in st.session_state:
            st.session_state.show_envelope_accepted = False
        if 'show_envelope_rejected' not in st.session_state:
            st.session_state.show_envelope_rejected = False

        envelope_plot_input = _latest_csv_with_suffix(out_dir, ['_extraction', '_extraction_test']) or step6_input
        can_gen_envelope_plots = bool(
            envelope_plot_input and os.path.exists(envelope_plot_input) and mzml_path and os.path.exists(mzml_path)
        )

        env_col1, env_col2, env_col3 = st.columns([1, 1, 2])
        with env_col1:
            if st.button('Generate plots (accepted envelopes)', key='gen_envelope_accepted'):
                st.session_state.show_envelope_accepted = True
                if not can_gen_envelope_plots:
                    st.warning('Need extraction CSV and mzML to generate envelope plots.')
                else:
                    plot_output_dir = _rotate_plot_version('envelope')
                    cmd = [
                        sys.executable, os.path.join(_SCRIPT_DIR, 'workflow_scripts', 'generate_envelope_summed_ms1_plots.py'),
                        '--input', os.path.abspath(envelope_plot_input),
                        '--mzml', os.path.abspath(mzml_path),
                        '--mode', 'accepted',
                        '--output-dir', os.path.abspath(os.path.join(plot_output_dir, 'envelope')),
                        '--clear-output',
                    ]
                    _queue_run(
                        cmd, 'Envelope chromatogram plots', '_envelope_plots',
                        needs_visualization=True, input_path=envelope_plot_input,
                        output_path=os.path.join(plot_output_dir, 'envelope')
                    )
                    _execute_pending_run(stream_container=envelope_run_live)
        with env_col2:
            if st.button('Generate plots (rejected envelopes)', key='gen_envelope_rejected'):
                st.session_state.show_envelope_rejected = True
                if not can_gen_envelope_plots:
                    st.warning('Need extraction CSV and mzML to generate envelope plots.')
                else:
                    plot_output_dir = _rotate_plot_version('envelope')
                    cmd = [
                        sys.executable, os.path.join(_SCRIPT_DIR, 'workflow_scripts', 'generate_envelope_summed_ms1_plots.py'),
                        '--input', os.path.abspath(envelope_plot_input),
                        '--mzml', os.path.abspath(mzml_path),
                        '--mode', 'rejected',
                        '--output-dir', os.path.abspath(os.path.join(plot_output_dir, 'envelope_rejected')),
                        '--clear-output',
                    ]
                    _queue_run(
                        cmd, 'Envelope chromatogram plots', '_envelope_plots',
                        needs_visualization=True, input_path=envelope_plot_input,
                        output_path=os.path.join(plot_output_dir, 'envelope_rejected')
                    )
                    _execute_pending_run(stream_container=envelope_run_live)
        with env_col3:
            if st.button('Hide envelope plots', key='hide_envelope_plots'):
                st.session_state.show_envelope_accepted = False
                st.session_state.show_envelope_rejected = False
                st.session_state.envelope_hires_plot = None
                st.session_state.envelope_hires_label = ''

        if 'envelope_hires_plot' not in st.session_state:
            st.session_state.envelope_hires_plot = None
        if 'envelope_hires_label' not in st.session_state:
            st.session_state.envelope_hires_label = ''

        # Single-peptide high-res viewer at TOP when one is selected (full-resolution PNG)
        selected_env_plot = st.session_state.get('envelope_hires_plot')
        if selected_env_plot and os.path.exists(selected_env_plot):
            st.markdown('**Envelope plot — full resolution**')
            if st.button('← Back to gallery', key='close_envelope_hires', type='primary', use_container_width=True):
                st.session_state.envelope_hires_plot = None
                st.session_state.envelope_hires_label = ''
                st.rerun()
            st.caption(st.session_state.get('envelope_hires_label') or os.path.basename(selected_env_plot))
            try:
                env_full_bytes = _read_image_bytes_cached(selected_env_plot, os.path.getmtime(selected_env_plot))
            except Exception:
                env_full_bytes = None
            if env_full_bytes:
                st.image(env_full_bytes, use_container_width=True)
                st.download_button('Download PNG', data=env_full_bytes, file_name=os.path.basename(selected_env_plot), mime='image/png', key='dl_envelope_hires')
            else:
                st.image(selected_env_plot, use_container_width=True)
            if st.button('← Back to gallery', key='close_envelope_hires_bottom', type='primary', use_container_width=True):
                st.session_state.envelope_hires_plot = None
                st.session_state.envelope_hires_label = ''
                st.rerun()
            st.divider()

        def _render_envelope_thumb_gallery(paths: list[str], gallery_key: str, thumbs_per_row: int = 4, thumb_width: int = 200) -> None:
            """One thumbnail per unique-peptide PNG; click Expand for full resolution (not the image expand icon)."""
            for i in range(0, len(paths), thumbs_per_row):
                chunk = paths[i:i + thumbs_per_row]
                cols = st.columns(thumbs_per_row)
                for j, p in enumerate(chunk):
                    if not os.path.exists(p):
                        continue
                    try:
                        mt = os.path.getmtime(p)
                        thumb_bytes = _read_thumbnail_bytes_cached(p, mt, max_width=360)
                    except Exception:
                        thumb_bytes = p
                    with cols[j]:
                        st.image(thumb_bytes, width=thumb_width, caption=os.path.basename(p))
                        # Stable unique key per path so Expand always works across reruns
                        btn_key = f"env_expand_{hashlib.md5(p.encode()).hexdigest()[:12]}"
                        if st.button('Expand (full res)', key=btn_key):
                            st.session_state.envelope_hires_plot = p
                            st.session_state.envelope_hires_label = os.path.basename(p)
                            st.rerun()

        env_accepted, env_rejected = _list_chromatogram_plots_by_status(out_dir, 'envelope')
        if st.session_state.show_envelope_accepted:
            st.markdown('**Accepted envelopes** (one PNG per unique peptide)')
            if env_accepted:
                st.caption('Click **Expand (full res)** below a thumbnail to view that peptide at full resolution above.')
                _render_envelope_thumb_gallery(env_accepted, 'env_accepted', thumbs_per_row=4, thumb_width=200)
            else:
                st.caption('No accepted envelope plots yet. Click generate.')
        if st.session_state.show_envelope_rejected:
            st.markdown('**Rejected envelopes** (one PNG per unique peptide)')
            if env_rejected:
                st.caption('Click **Expand (full res)** below a thumbnail to view that peptide at full resolution above.')
                _render_envelope_thumb_gallery(env_rejected, 'env_rejected', thumbs_per_row=4, thumb_width=200)
            else:
                st.caption('No rejected envelope plots yet. Click generate.')

    with filter_tabs[9]:  # Significance (combined: 5ppm + min sig frags + 1% max, all from summed MS2)
        sig_frag_run_live = st.container()
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
                           '--ppm', str(ppm_thresh), '--min-frac', str(min_frac), '--min-sig-frags', str(min_sig_frag_count),
                           '--min-overhangs', '2']
                    _queue_run(cmd, 'Significant fragmentation', '_significance.csv', needs_visualization=True, input_path=sig_frag_input)
                    _execute_pending_run(stream_container=sig_frag_run_live)
        st.subheader('Significant fragmentation')
        st.caption('5 ppm + min sig frags + min 2 significant single-AA overhangs + 1% max peak (all from summed extracted MS2). Run Extraction first.')
        st.info(
            f"Current significance metrics: PPM tolerance = `{ppm_thresh:.1f}` | "
            f"Min significant fragments = `{int(min_sig_frag_count)}` | "
            f"Min single-AA overhangs = `2` | "
            f"Min fragment intensity = `{sig_frac_pct:.1f}%` of max summed MS2"
        )
        sig_preview = _resolve_preview_output('Significant fragmentation', step_outputs['significance'])
        if not sig_preview or not os.path.exists(sig_preview):
            sig_preview = _latest_csv_with_suffix(out_dir, [
                '_prefilter_extraction_envelope_significance',
                '_confidence_accuracy_extraction_envelope_significance',
                '_confidence_extraction_envelope_significance',
                '_confidence_extraction_significance',
                '_confidence_accuracy_extraction_significance',
                '_prefilter_extraction_test_envelope_significance',
                '_confidence_accuracy_extraction_test_envelope_significance',
                '_confidence_extraction_test_envelope_significance',
                '_confidence_extraction_test_significance',
                '_confidence_accuracy_extraction_test_significance',
                '_significance',
            ])
        if sig_preview:
            _render_step_output_csv(sig_preview, 'sig_frag_output', expanded=(csv_path == sig_preview))
        else:
            st.caption('No Significance output detected yet. Run Significant fragmentation to generate ..._significance.csv.')
        if 'show_significance_plots' not in st.session_state:
            st.session_state.show_significance_plots = False
        sig_btn1, sig_btn2, _ = st.columns([1, 1, 3])
        with sig_btn1:
            if st.button('Load significance plots', key='load_significance_plots'):
                st.session_state.show_significance_plots = True
        with sig_btn2:
            if st.button('Hide significance plots', key='hide_significance_plots'):
                st.session_state.show_significance_plots = False
        if has_sig_frags:
            if st.session_state.show_significance_plots:
                sigfrag_plots = []
                counts = df_orig.apply(
                    lambda row: 0 if pd.isna(row.get(SIGNIFICANT_FRAGS_COL))
                    else len([p for p in str(row.get(SIGNIFICANT_FRAGS_COL)).replace(',', ' ').split() if p.strip()]),
                    axis=1,
                )
                effective_min = max(1, min_sig_frag_count) if enable_sig_frags else min_sig_frag_count
                if has_extraction and 'total_area' in df_orig.columns:
                    valid = df_orig['total_area'].notna() & counts.notna()
                    if valid.sum() > 0:
                        x = df_orig.loc[valid, 'total_area'].values
                        y = counts[valid].values
                        status = [_status_all(idx) for idx in df_orig[valid].index]
                        fig = _scatter_plotly(
                            x, y, color=status,
                            title=f'Sig fragment count vs total area (min={effective_min}) — Pass all filters',
                            xlabel='Total area', ylabel='Sig fragment count', log_x=True
                        )
                        if fig:
                            sigfrag_plots.append((fig, 'sig_area'))
                import plotly.express as px
                plot_df = pd.DataFrame({'Count': counts, 'All filters': [_status_all(idx) for idx in df_orig.index]})
                fig2 = px.histogram(
                    plot_df, x='Count', color='All filters', barmode='stack',
                    title='Sig fragment count — Pass all filters',
                    color_discrete_map={'Fail': 'red', 'Pass': 'dodgerblue'}
                )
                _reorder_traces_smaller_on_top(fig2, plot_df, 'All filters')
                fig2.update_layout(template='plotly_dark', height=PLOT_HEIGHT, margin=dict(l=40, r=20, t=40, b=40))
                sigfrag_plots.append((fig2, 'sig_hist'))
                _plot_grid(sigfrag_plots)
            else:
                st.caption('Click **Load significance plots** to render significance diagnostics.')
        else:
            st.info('No significant_frags column. Run Significant fragmentation.')

    with filter_tabs[10]:  # Sequence
        seq_cov_run_live = st.container()
        st.subheader('Sequence')
        st.caption('Unique peptides grid plots + valuable_sequence/protect_peptide columns. Output: ..._prefilter_extraction_envelope_significance_sequence.csv')
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
                    plot_output_dir = _rotate_plot_version('sequence')
                    seq_cov_dir = os.path.join(plot_output_dir, 'sequence_coverage')
                    os.makedirs(seq_cov_dir, exist_ok=True)
                    cmd = [sys.executable, os.path.join(_SCRIPT_DIR, 'workflow_scripts', 'compare_unique_peptides_sequence.py'),
                           '--input', seq_cov_input, '--output-dir', out_dir,
                           '--sequence-coverage-dir', seq_cov_dir, '--fasta', fasta_path]
                    _queue_run(cmd, 'Sequence coverage', '_sequence.csv', needs_visualization=True, input_path=seq_cov_input)
                    _execute_pending_run(stream_container=seq_cov_run_live)
            seq_output_path = _resolve_preview_output('Sequence coverage', step_outputs.get('sequence'))
            if not seq_output_path or not os.path.exists(seq_output_path):
                seq_output_path = _latest_csv_with_suffix(out_dir, [
                    '_prefilter_extraction_envelope_significance_sequence',
                    '_confidence_extraction_envelope_significance_sequence',
                    '_prefilter_extraction_test_envelope_significance_sequence',
                    '_confidence_extraction_test_envelope_significance_sequence',
                    '_significance_sequence',
                    '_sequence',
                ])
            if seq_output_path and os.path.exists(seq_output_path):
                try:
                    df_seq = load_csv(seq_output_path)
                    overhang_col = (
                        'significant_single_aa_overhangs_protein_positions'
                        if 'significant_single_aa_overhangs_protein_positions' in df_seq.columns
                        else ('single_aa_overhangs_protein_positions' if 'single_aa_overhangs_protein_positions' in df_seq.columns else None)
                    )
                    unique_positions = set()
                    if overhang_col:
                        pos_to_rows = {}
                        for ridx, r in df_seq.iterrows():
                            raw = r.get(overhang_col)
                            if pd.isna(raw) or raw is None:
                                continue
                            for tok in str(raw).split(','):
                                tok = tok.strip()
                                if not tok:
                                    continue
                                if tok[-1].isalpha() and tok[:-1]:
                                    tok = tok[:-1]
                                try:
                                    p = int(float(tok))
                                except Exception:
                                    continue
                                pos_to_rows.setdefault(p, set()).add(ridx)
                        unique_positions = {p for p, rows in pos_to_rows.items() if len(rows) == 1}
                    valuable_col = 'valuable_sequence' if 'valuable_sequence' in df_seq.columns else None
                    if valuable_col:
                        n_valuable = int(pd.to_numeric(df_seq[valuable_col], errors='coerce').fillna(0).astype(int).gt(0).sum())
                    else:
                        n_valuable = 0
                    st.info(
                        f"Sequence summary: unique overhang positions = `{len(unique_positions)}` | "
                        f"valuable peptides = `{n_valuable}` / `{len(df_seq)}`"
                    )
                except Exception:
                    pass
                _render_step_output_csv(seq_output_path, 'seq_cov_sequence', expanded=False)
            else:
                st.caption('No Sequence output detected yet. Run Sequence coverage to generate ..._sequence.csv.')
            if 'show_sequence_plots' not in st.session_state:
                st.session_state.show_sequence_plots = False
            seq_btn1, seq_btn2, _ = st.columns([1, 1, 3])
            with seq_btn1:
                if st.button('Load sequence plots', key='load_sequence_plots'):
                    st.session_state.show_sequence_plots = True
            with seq_btn2:
                if st.button('Hide sequence plots', key='hide_sequence_plots'):
                    st.session_state.show_sequence_plots = False

            if st.session_state.show_sequence_plots:
                def _render_sequence_thumb_gallery(paths: list[str], gallery_key: str, thumbs_per_row: int = 4, thumb_width: int = 180) -> None:
                    for i in range(0, len(paths), thumbs_per_row):
                        chunk = paths[i:i + thumbs_per_row]
                        cols = st.columns(thumbs_per_row)
                        for j, p in enumerate(chunk):
                            if not os.path.exists(p):
                                continue
                            try:
                                mt = os.path.getmtime(p)
                                thumb_bytes = _read_thumbnail_bytes_cached(p, mt, max_width=360)
                            except Exception:
                                thumb_bytes = p
                            with cols[j]:
                                st.image(thumb_bytes, width=thumb_width, caption=os.path.basename(p))

                seq_plot_base_dir = _plot_version_dir('sequence')
                seq_cov_dir = os.path.join(seq_plot_base_dir, 'sequence_coverage')
                seq_cov_plots = []
                if os.path.isdir(seq_cov_dir):
                    seq_cov_plots = [
                        os.path.join(seq_cov_dir, name)
                        for name in sorted(os.listdir(seq_cov_dir))
                        if name.lower().endswith('.png')
                        and 'lowres' not in name.lower()
                    ]
                if seq_cov_plots:
                    st.markdown('**Sequence coverage plots**')
                    _render_sequence_thumb_gallery(seq_cov_plots, 'seq_cov_manual')
                else:
                    st.caption('No sequence coverage PNGs found yet.')

                seq_chrom_dir = os.path.join(seq_plot_base_dir, 'sequence_unique_chromatograms')
                seq_chrom_plots = []
                if os.path.isdir(seq_chrom_dir):
                    for root, _, files in os.walk(seq_chrom_dir):
                        for fn in files:
                            if fn.lower().endswith('.png'):
                                seq_chrom_plots.append(os.path.join(root, fn))
                    seq_chrom_plots = sorted(seq_chrom_plots)
                if seq_chrom_plots:
                    st.markdown('**Sequence chromatogram plots**')
                    _render_sequence_thumb_gallery(seq_chrom_plots, 'seq_chrom_manual')
                else:
                    st.caption('No sequence chromatogram PNGs found yet.')
            else:
                st.caption('Click **Load sequence plots** to render saved sequence plots.')
        except Exception as e:
            st.error(f'Sequence tab error: {e}')
            import traceback
            st.code(traceback.format_exc())

    with filter_tabs[11]:  # Windows
        st.subheader('Windows')
        st.caption('Sequence-passed unique peptides only. Y-axis: peptide key. X-axis: MS1 RT (min). Each peptide window is drawn only during its collection RT range.')
        try:
            sequence_csv = _resolve_preview_output('Sequence coverage', step_outputs.get('sequence'))
            if not sequence_csv or not os.path.exists(sequence_csv):
                sequence_csv = _latest_csv_with_suffix(out_dir, [
                    '_prefilter_extraction_envelope_significance_sequence',
                    '_confidence_extraction_envelope_significance_sequence',
                    '_prefilter_extraction_test_envelope_significance_sequence',
                    '_confidence_extraction_test_envelope_significance_sequence',
                    '_significance_sequence',
                    '_sequence',
                ])

            dataframes_dir = os.path.join(out_dir, 'dataframes')
            metrics_csv_windows = os.path.join(dataframes_dir, 'chromatogram_metrics_all.csv')
            traces_npz_windows = os.path.join(dataframes_dir, 'chromatogram_traces.npz')

            if not sequence_csv or not os.path.exists(sequence_csv):
                st.info('Run Sequence first to generate a ..._sequence.csv output.')
            elif not os.path.exists(metrics_csv_windows) or not os.path.exists(traces_npz_windows):
                st.info('Run Extraction first to generate chromatogram_metrics_all.csv and chromatogram_traces.npz.')
            else:
                try:
                    import plotly.graph_objects as go
                except ImportError:
                    st.warning('Plotly is required for Windows plots.')
                    go = None

                if go is not None:
                    df_seq = load_csv(sequence_csv).copy()
                    df_metrics = pd.read_csv(metrics_csv_windows)

                    def _mods_norm(v):
                        if v is None or (isinstance(v, float) and pd.isna(v)):
                            return '-'
                        s = str(v).strip()
                        return s if s and s.lower() != 'nan' else '-'

                    def _pep_col(df_):
                        for c in ('plain_peptide', 'peptide', 'peptide_sequence', 'sequence'):
                            if c in df_.columns:
                                return c
                        return None

                    seq_pep_col = _pep_col(df_seq)
                    met_pep_col = _pep_col(df_metrics)
                    if not seq_pep_col or not met_pep_col or 'charge' not in df_seq.columns or 'charge' not in df_metrics.columns:
                        st.error('Need peptide + charge columns in both Sequence output and chromatogram metrics.')
                    else:
                        df_seq['_key_pep'] = df_seq[seq_pep_col].astype(str).str.strip()
                        df_seq['_key_charge'] = pd.to_numeric(df_seq['charge'], errors='coerce').fillna(0).astype(int)
                        df_seq['_key_mods'] = df_seq.apply(lambda r: _mods_norm(r.get('modifications', '-')), axis=1)
                        seq_keys = set(zip(df_seq['_key_pep'], df_seq['_key_charge'], df_seq['_key_mods']))

                        df_metrics['_key_pep'] = df_metrics[met_pep_col].astype(str).str.strip()
                        df_metrics['_key_charge'] = pd.to_numeric(df_metrics['charge'], errors='coerce').fillna(0).astype(int)
                        df_metrics['_key_mods'] = df_metrics.apply(lambda r: _mods_norm(r.get('modifications', '-')), axis=1)
                        df_metrics['_key'] = df_metrics.apply(lambda r: (r['_key_pep'], int(r['_key_charge']), r['_key_mods']), axis=1)
                        df_metrics = df_metrics[df_metrics['_key'].isin(seq_keys)].copy()
                        if df_metrics.empty:
                            st.info('No Sequence-passed peptides matched chromatogram metrics.')
                        else:
                            # Keep one representative chromatogram per unique sequence+charge+mods key.
                            if 'trace_index' in df_metrics.columns:
                                df_metrics = df_metrics.sort_values(['_key_pep', '_key_charge', '_key_mods', 'trace_index']).drop_duplicates(subset=['_key'], keep='first')
                            else:
                                df_metrics = df_metrics.drop_duplicates(subset=['_key'], keep='first')

                            rows = []
                            if 'collection_min_rt' not in df_metrics.columns or 'collection_max_rt' not in df_metrics.columns:
                                st.error('Missing required collection-window columns in chromatogram metrics: need collection_min_rt and collection_max_rt for every peptide.')
                            else:
                                missing_collection = int(
                                    (pd.to_numeric(df_metrics['collection_min_rt'], errors='coerce').isna() |
                                     pd.to_numeric(df_metrics['collection_max_rt'], errors='coerce').isna()).sum()
                                )
                                if missing_collection > 0:
                                    st.error(
                                        f'Missing collection-window values for {missing_collection} peptide(s). '
                                        'Every peptide must have collection_min_rt and collection_max_rt.'
                                    )
                                else:
                                    traces_data = np.load(traces_npz_windows)
                                    for _, row in df_metrics.iterrows():
                                        try:
                                            trace_idx = int(row['trace_index']) if 'trace_index' in df_metrics.columns and pd.notna(row.get('trace_index')) else int(_)
                                        except Exception:
                                            continue
                                        rts_key = f'trace_{trace_idx}_rts'
                                        int_key = f'trace_{trace_idx}_int'
                                        if rts_key not in traces_data or int_key not in traces_data:
                                            continue
                                        rts = np.asarray(traces_data[rts_key], dtype=float).flatten()
                                        intensity_matrix = np.asarray(traces_data[int_key], dtype=float)
                                        if intensity_matrix.ndim > 1:
                                            total_int = np.sum(intensity_matrix, axis=1)
                                        else:
                                            total_int = np.asarray(intensity_matrix, dtype=float).flatten()
                                        n = min(len(rts), len(total_int))
                                        if n < 2:
                                            continue
                                        rts = rts[:n]
                                        total_int = total_int[:n]
                                        coll_min = float(row.get('collection_min_rt'))
                                        coll_max = float(row.get('collection_max_rt'))
                                        if coll_max < coll_min:
                                            coll_min, coll_max = coll_max, coll_min
                                        win_mask = (rts >= coll_min) & (rts <= coll_max)
                                        if not np.any(win_mask):
                                            continue
                                        pep = str(row.get(met_pep_col, '')).strip()
                                        ch = int(row.get('charge', 0)) if pd.notna(row.get('charge')) else 0
                                        mods = _mods_norm(row.get('modifications', '-'))
                                        label = f"{pep} +{ch} | {mods}"
                                        rows.append({
                                            'label': label,
                                            'pep': pep,
                                            'charge': ch,
                                            'mods': mods,
                                            'rts_sec': rts,
                                            'rts_min': rts / 60.0,
                                            'ints': total_int,
                                            'mask': win_mask,
                                            'min_rt_sec': float(row.get('detected_peak_min_rt')) if pd.notna(row.get('detected_peak_min_rt')) else coll_min,
                                            'max_rt_sec': float(row.get('detected_peak_max_rt')) if pd.notna(row.get('detected_peak_max_rt')) else coll_max,
                                            'collection_min_rt_sec': coll_min,
                                            'collection_max_rt_sec': coll_max,
                                            'coll_min_min': coll_min / 60.0,
                                            'coll_max_min': coll_max / 60.0,
                                            'seq_start': int(row.get('sequence_start_pos', 999999)) if pd.notna(row.get('sequence_start_pos')) else 999999,
                                            'total_area': float(row.get('total_area')) if pd.notna(row.get('total_area')) else None,
                                        })

                            if not rows:
                                st.info('No usable traces found for Sequence-passed peptides.')
                            else:
                                windows_sort = st.selectbox(
                                    'Order windows by',
                                    ['RT (earliest to latest)', 'Position (N-terminus at top)'],
                                    key='windows_sort_mode'
                                )
                                if windows_sort.startswith('RT'):
                                    rows = sorted(rows, key=lambda d: (d['coll_min_min'], d['seq_start'], d['label']))
                                else:
                                    rows = sorted(rows, key=lambda d: (d['seq_start'], d['coll_min_min'], d['label']))

                                # 1) Windows plot: one horizontal segment per peptide, clipped to collection window.
                                fig_tracks = go.Figure()
                                try:
                                    from visualization.chromatograms import _total_area_colormap_for_overlay
                                except Exception:
                                    _total_area_colormap_for_overlay = None
                                area_norm_win = None
                                cmap_win = None
                                if _total_area_colormap_for_overlay is not None:
                                    valid_areas_win = [
                                        max(0.0, float(r.get('total_area') or 0.0))
                                        for r in rows
                                        if r.get('total_area') is not None
                                    ]
                                    valid_areas_win = [v for v in valid_areas_win if np.isfinite(v)]
                                    if len(valid_areas_win) > 1 and max(valid_areas_win) > min(valid_areas_win):
                                        from matplotlib.colors import Normalize
                                        area_norm_win = Normalize(vmin=min(valid_areas_win), vmax=max(valid_areas_win))
                                        cmap_win = _total_area_colormap_for_overlay()
                                y_order = [r['label'] for r in rows]
                                for i, r in enumerate(rows):
                                    if area_norm_win is not None and cmap_win is not None:
                                        ta = max(0.0, float(r.get('total_area') or 0.0))
                                        win_color = matplotlib.colors.to_hex(cmap_win(area_norm_win(ta)))
                                    else:
                                        win_color = '#8b8b8b'
                                    fig_tracks.add_trace(go.Scatter(
                                        x=[r['coll_min_min'], r['coll_max_min']],
                                        y=[r['label'], r['label']],
                                        mode='lines',
                                        line=dict(width=4, color=win_color),
                                        hovertemplate=(
                                            f"{r['label']}<br>"
                                            f"total_area: {float(r.get('total_area') or 0.0):.2e}<br>"
                                            "Collection window: %{x:.2f} min"
                                            "<extra></extra>"
                                        ),
                                        showlegend=False,
                                    ))
                                fig_tracks.update_layout(
                                    template='plotly_dark',
                                    height=max(350, min(1600, 22 * len(rows))),
                                    margin=dict(l=20, r=20, t=40, b=40),
                                    title=f'MS1 collection windows (Sequence-passed peptides; ordered by {windows_sort})',
                                    xaxis_title='MS1 RT (min)',
                                    yaxis_title='Peptide',
                                    yaxis=dict(type='category', categoryorder='array', categoryarray=y_order, autorange='reversed'),
                                )
                                st.plotly_chart(fig_tracks, width='stretch')

                                # 1b) Iterative channel thinning (valuable_sequence is annotation only).
                                st.markdown('**Channel thinning (iterative; no valuable-sequence protection)**')
                                st.caption(
                                    'Run iterative thinning so overlap at any RT is <= selected channel count. '
                                    'At each conflict RT, remove the lowest relative-area peptide regardless of valuable_sequence. '
                                    'After each thinning pass, valuable_sequence is recomputed on the remaining set for annotation only.'
                                )

                                def _parse_seq_start_for_positions(v):
                                    if v is None or (isinstance(v, float) and pd.isna(v)):
                                        return None
                                    s = str(v).strip()
                                    if not s:
                                        return None
                                    try:
                                        if '-' in s:
                                            return int(s.split('-')[0].strip())
                                        if ',' in s:
                                            return int(s.split(',')[0].strip())
                                        return int(float(s))
                                    except Exception:
                                        return None

                                def _parse_overhang_positions(val, seq_start=None):
                                    pos_set = set()
                                    if val is None or (isinstance(val, float) and pd.isna(val)):
                                        return pos_set
                                    txt = str(val).strip()
                                    if not txt or txt.lower() == 'nan':
                                        return pos_set
                                    for part in txt.split(','):
                                        p = part.strip()
                                        if not p:
                                            continue
                                        num = p[:-1] if p[-1].isalpha() else p
                                        if not num.replace('.', '').replace('-', '').isdigit():
                                            continue
                                        try:
                                            pos = int(float(num))
                                        except Exception:
                                            continue
                                        if seq_start is not None and seq_start > 0 and pos <= 1000:
                                            pos = seq_start + pos - 1
                                        pos_set.add(pos)
                                    return pos_set

                                def _recompute_valuable(entries, accepted_idx):
                                    coverage = {}
                                    for ii in accepted_idx:
                                        for pos in entries[ii].get('overhang_positions', set()):
                                            coverage[pos] = coverage.get(pos, 0) + 1
                                    unique_pos = {p for p, c in coverage.items() if c == 1}
                                    for ii in accepted_idx:
                                        pos = entries[ii].get('overhang_positions', set())
                                        val = 1 if (pos & unique_pos) else 0
                                        entries[ii]['valuable_sequence'] = val
                                        entries[ii]['protect_peptide'] = bool(val)
                                    n_val = sum(int(entries[ii].get('valuable_sequence', 0)) for ii in accepted_idx)
                                    return n_val, len(unique_pos)

                                def _max_overlap(entries, accepted_idx):
                                    worst_count, _, _ = _worst_overlap_segment(entries, accepted_idx)
                                    return int(worst_count)

                                def _worst_overlap_segment(entries, accepted_idx):
                                    """Return (count, mid_t_sec, overlapping_indices) for worst positive-width RT segment."""
                                    if not accepted_idx:
                                        return 0, None, []
                                    pts = []
                                    for ii in accepted_idx:
                                        a = float(entries[ii]['collection_min_rt_sec'])
                                        b = float(entries[ii]['collection_max_rt_sec'])
                                        if b < a:
                                            a, b = b, a
                                        pts.extend([a, b])
                                    pts = sorted(set(pts))
                                    if len(pts) < 2:
                                        return len(accepted_idx), None, list(accepted_idx)
                                    worst_count = 0
                                    worst_t = None
                                    worst_overlap = []
                                    for j in range(len(pts) - 1):
                                        left = float(pts[j])
                                        right = float(pts[j + 1])
                                        if right <= left:
                                            continue
                                        mid = 0.5 * (left + right)
                                        ov = []
                                        for ii in accepted_idx:
                                            e = entries[ii]
                                            lo = float(e['collection_min_rt_sec'])
                                            hi = float(e['collection_max_rt_sec'])
                                            if hi < lo:
                                                lo, hi = hi, lo
                                            # Positive-width overlap on segment interior.
                                            if lo < mid < hi:
                                                ov.append(ii)
                                        if len(ov) > worst_count:
                                            worst_count = len(ov)
                                            worst_t = mid
                                            worst_overlap = ov
                                    return int(worst_count), worst_t, worst_overlap

                                run_windows_thin = st.button(
                                    'Run window channel thinning',
                                    key='run_windows_channel_thinning',
                                    help='Iteratively thin overlap by channel limit, preserving valuable_sequence peptides.',
                                )

                                if run_windows_thin:
                                    # Add sequence-coverage metadata (valuable_sequence + overhang positions) onto rows by peptide key.
                                    overhang_col = (
                                        'significant_single_aa_overhangs_protein_positions'
                                        if 'significant_single_aa_overhangs_protein_positions' in df_seq.columns
                                        else ('single_aa_overhangs_protein_positions' if 'single_aa_overhangs_protein_positions' in df_seq.columns else None)
                                    )
                                    seq_start_col = 'sequence_start_pos' if 'sequence_start_pos' in df_seq.columns else ('sequence_positions' if 'sequence_positions' in df_seq.columns else None)
                                    val_col = 'valuable_sequence' if 'valuable_sequence' in df_seq.columns else None

                                    seq_meta = {}
                                    for _, srow in df_seq.iterrows():
                                        k = (
                                            str(srow.get('_key_pep', '')).strip(),
                                            int(pd.to_numeric(srow.get('_key_charge'), errors='coerce') or 0),
                                            _mods_norm(srow.get('_key_mods', '-')),
                                        )
                                        start_val = _parse_seq_start_for_positions(srow.get(seq_start_col)) if seq_start_col else None
                                        positions = _parse_overhang_positions(srow.get(overhang_col), seq_start=start_val) if overhang_col else set()
                                        meta = seq_meta.get(k, {'positions': set(), 'valuable_init': 0})
                                        meta['positions'] |= positions
                                        init_val = int(pd.to_numeric(pd.Series([srow.get(val_col, 0)]), errors='coerce').fillna(0).astype(int).iloc[0] > 0) if val_col else 0
                                        meta['valuable_init'] = max(meta['valuable_init'], init_val)
                                        seq_meta[k] = meta

                                    entries = []
                                    for r in rows:
                                        key = (str(r.get('pep', '')).strip(), int(r.get('charge', 0) or 0), _mods_norm(r.get('mods', '-')))
                                        meta = seq_meta.get(key, {'positions': set(), 'valuable_init': 0})
                                        e = dict(r)
                                        e['key'] = key
                                        e['overhang_positions'] = set(meta.get('positions', set()))
                                        e['valuable_sequence'] = int(meta.get('valuable_init', 0))
                                        e['protect_peptide'] = bool(meta.get('valuable_init', 0))
                                        e['removed_reason'] = ''
                                        e['removed_iteration'] = None
                                        entries.append(e)

                                    accepted = set(range(len(entries)))
                                    unresolvable_conflicts = 0
                                    max_iter = max(5, len(entries) * 3)
                                    thinning_logs = []

                                    def _log(msg):
                                        thinning_logs.append(msg)
                                        print(f"[Windows thinning] {msg}", flush=True)

                                    _log(f"Starting iterative thinning with channel limit = {int(channels_count)} across {len(entries)} peptides.")
                                    for iter_idx in range(1, max_iter + 1):
                                        if not accepted:
                                            _log(f"Iteration {iter_idx}: no accepted peptides remain, stopping.")
                                            break
                                        _log(f"Iteration {iter_idx}: scanning valuable sequence evaluation on {len(accepted)} current peptides.")
                                        n_val, n_unique_pos = _recompute_valuable(entries, accepted)
                                        _log(
                                            f"Iteration {iter_idx}: valuable sequence evaluation complete "
                                            f"(valuable_sequence=1 for {n_val}, unique overhang positions={n_unique_pos})."
                                        )
                                        areas = {ii: max(0.0, float(entries[ii].get('total_area') or 0.0)) for ii in accepted}
                                        area_sum = sum(areas.values())
                                        rel = {ii: (areas[ii] / area_sum if area_sum > 0 else 0.0) for ii in accepted}
                                        changed = False
                                        _log(f"Iteration {iter_idx}: scanning channel selection conflicts across RT windows.")
                                        while True:
                                            worst_count, worst_t, worst_overlap = _worst_overlap_segment(entries, sorted(accepted))
                                            if worst_count <= int(channels_count):
                                                break
                                            candidates = list(worst_overlap)
                                            if not candidates:
                                                unresolvable_conflicts += 1
                                                _log(
                                                    f"Iteration {iter_idx}: conflict at RT={float(worst_t or 0)/60.0:.2f} min has {len(worst_overlap)} overlaps, "
                                                    "but no removable candidates were found."
                                                )
                                                break
                                            drop_idx = min(
                                                candidates,
                                                key=lambda ii: (rel.get(ii, 0.0), areas.get(ii, 0.0), str(entries[ii].get('label', '')))
                                            )
                                            accepted.remove(drop_idx)
                                            _log(
                                                f"Iteration {iter_idx}: removed peptide '{entries[drop_idx].get('label', 'unknown')}' "
                                                f"at RT={float(worst_t or 0)/60.0:.2f} min (relative_area={rel.get(drop_idx, 0.0):.4f}, "
                                                f"total_area={areas.get(drop_idx, 0.0):.2f})."
                                            )
                                            entries[drop_idx]['removed_reason'] = (
                                                f'overlap>{int(channels_count)} and lowest relative area '
                                                f'at RT={float(worst_t or 0)/60.0:.2f} min'
                                            )
                                            entries[drop_idx]['removed_iteration'] = iter_idx
                                            changed = True
                                        if not changed:
                                            _log(f"Iteration {iter_idx}: no removals in channel selection scan, convergence reached.")
                                            break

                                    if accepted:
                                        _log(f"Final pass: scanning valuable sequence evaluation on {len(accepted)} accepted peptides.")
                                        n_val_final, n_unique_final = _recompute_valuable(entries, accepted)
                                        _log(
                                            f"Final pass: valuable sequence evaluation complete "
                                            f"(valuable_sequence=1 for {n_val_final}, unique overhang positions={n_unique_final})."
                                        )
                                    accepted_idx = sorted(list(accepted))
                                    rejected_idx = sorted([i for i in range(len(entries)) if i not in accepted])
                                    max_ov = _max_overlap(entries, accepted_idx)
                                    _log(
                                        f"Finished thinning: accepted={len(accepted_idx)}, rejected={len(rejected_idx)}, "
                                        f"max_overlap_after={int(max_ov)}, unresolvable_conflicts={int(unresolvable_conflicts)}."
                                    )

                                    # Persist accepted/rejected CSVs with updated valuable_sequence designation.
                                    seq_out = df_seq.copy()
                                    status_by_key = {}
                                    for ii, e in enumerate(entries):
                                        status_by_key[e['key']] = {
                                            'accepted': (ii in accepted),
                                            'valuable_sequence': int(e.get('valuable_sequence', 0)),
                                            'protect_peptide': bool(e.get('protect_peptide', False)),
                                            'removed_reason': e.get('removed_reason', ''),
                                        }
                                    seq_out['windows_channel_thinning_status'] = seq_out.apply(
                                        lambda rr: (
                                            'accepted'
                                            if status_by_key.get(
                                                (str(rr.get('_key_pep', '')).strip(), int(pd.to_numeric(rr.get('_key_charge'), errors='coerce') or 0), _mods_norm(rr.get('_key_mods', '-'))),
                                                {'accepted': False}
                                            )['accepted']
                                            else 'rejected'
                                        ),
                                        axis=1,
                                    )
                                    seq_out['valuable_sequence'] = seq_out.apply(
                                        lambda rr: status_by_key.get(
                                            (str(rr.get('_key_pep', '')).strip(), int(pd.to_numeric(rr.get('_key_charge'), errors='coerce') or 0), _mods_norm(rr.get('_key_mods', '-'))),
                                            {'valuable_sequence': int(pd.to_numeric(pd.Series([rr.get('valuable_sequence', 0)]), errors='coerce').fillna(0).astype(int).iloc[0])}
                                        )['valuable_sequence'],
                                        axis=1,
                                    )
                                    seq_out['protect_peptide'] = seq_out.apply(
                                        lambda rr: bool(status_by_key.get(
                                            (str(rr.get('_key_pep', '')).strip(), int(pd.to_numeric(rr.get('_key_charge'), errors='coerce') or 0), _mods_norm(rr.get('_key_mods', '-'))),
                                            {'protect_peptide': False}
                                        )['protect_peptide']),
                                        axis=1,
                                    )
                                    seq_out['windows_reject_reason'] = seq_out.apply(
                                        lambda rr: status_by_key.get(
                                            (str(rr.get('_key_pep', '')).strip(), int(pd.to_numeric(rr.get('_key_charge'), errors='coerce') or 0), _mods_norm(rr.get('_key_mods', '-'))),
                                            {'removed_reason': ''}
                                        )['removed_reason'],
                                        axis=1,
                                    )

                                    seq_base = os.path.splitext(os.path.basename(sequence_csv))[0]
                                    accepted_csv = os.path.join(out_dir, f'{seq_base}_windows_channels_accepted.csv')
                                    rejected_csv = os.path.join(out_dir, f'{seq_base}_windows_channels_rejected.csv')
                                    seq_out[seq_out['windows_channel_thinning_status'] == 'accepted'].to_csv(accepted_csv, index=False)
                                    seq_out[seq_out['windows_channel_thinning_status'] == 'rejected'].to_csv(rejected_csv, index=False)

                                    st.session_state.windows_thinning_result = {
                                        'accepted_csv': accepted_csv,
                                        'rejected_csv': rejected_csv,
                                        'channels_count': int(channels_count),
                                        'n_input': len(entries),
                                        'n_accepted': len(accepted_idx),
                                        'n_rejected': len(rejected_idx),
                                        'max_overlap_after': int(max_ov),
                                        'unresolvable_conflicts': int(unresolvable_conflicts),
                                        'accepted_rows': [entries[i] for i in accepted_idx],
                                        'rejected_rows': [entries[i] for i in rejected_idx],
                                        'log_lines': thinning_logs,
                                    }

                                thin_res = st.session_state.get('windows_thinning_result')
                                if thin_res:
                                    st.caption(
                                        f"Thinning summary: kept {thin_res.get('n_accepted', 0)} / {thin_res.get('n_input', 0)} "
                                        f"(rejected {thin_res.get('n_rejected', 0)}), "
                                        f"max overlap after thinning = {thin_res.get('max_overlap_after', 0)} "
                                        f"with channel limit = {thin_res.get('channels_count', int(channels_count))}."
                                    )
                                    if int(thin_res.get('unresolvable_conflicts', 0)) > 0:
                                        st.warning(
                                            f"{thin_res.get('unresolvable_conflicts', 0)} overlap region(s) could not be reduced "
                                            "because no removable candidates were found."
                                        )
                                    if thin_res.get('accepted_csv') and os.path.exists(thin_res['accepted_csv']):
                                        st.markdown(f"- Accepted CSV: `{thin_res['accepted_csv']}`")
                                    if thin_res.get('rejected_csv') and os.path.exists(thin_res['rejected_csv']):
                                        st.markdown(f"- Rejected CSV: `{thin_res['rejected_csv']}`")
                                    logs = thin_res.get('log_lines', [])
                                    if logs:
                                        with st.expander('Windows thinning log', expanded=False):
                                            st.text_area(
                                                'Windows thinning log output',
                                                value='\n'.join(logs),
                                                height=260,
                                                key='windows_thinning_logs_view',
                                                disabled=True,
                                                label_visibility='collapsed',
                                            )

                                # 2) Extraction-style overlay PNGs (linear/log/tracks), restricted to this subset.
                                st.markdown('**Extraction-style overlay plots (subset only)**')
                                st.caption('These are the same overlay plot types created during Extraction, regenerated here for the current Sequence-passed subset.')
                                try:
                                    from visualization.chromatograms import _create_overlay_figures, _create_zoom_all_linear_log_tracks_from_overlay
                                    if 'plot_version_windows' not in st.session_state:
                                        st.session_state.plot_version_windows = datetime.now().strftime('%Y%m%d_%H%M%S')
                                    windows_plot_dir = os.path.join(plots_root_dir, 'windows', str(st.session_state.plot_version_windows))
                                    os.makedirs(windows_plot_dir, exist_ok=True)

                                    overlay_subset = []
                                    for r in rows:
                                        overlay_subset.append({
                                            'rts': np.asarray(r['rts_sec'], dtype=float),
                                            'rts_min': np.asarray(r['rts_min'], dtype=float),
                                            'total_intensities': np.asarray(r['ints'], dtype=float),
                                            'label': r['label'],
                                            'total_area': r.get('total_area'),
                                            'sequence_start_pos': r.get('seq_start', 999999),
                                            'min_rt': r.get('min_rt_sec'),
                                            'max_rt': r.get('max_rt_sec'),
                                            'collection_min_rt': r.get('collection_min_rt_sec'),
                                            'collection_max_rt': r.get('collection_max_rt_sec'),
                                        })

                                    regen = st.button(
                                        'Regenerate extraction-style overlays (subset)',
                                        key='regen_windows_subset_overlays',
                                        help='Create linear/log/tracks overlay PNGs for only the Sequence-passed subset.'
                                    )
                                    linear_png = os.path.join(windows_plot_dir, 'windows_subset_chromatograms_zoom_peptides_all_linear.png')
                                    log_png = os.path.join(windows_plot_dir, 'windows_subset_chromatograms_zoom_peptides_all_log.png')
                                    tracks_png = os.path.join(windows_plot_dir, 'windows_subset_chromatograms_zoom_peptides_all_tracks.png')
                                    combined_png = os.path.join(
                                        windows_plot_dir,
                                        'windows_subset_chromatograms_zoom_peptides_all_linear_log_tracks.png',
                                    )
                                    if regen or not (
                                        os.path.exists(linear_png)
                                        and os.path.exists(log_png)
                                        and os.path.exists(tracks_png)
                                    ) and not os.path.exists(combined_png):
                                        _create_overlay_figures(overlay_subset, windows_plot_dir)
                                        _create_zoom_all_linear_log_tracks_from_overlay(
                                            overlay_subset, windows_plot_dir, 'windows_subset_chromatograms'
                                        )

                                    if os.path.exists(linear_png) and os.path.exists(log_png) and os.path.exists(tracks_png):
                                        show_paths = [linear_png, log_png, tracks_png]
                                        labels = ['All peptides overlay (linear)', 'All peptides overlay (log)', 'All peptides overlay (tracks)']
                                        for p, t in zip(show_paths, labels):
                                            st.markdown(f'**{t}**')
                                            st.image(p, use_container_width=True)
                                    elif os.path.exists(combined_png):
                                        st.markdown('**All peptides overlay (linear + log + tracks)**')
                                        st.image(combined_png, use_container_width=True)
                                    else:
                                        show_paths = [linear_png, log_png, tracks_png, combined_png]
                                        for p in show_paths:
                                            st.caption(f'Missing plot: {os.path.basename(p)}')

                                    # Generate accepted/rejected overlays after window channel thinning.
                                    thin_res = st.session_state.get('windows_thinning_result')
                                    if thin_res and (thin_res.get('accepted_rows') is not None or thin_res.get('rejected_rows') is not None):
                                        st.markdown('**Post-thinning subset overlays**')
                                        acc_col, rej_col = st.columns(2)
                                        with acc_col:
                                            show_acc_overlay = st.button(
                                                'Show Accepted overlay (linear + log + tracks)',
                                                key='show_windows_thinning_overlay_accepted',
                                                help='Generate/show combined all-peptides overlay for accepted subset.',
                                            )
                                            show_acc_zoom_segments = st.button(
                                                'Show Accepted zoom segments',
                                                key='show_windows_thinning_overlay_accepted_segments',
                                                help='Generate/show per-segment accepted overlays for closer overlap inspection.',
                                            )
                                        with rej_col:
                                            show_rej_overlay = st.button(
                                                'Show Rejected overlay (linear + log + tracks)',
                                                key='show_windows_thinning_overlay_rejected',
                                                help='Generate/show combined all-peptides overlay for rejected subset.',
                                            )

                                        def _render_thinning_subset_overlay(subset_name, subset_rows, force_regen=False):
                                            if not subset_rows:
                                                st.caption(f'No {subset_name} peptides available after thinning.')
                                                return
                                            subset_dir = os.path.join(windows_plot_dir, 'channel_thinning', subset_name)
                                            os.makedirs(subset_dir, exist_ok=True)
                                            overlay_subset = []
                                            for r in subset_rows:
                                                overlay_subset.append({
                                                    'rts': np.asarray(r['rts_sec'], dtype=float),
                                                    'rts_min': np.asarray(r['rts_min'], dtype=float),
                                                    'total_intensities': np.asarray(r['ints'], dtype=float),
                                                    'label': r['label'],
                                                    'total_area': r.get('total_area'),
                                                    'sequence_start_pos': r.get('seq_start', 999999),
                                                    'min_rt': r.get('min_rt_sec'),
                                                    'max_rt': r.get('max_rt_sec'),
                                                    'collection_min_rt': r.get('collection_min_rt_sec'),
                                                    'collection_max_rt': r.get('collection_max_rt_sec'),
                                                })
                                            base_nm = f'windows_{subset_name}_chromatograms'
                                            comb_png = os.path.join(subset_dir, f'{base_nm}_zoom_peptides_all_linear_log_tracks.png')
                                            if force_regen or regen or not os.path.exists(comb_png):
                                                _create_overlay_figures(overlay_subset, subset_dir)
                                                _create_zoom_all_linear_log_tracks_from_overlay(
                                                    overlay_subset,
                                                    subset_dir,
                                                    base_nm,
                                                    add_peak_labels=True,
                                                )
                                            if os.path.exists(comb_png):
                                                st.markdown(f'**{subset_name.capitalize()} subset overlay (linear + log + tracks)**')
                                                st.image(comb_png, use_container_width=True)
                                            else:
                                                st.caption(f'Missing plot: {os.path.basename(comb_png)}')

                                        def _build_collection_segments(subset_rows):
                                            intervals = []
                                            for rr in subset_rows:
                                                a = float(rr.get('collection_min_rt_sec', 0.0))
                                                b = float(rr.get('collection_max_rt_sec', 0.0))
                                                if b < a:
                                                    a, b = b, a
                                                intervals.append((a, b))
                                            if not intervals:
                                                return []
                                            intervals.sort(key=lambda t: t[0])
                                            merged = [list(intervals[0])]
                                            for a, b in intervals[1:]:
                                                if a <= merged[-1][1]:
                                                    merged[-1][1] = max(merged[-1][1], b)
                                                else:
                                                    merged.append([a, b])
                                            return [(float(a), float(b)) for a, b in merged]

                                        def _segment_max_overlap(subset_rows, seg_start, seg_end):
                                            pts = [seg_start, seg_end]
                                            for rr in subset_rows:
                                                a = float(rr.get('collection_min_rt_sec', 0.0))
                                                b = float(rr.get('collection_max_rt_sec', 0.0))
                                                if b < a:
                                                    a, b = b, a
                                                if b < seg_start or a > seg_end:
                                                    continue
                                                pts.extend([max(seg_start, a), min(seg_end, b)])
                                            pts = sorted(set(pts))
                                            if len(pts) < 2:
                                                return len(subset_rows)
                                            m = 0
                                            for j in range(len(pts) - 1):
                                                left, right = float(pts[j]), float(pts[j + 1])
                                                if right <= left:
                                                    continue
                                                mid = 0.5 * (left + right)
                                                c = 0
                                                for rr in subset_rows:
                                                    a = float(rr.get('collection_min_rt_sec', 0.0))
                                                    b = float(rr.get('collection_max_rt_sec', 0.0))
                                                    if b < a:
                                                        a, b = b, a
                                                    if a < mid < b:
                                                        c += 1
                                                m = max(m, c)
                                            return int(m)

                                        def _render_accepted_zoom_segments(subset_rows, force_regen=False):
                                            if not subset_rows:
                                                st.caption('No accepted peptides available for segment zoom plots.')
                                                return
                                            segments = _build_collection_segments(subset_rows)
                                            if not segments:
                                                st.caption('No valid accepted collection-window segments found.')
                                                return
                                            seg_dir = os.path.join(windows_plot_dir, 'channel_thinning', 'accepted', 'segments')
                                            os.makedirs(seg_dir, exist_ok=True)
                                            st.markdown('**Accepted overlay zoom segments**')
                                            for seg_idx, (seg_start, seg_end) in enumerate(segments, start=1):
                                                seg_rows = []
                                                for rr in subset_rows:
                                                    a = float(rr.get('collection_min_rt_sec', 0.0))
                                                    b = float(rr.get('collection_max_rt_sec', 0.0))
                                                    if b < a:
                                                        a, b = b, a
                                                    if not (b < seg_start or a > seg_end):
                                                        seg_rows.append(rr)
                                                if not seg_rows:
                                                    continue
                                                overlay_subset = []
                                                for r in seg_rows:
                                                    overlay_subset.append({
                                                        'rts': np.asarray(r['rts_sec'], dtype=float),
                                                        'rts_min': np.asarray(r['rts_min'], dtype=float),
                                                        'total_intensities': np.asarray(r['ints'], dtype=float),
                                                        'label': r['label'],
                                                        'total_area': r.get('total_area'),
                                                        'sequence_start_pos': r.get('seq_start', 999999),
                                                        'min_rt': r.get('min_rt_sec'),
                                                        'max_rt': r.get('max_rt_sec'),
                                                        'collection_min_rt': r.get('collection_min_rt_sec'),
                                                        'collection_max_rt': r.get('collection_max_rt_sec'),
                                                    })
                                                base_nm = f'windows_accepted_segment_{seg_idx:02d}'
                                                seg_png = os.path.join(seg_dir, f'{base_nm}_zoom_peptides_all_linear_log_tracks.png')
                                                if force_regen or regen or not os.path.exists(seg_png):
                                                    _create_overlay_figures(overlay_subset, seg_dir)
                                                    _create_zoom_all_linear_log_tracks_from_overlay(
                                                        overlay_subset,
                                                        seg_dir,
                                                        base_nm,
                                                        add_peak_labels=True,
                                                    )
                                                seg_max_overlap = _segment_max_overlap(seg_rows, seg_start, seg_end)
                                                st.caption(
                                                    f"Segment {seg_idx}: RT {seg_start/60.0:.2f}-{seg_end/60.0:.2f} min | "
                                                    f"peptides={len(seg_rows)} | max overlap={seg_max_overlap}"
                                                )
                                                if os.path.exists(seg_png):
                                                    st.image(seg_png, use_container_width=True)
                                                else:
                                                    st.caption(f'Missing plot: {os.path.basename(seg_png)}')

                                        if show_acc_overlay:
                                            _render_thinning_subset_overlay('accepted', thin_res.get('accepted_rows', []), force_regen=True)
                                        if show_acc_zoom_segments:
                                            _render_accepted_zoom_segments(thin_res.get('accepted_rows', []), force_regen=True)
                                        if show_rej_overlay:
                                            _render_thinning_subset_overlay('rejected', thin_res.get('rejected_rows', []), force_regen=True)
                                except Exception as e:
                                    st.warning(f'Could not generate extraction-style subset overlays: {e}')
        except Exception as e:
            st.error(f'Windows tab error: {e}')
            import traceback
            st.code(traceback.format_exc())

    with filter_tabs[14]:  # D_MZ (Estimated Deuterated MZ target range)
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
        _render_terminal('dmz')

    with filter_tabs[15]:  # SF (Source Fragmentation)
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
        _render_terminal('sf')

    with filter_tabs[13]:  # Inspect
        st.subheader('Inspect')
        st.caption('Individual extraction plots for peptides that survived Windows thinning and channel assignment.')
        ch_res = st.session_state.get('windows_accepted_channels_result')
        channelized_csv = ch_res.get('out_csv') if isinstance(ch_res, dict) else None
        if not channelized_csv or not os.path.exists(channelized_csv):
            channelized_csv = _latest_csv_with_suffix(out_dir, ['_channels'])
        dataframes_dir = os.path.join(out_dir, 'dataframes')
        metrics_csv = os.path.join(dataframes_dir, 'chromatogram_metrics_all.csv')
        traces_path = os.path.join(dataframes_dir, 'chromatogram_traces.npz')

        if not channelized_csv or not os.path.exists(channelized_csv):
            st.info('Run **Channels** assignment first to create the channelized accepted CSV.')
        elif not (os.path.exists(metrics_csv) and os.path.exists(traces_path)):
            st.info('Run Extraction first to generate chromatogram_metrics_all.csv and chromatogram_traces.npz.')
        else:
            if 'show_inspect_plots' not in st.session_state:
                st.session_state.show_inspect_plots = False
            if 'inspect_hires_plot' not in st.session_state:
                st.session_state.inspect_hires_plot = None
            if 'inspect_hires_label' not in st.session_state:
                st.session_state.inspect_hires_label = ''

            col_gen, col_hide = st.columns([1, 1])
            with col_gen:
                if st.button(
                    'Generate inspect plots',
                    key='gen_inspect_plots',
                    help='Generate individual extraction plots for channel-assigned accepted peptides only.',
                ):
                    inspect_plot_dir = _rotate_plot_version('inspect')
                    st.session_state.show_inspect_plots = True
                    st.session_state.inspect_hires_plot = None
                    st.session_state.inspect_hires_label = ''
                    plot_cmd = [
                        sys.executable,
                        os.path.join(_SCRIPT_DIR, 'workflow_scripts', 'plot_chromatograms_from_extraction.py'),
                        '--output-dir', os.path.abspath(inspect_plot_dir),
                        '--chromatogram-metrics-csv', os.path.abspath(metrics_csv),
                        '--chromatogram-traces', os.path.abspath(traces_path),
                        '--clear-output',
                        '--filter-csv', os.path.abspath(channelized_csv),
                    ]
                    _queue_run(plot_cmd, 'Inspect chromatogram plots', '_inspect_chromatograms', needs_visualization=True)
                    st.caption(f'Inspect plot output dir: `{inspect_plot_dir}`')
            with col_hide:
                if st.button('Hide inspect plots', key='hide_inspect_plots'):
                    st.session_state.show_inspect_plots = False
                    st.session_state.inspect_hires_plot = None
                    st.session_state.inspect_hires_label = ''

            inspect_plot_dir = _plot_version_dir('inspect')

            def _list_inspect_individual_plots(base_dir: str) -> list[str]:
                if not base_dir or not os.path.isdir(base_dir):
                    return []
                out = []
                for root, _, files in os.walk(base_dir):
                    for f in files:
                        fl = f.lower()
                        if not fl.endswith(('.png', '.jpg', '.jpeg')):
                            continue
                        if 'overlay' in fl or 'zoom_peptides' in fl or '_rejected' in fl:
                            continue
                        out.append(os.path.join(root, f))
                def _rt_key(path: str):
                    name = os.path.basename(path).lower()
                    m = re.search(r'_rt([0-9]+(?:p[0-9]+)?)_', name)
                    if m:
                        try:
                            return float(m.group(1).replace('p', '.'))
                        except Exception:
                            return float('inf')
                    return float('inf')
                return sorted(set(out), key=lambda p: (_rt_key(p), os.path.basename(p).lower()))

            def _render_inspect_thumb_gallery(paths: list[str], gallery_key: str, thumbs_per_row: int = 3, thumb_width: int = 240) -> None:
                for i in range(0, len(paths), thumbs_per_row):
                    chunk = paths[i:i + thumbs_per_row]
                    cols = st.columns(thumbs_per_row)
                    for j, p in enumerate(chunk):
                        if not os.path.exists(p):
                            continue
                        try:
                            mt = os.path.getmtime(p)
                            thumb_bytes = _read_thumbnail_bytes_cached(p, mt, max_width=360)
                        except Exception:
                            thumb_bytes = p
                        with cols[j]:
                            st.image(thumb_bytes, width=thumb_width, caption=os.path.basename(p))
                            if st.button('Expand', key=f'{gallery_key}_expand_{i}_{j}_{os.path.basename(p)}'):
                                st.session_state.inspect_hires_plot = p
                                st.session_state.inspect_hires_label = os.path.basename(p)

            if st.session_state.get('show_inspect_plots', False):
                inspect_paths = _list_inspect_individual_plots(inspect_plot_dir)
                if inspect_paths:
                    st.caption(f"Showing {len(inspect_paths)} individual accepted/channel-assigned extraction plots.")
                    _render_inspect_thumb_gallery(inspect_paths, 'inspect_gallery')
                else:
                    st.info('Click **Generate inspect plots** to build individual plots for the channel-assigned accepted subset.')

                selected_plot = st.session_state.get('inspect_hires_plot')
                if selected_plot and os.path.exists(selected_plot):
                    plot_label = st.session_state.get('inspect_hires_label') or os.path.basename(selected_plot)
                    st.markdown('**Inspect high-resolution viewer**')
                    if st.button('← Back to inspect gallery', key='close_inspect_hires', type='primary', use_container_width=True):
                        st.session_state.inspect_hires_plot = None
                        st.session_state.inspect_hires_label = ''
                        st.rerun()
                    st.caption(plot_label)
                    try:
                        original_bytes = _read_image_bytes_cached(selected_plot, os.path.getmtime(selected_plot))
                    except Exception:
                        original_bytes = None
                    if original_bytes:
                        st.image(original_bytes, use_container_width=True)
                    else:
                        st.image(selected_plot, use_container_width=True)

            _render_terminal('inspect')

    with filter_tabs[12]:  # Channels
        st.subheader('Channels')
        st.caption('Assign accepted Windows-thinning peptides to channels so no peptides overlap within a channel (collection-window RT).')
        thin_res = st.session_state.get('windows_thinning_result')
        windows_accepted_csv = _latest_csv_with_suffix(out_dir, ['_windows_channels_accepted'])
        if not thin_res and windows_accepted_csv and os.path.exists(windows_accepted_csv):
            st.caption(f'Loaded saved Windows accepted set: `{os.path.basename(windows_accepted_csv)}`')
        if (not thin_res or not thin_res.get('accepted_rows')) and not (windows_accepted_csv and os.path.exists(windows_accepted_csv)):
            st.info('Run Windows channel thinning first, then return here to assign channels on the accepted subset.')
        else:
            try:
                import plotly.graph_objects as go
            except ImportError:
                st.warning('Plotly is required for Channels (Accepted) visualization.')
                go = None

            accepted_rows = thin_res.get('accepted_rows', []) if thin_res else []
            if not accepted_rows and windows_accepted_csv and os.path.exists(windows_accepted_csv):
                try:
                    df_acc_in = load_csv(windows_accepted_csv).copy()
                    dataframes_dir = os.path.join(out_dir, 'dataframes')
                    metrics_csv = os.path.join(dataframes_dir, 'chromatogram_metrics_all.csv')
                    traces_path = os.path.join(dataframes_dir, 'chromatogram_traces.npz')
                    if os.path.exists(metrics_csv) and os.path.exists(traces_path):
                        df_metrics = pd.read_csv(metrics_csv)
                        traces_data = np.load(traces_path)
                        if 'peptide_sequence' in df_metrics.columns and 'plain_peptide' not in df_metrics.columns:
                            df_metrics['plain_peptide'] = df_metrics['peptide_sequence']
                        if 'peptide_sequence' in df_acc_in.columns and 'plain_peptide' not in df_acc_in.columns:
                            df_acc_in['plain_peptide'] = df_acc_in['peptide_sequence']
                        df_acc_in['_key_pep'] = df_acc_in.get('_key_pep', df_acc_in.get('plain_peptide', pd.Series([''] * len(df_acc_in)))).astype(str).str.strip()
                        df_acc_in['_key_charge'] = pd.to_numeric(df_acc_in.get('_key_charge', df_acc_in.get('charge', 0)), errors='coerce').fillna(0).astype(int)
                        df_acc_in['_key_mods'] = df_acc_in.apply(lambda r: _mods_norm(r.get('_key_mods', r.get('modifications', '-'))), axis=1)
                        acc_keys = set(zip(df_acc_in['_key_pep'], df_acc_in['_key_charge'], df_acc_in['_key_mods']))
                        df_metrics['_key_pep'] = df_metrics.get('plain_peptide', pd.Series([''] * len(df_metrics))).astype(str).str.strip()
                        df_metrics['_key_charge'] = pd.to_numeric(df_metrics.get('charge', 0), errors='coerce').fillna(0).astype(int)
                        df_metrics['_key_mods'] = df_metrics.apply(lambda r: _mods_norm(r.get('modifications', '-')), axis=1)
                        df_metrics['_key'] = df_metrics.apply(lambda r: (r['_key_pep'], int(r['_key_charge']), r['_key_mods']), axis=1)
                        df_metrics = df_metrics[df_metrics['_key'].isin(acc_keys)].copy()
                        if 'trace_index' in df_metrics.columns:
                            df_metrics = df_metrics.sort_values(['_key_pep', '_key_charge', '_key_mods', 'trace_index']).drop_duplicates(subset=['_key'], keep='first')
                        else:
                            df_metrics = df_metrics.drop_duplicates(subset=['_key'], keep='first')
                        for _, row in df_metrics.iterrows():
                            try:
                                trace_idx = int(row['trace_index']) if 'trace_index' in df_metrics.columns and pd.notna(row.get('trace_index')) else int(_)
                            except Exception:
                                continue
                            rts_key = f'trace_{trace_idx}_rts'
                            int_key = f'trace_{trace_idx}_int'
                            if rts_key not in traces_data or int_key not in traces_data:
                                continue
                            rts = np.asarray(traces_data[rts_key], dtype=float).flatten()
                            intensity_matrix = np.asarray(traces_data[int_key], dtype=float)
                            total_int = np.sum(intensity_matrix, axis=1) if intensity_matrix.ndim > 1 else np.asarray(intensity_matrix, dtype=float).flatten()
                            n = min(len(rts), len(total_int))
                            if n < 2:
                                continue
                            rts = rts[:n]
                            total_int = total_int[:n]
                            coll_min = row.get('collection_min_rt')
                            coll_max = row.get('collection_max_rt')
                            if pd.isna(coll_min) or pd.isna(coll_max):
                                continue
                            coll_min = float(coll_min); coll_max = float(coll_max)
                            if coll_max < coll_min:
                                coll_min, coll_max = coll_max, coll_min
                            pep = str(row.get('plain_peptide', '')).strip()
                            ch = int(row.get('charge', 0)) if pd.notna(row.get('charge')) else 0
                            mods = _mods_norm(row.get('modifications', '-'))
                            accepted_rows.append({
                                'label': f"{pep} +{ch} | {mods}",
                                'key': (pep, ch, mods),
                                'pep': pep,
                                'charge': ch,
                                'mods': mods,
                                'rts_sec': rts,
                                'rts_min': rts / 60.0,
                                'ints': total_int,
                                'min_rt_sec': float(row.get('detected_peak_min_rt')) if pd.notna(row.get('detected_peak_min_rt')) else coll_min,
                                'max_rt_sec': float(row.get('detected_peak_max_rt')) if pd.notna(row.get('detected_peak_max_rt')) else coll_max,
                                'collection_min_rt_sec': coll_min,
                                'collection_max_rt_sec': coll_max,
                                'seq_start': int(row.get('sequence_start_pos', 999999)) if pd.notna(row.get('sequence_start_pos')) else 999999,
                                'total_area': float(row.get('total_area')) if pd.notna(row.get('total_area')) else None,
                            })
                except Exception:
                    accepted_rows = []

            if not accepted_rows:
                st.info('No accepted peptides available to channelize.')
            else:
                run_acc_channels = st.button(
                    'Assign channels for accepted peptides',
                    key='run_windows_accepted_channels',
                    help='Greedy channel assignment from accepted peptides using collection-window overlap only.',
                )

                if run_acc_channels:
                    # Build interval table from accepted rows.
                    chan_rows = []
                    for r in accepted_rows:
                        a = float(r.get('collection_min_rt_sec', 0.0))
                        b = float(r.get('collection_max_rt_sec', 0.0))
                        if b < a:
                            a, b = b, a
                        if b <= a:
                            continue
                        label = str(r.get('label', '')).split('|')[0].strip() or str(r.get('label', ''))
                        key = r.get('key')
                        apex_rt_sec = None
                        try:
                            rr = np.asarray(r.get('rts_sec', []), dtype=float)
                            ii = np.asarray(r.get('ints', []), dtype=float)
                            if len(rr) > 0 and len(ii) > 0:
                                n_ap = min(len(rr), len(ii))
                                apex_idx = int(np.nanargmax(ii[:n_ap]))
                                if 0 <= apex_idx < n_ap:
                                    apex_rt_sec = float(rr[apex_idx])
                        except Exception:
                            apex_rt_sec = None
                        if apex_rt_sec is None or not np.isfinite(apex_rt_sec):
                            apex_rt_sec = 0.5 * (a + b)
                        chan_rows.append({
                            'key': key,
                            'label': label,
                            'collection_min_rt_sec': a,
                            'collection_max_rt_sec': b,
                            'apex_rt_sec': float(apex_rt_sec),
                            'total_area': float(r.get('total_area') or 0.0),
                        })

                    if not chan_rows:
                        st.error('No valid accepted collection windows found for channel assignment.')
                    else:
                        # Greedy interval coloring (minimal channel count), with spacing optimization:
                        # among feasible channels, pick the one that maximizes separation from previous apex.
                        order = sorted(range(len(chan_rows)), key=lambda i: (chan_rows[i]['collection_min_rt_sec'], chan_rows[i]['collection_max_rt_sec']))
                        channel_ends = []   # end RT per channel
                        channel_last_apex = []  # apex RT of last peptide in each channel
                        assigned = [None] * len(chan_rows)
                        for i in order:
                            start_i = chan_rows[i]['collection_min_rt_sec']
                            apex_i = chan_rows[i].get('apex_rt_sec', 0.5 * (
                                chan_rows[i]['collection_min_rt_sec'] + chan_rows[i]['collection_max_rt_sec']
                            ))
                            feasible = []
                            for ch_idx, ch_end in enumerate(channel_ends):
                                if start_i >= ch_end:
                                    gap = float(apex_i) - float(channel_last_apex[ch_idx])
                                    feasible.append((gap, ch_idx))
                            if not feasible:
                                assigned_ch = len(channel_ends)
                                channel_ends.append(chan_rows[i]['collection_max_rt_sec'])
                                channel_last_apex.append(float(apex_i))
                            else:
                                # Maximize peak spacing; tie-break by earlier channel index for stability.
                                feasible.sort(key=lambda t: (-t[0], t[1]))
                                assigned_ch = feasible[0][1]
                                channel_ends[assigned_ch] = chan_rows[i]['collection_max_rt_sec']
                                channel_last_apex[assigned_ch] = float(apex_i)
                            assigned[i] = assigned_ch + 1
                        for i, ch in enumerate(assigned):
                            chan_rows[i]['channel'] = int(ch)
                        ch_map = {tuple(rr['key']): int(rr['channel']) for rr in chan_rows if rr.get('key') is not None}
                        accepted_rows_with_channel = []
                        for rr in accepted_rows:
                            k = rr.get('key')
                            if k is None:
                                continue
                            chv = ch_map.get(tuple(k))
                            if chv is None:
                                continue
                            row_ch = dict(rr)
                            row_ch['channel'] = int(chv)
                            accepted_rows_with_channel.append(row_ch)

                        # Save channelized accepted CSV by mapping channel back onto accepted CSV rows.
                        accepted_csv = thin_res.get('accepted_csv') if thin_res else windows_accepted_csv
                        out_csv = None
                        if accepted_csv and os.path.exists(accepted_csv):
                            try:
                                df_acc = load_csv(accepted_csv).copy()
                                df_acc['channel'] = df_acc.apply(
                                    lambda rr: ch_map.get(
                                        (str(rr.get('_key_pep', '')).strip(), int(pd.to_numeric(rr.get('_key_charge'), errors='coerce') or 0), _mods_norm(rr.get('_key_mods', '-'))),
                                        np.nan,
                                    ),
                                    axis=1,
                                )
                                base_acc = os.path.splitext(os.path.basename(accepted_csv))[0]
                                clean_base = base_acc
                                for suf in ['_windows_channels_accepted', '_accepted']:
                                    if clean_base.endswith(suf):
                                        clean_base = clean_base[:-len(suf)]
                                        break
                                out_csv = os.path.join(out_dir, clean_base + '_channels.csv')
                                df_acc.to_csv(out_csv, index=False)
                                st.session_state.last_step_output = out_csv
                                st.session_state.last_step_name = 'Channels'
                            except Exception:
                                out_csv = None

                        st.session_state.windows_accepted_channels_result = {
                            'rows': chan_rows,
                            'n_channels': int(max(r['channel'] for r in chan_rows)),
                            'out_csv': out_csv,
                            'accepted_rows_with_channel': accepted_rows_with_channel,
                        }

                ch_res = st.session_state.get('windows_accepted_channels_result')
                if ch_res and ch_res.get('rows'):
                    ch_rows = ch_res.get('rows', [])
                    n_channels = int(ch_res.get('n_channels', 0))
                    st.caption(f'Assigned {len(ch_rows)} accepted peptides across {n_channels} channels (no overlap within each channel).')
                    if ch_res.get('out_csv') and os.path.exists(ch_res['out_csv']):
                        st.markdown(f"- Channelized accepted CSV: `{ch_res['out_csv']}`")

                    if go is not None:
                        try:
                            from visualization.chromatograms import _total_area_colormap_for_overlay
                        except Exception:
                            _total_area_colormap_for_overlay = None
                        area_norm_ui = None
                        cmap_ui = None
                        if _total_area_colormap_for_overlay is not None:
                            valid_areas_ui = [
                                max(0.0, float(r.get('total_area') or 0.0))
                                for r in ch_rows
                                if r.get('total_area') is not None
                            ]
                            valid_areas_ui = [v for v in valid_areas_ui if np.isfinite(v)]
                            if len(valid_areas_ui) > 1 and max(valid_areas_ui) > min(valid_areas_ui):
                                from matplotlib.colors import Normalize
                                area_norm_ui = Normalize(vmin=min(valid_areas_ui), vmax=max(valid_areas_ui))
                                cmap_ui = _total_area_colormap_for_overlay()

                        def _ui_row_color(row_):
                            if area_norm_ui is not None and cmap_ui is not None:
                                aval = max(0.0, float(row_.get('total_area') or 0.0))
                                return matplotlib.colors.to_hex(cmap_ui(area_norm_ui(aval)))
                            return '#8b8b8b'

                        fig_ch = go.Figure()
                        y_order = [f'Channel {i}' for i in range(1, n_channels + 1)]
                        for r in ch_rows:
                            ylab = f"Channel {int(r['channel'])}"
                            fig_ch.add_trace(go.Scatter(
                                x=[r['collection_min_rt_sec'] / 60.0, r['collection_max_rt_sec'] / 60.0],
                                y=[ylab, ylab],
                                mode='lines',
                                line=dict(width=5, color=_ui_row_color(r)),
                                hovertemplate=(
                                    f"{r['label']}<br>"
                                    f"Channel: {int(r['channel'])}<br>"
                                    f"total_area: {float(r.get('total_area') or 0.0):.2e}<br>"
                                    "Collection window: %{x:.2f} min"
                                    "<extra></extra>"
                                ),
                                showlegend=False,
                            ))
                        fig_ch.update_layout(
                            template='plotly_dark',
                            height=max(320, min(1400, 80 + 38 * n_channels)),
                            margin=dict(l=20, r=20, t=40, b=40),
                            title='Accepted peptides assigned to non-overlapping channels',
                            xaxis_title='MS1 RT (min)',
                            yaxis_title='Channel',
                            yaxis=dict(type='category', categoryorder='array', categoryarray=y_order),
                        )
                        st.plotly_chart(fig_ch, width='stretch')

                    labeled_tracks_btn, labeled_overlay_btn = st.columns(2)
                    with labeled_tracks_btn:
                        show_tracks_by_channel = st.button(
                            'Show labeled tracks plots by channel',
                            key='show_labeled_tracks_by_channel',
                            help='Generate tracks-only labeled chromatogram plots for each channel.',
                        )
                    with labeled_overlay_btn:
                        show_overlays_by_channel = st.button(
                            'Show labeled overlays by channel',
                            key='show_labeled_overlays_by_channel',
                            help='Generate labeled linear+log+tracks overlays for each channel.',
                        )

                    rows_with_channel = ch_res.get('accepted_rows_with_channel', [])
                    if (show_tracks_by_channel or show_overlays_by_channel) and rows_with_channel:
                        try:
                            from visualization.chromatograms import (
                                _create_zoom_all_linear_log_tracks_from_overlay,
                                _total_area_colormap_for_overlay,
                            )
                        except Exception:
                            _create_zoom_all_linear_log_tracks_from_overlay = None
                            _total_area_colormap_for_overlay = None

                        channel_plot_dir = os.path.join(plots_root_dir, 'channel_assignment', 'accepted_by_channel')
                        os.makedirs(channel_plot_dir, exist_ok=True)
                        channels_sorted = sorted(set(int(r.get('channel', 0)) for r in rows_with_channel if int(r.get('channel', 0)) > 0))
                        area_norm_global = None
                        cmap_global = None
                        if show_tracks_by_channel and _total_area_colormap_for_overlay is not None:
                            valid_areas_global = [
                                max(0.0, float(r.get('total_area') or 0.0))
                                for r in rows_with_channel
                                if r.get('total_area') is not None
                            ]
                            valid_areas_global = [v for v in valid_areas_global if np.isfinite(v)]
                            if len(valid_areas_global) > 1 and max(valid_areas_global) > min(valid_areas_global):
                                from matplotlib.colors import Normalize
                                area_norm_global = Normalize(vmin=min(valid_areas_global), vmax=max(valid_areas_global))
                                cmap_global = _total_area_colormap_for_overlay()

                        for chv in channels_sorted:
                            subset_rows = [r for r in rows_with_channel if int(r.get('channel', 0)) == chv]
                            if not subset_rows:
                                continue
                            overlay_subset = []
                            for r in subset_rows:
                                overlay_subset.append({
                                    'rts': np.asarray(r['rts_sec'], dtype=float),
                                    'rts_min': np.asarray(r['rts_min'], dtype=float),
                                    'total_intensities': np.asarray(r['ints'], dtype=float),
                                    'label': r['label'],
                                    'total_area': r.get('total_area'),
                                    'sequence_start_pos': r.get('seq_start', 999999),
                                    'min_rt': r.get('min_rt_sec'),
                                    'max_rt': r.get('max_rt_sec'),
                                    'collection_min_rt': r.get('collection_min_rt_sec'),
                                    'collection_max_rt': r.get('collection_max_rt_sec'),
                                })

                            if show_overlays_by_channel and _create_zoom_all_linear_log_tracks_from_overlay is not None:
                                base_nm = f'accepted_channel_{chv}'
                                overlay_png = os.path.join(channel_plot_dir, f'{base_nm}_zoom_peptides_all_linear_log_tracks.png')
                                _create_zoom_all_linear_log_tracks_from_overlay(
                                    overlay_subset,
                                    channel_plot_dir,
                                    base_nm,
                                    add_peak_labels=True,
                                )
                                if os.path.exists(overlay_png):
                                    st.markdown(f'**Channel {chv} overlay (linear + log + tracks)**')
                                    st.image(overlay_png, use_container_width=True)

                            if show_tracks_by_channel:
                                subset_rows = sorted(subset_rows, key=lambda x: (float(x.get('collection_min_rt_sec', 0.0)), str(x.get('label', ''))))
                                fig_h = max(8, min(44, 1.1 * len(subset_rows) + 4))
                                fig_tr, ax_tr = plt.subplots(figsize=(24, fig_h))
                                fig_tr.patch.set_facecolor('black')
                                ax_tr.set_facecolor('black')
                                area_sum_channel = sum(max(0.0, float(r.get('total_area') or 0.0)) for r in subset_rows)
                                x_vals = []
                                track_h = 1.45
                                track_gap = 0.35
                                placed_label_points = []
                                for idx, r in enumerate(subset_rows):
                                    rts = np.asarray(r.get('rts_min', []), dtype=float)
                                    ints = np.asarray(r.get('ints', []), dtype=float)
                                    if len(rts) < 2 or len(ints) < 2:
                                        continue
                                    imax = float(np.nanmax(ints)) if np.isfinite(np.nanmax(ints)) and float(np.nanmax(ints)) > 0 else 1.0
                                    y0 = idx * (track_h + track_gap)
                                    y = (ints / imax) * track_h + y0
                                    cmin = float(r.get('collection_min_rt_sec', 0.0)) / 60.0
                                    cmax = float(r.get('collection_max_rt_sec', 0.0)) / 60.0
                                    if cmax < cmin:
                                        cmin, cmax = cmax, cmin
                                    mask = (rts >= cmin) & (rts <= cmax)
                                    y_fill = np.where(mask, y, y0)
                                    area_val = max(0.0, float(r.get('total_area') or 0.0))
                                    if area_norm_global is not None and cmap_global is not None and np.isfinite(area_val):
                                        col = cmap_global(area_norm_global(area_val))
                                    else:
                                        col = (0.55, 0.55, 0.55, 0.95)
                                    ax_tr.fill_between(rts, y0, y_fill, color=col, alpha=0.95, zorder=3)
                                    ax_tr.plot(rts, y, color=col, linewidth=2.5, alpha=1.0, zorder=4)
                                    lab = str(r.get('label', f'#{idx+1}')).split('|')[0].strip()
                                    if len(lab) > 38:
                                        lab = lab[:35] + '...'
                                    try:
                                        i_apex = int(np.nanargmax(ints))
                                    except Exception:
                                        i_apex = -1
                                    rel_pct = (100.0 * area_val / area_sum_channel) if area_sum_channel > 0 else 0.0
                                    if 0 <= i_apex < len(rts):
                                        x_peak = float(rts[i_apex])
                                        y_peak = float(y[i_apex])
                                        x_offset = 0.040 * max(1e-6, float(np.nanmax(rts) - np.nanmin(rts)))
                                        seq_txt = lab
                                        rt_window_txt = f"[{cmin:.2f}, {cmax:.2f}] min"
                                        area_txt = f"A={area_val:.2e} | {rel_pct:.1f}% | {rt_window_txt}"
                                        # Place labels directly adjacent to the peak, then nudge right if needed to avoid overlap.
                                        y_seq = y_peak + track_h * 0.06
                                        y_area = y_peak - track_h * 0.22
                                        x_seq = x_peak + x_offset
                                        x_area = x_peak + x_offset
                                        x_span_local = max(1e-6, float(np.nanmax(rts) - np.nanmin(rts)))
                                        x_step = 0.018 * x_span_local
                                        x_min_sep = 0.020 * x_span_local
                                        y_min_sep = 0.18 * track_h

                                        def _occupied(xx, yy):
                                            for px, py in placed_label_points:
                                                if abs(xx - px) < x_min_sep and abs(yy - py) < y_min_sep:
                                                    return True
                                            return False

                                        guard = 0
                                        while (_occupied(x_seq, y_seq) or _occupied(x_area, y_area)) and guard < 40:
                                            x_seq += x_step
                                            x_area += x_step
                                            guard += 1
                                        ax_tr.text(
                                            x_seq,
                                            y_seq,
                                            seq_txt,
                                            va='bottom',
                                            ha='left',
                                            fontsize=17,
                                            fontfamily='serif',
                                            fontweight='bold',
                                            color='0.98',
                                            bbox=dict(boxstyle='round,pad=0.30', fc=(0, 0, 0, 0.72), ec=col, linewidth=1.2),
                                            zorder=7,
                                        )
                                        ax_tr.text(
                                            x_area,
                                            y_area,
                                            area_txt,
                                            va='bottom',
                                            ha='left',
                                            fontsize=14,
                                            fontfamily='serif',
                                            fontweight='bold',
                                            color='0.95',
                                            bbox=dict(boxstyle='round,pad=0.26', fc=(0, 0, 0, 0.68), ec=col, linewidth=1.1),
                                            zorder=6,
                                        )
                                        placed_label_points.append((x_seq, y_seq))
                                        placed_label_points.append((x_area, y_area))
                                    x_vals.extend([float(np.nanmin(rts)), float(np.nanmax(rts))])
                                if x_vals:
                                    x_min = min(x_vals)
                                    x_max = max(x_vals)
                                    pad = max(0.25, (x_max - x_min) * 0.18)
                                    ax_tr.set_xlim(x_min - pad, x_max + pad)
                                ax_tr.set_ylim(-0.2, len(subset_rows) * (track_h + track_gap) - track_gap + 0.2)
                                ax_tr.set_title(f'Channel {chv} tracks (labeled)', fontsize=16, fontweight='bold', fontfamily='serif', color='0.95')
                                ax_tr.set_xlabel('MS1 RT (min)', fontsize=14, fontfamily='serif', color='0.9')
                                ax_tr.set_ylabel('Track (shape norm.)', fontsize=13, fontfamily='serif', color='0.9')
                                ax_tr.set_yticks([])
                                ax_tr.tick_params(axis='x', colors='0.85', labelsize=12)
                                for spine in ax_tr.spines.values():
                                    spine.set_color('0.6')
                                ax_tr.grid(True, axis='x', alpha=0.25)
                                tracks_png = os.path.join(channel_plot_dir, f'accepted_channel_{chv}_tracks_labeled.png')
                                fig_tr.tight_layout()
                                fig_tr.savefig(tracks_png, dpi=320, bbox_inches='tight', facecolor='black', pad_inches=0.2)
                                plt.close(fig_tr)
                                if os.path.exists(tracks_png):
                                    st.markdown(f'**Channel {chv} tracks (labeled)**')
                                    st.image(tracks_png, use_container_width=True)

        _render_terminal('channels')

    with filter_tabs[16]:  # Method
        st.subheader('Method')
        st.caption('Method parameters and documentation.')

if __name__ == '__main__':
    main()
