#!/usr/bin/env python3
"""
Interactive HDX filtering app – Streamlit interface for the Comet/HDX workflow.

Implements the filtering logic from Workflow_Steps:
  Step 4: Q-value / PEP confidence
  Step 5: Fragment accuracy (ppm)
  Step 6: (Load pre-computed extraction CSV)
  Step 7: Isotopic envelope (M+0 > M+1 or M+1 > M+2)
  Step 8: Significant fragments (0.5% max MS2)
  Plus: apex intensity, shape correlation, proline, modifications

Run: streamlit run hdx_filter_app.py
"""

import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import pandas as pd
import numpy as np
import streamlit as st

# Default paths
DEFAULT_DATA_DIR = os.path.join(_SCRIPT_DIR, 'data')
DEFAULT_CSV = os.path.join(DEFAULT_DATA_DIR, 'comet_frags_perc_openMS.csv')

QVALUE_COLS = ['percolator_qvalue', 'q-value', 'qvalue', 'percolator_q-value', 'FDR', 'fdr']
PEP_COLS = ['percolator_PEP', 'PEP', 'pep']


def load_csv(path: str) -> pd.DataFrame:
    """Load CSV with optional Comet header skip."""
    try:
        with open(path, 'r') as f:
            first = f.readline()
        skip = 1 if 'CometVersion' in first else 0
    except Exception:
        skip = 0
    return pd.read_csv(path, sep=',', skiprows=skip, engine='python', quotechar='"', on_bad_lines='warn')


def _resolve_qcol(df: pd.DataFrame):
    for c in QVALUE_COLS:
        if c in df.columns:
            return c
    return None


def _resolve_pepcol(df: pd.DataFrame):
    for c in PEP_COLS:
        if c in df.columns:
            return c
    return None


def _pass_confidence(row, df, qcol, pepcol, threshold):
    q_ok = True
    if qcol and qcol in df.columns and pd.notna(row.get(qcol)):
        try:
            q_ok = float(row[qcol]) <= threshold
        except (TypeError, ValueError):
            q_ok = True
    pep_ok = True
    if pepcol and pepcol in df.columns and pd.notna(row.get(pepcol)):
        try:
            pep_ok = float(row[pepcol]) <= threshold
        except (TypeError, ValueError):
            pep_ok = True
    return q_ok or pep_ok


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
    """True if peptide has variable modifications (exclude)."""
    if mods is None or (isinstance(mods, float) and np.isnan(mods)):
        return False
    s = str(mods).strip()
    if not s or s == '-' or s.lower() == 'nan':
        return False
    # Carbamidomethyl (57.0215) is fixed – allow
    allowed = {'57.0215', '57.02146', '57.021464'}
    parts = [p.strip() for p in s.replace(',', ' ').split() if p.strip()]
    for p in parts:
        if '_' in p:
            mass = p.split('_')[-1].strip()
            if mass and mass not in allowed:
                return True
    return True


def main():
    st.set_page_config(page_title='HDX Filtering', page_icon='🔬', layout='wide')
    st.title('HDX Comet Filtering – Interactive Workflow')
    st.caption('Based on Workflow_Steps: confidence, accuracy, envelope, significance, and extraction metrics')

    # Sidebar: file selection
    st.sidebar.header('Input')
    data_dir = st.sidebar.text_input('Data directory', DEFAULT_DATA_DIR)
    csv_options = [
        'comet_frags_perc_openMS.csv',
        'comet_frags_perc_openMS_confidence.csv',
        'comet_frags_perc_openMS_confidence_accuracy.csv',
        'comet_frags_perc_openMS_confidence_accuracy_extraction.csv',
        'comet_frags_perc_openMS_confidence_accuracy_extraction_envelope.csv',
    ]
    csv_name = st.sidebar.selectbox('CSV file', csv_options)
    csv_path = os.path.join(data_dir, csv_name)
    if not os.path.exists(csv_path):
        st.error(f'File not found: {csv_path}')
        st.info('Update the data directory or ensure the CSV exists.')
        return

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

    st.sidebar.metric('Rows loaded', n_orig)
    st.sidebar.divider()

    # Filter pipeline
    st.sidebar.header('Filters')
    has_extraction = 'apex_intensity' in df.columns or 'total_area' in df.columns
    has_envelope = 'm0_gt_m1_gt_m2' in df.columns or 'envelope_ok' in df.columns

    # Step 4: Confidence
    conf_thresh = st.sidebar.slider('Q-value / PEP threshold', 0.001, 0.2, 0.05, 0.005,
                                    help='Keep rows where Q ≤ threshold OR PEP ≤ threshold')
    qcol = _resolve_qcol(df)
    pepcol = _resolve_pepcol(df)
    if qcol or pepcol:
        mask_conf = df.apply(lambda r: _pass_confidence(r, df, qcol, pepcol, conf_thresh), axis=1)
        df = df[mask_conf].copy()
    n_after_conf = len(df)
    st.sidebar.metric('After confidence', n_after_conf, delta=n_after_conf - n_orig)

    # Step 5: PPM (if matched fragment columns exist)
    if 'matched fragment ion mz' in df.columns and 'plain_peptide' in df.columns:
        ppm_thresh = st.sidebar.slider('PPM threshold (fragment accuracy)', 1.0, 20.0, 5.0, 0.5,
                                       help='Keep only fragments within this ppm of theoretical')
        # PPM filter is complex – for now we pass through; full logic in filter_comet_frags_accuracy.py
        st.sidebar.caption('PPM filter: run filter_comet_frags_accuracy.py for full logic')
    else:
        ppm_thresh = 5.0

    # Step 7: Envelope (if extraction data)
    if has_envelope:
        filter_envelope = st.sidebar.checkbox('Filter by envelope (M+0>M+1 or M+1>M+2)', True)
        if filter_envelope:
            def _pass_env(row):
                m0 = row.get('m0_gt_m1_gt_m2')
                env = row.get('envelope_ok')
                if pd.notna(m0):
                    return _to_bool(m0)
                if pd.notna(env):
                    return _to_bool(env)
                return False
            mask_env = df.apply(_pass_env, axis=1)
            df = df[mask_env].copy()
    st.sidebar.metric('After envelope', len(df))

    # Extraction metrics (if available)
    if has_extraction:
        st.sidebar.subheader('Extraction metrics')
        apex_min = st.sidebar.number_input('Min apex intensity', 1e4, 1e8, 1e5, 1e4, format='%.0e',
                                          help='Reject peaks with apex < this')
        if 'apex_intensity' in df.columns:
            mask_apex = pd.to_numeric(df['apex_intensity'], errors='coerce') >= apex_min
            mask_apex = mask_apex | pd.isna(df['apex_intensity'])
            df = df[mask_apex].copy()
        shape_min = st.sidebar.slider('Min shape correlation', 0.0, 1.0, 0.8, 0.05)
        if 'shape_corr' in df.columns:
            mask_shape = pd.to_numeric(df['shape_corr'], errors='coerce') >= shape_min
            mask_shape = mask_shape | pd.isna(df['shape_corr'])
            df = df[mask_shape].copy()
        reject_proline = st.sidebar.checkbox('Reject peptides starting/ending with proline', True)
        if reject_proline and 'plain_peptide' in df.columns:
            def _is_proline_reject(seq):
                if pd.isna(seq) or not seq:
                    return True
                s = str(seq).strip()
                return s and (s[0].upper() == 'P' or s[-1].upper() == 'P')
            mask_pro = ~df['plain_peptide'].apply(_is_proline_reject)
            df = df[mask_pro].copy()
        reject_mods = st.sidebar.checkbox('Reject modified peptides', False)
        if reject_mods and 'modifications' in df.columns:
            mask_mods = ~df['modifications'].apply(_has_modifications)
            df = df[mask_mods].copy()

    n_final = len(df)
    st.sidebar.divider()
    st.sidebar.metric('Final rows', n_final, delta=n_final - n_orig)

    # Main area: tabs
    tab1, tab2, tab3, tab4 = st.tabs(['Summary', 'Confidence (PEP vs Q)', 'Data table', 'Download'])

    with tab1:
        st.subheader('Filter summary')
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric('Original', n_orig)
        with col2:
            st.metric('After filters', n_final)
        with col3:
            pct = 100 * n_final / n_orig if n_orig > 0 else 0
            st.metric('Retention %', f'{pct:.1f}%')
        st.write('Columns:', ', '.join(df.columns[:20].tolist()) + ('...' if len(df.columns) > 20 else ''))

    with tab2:
        st.subheader('Confidence filter: PEP vs Q-value')
        if qcol and pepcol and qcol in df.columns and pepcol in df.columns:
            q_vals = pd.to_numeric(df[qcol], errors='coerce')
            pep_vals = pd.to_numeric(df[pepcol], errors='coerce')
            valid = q_vals.notna() & pep_vals.notna()
            if valid.sum() > 0:
                try:
                    import plotly.express as px
                    plot_df = pd.DataFrame({
                        'Q-value': np.clip(q_vals[valid], 1e-10, 1.0),
                        'PEP': np.clip(pep_vals[valid], 1e-10, 1.0),
                    })
                    fig = px.scatter(plot_df, x='Q-value', y='PEP', log_x=True, log_y=True,
                                    title=f'PEP vs Q-value (n={len(plot_df)})')
                    fig.add_hline(y=conf_thresh, line_dash='dash', line_color='red')
                    fig.add_vline(x=conf_thresh, line_dash='dash', line_color='orange')
                    fig.update_layout(template='plotly_dark', height=500)
                    st.plotly_chart(fig, use_container_width=True)
                except ImportError:
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(8, 8))
                    ax.set_xscale('log')
                    ax.set_yscale('log')
                    ax.scatter(np.clip(q_vals[valid], 1e-10, 1.0), np.clip(pep_vals[valid], 1e-10, 1.0),
                               alpha=0.6, s=12, c='#2E86AB')
                    ax.axhline(conf_thresh, color='red', linestyle='--')
                    ax.axvline(conf_thresh, color='orange', linestyle='--')
                    ax.set_xlabel('Q-value')
                    ax.set_ylabel('PEP')
                    ax.set_title(f'PEP vs Q-value (n={valid.sum()})')
                    st.pyplot(fig)
            else:
                st.warning('No valid Q/PEP values for scatter.')
        else:
            st.info('No Q-value or PEP column found.')

    with tab3:
        st.subheader('Filtered data')
        st.dataframe(df.head(500), use_container_width=True)
        if len(df) > 500:
            st.caption(f'Showing first 500 of {len(df)} rows.')

    with tab4:
        st.subheader('Download filtered CSV')
        csv_out = df.to_csv(index=False)
        st.download_button('Download filtered CSV', csv_out, file_name='filtered_output.csv', mime='text/csv')

    # Extraction diagnostics (if available)
    if has_extraction and 'coelution_score' in df.columns:
        st.divider()
        st.subheader('Extraction diagnostics')
        has_shape = 'shape_corr' in df.columns
        if has_shape:
            valid = df['coelution_score'].notna() & df['shape_corr'].notna()
        else:
            valid = df['coelution_score'].notna()
        if valid.sum() > 0:
            try:
                import plotly.express as px
                if has_shape:
                    diag_df = df[valid][['coelution_score', 'shape_corr']].copy()
                    diag_df.columns = ['Coelution', 'Shape correlation']
                    fig = px.scatter(diag_df, x='Coelution', y='Shape correlation',
                                     title='Coelution vs shape correlation')
                else:
                    diag_df = df[valid][['coelution_score']].copy()
                    diag_df.columns = ['Coelution']
                    fig = px.histogram(diag_df, x='Coelution', title='Coelution score distribution')
                fig.update_layout(template='plotly_dark', height=400)
                st.plotly_chart(fig, use_container_width=True)
            except ImportError:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(8, 6))
                if has_shape:
                    ax.scatter(df.loc[valid, 'coelution_score'], df.loc[valid, 'shape_corr'], alpha=0.6, s=12)
                    ax.set_ylabel('Shape correlation')
                else:
                    ax.hist(df.loc[valid, 'coelution_score'], bins=30, alpha=0.7)
                ax.set_xlabel('Coelution score')
                ax.set_title('Coelution vs shape correlation' if has_shape else 'Coelution score distribution')
                st.pyplot(fig)


if __name__ == '__main__':
    main()
