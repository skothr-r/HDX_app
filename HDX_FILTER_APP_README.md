# HDX Filtering – Interactive Streamlit App

Interactive web app for filtering Comet/HDX peptide data according to the workflow in `Workflow_Steps`.

## Quick start

**Option A – Conda (recommended, includes pyopenms):**
```bash
conda env create -f environment_hdx_app.yml
conda activate hdx_app
streamlit run hdx_filter_app.py
```

**Option A2 – Use run script** (avoids libjpeg/PIL conflicts on macOS):
```bash
conda env create -f environment_hdx_app.yml
./run_hdx_app.sh
```
The script uses the hdx_app Python and sets `DYLD_LIBRARY_PATH` so conda libs load before Homebrew.

**Option B – pip only (Steps 6 & 8 need pyopenms separately):**
```bash
pip install -r requirements_streamlit.txt
streamlit run hdx_filter_app.py
```

The app opens in your browser (default: http://localhost:8501).

**Steps 5, 6, and 8 require pyopenms** (accuracy, extraction, significance). If you see "Could not import pyopenms":
- **Conda:** `conda install -c bioconda pyopenms` then `conda activate <env>` before running the app
- **macOS:** `export DYLD_LIBRARY_PATH=$CONDA_PREFIX/lib:$DYLD_LIBRARY_PATH` (or `$(brew --prefix openms)/lib` if using Homebrew pyopenms)

**macOS – libjpeg / PIL conflict** (`_jpeg12_read_raw_data` symbol not found):
- Use the run script: `./run_hdx_app.sh` (it sets `DYLD_LIBRARY_PATH` to conda libs only).
- Or run manually with **only** conda libs (do not append `$DYLD_LIBRARY_PATH`—that can pull in Homebrew):
  ```bash
  export DYLD_LIBRARY_PATH=$CONDA_PREFIX/lib
  streamlit run hdx_filter_app.py
  ```

## Workflow steps implemented

| Step | Filter | Description |
|------|--------|-------------|
| 4 | Prefilter | PEP, Q-value, prolines, mods, ppm (MS1 peptide) |
| 5 | Significant_fragmentation | MS2 fragments within 5ppm of theoretical m/z; min significant fragment count |
| 6 | Extraction | Envelope, apex ≥10⁵, coelution, shape correlation, S/N fragments 0.5% max |

## Input files

1. **Data directory** – Path to the folder containing your CSV files (default: `data/`).
2. **CSV file** – Choose the pipeline stage:
   - `comet_frags_perc_openMS.csv` – Raw output from Step 3
   - `comet_frags_perc_openMS_confidence.csv` – After Step 4
   - `comet_frags_perc_openMS_confidence_accuracy.csv` – After Step 5
   - `comet_frags_perc_openMS_confidence_accuracy_extraction.csv` – After Step 6 (chromatogram extraction)
   - `comet_frags_perc_openMS_confidence_accuracy_extraction_envelope.csv` – After Step 7

## Features

- **Sidebar filters** – Adjust thresholds with sliders and checkboxes.
- **Summary** – Row counts before and after filtering.
- **PEP vs Q-value scatter** – Interactive plot with threshold lines.
- **Extraction diagnostics** – Coelution vs shape correlation (when extraction data is present).
- **Data table** – Browse filtered rows.
- **Download** – Export filtered CSV.

## Run pipeline scripts

The app can run Steps 4–8 directly (each outputs CSV + diagnostic plots). Steps 5, 6, and 8 require pyopenms.
