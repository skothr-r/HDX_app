# HDX Filtering – Interactive Streamlit App

Interactive web app for filtering Comet/HDX peptide data according to the workflow in `Workflow_Steps`.

## Quick start

```bash
# Install dependencies
pip install -r requirements_streamlit.txt

# Run the app
streamlit run hdx_filter_app.py
```

The app opens in your browser (default: http://localhost:8501).

## Workflow steps implemented

| Step | Filter | Description |
|------|--------|-------------|
| 4 | Q-value / PEP | Keep rows where Q ≤ threshold OR PEP ≤ threshold (default 0.05) |
| 5 | PPM | Fragment accuracy – run `filter_comet_frags_accuracy.py` for full logic |
| 6 | (Load) | Load pre-computed extraction CSV from chromatogram step |
| 7 | Envelope | M+0 > M+1 or M+1 > M+2 on summed MS1 |
| – | Apex intensity | Reject peaks with apex < 10⁵ |
| – | Shape correlation | Min shape correlation (when extraction data available) |
| – | Proline | Reject peptides starting/ending with proline |
| – | Modifications | Reject modified peptides |

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

## Note on Step 6 (chromatogram extraction)

Chromatogram extraction (Step 6) uses the mzML file and is run separately via `run_visualization.py --chromatograms` or `extract_chromatograms_from_comet_frags_csv_mzml.py`. The app loads the resulting extraction CSV; it does not run extraction itself.
