# HDX Visualization Module — Refactoring Plan

**Input (from Step 3):** `comet_frags_perc_openMS.csv` — output of OpenMS MS1 extraction (add_ms1_data_openms.py on Comet+Percolator CSV).

**KEEP (must never remove during refactoring):**
- `*_zoom_peptides_all_linear_log_tracks.png` — 3×1 combined zoom overlay (linear + log + tracks)
- `accepted/chromatograms_peptides/` — individual per-peptide chromatogram plots (keep as-is, they look great)
- `accepted/chromatograms_peptides/all_peptides_overlay.png`
- `accepted/chromatograms_peptides/all_peptides_overlay_skyline_style.png`
- `accepted/chromatograms_peptides/all_peptides_overlay_tracks_only.png`

---

## Step 4: Filter by confidence (Q/PEP)

**Script:** `filter_comet_frags_confidence.py`

**Input:** `comet_frags_perc_openMS.csv`  
**Output:** `comet_frags_perc_openMS_confidence.csv`  
**Criteria:** PEP ≤ 0.05 OR Q-value ≤ 0.05 (keep rows passing either). Include all columns; delete rows that don't pass.

**Actions:**
- `mkdir diagnostics` (create if not exists)
- Output diagnostic scatter plot: PEP vs Q-value scatter, draw thresholds (horizontal/vertical lines at 0.05), legend with count passing / total
- Write filtered CSV

**Current implementation:** `run_visualization.py` `_default_filtered_csv()` does Q/PEP filter; no separate diagnostics dir or PEP vs Q scatter. Need to extract into standalone script.

---

## Step 5: Filter by fragment accuracy (5 ppm)

**Script:** `filter_comet_frags_accuracy.py`

**Input:** `comet_frags_perc_openMS_confidence.csv`  
**Output:** `comet_frags_perc_openMS_confidence_accuracy.csv`

**Actions:**
- For each row: keep only fragments whose observed m/z is within 5 ppm of theoretical (c/z/z+1). Overwrite matched fragment ions, pairs, single-AA overhangs with significant-only.
- If a peptide has 0 significant fragments after 5 ppm → discard that peptide.
- Output diagnostic scatter: all fragments' observed m/z vs expected m/z; colorbar = ±ppm; dashed line at 5 ppm; legend with fragments excluded count; outline rejected-peptide fragments (e.g. colored outline) vs thin black outline for others.
- Write filtered CSV

**Current implementation:** `run_visualization.py` `_filter_fragment_columns_to_significant_ppm()` + min overhangs. No diagnostic scatter. Need to extract and add scatter.

**Note:** If theoretical m/z (from `visualization.chromatograms.theoretical_c_z_z1_ion_mz`) does not match Comet-observed m/z (e.g. mass constant differences), use `--ppm 2000` for testing. Align theoretical with Comet source for strict 5 ppm.

---

## Step 6: Extract chromatograms

**Script:** `extract_chromatograms_from_comet_frags_csv_mzml.py`

**Input:** mzML, `comet_frags_perc_openMS_confidence_accuracy.csv` (defaults; flags for custom)  
**Output:** `comet_frags_perc_openMS_confidence_accuracy_extraction.csv`

**Actions:**
- Do all extraction from mzML → add metrics to CSV as you go.
- Add: total_area, apex_peak, integration windows (min_rt, max_rt), collection window (from integration), peak drift buffer window (from collection).
- Save integration, collection, peak drift windows in output CSV.
- Record fragment signals and max MS2 signal for further filtering.
- Make individual chromatogram plots in `accepted/chromatograms_peptides/` — **keep as is, they look great**.
- **KEEP** the zoom overlay `*_zoom_peptides_all_linear_log_tracks.png` (3×1 linear + log + tracks).

**Implementation:** `extract_chromatograms_from_comet_frags_csv_mzml.py` — wrapper that calls `run_chromatograms()` with workflow input/output naming.

---

## Step 7: Refine significant fragments

**Script:** `refine_significant_frags.py`

**Input:** `comet_frags_perc_openMS_confidence_accuracy_extraction.csv`  
**Output:** `comet_frags_perc_openMS_confidence_accuracy_extraction_significance.csv`

**Criteria:**
- Matched frags must be ≥ 0.5% of max MS2 signal.
- Replace significant_frags and related columns with refined data only.
- Remove rows with 0 significant fragments after refinement.

**Diagnostic plots (to diagnostics/):**
1. MS2 max signal vs fragment signal (denote fragments that do not pass).
2. Comparative total peptide overlay (same as all-peptide overlay on chromatograms) on log scale: before vs after significance refinement.
3. All peptides overlay showing rejected peaks only, labeled with unique peptide ID (sequence, charge, mods, shared theoretical mz).

**Print:** Comparison of input vs output dataframes (row counts, etc.).

**Implementation:** `refine_significant_frags.py` — 0.5% max MS2 filter; diagnostic MS2 scatter plot; replaces significant_fragment_* columns.

---

## Step 8: Assess sequence coverage

**Script:** `compare_unique_peptides_sequence.py`

**Input:** `comet_frags_perc_openMS_confidence_accuracy_extraction_significance.csv`  
**Output:** `comet_frags_perc_openMS_confidence_accuracy_extraction_significance_valuable_sequences.csv`

**Actions:**
- Create all unique peptides plots.
- Identify parts of full protein sequence with very little coverage (< 2 single-AA overhangs).
- Add column `valuable_sequence` (0 = FALSE, 1 = TRUE).
- Write output CSV.

**Implementation:** `compare_unique_peptides_sequence.py` — adds `valuable_sequence` column; creates unique peptides plot when FASTA provided.

---

## Step 9: Assign channels

**Script:** `select_unique_peptides_assign_channels.py`

**Input:** `comet_frags_perc_openMS_confidence_accuracy_extraction_significance_valuable_sequences.csv`

**Logic:**
- If `valuable_sequence == 1`: protect from intensity-based filters, prioritize for channels 1–3.
- Channel assignment: need enough channels for all fragments > 1% relative area.
- Fit fragments into sets of 3 channels ("method"); prioritize by % relative area.
- No repeat peak within a method; no overlapping RT/integration/collection/drift windows within a channel.
- First 3 channels: high-signal (>5%) + valuable_sequence; then backfill 1–5%; then <1%.
- Remaining high-signal → next method.
- Create combined and individual channel plots + all peptides overlays.

**Implementation:** `select_unique_peptides_assign_channels.py` — wrapper for `run_rt_windows`; `valuable_sequence` support added to `regenerate_rt_windows_from_csv.py`.

---

## Key additions / clarifications

1. **Diagnostics directory:** Created in Step 4; used by Steps 4, 5, 7. Single `diagnostics/` at project root or per-run.
2. **CSV naming chain:** Each step appends a suffix; full chain is traceable.
3. **all_linear_log_tracks.png:** Produced in Step 6 (chromatogram extraction); must never be removed — it's the 3×1 zoom overlay (linear + log + tracks).
4. **accepted/chromatograms_peptides/:** Individual per-peptide chromatogram plots; keep as-is, they look great. Also: `all_peptides_overlay.png`, `all_peptides_overlay_skyline_style.png`, `all_peptides_overlay_tracks_only.png`.
5. **FASTA:** Required for Steps 8–9 (sequence coverage, channel assignment). Pass from Step 3 output or as config.
6. **Run order:** 4 → 5 → 6 → 7 → 8 → 9. Each step reads previous step's output.
