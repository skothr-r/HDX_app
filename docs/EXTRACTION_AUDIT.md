# Chromatogram Extraction Logic, Name Matching, and RT Times – Audit

## 1. Extraction Logic

### Data flow
1. **Input**: Comet CSV (`plain_peptide`, `charge`, `MS1_retention_time_sec`, `mz`/`theoretical_mz`) + mzML
2. **Grouping**: By `peptide_key` = `{plain_peptide}_{charge}_{mods_norm}`
3. **Per peptide**: `extract_aligned_isotope_chromatograms(raw_file, isotope_mzs)` → `(rts, intensity_matrix)` in seconds
4. **Peak detection**: `detect_chromatographic_peaks_windowed(rts, intensities, rt_anchor, window_sec=180)` → integration/collection windows (seconds)
5. **Output**: `all_peak_windows` (one dict per peptide) + `traces_list` (rts, intensity per peptide)

### RT sources (all in **seconds**)
- **mzML**: `spec.getRT()` (OpenMS)
- **MS1_retention_time_sec**: From `add_ms1_data_openms.py` (MS1 scan before each MS2)
- **anchor_rt**: Median of `MS1_retention_time_sec` in group, or best E-value spectrum RT
- **detected_peak_min_rt / max_rt**: From `detect_chromatographic_peaks_windowed` (same `rts` as trace)
- **collection_min_rt / max_rt**: Wider window from peak_widths rel_height=0.8

### Units
- All stored RT values: **seconds**
- Plotting: convert to minutes (`rt / 60.0`) for x-axis

---

## 2. Name Matching

### peptide_key
- Format: `{plain_peptide}_{charge}_{mods_norm}`
- `mods_norm`: `_norm_mods_for_key(modifications)` – strips, rounds floats in mod strings

### Column names
| Context | Sequence column | Notes |
|---------|-----------------|-------|
| Comet/input CSV | `plain_peptide` | Required |
| Extraction output (metrics) | `peptide` | Same value as plain_peptide |
| Filter CSV | `plain_peptide` | For merge/filter |

### Filter merge (run_plots_from_dataframe)
- Filter keys: `(plain_peptide, charge, _mods)` from filter CSV
- Metrics keys: `(peptide, charge, _mods)` from metrics CSV (`pep_col = 'peptide'` when `plain_peptide` absent)
- Mods: `_norm_mods(modifications)` → `'-'` for None/NaN/empty

### Windows merge (extract → base CSV)
- Lookup key: `(peptide, charge, mods)` from `all_peak_windows` (peptide = sequence)
- Base CSV key: `(plain_peptide, charge, modifications)` – same sequence

---

## 3. trace_index and npz

### Assignment
- `trace_index = i` for row `i` in `all_peak_windows`
- npz: `trace_{i}_rts`, `trace_{i}_int` for `i` in `range(len(traces_list))`
- Order: `traces_list` and `all_peak_windows` are built in the same loop

### Plot-from-dataframe
- Uses `trace_index` from each row to load `trace_{trace_index}_rts` and `trace_{trace_index}_int`
- Fallback: `orig_idx` when `trace_index` is missing (older CSVs)
- **Warning**: If `trace_index` is missing and `filter_csv` is used, `orig_idx` can misalign with npz

---

## 4. RT Consistency Checks

### Fallback in create_chromatogram_subplot_figure
If `detected_peak_min_rt` / `max_rt` do not contain the trace peak apex:
- Recompute window from trace apex ± half_span
- Avoids mismatch between stored windows and trace coordinates

### Window fallback in _draw_peptide_figure_from_dataframe
- `min_rt` = `detected_peak_min_rt` or `collection_min_rt`
- `max_rt` = `detected_peak_max_rt` or `collection_max_rt`
- Avoids using `anchor_rt` for both (zero-width window)

---

## 5. Fixes Applied

1. **combined.py**: `detected_peak_window_window_size` → `detected_peak_window_size`
2. **chromatograms.py**: Warning when `trace_index` is missing in metrics CSV
3. **_draw_peptide_figure_from_dataframe**: Use `collection_min_rt` / `collection_max_rt` when `detected_peak_*` are missing

---

## 6. Recommendations

1. **Re-run extraction** to regenerate `chromatogram_metrics_all.csv` with `trace_index`
2. **Filter CSV**: Use same `(plain_peptide, charge, modifications)` normalization as extraction
3. **RT columns**: Prefer `detected_peak_min_rt` / `max_rt`; fall back to `collection_*`; avoid `anchor_rt` for both min and max
