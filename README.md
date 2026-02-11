<img src="https://uwpr.github.io/Comet/images/cometlogo_1_small.png" align="right">

# Comet MS/MS

Comet is an open source tandem mass spectrometry (MS/MS) sequence database search tool written primarily in C/C++. The original Comet repository lived on [SourceForge](https://sourceforge.net/projects/comet-ms/) since 2012. It was migrated to GitHub on September 2021.

The project website [can be found here](https://uwpr.github.io/Comet/). This includes release notes and search parameters documentation.

To compile on linux and macOS:

- Type 'make'.  This will generate a binary "comet.exe".

To compile with Microsoft Visual Studio:

- We current use build tools v143 with Microsoft Visual Studio 2022.

- First install [MSFileReader from Thermo Fischer Scientific](https://uwpr.github.io/Comet/notes/20220228_rawfile.html).

- Load "Comet.sln" in Visual Studio.

- Set the build to "Release" and "x64".

- Right-mouse-click on the "Comet" project and choose "Build". This should generate a binary "Comet.exe" in x64/Release.

Comet integrates:
- Mike Hoopmann's [MSToolkit library](https://github.com/mhoopmann/mstoolkit) for reading various file formats.
- Matthew Belmonte's C implementation of the [Twiddle algorithm](https://www.netlib.org/toms-2014-06-10/382) used in generating modification permutations.
- C++ port of Gygi Lab's [AScorePro](https://github.com/gygilab/MPToolkit/) for modification localization.

## Documentation

- **[MATCHED_FRAGMENT_IONS_IMPLEMENTATION.md](MATCHED_FRAGMENT_IONS_IMPLEMENTATION.md)**: Detailed documentation on matched fragment ions output and the two-layer ion series strategy
- **[TWO_PASS_ALGORITHM.md](TWO_PASS_ALGORITHM.md)**: Comprehensive guide to the two-pass selection algorithm for peptide selection
- **[PERCOLATOR_SETUP.md](PERCOLATOR_SETUP.md)**: Guide for integrating Percolator q-values with Comet results
- **[Q_VALUE_FORMATS.md](Q_VALUE_FORMATS.md)**: Guide to q-value formats (true q-values, q-scores, percentages) and automatic detection
- **[COMET_REPOSITORY_AND_FRAGMENT_MATCHING.md](COMET_REPOSITORY_AND_FRAGMENT_MATCHING.md)**: Detailed explanation of Comet's repository and fragment matching algorithms

## How Comet actually matches fragments

In Comet, fragment matching is controlled primarily by these parameters:

- **`fragment_bin_tol`**
- **`fragment_bin_offset`**

### fragment_bin_tol (the main one)

- **Units:** Daltons  
- Defines the **half-width** of the fragment m/z matching window.  
- A theoretical fragment is considered matched if an observed peak falls within:  
  **|m/z_obs − m/z_theory| ≤ fragment_bin_tol**

**Typical values:**

| Instrument / mode        | fragment_bin_tol   |
|--------------------------|--------------------|
| Ion trap (low-res MS2)   | ~1.0005 Da         |
| Orbitrap / FT MS2        | 0.02 – 0.05 Da (sometimes tighter) |

So if you're running high-res ETD or UVPD MS2 on an Orbitrap, you've probably set something like `fragment_bin_tol = 0.02`. That corresponds roughly to:

- ~10 ppm at m/z 2000  
- ~40 ppm at m/z 500  

…but that ppm equivalence is **implicit**, not explicit.

### What Comet does *not* do

Comet does **not**:

- adapt tolerance based on m/z  
- use ppm-based matching  
- use S/N or noise models for fragment inclusion  
- require isotope patterns  
- distinguish noise vs signal beyond “is there a peak in the bin?”

So from Comet’s point of view: **any peak inside the bin is equally valid evidence.** Intensity matters only insofar as it contributes to XCorr, not whether the fragment is considered “real.”

## Future work

Plans include exposing the MS1 chromatogram filtering and data analysis workflow (e.g. from `extract_ms1_chromatograms.py`) through a **Streamlit** web app, so users can run filtering, inspect results, and explore data interactively in the browser.
