#!/bin/bash
# Run the HDX Filter app with the correct conda environment.
# Use this if "conda activate hdx_app" + "streamlit run" gives import errors.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Prefer conda env python; fallback to streamlit on PATH
if [[ -n "$CONDA_PREFIX" ]] && [[ "$CONDA_PREFIX" == *"hdx_app"* ]]; then
    PYTHON="$CONDA_PREFIX/bin/python"
    STREAMLIT="$CONDA_PREFIX/bin/streamlit"
else
    # Try hdx_app env by path (common anaconda/miniconda locations)
    for PREFIX in "$HOME/anaconda3/envs/hdx_app" "$HOME/miniconda3/envs/hdx_app" "/opt/anaconda3/envs/hdx_app" "/opt/miniconda3/envs/hdx_app"; do
        if [[ -x "$PREFIX/bin/python" ]]; then
            export CONDA_PREFIX="$PREFIX"
            PYTHON="$PREFIX/bin/python"
            STREAMLIT="$PREFIX/bin/streamlit"
            break
        fi
    done
fi

if [[ -z "$PYTHON" ]] || [[ ! -x "$PYTHON" ]]; then
    echo "Error: hdx_app conda environment not found."
    echo "Create it with: conda env create -f environment_hdx_app.yml"
    echo "Then run: conda activate hdx_app && ./run_hdx_app.sh"
    exit 1
fi

# Avoid Homebrew/conda lib conflicts (libjpeg, libtiff, PIL).
# Use ONLY conda libs - appending $DYLD_LIBRARY_PATH can pull in Homebrew's libjpeg.
export DYLD_LIBRARY_PATH="${CONDA_PREFIX}/lib"

if [[ -x "$STREAMLIT" ]]; then
    exec "$STREAMLIT" run hdx_filter_app.py "$@"
else
    exec "$PYTHON" -m streamlit run hdx_filter_app.py "$@"
fi
