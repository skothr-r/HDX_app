#!/bin/bash
# Run the HDX Filter app with the correct conda environment.
# Use this if "conda activate hdx_app" + "streamlit run" gives import errors.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

progress_line() {
    local pct="$1"
    local msg="$2"
    local width=20
    local filled=$((pct * width / 100))
    local empty=$((width - filled))
    local bar_filled=""
    local bar_empty=""
    if ((filled > 0)); then
        printf -v bar_filled "%*s" "$filled" ""
        bar_filled="${bar_filled// /#}"
    fi
    if ((empty > 0)); then
        printf -v bar_empty "%*s" "$empty" ""
        bar_empty="${bar_empty// /-}"
    fi
    echo "[run_hdx_app] [${bar_filled}${bar_empty}] ${pct}%  ${msg}"
}

resolve_port() {
    local port="8501"
    local i=1
    while ((i <= $#)); do
        local arg="${!i}"
        if [[ "$arg" == "--server.port" ]]; then
            local j=$((i + 1))
            if ((j <= $#)); then
                port="${!j}"
            fi
            break
        elif [[ "$arg" == --server.port=* ]]; then
            port="${arg#--server.port=}"
            break
        fi
        i=$((i + 1))
    done
    echo "$port"
}

progress_line 10 "Resolving Python and Streamlit environment..."

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

progress_line 35 "Environment resolved: ${CONDA_PREFIX:-system python fallback}"

# Avoid Homebrew/conda lib conflicts (libjpeg, libtiff, PIL).
# Use ONLY conda libs - appending $DYLD_LIBRARY_PATH can pull in Homebrew's libjpeg.
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export DYLD_LIBRARY_PATH="${CONDA_PREFIX}/lib"

if [[ -x "$STREAMLIT" ]]; then
    APP_CMD=("$STREAMLIT" run hdx_filter_app.py "$@")
else
    APP_CMD=("$PYTHON" -m streamlit run hdx_filter_app.py "$@")
fi

APP_PORT="$(resolve_port "$@")"
progress_line 55 "Launching Streamlit on port ${APP_PORT}..."

# Ctrl+C behavior:
# - First Ctrl+C: cancel active workflow subprocesses spawned by Streamlit, keep app alive.
# - If no child jobs are active, stop Streamlit.
_descendants_of() {
    local parent_pid="$1"
    local child
    local kids
    kids="$(pgrep -P "$parent_pid" || true)"
    for child in $kids; do
        echo "$child"
        _descendants_of "$child"
    done
}

cancel_streamlit_children() {
    local app_pid="$1"
    local p
    local killed=0
    local descendants
    descendants="$(_descendants_of "$app_pid" | sort -u || true)"
    for p in $descendants; do
        if [[ "$p" != "$app_pid" ]] && kill -0 "$p" 2>/dev/null; then
            kill -INT "$p" 2>/dev/null || true
            killed=1
        fi
    done
    if [[ "$killed" -eq 1 ]]; then
        echo ""
        echo "[run_hdx_app] Ctrl+C: sent SIGINT to active workflow subprocess(es)."
        return 0
    fi
    return 1
}

"${APP_CMD[@]}" &
APP_PID=$!

progress_line 70 "Waiting for app health endpoint..."
ready=0
for attempt in $(seq 1 30); do
    if ! kill -0 "$APP_PID" 2>/dev/null; then
        break
    fi
    if command -v curl >/dev/null 2>&1; then
        if curl -fsS "http://127.0.0.1:${APP_PORT}/_stcore/health" >/dev/null 2>&1 || \
           curl -fsS "http://127.0.0.1:${APP_PORT}" >/dev/null 2>&1; then
            ready=1
            break
        fi
    fi
    progress_line 70 "Still starting... (${attempt}s elapsed)"
    sleep 1
done

if [[ "$ready" -eq 1 ]]; then
    progress_line 100 "App is ready at http://localhost:${APP_PORT}"
else
    progress_line 85 "Startup is still in progress; waiting for Streamlit output..."
fi

on_interrupt() {
    if cancel_streamlit_children "$APP_PID"; then
        return
    fi
    echo ""
    echo "[run_hdx_app] Ctrl+C: no active workflow subprocess found; stopping Streamlit..."
    kill -INT "$APP_PID" 2>/dev/null || true
}

trap on_interrupt INT
wait "$APP_PID"
