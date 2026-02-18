#!/bin/bash
# Script to generate .pin file from Comet search and run Percolator

set -e

CSV_FILE="${1:-data/WT_nep2_0MUrea_08.csv}"
FASTA_FILE="${2:-data/Ube2D3.fasta}"
MZXML_FILE="${3:-data/WT_nep2_0MUrea_08.mzXML}"

if [ ! -f "$CSV_FILE" ]; then
    echo "Error: CSV file not found: $CSV_FILE"
    exit 1
fi

# Extract base name
BASE_NAME=$(basename "$CSV_FILE" .csv)
BASE_NAME="${BASE_NAME%_PP}"  # Remove _PP suffix if present
DIR_NAME=$(dirname "$CSV_FILE")

PIN_FILE="${DIR_NAME}/${BASE_NAME}.pin"
PARAMS_FILE="comet.params.new"

echo "=========================================="
echo "Generating .pin file and running Percolator"
echo "=========================================="
echo "CSV file: $CSV_FILE"
echo "FASTA file: $FASTA_FILE"
echo "mzXML file: $MZXML_FILE"
echo "Expected .pin file: $PIN_FILE"
echo ""

# Check if .pin file already exists
if [ -f "$PIN_FILE" ]; then
    echo "Found existing .pin file: $PIN_FILE"
    echo "Skipping Comet search..."
else
    echo "=========================================="
    echo "Step 1: Generate .pin file with Comet"
    echo "=========================================="
    echo ""
    echo "To generate a .pin file, you need to:"
    echo "1. Edit comet.params.new and set:"
    echo "   output_percolatorfile = 1"
    echo ""
    echo "2. Run Comet:"
    echo "   ./comet.exe -P$PARAMS_FILE $MZXML_FILE"
    echo ""
    echo "This will generate: $PIN_FILE"
    echo ""
    read -p "Press Enter if you've already generated the .pin file, or Ctrl+C to exit..."
fi

# Check if .pin file exists now
if [ ! -f "$PIN_FILE" ]; then
    echo "Error: .pin file not found: $PIN_FILE"
    echo "Please generate it first by running Comet with output_percolatorfile = 1"
    exit 1
fi

echo ""
echo "=========================================="
echo "Step 2: Run Percolator"
echo "=========================================="
echo ""

# Check if Percolator is installed
if ! command -v percolator &> /dev/null; then
    echo "Percolator not found. Installing..."
    echo ""
    echo "Option 1 (recommended): Install via conda"
    echo "  conda install -c bioconda percolator"
    echo ""
    echo "Option 2: Download from"
    echo "  https://github.com/percolator/percolator/releases"
    echo ""
    exit 1
fi

# Run Percolator
PERCOLATOR_OUTPUT="${DIR_NAME}/${BASE_NAME}.psms"
echo "Running Percolator on $PIN_FILE..."
echo "Output will be: $PERCOLATOR_OUTPUT"
echo ""

percolator -Y -m "${DIR_NAME}/${BASE_NAME}" "$PIN_FILE"

if [ ! -f "$PERCOLATOR_OUTPUT" ]; then
    echo "Error: Percolator output not found: $PERCOLATOR_OUTPUT"
    echo "Checking for alternative output files..."
    if [ -f "${DIR_NAME}/${BASE_NAME}.percolator" ]; then
        PERCOLATOR_OUTPUT="${DIR_NAME}/${BASE_NAME}.percolator"
        echo "Found: $PERCOLATOR_OUTPUT"
    else
        exit 1
    fi
fi

echo ""
echo "=========================================="
echo "Step 3: Merge q-values into CSV"
echo "=========================================="
echo ""

# Run Python script to merge
OUTPUT_CSV="${DIR_NAME}/${BASE_NAME}_with_qvalues.csv"
python3 add_percolator_qvalues.py \
    --csv "$CSV_FILE" \
    --percolator-output "$PERCOLATOR_OUTPUT" \
    --output "$OUTPUT_CSV"

if [ -f "$OUTPUT_CSV" ]; then
    echo ""
    echo "=========================================="
    echo "Success!"
    echo "=========================================="
    echo "CSV with q-values: $OUTPUT_CSV"
    echo ""
    echo "You can now use this file with the visualization script:"
    echo "  python3 combined_overhang_visualization.py --csv $OUTPUT_CSV --fasta $FASTA_FILE --two-pass"
else
    echo "Error: Failed to create output CSV"
    exit 1
fi
