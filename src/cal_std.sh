#!/bin/bash
# Script: calculate_total_std_sum_arg.sh
# Usage: ./calculate_total_std_sum_arg.sh <path/to/your/input_file.nc>

# --- Configuration ---
# Set desired output format (e.g., '%.4f' for 4 decimal places)
OUTPUT_FORMAT='%.6f'

# --- Error Handling ---
# Exit immediately if a command exits with a non-zero status
set -e
# Exit if any variable is used uninitialized
set -u

# Check if an argument was provided
if [ $# -eq 0 ]; then
    echo "Error: No input file path provided." >&2
	    echo "Usage: $0 <path/to/your/input_file.nc>" >&2
    exit 1
fi

# Assign the first command-line argument to a variable
INPUT_FILE="$1"

# Check if the input file exists and is readable
if [ ! -r "$INPUT_FILE" ]; then
    echo "Error: Input file '$INPUT_FILE' not found or is not readable." >&2
    exit 1
fi

echo "--- CDO Calculation Initiated ---"
echo "Processing file: $INPUT_FILE"

# --- Main CDO Operation ---
# Calculates the Spatial Sum of the Time-Series Standard Deviation
#TOTAL_STD_SUM=$(cdo outputf,"$OUTPUT_FORMAT" -fldsum -timstd "$INPUT_FILE")
TOTAL_AVG_OF_STD=$(cdo outputf,"$OUTPUT_FORMAT" -fldmean -timstd "$INPUT_FILE")

# --- Output and Verification ---
echo ""
echo "--- Results ---"
echo "Operation: Spatial Sum of Time-Series Standard Deviation"
echo "Total Sum (Formatted as $OUTPUT_FORMAT): $TOTAL_AVG_OF_STD"
echo "-----------------"

exit 0
