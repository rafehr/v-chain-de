#!/bin/bash

###############################################################################
# Script: 01_download_data.sh
# Purpose: Downloads the openfoodfacts data in JSON Lines format and extracts
#          only those products that are vegan and sold in Germany.
# Usage: ./scripts/01_download_data.sh
# Dependencies: curl, zcat, grep
# Output: Generates data/openfoodfacts-products-de.jsonl
###############################################################################

SCRIPT_DIR=$(dirname "$0")

source "$SCRIPT_DIR/../.env"

DATA_DIR="$SCRIPT_DIR/../data"
mkdir -p "$DATA_DIR"

OUTPUT_FILE="$SCRIPT_DIR/../$VEGAN_DATA_PATH"

curl -L "$OFF_URL" | \
    zcat | \
    grep '"en:germany"' | \
    grep -E '"vegan":"yes"|"vegan":"maybe"' > "$OUTPUT_FILE"