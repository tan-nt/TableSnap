#!/bin/bash

# Install gdown if not available
pip install -q gdown

# File ID and output filename
FILE_ID="1z3y-0lSEadWgo5mI4uaijBPc_Ts-dbSB"
OUTPUT="downloaded_file.zip"
TARGET_DIR="/dataset"

# Download the file
gdown "https://drive.google.com/uc?id=${FILE_ID}" -O "${OUTPUT}"

# Create target directory if it doesn't exist
mkdir -p "${TARGET_DIR}"

# Unzip into the target folder
unzip -q "${OUTPUT}" -d "${TARGET_DIR}"

# Optionally: remove the zip after extraction
rm "${OUTPUT}"

echo "✅ Unzipped to ${TARGET_DIR}"
