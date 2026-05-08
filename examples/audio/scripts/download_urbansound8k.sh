#!/bin/bash

# Copyright (c) 2018-2025, Texas Instruments
# All Rights Reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

##################################################################
# UrbanSound8K Dataset Download Script
#
# Usage:
#   bash examples/audio/scripts/download_urbansound8k.sh            # default Zenodo URL
#   bash examples/audio/scripts/download_urbansound8k.sh <URL>      # custom URL override
#
# Attribution (required by dataset terms):
#   J. Salamon, C. Jacoby and J. P. Bello, "A Dataset and Taxonomy for Urban
#   Sound Research", 22nd ACM International Conference on Multimedia, Orlando
#   USA, Nov. 2014. DOI: 10.1145/2647868.2655045
#   https://urbansounddataset.weebly.com/urbansound8k.html
#
# Dataset: UrbanSound8K — 8732 labeled urban sound clips (<=4s), 10 classes
# License: Creative Commons Attribution (CC BY 4.0)
# Size:    ~5.6 GB (compressed)
##################################################################

set -e

DATASET_ARCHIVE="UrbanSound8K.tar.gz"
DEFAULT_URL="https://zenodo.org/records/1203745/files/UrbanSound8K.tar.gz"

# Resolve repo root (script lives at examples/audio/scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

DATASET_DIR="${REPO_ROOT}/data/datasets/UrbanSound8K"
DOWNLOAD_DIR="${REPO_ROOT}/data/datasets"

##################################################################
# Skip if already downloaded
##################################################################
if [ -d "${DATASET_DIR}" ] && [ -f "${DATASET_DIR}/metadata/UrbanSound8K.csv" ]; then
    echo "INFO: UrbanSound8K already exists at ${DATASET_DIR} — skipping download."
    echo "      Use 'rm -rf ${DATASET_DIR}' to force a fresh download."
    exit 0
fi

##################################################################
# Resolve download URL (argument overrides default)
##################################################################
DATASET_URL="${1:-${DEFAULT_URL}}"

echo ""
echo "======================================================================"
echo " UrbanSound8K Dataset Download"
echo "======================================================================"
echo ""
echo " By using this dataset you agree to cite:"
echo "   J. Salamon, C. Jacoby and J. P. Bello, 'A Dataset and Taxonomy"
echo "   for Urban Sound Research', ACM-MM 2014."
echo "   DOI: 10.1145/2647868.2655045"
echo "   https://urbansounddataset.weebly.com/urbansound8k.html"
echo "======================================================================"
echo ""

##################################################################
# Check for wget
##################################################################
if ! command -v wget &> /dev/null; then
    echo "ERROR: wget not found. Install it with: sudo apt-get install wget"
    exit 1
fi

##################################################################
# Download
##################################################################
mkdir -p "${DOWNLOAD_DIR}"

echo ""
echo "INFO: Downloading UrbanSound8K (~5.6 GB)..."
echo "      URL: ${DATASET_URL}"
echo "      Destination: ${DOWNLOAD_DIR}/${DATASET_ARCHIVE}"
echo ""

wget --show-progress -O "${DOWNLOAD_DIR}/${DATASET_ARCHIVE}" "${DATASET_URL}"

##################################################################
# Extract
##################################################################
echo ""
echo "INFO: Extracting ${DATASET_ARCHIVE}..."
tar -xzf "${DOWNLOAD_DIR}/${DATASET_ARCHIVE}" -C "${DOWNLOAD_DIR}"

##################################################################
# Cleanup tarball
##################################################################
echo "INFO: Removing archive..."
rm -f "${DOWNLOAD_DIR}/${DATASET_ARCHIVE}"

##################################################################
# Verify
##################################################################
if [ ! -d "${DATASET_DIR}/audio/fold1" ]; then
    echo "ERROR: Extraction failed — ${DATASET_DIR}/audio/fold1/ not found."
    exit 1
fi

if [ ! -f "${DATASET_DIR}/metadata/UrbanSound8K.csv" ]; then
    echo "ERROR: Extraction failed — ${DATASET_DIR}/metadata/UrbanSound8K.csv not found."
    exit 1
fi

echo ""
echo "INFO: UrbanSound8K downloaded and extracted successfully."
echo "      Path: ${DATASET_DIR}"
echo "      Structure: audio/fold1..fold10/, metadata/UrbanSound8K.csv"
