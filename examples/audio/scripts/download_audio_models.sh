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
# Audio Model Download Script
#
# Models:
#   VGGish11  (sound classification, PTQ) — ~10.4 MB
#   YAMNet    (sound classification)      — ~14.3 MB
#   GTCRN     (speech enhancement)        —  ~286 KB
#
# Source: https://software-dl.ti.com/jacinto7/esd/modelzoo/audioai/models/onnx
#
# Models are saved next to their .link files so the tidlrunner pipeline
# auto-download mechanism does not re-fetch them.
#
# Usage:
#   bash examples/audio/scripts/download_audio_models.sh
##################################################################

set -e

# Resolve repo root (script lives at examples/audio/scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

MODELS_DIR="${REPO_ROOT}/data/configs/subset/models/audio"

##################################################################
# Check for wget
##################################################################
if ! command -v wget &> /dev/null; then
    echo "ERROR: wget not found. Install it with: sudo apt-get install wget"
    exit 1
fi

##################################################################
# Find and download each .link file
##################################################################
LINK_FILES=()
while IFS= read -r -d '' f; do
    LINK_FILES+=("$f")
done < <(find "${MODELS_DIR}" -name "*.onnx.link" -print0 | sort -z)

if [ ${#LINK_FILES[@]} -eq 0 ]; then
    echo "ERROR: No .link files found under ${MODELS_DIR}"
    exit 1
fi

echo ""
echo "INFO: Found ${#LINK_FILES[@]} model(s) to download."
echo ""

SUCCESS=0
SKIP=0
FAIL=0

for LINK_FILE in "${LINK_FILES[@]}"; do
    # Derive target .onnx path (strip trailing .link)
    TARGET="${LINK_FILE%.link}"
    MODEL_NAME="$(basename "${TARGET}")"
    MODEL_DIR="$(dirname "${TARGET}")"

    # Skip if already downloaded
    if [ -s "${TARGET}" ]; then
        echo "INFO: ${MODEL_NAME} already exists — skipping."
        SKIP=$((SKIP + 1))
        continue
    fi

    # Read URL from .link file (first non-empty line)
    URL="$(grep -m1 . "${LINK_FILE}" | tr -d '[:space:]')"
    if [ -z "${URL}" ]; then
        echo "ERROR: ${LINK_FILE} is empty."
        FAIL=$((FAIL + 1))
        continue
    fi

    echo "INFO: Downloading ${MODEL_NAME} ..."
    echo "      URL: ${URL}"
    echo "      Destination: ${TARGET}"

    mkdir -p "${MODEL_DIR}"
    if wget --show-progress -O "${TARGET}" "${URL}"; then
        if [ -s "${TARGET}" ]; then
            echo "INFO: ${MODEL_NAME} downloaded successfully."
            SUCCESS=$((SUCCESS + 1))
        else
            echo "ERROR: Downloaded file is empty — removing: ${TARGET}"
            rm -f "${TARGET}"
            FAIL=$((FAIL + 1))
        fi
    else
        echo "ERROR: Failed to download ${MODEL_NAME}."
        rm -f "${TARGET}"
        FAIL=$((FAIL + 1))
    fi
    echo ""
done

##################################################################
# Summary
##################################################################
echo "======================================================================"
echo " Download complete: ${SUCCESS} downloaded, ${SKIP} skipped, ${FAIL} failed"
echo " Models directory:  ${MODELS_DIR}"
echo "======================================================================"

if [ "${FAIL}" -gt 0 ]; then
    exit 1
fi
