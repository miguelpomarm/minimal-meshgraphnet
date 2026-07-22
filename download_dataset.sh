#!/bin/bash
# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
#
# Download a MeshGraphNets dataset from Google Cloud Storage.
#
# NOTE: This script downloads the ORIGINAL dataset from DeepMind. To use it
# with the proposed architecture, you must run the Sinkhorn-Knopp
# preprocessing step afterwards (see preprocessing/preprocess_dataset.py).
#
# Available dataset names:
#   airfoil, cylinder_flow, deforming_plate, flag_minimal, flag_simple,
#   flag_dynamic, flag_dynamic_sizing, sphere_simple, sphere_dynamic,
#   sphere_dynamic_sizing
#
# Usage:
#   bash download_dataset.sh <dataset_name> <target_directory>
#
# Example:
#   bash download_dataset.sh cylinder_flow /tmp/datasets

set -e

DATASET_NAME="${1}"
OUTPUT_DIR="${2}"

if [[ -z "${DATASET_NAME}" || -z "${OUTPUT_DIR}" ]]; then
    echo "Usage: bash download_dataset.sh <dataset_name> <target_directory>"
    echo "Example: bash download_dataset.sh cylinder_flow /tmp/datasets"
    exit 1
fi

BASE_URL="https://storage.googleapis.com/dm-meshgraphnets/${DATASET_NAME}"
TARGET_DIR="${OUTPUT_DIR}/${DATASET_NAME}"

mkdir -p "${TARGET_DIR}"

echo "Downloading ${DATASET_NAME} into ${TARGET_DIR}..."

for file in meta.json train.tfrecord valid.tfrecord test.tfrecord; do
    echo "  ${file}"
    wget -O "${TARGET_DIR}/${file}" "${BASE_URL}/${file}"
done

echo "✓ Done. Dataset saved to ${TARGET_DIR}"