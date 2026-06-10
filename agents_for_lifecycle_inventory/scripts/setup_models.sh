#!/usr/bin/env bash
# Agentic LCA
# Setup script for the accompanying research codebase.
#
# See the LICENSE file in the repository root for details.
#
# Note: This script downloads third-party model weights.
# See docs/THIRD_PARTY_LICENSES.md for details.

# create hidden model directory
MODEL_DIR="./.models"
mkdir -p $MODEL_DIR

# download https://github.com/SanderGi/PCB-Detection
curl -L https://huggingface.co/SanderGi/PCB-OBB/resolve/main/best.pt > $MODEL_DIR/pcb_obb_yolov11.pt
curl -L https://huggingface.co/SanderGi/PCB-SEG/resolve/main/best.pt > $MODEL_DIR/pcb_seg_yolov11.pt

# download segment anything models
curl -L https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth > $MODEL_DIR/sam_vit_b.pth
curl -L https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt > $MODEL_DIR/sam2_hiera_tiny.pt
