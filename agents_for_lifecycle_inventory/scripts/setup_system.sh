#!/usr/bin/env bash
# Agentic LCA
# Setup script for the accompanying research codebase.
#
# See the LICENSE file in the repository root for details.

if [ "$(uname)" == "Darwin" ]; then
    # Mac OS X specific  
    brew install tesseract
elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    # GNU/Linux specific
    sudo apt update
    sudo apt install tesseract-ocr tesseract-data-eng
else
    echo "please install tesseract manually for this OS"
fi
