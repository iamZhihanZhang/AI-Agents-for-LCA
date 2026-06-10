#!/usr/bin/env bash
# Agentic LCA
# Setup script for the accompanying research codebase.
#
# See the LICENSE file in the repository root for details.

# make sure the python version is 3.10 or higher
test $(python3 --version | cut -d ' ' -f 2 | cut -d '.' -f 1) -ge 3 && test $(python3 --version | cut -d ' ' -f 2 | cut -d '.' -f 2) -ge 10 || echo "Please install python 3.10 or higher"

# create virtual environment
python3 -m venv venv
. venv/bin/activate
pip3 install --upgrade pip setuptools wheel

# install lca-agentic as editable module
# pip3 install -e .                      # minimal (agents + MCP only)
# pip3 install -e ".[vision,sam]"        # With vision + SAM
pip3 install -e ".[vision,sam,ui,dev]"   # full research environment
