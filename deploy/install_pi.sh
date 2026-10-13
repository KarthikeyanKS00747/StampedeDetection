#!/usr/bin/env bash
set -euo pipefail
sudo apt update
sudo apt install -y python3-venv python3-opencv libatlas-base-dev
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo 'Installation complete. Run the fog service with: .venv/bin/python -m fog.app'
