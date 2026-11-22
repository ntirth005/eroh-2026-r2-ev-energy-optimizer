#!/usr/bin/env bash
# Full pipeline for one data mode.  Usage:  ./run_pipeline.sh real      (IEX + ERA5 files in data/)
#                                           ./run_pipeline.sh synthetic (fully synthetic weather and price)
set -euo pipefail
MODE=${1:-real}
export DATA_MODE=$MODE
cd "$(dirname "$0")/src"
mkdir -p ../data ../out/tables ../out/charts
python degradation.py      # fit battery ageing from the state-of-health records -> data/deg.json
python calibrate.py        # calibrate the incumbent baseline to the case KPIs  -> data/calib.json
python forecast.py         # LightGBM forecasts, reconciliation, error scenarios -> data/forecasts.pkl
python run_all.py          # ladder, steering sweep, stress scenarios, sensitivities (uses all CPU cores)
python analyze.py          # tables + charts -> out/
