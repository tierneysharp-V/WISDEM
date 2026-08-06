#!/usr/bin/env bash
set -euo pipefail
source ~/miniforge3/etc/profile.d/conda.sh
conda activate wisdem-env
cd /home/s2795056/projects/GitHub/WISDEM/examples/foundation_batch
cat scenarios_22mw.csv
python run_foundation_batch.py scenarios_22mw.csv --turbine-csv turbine_lookup.csv --foundation monopile
