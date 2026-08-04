#!/usr/bin/env bash
set -uo pipefail
source ~/miniforge3/etc/profile.d/conda.sh
conda activate wisdem-env
cd /home/s2795056/projects/GitHub/WISDEM/examples/foundation_batch
mkdir -p logs
echo "python=$(which python)"
echo "=== START moment sensitivity $(date -Is) ==="
python run_foundation_batch.py foundation_sensitivity_inputs_moment.csv --foundation both
ec=$?
echo "=== EXIT ${ec} $(date -Is) ==="
echo "${ec}" > logs/moment_sensitivity.exitcode
exit "${ec}"
