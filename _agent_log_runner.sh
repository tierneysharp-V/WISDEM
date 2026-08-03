#!/usr/bin/env bash
LOG=/home/s2795056/projects/GitHub/WISDEM/_agent_log.txt
exec >"$LOG" 2>&1
set -x
source ~/miniforge3/etc/profile.d/conda.sh
conda activate wisdem-env
echo "=== pip show ==="
pip show wisdem
echo "=== import from /tmp ==="
cd /tmp
python /home/s2795056/projects/GitHub/WISDEM/_agent_import_test.py
echo "=== import exit $? ==="
