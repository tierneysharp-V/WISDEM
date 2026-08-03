#!/bin/bash
set -e
cd /home/s2795056/projects/GitHub/WISDEM
for init in ~/miniforge3/etc/profile.d/conda.sh ~/miniconda3/etc/profile.d/conda.sh ~/anaconda3/etc/profile.d/conda.sh; do
  if [ -f "$init" ]; then
    # shellcheck source=/dev/null
    source "$init"
    break
  fi
done
conda activate wisdem-env 2>/dev/null || conda activate base
python _agent_import_test.py
