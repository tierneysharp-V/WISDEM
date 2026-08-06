#!/usr/bin/env bash
set -uo pipefail
source ~/miniforge3/etc/profile.d/conda.sh
conda activate wisdem-env
cd /home/s2795056/projects/GitHub/WISDEM/examples/foundation_batch
mkdir -p logs

run_one() {
  local foundation="$1"
  local logfile="logs/diagnostic_${foundation}.log"
  local ecfile="logs/diagnostic_${foundation}.exitcode"
  {
    echo "=== START ${foundation} $(date -Is) ==="
    python run_foundation_batch.py diagnostic_scenarios.csv \
      --turbine-csv turbine_lookup.csv \
      --foundation "${foundation}"
    local ec=$?
    echo "=== ${foundation} EXIT ${ec} $(date -Is) ==="
    echo "${ec}" > "${ecfile}"
    return "${ec}"
  } 2>&1 | tee "${logfile}"
  # tee masks exit; read from exitcode file
  return "$(cat "${ecfile}")"
}

echo "python=$(which python)"
echo "scenarios:"
cat diagnostic_scenarios.csv
run_one jacket
JEC=$?
run_one monopile
MEC=$?
echo "DONE jacket=${JEC} monopile=${MEC}"
# overall fail if either failed
if [[ "${JEC}" != "0" || "${MEC}" != "0" ]]; then
  exit 1
fi
exit 0
