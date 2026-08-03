#!/bin/bash
set -e
cd /home/s2795056/projects/GitHub/WISDEM
{
  echo "=== which ==="
  which python3 || true
  which conda || true
  echo "=== conda env list ==="
  conda env list 2>&1 || true
  echo "=== try common pythons ==="
  for p in \
    /home/s2795056/miniforge3/envs/wisdem-env/bin/python \
    /home/s2795056/miniforge3/envs/*/bin/python \
    /home/s2795056/miniconda3/envs/wisdem-env/bin/python \
    /home/s2795056/miniconda3/envs/*/bin/python \
    /home/s2795056/anaconda3/envs/wisdem-env/bin/python \
    /home/s2795056/anaconda3/envs/*/bin/python; do
    if [ -x "$p" ]; then
      echo "FOUND: $p"
      "$p" -c "import wisdem; from ORBIT import ProjectManager; print('ok', '$p')" 2>&1 || true
    fi
  done
} > shell_probe.txt 2>&1
echo done >> shell_probe.txt
