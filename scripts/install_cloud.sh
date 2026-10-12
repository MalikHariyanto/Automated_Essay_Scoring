#!/usr/bin/env bash
# Dari checkout yang sudah tersedia. Tidak membuat worktree atau mengubah CSV/JSON.
set -euo pipefail
cd /workspace/Automated_Essay_Scoring
if ! [[ -x /workspace/aes-venv/bin/python ]]; then
  python -m venv /workspace/aes-venv
fi
export HF_HOME=/workspace/hf-cache
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
/workspace/aes-venv/bin/python -m pip install -r requirements.lock.txt
/workspace/aes-venv/bin/python -m pip install -e . --no-deps
bash scripts/setup_r_local.sh
AES_RSCRIPT=/workspace/aes-r-local/bin/Rscript /workspace/aes-venv/bin/python -m pytest -q
/workspace/aes-venv/bin/python -m aes.cli inspect --models
/workspace/aes-venv/bin/python scripts/verify_ac2.py --require-r --rscript /workspace/aes-r-local/bin/Rscript
