#!/usr/bin/env bash
set -euo pipefail
# CPU/Python 3.12. Colab menyediakan apt sebagai root; Python AES tetap terisolasi.
cd "$(dirname "$0")/.."
python scripts/setup_notebook.py
apt-get update
apt-get install -y --no-install-recommends r-base-core r-cran-dplyr
Rscript -e 'install.packages("irrCAC", repos="https://cloud.r-project.org")'
outputs/notebook_env/bin/python scripts/verify_ac2.py --require-r
