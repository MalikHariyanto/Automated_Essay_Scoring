#!/usr/bin/env bash
set -euo pipefail
# Jalankan dari root repo pada Colab. Colab menyediakan apt sebagai root.
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
apt-get update
apt-get install -y --no-install-recommends r-base-core r-cran-dplyr
Rscript -e 'install.packages("irrCAC", repos="https://cloud.r-project.org")'
python scripts/verify_ac2.py --require-r
