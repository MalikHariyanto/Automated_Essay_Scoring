#!/usr/bin/env bash
set -euo pipefail
# CPU/Python 3.12. Colab menyediakan apt sebagai root; Python AES tetap terisolasi.
cd "$(dirname "$0")/.."
python scripts/setup_notebook.py
if ! command -v Rscript >/dev/null; then
  apt-get update
  apt-get install -y --no-install-recommends r-base-core r-cran-dplyr
fi
Rscript -e 'if (!requireNamespace("irrCAC", quietly=TRUE)) install.packages("irrCAC", repos="https://cloud.r-project.org"); if (!requireNamespace("irrCAC", quietly=TRUE)) stop("irrCAC belum berhasil dipasang"); cat(R.version.string, "; irrCAC", as.character(packageVersion("irrCAC")), "\n")'
outputs/notebook_env/bin/python scripts/verify_ac2.py --require-r
