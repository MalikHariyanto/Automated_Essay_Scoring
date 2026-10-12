#!/usr/bin/env bash
# R lokal Debian; tidak menulis ke /usr dan mempertahankan verifikasi APT/TLS/checksum.
set -euo pipefail
export AES_R_PREFIX="${AES_R_PREFIX:-/workspace/aes-r-local}"
export AES_APT_PREFIX="${AES_APT_PREFIX:-/workspace/aes-apt}"
if [[ -x "$AES_R_PREFIX/bin/Rscript" ]] && "$AES_R_PREFIX/bin/Rscript" -e 'stopifnot(as.character(packageVersion("irrCAC"))=="1.4")' >/dev/null 2>&1; then
  echo 'R/irrCAC 1.4 sudah siap'; exit 0
fi
if ! [[ -f /usr/share/keyrings/debian-archive-keyring.gpg ]]; then
  echo 'Helper ini khusus Debian. Pada Colab gunakan scripts/setup_colab.sh.' >&2; exit 2
fi
mkdir -p "$AES_APT_PREFIX/lists/partial" "$AES_APT_PREFIX/cache/archives/partial" "$AES_R_PREFIX/sources" "$AES_R_PREFIX/bin"
cat > "$AES_APT_PREFIX/sources.list" <<'SOURCES'
deb [signed-by=/usr/share/keyrings/debian-archive-keyring.gpg] https://deb.debian.org/debian trixie main
SOURCES
AES_APT_ARGS=(-o Debug::NoLocking=true -o "Dir::State::lists=$AES_APT_PREFIX/lists" -o "Dir::Cache=$AES_APT_PREFIX/cache" -o "Dir::Etc::sourcelist=$AES_APT_PREFIX/sources.list" -o Dir::Etc::sourceparts=- -o Dir::Etc::parts=- -o "APT::Sandbox::User=$(id -un)")
apt-get "${AES_APT_ARGS[@]}" update
apt-get "${AES_APT_ARGS[@]}" --download-only --no-install-recommends -y install r-base-core r-cran-dplyr r-cran-tidyr r-cran-stringr
python - <<'PY'
import os,subprocess,urllib.request,hashlib
from pathlib import Path
root=Path(os.environ['AES_R_PREFIX']);apt=Path(os.environ['AES_APT_PREFIX'])
for p in (apt/'cache/archives').glob('*.deb'):subprocess.run(['dpkg-deb','-x',str(p),str(root)],check=True)
for p in (root/'usr/lib/R/etc').iterdir():
    if p.is_symlink() and str(p.readlink()).startswith('/etc/R/'):
        target=root/str(p.readlink()).lstrip('/');p.unlink();p.symlink_to(target)
p=root/'etc/R/Renviron'
if not p.exists():p.write_bytes((root/'usr/lib/R/etc/Renviron.ucf').read_bytes())
p=root/'usr/lib/R/bin/R';s=p.read_text()
for key,path in [('R_HOME_DIR','/usr/lib/R'),('R_SHARE_DIR','/usr/share/R/share'),('R_INCLUDE_DIR','/usr/share/R/include'),('R_DOC_DIR','/usr/share/R/doc')]:
    s=s.replace(f'{key}={path}',f'{key}={root}{path}')
p.write_text(s)
# Rscript membutuhkan RHOME, R binary membutuhkan R_HOME. Wrapper lokal menautkan keduanya.
import shlex
q=lambda v:shlex.quote(str(v))
wrapper='#!/bin/sh\n'
for key,rel in [('R_HOME','usr/lib/R'),('R_SHARE_DIR','usr/share/R/share'),('R_INCLUDE_DIR','usr/share/R/include'),('R_DOC_DIR','usr/share/R/doc'),('R_LIBS_SITE','usr/lib/R/site-library'),('LD_LIBRARY_PATH','usr/lib/R/lib')]:wrapper+=f'export {key}={q(root/rel)}\n'
wrapper+=f'exec {q(root/"usr/lib/R/bin/exec/R")} "$@"\n'
(root/'bin/R').write_text(wrapper)
(root/'bin/Rscript').write_text(f'#!/bin/sh\nexport RHOME={q(root)}\nexec {q(root/"usr/bin/Rscript")} "$@"\n')
for p in [root/'bin/R',root/'bin/Rscript']:p.chmod(0o755)
source=root/'sources/irrCAC_1.4.tar.gz'
expected='07232bb9e8cd5989beba0447699d40caa15d928b40c52683c29f603fcdbde125'
if not source.exists():
    for base in ['https://cloud.r-project.org/src/contrib/','https://cloud.r-project.org/src/contrib/Archive/irrCAC/']:
        try:urllib.request.urlretrieve(base+source.name,source);break
        except urllib.error.HTTPError as exc:
            if exc.code!=404:raise
if hashlib.sha256(source.read_bytes()).hexdigest()!=expected:raise ValueError('Checksum irrCAC 1.4 berbeda; instalasi dihentikan')
subprocess.run([str(root/'bin/Rscript'),'-e',f'install.packages({str(source)!r}, repos=NULL, lib={str(root/"usr/lib/R/site-library")!r}); stopifnot(as.character(packageVersion("irrCAC"))=="1.4")'],check=True)
PY
"$AES_R_PREFIX/bin/Rscript" -e 'cat(R.version.string,"; irrCAC",as.character(packageVersion("irrCAC")),"\n")'
