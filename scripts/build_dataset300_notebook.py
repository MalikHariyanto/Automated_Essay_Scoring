"""Entrypoint kompatibel: notebook03 berisi kode inline dan ekspor Excel."""
from build_dataset300_inline import build

if __name__ == '__main__':
    print('Notebook publik:', len(build()['cells']), 'cell; dua mirror identik tanpa output.')
