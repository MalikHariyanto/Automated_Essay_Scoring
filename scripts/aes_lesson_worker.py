"""Kernel numerik persisten untuk %%aes; source cell benar-benar dieksekusi."""
import argparse
import contextlib
import json
from pathlib import Path
import sys
import traceback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root/'src'))
    def event(kind, **values):
        print(args.prefix+json.dumps({'kind': kind, **values}, ensure_ascii=False, allow_nan=False), flush=True)
    def show(frame, title='', rows=8, columns=8):
        # Preview ringkas saja; exporter Excel tetap menyimpan seluruh koordinat.
        import pandas as pd
        frame = frame if isinstance(frame, pd.DataFrame) else pd.DataFrame(frame)
        view = frame.iloc[:rows, :columns].copy()
        view = view.map(lambda value: value.item() if hasattr(value, 'item') else value)
        safe = json.loads(view.to_json(orient='split', force_ascii=False, default_handler=str))
        event('table', title=title, columns=safe['columns'], rows=safe['data'], shape=list(frame.shape))
    namespace = {'__name__': '__aes_notebook__', 'ROOT': root, 'show': show}
    event('ready', python=sys.version.split()[0])
    for line in sys.stdin:
        try:
            payload = json.loads(line)
            if payload.get('command') == 'close':
                break
            code = compile(payload['code'], payload.get('filename', '<aes-cell>'), 'exec')
            exec(code, namespace)
        except BaseException:
            event('error', traceback=traceback.format_exc())
        else:
            event('done')


if __name__ == '__main__':
    main()
