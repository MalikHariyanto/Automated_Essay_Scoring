"""Browser checks buku Colab; fixture numerik hanya untuk verifikasi UI."""
import argparse
import hashlib
import json
import os
import random
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from playwright.sync_api import sync_playwright, expect

from aes.metrics import evaluate

ROOT = Path(__file__).resolve().parents[1]


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def check(private_notebook=None, private_html=None):
    out = ROOT / 'outputs/colab_study'
    out.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer(('127.0.0.1', 0),
        partial(QuietHandler, directory=str(ROOT)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    origin = 'http://127.0.0.1:' + str(server.server_port)
    rubric = {'traits': [{'name': t, 'categories': [1, 2, 3, 4, 5], 'min': 1, 'max': 5}
                        for t in ['isi', 'organisasi', 'kebahasaan', 'mekanik']]}
    result = {}
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(executable_path='/usr/bin/chromium',
                                       headless=True, args=['--no-sandbox'])
            page = browser.new_page(viewport={'width': 1440, 'height': 1050})
            errors, requests = [], []
            page.on('pageerror', lambda e: errors.append(str(e)))
            page.on('request', lambda r: requests.append(r.url))
            page.goto(origin + '/docs/belajar_colab_AI_interaktif.html')
            assert page.locator('.palette-choice').count() == 3
            assert page.locator('html').get_attribute('data-palette') == 'rose'
            assert page.locator('.chapter-tab').count() == 11
            assert page.locator('.page:visible').count() == 1
            assert page.locator('#setup-output').inner_text().find('2.6.0+cpu') >= 0
            page.screenshot(path=str(out / 'desktop_home.png'))
            for chapter in ['input', 'cleaning', 'token', 'vector', 'scenario',
                            'train', 'metric', 'bias', 'lanjut', 'code', 'mulai']:
                page.locator('#tab-' + chapter).click()
                assert page.locator('#page-' + chapter).is_visible()
                assert page.locator('.page:visible').count() == 1
                assert page.locator('#tab-' + chapter).get_attribute('aria-selected') == 'true'
            page.locator('#tab-mulai').focus()
            page.keyboard.press('ArrowRight')
            assert page.locator('#page-input').is_visible()
            page.keyboard.press('End')
            assert page.locator('#page-code').is_visible()
            page.locator('#tab-token').click()
            for n in [0, 1, 510, 511, 514, 892, 1020, 3000, 10000]:
                s = page.evaluate('(n)=>ColabStudy.windowStats(n)', n)
                covered = set()
                for start, length in zip(s['starts'], s['lengths']):
                    covered.update(range(start, start + length))
                    assert length + 2 <= 512
                assert covered == set(range(n))
            page.locator('#token-boundary').click()
            assert page.locator('#window-list .windowbar').count() == 2
            page.locator('#token-count').fill('')
            assert 'kosong' in page.locator('#token-error').inner_text()
            page.locator('#token-count').fill('514')
            page.locator('#tab-scenario').click()
            for s in ['S1', 'S2', 'S3']:
                page.locator('[data-scenario="' + s + '"]').click()
                page.locator('#batch').select_option('8')
                assert '[8,4]' in page.locator('#scenario-flow').inner_text()
            assert '[8,256]' in page.locator('#scenario-flow').inner_text()
            page.locator('#tab-train').click()
            for n in [1, 4, 8, 9, 16, 64]:
                for epochs in [1, 2, 5]:
                    s = page.evaluate('(a)=>ColabStudy.optimizerStats(a[0],a[1])', [n, epochs])
                    total = int(np.ceil(n / 8)) * epochs
                    warmup = min(int(np.ceil(.1 * total)), total - 1)
                    assert s['total'] == total and s['warmup'] == warmup
                    assert s['active'] == total - (1 if warmup else 0)
            page.locator('#train-n').fill('8')
            assert 'Step 2' in page.locator('#step-timeline').inner_text()
            assert page.locator('#step-timeline .zero').count() == 1
            page.locator('#tab-metric').click()
            assert page.locator('#actual-metrics tbody tr').count() == 12
            page.locator('#desk-tab-metric-output').click()
            page.locator('#scenario-filter').select_option('S3')
            assert page.locator('#actual-metrics tbody tr').count() == 4
            for metric in ['Pearson', 'QWK', 'AC2', 'RMSE']:
                page.locator('[data-metric="' + metric + '"]').click()
                assert metric in page.locator('#metric-walkthrough').inner_text()
            page.locator('[data-metric="AC2"]').click()
            assert '0.951807' in page.locator('#metric-walkthrough').inner_text()
            rng = random.Random(2026)
            cases = [([1, 2, 3, 4, 5], [1.5, 2.5, 3.5, 4.5, 8]),
                     ([3, 3, 4, 4], [3, 3, 3, 3]),
                     ([3, 3], [3, 3]), ([1, 2, 3], [.1, .1, .1]), ([1, 5], [5, 1])]
            for _ in range(50):
                n = rng.randint(2, 25)
                cases.append(([rng.randint(1, 5) for _ in range(n)],
                              [rng.uniform(-3, 8) for _ in range(n)]))
            for y, prediction in cases:
                actual = page.evaluate('(a)=>ColabStudy.metrics(a[0],a[1])', [y, prediction])
                reference = evaluate(np.repeat(np.array(y)[:, None], 4, axis=1),
                                     np.repeat(np.array(prediction)[:, None], 4, axis=1),
                                     rubric)['isi']
                for key in ['RMSE', 'Pearson', 'QWK', 'AC2']:
                    if reference[key] is None:
                        assert actual[key] is None
                    else:
                        assert abs(actual[key] - reference[key]) < 1e-10
            page.locator('[data-preset="konstan"]').click()
            assert 'undefined' in page.locator('#toy-results').inner_text()
            page.locator('#toy-prediction').fill('1,')
            assert 'kosong' in page.locator('#toy-error').inner_text()
            page.locator('#toy-prediction').fill('1e308,1e308')
            assert 'dibatasi' in page.locator('#toy-error').inner_text()
            page.locator('[data-preset="dekat"]').click()
            page.locator('[data-level="santai"]').click()
            assert page.locator('#page-metric .technical').is_hidden()
            page.locator('[data-level="teknis"]').click()
            assert page.locator('#page-metric .simple').is_hidden()
            page.locator('[data-level="dua"]').click()
            page.locator('#mark-read').check()
            assert page.locator('#progress-text').inner_text().startswith('1 dari')
            page.reload()
            assert page.locator('#mark-read').is_checked()
            page.locator('#reset').click()
            assert page.locator('#progress-text').inner_text().startswith('0 dari')
            assert page.locator('[data-quiz]').count() == 0
            assert page.locator('.code-lesson').count() == 27
            page.locator('#tab-scenario').click()
            assert 'AutoModel.from_pretrained' in page.locator('#page-scenario .guided-code').inner_text()
            assert 'self.roberta' in page.locator('#page-scenario .guided-code').inner_text()
            page.locator('#tab-metric').click()
            page.locator('#theme').click()
            assert page.locator('html').get_attribute('data-theme') == 'dark'
            page.screenshot(path=str(out / 'desktop_metrics_dark.png'))
            page.locator('#tab-lanjut').click()
            page.locator('[data-split="sepuluh"]').click()
            assert page.locator('#split-boxes .essaybox:not(.validation)').count() == 8
            assert 'Hipotetis' in page.locator('#split-caption').inner_text()
            page.locator('[data-split="delapan"]').click()
            assert page.locator('#split-boxes .excluded').count() == 2
            page.locator('#tab-code').click()
            page.locator('#source-select').select_option('src/aes/model.py')
            assert 'loss_mse' in page.locator('#source-code').inner_text()
            page.evaluate("Object.defineProperty(navigator,'clipboard',{value:{writeText:async t=>window.COPIED_SOURCE=t},configurable:true})")
            page.locator('#copy-source').click()
            assert 'loss_mse' in page.evaluate('window.COPIED_SOURCE')
            page.locator('#source-select').select_option('cell:11')
            assert 'metric' in page.locator('#source-code').inner_text()
            # The private viewer treats untrusted output as text, never executable HTML.
            fixture = {'cells': [{'cell_type': 'code', 'source': ['# TECHNICAL FIXTURE ONLY\n'],
                        'execution_count': 1, 'outputs': [{'output_type': 'display_data',
                        'data': {'text/html': ['<table><tr><td>TECHNICAL_OUTPUT</td></tr></table>',
                        '<script>window.PRIVATE_EXECUTED=true</script>',
                        '<img src="https://example.invalid/private" onerror="window.PRIVATE_EXECUTED=true">']}}]}]}
            page.locator('#tab-mulai').click()
            page.locator('#private-file').set_input_files({'name': 'technical.ipynb',
                'mimeType': 'application/json', 'buffer': json.dumps(fixture).encode()})
            page.locator('#desk-tab-mulai-colab').click()
            assert page.locator('#private-output').is_visible()
            assert 'TECHNICAL_OUTPUT' in page.locator('#private-output-cells').inner_text()
            assert page.evaluate('Boolean(window.PRIVATE_EXECUTED)') is False
            keys = page.evaluate('Object.keys(localStorage)')
            assert all(k in ['aes-study-v2:theme', 'aes-study-v2:level', 'aes-study-v2:read', 'aes-study-v2:palette', 'aes-study-v2:palette-main', 'aes-study-v2:palette-default'] for k in keys)
            page.locator('#forget-private-inline').click()
            assert page.locator('#private-output').is_hidden()
            private_cells = 0
            if private_notebook:
                page.locator('#private-file').set_input_files(str(private_notebook))
                expect(page.locator('#private-file-status')).to_contain_text('Source cell cocok', timeout=5000)
                for chapter, expected in [('input', 2), ('token', 1), ('metric', 1), ('code', 11)]:
                    page.locator('#tab-' + chapter).click()
                    page.locator('#desk-tab-' + chapter + '-colab').click()
                    assert page.locator('#private-cell-select option').count() == expected
                private_cells = 11
                page.locator('#forget-private-inline').click()
            page.set_viewport_size({'width': 390, 'height': 844})
            page.locator('#theme').click()
            page.locator('#tab-mulai').click()
            assert page.locator('#chapter-tabs').get_attribute('aria-orientation') == 'horizontal'
            page.screenshot(path=str(out / 'mobile_home.png'))
            for chapter in ['input', 'token', 'vector', 'scenario', 'train', 'metric', 'code']:
                page.locator('#tab-' + chapter).click()
                assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), chapter
            page.locator('#tab-scenario').click()
            page.screenshot(path=str(out / 'mobile_scenario.png'))
            page.locator('#tab-metric').click()
            page.screenshot(path=str(out / 'mobile_metrics.png'))
            assert page.locator('#print').count() == 0
            assert page.locator('.study-desk').count() == 11
            # Each code viewport scrolls without stretching the document.
            page.set_viewport_size({'width': 1440, 'height': 1050})
            page.locator('#tab-scenario').click()
            page.locator('#desk-tab-scenario-guided').click()
            viewport = page.locator('#page-scenario .desk-scroll')
            assert viewport.evaluate('(el)=>el.scrollHeight>el.clientHeight')
            before = page.evaluate('document.documentElement.scrollHeight')
            viewport.evaluate('(el)=>el.scrollTop=el.scrollHeight')
            page.wait_for_function('(el)=>el.scrollTop>0', arg=viewport.element_handle())
            assert page.evaluate('document.documentElement.scrollHeight') == before
            assert page.locator('#page-scenario .code-focus').count() >= 7
            assert page.locator('#page-scenario .margin-note').count() >= 7
            # Inner tabs are also operable with the keyboard.
            page.locator('#desk-tab-scenario-guided').focus()
            page.keyboard.press('ArrowRight')
            assert page.locator('#desk-pane-scenario-output').is_visible()
            page.keyboard.press('End')
            assert page.locator('#desk-pane-scenario-source').is_visible()
            embedded_outputs = 0
            if private_html:
                page.goto(origin + '/' + private_html.relative_to(ROOT).as_posix())
                assert 'sudah terpasang' in page.locator('#private-file-status').inner_text()
                page.locator('#tab-code').click()
                page.locator('#desk-tab-code-colab').click()
                assert page.locator('#private-cell-select option').count() == 11
                for cell in range(1, 12):
                    page.locator('#private-cell-select').select_option(str(cell))
                    embedded_outputs += page.locator('.saved-output').count()
                    assert page.locator('.output-note').count() == 1
                assert embedded_outputs == 44
                assert page.locator('.output-table table').count() > 0
                assert page.evaluate('Boolean(window.PRIVATE_EXECUTED)') is False
                page.locator('#forget-private-inline').click()
                assert page.locator('#private-output-cells').inner_text() == ''
            assert not errors, errors
            assert all(r.startswith(origin + '/') for r in requests), requests
            r_references = []
            if os.environ.get('AES_RSCRIPT'):
                from verify_ac2 import verify, FIXTURES
                r_references = verify(os.environ['AES_RSCRIPT'])
                for (y, prediction), reference in zip(FIXTURES, r_references):
                    actual = page.evaluate('(a)=>ColabStudy.metrics(a[0],a[1])', [y, prediction])
                    assert abs(actual['AC2'] - reference['irrCAC_AC2']) <= 5.01e-6
            result = {'exit_code': 0, 'browser': 'Chromium via Playwright', 'browser_version': browser.version,
                      'AC2_fixtures_matching_R': len(r_references),
                      'origin': 'Local HTTP; no user-facing localhost preview.',
                      'page_errors': errors, 'external_resource_requests': 0,
                      'chapters': 11, 'source_files': 17, 'code_cells': 11, 'guided_code_steps': 27, 'quizzes': 0,
                      'actual_metric_rows': 12, 'metric_fixtures_matching_Python': len(cases),
                      'window_coverage_cases': 9, 'optimizer_schedule_cases': 18,
                      'private_actual_notebook_cells_checked': private_cells,
                      'embedded_private_outputs_checked': embedded_outputs,
                      'untrusted_HTML_not_executed': True, 'private_data_not_in_localStorage': True,
                      'mobile_horizontal_overflow': False, 'print_control_removed': True, 'bounded_code_output_scroll': True,
                      'tested_interactions': ['tabs/keyboard', 'language modes', 'theme',
                          'reading persistence/reset', 'guided code names and explanation', 'source dropdown',
                          'S1/S2/S3 batch shapes', 'window boundaries',
                          'LR scheduling', 'metric filter/trace', 'numeric input errors',
                          '8/2 hypothetical diagram', 'local notebook open/forget', 'line highlight/margin notes', 'inner reading tabs/keyboard', 'bounded scrolling']}
            browser.close()
    finally:
        server.shutdown()
    path = ROOT / 'docs/belajar_colab_AI_interaktif.html'
    result['artifact_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    (out / 'browser_check.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--private-notebook', type=Path)
    parser.add_argument('--private-html', type=Path)
    args = parser.parse_args()
    check(args.private_notebook, args.private_html)
