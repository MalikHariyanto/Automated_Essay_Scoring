"""Perbandingan hasil tersimpan: integritas sumber, parser, dan batas bukti."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import build_mlp_comparison as compare


class ComparisonIntegrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=json.loads(compare.TARGET.read_text())

    def test_baseline_and_intermediate_evidence_match_their_own_commits(self):
        for key in ['baseline','intermediate']:
            version=self.data[key]
            actual=compare.code_evidence(compare.committed(version['commit']))
            self.assertEqual(version['evidence'],actual)

    def test_three_scenarios_have_all_five_original_folds(self):
        first=self.data['baseline']
        for scenario in ['S1','S2','S3']:
            folds=[r for r in first['folds'] if r['Skenario']==scenario]
            self.assertEqual([r['Fold'] for r in folds],[1,2,3,4,5])
            for fold in range(1,6):
                history=[r for r in first['history'][scenario] if r['Fold']==fold]
                self.assertEqual([r['Epoch'] for r in history],[1,2,3,4,5])
                reported=next(r['RMSE (poin)'] for r in folds if r['Fold']==fold)
                self.assertAlmostEqual(min(r['RMSE Validasi (poin)'] for r in history),reported,delta=.0001)

    def test_missing_intermediate_results_are_not_invented(self):
        self.assertIsNone(self.data['intermediate']['metrics'])
        self.assertIsNone(self.data['intermediate']['history'])
        self.assertNotIn('prediction',self.data['baseline'])
        self.assertNotIn('native',self.data['baseline'])
        self.assertGreaterEqual(len(self.data['comparison_limits']),4)

    def test_html_table_parser_preserves_negative_metrics_and_column_names(self):
        parser=compare.SavedTable('<table><thead><tr><th>Skenario</th><th>QWK</th></tr></thead>'
                                 '<tbody><tr><td>S3</td><td>-0.21238397</td></tr></tbody></table>')
        self.assertEqual(parser.records(),[{'Skenario':'S3','QWK':-.21238397}])

    def test_log_parser_separates_cv_and_refit_even_without_newlines(self):
        text=('S3: validasi silang, fold 2 dari 5.'
              'S3 · epoch 1/5 · loss latih (MSE, target 0–1): 0.2 | RMSE validasi: 12.34 poin'
              'S3 · epoch 1/3 · loss latih (MSE, target 0–1): 0.1 | pelatihan final; data uji belum dibuka')
        rows=compare.history_from_log(text)['S3']
        self.assertEqual(rows[0]['Fold'],2)
        self.assertEqual(rows[0]['RMSE Validasi (poin)'],12.34)
        self.assertIsNone(rows[1]['Fold'])
        self.assertIsNone(rows[1]['RMSE Validasi (poin)'])
        self.assertEqual(rows[1]['Epoch'],1)

    def test_unmatched_baseline_code_is_rejected_before_reading_results(self):
        notebook=compare.committed(compare.BASE)
        notebook['cells'][30]['source']='raise RuntimeError("do not execute")'
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'notebook.ipynb'
            path.write_text(json.dumps(notebook))
            with self.assertRaisesRegex(ValueError,'Source notebook awal'):
                compare.extract(path)


if __name__=='__main__':
    unittest.main()
