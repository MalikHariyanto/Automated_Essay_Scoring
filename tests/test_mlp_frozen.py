"""Integritas agregat hasil frozen; tidak menjalankan model atau training."""
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import build_mlp_frozen as builder

class FrozenBookEvidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=json.loads(builder.TARGET.read_text())

    def test_complete_folds_and_epoch_selection(self):
        d=self.data
        self.assertEqual((d['rows'],d['development'],d['test']),(300,240,60))
        for s in ('S1','S2','S3'):
            fold=[r for r in d['folds'] if r['Skenario']==s]
            self.assertEqual(sorted(r['Fold'] for r in fold),[1,2,3,4,5])
            selected=[]
            for f in fold:
                summary=next(r for r in d['foldSummary'] if r['Skenario']==s and r['Fold']==f['Fold'])
                rows=[r for r in d['history'][s] if r.get('Fold')==f['Fold'] and r.get('RMSE Validasi (poin)') is not None]
                row=next(r for r in rows if r['Epoch']==summary['Epoch Terpilih'])
                self.assertAlmostEqual(row['RMSE Validasi (poin)'],f['RMSE (poin)'])
                self.assertLessEqual(summary['Epoch Terpilih'],summary['Epoch Dijalankan'])
                self.assertEqual(summary['Epoch Maksimal'],25)
                selected.append(summary['Epoch Terpilih'])
            refit=[r for r in d['history'][s] if r['Tahap']=='Latihan final']
            self.assertEqual(refit[-1]['Epoch'],math.ceil(statistics.median(selected)))
            summary=next(r for r in d['cvSummary'] if r['Skenario']==s)
            self.assertAlmostEqual(summary['mean'],statistics.mean(r['RMSE (poin)'] for r in fold))
            self.assertAlmostEqual(summary['sd'],statistics.stdev(r['RMSE (poin)'] for r in fold))

    def test_pairing_privacy_and_vector_proof(self):
        d=self.data
        self.assertTrue(d['teacher_test_pairs_identical'])
        self.assertEqual([r['rows'] for r in d['pairing'].values()],[300,300,1200])
        self.assertTrue(all(r['identical'] for r in d['pairing'].values()))
        self.assertEqual(d['vector_check']['dimensions'],768)
        self.assertEqual(d['vector_check']['max_abs_difference_S3_vs_pretrained'],0)
        self.assertLessEqual(d['metrics_recheck']['max_absolute_difference'],1e-6)
        for r in d['predictionImpact']:
            self.assertEqual(r['new_lower_error']+r['old_lower_error']+r['equal_error'],60)
        self.assertFalse(any(k in d for k in ['predictions','student_texts','pairs','vectors']))
        self.assertEqual(len(d['books']),5)
        self.assertEqual(sum(len(b['sheets']) for b in d['books']),41)

    def test_reference_source_is_exact_and_runtime_not_claimed(self):
        ref=self.data['source_reference']
        nb=json.loads(subprocess.check_output(['git','show',ref['commit']+':'+ref['path']],cwd=ROOT))
        for evidence in ref['evidence'].values():
            self.assertEqual(evidence['source'],builder.source(nb['cells'][evidence['cell']]))
        self.assertIn('Notebook eksekusi run ini belum diunggah',ref['status'])
        model=next(r for r in self.data['models'] if r['Skenario']=='S3')
        self.assertIn('frozen',model['Encoder'])
        self.assertIn('MLP',model['Lapisan Prediksi'])

if __name__=='__main__':unittest.main()
