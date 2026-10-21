"""Bukti integritas buku: bukan training atau pengukuran kualitas AES baru."""
import json
from html.parser import HTMLParser
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import build_mlp_study as builder


class Plain(HTMLParser):
    def __init__(self, text):
        super().__init__(convert_charrefs=True)
        self.parts=[]
        self.feed(text)

    def handle_data(self, text):
        self.parts.append(text)


class MLPBookIntegrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.snapshot=json.loads(builder.SNAPSHOT.read_text())
        cls.data=builder.content(cls.snapshot)

    def test_every_code_cell_matches_pinned_source(self):
        notebook=json.loads(subprocess.check_output(['git','show',builder.COMMIT+':'+builder.NOTEBOOK_PATH],cwd=ROOT))
        originals={i:builder.source(c) for i,c in enumerate(notebook['cells']) if c['cell_type']=='code'}
        self.assertEqual({c['index']:c['source'] for c in self.data['cells']},originals)
        self.assertEqual(len(originals),30)
        self.assertEqual(len(self.data['markdown']),17)

    def test_highlighting_is_reversible_and_escapes_markup(self):
        for c in self.data['cells']:
            restored='\n'.join(''.join(Plain(line).parts) for line in c['htmlLines'])
            self.assertEqual(restored,c['source'])
        hostile='message = "</script><img src=x onerror=alert(1)>"\n# < > &'
        highlighted='\n'.join(builder.highlighted_lines(hostile))
        self.assertNotIn('</script>',highlighted)
        self.assertEqual(''.join(Plain(highlighted).parts),hostile)

    def test_fundamental_notes_cover_exact_original_lines(self):
        found=0
        for index,anchor,title,_,_ in builder.PINS:
            cell=next(c for c in self.data['cells'] if c['index']==index)
            note=next(n for n in cell['notes'] if n['title']==title)
            self.assertIn(anchor,cell['source'].split('\n')[note['line']-1])
            found+=1
        self.assertEqual(found,52)
        self.assertEqual(self.data['coverage']['annotations'],120)

    def test_output_whitelist_and_full_workbook_inventory(self):
        self.assertEqual(set(self.snapshot['native']),{'12','18','32','39'})
        self.assertEqual(len(self.snapshot['books']),5)
        self.assertEqual(sum(len(b['sheets']) for b in self.snapshot['books']),41)
        self.assertEqual(self.snapshot['saved_outputs'],34)
        self.assertNotIn('pairs',self.snapshot)
        self.assertNotIn('responses',self.snapshot)

    def test_aggregate_matrices_preserve_all_test_observations(self):
        for scenario,matrix in self.snapshot['matrices'].items():
            self.assertEqual(len(matrix),11)
            self.assertTrue(all(len(row)==11 for row in matrix))
            self.assertTrue(all(type(v) is int and v>=0 for row in matrix for v in row))
            self.assertEqual(sum(map(sum,matrix)),self.snapshot['test'])

    def test_cv_selection_matches_actual_history(self):
        for entry in self.snapshot['foldSummary']:
            rows=[r for r in self.snapshot['history'][entry['Skenario']] if r['Fold']==entry['Fold']]
            valid=[r for r in rows if isinstance(r.get('RMSE Validasi (poin)'),(float,int))]
            chosen=next(r for r in valid if r['Epoch']==entry['Epoch Terpilih'])
            self.assertAlmostEqual(chosen['RMSE Validasi (poin)'],min(r['RMSE Validasi (poin)'] for r in valid))


if __name__=='__main__':
    unittest.main()
