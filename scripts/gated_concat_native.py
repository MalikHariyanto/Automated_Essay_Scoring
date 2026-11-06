"""Visible E51 gate and audit helpers; model labels never enter the gate."""


class GatedConcatenation(nn.Module):
    """Two per-answer softmax weights; factor2 starts at the identity concat."""
    def __init__(self, mode='softmax_2x_identity'):
        super().__init__()
        if mode!='softmax_2x_identity':raise ValueError('E51 mengunci softmax dua cabang dengan pengali2.')
        self.mode=mode
        with torch.random.fork_rng(devices=[]):
            self.logits=nn.Linear(1024,2)
        nn.init.zeros_(self.logits.weight)
        nn.init.zeros_(self.logits.bias)
        self.last_alpha=None

    def forward(self, roberta, bge):
        if roberta.ndim!=2 or bge.ndim!=2 or roberta.shape!=(bge.shape[0],768) or bge.shape[1]!=256:
            raise ValueError('Gate memerlukan RoBERTa[B,768] dan proyeksiBGE[B,256].')
        joined=torch.cat([roberta,bge],dim=-1)
        with torch.autocast(device_type=joined.device.type,enabled=False):
            alpha=torch.softmax(self.logits(joined.float()),dim=-1)
        self.last_alpha=alpha.detach()
        gated=torch.cat([roberta*(2*alpha[:,0:1]).to(roberta.dtype),
                         bge*(2*alpha[:,1:2]).to(bge.dtype)],dim=-1)
        return gated,alpha

    def report(self):
        return {'Mode':self.mode,'Dimensi Input':1024,'Jumlah Cabang':2,'Pengali Alpha':2,
                'Jumlah Parameter':sum(p.numel() for p in self.parameters()),
                'Norma Bobot':float(self.logits.weight.detach().float().norm().cpu()),
                'Norma Bias':float(self.logits.bias.detach().float().norm().cpu())}


def gate_prediction_frame(bundle, indices, prediction, gate_rows):
    """Persist the gates captured in the SAME forward as held-out predictions."""
    indices=[int(i) for i in indices]
    expected=[bundle.rows[i]['row_id'] for i in indices]
    observed=[]
    for r in gate_rows:
        if 'row_id' in r:observed.append(r['row_id'])
        else:
            i=r.get('dataset_index')
            if type(i) is not int or not 0<=i<len(bundle.rows):raise ValueError('Indeks gate tidak sah.')
            observed.append(bundle.rows[i]['row_id'])
    if len(prediction)!=len(indices) or len(gate_rows)!=len(indices) or observed!=expected:
        raise ValueError('Gate/prediksi tidak cocok jumlah atau urutan ID.')
    rows=[]
    for i,p,r in zip(indices,prediction,gate_rows):
        a,b=float(r['gate_roberta']),float(r['gate_bge'])
        if not math.isfinite(a) or not math.isfinite(b) or not 0<=a<=1 or not 0<=b<=1 or abs(a+b-1)>1e-6:
            raise ValueError('Alpha gate harus finite,0–1,dan berjumlah1.')
        rows.append({'ID Jawaban':friendly_value('row_id',bundle.rows[i]['row_id']),
            'Nilai Guru':float(bundle.y[i]),'Prediksi S3':float(p),'Alpha RoBERTa':a,'Alpha BGE':b,
            'Pengali RoBERTa':2*a,'Pengali BGE':2*b})
    return pd.DataFrame(rows)


def gate_summary_records(oof, test):
    groups=[('Validasi fold',f,[r for r in oof if r['Fold']==f]) for f in sorted({r['Fold'] for r in oof})]
    groups += [('OOF development',None,oof),('Test',None,test)]
    rows=[]
    for stage,fold,part in groups:
        if not part:continue
        row={'Cakupan':stage,'Fold':fold,'Jumlah Jawaban':len(part)}
        for column in ['Alpha RoBERTa','Alpha BGE']:
            x=np.asarray([r[column] for r in part],dtype=np.float64)
            row.update({column+' Mean':float(x.mean()),column+' SD':float(x.std(ddof=0)),
                        column+' Min':float(x.min()),column+' Max':float(x.max())})
        rows.append(row)
    return rows


def gate_report_tables(source):
    oof=[];configs=[]
    for file in sorted(source.files):
        match=re.fullmatch(r'09_S3_fold_(\d+)\.xlsx',file)
        if not match and file!='10_S3_refit.xlsx':continue
        if match:
            for row in source.records(file,'Gate_validation'):oof.append({'Fold':int(match[1]),**row})
        for row in source.records(file,'Fusion_gate_config'):
            configs.append({'Tahap':'Validasi silang' if match else 'Latihan final','Fold':int(match[1]) if match else None,**row})
    test=list(source.records('11_S3_test.xlsx','Gate_test'))
    result={}
    for title,rows in [('Gate Validasi OOF',oof),('Gate Test',test),('Ringkasan Gate',gate_summary_records(oof,test)),('Konfigurasi Gate',configs)]:
        if rows:
            keys=list(rows[0]);result[title]=Table(keys,[[r[k] for k in keys] for r in rows])
    return result


def verify_gate_exports(oof_gates, test_gates, summaries, configs, oof, test, cfg):
    """Check serialized alpha, scaling, split, scores, summaries and gate recipe."""
    def check(rows, originals, with_fold):
        key=lambda r:(r['Fold'],r['ID Jawaban']) if with_fold else r['ID Jawaban']
        expected={key(r):r for r in originals}
        if len(rows)!=len(expected) or {key(r) for r in rows}!=set(expected):
            raise ValueError('ID/jumlah gate tidak cocok dengan prediksi OOF/test.')
        for r in rows:
            wanted=expected[key(r)]
            for col in ['Nilai Guru','Prediksi S3']:
                if not math.isclose(float(r[col]),float(wanted[col]),rel_tol=1e-6,abs_tol=1e-5):
                    raise ValueError('Nilai/prediksi pada gate berbeda dari pasangan evaluasi.')
            a,b=r['Alpha RoBERTa'],r['Alpha BGE']
            if not all(isinstance(x,(int,float)) and math.isfinite(x) and 0<=x<=1 for x in [a,b]) or abs(a+b-1)>1e-6:
                raise ValueError('Alpha gate tidak finite/di luar0–1/tidak berjumlah1.')
            if not math.isclose(r['Pengali RoBERTa'],2*a,rel_tol=1e-7,abs_tol=1e-7) or not math.isclose(r['Pengali BGE'],2*b,rel_tol=1e-7,abs_tol=1e-7):
                raise ValueError('Pengali gate bukan2×alpha.')
    check(oof_gates,oof,True);check(test_gates,test,False)
    expected=gate_summary_records(oof_gates,test_gates)
    if len(summaries)!=len(expected):raise ValueError('Ringkasan gate tidak lengkap.')
    for a,b in zip(summaries,expected):
        for key,value in b.items():
            if isinstance(value,(int,float)):
                if not isinstance(a.get(key),(int,float)) or not math.isclose(a[key],value,rel_tol=1e-7,abs_tol=1e-7):
                    raise ValueError('Statistik ringkasan gate tidak cocok.')
            elif a.get(key)!=value:raise ValueError('Identitas ringkasan gate tidak cocok.')
    if len(configs)!=cfg['folds']+1:raise ValueError('Konfigurasi gate checkpoint tidak lengkap.')
    for fold in [*range(1,cfg['folds']+1),None]:
        rows=[r for r in configs if r['Fold']==fold and r['Tahap']==('Latihan final' if fold is None else 'Validasi silang')]
        if len(rows)!=1:raise ValueError('Konfigurasi gate fold duplikat/hilang.')
        r=rows[0]
        if r['Mode']!=cfg['fusion_gate_mode'] or r['Dimensi Input']!=1024 or r['Jumlah Cabang']!=2 or r['Pengali Alpha']!=2 or r['Jumlah Parameter']!=2050:
            raise ValueError('Resep gate checkpoint berbeda dariE51.')
        if any(not isinstance(r[k],(int,float)) or not math.isfinite(r[k]) or r[k]<0 for k in ['Norma Bobot','Norma Bias']):
            raise ValueError('Norma parameter gate tidak finite.')
    return {'gate_oof_rows':len(oof_gates),'gate_test_rows':len(test_gates),'gate_config_rows':len(configs),
            'gate_values_and_summaries_verified':True}
