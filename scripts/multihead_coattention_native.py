"""Visible E50 module: bidirectional multi-head cross-attention with round trip."""


class MultiHeadCoAttention(nn.Module):
    """Share Q/K/V/O between directions; softmax matrices are computed independently."""
    def __init__(self, dim=128, heads=4):
        super().__init__()
        if type(heads) is not int or heads!=4 or dim!=128 or dim%heads:
            raise ValueError('E50 mengunci dim128, empat head, masing-masing32dimensi.')
        self.dim,self.heads,self.head_dim=dim,heads,dim//heads
        self.q=nn.Linear(dim,dim,bias=False)
        self.k=nn.Linear(dim,dim,bias=False)
        self.v=nn.Linear(dim,dim,bias=False)
        self.out=nn.Linear(dim,dim,bias=False)

    def forward(self, response, reference, valid, ref_valid):
        if response.ndim!=3 or reference.ndim!=3 or response.shape[0]!=reference.shape[0] or response.shape[-1]!=self.dim or reference.shape[-1]!=self.dim:
            raise ValueError('Multi-head memerlukan response[B,T,128] dan reference[B,R,128].')
        if valid.shape!=response.shape[:2] or ref_valid.shape!=reference.shape[:2] or valid.dtype!=torch.bool or ref_valid.dtype!=torch.bool:
            raise ValueError('Mask multi-head harus bool[B,T] dan bool[B,R].')
        if not bool(valid.any(1).all()) or not bool(ref_valid.any(1).all()):
            raise ValueError('Setiap jawaban/referensi harus mempunyai token valid.')
        def split(x):return x.reshape(x.shape[0],x.shape[1],self.heads,self.head_dim).transpose(1,2)
        def merge(x):return x.transpose(1,2).contiguous().reshape(x.shape[0],x.shape[2],self.dim)
        # FP32 menjaga softmax/produk attention stabil saat encoder memakai AMP.
        with torch.autocast(device_type=response.device.type,enabled=False):
            response,reference=response.float(),reference.float()
            rq,rk,rv=[split(layer(response)) for layer in (self.q,self.k,self.v)]
            fq,fk,fv=[split(layer(reference)) for layer in (self.q,self.k,self.v)]
            scores_a=torch.matmul(rq,fk.transpose(-1,-2))/math.sqrt(self.head_dim)
            scores_r=torch.matmul(fq,rk.transpose(-1,-2))/math.sqrt(self.head_dim)
            a_to_r=torch.softmax(scores_a.masked_fill(~ref_valid[:,None,None,:],-torch.inf),dim=-1)
            a_to_r=a_to_r.masked_fill(~valid[:,None,:,None],0.)
            r_to_a=torch.softmax(scores_r.masked_fill(~valid[:,None,None,:],-torch.inf),dim=-1)
            r_to_a=r_to_a.masked_fill(~ref_valid[:,None,:,None],0.)
            aligned=self.out(merge(torch.matmul(a_to_r,fv)))
            round_trip=self.out(merge(torch.matmul(a_to_r,torch.matmul(r_to_a,rv))))
        return aligned,round_trip,a_to_r,r_to_a

    def report(self):
        rows=[]
        for h in range(self.heads):
            a,b=h*self.head_dim,(h+1)*self.head_dim
            rows.append({'Head':h+1,'Jumlah Head':self.heads,'Dimensi Total':self.dim,
                'Dimensi per Head':self.head_dim,'Parameter Proyeksi Total':sum(p.numel() for p in self.parameters()),
                'Norma Q':float(self.q.weight[a:b].detach().float().norm().cpu()),
                'Norma K':float(self.k.weight[a:b].detach().float().norm().cpu()),
                'Norma V':float(self.v.weight[a:b].detach().float().norm().cpu()),
                'Norma Output per Head':float(self.out.weight[:,a:b].detach().float().norm().cpu())})
        return rows


def multihead_report_tables(source):
    rows=[]
    for file in sorted(source.files):
        match=re.fullmatch(r'09_S3_fold_(\d+)\.xlsx',file)
        if not match and file!='10_S3_refit.xlsx':continue
        for r in source.records(file,'Proyeksi_multihead'):
            rows.append({'Tahap':'Validasi silang' if match else 'Latihan final','Fold':int(match[1]) if match else None,**r})
    if not rows:return {}
    keys=list(rows[0])
    return {'Proyeksi Multihead':Table(keys,[[r[k] for k in keys] for r in rows])}
