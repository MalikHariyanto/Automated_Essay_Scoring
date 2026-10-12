import torch
from torch import nn
from .windows import mean_by_essay

class AESModel(nn.Module):
    def __init__(self,scenario,roberta=None,dropout=.1,hidden=256,projection=256):
        super().__init__()
        if scenario not in ['S1','S2','S3']: raise ValueError('Skenario tidak sah')
        self.scenario=scenario;self.roberta=roberta
        if scenario!='S2' and roberta is None: raise ValueError('RoBERTa wajib untuk S1/S3')
        if roberta is not None: roberta.requires_grad_(True)
        self.dropout=nn.Dropout(dropout)
        if scenario=='S1':self.head=nn.Linear(768,4)
        elif scenario=='S2':self.head=nn.Sequential(nn.Linear(1024,hidden),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden,4))
        else:
            if projection!=256:raise ValueError('S3 proyeksi harus 256')
            self.projection=nn.Sequential(nn.Linear(1024,256),nn.LayerNorm(256))
            self.head=nn.Linear(1024,4)

    def forward(self, windows=None, chunk_to_essay=None, bge=None, n_essays=None, return_features=False):
        features={}
        if self.scenario!='S2':
            cls=self.roberta(**windows).last_hidden_state[:,0,:]
            essay=mean_by_essay(cls,chunk_to_essay,n_essays)
            features['roberta']=essay
        if self.scenario=='S1':pred=self.head(self.dropout(essay))
        elif self.scenario=='S2':
            if bge is None or bge.shape[1]!=1024:raise ValueError('BGE [B,1024] wajib')
            features['bge']=bge;pred=self.head(bge)
        else:
            if bge is None or bge.shape!=(n_essays,1024):raise ValueError('BGE [B,1024] wajib')
            features['bge']=bge;features['projection']=self.projection(bge)
            features['concat']=torch.cat([essay,features['projection']],dim=1)
            pred=self.head(self.dropout(features['concat']))
        if pred.ndim!=2 or pred.shape[1]!=4:raise ValueError('Output harus [B,4]')
        return (pred,features) if return_features else pred

def loss_mse(prediction,labels):
    if prediction.shape!=labels.shape or prediction.ndim!=2 or prediction.shape[1]!=4:
        raise ValueError('Loss menolak broadcasting: prediksi dan label harus [B,4]')
    if not torch.isfinite(prediction).all() or not torch.isfinite(labels).all():raise ValueError('Loss nonfinite')
    return ((prediction-labels)**2).mean()
