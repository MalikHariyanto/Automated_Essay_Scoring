from .checkpoint import load_checkpoint
from .cleaning import clean
from .encoders import BGEEncoder
from .trainer import BatchEngine,device_name
from .metrics import ordinal_prediction
import torch


def predict(checkpoint,texts,expected_rubric=None):
    device=device_name();model,p=load_checkpoint(checkpoint,device,expected_rubric)
    if not p['training'].get('trained'):raise ValueError('Checkpoint UNTRAINED tidak boleh digunakan untuk penilaian')
    cfg=p['config'];texts=[clean(t,**cfg['preprocessing']) for t in texts]
    if any(not t.strip() for t in texts):raise ValueError('Teks kosong')
    bge=None
    if model.scenario!='S1':
        encoder=BGEEncoder(p['resolved_models']['bge'],device);bge=torch.from_numpy(encoder.encode(texts));del encoder
    engine=BatchEngine(model,texts,bge,p['resolved_models'],cfg,device);values=engine.predict(list(range(len(texts))))
    return [{'continuous':{t['name']:float(row[j]) for j,t in enumerate(p['rubric']['traits'])},
             'categories':{t['name']:int(ordinal_prediction([row[j]],t)[0]) for j,t in enumerate(p['rubric']['traits'])}}
             for row in values]
