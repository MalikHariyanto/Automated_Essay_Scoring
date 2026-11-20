import argparse
import json
from pathlib import Path
import sys
from .io import load_config,load_data,load_rubric,dump,targets
from .audit import EligibilityError,validate_inputs_for_training
from .inspection import inspect_data,inspect_models
from .inference import predict
from .metrics import evaluate


def main(argv=None):
    p=argparse.ArgumentParser(description='AES empat trait: inspect tidak memerlukan skor guru')
    p.add_argument('--config',default='configs/default.json');p.add_argument('--csv');p.add_argument('--rubric');p.add_argument('--output')
    sub=p.add_subparsers(dest='command',required=True)
    i=sub.add_parser('inspect');i.add_argument('--models',action='store_true',help='Unduh checkpoint asli dan forward UNTRAINED')
    t=sub.add_parser('train');t.add_argument('--mode',choices=['ai_learning','learning','proposal'],default='learning');t.add_argument('--scenarios',nargs='+',choices=['S1','S2','S3'],default=['S1','S2','S3'])
    pred=sub.add_parser('predict');pred.add_argument('--checkpoint',required=True)
    g=pred.add_mutually_exclusive_group(required=True);g.add_argument('--text');g.add_argument('--file')
    e=sub.add_parser('evaluate');e.add_argument('--checkpoint',required=True);e.add_argument('--partition',choices=['validation','test'],required=True)
    e.add_argument('--manifest',required=True);e.add_argument('--repeat',type=int,default=1);e.add_argument('--fold',type=int,default=1)
    args=p.parse_args(argv);cfg=load_config(args.config)
    for k in ['csv','rubric','output']:
        if getattr(args,k):cfg[k]=str(Path(getattr(args,k)).resolve())
    if args.command in ['inspect','train','evaluate']:
        missing=[cfg[k] for k in ['csv','rubric'] if not Path(cfg[k]).is_file()]
        if missing:
            p.error('Input penelitian lokal belum tersedia: '+', '.join(missing)+
                    '. Data tidak dibagikan di repo publik; atur --csv/--rubric atau configs/default.json.')
    if args.rubric and not Path(cfg['rubric']).is_file():
        p.error('--rubric menunjuk file yang tidak tersedia')
    try:
        if args.command=='inspect':
            df,r,a=inspect_data(cfg);print(json.dumps(a,ensure_ascii=False,indent=2))
            if args.models:print(json.dumps(inspect_models(cfg,df),ensure_ascii=False,indent=2))
        elif args.command=='train':
            if args.mode=='ai_learning':
                from .ai_learning import run_ai_learning
                cfg['ai_learning']['enabled']=True
                run_ai_learning(cfg)
            else:
                from .experiment import run
                run(cfg,args.mode,args.scenarios)
        elif args.command=='predict':
            texts=[args.text] if args.text is not None else [Path(args.file).read_text(encoding='utf-8')]
            expected=cfg['rubric'] if Path(cfg['rubric']).is_file() else None
            print(json.dumps(predict(args.checkpoint,texts,expected_rubric=expected),ensure_ascii=False,indent=2))
        else:
            from .checkpoint import load_checkpoint
            from .io import sha256
            df=load_data(cfg['csv']);r=load_rubric(cfg['rubric']);review=json.loads(Path(cfg['split']['review_file']).read_text())
            validate_inputs_for_training(df,r,review,sha256(cfg['csv']))
            manifest=json.loads(Path(args.manifest).read_text());model,payload=load_checkpoint(args.checkpoint,'cpu',cfg['rubric'],cfg['preprocessing']);del model
            if manifest['dataset_sha256']!=sha256(cfg['csv']) or manifest['rubric_sha256']!=sha256(cfg['rubric']):raise ValueError('Manifest tidak cocok dataset/rubrik')
            if args.partition=='test':
                if payload['training']['validation_essay_count']!=0:raise ValueError('Test hanya untuk checkpoint final refit tanpa validation/tuning')
                lock=Path(args.checkpoint).parent.parent/'locked_config.json'
                if not lock.exists():raise ValueError('Konfigurasi belum dikunci')
                ids=manifest['test']
            else:
                fold=next(f for f in manifest['folds'] if f['repeat']==args.repeat and f['fold']==args.fold);ids=fold['validation']
            subset=df.set_index('essay_id').loc[ids].reset_index();outputs=predict(args.checkpoint,subset.text_raw.tolist(),cfg['rubric'])
            import numpy as np
            predicted=np.array([[row['continuous'][t['name']] for t in r['traits']] for row in outputs])
            metrics=evaluate(targets(subset,r),predicted,r);dump(Path(cfg['output'])/'evaluation.json',metrics);print(json.dumps(metrics,indent=2))
    except EligibilityError as error:
        print('SUPERVISED DIHENTIKAN; INSPEKSI TETAP TERSEDIA:\n'+str(error),file=sys.stderr);return 2
    return 0

if __name__=='__main__':raise SystemExit(main())
