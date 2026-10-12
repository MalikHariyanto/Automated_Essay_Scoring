"""Pintasan script menerima opsi path sebelum atau sesudah opsi perintah."""
import sys
from .cli import main

def run_command(command,argv=None):
    argv=list(sys.argv[1:] if argv is None else argv);prefix=[];rest=[];i=0
    options={'--config','--csv','--rubric','--output'}
    while i<len(argv):
        arg=argv[i]
        if arg in options:
            if i+1>=len(argv):raise SystemExit(f'{arg} perlu nilai')
            prefix.extend(argv[i:i+2]);i+=2
        elif arg.split('=',1)[0] in options:
            prefix.append(arg);i+=1
        else:rest.append(arg);i+=1
    return main(prefix+[command]+rest)
