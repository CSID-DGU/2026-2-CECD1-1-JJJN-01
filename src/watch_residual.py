"""Export newly selected best checkpoints while independent PPO training runs."""
import argparse,json,subprocess,sys,time
from pathlib import Path
from humanoid_model import ROOT


def watch(output,once=False):
    output=Path(output).resolve();last=None
    while True:
        progress=json.loads((output/'progress.json').read_text(encoding='utf-8')) if (output/'progress.json').exists() else {}
        best_file=output/'best.json'
        if best_file.exists():
            best=json.loads(best_file.read_text(encoding='utf-8'));selected=best['checkpoint']
            if selected!=last:
                checkpoint=output/selected;name=checkpoint.name;destination=output/'evaluations'/name
                if not (destination/'evaluation.json').exists():
                    result=subprocess.run([sys.executable,str(ROOT/'src/evaluate_residual.py'),'--checkpoint',str(checkpoint),'--output',str(destination),'--render'])
                    if result.returncode==0:
                        (output/'best_evaluation.json').write_text(json.dumps({'checkpoint':selected,'review':str((destination/'review.html').relative_to(output))},indent=2),encoding='utf-8')
                last=selected
        if once or progress.get('status') in ['finished','stopped','failed','interrupted']:break
        time.sleep(60)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'runs/residual_ppo');p.add_argument('--once',action='store_true')
    a=p.parse_args();watch(a.output,a.once)
