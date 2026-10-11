"""SPL pipeline: candidate review, editable crop, retargeting, physical tracking."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import argparse
import hashlib
import json
from spl_motion import ROOT,scan
from retarget_spl import retarget
from track_spl import track
from render_spl import render_results,review_page


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['all','scan','select','review','retarget','track','render','shoot','calibrate','prepare-rl'],default='all')
    p.add_argument('--selection',type=Path,default=ROOT/'runs/spl_tracking/selection.json')
    p.add_argument('--candidate',type=int,help='1-based rank in candidates.json (select stage)')
    p.add_argument('--start',type=int);p.add_argument('--release',type=int);p.add_argument('--end',type=int)
    p.add_argument('--speed',type=float,default=1.)
    a=p.parse_args();out=ROOT/'runs/spl_tracking';ref=ROOT/'data/motions/processed/free_throw_reference.npz'
    if a.stage=='prepare-rl':
        from prepare_residual import run
        run(render=True)
        return
    if a.stage=='calibrate':
        from calibrate_release import run
        run(render=True)
        return
    if a.stage=='shoot':
        from shoot_spl import shoot,OUTPUT
        from render_shooting import render
        shoot(ref,OUTPUT);render(OUTPUT)
        return
    if a.stage=='scan' or not (out/'candidates.json').exists():scan(out)
    if a.stage=='select':
        if a.candidate is not None:
            ranked=json.loads((out/'candidates.json').read_text())['ranked']
            if not 1<=a.candidate<=len(ranked):p.error('Candidate rank out of range')
            selection=ranked[a.candidate-1].copy()
        else:selection=json.loads(a.selection.read_text())
        for key in ['start','release','end']:
            if getattr(a,key) is not None:selection[key+'_frame']=getattr(a,key)
        if not selection['start_frame']<selection['release_frame']<selection['end_frame']:p.error('Require start < release < end')
        selection['source_sha256']=hashlib.sha256((ROOT/selection['path']).read_bytes()).hexdigest()
        selection['selection_method']='Explicit CLI selection'
        a.selection.parent.mkdir(parents=True,exist_ok=True)
        a.selection.write_text(json.dumps(selection,indent=2),encoding='utf-8')
    if a.stage in ['all','retarget']:retarget(a.selection,ref,a.speed)
    if a.stage in ['all','track']:track(ref,out)
    if a.stage in ['all','render']:render_results(out,ref)
    if a.stage in ['all','scan','select','review','render']:
        if a.selection.resolve()!=(out/'selection.json').resolve():
            (out/'selection.json').write_text(a.selection.read_text(encoding='utf-8'),encoding='utf-8')
        review_page(out)
    print('Results:',out)


if __name__=='__main__':main()
