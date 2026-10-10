"""Read-only SPL skeleton preprocessing; points are observations, not rotations."""
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
from .smplh_motion import SMPL_TO_WORLD

BODY_MAP = {'MID_HIP':'Pelvis', 'NECK':'Neck', **{
    f'{side}_{part.upper()}':f'{short}_{part}'
    for side,short in [('LEFT','L'),('RIGHT','R')]
    for part in ('Hip','Knee','Ankle','Shoulder','Elbow','Wrist')}}
FINGER_MAP = {}
for side,short in [('LEFT','L'),('RIGHT','R')]:
    for finger,part in [('FIRST','Thumb'),('SECOND','Index'),('THIRD','Middle'),('FOURTH','Ring'),('FIFTH','Pinky')]:
        segments = ('CMC','MCP','IP') if part=='Thumb' else ('MCP','PIP','DIP')
        for i,segment in enumerate(segments,1):
            FINGER_MAP[f'{side}_{finger}_FINGER_{segment}'] = f'{short}_{part}{i}'

def mapping_table(names):
    """Account for all source keys, including landmarks without joint-center matches."""
    return [{'spl':n,'smplh':BODY_MAP.get(n,FINGER_MAP.get(n,'')),
             'kind':'body_joint_candidate' if n in BODY_MAP else
                    'finger_joint_candidate' if n in FINGER_MAP else 'unfitted_landmark'} for n in names]

def fill_short_gaps(points, max_gap):
    result=points.copy();filled=np.zeros(points.shape[:2],bool)
    for j in range(points.shape[1]):
        ok=np.isfinite(result[:,j]).all(axis=1);i=0
        while i<len(ok):
            if ok[i]:i+=1;continue
            end=i
            while end<len(ok) and not ok[end]:end+=1
            if i>0 and end<len(ok) and end-i<=max_gap:
                for k in range(i,end):
                    a=(k-i+1)/(end-i+1);result[k,j]=(1-a)*result[i-1,j]+a*result[end,j];filled[k,j]=True
            i=end
    return result,filled

@dataclass
class SPLTrial:
    names: list
    original_m: np.ndarray
    normalized_world_m: np.ndarray
    smpl_m: np.ndarray
    valid_original: np.ndarray
    outliers: np.ndarray
    filled: np.ndarray
    weights: np.ndarray
    fps: float
    timestamps_ms: np.ndarray
    origin_m: np.ndarray
    world_rotation: np.ndarray
    metadata: dict

    def observations(self, include_fingers=False):
        mapping=dict(BODY_MAP)
        if include_fingers:mapping.update(FINGER_MAP)
        keys=[k for k in mapping if k in self.names]
        ids=[self.names.index(k) for k in keys]
        return keys,[mapping[k] for k in keys],self.smpl_m[:,ids],self.weights[:,ids]

    def restore_world(self, normalized):
        return np.asarray(normalized)@self.world_rotation+self.origin_m


def load_trial(path, max_gap=3, speed_limit=10.):
    raw=json.loads(Path(path).read_text());frames=raw['tracking'];fps=float(raw['sampling_rate'])
    if fps<=0 or len(frames)<2:raise ValueError('Invalid FPS or empty trial')
    ids=np.array([f['frame'] for f in frames]);times=np.array([f['time'] for f in frames])
    if not np.array_equal(ids,np.arange(len(frames))) or not np.all(np.diff(times)>0):
        raise ValueError('Expected consecutive frames and increasing timestamps')
    if np.max(abs(times-ids*1000/fps))>1.01:raise ValueError('Unexpected timestamp convention')
    names=sorted(set().union(*(f['data']['player'] for f in frames)))
    original=np.array([[f['data']['player'].get(n,[None]*3) for n in names] for f in frames],float)*.3048
    valid=np.isfinite(original).all(axis=-1);bad=np.zeros(valid.shape,bool)
    # Conservative speed outliers: mark both ends rather than privileging one sample.
    speed=np.linalg.norm(np.diff(original,axis=0),axis=-1)*fps
    jumps=np.isfinite(speed)&(speed>speed_limit);bad[:-1]|=jumps;bad[1:]|=jumps
    # Robust limb-length screen, with a 3 cm noise floor to avoid rejecting real soft-tissue motion.
    for side in ('LEFT','RIGHT'):
        for a,b in [('SHOULDER','ELBOW'),('ELBOW','WRIST'),('HIP','KNEE'),('KNEE','ANKLE')]:
            ia,ib=names.index(f'{side}_{a}'),names.index(f'{side}_{b}')
            lengths=np.linalg.norm(original[:,ia]-original[:,ib],axis=-1)
            median=np.nanmedian(lengths);mad=np.nanmedian(abs(lengths-median))
            flag=np.isfinite(lengths)&(abs(lengths-median)>max(.03,6*1.4826*mad))
            bad[:,ia]|=flag;bad[:,ib]|=flag
    screened=original.copy();screened[bad]=np.nan
    cleaned,filled=fill_short_gaps(screened,max_gap)
    # Root/neck fallback is a derived landmark, with half weight rather than claimed ground truth.
    derived=np.zeros(valid.shape,bool)
    for name,left,right in [('MID_HIP','LEFT_HIP','RIGHT_HIP'),('NECK','LEFT_SHOULDER','RIGHT_SHOULDER')]:
        if name not in names:continue
        j=names.index(name);mid=(cleaned[:,names.index(left)]+cleaned[:,names.index(right)])/2
        use=~np.isfinite(cleaned[:,j]).all(axis=1)&np.isfinite(mid).all(axis=1)
        cleaned[use,j]=mid[use];derived[use,j]=True
    pelvis=cleaned[:,names.index('MID_HIP')];first=np.flatnonzero(np.isfinite(pelvis).all(axis=1))
    if not len(first):raise ValueError('No usable pelvis')
    origin=pelvis[first[0]].copy();origin[2]=0  # Preserve court vertical coordinate.
    lateral=cleaned[:min(30,len(frames)),names.index('LEFT_HIP')]-cleaned[:min(30,len(frames)),names.index('RIGHT_HIP')]
    lateral=np.nanmedian(lateral,axis=0);lateral[2]=0
    if not np.isfinite(lateral).all() or np.linalg.norm(lateral)<1e-5:raise ValueError('Degenerate hip heading')
    forward=np.cross(lateral,[0,0,1]);yaw=np.arctan2(forward[1],forward[0]);c,s=np.cos(yaw),np.sin(yaw)
    rotation=np.array([[c,s,0],[-s,c,0],[0,0,1]])
    world=(cleaned-origin)@rotation.T
    smpl=SMPL_TO_WORLD.inv().apply(world.reshape(-1,3)).reshape(world.shape)
    weights=np.isfinite(smpl).all(axis=-1).astype(float);weights[filled]=.25;weights[derived]=.5
    return SPLTrial(names,original,world,smpl,valid,bad,filled,weights,fps,times,origin,rotation,
                    {k:v for k,v in raw.items() if k!='tracking'})
