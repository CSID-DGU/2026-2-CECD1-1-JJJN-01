"""SPL ingestion, candidate ranking and explicit frame selection. Positions in metres."""
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/motions/spl_open_data/basketball/freethrow/data/2025-12-18'
KEYS = ['LEFT_HIP','RIGHT_HIP','LEFT_KNEE','RIGHT_KNEE','LEFT_ANKLE','RIGHT_ANKLE',
        'LEFT_SHOULDER','RIGHT_SHOULDER','LEFT_ELBOW','RIGHT_ELBOW','LEFT_WRIST','RIGHT_WRIST',
        'LEFT_HEEL','RIGHT_HEEL','LEFT_BIG_TOE','RIGHT_BIG_TOE','NOSE']


def read_trial(path):
    raw = json.loads(Path(path).read_text(encoding='utf-8'))
    frames = raw['tracking']
    fps = float(raw['sampling_rate'])
    if not np.isfinite(fps) or fps <= 0: raise ValueError('Invalid sampling rate')
    ids = np.array([f['frame'] for f in frames], dtype=int)
    if len(ids) < 30 or np.any(np.diff(ids) != 1):
        raise ValueError('Require at least 30 contiguous ordered frames')
    # Frame IDs, not millisecond timestamps rounded to integers, determine sample time.
    times = (ids - ids[0]) / fps
    keys = sorted(set().union(*(f['data']['player'].keys() for f in frames)))
    tracks = {}
    missing = {}
    for key in keys + ['BALL']:
        positions = np.array([f['data'].get('ball') if key == 'BALL'
                              else f['data']['player'].get(key, [None]*3) for f in frames], dtype=float) * .3048
        if positions.shape != (len(ids), 3):
            raise ValueError(f'Bad coordinate shape: {key}')
        missing[key] = float(np.mean(~np.isfinite(positions)))
        for axis in range(3):
            valid = np.isfinite(positions[:, axis])
            if valid.sum() < 2:
                positions[:, axis] = np.nan
            else:
                positions[:, axis] = np.interp(times, times[valid], positions[valid, axis])
        tracks[key] = positions
    if not all(k in tracks and np.isfinite(tracks[k]).all() for k in KEYS):
        raise ValueError('Missing essential body landmarks')
    return raw, fps, ids, times, tracks, missing


def smooth(points, fps):
    window = max(5, int(round(fps * .12)) | 1)
    return savgol_filter(median_filter(points, size=(3, 1)), window, 2, axis=0)


def propose(path):
    raw, fps, ids, times, tracks, missing = read_trial(path)
    p = {k:smooth(v, fps) for k,v in tracks.items()}
    ball = p['BALL']
    distance = np.minimum(np.linalg.norm(ball-p['RIGHT_WRIST'], axis=1),
                          np.linalg.norm(ball-p['LEFT_WRIST'], axis=1))
    height = np.maximum(p['LEFT_SHOULDER'][:,2],p['RIGHT_SHOULDER'][:,2])
    velocity = np.gradient(ball, 1/fps, axis=0)
    crossing = np.flatnonzero((distance[1:]>.33)&(distance[:-1]<=.33)&
                              (ball[1:,2]>height[1:]+.2)&(velocity[1:,2]>1.)) + 1
    crossing = [i for i in crossing if i > int(.8*fps) and i+int(.55*fps) < len(ids)]
    if not crossing:
        raise ValueError('No reliable release candidate')
    release = int(crossing[0])
    # The separation threshold occurs shortly after physical release. Backtrack to .20 m.
    while release > 1 and distance[release-1] > .20:
        release -= 1
    start = max(0, release-int(1.05*fps))
    end = min(len(ids)-1, release+int(.65*fps))
    sl = slice(start,end+1)
    foot_motion = sum(np.max(np.linalg.norm(p[k][sl]-p[k][start], axis=1)) for k in ['LEFT_ANKLE','RIGHT_ANKLE'])
    jitter = np.mean([np.mean(np.linalg.norm(np.diff(tracks[k][sl],n=2,axis=0),axis=1)) for k in KEYS])
    score = float(foot_motion+10*jitter+sum(missing[k] for k in KEYS)+(0 if raw['result']=='made' else 1))
    return {'path':str(Path(path).relative_to(ROOT)), 'participant':raw['participant_id'],
            'trial':raw['trial_id'], 'result':raw['result'], 'fps':fps,
            'start_frame':int(ids[start]), 'release_frame':int(ids[release]), 'end_frame':int(ids[end]),
            'foot_motion_m':float(foot_motion), 'jitter_m':float(jitter), 'score':score,
            'release_note':'Heuristic hand-ball separation estimate; not ground-truth annotation.'}


def scan(output, limit=3):
    output.mkdir(parents=True,exist_ok=True)
    candidates, rejected = [], []
    for path in sorted(DATA.rglob('*.json')):
        try:
            candidates.append(propose(path))
        except (ValueError,KeyError,TypeError) as error:
            rejected.append({'file':str(path.relative_to(ROOT)), 'reason':str(error)})
    candidates.sort(key=lambda x:x['score'])
    if not candidates:
        raise RuntimeError('No usable SPL trials')
    (output/'candidates.json').write_text(json.dumps({'ranked':candidates,'rejected':rejected},indent=2),encoding='utf-8')
    selected = candidates[0].copy()
    selected['source_sha256'] = hashlib.sha256((ROOT/selected['path']).read_bytes()).hexdigest()
    selected['selection_method'] = 'Automatic low foot-motion / low jitter ranking; crop editable using the local review page or CLI.'
    selected['frame_bounds'] = 'Inclusive source frame IDs'
    (output/'selection.json').write_text(json.dumps(selected,indent=2),encoding='utf-8')
    return candidates[:limit]
