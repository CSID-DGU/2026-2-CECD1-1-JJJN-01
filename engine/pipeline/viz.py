"""Side-by-side viewer for a run_video output: video frame | source person | targets.

The source panel also shows the MediaPipe landmarks (same frame as the IK, no smoothing):
black = the 14 points the IK fits, grey = other MediaPipe points; grey lines = MediaPipe's
own body connections. Point size follows MediaPipe visibility.

  python -m pipeline.viz runs/video/squat_side --targets Abe_m Mousey_m
Writes <run>/comparison.html (self-contained, plotly).
"""
import argparse
import base64
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.core.paths import enter_repo  # noqa: E402

COLORS = ['#2f6fdf', '#d9480f', '#2b8a3e', '#9c36b5', '#c2255c']
MP_EDGES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24),
            (23, 25), (25, 27), (27, 29), (29, 31), (27, 31), (24, 26), (26, 28), (28, 30),
            (30, 32), (28, 32), (0, 7), (0, 8)]


def video_frames(video, indices, width=360):
    with tempfile.TemporaryDirectory() as d:
        subprocess.run(['ffmpeg', '-loglevel', 'error', '-i', str(video), '-vf', f'scale={width}:-2',
                        '-q:v', '5', f'{d}/f%05d.jpg'], check=True)
        files = sorted(Path(d).glob('f*.jpg'))
        out = []
        for i in indices:
            data = files[min(i, len(files) - 1)].read_bytes()
            out.append('data:image/jpeg;base64,' + base64.b64encode(data).decode())
    return out


def skeleton(path):
    import outer_utils.Animation as Animation
    import outer_utils.BVH as BVH
    anim, names, _ = BVH.load(str(path))
    pos = Animation.positions_global(anim)
    shift = pos[0, 0] * np.array([1, 0, 1])                         # start above the origin
    return (pos - shift)[..., [0, 2, 1]], anim.parents, shift      # plot with Z up


def main(argv=None):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('--targets', nargs='+', default=['Abe_m', 'Mousey_m'])
    opt = ap.parse_args(argv)
    run = Path(opt.run).resolve()
    enter_repo()
    report = json.loads((run / 'report.json').read_text())
    panels = [('Source: IK skeleton (blue) + MediaPipe (black/grey)', run / 'source_person.bvh')] + \
        [(f'PAN → {t}', run / f'{t}.bvh') for t in opt.targets]
    loaded = [skeleton(p) for _, p in panels]
    skels = [(pos, par) for pos, par, _ in loaded]
    obs_file = np.load(run / 'observation.npz')
    if 'landmarks' not in obs_file:
        raise RuntimeError('observation.npz has no estimator points; rerun pipeline.run_video')
    mp_pts = (obs_file['landmarks'] - loaded[0][2])[..., [0, 2, 1]]   # same shift as the source
    mp_vis = obs_file['landmark_visibility']
    if 'landmark_edges' in obs_file:
        edges = [tuple(e) for e in obs_file['landmark_edges']]
        fitted = set(obs_file['landmark_fitted'].tolist())
        shown = obs_file['landmark_shown'].tolist()
    else:                                                          # older MediaPipe runs
        from pipeline.core.mediapipe_format import LANDMARKS, TO_MIXAMO
        edges, fitted = MP_EDGES, {LANDMARKS[lm] for lm, _ in TO_MIXAMO.values()}
        shown = sorted(fitted | {0, 7, 8, 29, 30})
    est = json.loads((run / 'report.json').read_text()).get('estimator', 'mediapipe')
    est_label = 'WHAM' if est == 'wham' else 'MediaPipe'
    panels[0] = (f'Source: IK skeleton (blue) + {est_label} (black/grey)', panels[0][1])
    T = min(len(s[0]) for s in skels)
    first = report['frames_used'][0]
    idx = [first + int(round(t * report['fps_in'] / 30.)) for t in range(T)]
    imgs = video_frames(report['video'], idx)

    allp = np.concatenate([s[0][:T].reshape(-1, 3) for s in skels])
    lo, hi = allp.min(0), allp.max(0)
    c, r = (lo + hi) / 2, float((hi - lo).max()) * 0.55
    rng = [[float(c[k] - r), float(c[k] + r)] for k in range(3)]
    rng[2] = [float(min(lo[2], 0) - 5), float(min(lo[2], 0) - 5 + 2 * r)]

    def lines(pos, parents, t, color):
        seg = []
        for j, p in enumerate(parents):
            if p >= 0:
                seg += [pos[t, p].tolist(), pos[t, j].tolist(), [None] * 3]
        x, y, z = zip(*seg)
        return go.Scatter3d(x=x, y=y, z=z, mode='lines+markers', line=dict(color=color, width=6),
                            marker=dict(size=2, color=color), showlegend=False)

    def mediapipe(t):
        p, v = mp_pts[min(t, len(mp_pts) - 1)], mp_vis[min(t, len(mp_vis) - 1)]
        seg = []
        for a, b in edges:
            seg += [p[a].tolist(), p[b].tolist(), [None] * 3]
        x, y, z = zip(*seg)
        body = shown
        return [go.Scatter3d(x=x, y=y, z=z, mode='lines', line=dict(color='#868e96', width=2, dash='dot'),
                             showlegend=False, hoverinfo='skip'),
                go.Scatter3d(x=p[body, 0], y=p[body, 1], z=p[body, 2], mode='markers', showlegend=False,
                             marker=dict(size=[3 + 4 * float(v[k]) for k in body],
                                         color=['#000000' if k in fitted else '#adb5bd' for k in body]),
                             text=[f'{est_label} {k}, visibility {v[k]:.2f}' for k in body], hoverinfo='text')]

    def floor():
        g = rng[0] + rng[0][::-1]
        return go.Mesh3d(x=[rng[0][0], rng[0][1], rng[0][1], rng[0][0]],
                         y=[rng[1][0], rng[1][0], rng[1][1], rng[1][1]], z=[0, 0, 0, 0],
                         i=[0, 0], j=[1, 2], k=[2, 3], color='#adb5bd', opacity=0.25, hoverinfo='skip')

    n = len(panels) + 1
    fig = make_subplots(rows=1, cols=n, specs=[[{'type': 'xy'}] + [{'type': 'scene'}] * (n - 1)],
                        subplot_titles=['Video'] + [p[0] for p in panels], horizontal_spacing=0.01)
    fig.add_trace(go.Image(source=imgs[0]), 1, 1)
    for k, (pos, par) in enumerate(skels):
        fig.add_trace(lines(pos, par, 0, COLORS[k % len(COLORS)]), 1, k + 2)
    for tr in mediapipe(0):
        fig.add_trace(tr, 1, 2)
    for k in range(len(skels)):
        fig.add_trace(floor(), 1, k + 2)
    dyn = list(range(1 + len(skels) + 2))
    fig.frames = [go.Frame(name=str(t), traces=dyn,
                           data=[go.Image(source=imgs[t])] +
                                [lines(pos, par, t, COLORS[k % len(COLORS)]) for k, (pos, par) in enumerate(skels)] +
                                mediapipe(t))
                  for t in range(T)]
    scene = dict(xaxis=dict(range=rng[0], visible=False), yaxis=dict(range=rng[1], visible=False),
                 zaxis=dict(range=rng[2], visible=False), aspectmode='cube',
                 camera=dict(eye=dict(x=1.6, y=-1.6, z=0.6)))
    fig.update_layout(**{f'scene{"" if k == 0 else k + 1}': scene for k in range(len(skels))})
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(
        height=520, margin=dict(l=10, r=10, t=60, b=10),
        title=f'{Path(report["video"]).name} [{est_label}]: {T} frames @30fps, person {report["person_height_cm"]:.0f} cm, '
              f'PAN input |z|>4: {100 * report["pan_input"]["frac_abs_z_gt"]:.2f}%',
        updatemenus=[dict(type='buttons', x=0, y=-0.05, xanchor='left', buttons=[
            dict(label='Play', method='animate', args=[None, dict(fromcurrent=True, frame=dict(
                duration=33, redraw=True), transition=dict(duration=0))]),
            dict(label='Pause', method='animate', args=[[None], dict(mode='immediate', frame=dict(
                duration=0, redraw=False))])])],
        sliders=[dict(x=0.12, len=0.88, y=-0.02, currentvalue=dict(prefix='Frame '), steps=[
            dict(label=str(t), method='animate', args=[[str(t)], dict(mode='immediate', frame=dict(
                duration=0, redraw=True), transition=dict(duration=0))]) for t in range(T)])])
    out = run / 'comparison.html'
    fig.write_html(str(out), include_plotlyjs=True, auto_play=False)
    print(f'Wrote {out} ({out.stat().st_size / 1e6:.1f} MB)')


if __name__ == '__main__':
    main()
