"""Viewer for a coach_to_user output: coach video | coach | coach->user (PAN, copy) | user | user video.

  python -m pipeline.coach_user_viz runs/coach_user/<name>
Writes <name>/comparison.html (self-contained, plotly). Coach and user clips may differ in
length; each panel holds its last frame once its clip ends (no time alignment here).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.core.paths import enter_repo  # noqa: E402

COLORS = {'coach': '#2f6fdf', 'pan': '#d9480f', 'copy': '#868e96', 'user': '#2b8a3e'}


def _frames(run, n):
    from pipeline.viz import video_frames
    rep = json.loads((run / 'report.json').read_text())
    first = rep['frames_used'][0]
    return video_frames(rep['video'], [first + int(round(t * rep['fps_in'] / 30.)) for t in range(n)], width=300)


def render(out, coach_run, user_run=None):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from pipeline.viz import skeleton
    enter_repo()
    out, coach_run = Path(out), Path(coach_run)
    rep = json.loads((out / 'report.json').read_text())
    h = rep['height_cm']
    panels = [(f'Coach ({h["coach"]:.0f} cm)', out / 'coach.bvh', 'coach'),
              (f'Coach → user, PAN ({h["user"]:.0f} cm)', out / 'coach_on_user_pan.bvh', 'pan'),
              ('Coach → user, rotation copy', out / 'coach_on_user_copy.bvh', 'copy')]
    if (out / 'user.bvh').exists():
        panels.append(('User (own motion)', out / 'user.bvh', 'user'))
    skels = [skeleton(p)[:2] for _, p, _ in panels]
    T = max(len(s[0]) for s in skels)
    videos = [('Coach video', _frames(coach_run, len(skels[0][0])))]
    if user_run:
        videos.append(('User video', _frames(Path(user_run), len(skels[-1][0]))))

    import numpy as np
    allp = np.concatenate([s[0].reshape(-1, 3) for s in skels])
    lo, hi = allp.min(0), allp.max(0)
    c, r = (lo + hi) / 2, float((hi - lo).max()) * 0.55
    rng = [[float(c[k] - r), float(c[k] + r)] for k in range(3)]
    rng[2] = [float(min(lo[2], 0) - 5), float(min(lo[2], 0) - 5 + 2 * r)]

    def lines(pos, parents, t, color):
        t = min(t, len(pos) - 1)
        seg = []
        for j, p in enumerate(parents):
            if p >= 0:
                seg += [pos[t, p].tolist(), pos[t, j].tolist(), [None] * 3]
        x, y, z = zip(*seg)
        return go.Scatter3d(x=x, y=y, z=z, mode='lines+markers', line=dict(color=color, width=6),
                            marker=dict(size=2, color=color), showlegend=False)

    def floor():
        return go.Mesh3d(x=[rng[0][0], rng[0][1], rng[0][1], rng[0][0]],
                         y=[rng[1][0], rng[1][0], rng[1][1], rng[1][1]], z=[0, 0, 0, 0],
                         i=[0, 0], j=[1, 2], k=[2, 3], color='#adb5bd', opacity=0.25, hoverinfo='skip')

    # columns: coach video, skeleton panels, (user video)
    specs = [{'type': 'xy'}] + [{'type': 'scene'}] * len(panels) + ([{'type': 'xy'}] if user_run else [])
    titles = [videos[0][0]] + [p[0] for p in panels] + ([videos[1][0]] if user_run else [])
    fig = make_subplots(rows=1, cols=len(specs), specs=[specs], subplot_titles=titles, horizontal_spacing=0.01)

    def img(v, t):
        return go.Image(source=v[min(t, len(v) - 1)])

    def frame_data(t):
        data = [img(videos[0][1], t)]
        data += [lines(pos, par, t, COLORS[panels[k][2]]) for k, (pos, par) in enumerate(skels)]
        if user_run:
            data.append(img(videos[1][1], t))
        return data

    first = frame_data(0)
    fig.add_trace(first[0], 1, 1)
    for k in range(len(skels)):
        fig.add_trace(first[1 + k], 1, k + 2)
    if user_run:
        fig.add_trace(first[-1], 1, len(specs))
    for k in range(len(skels)):
        fig.add_trace(floor(), 1, k + 2)
    dyn = list(range(len(first)))
    fig.frames = [go.Frame(name=str(t), traces=dyn, data=frame_data(t)) for t in range(T)]
    scene = dict(xaxis=dict(range=rng[0], visible=False), yaxis=dict(range=rng[1], visible=False),
                 zaxis=dict(range=rng[2], visible=False), aspectmode='cube',
                 camera=dict(eye=dict(x=1.6, y=-1.6, z=0.6)))
    fig.update_layout(**{f'scene{"" if k == 0 else k + 1}': scene for k in range(len(skels))})
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    feet = rep['feet']
    fig.update_layout(
        height=480, margin=dict(l=10, r=10, t=70, b=10),
        title=(f'{out.name}: leg ratio coach {rep["leg_ratio_to_bigvegas"]["coach"]:.2f} → user '
               f'{rep["leg_ratio_to_bigvegas"]["user"]:.2f} | PAN vs copy {rep["pan_vs_copy_pos_cm"]:.1f} cm | '
               f'toe min y PAN {feet["pan"]["toe_min_y"]:.1f} / copy {feet["copy"]["toe_min_y"]:.1f} cm'),
        updatemenus=[dict(type='buttons', x=0, y=-0.05, xanchor='left', buttons=[
            dict(label='Play', method='animate', args=[None, dict(fromcurrent=True, frame=dict(
                duration=33, redraw=True), transition=dict(duration=0))]),
            dict(label='Pause', method='animate', args=[[None], dict(mode='immediate', frame=dict(
                duration=0, redraw=False))])])],
        sliders=[dict(x=0.12, len=0.88, y=-0.02, currentvalue=dict(prefix='Frame '), steps=[
            dict(label=str(t), method='animate', args=[[str(t)], dict(mode='immediate', frame=dict(
                duration=0, redraw=True), transition=dict(duration=0))]) for t in range(T)])])
    path = out / 'comparison.html'
    fig.write_html(str(path), include_plotlyjs=True, auto_play=False)
    print(f'Wrote {path} ({path.stat().st_size / 1e6:.1f} MB)')


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    opt = ap.parse_args(argv)
    out = Path(opt.out).resolve()
    rep = json.loads((out / 'report.json').read_text())
    render(out, rep['coach'], rep['user'])


if __name__ == '__main__':
    main()
