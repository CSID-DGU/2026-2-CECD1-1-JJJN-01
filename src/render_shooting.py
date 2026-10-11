"""Render the recorded physical shot, trajectory, and a local Korean review page."""
import json
import os
import html
from pathlib import Path
import numpy as np
import mujoco
import imageio.v2 as imageio
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from shoot_spl import ROOT, OUTPUT, HOOP, RADIUS


def render(output=OUTPUT):
    output=Path(output);report=json.loads((output/'shooting_report.json').read_text(encoding='utf-8'))
    run=np.load(output/'shooting.npz');m=mujoco.MjModel.from_xml_string((output/'scene.xml').read_text(encoding='utf-8'))
    d=mujoco.MjData(m);option=mujoco.MjvOption();option.sitegroup[:]=0
    calibrated='release_diagnostics' in report
    contact=report.get('release_mode')=='natural_contact_separation'
    label='Contact release' if contact else 'Wrist correction' if calibrated else 'Baseline'
    release=report['release']['time_s'];settle=report['settle_s'];end=settle+report['motion_duration_s']
    cameras=[]
    for look,distance,azimuth,elevation in [([2.1,0,1.6],7.7,90,-10),([.1,0,1.2],3.25,125,-12)]:
        camera=mujoco.MjvCamera();camera.lookat[:]=look;camera.distance=distance
        camera.azimuth=azimuth;camera.elevation=elevation;cameras.append(camera)
    frames=[];snapshots={};phases=[('prepare',.45),('lift',release-.22),('release',release),('followthrough',end+.35)]
    with mujoco.Renderer(m,height=640,width=960) as wide, mujoco.Renderer(m,height=640,width=480) as close, \
         imageio.get_writer(str(output/'shooting.mp4'),fps=30,codec='libx264',quality=8,macro_block_size=16) as writer:
        for t in np.arange(0,run['times'][-1],1/30):
            k=int(np.argmin(abs(run['times']-t)));d.qpos[:]=run['qpos'][k];d.qvel[:]=run['qvel'][k]
            d.eq_active[:]=run['grasp_active'][k];mujoco.mj_forward(m,d)
            canvas=Image.new('RGB',(1440,640))
            for j,(renderer,camera) in enumerate(zip([wide,close],cameras)):
                renderer.update_scene(d,camera,scene_option=option)
                if j==0:
                    pts=run['ball_positions'][(run['ball_times']>=release)&(run['ball_times']<=t)][::25]
                    for point in pts:
                        scene=renderer.scene
                        if scene.ngeom>=scene.maxgeom:break
                        mujoco.mjv_initGeom(scene.geoms[scene.ngeom],mujoco.mjtGeom.mjGEOM_SPHERE,
                                           np.array([.012,.012,.012]),point,np.eye(3).ravel(),np.array([1.,.65,.05,1.],dtype=np.float32))
                        scene.ngeom+=1
                tile=Image.fromarray(renderer.render());draw=ImageDraw.Draw(tile)
                draw.rectangle((0,0,tile.width,68),fill=(18,27,42))
                title='Physical ball trajectory / fixed hoop' if j==0 else 'Loaded motion / balance + follow-through'
                draw.text((16,12),title,fill='white')
                phase='PREPARE / GRASP' if t<settle else 'LIFT / GRASP' if t<release-.0008 else 'RELEASE / FREE FLIGHT' if t<end else 'FOLLOW-THROUGH / HOLD'
                draw.text((16,36),f'{phase}   t={t:.2f}s',fill=(255,180,65))
                draw.rectangle((0,598,tile.width,640),fill=(18,27,42))
                text=f'{label}: MISS | no launch impulse | ball mass 0.62 kg' if not report['scored'] else f'{label}: SCORED | no launch impulse'
                if j==1:text='Preparation weld / physical finger contact / natural release' if contact else 'Grasp: temporary hand-ball weld (simplified)'
                draw.text((14,612),text,fill='white');canvas.paste(tile,(0 if j==0 else 960,0))
            writer.append_data(np.asarray(canvas));frames.append(canvas.resize((1080,480)))
            for name,when in phases:
                if abs(t-when)<1/60+.0001:snapshots[name]=canvas.copy()
    gifs=frames[::2];gifs[0].save(output/'shooting.gif',save_all=True,append_images=gifs[1:],duration=67,loop=0)
    snapshots['release'].save(output/'release.png')
    storyboard=Image.new('RGB',(1440,640))
    for k,(name,_) in enumerate(phases):
        snapshots[name].save(output/f'{name}.png')
        storyboard.paste(snapshots[name].resize((720,320)),((k%2)*720,(k//2)*320))
    storyboard.save(output/'storyboard.png')
    t=run['ball_times'];p=run['ball_positions'];flight=t>=release
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    ax=axes[0,0];ax.plot(p[~flight,0],p[~flight,2],color='gray',label='Held')
    ax.plot(p[flight,0],p[flight,2],color='darkorange',label='Physical flight')
    ax.plot([HOOP[0]-.225,HOOP[0]+.225],[HOOP[2]]*2,'r-',lw=3,label='Rim')
    ax.scatter(*np.array(report['release']['position_m'])[[0,2]],label='Release')
    ax.set(xlabel='Forward X (m)',ylabel='Height Z (m)',title='Side view',ylim=(0,3.7));ax.set_aspect('equal',adjustable='box');ax.legend(fontsize=8)
    ax=axes[0,1];ax.plot(p[:,0],p[:,1],color='darkorange');ax.add_patch(plt.Circle(HOOP[:2],.225,fill=False,color='red'))
    ax.set(xlabel='Forward X (m)',ylabel='Left Y (m)',title='Top view');ax.set_aspect('equal',adjustable='datalim')
    ax=axes[1,0]
    for i,label in enumerate(['vx','vy','vz']):ax.plot(t,run['ball_velocities'][:,i],label=label)
    ax.axvline(release,color='gray',ls='--');ax.set(xlabel='Time (s)',ylabel='Ball velocity (m/s)');ax.legend()
    ax=axes[1,1]
    for i in [0,1,4]:ax.plot(run['times'],run['errors'][:,i]*100,label=str(run['error_names'][i]))
    ax.axvline(release,color='gray',ls='--');ax.set(xlabel='Time (s)',ylabel='Tracking error (cm)');ax.legend()
    for ax in axes.flat:ax.grid(alpha=.2)
    fig.suptitle(f'Loaded reference shot | {label}, without scoring optimization')
    fig.tight_layout();fig.savefig(output/'trajectory.png',dpi=150);plt.close(fig)
    speed=float(np.linalg.norm(report['release']['linear_velocity_m_s']))
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SMPL+H 공을 포함한 자유투 실행</title><style>
body{margin:0;background:#edf1f6;color:#16243b;font:16px system-ui,sans-serif;line-height:1.65}main{max-width:1200px;margin:auto;padding:28px}section{background:white;padding:22px;border-radius:14px;margin:20px 0}video,img{width:100%;border-radius:8px}a{color:#2459bf}table{border-collapse:collapse;width:100%}td,th{text-align:left;padding:9px;border-bottom:1px solid #ddd}.note{color:#526078}canvas{width:100%;background:#18202b;border-radius:8px}input[type=range]{width:95%}pre{white-space:pre-wrap;background:#edf1f6;padding:15px}</style>
<main><h1>공을 든 준비 자세 → 슛 → 릴리스 → 팔로스루</h1>
<p>현재 기준 동작으로 실행한 물리 시뮬레이션입니다. 강화학습이나 득점 최적화는 수행하지 않았습니다.</p>
<section><h2>실행 영상</h2><video src="shooting.mp4" controls muted loop playsinline></video>
<p><a href="shooting.mp4">MP4 영상</a> · <a href="shooting.gif">GIF</a> · <a href="storyboard.png">주요 장면</a> · <a href="__TRACKING__">기존 후보·구간 선택</a></p>
<p class="note">왼쪽: 골대와 공의 실제 궤적. 오른쪽: 전신과 손 자세. 공을 놓을 때 위치·속도를 바꾸지 않고 손과 공의 제약만 해제했습니다.</p></section>
<section><h2>결과</h2><table>__ROWS__</table>
<p>__OUTCOME__ 원본 데이터의 성공 기록과 현재 체형·손 자세의 물리 실행 결과는 구분해야 합니다.</p>
<p class="note">잡기는 오른손과 공 사이의 임시 weld 제약입니다. 공의 질량·관성은 제어에 포함되지만, 손가락 마찰로 잡는 것은 아닙니다. 양손과 공의 충돌은 제외했고 왼손은 보조 손 자세만 추종합니다. 몸통·바닥·림·백보드와 공의 충돌은 계산합니다. 다리는 기존의 발 고정 추종 기준을 사용합니다.</p></section>
__CALIBRATION__
<section><h2>궤적 살펴보기</h2><canvas id="plot" width="1100" height="400"></canvas><input id="time" type="range" min="0" value="0"><span id="label"></span>
<img src="trajectory.png" alt="공의 궤적과 속도, 관절 추종 오차"><p><a href="ball_trajectory.csv">1ms 궤적 CSV</a> · <a href="shooting.npz">상태 NPZ</a> · <a href="shooting_report.json">측정 결과 JSON</a> · <a href="holding_reference.npz">공을 위한 보정 기준 동작</a> · <a href="scene.xml">물리 모델 XML</a></p></section>
<section><h2>재실행</h2><pre>__COMMAND__</pre>
<p>프로젝트 루트에서 실행합니다. 지정한 출력 폴더의 결과를 갱신합니다. SPL 원본과 기존 리타기팅 기준 파일은 수정하지 않습니다.</p></section></main>
<script>const D=__DATA__,R=__REPORT__,c=document.getElementById('plot'),ctx=c.getContext('2d'),slider=document.getElementById('time');slider.max=D.length-1;
function xy(p){return [75+p[1]*190,365-p[3]*90]}
function draw(){const i=+slider.value;ctx.fillStyle='#18202b';ctx.fillRect(0,0,1100,400);ctx.strokeStyle='#596478';ctx.beginPath();ctx.moveTo(20,365);ctx.lineTo(1080,365);ctx.stroke();ctx.strokeStyle='#f87148';ctx.lineWidth=4;ctx.beginPath();ctx.moveTo(75+(4.225-.225)*190,365-3.05*90);ctx.lineTo(75+(4.225+.225)*190,365-3.05*90);ctx.stroke();ctx.lineWidth=2;ctx.strokeStyle='#efab40';ctx.beginPath();for(let j=0;j<=i;j++){let p=xy(D[j]);j?ctx.lineTo(...p):ctx.moveTo(...p)}ctx.stroke();const p=xy(D[i]);ctx.fillStyle='#ffb43c';ctx.beginPath();ctx.arc(...p,11,0,2*Math.PI);ctx.fill();ctx.fillStyle='white';ctx.font='18px sans-serif';ctx.fillText('Side view / X-Z projection',25,30);document.getElementById('label').textContent=`${D[i][0].toFixed(3)}초 | ${D[i][0]<R.release.time_s?'공 잡기':'릴리스 이후'} | 높이 ${D[i][3].toFixed(2)}m`;}
slider.oninput=draw;draw();</script></html>'''
    rows=[('균형 유지','넘어짐 없음' if not report['fell'] else '넘어짐 발생'),('득점','성공' if report['scored'] else '실패 — 현재 기준 실행'),
          ('릴리스 시각',f'{release:.3f}초'),('릴리스 속력',f'{speed:.2f} m/s'),('최고 높이',f'{report["apex_position_m"][2]:.2f} m (림 3.05 m)'),
          ('손목 RMS 오차',f'왼쪽 {report["rms_error_m"]["left_wrist"]*100:.2f} cm / 오른쪽 {report["rms_error_m"]["right_wrist"]*100:.2f} cm'),
          ('발목 최대 수평 이동',f'{report["maximum_ankle_horizontal_displacement_m"]*1000:.2f} mm'),('외부 보조 힘',str(report['max_external_applied_force'])),
          ('릴리스 시 속도 덧붙이기',str(report['release']['qvel_change_at_release'])),('QP 대체 제어 횟수',str(report['qp_fallback_count']))]
    page=page.replace('__ROWS__',''.join(f'<tr><th>{k}</th><td>{v}</td></tr>' for k,v in rows))
    section=''
    try:relative=output.resolve().relative_to(ROOT).as_posix()
    except ValueError:relative=str(output.resolve())
    script='calibrate_release.py' if calibrated else 'shoot_spl.py'
    command=f'.\\.venv\\Scripts\\python.exe src/{script} --output "{relative}" --render'
    if calibrated:
        source=report['adaptation']['source_release_analysis'];diag=report['release_diagnostics']
        params=report['adaptation']['release_correction']
        command+=f' --weight {params["task_weight"]:g} --release-delay {report["release_offset_s"]:g}'
        checks='통과' if diag['tracking_accepted'] else '미통과 — 아래 JSON의 실패 항목 확인'
        section=f'''<section><h2>릴리스 주변 손 움직임 보정</h2>
<p>릴리스 추정: {source['initial_release_frame']} → {source['estimated_release_frame']} 프레임. 원본 공 속력 {source['aligned_target_speed_m_s']:.2f}m/s, 원본 손목 속력 {source['source_wrist_speed_m_s']:.2f}m/s입니다. 현재 체형의 팔 길이에 맞춘 손목 궤적과 MCP 표식으로 추정한 손바닥 방향을 추종합니다. 공 속도를 팔의 목표 속도로 사용하지 않습니다.</p>
<p>보정 범위: 어깨 각 축 ±0.15rad, 팔꿈치 ±0.20rad, 손목 축별 ±0.65/0.65/0.45rad. 몸통·다리·왼팔 기준은 유지하고 보정은 릴리스 전후에서만 적용합니다. 공학적 추종 기준: {checks}. 릴리스 손목 속도 오차(보정 기준 대비) {diag['reference_wrist_velocity_error_m_s']:.3f}m/s.</p>
<p>공 속도의 원본 대비 차이는 {diag['source_velocity_error_m_s']:.2f}m/s입니다. 고정 잡기 제약은 손 안에서 구르거나 손가락 끝을 거쳐 분리되는 공의 운동을 표현하지 않습니다. 이 결과는 원본 공 속도 재현이나 득점 학습 완료를 뜻하지 않습니다.</p>
<img src="release_comparison.png" alt="보정 전후 궤적과 손목·공 속도 비교">
<p><a href="source_release_analysis.json">원본 릴리스 추정과 피팅 구간</a> · <a href="release_diagnostics.json">추종·팔로스루 검증</a> · <a href="release_diagnostics.npz">실제 손목과 잡기 지점의 속도</a> · <a href="__BASELINE__">보정 전 실행</a></p></section>'''
    page=page.replace('__CALIBRATION__',section).replace('__COMMAND__',html.escape(command))
    if contact:
        page=page.replace('공을 놓을 때 위치·속도를 바꾸지 않고 손과 공의 제약만 해제했습니다.',
                          '준비 단계의 잡기 제약을 해제한 뒤 손·손가락 접촉으로 공을 밀고 자연스럽게 분리합니다. 공의 위치·속도를 주입하지 않습니다.')
        page=page.replace('잡기는 오른손과 공 사이의 임시 weld 제약입니다. 공의 질량·관성은 제어에 포함되지만, 손가락 마찰로 잡는 것은 아닙니다. 양손과 공의 충돌은 제외했고 왼손은 보조 손 자세만 추종합니다. 몸통·바닥·림·백보드와 공의 충돌은 계산합니다. 다리는 기존의 발 고정 추종 기준을 사용합니다.',
                          '준비 중에는 임시 weld로 공을 지지합니다. 양손·손가락과 공의 접촉은 항상 계산하며, 제약 해제 후 실제 접촉과 중력으로 발사·분리가 발생합니다. 전완 회내·회외 축을 추가했고 양손 위치를 함께 추종합니다. 손가락 목표는 합성 동작이며 완전한 마찰 파지와 실제 인체 검증은 남아 있습니다.')
        page=page.replace(html.escape(command),html.escape(f'.\\.venv\\Scripts\\python.exe src/prepare_residual.py --output "{relative}" --render'))
        if report.get('adaptation',{}).get('preparation')=='overhand':
            page=page.replace(html.escape(f'.\\.venv\\Scripts\\python.exe src/prepare_residual.py --output "{relative}" --render'),
                              html.escape(f'.\\.venv\\Scripts\\python.exe src/prepare_overhand.py --output "{relative}" --render'))
            page=page.replace('</main>','<section><h2>공 위에서 시작하는 준비 자세</h2><p>오른손 손바닥이 공 위에서 아래를 향하게 시작합니다. 공을 들어 올리는 동안 손 방향을 바꾸며, 기존의 슛 직전 180도 방향 회전을 제거했습니다. 준비 weld는 손과 공의 상대 위치를 유지하므로 공도 함께 이동합니다.</p><p><a href="preparation_review.json">준비 자세와 물리 검사</a> · <a href="nominal_reference.json">동작 설정</a></p></section></main>')
        ready=output/'readiness.json'
        if ready.exists():
            from analyze_residual import analyze
            analyze(output)
            readiness=json.loads(ready.read_text(encoding='utf-8'))
            addition=f'<section><h2>잔차 강화학습 준비 검사</h2><p>물리 기준 통과: {readiness["physics_ready"]}. 득점 학습 준비 완료: {readiness["scoring_curriculum_ready"]}. 강화학습은 실행하지 않았습니다.</p><p>잔차는 오른쪽 팔꿈치 ±0.10rad, 손목 ±0.16rad, 전완 ±0.10rad, 손가락 ±0.10rad, 잡기 해제 시점 ±0.012초로 제한합니다. 몸통·다리 기준과 토크 상한은 유지합니다.</p><p>현재 최고 높이가 림보다 낮습니다. 다음 단계에서는 궤적·조준 보상을 먼저 연결해 위쪽 발사 속도와 방향을 개선하고, 통과 검사를 유지한 채 득점 보상을 추가해야 합니다.</p><p><a href="readiness.json">기준·잔차 시험 결과</a> · <a href="nominal_reference.npz">고정된 기준 동작</a> · <a href="nominal_reference.json">기준 설정</a></p></section>'
            addition=addition.replace('</section>','<img src="residual_comparison.png" alt="작은 잔차 입력에 따른 실제 공 궤적 비교"></section>')
            page=page.replace('</main>',addition+'</main>')
    page=page.replace('__TRACKING__',os.path.relpath(ROOT/'runs/spl_tracking/review.html',output).replace('\\','/'))
    page=page.replace('__BASELINE__',os.path.relpath(ROOT/'runs/spl_shooting/review.html',output).replace('\\','/'))
    outcome='이 기준 실행은 득점했습니다.' if report['scored'] else '이 기준 실행은 득점하지 못했습니다.'
    if report['apex_position_m'][2]<HOOP[2]:outcome+=' 공의 최고 높이가 림 높이보다 낮습니다.'
    page=page.replace('__OUTCOME__',outcome)
    page=page.replace('__DATA__',json.dumps(np.c_[t,p][::10].tolist())).replace('__REPORT__',json.dumps(report).replace('</','<\\/'))
    (output/'review.html').write_text(page,encoding='utf-8')
    print('Shooting review:',output/'review.html',flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=OUTPUT)
    render(parser.parse_args().output)
