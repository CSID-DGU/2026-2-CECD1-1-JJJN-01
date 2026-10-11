"""Standalone candidate review page, comparison video/GIF and tracking plots."""
from pathlib import Path
import hashlib
import json
import numpy as np
import imageio.v2 as imageio
import mujoco
from PIL import Image,ImageDraw,ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from spl_motion import ROOT,KEYS,read_trial,smooth
from humanoid_model import load_model
from retarget_spl import normalize
from track_spl import reference_at

EDGES=[('LEFT_HIP','RIGHT_HIP'),('LEFT_SHOULDER','RIGHT_SHOULDER'),
       ('LEFT_HIP','LEFT_SHOULDER'),('RIGHT_HIP','RIGHT_SHOULDER'),
       ('LEFT_SHOULDER','LEFT_ELBOW'),('LEFT_ELBOW','LEFT_WRIST'),
       ('RIGHT_SHOULDER','RIGHT_ELBOW'),('RIGHT_ELBOW','RIGHT_WRIST'),
       ('LEFT_HIP','LEFT_KNEE'),('LEFT_KNEE','LEFT_ANKLE'),
       ('RIGHT_HIP','RIGHT_KNEE'),('RIGHT_KNEE','RIGHT_ANKLE'),
       ('LEFT_ANKLE','LEFT_HEEL'),('LEFT_ANKLE','LEFT_BIG_TOE'),
       ('RIGHT_ANKLE','RIGHT_HEEL'),('RIGHT_ANKLE','RIGHT_BIG_TOE')]


def source_for(selection):
    _,fps,ids,times,p,_=read_trial(ROOT/selection['path'])
    m=load_model();d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    start=int(np.flatnonzero(ids==selection['start_frame'])[0])
    p,meta=normalize(p,fps,start,m,d)
    return p,ids,fps


def skeleton_image(points,ball,frame,title,width=480,height=480):
    image=Image.new('RGB',(width,height),(24,31,42));draw=ImageDraw.Draw(image)
    az=np.deg2rad(125);el=np.deg2rad(10)
    right=np.array([-np.sin(az),np.cos(az),0]);up=np.array([-np.sin(el)*np.cos(az),-np.sin(el)*np.sin(az),np.cos(el)])
    def xy(p):
        p=p-np.array([0,0,1.]);return (int(width/2+p@right*180),int(height/2-p@up*180))
    for x in np.linspace(-1,1,9):draw.line([xy(np.array([x,-.8,0])),xy(np.array([x,.8,0]))],fill=(46,58,72))
    for a,b in EDGES:draw.line([xy(points[a][frame]),xy(points[b][frame])],fill=(80,200,240),width=4)
    for key in KEYS:
        x,y=xy(points[key][frame]);draw.ellipse((x-3,y-3,x+3,y+3),fill=(210,240,255))
    x,y=xy(ball[frame]);draw.ellipse((x-10,y-10,x+10,y+10),fill=(255,160,45))
    draw.text((14,14),title,fill='white')
    draw.text((14,height-25),'Observed ball: visualization only',fill=(245,190,100))
    return image


def render_results(output,reference_path):
    output=Path(output);ref=np.load(reference_path);tracking=np.load(output/'tracking.npz')
    info=json.loads(Path(reference_path).with_suffix('.json').read_text())
    selection=info['selection'];p,ids,fps=source_for(selection)
    start=np.flatnonzero(ids==selection['start_frame'])[0]
    m=load_model();d=mujoco.MjData(m);option=mujoco.MjvOption();option.sitegroup[:]=0
    camera=mujoco.MjvCamera();camera.lookat[:]=[0,0,1.];camera.distance=3.2;camera.azimuth=125;camera.elevation=-10
    settle=float(tracking['settle']);duration=float(tracking['times'][-1]);frames=[]
    release_time=settle+(selection['release_frame']-selection['start_frame'])/fps/info['playback_speed']
    snapshot=None;closest_release=float('inf')
    with mujoco.Renderer(m,height=480,width=480) as renderer, imageio.get_writer(str(output/'comparison.mp4'),fps=30,codec='libx264',quality=8,macro_block_size=16) as writer:
        for t in np.arange(0,duration,1/30):
            ti=int(np.argmin(abs(tracking['times']-t)))
            motion_time=float(np.clip(t-settle,0,ref['times'][-1]))
            si=min(start+int(round(motion_time*fps*info['playback_speed'])),int(np.flatnonzero(ids==selection['end_frame'])[0]))
            source=skeleton_image(p,p['BALL'],si,f'SPL source / frame {ids[si]}')
            tiles=[source]
            for q,title in [(tracking['reference_qpos'][ti],'Retargeted reference / kinematic'),(tracking['qpos'][ti],'Physical tracking / joint torques only')]:
                d.qpos[:]=q;mujoco.mj_forward(m,d);renderer.update_scene(d,camera,scene_option=option)
                image=Image.fromarray(renderer.render());draw=ImageDraw.Draw(image)
                draw.text((12,14),title,fill='white');draw.text((12,450),f't = {t:.2f} s',fill='white');tiles.append(image)
            canvas=Image.new('RGB',(1440,480))
            for k,tile in enumerate(tiles):canvas.paste(tile,(480*k,0))
            array=np.asarray(canvas);writer.append_data(array)
            frames.append(canvas.resize((960,320)))
            if abs(t-release_time)<closest_release:
                snapshot=canvas.copy();closest_release=abs(t-release_time)
        snapshot.save(output/'comparison.png')
    # PIL GIF is intentionally slower (0.5x) for inspecting the short motion.
    gifframes=frames[::2]
    gifframes[0].save(output/'comparison.gif',save_all=True,append_images=gifframes[1:],duration=133,loop=0)
    storyboard=Image.new('RGB',(1440,960))
    selected=[0,selection['start_frame'],(selection['start_frame']+selection['release_frame'])//2,
              selection['release_frame'],selection['end_frame'],int(ids[-1])]
    for k,frameid in enumerate(selected):
        idx=int(np.argmin(abs(ids-frameid)))
        tile=skeleton_image(p,p['BALL'],idx,f'Original frame {frameid}')
        storyboard.paste(tile,((k%3)*480,(k//3)*480))
    storyboard.save(output/'candidate_storyboard.png')
    fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True)
    t=tracking['times']
    for i,name in enumerate(tracking['error_names']):axes[0].plot(t,tracking['errors'][:,i]*100,label=name)
    axes[0].set_ylabel('Reference error (cm)');axes[0].legend(ncol=3,fontsize=8)
    axes[1].plot(t,tracking['qpos'][:,2],label='Physical pelvis height')
    axes[1].plot(t,tracking['reference_qpos'][:,2],'--',label='Reference pelvis height')
    axes[1].set_ylabel('Height (m)');axes[1].set_xlabel('Simulation time (s)');axes[1].legend()
    for ax in axes:
        ax.axvspan(settle,settle+float(ref['times'][-1]),color='green',alpha=.07)
        ax.axvline(release_time,color='orange',linestyle=':');ax.grid(alpha=.2)
    fig.suptitle(f"Physical tracking at {info['playback_speed']:g}x source speed | no external root support")
    fig.tight_layout();fig.savefig(output/'tracking_metrics.png',dpi=160);plt.close(fig)


def review_page(output):
    output=Path(output)
    ranking=json.loads((output/'candidates.json').read_text())['ranked'][:3]
    selected=json.loads((output/'selection.json').read_text())
    if selected['path'] not in [r['path'] for r in ranking]:ranking.insert(0,selected)
    elif ranking[0]['path']!=selected['path']:
        ranking=[selected]+[r for r in ranking if r['path']!=selected['path']]
    payload=[]
    for entry in ranking:
        p,ids,fps=source_for(entry)
        meta=entry.copy();meta['source_sha256']=hashlib.sha256((ROOT/entry['path']).read_bytes()).hexdigest()
        if entry['path']==selected['path']:meta.update(selected)
        payload.append({'meta':meta,'ids':ids.tolist(),'points':{k:np.round(p[k],5).tolist() for k in KEYS+['BALL']}})
    report=json.loads((output/'tracking_report.json').read_text()) if (output/'tracking_report.json').exists() else {}
    page=r'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>SPL 자유투 검토</title><style>
body{font:16px system-ui,sans-serif;background:#eef2f7;color:#14233b;margin:0}main{max-width:1150px;margin:auto;padding:28px}h1{margin:0 0 8px}section{background:white;border-radius:14px;padding:22px;margin:18px 0;box-shadow:0 3px 15px #2233550a}canvas{width:100%;max-height:550px;background:#18202b;border-radius:8px}video,img{width:100%;border-radius:8px}button,select,input{font:inherit;padding:8px;border:1px solid #b7c2d0;border-radius:6px}button{background:#2459bf;color:white;cursor:pointer}label{display:inline-block;margin:8px}input[type=number]{width:80px}input[type=range]{width:90%}.hint{color:#52657c;line-height:1.7}.row{display:flex;gap:14px;flex-wrap:wrap;align-items:center}pre{white-space:pre-wrap;font-size:13px;background:#f4f6fa;padding:12px}a{color:#2459bf}</style>
<main><h1>SPL 자유투: 후보 검토와 추종 결과</h1><p class="hint">원본 준비 루틴을 확인하고 슛 구간만 선택합니다. 동작 추종 결과는 기준 동작 변환과 실제 물리 실행을 구분해서 표시합니다.</p>
<section><h2>1. 후보 재생 · 슛 구간 지정</h2><div class="row"><select id="candidate"></select><button id="play">재생</button><label><input type="checkbox" id="crop">선택 구간만 반복</label><label>속도 <select id="speed"><option value="0.5">0.5배</option><option value="1" selected>1배</option></select></label></div>
<canvas id="canvas" width="1100" height="550"></canvas><input type="range" id="frame" min="0" value="0"><span id="frameText"></span>
<div><label>시점 회전 <input id="angle" type="range" min="0" max="360" value="125"></label></div>
<div class="row"><label>시작 <input id="start" type="number"></label><label>릴리스 추정 <input id="release" type="number"></label><label>종료 <input id="end" type="number"></label></div>
<div class="row"><button data-mark="start">현재 프레임 → 시작</button><button data-mark="release">현재 프레임 → 릴리스</button><button data-mark="end">현재 프레임 → 종료</button><button id="save">구간 JSON 저장</button></div>
<p class="hint">프레임 경계는 양 끝을 포함합니다. 주황색 공은 측정 위치이며 실제 물리 공이 아닙니다. 릴리스 표시는 자동 추정값으로 직접 조정할 수 있습니다. 여기서 구간을 변경해도 아래의 기존 계산 결과는 바뀌지 않습니다. JSON을 저장한 후 파이프라인을 다시 실행하세요.</p><pre id="command"></pre></section>
<section><h2>2. 원본 → 리타기팅 → 물리 추종</h2><p class="hint" id="resultLabel"></p><video src="comparison.mp4" controls muted loop playsinline></video><p><a href="comparison.mp4" download>MP4 다운로드</a> · <a href="comparison.gif">느린 GIF 보기</a> · <a href="candidate_storyboard.png">원본 주요 장면</a></p><p class="hint">왼쪽: 측정된 원본 골격과 공. 가운데: 체형·관절 한계에 맞춘 운동학적 기준 동작. 오른쪽: 중력과 접촉을 적용한 실제 물리 추종. 발은 지면에 유지하도록 변환했으며 손가락은 단순한 자세를 사용합니다. 공 접촉·득점·강화학습은 포함하지 않았습니다.</p></section>
<section><h2>3. 추종 측정 결과</h2><div id="metrics"></div><img src="tracking_metrics.png" alt="추종 오차와 골반 높이"><p><a href="tracking_report.json">전체 측정 결과 JSON</a></p></section></main>
<script>const DATA=__DATA__,REPORT=__REPORT__,EDGES=__EDGES__;
const $=id=>document.getElementById(id),ctx=$('canvas').getContext('2d');let current=0,playing=false,last=0,acc=0;
DATA.forEach((d,i)=>{let o=document.createElement('option');o.value=i;o.textContent=`후보 ${i+1}: ${d.meta.participant} / ${d.meta.trial} (${d.meta.result})`;$('candidate').append(o)});
function choose(){current=+$('candidate').value;let d=DATA[current];$('frame').max=d.ids.length-1;$('frame').value=0;for(let k of ['start','release','end'])$(k).value=d.meta[k+'_frame'];draw()}
function project(p){let a=+$('angle').value*Math.PI/180,e=.16;let v=[p[0],p[1],p[2]-1.05];return[550+(-Math.sin(a)*v[0]+Math.cos(a)*v[1])*210,285-(-Math.sin(e)*Math.cos(a)*v[0]-Math.sin(e)*Math.sin(a)*v[1]+Math.cos(e)*v[2])*210]}
function draw(){let d=DATA[current],i=+$('frame').value,f=d.ids[i];ctx.fillStyle='#18202b';ctx.fillRect(0,0,1100,550);ctx.strokeStyle='#344052';ctx.lineWidth=1;for(let x=-1;x<=1;x+=.25){let a=project([x,-1,0]),b=project([x,1,0]);ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke()}ctx.strokeStyle='#51d4f0';ctx.lineWidth=5;for(let [a,b]of EDGES){ctx.beginPath();ctx.moveTo(...project(d.points[a][i]));ctx.lineTo(...project(d.points[b][i]));ctx.stroke()}for(let[k,pts]of Object.entries(d.points)){let p=project(pts[i]);ctx.fillStyle=k==='BALL'?'#ffa838':'#d3f7ff';ctx.beginPath();ctx.arc(...p,k==='BALL'?12:4,0,7);ctx.fill()}ctx.fillStyle='white';ctx.font='18px sans-serif';let label=f<+$('start').value?'제외: 준비 루틴':f>+$('end').value?'제외: 후속 동작':'선택한 슛 구간';ctx.fillText(label,22,32);ctx.fillText(`frame ${f} / ${(f/d.meta.fps).toFixed(2)} s`,22,58);$('frameText').textContent=` ${f} 프레임`;$('command').textContent='저장한 JSON을 runs/spl_tracking/selection.json에 배치한 뒤:\n.\\.venv\\Scripts\\python.exe src/run_spl_pipeline.py --stage all';}
$('candidate').onchange=choose;$('frame').oninput=draw;$('angle').oninput=draw;for(let k of ['start','release','end'])$(k).oninput=draw;
$('play').onclick=()=>{playing=!playing;$('play').textContent=playing?'일시정지':'재생';last=0};document.querySelectorAll('[data-mark]').forEach(b=>b.onclick=()=>{$(b.dataset.mark).value=DATA[current].ids[+$('frame').value];draw()});
$('save').onclick=()=>{let d=DATA[current],s=+$('start').value,r=+$('release').value,e=+$('end').value;if(!(s<r&&r<e&&d.ids.includes(s)&&d.ids.includes(r)&&d.ids.includes(e))){alert('데이터에 있는 프레임으로 시작 < 릴리스 < 종료를 지정하세요.');return}let v={...d.meta,start_frame:s,release_frame:r,end_frame:e,selection_method:'User-selected in review.html',frame_bounds:'Inclusive source frame IDs'};let u=URL.createObjectURL(new Blob([JSON.stringify(v,null,2)],{type:'application/json'}));let a=document.createElement('a');a.href=u;a.download='selection.json';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000)};
function tick(ts){if(playing){if(last)acc+=(ts-last)/1000*DATA[current].meta.fps*+$('speed').value;if(acc>=1){let d=DATA[current],i=+$('frame').value+Math.floor(acc);acc%=1;let lo=$('crop').checked?d.ids.indexOf(+$('start').value):0,hi=$('crop').checked?d.ids.indexOf(+$('end').value):d.ids.length-1;if(lo<0||hi<lo){lo=0;hi=d.ids.length-1}if(i>hi||i<lo)i=lo;$('frame').value=i;draw()}last=ts}else last=0;requestAnimationFrame(tick)}
$('resultLabel').textContent=REPORT.source_selection?`현재 결과: ${REPORT.source_selection.participant} ${REPORT.source_selection.trial}, ${REPORT.source_selection.start_frame}–${REPORT.source_selection.end_frame}프레임. 준비 ${REPORT.settle_s}초 + 슛 ${REPORT.motion_duration_s}초 + 마무리 ${REPORT.hold_s}초.`:'아직 물리 실행 결과가 없습니다.';
if(REPORT.rms_error_m){$('metrics').textContent=`넘어짐: ${REPORT.fell?'있음':'없음'} | 왼손목 RMS ${(REPORT.rms_error_m.left_wrist*100).toFixed(1)}cm | 오른손목 RMS ${(REPORT.rms_error_m.right_wrist*100).toFixed(1)}cm | 외부 지지력 ${REPORT.max_external_root_or_body_force}`;}choose();requestAnimationFrame(tick);</script></html>'''
    for key,value in [('__DATA__',payload),('__REPORT__',report),('__EDGES__',EDGES)]:page=page.replace(key,json.dumps(value,ensure_ascii=False).replace('</','<\\/'))
    page=page.replace('<main>','<main><p><a href="../spl_shooting/review.html">공을 포함한 슛·팔로스루 결과 보기</a></p>',1)
    (output/'review.html').write_text(page,encoding='utf-8')


if __name__=='__main__':
    out=ROOT/'runs/spl_tracking'
    render_results(out,ROOT/'data/motions/processed/free_throw_reference.npz');review_page(out)
