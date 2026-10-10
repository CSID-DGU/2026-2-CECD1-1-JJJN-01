"""M4-2D-1: measured hand FK, bounded grasp search and contact-only static trials.
All old code/assets remain read-only. Anatomical labels are geometric estimates.
"""
import argparse, itertools, json, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import mujoco
import numpy as np
from scipy.optimize import least_squares
from examples import m4_smpl_contact as old
from examples.m3_control import write_csv,FINGERS
from smpl_sim.physics.basketball import HumanoidControllerBridge


def capsule(m,d,g):
    R=d.geom_xmat[g].reshape(3,3);c=d.geom_xpos[g];h=m.geom_size[g,1]
    return c-R[:,2]*h,c+R[:,2]*h,float(m.geom_size[g,0])


def closest(p,a,b):
    return a+np.clip(np.dot(p-a,b-a)/max(np.dot(b-a,b-a),1e-20),0,1)*(b-a)


def tip(m,d,f):
    b=m.body('R_'+f+'3').id;g=m.geom('R_'+f+'3').id
    a,z,_=capsule(m,d,g)
    return max((a,z),key=lambda p:np.linalg.norm(p-d.xpos[b]))


def setup(args):
    args.output.mkdir(parents=True,exist_ok=True)
    path=old.build_scene(args.models/'body_01.xml',args.output/'contact.xml')
    m=mujoco.MjModel.from_xml_path(str(path));d=mujoco.MjData(m)
    bridge=HumanoidControllerBridge(args.models/'body_01.xml',m,args.source)
    adapter=old.ContactTargets(bridge,'hold',0)
    bridge.initialize(m,d);q=adapter.palm_up(adapter.base.copy());d.qpos[:bridge.model.nq]=q
    mujoco.mj_forward(m,d)
    return m,d,bridge,adapter,q


def measure(m,d,q,out):
    d.qpos[:len(q)]=q;mujoco.mj_forward(m,d)
    wrist=m.body('R_Wrist').id;W=d.xmat[wrist].reshape(3,3).copy();normal=-W[:,1]
    joints=[];rows=[];mapping=[]
    for f in FINGERS:
        base_tip=tip(m,d,f).copy()
        for part in (1,2,3):
            name=f'R_{f}{part}';bid=m.body(name).id;g=m.geom(name).id
            a,b,_=capsule(m,d,g);bone=b-a
            if np.dot(bone,(a+b)/2-d.xpos[bid])<0:bone=-bone
            bone/=np.linalg.norm(bone)
            flex=np.cross(bone,normal);flex/=np.linalg.norm(flex)
            abd=np.cross(flex,bone);abd/=np.linalg.norm(abd)
            ideals=np.array([flex,abd,bone]);axes=[]
            for ax in 'xyz':axes.append(d.xaxis[m.joint(name+'_'+ax).id].copy())
            axes=np.array(axes);scores=np.abs(axes@ideals.T)
            perm=max(itertools.permutations(range(3)),key=lambda p:sum(scores[p[k],k] for k in range(3)))
            for role,k in zip(('flexion','abduction','twist'),range(3)):
                ai=perm[k];ax='xyz'[ai];jid=m.joint(name+'_'+ax).id;adr=int(m.jnt_qposadr[jid]);sign=1 if np.dot(axes[ai],ideals[k])>=0 else -1
                mapping.append(dict(segment=name,role=role,axis=ax,sign=sign,alignment=float(scores[ai,k]),bone_palm_normal_absdot=float(abs(np.dot(bone,normal))),joint=name+'_'+ax,qposadr=adr))
            for ai,ax in enumerate('xyz'):
                jid=m.joint(name+'_'+ax).id;adr=int(m.jnt_qposadr[jid]);joints.append(adr)
                for delta in (-.2,.2):
                    d.qpos[:len(q)]=q;d.qpos[adr]+=delta;mujoco.mj_forward(m,d)
                    dp=tip(m,d,f)-base_tip;local=W.T@dp
                    rows.append(dict(joint=name+'_'+ax,delta_rad=delta,tip_dx_m=float(dp[0]),tip_dy_m=float(dp[1]),tip_dz_m=float(dp[2]),
                                     palm_dx_m=float(local[0]),palm_dy_m=float(local[1]),palm_dz_m=float(local[2]),tip_motion_mm=float(np.linalg.norm(dp)*1000),
                                     axis_world_x=float(axes[ai,0]),axis_world_y=float(axes[ai,1]),axis_world_z=float(axes[ai,2])))
                d.qpos[:len(q)]=q;mujoco.mj_forward(m,d)
    thumb_rows=[]
    for x,y,z in itertools.product((-.35,0,.35),repeat=3):
        d.qpos[:len(q)]=q
        for ax,a in zip('xyz',(x,y,z)):d.qpos[int(m.joint('R_Thumb1_'+ax).qposadr[0])]=a
        mujoco.mj_forward(m,d);tp=tip(m,d,'Thumb');ip=tip(m,d,'Index');local=W.T@(tp-d.xpos[wrist])
        thumb_rows.append(dict(x_rad=x,y_rad=y,z_rad=z,index_tip_distance_mm=float(np.linalg.norm(tp-ip)*1000),palm_x_m=local[0],palm_y_m=local[1],palm_z_m=local[2]))
    d.qpos[:len(q)]=q;mujoco.mj_forward(m,d)
    write_csv(out/'thumb_opposition.csv',thumb_rows)
    write_csv(out/'axis_perturbations.csv',rows);write_csv(out/'axis_mapping.csv',mapping)
    (out/'axis_mapping.json').write_text(json.dumps(mapping,indent=2))
    return mapping,joints


def distances(m,d,center):
    records=[]
    for name in ['R_Wrist']+[f'R_{f}{p}' for f in FINGERS for p in (1,2,3)]:
        g=m.geom(name).id;a,b,r=capsule(m,d,g);c=closest(center,a,b);delta=center-c;length=np.linalg.norm(delta);n=delta/max(length,1e-10)
        records.append(dict(geom=name,surface_gap_mm=float((length-r-.121)*1000),normal_x=float(n[0]),normal_y=float(n[1]),normal_z=float(n[2]),
                            centerline_x=float(c[0]),centerline_y=float(c[1]),centerline_z=float(c[2]),radius_mm=r*1000))
    return records


def search(m,d,q,mapping,out):
    """Geometric screening; dynamic stability is evaluated separately, never inferred."""
    wrist=m.body('R_Wrist').id;d.qpos[:len(q)]=q;mujoco.mj_forward(m,d)
    W=d.xmat[wrist].reshape(3,3).copy();origin=d.xpos[wrist].copy()
    variables=[r for r in mapping if r['role']=='flexion' or (r['role']=='abduction' and r['segment'].endswith('1')) or r['segment']=='R_Thumb1']
    # Bounds are experimental conservative limits relative to the model's curved rest pose.
    bounds=[(-.25,1.1) if r['role']=='flexion' else ((-.65,.65) if 'Thumb' in r['segment'] else (-.25,.25)) for r in variables]
    lo=np.array([b[0] for b in bounds]+[-.125,-.035,-.030]);hi=np.array([b[1] for b in bounds]+[-.055,.025,.030])
    def decode(x):
        target=q.copy()
        for r,val in zip(variables,x):target[r['qposadr']]=r['sign']*val
        d.qpos[:len(q)]=target;mujoco.mj_forward(m,d)
        # local -Y is the palm-facing direction; center is placed .12 above its plane.
        center=origin+W@np.array([x[-3],x[-2]-.12,x[-1]])
        return target,center
    def residual(x):
        target,c=decode(x);rs=distances(m,d,c)
        gaps=np.array([r['surface_gap_mm']/1000 for r in rs]);normals=np.array([[r['normal_x'],r['normal_y'],r['normal_z']] for r in rs])
        # Gap fit, strong penetration penalty, spread in lower-sphere contact normals.
        distal=[1+3*fi+2 for fi in range(5)]
        touch=gaps[[0]+distal]
        return np.r_[touch*35,np.minimum(gaps,0)*100, np.maximum(.15-normals[distal,2],0)*.2, x[:-3]*.025]
    candidates=[]
    # Include old behavior for an apples-to-apples reference and bounded geometric candidates.
    baseline=q.copy()
    for f in FINGERS:
        if f!='Thumb':
            for p in (1,2,3):baseline[int(m.joint(f'R_{f}{p}_z').qposadr[0])]=.3*(.7 if p==3 else 1)
    candidates.append(dict(name='old_z_reference',angles={r['joint']:float(baseline[r['qposadr']]) for r in mapping},center_x=-.1,center_z=-.007,wrist_roll=0.))
    for i,cx in enumerate((-.07,-.09,-.11)):
        x=np.r_[[.15 if r['role']=='flexion' else 0 for r in variables],cx,0,-.007]
        fit=least_squares(residual,x,bounds=(lo,hi),max_nfev=180,diff_step=1e-4)
        target,c=decode(fit.x);rs=distances(m,d,c)
        candidates.append(dict(name=f'fk_fit_{i+1}',angles={r['joint']:float(target[r['qposadr']]) for r in mapping},
                               center_x=float(fit.x[-3]),center_z=float(fit.x[-1]),center_y=float(fit.x[-2]),wrist_roll=0.,geometry_cost=float(np.linalg.norm(fit.fun)),
                               variables=[dict(joint=r['joint'],role=r['role'],signed_angle=float(v)) for r,v in zip(variables,fit.x)]))
        write_csv(out/f'fk_fit_{i+1}_geometry.csv',rs)
    # Lower-curvature and rotated-thumb alternatives to test geometry beyond the optimizer minimum.
    for scale in (.0,.5):
        c=json.loads(json.dumps(candidates[2]));c['name']=f'flex_scale_{scale:g}'
        for r in mapping:
            if r['role']=='flexion':c['angles'][r['joint']]*=scale
        candidates.append(c)
    for angle in (-.35,.35):
        c=json.loads(json.dumps(candidates[2]));c['name']=f'thumb_opposition_{angle:+g}';c['angles']['R_Thumb1_y']=angle
        candidates.append(c)
    (out/'candidates.json').write_text(json.dumps(candidates,indent=2));return candidates


class GraspTargets(old.ContactTargets):
    def __init__(self,bridge,candidate):
        super().__init__(bridge,'hold',0);self.candidate=candidate
    def target(self,t,actual=None):
        q=self.palm_up(self.base.copy(),actual)
        for name,value in self.candidate['angles'].items():q[int(self.m.joint(name).qposadr[0])]=value
        return q


def trial(args,candidate,duration=1.,no_pd=False):
    folder=args.output/'trials'/(candidate['name']+('_no_pd' if no_pd else ''));folder.mkdir(parents=True,exist_ok=True)
    m=mujoco.MjModel.from_xml_path(str(args.output/'contact.xml'));d=mujoco.MjData(m)
    bridge=HumanoidControllerBridge(args.models/'body_01.xml',m,args.source);adapter=GraspTargets(bridge,candidate)
    bridge.initialize(m,d);warm=mujoco.MjData(bridge.model);warm.qpos[:]=adapter.target(0);mujoco.mj_forward(bridge.model,warm)
    for _ in range(round(1.5/m.opt.timestep)):
        adapter.control(warm,adapter.target(0,warm));mujoco.mj_step(bridge.model,warm);mujoco.mj_forward(bridge.model,warm)
    d.qpos[:bridge.model.nq]=warm.qpos;d.qvel[:bridge.model.nv]=0;mujoco.mj_forward(m,d)
    # Initial placement only: binary search vertically above the actual settled palm.
    wrist=m.body('R_Wrist').id;W=d.xmat[wrist].reshape(3,3)
    origin=d.xpos[wrist]+W@np.array([candidate['center_x'],candidate.get('center_y',0),candidate['center_z']])
    q=int(m.joint('basketball_free').qposadr[0]);v=int(m.joint('basketball_free').dofadr[0]);bg=m.geom('basketball_geom').id
    low,high=.121,.35
    for _ in range(40):
        height=(low+high)/2;d.qpos[q:q+7]=np.r_[origin+[0,0,height],1,0,0,0];mujoco.mj_forward(m,d)
        if any(bg in (c.geom1,c.geom2) and c.dist<0 for c in d.contact):low=height
        else:high=height
    d.qpos[q:q+7]=np.r_[origin+[0,0,high+.0002],1,0,0,0];d.qvel[v:v+6]=0;mujoco.mj_forward(m,d)
    write_csv(folder/'initial_geometry.csv',distances(m,d,d.qpos[q:q+3]))
    rel0=W.T@(d.qpos[q:q+3]-d.xpos[wrist]);initial=d.qpos[q:q+3].copy();rows=[];contact_rows=[];torques=[];states=[];times=[];seen=False;loss=None;loss_start=None
    joint_names=[r['joint'] for r in json.loads((args.output/'axis_mapping.json').read_text())]
    viewer=None
    if args.gui:
        from mujoco import viewer as mv
        viewer=mv.launch_passive(m,d);viewer.cam.lookat[:]=initial;viewer.cam.distance=.65;viewer.cam.azimuth=110;viewer.cam.elevation=-25
    started=time.monotonic()
    for step in range(round(duration/m.opt.timestep)):
        before=d.qpos[q:q+7].copy();vel=d.qvel[v:v+6].copy();adapter.control(d,adapter.target(d.time,d),no_pd)
        assert np.array_equal(before,d.qpos[q:q+7]) and np.array_equal(vel,d.qvel[v:v+6])
        assert not d.qfrc_applied.any() and not d.xfrc_applied.any()
        mujoco.mj_step(m,d);mujoco.mj_forward(m,d)
        n,nf,F,ff,depth,slip=old.hand_contacts(m,d);W=d.xmat[wrist].reshape(3,3);rel=W.T@(d.qpos[q:q+3]-d.xpos[wrist]);t=float(d.time)
        seen=seen or n>0
        if n>0:loss_start=None
        elif seen and loss_start is None:loss_start=t
        if loss is None and loss_start is not None and t-loss_start>=.05:loss=loss_start
        feet,margin,_=old.contacts(m,d)
        jp,jr=np.zeros((3,m.nv)),np.zeros((3,m.nv));mujoco.mj_jac(m,d,jp,jr,d.qpos[q:q+3],wrist)
        relvel=W.T@(d.qvel[v:v+3]-jp@d.qvel)
        rows.append(dict(time_s=t,contacts=n,finger_contacts=nf,force_z_N=F[2],finger_force_z_N=ff[2],force_x_N=F[0],force_y_N=F[1],slip_ms=slip,
                         relative_x_m=rel[0],relative_y_m=rel[1],relative_z_m=rel[2],drift_m=np.linalg.norm(rel-rel0),
                         ball_x_m=d.qpos[q],ball_y_m=d.qpos[q+1],ball_z_m=d.qpos[q+2],vx_ms=d.qvel[v],vy_ms=d.qvel[v+1],vz_ms=d.qvel[v+2],
                         ball_hand_vx_ms=float((W.T@d.qvel[v:v+3])[0]),ball_hand_vy_ms=float((W.T@d.qvel[v:v+3])[1]),ball_hand_vz_ms=float((W.T@d.qvel[v:v+3])[2]),
                         relative_vx_ms=relvel[0],relative_vy_ms=relvel[1],relative_vz_ms=relvel[2],palm_normal_up=float(-W[2,1]),penetration_mm=depth*1000,com_margin_m=margin,both_feet=int(min(feet.values())>.1),root_height_m=d.qpos[2],pelvis_tilt_deg=float(np.degrees(np.arccos(np.clip(d.xmat[m.body('Pelvis').id].reshape(3,3)[2,1],-1,1))))))
        if step%15==0:
            states.append(d.qpos.copy());times.append(t)
            torques.append(dict(time_s=t,**{name:float(d.qfrc_actuator[int(m.joint(name).dofadr[0])]) for name in joint_names}))
            for ci in range(d.ncon):
                c=d.contact[ci]
                if bg not in (c.geom1,c.geom2):continue
                other=c.geom2 if c.geom1==bg else c.geom1;name=m.geom(int(other)).name
                if name!='R_Wrist' and not old.is_finger(name):continue
                f=np.zeros(6);mujoco.mj_contactForce(m,d,ci,f)
                normal=c.frame[:3]*(-1 if c.geom1==bg else 1)
                contact_rows.append(dict(time_s=t,geom=name,gap_mm=c.dist*1000,normal_x=normal[0],normal_y=normal[1],normal_z=normal[2],normal_force_N=f[0]))
        if viewer:
            if not viewer.is_running():break
            viewer.sync();time.sleep(max(0,t-(time.monotonic()-started)))
    if viewer:viewer.close()
    write_csv(folder/'telemetry.csv',rows);write_csv(folder/'joint_torques.csv',torques)
    if contact_rows:write_csv(folder/'contacts.csv',contact_rows)
    np.savez_compressed(folder/'states.npz',qpos=states,time_s=times)
    held_until=loss if loss is not None else d.time;window=[r for r in rows if .1<=r['time_s']<held_until];early=[r for r in rows if .1<=r['time_s']<=.25]
    gap=longest=0
    for r in rows:
        if r['time_s']<.2:continue
        gap=gap+1 if r['contacts']==0 else 0;longest=max(longest,gap)
    summary=dict(candidate=candidate['name'],no_finger_pd=no_pd,duration_s=float(d.time),first_loss_s=loss,held_3s=bool(seen and held_until>=3.2 and d.time>=3.2-1e-8),
                 holding_after_settle_s=max(0.,held_until-.2) if seen else 0.,max_contacts=max(r['contacts'] for r in rows),
                 max_contact_gap_after_settle_s=longest*m.opt.timestep,max_pelvis_tilt_deg=max(r['pelvis_tilt_deg'] for r in rows),
                 fallen=bool(any(r['root_height_m']<.65*rows[0]['root_height_m'] or r['pelvis_tilt_deg']>35 for r in rows)),
                 supported_finger_force_z_N=float(np.mean([r['finger_force_z_N'] for r in window])) if window else 0.,supported_force_z_N=float(np.mean([r['force_z_N'] for r in window])) if window else 0.,
                 contact_ratio=float(np.mean([r['contacts']>0 for r in rows if r['time_s']>=.2])),supported_slip_rms_ms=float(np.sqrt(np.mean([r['slip_ms']**2 for r in window]))) if window else 0.,
                 max_relative_drift_m=max(r['drift_m'] for r in rows),max_penetration_mm=max(r['penetration_mm'] for r in rows),
                 initial_relative_velocity_hand_ms=np.mean([[r['relative_v'+a+'_ms'] for a in 'xyz'] for r in early],axis=0).tolist() if early else None,
                 initial_ball_velocity_hand_ms=np.mean([[r['ball_hand_v'+a+'_ms'] for a in 'xyz'] for r in early],axis=0).tolist() if early else None,
                 both_feet_ratio=float(np.mean([r['both_feet'] for r in rows])),min_com_margin_m=min(r['com_margin_m'] for r in rows),
                 warnings=[int(w.number) for w in d.warning],neq=m.neq,ball_state_writes_after_initialization=0,
                 max_finger_torque_Nm=max(abs(row[name]) for row in torques for name in joint_names),initial_position_m=initial.tolist())
    (folder/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary));return summary


def render(args,kind,candidate=None):
    from PIL import Image,ImageDraw
    m=mujoco.MjModel.from_xml_path(str(args.output/'contact.xml'));d=mujoco.MjData(m)
    cam=mujoco.MjvCamera();frames=[];renderer=None if args.gui else mujoco.Renderer(m,540,960)
    viewer=None
    if args.gui:
        from mujoco import viewer as mv
        viewer=mv.launch_passive(m,d)
    if kind=='axes':
        data=np.load(args.output/'fk_baseline.npz');q=data['qpos'];mapping=json.loads((args.output/'axis_mapping.json').read_text())
        seq=[]
        for f in FINGERS:
            # Every segment/axis is shown, not just one representative finger.
            for part in (1,2,3):
                for ax in 'xyz':
                    name=f'R_{f}{part}_{ax}';adr=int(m.joint(name).qposadr[0])
                    for a in np.r_[np.linspace(0,.2,5),np.linspace(.2,-.2,9),np.linspace(-.2,0,5)]:
                        state=q.copy();state[adr]=a;seq.append((state,name+f' {a:+.2f} rad'))
        groups=[(f,seq[i*171:(i+1)*171]) for i,f in enumerate(FINGERS)]
    else:
        folder=args.output/'trials'/candidate;data=np.load(folder/'states.npz');groups=[(candidate,list(zip(data['qpos'],[f't={t:.2f}s' for t in data['time_s']])))]
    for label,seq in groups:
        frames=[]
        for state,caption in seq:
            d.qpos[:]=state;mujoco.mj_forward(m,d);cam.lookat[:]=d.xpos[m.body('R_Wrist').id];cam.distance=.52;cam.azimuth=110;cam.elevation=-30
            if kind!='axes':
                cam.lookat[:]=(d.xpos[m.body('R_Wrist').id]+d.xpos[m.body('basketball').id])/2;cam.distance=.7
            if viewer:
                if not viewer.is_running():break
                viewer.cam.lookat[:]=cam.lookat;viewer.cam.distance=cam.distance;viewer.cam.azimuth=cam.azimuth;viewer.cam.elevation=cam.elevation
                viewer.sync();print(caption,flush=True);time.sleep(.06 if kind=='axes' else .033);continue
            renderer.update_scene(d,camera=cam)
            # Show only actual hand collision geometry and ball in this close-up.
            for g in renderer.scene.geoms[:renderer.scene.ngeom]:
                if g.objtype==mujoco.mjtObj.mjOBJ_GEOM and g.objid>=0:
                    name=m.geom(g.objid).name
                    if name not in ('R_Wrist','basketball_geom') and not old.is_finger(name):g.rgba[3]=0
                    elif name=='R_Wrist':g.rgba[:]=[.3,.7,.9,1]
                    elif name!='basketball_geom':g.rgba[:]=[.85,.6,.4,1]
            if kind=='axes':
                name=caption.split()[0];jid=m.joint(name).id
                gi=renderer.scene.ngeom;renderer.scene.ngeom+=1;g=renderer.scene.geoms[gi]
                color={'x':[1,0,0,1],'y':[0,1,0,1],'z':[.2,.5,1,1]}[name[-1]]
                mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_ARROW,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array(color,dtype=np.float32))
                mujoco.mjv_connector(g,mujoco.mjtGeom.mjGEOM_ARROW,.002,d.xanchor[jid],d.xanchor[jid]+.05*d.xaxis[jid])
            im=Image.fromarray(renderer.render());ImageDraw.Draw(im).text((15,15),label+' | '+caption,fill='white');frames.append(im)
        if not frames:continue
        frames[0].save(args.output/(label+'.gif'),save_all=True,append_images=frames[1:],duration=60 if kind=='axes' else 33,loop=0)
        frames[min(30,len(frames)-1)].save(args.output/(label+'.png'))
    if renderer:renderer.close()
    if viewer:viewer.close()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--models',type=Path,default=ROOT.parents[2]/'basketball-r1/SMPLSim/outputs/smplh_physics');p.add_argument('--source',type=Path,default=ROOT/'outputs/m3_spl_fitting/spl_P0002_T0049.npz')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/m4_2d_1_hand');p.add_argument('--mode',choices=['analyze','hold','axes','render'],default='analyze');p.add_argument('--candidate',default='fk_fit_2');p.add_argument('--duration',type=float,default=3.5);p.add_argument('--gui',action='store_true');p.add_argument('--no-finger-pd',action='store_true')
    args=p.parse_args()
    if args.duration<.3 or not np.isfinite(args.duration):p.error('duration must be finite and >= .3')
    if args.mode=='analyze':
        m,d,b,a,q=setup(args);np.savez_compressed(args.output/'fk_baseline.npz',qpos=d.qpos)
        mapping,joints=measure(m,d,q,args.output);candidates=search(m,d,q,mapping,args.output)
        results=[trial(args,c,1.) for c in candidates];write_csv(args.output/'screening.csv',results)
        best=max(results,key=lambda r:(r['holding_after_settle_s'],-r['max_relative_drift_m']))['candidate']
        (args.output/'selected.json').write_text(json.dumps({'candidate':best,'selection':'contact-loss time, tie-break relative drift; 1s screen is not success'}))
        c=next(c for c in candidates if c['name']==best);trial(args,c,3.5);trial(args,c,3.5,True)
    elif args.mode=='axes':render(args,'axes')
    elif args.mode=='render':render(args,'holding',args.candidate)
    else:
        candidates=json.loads((args.output/'candidates.json').read_text());trial(args,next(c for c in candidates if c['name']==args.candidate),args.duration,args.no_finger_pd)

if __name__=='__main__':main()
