"""Paused, recorded-physics pose editor. No mj_step or source-motion writes.
Arrows: source frame, 1..9 / [ ]: selected joint,
minus/equals: selected angle, S: save offsets, R: reset frame, Space: replay.
"""
import argparse,hashlib,json,queue,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,mujoco
from scipy.spatial.transform import Rotation,Slerp
from examples.m4_forearm_rotation import OUT,NAMES,dyn

class PoseEdits:
    def __init__(self,m,states,times):
        self.m=m;self.states=np.array(states);self.times=np.array(times);self.frame=108;self.selected=0;self.edits={}
        self.names=dyn.lift.ARM+[m.joint(j).name for j in range(m.njnt) if m.joint(j).name.startswith('R_') and any(f in m.joint(j).name for f in dyn.FINGERS)]
        self.ids=np.array([int(m.joint(n).qposadr[0]) for n in self.names]);self.bounds=np.array([m.jnt_range[m.joint(n).id] for n in self.names])
    def base(self,frame):
        t=np.clip(3+frame/60,self.times[0],self.times[-1]);i=min(max(0,np.searchsorted(self.times,t)-1),len(self.times)-2);a=(t-self.times[i])/(self.times[i+1]-self.times[i]);q=(1-a)*self.states[i]+a*self.states[i+1]
        for adr in [3,int(self.m.joint('basketball_free').qposadr[0])+3]:
            q[adr:adr+4]=np.roll(Slerp([0,1],Rotation.from_quat(np.roll(self.states[i:i+2,adr:adr+4],-1,axis=1)))(a).as_quat(),1)
        return q
    def pose(self):
        q=self.base(self.frame);q[self.ids]+=self.edits.get(str(self.frame),np.zeros(len(self.ids)));return q
    def remember(self,q):
        self.edits[str(self.frame)]=np.clip(q[self.ids],self.bounds[:,0],self.bounds[:,1])-self.base(self.frame)[self.ids]
    def save(self,path,metadata):
        path.parent.mkdir(parents=True,exist_ok=True)
        # Always a separate timestamped file, never overwrite existing edits.
        target=path.parent/(path.stem+'_'+time.strftime('%Y%m%d_%H%M%S')+'_'+str(time.time_ns()%1000000)+path.suffix)
        target.write_text(json.dumps(dict(**metadata,joint_names=self.names,frame_offsets_rad={k:np.asarray(v).tolist() for k,v in self.edits.items()},fps=60,reference='offsets from recorded physical pose; not fitted SMPL+H poses',physics_verified=False),indent=2));return target

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--condition',choices=['A','B','C'],default='A');p.add_argument('--frame',type=int,default=108);p.add_argument('--save',type=Path,default=ROOT/'outputs/m4_pose_editor/pose_edits.json');p.add_argument('--load',type=Path);p.add_argument('--smoke-seconds',type=float,default=0);args=p.parse_args()
    if not 0<=args.frame<240:raise ValueError('frame must be 0..239')
    folder=OUT/args.condition/'C_1';scene=Path((folder/'scene_path.txt').read_text()).resolve();m=mujoco.MjModel.from_xml_path(str(scene));d=mujoco.MjData(m);states=np.load(folder/'states.npz');editor=PoseEdits(m,states['qpos'],states['time_s']);editor.frame=args.frame
    source=ROOT/'outputs/m3_spl_fitting/spl_P0002_T0049.npz';digest=hashlib.sha256(source.read_bytes()).hexdigest();metadata=dict(condition=args.condition,source_npz_sha256=digest,scene=str(scene),states_sha256=hashlib.sha256((folder/'states.npz').read_bytes()).hexdigest())
    if args.load:
        saved=json.loads(args.load.read_text())
        if saved['condition']!=args.condition or saved['source_npz_sha256']!=digest or saved['joint_names']!=editor.names or saved['states_sha256']!=metadata['states_sha256']:raise ValueError('Edit file reference does not match')
        for key,value in saved['frame_offsets_rad'].items():
            value=np.array(value,float)
            if not 0<=int(key)<240 or value.shape!=(len(editor.ids),) or not np.isfinite(value).all():raise ValueError('Invalid edit data')
            editor.edits[key]=value
    events=queue.Queue();replaying=False;next_frame=time.monotonic();started=time.monotonic()
    import mujoco.viewer as viewer
    d.qpos[:]=editor.pose();d.qvel[:]=0;mujoco.mj_forward(m,d)
    with viewer.launch_passive(m,d,key_callback=events.put,show_left_ui=False,show_right_ui=False) as v:
        v.cam.lookat[:]=[.25,0,1.1];v.cam.distance=2.5;v.cam.azimuth=110;v.cam.elevation=-10
        print(f'EDITOR_READY: condition {args.condition}, frame {editor.frame}, source {editor.frame/60:.3f}s; physics frozen',flush=True)
        while v.is_running():
            v.sync()
            with v.lock():
                editor.remember(d.qpos)
                while not events.empty():
                    key=events.get();changed=False
                    if key in (262,263):editor.frame=int(np.clip(editor.frame+(1 if key==262 else -1),0,239));replaying=False;changed=True
                    elif key in (266,267):editor.frame=int(np.clip(editor.frame+(10 if key==266 else -10),0,239));replaying=False;changed=True
                    elif 49<=key<=57:editor.selected=key-49
                    elif key in (91,93):editor.selected=(editor.selected+(1 if key==93 else -1))%len(editor.ids)
                    elif key in (45,61):d.qpos[editor.ids[editor.selected]]+=.02*(1 if key==61 else -1);editor.remember(d.qpos);changed=True
                    elif key==82:editor.edits.pop(str(editor.frame),None);changed=True
                    elif key==83:print('SAVED:',editor.save(args.save,metadata),flush=True)
                    elif key==32:replaying=not replaying;next_frame=time.monotonic()
                    if changed:d.qpos[:]=editor.pose()
                if replaying and time.monotonic()>=next_frame:
                    editor.frame=(editor.frame+1)%240;d.qpos[:]=editor.pose();next_frame=time.monotonic()+1/30
                # Restore floating roots/ball from the reference. Only humanoid
                # right-arm/finger edits are supported; no fictitious grasp result.
                q=editor.pose();d.qpos[:]=q;d.qvel[:]=0;mujoco.mj_forward(m,d)
            selected=editor.names[editor.selected];angle=d.qpos[editor.ids[editor.selected]]
            v.set_texts((None,None,'POSE EDITOR | FROZEN PHYSICS\n'+f'{args.condition} frame {editor.frame}/239 | source {editor.frame/60:.3f}s\n'+f'{selected}: {angle:.3f} rad\nArrows: frame | PgUp/PgDn: 10 frames\n[ ]: joint | - / =: angle | R: reset | S: save\n1..9: arm joint | Space: 0.5x recorded replay','Ball is a frozen reference.\nEdited pose is NOT validated Holding.'))
            if args.smoke_seconds and time.monotonic()-started>=args.smoke_seconds:break
            time.sleep(.02)
    assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
if __name__=='__main__':main()
