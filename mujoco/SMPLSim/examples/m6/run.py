"""python -m examples.m6.run validate/train/evaluate/replay/audit/regression."""
import argparse,json,time,hashlib,sys,runpy
from pathlib import Path
from examples.m6.core import OUT,ROOT,WeldShootingEnv,generate_reference
import numpy as np

def evaluate(args):
 from stable_baselines3 import PPO
 model=PPO.load(args.model,device='cpu') if args.model else None
 records=[]
 for seed in range(args.seed,args.seed+args.episodes):
  e=WeldShootingEnv(stage=args.stage,mode=args.mode,reference=args.reference or generate_reference(args.mode,args.motion_seconds),record=True,render_mode='human' if args.gui else None);o,_=e.reset(seed=seed);total=0;started=time.monotonic()
  while True:
   a=model.predict(o,deterministic=True)[0] if model else np.zeros(e.action_space.shape)
   o,r,t,u,info=e.step(a);total+=r
   if args.gui:time.sleep(max(0,e.d.time/args.speed-(time.monotonic()-started)))
   if t or u:break
  path=args.output/f'seed_{seed}';z=e.save(path);z.update(seed=seed,episode_reward=total,wall_seconds=time.monotonic()-started,env_steps=len(e.rows));e.close();records.append(z);print(json.dumps(z),flush=True)
 args.output.mkdir(parents=True,exist_ok=True);(args.output/'evaluation.json').write_text(json.dumps(dict(model=str(args.model),stage=args.stage,score_rate=float(np.mean([r['score'] for r in records])),episodes=records),indent=2))

def train(args):
 import torch
 torch.set_num_threads(1)
 from stable_baselines3 import PPO
 from stable_baselines3.common.monitor import Monitor
 from stable_baselines3.common.callbacks import CheckpointCallback,EvalCallback
 from stable_baselines3.common.logger import configure
 args.output.mkdir(parents=True,exist_ok=True)
 config=dict(stage=args.stage,motion_seconds=args.motion_seconds,action_dim=30,obs_dim=169,reference_sha256=hashlib.sha256(Path(args.reference or generate_reference('shoot',args.motion_seconds)).read_bytes()).hexdigest())
 if args.resume:
  saved=json.loads((args.resume.parent/'environment.json').read_text()) if (args.resume.parent/'environment.json').exists() else json.loads((args.resume.parent.parent/'environment.json').read_text())
  if saved!=config:raise ValueError('Resume environment configuration differs')
 (args.output/'environment.json').write_text(json.dumps(config,indent=2))
 if (args.output/'final.zip').exists():raise FileExistsError('Use a new output directory; existing training preserved')
 env=Monitor(WeldShootingEnv(stage=args.stage,reference=args.reference or generate_reference('shoot',args.motion_seconds)),str(args.output/'monitor.csv'));ev=Monitor(WeldShootingEnv(stage=args.stage,reference=args.reference or generate_reference('shoot',args.motion_seconds)));model=PPO.load(args.resume or args.init_model,env=env,device='cpu') if args.resume or args.init_model else PPO('MlpPolicy',env,n_steps=128,batch_size=32,n_epochs=3,learning_rate=3e-4,seed=args.seed,device='cpu',policy_kwargs=dict(net_arch=[64,64]),verbose=1)
 model.set_logger(configure(str(args.output),['stdout','csv']))
 callback=[CheckpointCallback(save_freq=args.checkpoint,save_path=str(args.output/'checkpoints'),name_prefix='m6'),EvalCallback(ev,best_model_save_path=str(args.output/'best'),log_path=str(args.output/'eval'),eval_freq=max(128,args.checkpoint),n_eval_episodes=1,deterministic=True)]
 before=model.num_timesteps;started=time.monotonic();model.learn(total_timesteps=args.steps,reset_num_timesteps=not bool(args.resume),callback=callback);model.save(args.output/'final');elapsed=time.monotonic()-started
 (args.output/'training.json').write_text(json.dumps(dict(requested_steps=args.steps,initial_steps=before,final_steps=model.num_timesteps,elapsed_seconds=elapsed,steps_per_wall_second=(model.num_timesteps-before)/elapsed,seed=args.seed,stage=args.stage,device='cpu',resume=str(args.resume),local_only=True,actual_scoring_learning_verified=False),indent=2));env.close();ev.close()

def validate(args):
 args.output.mkdir(parents=True,exist_ok=True)
 for mode in 'ABCD':
  e=WeldShootingEnv(stage=0,mode=mode,record=True);e.reset(seed=args.seed)
  while True:
   _,_,t,u,_=e.step(np.zeros(e.action_space.shape))
   if t or u:break
  z=e.save(args.output/mode);print(mode,json.dumps(z),flush=True);e.close()

def replay(args):
 import mujoco,mujoco.viewer
 from contextlib import nullcontext
 from PIL import Image,ImageDraw
 states=np.load(args.states);q=states['qpos'];ts=states['time_s'];e=WeldShootingEnv(mode=args.mode,reference=args.reference);e.d.eq_active[:]=0
 cam=mujoco.MjvCamera();mujoco.mjv_defaultCamera(cam);cam.lookat[:]=[1.8,0,1.65];cam.distance=6;cam.azimuth=100;cam.elevation=-12
 renderer=mujoco.Renderer(e.m,height=480,width=800) if args.gif else None;frames=[]
 context=mujoco.viewer.launch_passive(e.m,e.d) if args.gui else nullcontext(None)
 with context as viewer:
  if viewer:
   viewer.cam.lookat[:]=cam.lookat;viewer.cam.distance=cam.distance;viewer.cam.azimuth=cam.azimuth;viewer.cam.elevation=cam.elevation
  repeat=0
  while True:
   start=time.monotonic()
   for i,(t,pose) in enumerate(zip(ts,q)):
    if viewer and not viewer.is_running():break
    with viewer.lock() if viewer else nullcontext():
     e.d.qpos[:]=pose;mujoco.mj_forward(e.m,e.d)
     if viewer:
      from examples.m4_basketball import trail
      viewer.user_scn.ngeom=0;trail(viewer.user_scn,q[:i+1:3,e.qa:e.qa+3])
     if renderer and i%3==0:
      renderer.update_scene(e.d,camera=cam)
      from examples.m4_basketball import trail
      trail(renderer.scene,q[:i+1:3,e.qa:e.qa+3]);im=Image.fromarray(renderer.render());ImageDraw.Draw(im).text((12,12),f'M6 {args.mode} | t={t:.2f}s | '+('WELD' if t<e.reference.release else 'RELEASED')+' | saved physics replay',fill='yellow');frames.append(im.quantize(colors=256))
    if viewer:viewer.sync();time.sleep(max(0,(t-ts[0])/args.speed-(time.monotonic()-start)))
   repeat+=1
   if renderer or not viewer or not viewer.is_running() or (args.repeats and repeat>=args.repeats):break
 if renderer:renderer.close();frames[0].save(args.gif,save_all=True,append_images=frames[1:],duration=100,loop=0,optimize=False)
 e.close()

def audit(args):
 old=json.loads((OUT/'preservation_before.json').read_text());changed=[p for p,h in old.items() if not Path(p).exists() or hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h];result=dict(protected_files=len(old),changed=changed);(OUT/'preservation_audit.json').write_text(json.dumps(result,indent=2));print(result)
 if changed:raise AssertionError('Protected files changed')

def regression(args):
 from examples import m5_coordinated_eval
 target=OUT/'regression_audit';target.mkdir(exist_ok=True);(target/'preservation_before.json').write_bytes((m5_coordinated_eval.OUT/'preservation_before.json').read_bytes());m5_coordinated_eval.OUT=target
 runpy.run_path(str(ROOT/'tests/run_shooting_grip_regressions.py'),run_name='__main__')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=['validate','train','evaluate','replay','audit','regression']);p.add_argument('--stage',type=int,choices=[0,1,2],default=2);p.add_argument('--mode',choices=['shoot','A','B','C','D'],default='shoot');p.add_argument('--output',type=Path,default=OUT/'evaluation');p.add_argument('--seed',type=int,default=7);p.add_argument('--episodes',type=int,default=1);p.add_argument('--reference',type=Path);p.add_argument('--motion-seconds',type=float,default=2.4);p.add_argument('--steps',type=int,default=256);p.add_argument('--checkpoint',type=int,default=128);p.add_argument('--init-model',type=Path);p.add_argument('--resume',type=Path);p.add_argument('--model',type=Path);p.add_argument('--states',type=Path);p.add_argument('--gui',action='store_true');p.add_argument('--gif',type=Path);p.add_argument('--speed',type=float,default=1);p.add_argument('--repeats',type=int,default=1);a=p.parse_args()
 if a.speed<=0 or a.steps<=0 or a.checkpoint<=0 or a.episodes<=0 or a.repeats<0:p.error('Invalid positive count/speed')
 globals()[a.command](a)
