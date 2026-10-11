"""PPO with portable, atomic checkpoints and controlled long-running training."""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import argparse,json,time,hashlib,shutil,subprocess,sys,random,platform
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv,SubprocVecEnv,VecNormalize,sync_envs_normalization
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from free_throw_env import FreeThrowEnv
from humanoid_model import ROOT


def write_json(path,value):
    path=Path(path);temp=path.with_name(f'{path.name}.{os.getpid()}.{time.time_ns()}.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')
    try:
        for attempt in range(10):
            try:
                temp.replace(path)
                return
            except PermissionError:
                if attempt==9:raise
                time.sleep(.05*(attempt+1))
    finally:
        if temp.exists():temp.unlink()


class Episodes(BaseCallback):
    def __init__(self,output,state):super().__init__();self.output=output;self.state=state;self.last_status=0.
    def _on_step(self):
        for info in self.locals['infos']:
            if 'metrics' not in info:continue
            metrics=info['metrics'];self.state['episodes']+=1
            metrics.update(episode=self.state['episodes'],steps=self.num_timesteps,wall_time_s=time.time()-self.state['started_at'])
            with (self.output/'episodes.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(metrics)+'\n')
            self.state['last_episode']=metrics
            if self.state['episodes']%10==0:print(json.dumps({'episode':self.state['episodes'],'steps':self.num_timesteps,'last':metrics}),flush=True)
        if time.time()-self.last_status>15:
            self.state['steps']=self.num_timesteps;self.state['updated_at']=time.time();write_json(self.output/'progress.json',self.state);self.last_status=time.time()
        return True


def vector(bundle,workers):
    factories=[lambda:FreeThrowEnv(bundle) for _ in range(workers)]
    return DummyVecEnv(factories) if workers==1 else SubprocVecEnv(factories,start_method='spawn')


def evaluate(model,norm,bundle):
    env=DummyVecEnv([lambda:FreeThrowEnv(bundle)])
    evaluation=VecNormalize(env,training=False,norm_obs=True,norm_reward=False,clip_obs=10)
    sync_envs_normalization(norm,evaluation)
    obs=evaluation.reset();done=False;total=0.
    while not done:
        action,_=model.predict(obs,deterministic=True);obs,reward,dones,infos=evaluation.step(action)
        total+=float(reward[0]);done=bool(dones[0])
    result=infos[0]['metrics'];result['return']=total;evaluation.close();return result


def checkpoint(output,name,model,norm,state,config,bundle):
    destination=output/'checkpoints'/name
    if destination.exists():raise FileExistsError(destination)
    # Windows can deny directory renames while an indexer scans new files.
    # Publish only after an atomic completion marker; never load an incomplete
    # directory, and keep latest.json pointing to the previous complete save.
    temp=destination;temp.mkdir(parents=True)
    model.save(temp/'policy.zip');norm.save(temp/'normalization.pkl')
    torch.save({'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state()},temp/'rng.pt')
    shutil.copytree(bundle,temp/'environment',ignore=shutil.ignore_patterns('*.mp4','*.gif','*.png','*.html','*.csv','shooting.npz','probes','checkpoints'))
    # Minimal runtime input bundle: scene, frozen nominal motion, configuration.
    for path in list((temp/'environment').iterdir()):
        if path.is_file() and path.name not in ['scene.xml','nominal_reference.npz','nominal_reference.json']:path.unlink()
    (temp/'src').mkdir();source_dir=Path(__file__).resolve().parent
    for source in source_dir.glob('*.py'):shutil.copy2(source,temp/'src'/source.name)
    (temp/'assets').mkdir()
    for name in ['smplh_humanoid.xml','smplh_shooting.xml','smplh_shooting.json']:
        source=ROOT/'assets'/name
        if source.exists():shutil.copy2(source,temp/'assets'/name)
    write_json(temp/'training_state.json',dict(state,checkpoint_boundary='Policy contains completed PPO updates; active rollout is not serialized.',resume_semantics='New episodes; active environments and rollout fragments are not restored.'))
    write_json(temp/'config.json',config)
    frozen=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)
    (temp/'requirements.txt').write_text(frozen,encoding='utf-8')
    (temp/'RESUME.txt').write_text('Run with matching Python and dependency versions.\npython src/train_residual.py --resume . --output resumed_training --episodes 10000\nResume starts new episodes, preserving policy, optimizer, normalization and counters.\nA compatible pose-only refinement can be selected with --transfer-bundle PATH.\nFor CPU PyTorch packages an appropriate official PyTorch package index may be needed.\n',encoding='utf-8')
    hashes={p.relative_to(temp).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in temp.rglob('*') if p.is_file()}
    write_json(temp/'manifest.json',{'sha256':hashes,'python':platform.python_version(),'platform':platform.platform()})
    write_json(temp/'COMPLETE.json',{'complete':True,'steps':model.num_timesteps})
    write_json(output/'latest.json',{'checkpoint':str(destination.relative_to(output)),'episodes':state['episodes'],'steps':model.num_timesteps})
    state['last_checkpoint']=str(destination.relative_to(output));print('CHECKPOINT',destination,flush=True)
    return destination


def validate_bundle(path):
    if not (path/'COMPLETE.json').exists():raise ValueError('Incomplete checkpoint; completion marker missing')
    manifest=json.loads((path/'manifest.json').read_text(encoding='utf-8'))
    for name,expected in manifest['sha256'].items():
        if hashlib.sha256((path/name).read_bytes()).hexdigest()!=expected:raise ValueError('Checkpoint integrity mismatch: '+name)
    # Use the supplied runtime code when transferring the directory to another
    # machine, or refuse silently changed training/environment implementation.
    for name in ['free_throw_env.py','track_spl.py','contact_shot.py']:
        source=Path(__file__).resolve().parent/name
        if source.read_bytes()!=(path/'src'/name).read_bytes():raise ValueError('Runtime source differs from checkpoint: '+name)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'runs/residual_ppo');p.add_argument('--resume',type=Path)
    p.add_argument('--episodes',type=int,default=10000);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--checkpoint-every',type=int,default=500);p.add_argument('--eval-every',type=int,default=1000)
    p.add_argument('--checkpoint-seconds',type=float,default=300);p.add_argument('--eval-first',type=int,default=500)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--bundle',type=Path,default=ROOT/'runs/spl_residual_ready')
    p.add_argument('--transfer-bundle',type=Path,help='Compatible replacement reference used with --resume; policy, optimizer, normalization and counters are retained')
    a=p.parse_args()
    if a.transfer_bundle and not a.resume:p.error('--transfer-bundle requires --resume')
    if min(a.episodes,a.workers,a.checkpoint_every,a.eval_every,a.eval_first,a.checkpoint_seconds)<=0:p.error('Require positive budgets and intervals')
    output=a.output.resolve();output.mkdir(parents=True,exist_ok=True)
    if (output/'training.lock').exists():raise RuntimeError('Training lock exists; check active process before recovery')
    if not a.resume and (output/'progress.json').exists():raise RuntimeError('Existing run; use --resume with its checkpoint')
    torch.set_num_threads(1);raw=None;norm=None;lock=output/'training.lock'
    lock.write_text(str(os.getpid()),encoding='ascii')
    try:
        config={'algorithm':'PPO','policy_hz':20,'physics_hz':1000,'controller_hz':200,'action_limits':[.10,.16,.10,.10,.012],
                'learning_rate':1e-4,'n_steps':88,'batch_size':44,'n_epochs':5,'gamma':.995,'gae_lambda':.95,'clip_range':.15,
                'initial_log_std':-2.3,'seed':a.seed,'workers':a.workers,'reward':'Dense physically measured launch/trajectory reward + actual gated score; posture/balance/residual penalties.',
                'max_episodes':a.episodes,'checkpoint_every':a.checkpoint_every,'eval_every':a.eval_every,'checkpoint_seconds':a.checkpoint_seconds}
        bundle=a.bundle.resolve()
        state={'status':'starting','episodes':0,'steps':0,'started_at':time.time(),'pid':os.getpid(),'best_quality':-1e9,'best_episode':0,
               'baseline_velocity_error':None,'evaluations':[],'updates':0}
        if a.resume:
            resume=a.resume.resolve();validate_bundle(resume)
            bundle=a.transfer_bundle.resolve() if a.transfer_bundle else resume/'environment'
            state.update(json.loads((resume/'training_state.json').read_text(encoding='utf-8')))
            saved=json.loads((resume/'config.json').read_text(encoding='utf-8'))
            for key in ['learning_rate','n_steps','batch_size','n_epochs','gamma','gae_lambda','clip_range','seed']:config[key]=saved[key]
            state.update(status='resuming',pid=os.getpid(),started_at=time.time(),resumed_from=str(resume))
            state.pop('stop_reason',None);state.pop('error',None)
            if a.transfer_bundle:
                transfer_env=FreeThrowEnv(bundle);check_env(transfer_env,warn=True)
                obs,_=transfer_env.reset(seed=a.seed);done=False
                while not done:obs,_,done,_,info=transfer_env.step(np.zeros(5))
                baseline=info['metrics'];transfer_env.close()
                if not baseline['accepted']:raise RuntimeError('Transfer baseline failed physics gates')
                state['baseline_velocity_error']=baseline['release_velocity_error_m_s']
                state['transferred_environment_from']=str(resume/'environment')
                state['transferred_environment_to']=str(bundle)
                config['environment_transfer']='Fixed thumb-reference refinement; policy and normalization retained.'
            # Carry the previous best into a new run directory, including when
            # the last checkpoint itself contains a lower-performing policy.
            previous=resume
            for _ in range(10):
                prior_run=previous.parent.parent;best_file=prior_run/'best.json'
                if best_file.exists():
                    best=json.loads(best_file.read_text(encoding='utf-8'));best_path=prior_run/best['checkpoint']
                    if best_path.exists() and not (output/'best.json').exists():
                        inherited=output/'checkpoints/inherited_best';shutil.copytree(best_path,inherited)
                        best['checkpoint']='checkpoints/inherited_best';write_json(output/'best.json',best)
                    break
                prior_state=json.loads((previous/'training_state.json').read_text(encoding='utf-8'))
                if not prior_state.get('resumed_from'):break
                previous=Path(prior_state['resumed_from'])
                if not previous.exists():break
        else:
            baseline_env=FreeThrowEnv(bundle);check_env(baseline_env,warn=True)
            obs,_=baseline_env.reset(seed=a.seed);done=False
            while not done:obs,_,done,_,info=baseline_env.step(np.zeros(5))
            baseline=info['metrics'];write_json(output/'baseline.json',baseline)
            if not baseline['accepted']:raise RuntimeError('Baseline failed physics gates')
            state['baseline_velocity_error']=baseline['release_velocity_error_m_s']
        raw=vector(bundle,a.workers)
        if a.resume:
            norm=VecNormalize.load(a.resume/'normalization.pkl',raw);norm.training=True;norm.norm_reward=False
            model=PPO.load(a.resume/'policy.zip',env=norm,device='cpu')
            rng=torch.load(a.resume/'rng.pt',map_location='cpu',weights_only=False)
            random.setstate(rng['python']);np.random.set_state(rng['numpy']);torch.set_rng_state(rng['torch'])
        else:
            norm=VecNormalize(raw,norm_obs=True,norm_reward=False,clip_obs=10)
            model=PPO('MlpPolicy',norm,device='cpu',seed=a.seed,learning_rate=config['learning_rate'],n_steps=88,batch_size=44,n_epochs=5,
                      gamma=.995,gae_lambda=.95,clip_range=.15,policy_kwargs={'net_arch':dict(pi=[64,64],vf=[64,64]),'log_std_init':-2.3},verbose=0)
        config['max_episodes']=a.episodes;write_json(output/'config.json',config)
        initial=checkpoint(output,f'initial_{model.num_timesteps:09d}_{int(time.time())}',model,norm,state,config,bundle)
        callback=Episodes(output,state);last_save=time.time();last_saved_episodes=state['episodes']
        next_eval=((state['episodes']//a.eval_every)+1)*a.eval_every
        if state['episodes']<a.eval_first:next_eval=min(next_eval,a.eval_first)
        state['status']='training'
        stop_reason='episode_budget'
        while state['episodes']<a.episodes:
            model.learn(total_timesteps=model.n_steps*a.workers,reset_num_timesteps=False,callback=callback,progress_bar=False)
            state['updates']+=1;state['steps']=model.num_timesteps
            due_eval=state['episodes']>=next_eval or state['episodes']>=a.episodes
            improved=False
            if due_eval:
                state['status']='evaluating';write_json(output/'progress.json',state)
                result=evaluate(model,norm,bundle);result.update(episode=state['episodes'],steps=model.num_timesteps)
                state['evaluations'].append(result);print('EVALUATION',json.dumps(result),flush=True)
                # A physically invalid policy can never replace best.
                quality=float(result['trajectory_reward'])+20*int(result['valid_success'])
                if result['accepted'] and quality>state['best_quality']+.01:
                    state['best_quality']=quality;state['best_episode']=state['episodes'];improved=True
                next_eval=((state['episodes']//a.eval_every)+1)*a.eval_every;state['status']='training'
                if state['episodes']>=2000:
                    successful=any(e['valid_success'] for e in state['evaluations'])
                    better=any(e['accepted'] and e['release_velocity_error_m_s']<state['baseline_velocity_error']-.10 for e in state['evaluations'])
                    if not successful and not better:stop_reason='no_trajectory_improvement_at_2000';break
                current_evaluations=[e for e in state['evaluations'] if e['episode']>=a.eval_first]
                if state['episodes']>=a.eval_first and current_evaluations and all(not e['accepted'] for e in current_evaluations):stop_reason='evaluation_physics_gates_failed';break
                if state['episodes']>=2000 and state['episodes']-state['best_episode']>=2000:stop_reason='trajectory_plateau_for_2000_episodes';break
            periodic=state['episodes']//a.checkpoint_every>last_saved_episodes//a.checkpoint_every
            if periodic or improved or time.time()-last_save>=a.checkpoint_seconds:
                saved=checkpoint(output,f'step_{model.num_timesteps:09d}',model,norm,state,config,bundle);last_save=time.time();last_saved_episodes=state['episodes']
                if improved:write_json(output/'best.json',{'checkpoint':str(saved.relative_to(output)),'evaluation':result})
            state['updated_at']=time.time();write_json(output/'progress.json',state)
            if (output/'STOP').exists():stop_reason='stop_file';break
        state.update(status='finished' if stop_reason=='episode_budget' else 'stopped',stop_reason=stop_reason,steps=model.num_timesteps,updated_at=time.time())
        checkpoint(output,f'final_{model.num_timesteps:09d}_{int(time.time())}',model,norm,state,config,bundle)
        write_json(output/'progress.json',state)
    except BaseException as error:
        if 'state' in locals():
            state.update(status='interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',error=repr(error),updated_at=time.time())
            if 'model' in locals() and norm is not None:
                checkpoint(output,f'interrupted_{model.num_timesteps:09d}_{int(time.time())}',model,norm,state,config,bundle)
            write_json(output/'progress.json',state)
        raise
    finally:
        if norm is not None:norm.close()
        elif raw is not None:raw.close()
        lock.unlink(missing_ok=True)


if __name__=='__main__':main()
