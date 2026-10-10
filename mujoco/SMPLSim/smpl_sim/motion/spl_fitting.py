"""Single-trial differentiable SMPL+H joint-position fitting on CPU."""
import numpy as np
import torch
from smplx.lbs import batch_rodrigues, batch_rigid_transform
from scipy.spatial.transform import Rotation
from .smplh_motion import SMPLHMotion
from smpl_sim.smpllib.smpl_joint_names import SMPLH_BONE_ORDER_NAMES as NAMES

def observation_positions(joints, names):
    result=joints[:,[NAMES.index(n) for n in names]].clone()
    # SPL MID_HIP is a hip-center landmark, not the SMPL pelvis regressor center.
    if 'Pelvis' in names:
        result[:,names.index('Pelvis')]=(joints[:,NAMES.index('L_Hip')]+joints[:,NAMES.index('R_Hip')])/2
    return result

class JointFK(torch.nn.Module):
    """Exactly the kinematic-joint path of smplx LBS, without mesh skinning."""
    def __init__(self, parser):
        super().__init__()
        self.register_buffer('rest',parser.J_regressor@parser.v_template)
        self.register_buffer('shape',torch.einsum('jv,vck->jck',parser.J_regressor,parser.shapedirs))
        self.register_buffer('parents',parser.parents.clone())
    def forward(self, pose, beta, translation):
        rest=self.rest+torch.einsum('jck,k->jc',self.shape,beta)
        rotations=batch_rodrigues(pose.reshape(-1,3)).reshape(len(pose),52,3,3)
        joints,_=batch_rigid_transform(rotations,rest.expand(len(pose),-1,-1),self.parents)
        return joints+translation[:,None],rotations


def fit(trial, parser, iterations=800, include_fingers=False, fit_betas=True):
    keys,names,observations,confidence=trial.observations(include_fingers)
    if np.count_nonzero(confidence)==0:raise ValueError('No fitting observations')
    target=torch.tensor(np.nan_to_num(observations),dtype=torch.float32)
    weight=torch.tensor(confidence,dtype=torch.float32)
    # MID_HIP/NECK are approximate regressor-center matches, not independent exact markers.
    for k in ('MID_HIP','NECK'):
        if k in keys:weight[:,keys.index(k)]*=.5
    indices=[NAMES.index(n) for n in names];fk=JointFK(parser)
    count=len(target);active=156 if include_fingers else 66
    pose=torch.nn.Parameter(torch.zeros(count,active));beta=torch.nn.Parameter(torch.zeros(16),requires_grad=fit_betas)
    # Orthogonal hip/up frame initializes global orientation in the SMPL rest basis.
    p=trial.smpl_m;left=p[:,trial.names.index('LEFT_HIP')];right=p[:,trial.names.index('RIGHT_HIP')]
    root=p[:,trial.names.index('MID_HIP')];neck=p[:,trial.names.index('NECK')]
    if not np.isfinite(root).all():raise ValueError('Long pelvis gap: choose another clip')
    matrices=[]
    for t in range(count):
        x=left[t]-right[t];y=neck[t]-root[t]
        if not np.isfinite(x).all() or not np.isfinite(y).all():matrices.append(np.eye(3));continue
        x=x/max(np.linalg.norm(x),1e-8);y=y-x*np.dot(x,y);y=y/max(np.linalg.norm(y),1e-8)
        z=np.cross(x,y)
        if np.linalg.det(np.column_stack([x,y,z]))<.9:matrices.append(np.eye(3))
        else:matrices.append(np.column_stack([x,y,z]))
    with torch.no_grad():pose[:,:3]=torch.tensor(Rotation.from_matrix(matrices).as_rotvec(),dtype=torch.float32)
    def full():return torch.cat([pose,torch.zeros(count,156-active)],dim=1)
    with torch.no_grad():
        zero_joints=fk(full(),beta,torch.zeros(count,3))[0]
        center=(zero_joints[:,NAMES.index('L_Hip')]+zero_joints[:,NAMES.index('R_Hip')])/2
    translation=torch.nn.Parameter(torch.tensor(root,dtype=torch.float32)-center)
    with torch.no_grad():initial=fk(full(),beta,translation)[0].numpy()
    parameters=[pose,translation]+([beta] if fit_betas else [])
    opt=torch.optim.Adam(parameters,lr=.025);history=[];fps=trial.fps
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,iterations,eta_min=.0005)
    def objective():
        joints,r=fk(full(),beta,translation);err=observation_positions(joints,names)-target
        # Huber on coordinates: quadratic near the observation, robust beyond 5 cm.
        loss=torch.nn.functional.huber_loss(err,torch.zeros_like(err),delta=.05,reduction='none').sum(-1)
        joint=(loss*weight).sum()/weight.sum().clamp_min(1)
        velocity=((r[1:]-r[:-1])*fps).square().mean()
        acceleration=((r[2:]-2*r[1:-1]+r[:-2])*fps**2).square().mean()
        root_acc=((translation[2:]-2*translation[1:-1]+translation[:-2])*fps**2).square().mean()
        prior=pose[:,3:].square().mean()
        # Soft flexion sign/range constraints in this model's dominant anatomical axes.
        limit=torch.zeros(())
        for side,sign in [('L',-1),('R',1)]:
            knee=pose[:,3*NAMES.index(side+'_Knee')]
            elbow=sign*pose[:,3*NAMES.index(side+'_Elbow')+1]
            limit+=torch.relu(-knee).square().mean()+torch.relu(knee-2.6).square().mean()
            limit+=torch.relu(-elbow).square().mean()+torch.relu(elbow-2.8).square().mean()
        total=joint+1e-6*velocity+1e-9*acceleration+1e-6*root_acc+2e-5*prior+1e-5*beta.square().mean()+1e-4*limit
        return total,{'joint':joint,'velocity':velocity,'acceleration':acceleration,'root_acceleration':root_acc,'pose_prior':prior,'limits':limit}
    for i in range(iterations):
        opt.zero_grad();loss,parts=objective()
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite fitting loss')
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in parameters):raise RuntimeError('Nonfinite gradient')
        torch.nn.utils.clip_grad_norm_(parameters,1.);opt.step();scheduler.step()
        with torch.no_grad():beta.clamp_(-3,3)
        if i%50==0 or i==iterations-1:
            row={'iteration':i+1,'total':float(loss.detach()),**{k:float(v.detach()) for k,v in parts.items()}};history.append(row);print(row,flush=True)
    with torch.no_grad():
        poses=full().numpy();betas=beta.numpy();trans=translation.numpy();joints=fk(full(),beta,translation)[0].numpy()
    # Actual pelvis position: library transl != root joint position.
    motion=SMPLHMotion(poses,joints[:,0],trial.fps,'smpl','root_position')
    return {'motion':motion,'betas':betas,'translation':trans,'joints':joints,'initial_joints':initial,
            'keys':keys,'names':names,'observations':observations,'confidence':confidence,'history':history,'fk':fk}
