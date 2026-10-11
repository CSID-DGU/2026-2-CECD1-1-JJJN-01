"""Close full-body M6 preview, actual wrist direction and ball trail."""
import argparse,time
import numpy as np,mujoco,mujoco.viewer
from PIL import Image,ImageDraw
from examples.m6.core import WeldShootingEnv
from examples.m4_basketball import trail
from contextlib import nullcontext

def run():
 p=argparse.ArgumentParser();p.add_argument('--states',required=True);p.add_argument('--reference',required=True);p.add_argument('--gif');p.add_argument('--gui',action='store_true');p.add_argument('--repeats',type=int,default=1);a=p.parse_args();e=WeldShootingEnv(reference=a.reference);s=np.load(a.states);cam=mujoco.MjvCamera();mujoco.mjv_defaultCamera(cam);cam.lookat[:]=[.25,0,1.05];cam.distance=3;cam.azimuth=110;cam.elevation=-10;frames=[];renderer=mujoco.Renderer(e.m,height=540,width=720) if a.gif else None
 def extras(scene,i):
  trail(scene,s['qpos'][:i+1:2,e.qa:e.qa+3]);w=e.d.xpos[e.wrist];finger=e.d.xpos[e.m.body('R_Middle1').id]-w;finger/=max(np.linalg.norm(finger),1e-9)
  if scene.ngeom<scene.maxgeom:
   g=scene.geoms[scene.ngeom];mujoco.mjv_initGeom(g,mujoco.mjtGeom.mjGEOM_ARROW,np.zeros(3),np.zeros(3),np.eye(3).ravel(),np.array([0,1,0,1],np.float32));mujoco.mjv_connector(g,mujoco.mjtGeom.mjGEOM_ARROW,.008,w,w+.20*finger);scene.ngeom+=1
 with mujoco.viewer.launch_passive(e.m,e.d) if a.gui else nullcontext(None) as v:
  if v:v.cam.lookat[:]=cam.lookat;v.cam.distance=cam.distance;v.cam.azimuth=cam.azimuth;v.cam.elevation=cam.elevation
  n=0
  while True:
   start=time.monotonic()
   for i,(t,q) in enumerate(zip(s['time_s'],s['qpos'])):
    if v and not v.is_running():break
    with v.lock() if v else nullcontext():
     e.d.qpos[:]=q;mujoco.mj_forward(e.m,e.d)
     if v:v.user_scn.ngeom=0;extras(v.user_scn,i)
     if renderer and i%2==0:
      renderer.update_scene(e.d,camera=cam);extras(renderer.scene,i);im=Image.fromarray(renderer.render());ImageDraw.Draw(im).text((12,12),f'Reversed wrist / synchronized extension / down snap | t={t:.2f}s | '+('WELD' if t<e.reference.release else 'RELEASED'),fill='yellow');frames.append(im.quantize(colors=256))
    if v:v.sync();time.sleep(max(0,t-s['time_s'][0]-(time.monotonic()-start)))
   n+=1
   if renderer or not v or not v.is_running() or (a.repeats and n>=a.repeats):break
 if renderer:renderer.close();frames[0].save(a.gif,save_all=True,append_images=frames[1:],duration=[70,60,70]*((len(frames)+2)//3),loop=0,optimize=False)
 e.close()
if __name__=='__main__':run()
