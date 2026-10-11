"""Isolated M4-style ball fixtures, NOT humanoid release or PPO scores."""
import json
import numpy as np,mujoco
from examples.m6.core import OUT,MODELS,ScoringDetector
from smpl_sim.physics.basketball import build_xml,test_ball_initial_state,RIM_CENTER,RIM_CENTERLINE_RADIUS,BOARD_FACE_X

def run():
 path=build_xml(MODELS/'body_01.xml',OUT/'detector_fixture_scene.xml');results=[]
 cases=[('center',RIM_CENTER+[0,0,.5],[0,0,0]),('outside',RIM_CENTER+[.5,0,.5],[0,0,0]),('rim',RIM_CENTER+[RIM_CENTERLINE_RADIUS,0,.5],[0,0,0]),('backboard',[BOARD_FACE_X-.7,0,3.8],[2.5,0,0]),('upward',RIM_CENTER-[0,0,.4],[0,0,4])]
 for name,pos,vel in cases:
  m=mujoco.MjModel.from_xml_path(str(path));d=mujoco.MjData(m);test_ball_initial_state(m,d,pos,vel);det=ScoringDetector();det.update(d.qpos[160:163],True);hits=set();score=False
  for _ in range(90 if name=='upward' else 900):
   mujoco.mj_step(m,d);mujoco.mj_forward(m,d);score|=det.update(d.qpos[160:163],True)
   for c in d.contact:
    if m.geom('basketball_geom').id in (c.geom1,c.geom2):hits.add(m.geom(int(c.geom2 if c.geom1==m.geom('basketball_geom').id else c.geom1)).name)
  results.append(dict(fixture=name,score=bool(score),contacts=sorted(hits),warnings=sum(int(w.number) for w in d.warning),scope='Initial test launcher fixture, excluded from reference/PPO score statistics'))
 (OUT/'detector_physical_fixtures.json').write_text(json.dumps(results,indent=2));print(results)
 if not results[0]['score'] or results[1]['score'] or not any(x.startswith('rim_') for x in results[2]['contacts']) or 'backboard' not in results[3]['contacts']:raise AssertionError('Fixture physics failure')
if __name__=='__main__':run()
