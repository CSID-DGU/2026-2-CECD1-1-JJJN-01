"""The exercise modules (app/result/exercises) against what the rest of the app asks of them. No GPU, no video.

  cd backend && ../.venv/bin/python verification/tests/test_exercises.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app import constants  # noqa: E402
from app.result import exercises, result_builder  # noqa: E402
from app.result.llm import feedback  # noqa: E402
from app.result.pose_metrics import Body  # noqa: E402

SCORING = constants.PIPELINE_REPO / 'pipeline/motion_scoring'
sys.path[:0] = [str(SCORING / 'validation'), str(SCORING)]            # synthetic.py, the motion_scoring package
import synthetic as syn  # noqa: E402
from motion_scoring.step1_preprocess import SMPL_NAMES  # noqa: E402

NEEDS = ('MAIN_BONES', 'labeller', 'key_frames', 'rest_frames', 'moves', 'PROMPT')
PROMPT_PARTS = ('missing_example', 'extra_example', 'rest_where', 'rest_labels', 'still',
                'deviations', 'pose_metrics', 'rules')


def check_modules():
    # every exercise the API accepts has a module, and every module is named as the API names the exercise
    assert set(constants.SUPPORTED_EXERCISES) <= set(exercises.MODULES) <= set(constants.EXERCISES), \
        (constants.SUPPORTED_EXERCISES, sorted(exercises.MODULES), constants.EXERCISES)
    bones = set(result_builder.FRONT_BONES)                           # the bones of motion_scoring
    for name, module in exercises.MODULES.items():
        missing = [n for n in NEEDS if not hasattr(module, n)]
        assert not missing, f'{name}: no {missing}'
        assert module.MAIN_BONES and set(module.MAIN_BONES) <= bones, (name, module.MAIN_BONES)
        assert exercises.get(name) is module and exercises.main_bones(name) == module.MAIN_BONES
        assert set(module.PROMPT) == set(PROMPT_PARTS), (name, sorted(module.PROMPT))
        prompt = feedback.system_prompt(name)
        for part in PROMPT_PARTS:
            assert module.PROMPT[part].strip() and module.PROMPT[part] in prompt, (name, part)
        # the two headings verification/experiments/llm_compare.py cuts the instruction at
        assert prompt.count('입력 설명\n') == 1 and prompt.count('\n규칙\n') == 1, name
        print(f'{name}: module ok, instruction of {len(prompt)} characters')
    assert feedback.SYSTEM_PROMPT == feedback.system_prompt('squat')
    # an exercise without a module: refused where a result is built; the scorer would count every bone alike
    for other in sorted(set(constants.EXERCISES) - set(exercises.MODULES)):
        assert exercises.main_bones(other) == ()
        try:
            exercises.get(other)
        except ValueError:
            print(f'{other}: no module yet, refused')
        else:
            raise AssertionError(f'{other} has no module but exercises.get() returned one')


def check_squat():
    """the squat module on a clip whose movements are known: 0.8 s standing, 3 x (1 s down, 1 s up) with
    0.6 s standing between them, 0.8 s standing"""
    squat = exercises.get('squat')
    names = list(SMPL_NAMES[:22])
    joints, fps = syn.skeleton(syn.phases(reps=3))
    body = Body(joints, names, fps)
    bottoms = squat.key_frames(body)
    assert len(bottoms) == 3 and all(abs(f / fps - (1.8 + 2.6 * k)) < 0.15 for k, f in enumerate(bottoms)), bottoms
    rest = set(squat.rest_frames(body).tolist())
    assert 0 in rest and len(joints) - 1 in rest and not rest & set(bottoms), sorted(rest)[:5]
    assert squat.moves(body, (0.8, 2.8), bottoms[0]) and not squat.moves(body, (0.0, 0.7), bottoms[0])   # a repetition / standing
    label = squat.labeller(joints, names, fps)
    got = [label(0.0, 0.7, True, False), label(0.8, 1.8, False, False), label(1.8, 2.8, False, False),
           label(0.8, 2.8, False, False), label(2.9, 3.3, False, False), label(8.1, 8.8, False, True)]
    assert got == ['시작 자세', '하강', '상승', '하강 · 상승', '선 자세 유지', '마무리 자세'], got
    joints, fps = syn.skeleton(syn.phases(reps=1, bottom=1.0))        # 1 s held at the bottom, from 1.8 s
    assert squat.labeller(joints, names, fps)(2.0, 2.6, False, False) == '바닥 유지'
    assert squat.key_frames(Body(syn.skeleton([(3.0, 0.0)])[0], names, fps)) == []     # nobody squats: no bottom
    print('squat: bottoms, standing, movement and segment names ok')


def deadlift_clip(ph, from_floor=False, lean=75., thigh=60., shank=15.):
    """A deadlift with the timing of the squat phases ph -> (joints [T, 22, 3] cm, fps): in the floor pose the
    trunk leans `lean` deg, the thighs point `thigh` deg back and the shanks tip `shank` deg forward; the arms
    hang straight down all the time. from_floor: the same movement turned round, so that the clip starts and
    ends in the floor pose and what the phases hold at the bottom is held at the lockout."""
    fps = 30.
    t = np.arange(int(round(sum(s for s, _ in ph) * fps)) + 1) / fps
    legs, trunk = syn._depth(ph, t), syn._depth(ph, t - 0.09)            # the trunk follows the legs
    if from_floor:
        legs, trunk = 1 - legs, 1 - trunk
    zero = np.zeros(len(t))
    xyz = lambda x, y, z: np.stack([x + zero, y + zero, z + zero], 1)
    sh, th, fw = np.radians(shank) * legs, np.radians(thigh) * legs, np.radians(lean) * trunk
    knee_y, knee_z = 8 + 42 * np.cos(sh), 42 * np.sin(sh)
    hip_y, hip_z = knee_y + 44 * np.cos(th), knee_z - 44 * np.sin(th)
    pelvis = xyz(0, hip_y + 5, hip_z)
    axis = xyz(0, np.cos(fw), np.sin(fw))                               # along the trunk
    neck = pelvis + 50 * axis
    J = dict(pelvis=pelvis, neck=neck, head=neck + 18 * xyz(0, np.cos(fw / 2), np.sin(fw / 2)))
    for k, f in enumerate((0.25, 0.5, 0.75), 1):
        J[f'spine{k}'] = pelvis + f * (neck - pelvis)
    for side, sx in (('left', 1.), ('right', -1.)):                     # the person's left is +X
        J[f'{side}_ankle'], J[f'{side}_foot'] = xyz(sx * 12, 8, 0), xyz(sx * 12, 2, 14)
        J[f'{side}_knee'], J[f'{side}_hip'] = xyz(sx * 12, knee_y, knee_z), xyz(sx * 9, hip_y, hip_z)
        J[f'{side}_collar'] = neck + xyz(sx * 7, 0, 0) - 3 * axis
        J[f'{side}_shoulder'] = neck + xyz(sx * 18, 0, 0) - 5 * axis
        J[f'{side}_elbow'] = J[f'{side}_shoulder'] + xyz(0, -28, 0)
        J[f'{side}_wrist'] = J[f'{side}_elbow'] + xyz(0, -26, 0)
    return np.stack([J[n] for n in SMPL_NAMES[:22]], 1), fps


def check_deadlift():
    """the deadlift module on clips whose movements are known, with the timing of the squat clip above"""
    deadlift = exercises.get('deadlift')
    names = list(SMPL_NAMES[:22])
    # from standing: 0.8 s standing, 3 x (1 s down to the bar, 1 s up) with 0.6 s standing between them, 0.8 s standing
    joints, fps = deadlift_clip(syn.phases(reps=3))
    body = Body(joints, names, fps)
    keys = deadlift.key_frames(body)
    assert len(keys) == 3 and all(abs(f / fps - (1.8 + 2.6 * k)) < 0.2 for k, f in enumerate(keys)), keys   # the turning points
    rest = set(deadlift.rest_frames(body).tolist())
    assert 0 in rest and len(joints) - 1 in rest and not rest & set(keys), sorted(rest)[:5]
    # the repetition from its key pose on is in the range / standing still / only the way down to the bar
    assert deadlift.moves(body, (0.8, 2.8), keys[0]) and not deadlift.moves(body, (0.0, 0.7), keys[0])
    assert not deadlift.moves(body, (0.8, 1.8), keys[0])
    label = deadlift.labeller(joints, names, fps)
    got = [label(0.0, 0.7, True, False), label(0.8, 1.8, False, False), label(1.8, 2.8, False, False),
           label(2.9, 3.3, False, False), label(3.4, 4.4, False, False), label(3.4, 5.4, False, False),
           label(8.1, 8.8, False, True)]
    assert got == ['시작 자세', '셋업', '리프트', '락아웃', '내리기', '내리기 · 리프트', '마무리 자세'], got
    # from the floor: 0.8 s in the floor pose, 3 x (1 s up, 0.5 s lockout, 1 s down) with 0.6 s at the floor between
    # them, 0.8 s at the floor. The three lifts start at 0.8, 3.9 and 7.0 s
    joints, fps = deadlift_clip(syn.phases(reps=3, bottom=0.5), from_floor=True)
    body = Body(joints, names, fps)
    keys = [f / fps for f in deadlift.key_frames(body)]            # where each floor pose ends, not the pose the clip ends in
    assert len(keys) == 3 and all(abs(k - t) < 0.2 for k, t in zip(keys, (0.8, 3.9, 7.0))), keys
    assert not set(deadlift.rest_frames(body).tolist()) & {0, len(joints) - 1}                            # the lockouts only
    label = deadlift.labeller(joints, names, fps)
    got = [label(0.0, 0.7, True, False), label(0.8, 1.8, False, False), label(1.9, 2.2, False, False),
           label(2.3, 3.3, False, False), label(3.4, 3.8, False, False), label(0.8, 3.3, False, False),
           label(9.6, 10.2, False, True)]
    assert got == ['셋업', '리프트', '락아웃', '내리기', '셋업', '리프트 · 내리기', '마무리 자세'], got
    # the same clip from the middle of the first lift on: the pose it starts in is not held, so it is no floor pose
    assert len(deadlift.key_frames(Body(joints[int(1.2 * fps):], names, fps))) == 2
    assert deadlift.key_frames(Body(deadlift_clip([(3.0, 0.0)])[0], names, fps)) == []     # nobody lifts: none
    print('deadlift: floor poses, lockout, movement and segment names ok')


def bench_clip(ph, flare=55., drop=15.):
    """A bench press with the timing of the squat phases ph -> (joints [T, 22, 3] cm, fps): lying on the back
    (head towards -Z, feet on the floor), the upper arms come down from upright to `drop` deg below the
    shoulder line, `flare` deg out from the line from the head to the feet; the forearms stay upright and
    nothing else moves."""
    fps = 30.
    t = np.arange(int(round(sum(s for s, _ in ph) * fps)) + 1) / fps
    q = syn._depth(ph, t)
    zero = np.zeros(len(t))
    xyz = lambda x, y, z: np.stack([x + zero, y + zero, z + zero], 1)
    rad = np.radians
    pelvis = xyz(0, 58, 0)
    neck = pelvis + 50 * xyz(0, np.sin(rad(8)), -np.cos(rad(8)))
    J = dict(pelvis=pelvis, neck=neck, head=neck + 18 * xyz(0, 0, -1))
    for k, f in enumerate((0.25, 0.5, 0.75), 1):
        J[f'spine{k}'] = pelvis + f * (neck - pelvis)
    for side, sx in (('left', 1.), ('right', -1.)):                     # the person's left is +X
        J[f'{side}_hip'] = xyz(sx * 9, 56, 2)
        J[f'{side}_knee'] = J[f'{side}_hip'] + 44 * xyz(sx * np.sin(rad(10)) * np.cos(rad(15)), -np.sin(rad(15)),
                                                        np.cos(rad(10)) * np.cos(rad(15)))
        J[f'{side}_ankle'] = J[f'{side}_knee'] + xyz(0, -42, 0)
        J[f'{side}_foot'] = J[f'{side}_ankle'] + xyz(0, -4, 13)
        J[f'{side}_collar'], J[f'{side}_shoulder'] = neck + xyz(sx * 7, 0, 3), neck + xyz(sx * 18, -1, 5)
        top = np.array([0., 1., 0.])
        low = np.array([sx * np.sin(rad(flare)) * np.cos(rad(drop)), -np.sin(rad(drop)), np.cos(rad(flare)) * np.cos(rad(drop))])
        om = np.arccos(top @ low)                                       # the upper arm turns from `top` to `low`
        upper = (np.sin((1 - q) * om)[:, None] * top + np.sin(q * om)[:, None] * low) / np.sin(om)
        J[f'{side}_elbow'] = J[f'{side}_shoulder'] + 28 * upper
        J[f'{side}_wrist'] = J[f'{side}_elbow'] + xyz(0, 26, 0)
    return np.stack([J[n] for n in SMPL_NAMES[:22]], 1), fps


def check_bench_press():
    """the bench-press module on clips whose movements are known, with the timing of the squat clip above"""
    bench = exercises.get('bench_press')
    names = list(SMPL_NAMES[:22])
    joints, fps = bench_clip(syn.phases(reps=3))
    body = Body(joints, names, fps)
    a = bench.arms(body)
    chest = bench.key_frames(body)
    assert len(chest) == 3 and all(abs(f / fps - (1.8 + 2.6 * k)) < 0.2 for k, f in enumerate(chest)), chest
    # the arms as they were put in: straight at the lockout; at the chest bent 90 + 15 deg and 55 deg out from the body
    assert a['elbow_left'][0] < 1 and abs(a['elbow_left'][chest[0]] - 105) < 2 and abs(a['flare_right'][chest[0]] - 55) < 2
    assert np.isnan(a['flare_left'][0]) and abs(a['arm'] - 54) < 0.5 and abs(a['hand'][0] - a['hand'][chest[0]] - 35.2) < 1
    rest = set(bench.rest_frames(body).tolist())
    assert 0 in rest and len(joints) - 1 in rest and not rest & set(chest), sorted(rest)[:5]
    assert bench.moves(body, (0.8, 2.8), chest[0]) and not bench.moves(body, (0.0, 0.7), chest[0])
    label = bench.labeller(joints, names, fps)
    got = [label(0.0, 0.7, True, False), label(0.8, 1.8, False, False), label(1.8, 2.8, False, False),
           label(0.8, 2.8, False, False), label(2.9, 3.3, False, False), label(8.1, 8.8, False, True)]
    assert got == ['시작 자세', '하강', '프레스', '하강 · 프레스', '락아웃', '마무리 자세'], got
    # the bar held at the chest for 1 s (from 1.8 s): named as such, and the key pose is where the hold ends
    held, fps = bench_clip(syn.phases(reps=1, bottom=1.0))
    assert bench.labeller(held, names, fps)(2.0, 2.6, False, False) == '가슴 터치'
    key = bench.key_frames(Body(held, names, fps))
    assert len(key) == 1 and abs(key[0] / fps - 2.7) < 0.2, key
    assert bench.key_frames(Body(bench_clip([(3.0, 0.0)])[0], names, fps)) == []          # nobody presses: none
    # the values the LLM gets: a user whose elbows flare 80 deg against the coach's 55, the repetitions as poseMetrics pairs them
    user = Body(bench_clip(syn.phases(reps=3), flare=80.)[0], names, fps)
    pm = dict(thresholds=dict(sameAngleDeg=8., sameHeightCm=3., sameKneeWidthRatio=0.15), repCount=dict(user=3, coach=3),
              reps=[dict(index=k + 1, userSec=f / fps, coachSec=f / fps, segmentId=None) for k, f in enumerate(chest)])
    m = bench.llm_metrics(user, body, pm, mirrored=False)
    assert set(m) == {'armLengthCm', 'thresholds', 'repCount', 'lockout', 'reps', 'chestSummary'} and len(m['reps']) == 3
    assert all(abs(r['chest']['diff']['elbowFlareDeg'][s] - 25) <= 2 for r in m['reps'] for s in ('left', 'right')), m['reps'][0]
    assert m['lockout']['user']['elbowFlareDeg'] == dict(left=None, right=None) and m['chestSummary']['reps'] == 3
    # lying down: the response does not name a difference towards the head or the feet, the LLM is told in those words
    bones = [dict(bone='l_upper_arm', student_bone='l_upper_arm', deg=25., dir=dict(up=0.2, forward=-23., out=9.9))]
    lying, = result_builder.deviations(bones, 'bench_press')
    standing, = result_builder.deviations(bones, 'squat')
    assert 'direction' not in lying and lying['components'] == standing['components'] and standing['direction'] == 'backward'
    tucked, = result_builder.deviations([dict(bones[0], dir=dict(up=0.4, forward=17., out=-18.6))], 'bench_press')
    assert tucked['direction'] == 'inward', tucked
    result = dict(exercise='bench_press', poseMetrics=dict(repCount=dict(user=3, coach=3)),
                  segments=[dict(id='seg-0', label='하강', verdict='match', deviations=[lying])])
    told = feedback.llm_input(result, dict(segments=[{}], exerciseMetrics=m))
    assert told['segments'][0]['deviations'][0]['direction'] == '머리 쪽으로' and told['poseMetrics'] is m, told['segments'][0]
    print('bench press: chest poses, lockout, arm values, segment names and directions ok')


def main():
    check_modules()
    check_squat()
    check_deadlift()
    check_bench_press()
    print('exercise modules ok')


if __name__ == '__main__':
    main()
