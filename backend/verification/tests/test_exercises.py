"""The exercise modules (app/result/exercises) against what the rest of the app asks of them. No GPU, no video.

  cd backend && ../.venv/bin/python verification/tests/test_exercises.py
"""
import sys
from pathlib import Path

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
PROMPT_PARTS = ('deviations', 'pose_metrics', 'rules')


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
    assert squat.moves(body, (0.8, 2.8)) and not squat.moves(body, (0.0, 0.7))      # a repetition / standing still
    label = squat.labeller(joints, names, fps)
    got = [label(0.0, 0.7, True, False), label(0.8, 1.8, False, False), label(1.8, 2.8, False, False),
           label(0.8, 2.8, False, False), label(2.9, 3.3, False, False), label(8.1, 8.8, False, True)]
    assert got == ['시작 자세', '하강', '상승', '하강 · 상승', '선 자세 유지', '마무리 자세'], got
    joints, fps = syn.skeleton(syn.phases(reps=1, bottom=1.0))        # 1 s held at the bottom, from 1.8 s
    assert squat.labeller(joints, names, fps)(2.0, 2.6, False, False) == '바닥 유지'
    assert squat.key_frames(Body(syn.skeleton([(3.0, 0.0)])[0], names, fps)) == []     # nobody squats: no bottom
    print('squat: bottoms, standing, movement and segment names ok')


def main():
    check_modules()
    check_squat()
    print('exercise modules ok')


if __name__ == '__main__':
    main()
