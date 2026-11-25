"""What the app knows about each exercise: one module per exercise, named as the API names it
(constants.EXERCISES, frontend ExerciseType).

Everything else in app/result is the same for every exercise and asks the module of the job's exercise:

  MAIN_BONES            the bones the exercise is judged by (motion_scoring names). The scorer lines the two
                        clips up mostly by them (result_builder.scorer_setup) and a segment's deviations name
                        them first (result_builder.deviations)
  labeller(joints, names, fps) -> label(t0, t1, first, last)
                        the Korean name of the segment [t0, t1] s of that clip (result_builder.build_result)
  key_frames(body)      the frame of every repetition's key pose (squat: the bottoms)    | poseMetrics is
  rest_frames(body)     the frames of the pose the exercise rests in (squat: standing)   | measured at these
  moves(body, r)        the range r (s) holds a whole repetition's movement, not a hold  | (pose_metrics.py)
  PROMPT                what the LLM is told about the exercise: `deviations` (what a segment's deviations
                        are in it), `pose_metrics` (what poseMetrics holds), `rules` (what to look at first).
                        llm/feedback.py puts them into the instruction that is the same for every exercise

body is a pose_metrics.Body. motion_scoring itself knows no exercise: it is given the main bones as weights
and nothing else.

Adding an exercise is adding its module here. The API only accepts it once it is in
constants.SUPPORTED_EXERCISES as well.
"""
from . import squat

MODULES = {'squat': squat}


def get(exercise):
    """The module of an exercise."""
    if exercise not in MODULES:
        raise ValueError(f'no exercise module for {exercise!r} in app/result/exercises (there are: {sorted(MODULES)})')
    return MODULES[exercise]


def main_bones(exercise):
    """The main bones of an exercise; () for one without a module (the scorer then counts every bone alike)."""
    return MODULES[exercise].MAIN_BONES if exercise in MODULES else ()
