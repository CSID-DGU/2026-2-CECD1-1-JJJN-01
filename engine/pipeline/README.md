# pipeline/ — video-skeleton → pretrained PAN retargeting (work in progress)

## 실행 방법

환경(`.venv`, `.venv-pose`, `.venv-wham`, `third_party/WHAM`)은 `backend/README.md` 의 "환경 구성" 과
`scripts/setup_venvs.sh` 를 따른다. 외부 경로는 `pipeline/core/paths.py` 에 모여 있고 환경변수로 바꿀 수 있다.

```bash
cd engine
source ../.venv/bin/activate

# 실제 영상: 코치 → 사용자 체형 리타게팅 → 채점 (한 번에)
python -m pipeline.score_user --coach pipeline/data/squat_diag.mp4 --user pipeline/data/child_squat.mp4 --name coach_to_child
```

- `--coach`, `--user`: 영상 파일 또는 이미 처리된 run 폴더(`runs/video_wham/<이름>`).
  영상이면 WHAM → IK를 실행하고(영상당 약 2–4분), `runs/video_wham/<파일명>`이 이미 있으면 재사용합니다.
  60 fps·AV1 영상은 자동으로 H.264 30 fps로 변환됩니다.
- 결과
  - `runs/scoring/<name>/score.json` — 최종 채점 결과 (motion_scoring `Result.to_dict()`)
  - `runs/coach_user/<name>/` — `coach_on_user_pan.bvh`(리타게팅 결과), `user.bvh`(사용자 동작), `comparison.html`(시각화), `report.json`
  - `runs/video_wham/<영상>/` — 영상별 골격·동작(`source_person.bvh`), `report.json`
- 처음 보는 영상처럼 처리: `--rerun user`(사용자만), `--rerun coach`(코치만), `--rerun`(둘 다).
  WHAM 캐시(`wham_work/`, `video_wham.mp4`)까지 지우고 다시 처리합니다.
- 주요 옵션: `--compare-original`(원본 코치 기준 점수도 저장),
  `--ground clip`(점프 등 발이 뜨는 동작), `--no-mirror`, `--no-viz`
- 단계별 실행
  ```bash
  python -m pipeline.run_video pipeline/data/child_squat.mp4 --estimator wham --ground frame      # 영상 → 골격
  python -m pipeline.coach_to_user --coach runs/video_wham/squat_diag_gframe --user runs/video_wham/child_squat --name coach_to_child
  python -m pipeline.bvh_to_npz runs/coach_user/coach_to_child/user.bvh user.npz                   # BVH → 채점 입력
  ```
- 검증용 (실제 영상 아님, Mixamo 정답 데이터로 IK/PAN 정확도 측정)
  ```bash
  python tests/test_pipeline_core.py                                       # consistency checks
  python -m pipeline.harness --name <폴더이름> --max-clips 5 --levels L0 L1 --presets full   # → runs/harness/<폴더이름>/
  ```

Wraps the official PAN code without modifying it.

## Modules (`pipeline/core`)
| file | role |
|---|---|
| `paths.py` | repo / data locations; `PAN_TRAIN_NPY_DIR` env overrides the training npy dir |
| `quat.py` | quaternion ops (w,x,y,z), log map, twist/swing split, Horn rotation fit |
| `skeleton.py` | joint-space skeleton, FK (matches official FK), joint ↔ PAN raw conversion |
| `priors.py` | data-driven priors from the training npy → `assets/priors_<char>.json` |
| `ik.py` | constrained IK (`PRESETS`: free / init / limits / full) |
| `pan_runtime.py` | pretrained PAN loader + encoder→decoder inference, sign canonicalisation, OOD stats |

## Facts measured on the training data (BigVegas, 243 clips, 29 258 frames)
* edges sharing a parent joint always carry the same rotation → IK solves one rotation per joint
* `Hips` rotation is constant (identity) → pinned in `limits`/`full`
* knees/elbows: one dominant axis, off-axis median ≈ 8° (knee) / 3° (elbow) → soft limits, not hard hinges

## Harness (`pipeline/harness.py`)
Test motions (none in the training npy) give paired ground truth: `BigVegas/X.bvh` → `<target>/X.bvh`.
Observation levels: `L0` exact positions; `L0b` exact positions of a perturbed body (IK on the
person's own estimated skeleton); `L1` = `L0b` + noise (extra along the camera depth axis) + unobserved spine.
Outputs `runs/harness/<name>/{config.json,results.json,summary.json,summary.md,bvh/}`.

## Results so far (106 test clips, 4 unseen targets; PAN error = mean(dpos²)/height² ×1e3)
| run | level | condition | rot_deg | pan_err |
|---|---|---|---|---|
| full_v1 | – | original rotations (upper bound) | – | 0.369 |
| full_v1 | L0 | free (old unconstrained IK) | 27.4 | 1.639 |
| full_v1 | L0 | full | 7.2 | 0.372 |
| full_v1 | L1 | free | 33.6 | 5.287 |
| full_v1 | L1 | full | 14.0 | 1.651 |
| root_s0 | L1 | full_root | 12.5 | 1.085 |
| root_s15 | L1 | full_root + `--obs-sigma 1.5` | 9.8 | 0.464 |

Caveat: L1 noise is i.i.d. per frame, which a Gaussian filter removes unusually well; real
pose-estimator errors are temporally correlated, so expect a smaller gain on real input (L2).
Recommended defaults for real input: preset `full_root`, observation filter sigma ≈ 1.5 frames.

## L2: MediaPipe-like input through the real S1 code (`--levels L2 --obs-sigma 1.5`)
Root is optimised in every preset at L2 (no observed pelvis), so `full` == `full_root`.
`pan_err_al` = error after per-clip yaw+translation alignment (global frame is unknowable).

| run | condition | rot_deg | pan_err_al mean / median | gt (upper bound) mean |
|---|---|---|---|---|
| l2_diagonal | free | 36.7 | 3.21 / – | 0.345 |
| l2_diagonal | full | 17.9 | 2.03 / 0.58 | 0.345 |
| l2_side | full | 17.8 | 1.99 / – | 0.345 |

Diagonal, `full`: stationary clips (68, travel < 30 cm) median 0.51 vs 0.29 upper bound;
moving/jumping clips (38) median 1.00 — the foot-contact root reconstruction assumes one
foot on the ground and no foot sliding. `Standing To Crouch`: 0.51 (diag) / 0.58 (side) vs 0.26.

## Real video
`python -m pipeline.run_video VIDEO [--targets ...]` → `runs/video/<name>/`.
Known gap found on a real image (not visible in synthetic L2): MediaPipe hip landmarks are
~20 cm apart vs ~37 cm Mixamo hip joints → hip width is now measured (`core/person.py`).

## First real video: `pipeline/data/squat_side.mp4` (pure side view, 96 frames)
`python -m pipeline.run_video pipeline/data/squat_side.mp4 --name squat_side` then
`python -m pipeline.viz runs/video/squat_side --targets Abe_m Mousey_m Vampire_m`.
Fixes found on it (none visible in synthetic L2): shoulder width (BigVegas arm joints are
81 cm apart) now measured like hip width; occluded landmarks keep MediaPipe's estimate with a
low weight instead of being gap-filled from distant frames; toe weight 0.25.
PAN-input |z|>4 fraction 7.3% → 0.41%, IK fit 4.8 → 4.2 cm. L2 diagonal after the changes: 2.03 (unchanged).
Open: far-side arm while standing (occluded), toes below the floor and ~13–21 cm foot sliding
in the targets → needs S5 (foot-contact cleanup).

## WHAM as the pose estimator (`--estimator wham`, outputs in `runs/video_wham/`)
Setup: `.venv-wham` (repo root) (torch 2.8 cu128, mmcv 1.7.2, ViTPose/mmpose 0.24 with the
mmcv upper-bound check relaxed), WHAM in `third_party/WHAM` (repo root, or $WHAM_DIR), SMPL neutral model
supplied by the user, no DPVO (static camera). `tools/pose_estimation/wham_extract.py` allow-lists
the numpy types in WHAM's checkpoints for torch>=2.6's weights-only loader.
Mapping in `pipeline/core/wham_format.py`; S1 = resample + face +Z + `ground_align` (WHAM gives the trajectory).

| video | estimator | IK fit | PAN-input |z|>4 | source knee min R/L | target toe min | target foot slide |
|---|---|---|---|---|---|---|
| side | MediaPipe | 4.2 cm | 0.41 % | 70/88 | −27.7 cm | 5.9–22.0 cm |
| side | WHAM, all joints | 2.5 | 5.86 % | 78/76 | −13.4 | 5.6–11.6 |
| side | WHAM, default (no clavicle/Spine2/Neck) | 1.8 | 2.24 % | 77/76 | −13.5 | 5.2–11.5 |
| diag | MediaPipe | 4.8 | 0.81 % | 56/64 | −19.5 | 7.7–16.2 |
| diag | WHAM, all joints | 2.5 | 4.86 % | 61/63 | −14.0 | 4.4–7.5 |
| diag | WHAM, default | 1.9 | 0.70 % | 61/62 | −13.3 | 4.0–7.7 |

Fitting WHAM's clavicles (or chest + neck with the arms) forces BigVegas' clavicles ~50–60° off
PAN's training mean: the rest-pose geometry mismatch predicted earlier, now measurable because
more joints are observed. Default therefore skips LeftShoulder, RightShoulder, Spine2, Neck.

## Coach → user (same topology, PAN group-A decoder)
`python -m pipeline.coach_to_user --coach runs/video_wham/squat_diag --user runs/video_wham/<user>`
(or `--user-scale legs=1.15 arms=0.9` for a synthetic user) → `runs/coach_user/<name>/`
(`coach_on_user_pan.bvh` main output, `coach_on_user_copy.bvh` baseline, `user.bvh`, `comparison.html`).
Coach and user both come from `run_video`; the user's body = BigVegas topology + measured bone
lengths (`person_offsets`), no T-pose needed. `PanRuntime.retarget_intra` encodes the coach at
BigVegas scale and decodes with the user's offsets via `static_encoder`. A user has no mean/var:
BigVegas stats are used, root channels scaled by the **leg ratio** (rest hip height) — on Aj/Kaya/
SportyGranny it matches the real root stats, total height does not (Kaya 0.73 vs 0.88).

`python -m pipeline.intra_check --name v1`: BigVegas test clips (GT rotations) → Aj, Kaya,
SportyGranny with paired GT (these three test folders are needed for it). 106 clips:

| condition | pan_err×1e3 mean / median | rot_deg mean (median) |
|---|---|---|
| copy (rotations copied, root × leg ratio) | 1.57 / ≈0.40 | 3.24 (2.91) |
| pan_own (target's own mean/var, official-like) | 1.52 / ≈0.41 | 6.32 (6.03) |
| pan_user (BigVegas stats, root × leg ratio) | 1.87 / ≈0.66 | 7.46 (7.24) |

Findings: Mixamo's own same-topology GT is nearly a rotation copy (2.9° median), so this benchmark
cannot reward body-aware adaptation. On squat_diag with synthetic users (legs ×0.85 / ×1.0 /
×1.15 + arms ×0.9) the PAN rotations change by ≤0.2° on average (max 0.7°) with the skeleton:
the pretrained decoder does not adapt joint angles to bone lengths; its gap to the copy
(legs 2°, arms 14° on the WHAM input) is reconstruction error. Root/foot height do follow the
user (toe min ≈ 0.2–0.5 cm). Body-aware changes would need explicit constraints (foot contact,
end-effector targets) — not given by pretrained PAN.

### First real user: `pipeline/data/student_squat.mp4` (AV1 → `student_squat_h264.mp4`, WHAM's OpenCV cannot decode AV1)
* Body size is NOT recovered from monocular video: WHAM SMPL stature coach 174.6 / student 172.8 cm vs
  SMPL mean shape 171.7 cm (MediaPipe likewise similar). Bone lengths agree within 1–2 %; only
  shoulder width differs (student 0.83×, seen by both estimators). Absolute size needs an external
  reference (e.g. the user's real height).
* WHAM drifted ~15 cm vertically during the squat hold (feet floating) → `run_video --ground frame`
  (per-frame floor, smoothed; only for motions with a foot always on the ground). Runs:
  `runs/video_wham/{student_squat,squat_diag_gframe}`, result `runs/coach_user/coach_to_student`.

## Scoring (coach → user → motion_scoring)

```
python -m pipeline.score_user --coach <video|run dir> --user <video|run dir> [--name N] [--compare-original]
```
Video arguments run `run_video` (WHAM, `--ground frame`; AV1 transcoded) or reuse `runs/video_wham/<stem>`;
then `coach_to_user` (PAN) → `bvh_to_npz` (official BVH FK, BigVegas → SMPL-22 names, cm, Y up, 30 fps)
→ `motion_scoring` (real-video preset: auto tau, pose_trim 0.3, mirror check).
Output `runs/scoring/<name>/score.json` (+ `score_coach_original.json`, npz inputs, `inputs.json`).
Self-check: coach scored against itself gives tempo 1.0 and no pose labels.
Note: motion_scoring compares bone *directions* only, so bone lengths (the body part of the
retargeting) do not enter the score; only PAN's rotation changes do.

## WHAM speed (2026-11-07)
GPU here is a MIG 1g.24gb slice. Per run: ~15 s imports + ~80 s model loading (YOLOv8x, ViTPose-H,
HMR2 ViT-H) + ~0.19 s/frame ViTPose-H (YOLO 0.01 s) + HMR2 features (x2 with flip).
* `run_video.wham_input`: sources above 30 fps or not decodable by OpenCV (AV1) are converted to
  H.264 @ 30 fps (`<run>/video_wham.mp4`, = report `video`; `video_original` keeps the source).
  student_squat (60 fps AV1): same reprojection error / jitter as at 60 fps, half the frames.
* `wham_extract.skip_random_init`: the ViT-H models are built without random init (weights come
  from the checkpoints): outputs identical, ~30 s less. `--slow-init` restores it.
