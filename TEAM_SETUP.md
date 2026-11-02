# kinematic-refinement 팀 실행 가이드

원본: [seokhyeonhong/kinematic-refinement](https://github.com/seokhyeonhong/kinematic-refinement) (SIGGRAPH Asia 2026, Hong et al.)

이 브랜치는 원본 코드를 **WSL2 + RTX 3060 12GB 환경에서 테스트가 끝까지 돌아가도록** 고친 버전입니다.
원본 README와 함께 읽어 주세요. 테스트 데이터와 출력 결과는 용량 문제로 이 저장소에 없고, 팀 드라이브로 공유합니다.

---

## 1. 검증된 환경

| 항목 | 버전 |
|---|---|
| OS | Windows 11 + WSL2 Ubuntu 22.04 |
| GPU | NVIDIA RTX 3060 12GB (드라이버는 **Windows에만** 설치) |
| Python | 3.8.20 (conda env `kinref`) |
| PyTorch | 2.4.1 + CUDA 12.1 (`torch 2.4.1+cu121`) |

> WSL 안에 리눅스용 NVIDIA 드라이버(`nvidia-driver-xxx`, `cuda-drivers`)를 설치하지 마세요. WSL은 Windows 드라이버를 그대로 사용합니다.

### WSL 메모리 (권장)

기본값(PC RAM의 50%)이면 CPU 쪽 작업에서 메모리 부족으로 프로세스가 조용히 종료될 수 있습니다.
Windows의 `%USERPROFILE%\.wslconfig`에 다음을 넣고 `wsl --shutdown` 후 재시작합니다. 값은 PC RAM에 맞게 조정하세요.

```ini
[wsl2]
memory=12GB
swap=16GB
```

## 2. 설치

```bash
git clone <팀 저장소 URL> kinematic-refinement
cd kinematic-refinement
conda create -n kinref python=3.8 -y
conda activate kinref
bash install.sh
```

또는 내보낸 환경 파일로 재현합니다.

```bash
conda env create -f environment.yml
conda activate kinref
```

설치 확인:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 3. 데이터 배치

팀 드라이브에서 아래 파일을 받아 저장소 루트에서 압축을 풉니다.

- 테스트 데이터: `test_geo_uc.zip` → `data/test_geo_uc/` (드라이브 링크: _여기에 기입_)
- (선택) 결과 BVH: `save_bvh.zip` → `src/save_bvh/` (드라이브 링크: _여기에 기입_)

배치 후 확인:

```bash
ls data/test_geo_uc            # pair.txt, motion/ 등이 보여야 함
wc -l data/test_geo_uc/pair.txt
```

체크포인트(`result/kin`, `result/geo`)는 원본 저장소에 포함되어 있습니다.

## 4. 실행

```bash
cd src
python -m kinref.geo_test \
  --model_epoch geo \
  --data_dir test_geo_uc/motion/processed \
  --device cuda:0 \
  --oom_frame_chunk_size 12 \
  2>&1 | tee ~/geo_test.log
```

- RTX 3060 12GB 기준 **청크 12에서 GPU 메모리 약 6GB, 전체 약 1시간**.
- GPU 메모리가 더 작으면 청크를 8 또는 4로 낮추세요.
- 실행 중 터미널을 닫거나 PC가 절전 모드에 들어가면 중단됩니다.

## 5. 출력

결과는 쌍마다 즉시 저장되므로, 중간에 멈춰도 끝난 쌍은 남습니다.

```
src/save_bvh/
├── Source/{소스}2{타깃}/{동작}.bvh   # 원본 소스 동작
├── Target/{소스}2{타깃}/{동작}.bvh   # 타깃 캐릭터 기준 동작
└── geo/{소스}2{타깃}/{동작}.bvh      # 모델의 리타게팅·보정 결과
```

예: `geo/Claire2Mutant/Bat_Swing.bvh` = Claire의 배트 스윙을 Mutant 체형에 옮긴 결과.
BVH는 뼈대 동작만 담고 있으므로 Blender의 `File → Import → Motion Capture (.bvh)`로 확인합니다.

완료 확인:

```bash
find src/save_bvh/geo -name "*.bvh" | wc -l   # pair.txt의 쌍 수와 같아야 함
grep -i -n "oom\|out of memory\|skip\|error" ~/geo_test.log | head
```

## 6. 원본 대비 수정 사항

| # | 파일 | 수정 | 이유 |
|---|---|---|---|
| 1 | `install.sh` 1행 | `cpip` → `pip` | 오타로 설치 실패 |
| 2 | `src/kinref/geo_test.py` (3곳) | `torch.inference_mode()` → `torch.enable_grad()` | Jacobian을 autograd로 계산하므로 그래디언트 비활성 상태에서는 동작 불가 |
| 3 | `src/kinref/geo_test.py` | `z, hatD = model(...)` → `_, _, hatD = model(...)` | `geo_model.forward`가 `(delta_z, delta_joint, hatD)` 3개를 반환 |
| 4 | `src/kinref/geo_test.py` | `run_retarget_with_oom_fallback` → `retarget_frame_wise` | 전체 프레임 일괄 처리 시 충돌 검사(`cdist`)의 메모리가 GPU 용량을 넘음. 처음부터 프레임 청크 단위로 처리 |

정확한 변경 내용은 `git log`와 `git diff upstream/main`으로 확인할 수 있습니다.

## 7. 문제 해결

| 증상 | 원인 | 대응 |
|---|---|---|
| `RuntimeError: CUDA error: unknown error` (`geo_utils.py`의 `compute_chamfer_normal_collision` 부근) | 청크가 커서 GPU 메모리 초과. WSL2에서는 OOM 메시지 대신 이 오류로 나타남 | `--oom_frame_chunk_size`를 낮춤 (12 → 8 → 4) |
| 에러 없이 프롬프트로 돌아옴 | 시스템 RAM 부족으로 OOM killer가 종료 (`dmesg \| grep -i "killed process"`로 확인) | `.wslconfig`로 WSL 메모리 증가, 청크 축소 |
| 매우 느림 (MI당 수 분) | 청크가 너무 작음 (예: 4) | 메모리가 허용하는 한 청크를 키움 |
| `CUDA_LAUNCH_BLOCKING=1`로만 되는 것처럼 보임 | 실제로는 그때 사용한 청크가 작아서 통과한 것 | 동기 모드 불필요. 청크 크기로 조절 |

참고: 청크 처리 시 원본의 OOM 대체 처리 경로를 거치지 않으므로, 메모리 문제로 실패한 쌍은 자동으로 재시도되지 않습니다. 5절의 완료 확인으로 빠진 쌍이 없는지 꼭 확인하세요.
