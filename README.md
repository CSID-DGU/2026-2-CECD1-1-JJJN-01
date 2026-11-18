# 2026-2-CECD1-1-JJJN-01

리타겟팅 기반 자세 비교 코칭 서비스. 사용자 영상과 코치 영상에서 스켈레톤을 뽑고,
코치 동작을 사용자 체형으로 리타겟팅한 뒤 CP-DTW 로 구간별 비교해서 점수 · 피드백 · 3D 오버레이를 보여준다.

## 폴더 구조

우리가 직접 수정하는 폴더만 적었다.

```
frontend/
└── src/

backend/
├── app/
│   ├── main.py
│   ├── constants.py
│   ├── jobs/
│   │   ├── job_queue.py
│   │   └── pipeline_runner.py
│   ├── result/
│   │   ├── result_builder.py
│   │   ├── pose_metrics.py
│   │   ├── frame_map.py
│   │   └── llm/
│   │       └── feedback.py
│   └── debug/
└── verification/
    ├── tests/
    └── experiments/

engine/
└── pipeline/
    ├── skeleton_extract/
    │   └── to_mixamo_format/
    ├── retarget/
    ├── common/
    ├── motion_scoring/
    ├── bvh_to_npz.py
    └── validation/
```

### backend

- `app/` — API 요청을 받는 인터페이스 (`main.py`)
- `app/jobs/` — 받은 요청으로 분석 함수를 호출하는 로직
- `app/result/` — 프론트에게 넘겨줄 데이터를 조합하는 곳

### engine

- `pipeline/` 폴더만 신경 쓰면 된다. 나머지는 대부분 PAN 원본 코드다.
- `pipeline/` 안은 폴더 이름으로 역할이 구분되게 해 두었다.

### 테스트 · 검증 코드

서비스 로직이 아닌 코드는 `debug/`, `verification/`, `validation/` 폴더로 따로 모아 분리했다.

### frontend

`frontend/README.md` 참고.
