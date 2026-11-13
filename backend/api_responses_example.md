### POST /analysis (정상)
HTTP 202
{
 "jobId": "c24c88cb52ba"
}

### GET /analysis/c24c88cb52ba  (0s)
HTTP 200
{
 "jobId": "c24c88cb52ba",
 "status": "extracting",
 "progress": 0.0
}

### GET /analysis/c24c88cb52ba  (4s)
HTTP 200
{
 "jobId": "c24c88cb52ba",
 "status": "extracting",
 "progress": 0.04
}

### GET /analysis/c24c88cb52ba  (110s)
HTTP 200
{
 "jobId": "c24c88cb52ba",
 "status": "extracting",
 "progress": 0.6
}

### GET /analysis/c24c88cb52ba  (113s)
HTTP 200
{
 "jobId": "c24c88cb52ba",
 "status": "extracting",
 "progress": 0.64
}

### GET /analysis/c24c88cb52ba  (149s)
HTTP 200
{
 "jobId": "c24c88cb52ba",
 "status": "extracting",
 "progress": 0.8
}

### GET /analysis/c24c88cb52ba  (149s)
HTTP 200
{
 "jobId": "c24c88cb52ba",
 "status": "retargeting",
 "progress": 0.8
}

### GET /analysis/c24c88cb52ba  (154s)
HTTP 200
{
 "jobId": "c24c88cb52ba",
 "status": "done",
 "progress": 1.0,
 "result": {
  "jobId": "c24c88cb52ba",
  "exercise": "squat",
  "score": {
   "total": 0
  },
  "segments": [
   {
    "id": "seg-0",
    "label": "바닥 유지",
    "verdict": "missing",
    "deviations": [],
    "studentRange": [
     0.567,
     0.567
    ],
    "coachRange": [
     1.367,
     1.967
    ]
   }
  ],
  "student": {
   "jointNames": [
    "nose",
    "neck",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "pelvis",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle"
   ],
   "fps": 30.000003000000298,
   "frames": [
    [
     [
      0.0511,
      1.436,
      0.0717
     ],
     [
      0.0448,
      1.317,
      0.0345
     ],
     "... 관절 15개"
    ],
    "... 프레임 35개"
   ]
  },
  "coachRetargeted": {
   "jointNames": [
    "nose",
    "neck",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "pelvis",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle"
   ],
   "fps": 30.000003000000298,
   "frames": [
    [
     [
      0.0015,
      1.4262,
      0.0229
     ],
     [
      0.0007,
      1.3035,
      -0.0002
     ],
     "... 관절 15개"
    ],
    "... 프레임 88개"
   ]
  },
  "alignment": [
   [
    0,
    0
   ],
   [
    1,
    0
   ],
   [
    2,
    0
   ],
   "... 총 107쌍"
  ]
 }
}

### GET /analysis/3aa0b407ba99  (사람 없는 영상)
HTTP 200
{
 "jobId": "3aa0b407ba99",
 "status": "failed",
 "progress": 0.04,
 "error": "영상에서 사람을 찾지 못했어요. 전신이 3초 이상 나오게 다시 찍어 주세요."
}

### POST /analysis (데드리프트)
HTTP 400
{
 "message": "아직 스쿼트만 분석할 수 있어요."
}

### POST /analysis (영상 아님)
HTTP 400
{
 "message": "mp4, mov, webm 영상만 올릴 수 있어요."
}

### POST /analysis (video 빠짐)
HTTP 400
{
 "message": "요청에 빠진 항목이 있어요: video"
}

### GET /analysis/없는id
HTTP 404
{
 "message": "분석 작업을 찾을 수 없어요."
}