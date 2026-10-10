from pathlib import Path

import smplx
import torch
import trimesh

# 모델 디렉터리
MODEL_DIR = Path("assets/body_models")

# SMPL+H 모델 생성
model = smplx.create(
    model_path=str(MODEL_DIR),
    model_type="smplh",
    gender="male",
    ext="pkl",
    num_betas=16,
    use_pca=False,
)

# 기본 체형 파라미터
betas = torch.zeros(1, 16)

# 인체 모델 생성
with torch.no_grad():
    output = model(betas=betas)

vertices = output.vertices[0].cpu().numpy()
faces = model.faces

print("Vertices:", vertices.shape)
print("Faces:", faces.shape)
print("Joints:", output.joints.shape)

# 3D Mesh 저장
mesh = trimesh.Trimesh(
    vertices=vertices,
    faces=faces,
    process=False,
)

mesh.export("smplh_neutral.obj")

print("SMPL+H 모델 저장 완료!")
####
from pathlib import Path
import torch
import trimesh

# 저장 디렉터리
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

# 서로 다른 체형 4개
body_shapes = {
    "body_01": torch.zeros(1, 16),
    "body_02": torch.tensor([[2.0] + [0.0] * 15]),
    "body_03": torch.tensor([[-2.0] + [0.0] * 15]),
    "body_04": torch.tensor([[0.0, 2.0] + [0.0] * 14]),
}

for name, betas in body_shapes.items():

    with torch.no_grad():
        output = model(betas=betas)

    vertices = output.vertices[0].cpu().numpy()

    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=model.faces,
        process=False,
    )

    mesh.export(output_dir / f"{name}.obj")

    print(f"{name} 생성 완료")