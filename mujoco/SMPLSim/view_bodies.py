from pathlib import Path
import trimesh

scene = trimesh.Scene()

for i in range(1, 5):
    path = Path(f"outputs/body_{i:02d}.obj")

    mesh = trimesh.load(path, force="mesh")

    # 인체들이 겹치지 않도록 X축으로 이동
    mesh.apply_translation([(i - 1) * 1.5, 0, 0])

    scene.add_geometry(mesh, node_name=f"body_{i:02d}")

scene.show()