import os
import glob
import nibabel as nib
import numpy as np
from skimage import measure
import trimesh

# 根路徑
root_dir = 'datasets'
# 找出所有 pid_xx_gt.nii.gz 檔案
nii_paths = glob.glob(os.path.join(root_dir, '**', '*.nii.gz'), recursive=True)

for nii_path in nii_paths:
    # 讀取 NIfTI
    nii = nib.load(nii_path)
    vol = nii.get_fdata()

    # 二值化（可依需求調整門檻）
    mask = vol > 0.5

    # Marching Cubes 提取面片
    verts, faces, normals, _ = measure.marching_cubes(
        mask.astype(np.uint8),
        level=0,
        spacing=nii.header.get_zooms()
    )

    # 建立 Trimesh
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)

    # 準備輸出路徑：在 nii 檔同層下創 objs 資料夾
    parent_dir = os.path.dirname(nii_path)
    out_dir = os.path.join(parent_dir, 'objs')
    os.makedirs(out_dir, exist_ok=True)

    # 輸出檔名：原檔名去除路徑與副檔名，改為 .obj
    base_name = os.path.splitext(os.path.basename(nii_path))[0]
    out_path = os.path.join(out_dir, f'{base_name}.obj')

    # 寫出 OBJ
    mesh.export(out_path)
    print(f'Exported: {out_path}')

