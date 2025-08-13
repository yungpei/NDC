import trimesh
import numpy as np
import open3d as o3d
from scipy.spatial import KDTree as ScipyKDTree
from scipy.spatial.distance import directed_hausdorff
import json
import os
import pandas as pd  # 新增導入 pandas

# --- 0. 參數設定 ---
N_SAMPLING_POINTS_METRICS = 100000  # 指標計算用採樣點數
N_SAMPLING_POINTS_ICP = 5000        # ICP 對齊用採樣點數
ICP_THRESHOLD_FACTOR = 0.05         # ICP距離閾值因子（乘以模型尺寸）
F1_THRESHOLD_FACTOR = 0.01         # F1分數距離閾值因子

# 要比較的輸入格式
INPUT_FORMATS = ["sdf", "voxel"]
# 只比較 NDC 方法
METHOD_KEY = "ndc"
METHOD_NAME = "Neural Dual Contouring"
# Ground truth 來源
GT_TOOL = "3d_slicer"

# --- 1. 輔助函數 ---
def normalize_mesh(mesh):
    bounds = mesh.bounds
    center = (bounds[0] + bounds[1]) / 2.0
    scale = np.max(bounds[1] - bounds[0])
    mesh.apply_translation(-center)
    if scale > 0:
        mesh.apply_scale(1.0 / scale)
    return mesh, scale, center


def sample_points_and_normals(mesh, n_points):
    pts, face_idx = trimesh.sample.sample_surface(mesh, n_points)
    norms = mesh.face_normals[face_idx]
    return pts, norms


def calculate_chamfer_l2(gt_pts, pred_pts):
    tree_gt = ScipyKDTree(gt_pts)
    d1, _ = tree_gt.query(pred_pts)
    tree_pred = ScipyKDTree(pred_pts)
    d2, _ = tree_pred.query(gt_pts)
    return np.mean(d2**2) + np.mean(d1**2)


def calculate_f1(gt_pts, pred_pts, thr):
    tree_pred = ScipyKDTree(pred_pts)
    d_gt2pred, _ = tree_pred.query(gt_pts)
    recall = np.sum(d_gt2pred < thr) / len(gt_pts)
    tree_gt = ScipyKDTree(gt_pts)
    d_pred2gt, _ = tree_gt.query(pred_pts)
    precision = np.sum(d_pred2gt < thr) / len(pred_pts)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
    return f1, precision, recall


def calculate_normal_consistency(src_pts, src_norms, tgt_pts, tgt_norms):
    tree_tgt = ScipyKDTree(tgt_pts)
    _, idx = tree_tgt.query(src_pts)
    corresponding = tgt_norms[idx]
    return np.mean(np.abs(np.sum(src_norms * corresponding, axis=1)))


def mesh_to_pcd(mesh, n=None):
    if n:
        pts, _ = trimesh.sample.sample_surface(mesh, n)
    else:
        pts = mesh.vertices
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd


def align_icp(src_mesh, tgt_mesh, n_pts, thr_factor):
    src = mesh_to_pcd(src_mesh, n_pts)
    tgt = mesh_to_pcd(tgt_mesh, n_pts)
    reg = o3d.pipelines.registration.registration_icp(
        src, tgt, thr_factor, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
    )
    src_mesh.apply_transform(reg.transformation)
    return src_mesh

# --- 主流程: 只比較 NDC ---
all_results = {}
# 載入並歸一化 GT
gt_path = os.path.join("pid_02", f"3d_slicer.stl")
gt_mesh = trimesh.load_mesh(gt_path, process=True)
gt_mesh, _, _ = normalize_mesh(gt_mesh)
gt_pts, gt_norms = sample_points_and_normals(gt_mesh, N_SAMPLING_POINTS_METRICS)

for fmt in INPUT_FORMATS:
    print(f"\n=== Processing format: {fmt.upper()} for {METHOD_NAME} ===")
    # 載入模型
    model_file = f"pid_02/pid02_{METHOD_KEY}_{fmt}.obj"
    mesh = trimesh.load_mesh(model_file, process=True)
    mesh, _, _ = normalize_mesh(mesh)
    mesh = align_icp(mesh, gt_mesh, N_SAMPLING_POINTS_ICP, ICP_THRESHOLD_FACTOR)
    pts, norms = sample_points_and_normals(mesh, N_SAMPLING_POINTS_METRICS)

    # 計算指標
    cd = calculate_chamfer_l2(gt_pts, pts)
    hd1 = directed_hausdorff(gt_pts, pts)[0]
    hd2 = directed_hausdorff(pts, gt_pts)[0]
    hd = max(hd1, hd2)
    f1, prec, rec = calculate_f1(gt_pts, pts, F1_THRESHOLD_FACTOR)
    nc1 = calculate_normal_consistency(gt_pts, gt_norms, pts, norms)
    nc2 = calculate_normal_consistency(pts, norms, gt_pts, gt_norms)
    nc = (nc1 + nc2) / 2

    # 儲存結果
    all_results[fmt.upper()] = {
        "ChamferL2": cd,
        "Hausdorff_max": hd,
        "F1": f1,
        "Precision": prec,
        "Recall": rec,
        "NormalConsistency": nc
    }
    print(f"Results ({fmt}):", all_results[fmt.upper()])

# 寫入並讀取 JSON，產生 DataFrame 比較
out_json = "quant_results_ndc_sdf_voxel.json"
with open(out_json, "w") as jf:
    json.dump(all_results, jf, indent=2)
print(f"\nSaved results to {out_json}")

# 讀取並顯示比較表
data = pd.read_json(out_json)
data = data.T  # 行為格式、列為指標
print("\nComparison Table:")
print(data)