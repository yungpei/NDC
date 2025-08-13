import trimesh
import numpy as np
import open3d as o3d
from scipy.spatial import KDTree as ScipyKDTree
from scipy.spatial.distance import directed_hausdorff
import json

# --- 0. 參數設定 ---
N_SAMPLING_POINTS_METRICS = 100000  # 指標計算用採樣點數
N_SAMPLING_POINTS_ICP = 5000        # ICP 對齊用採樣點數
ICP_THRESHOLD_FACTOR = 0.05          # ICP距離閾值因子（乘以模型尺寸）
F1_THRESHOLD_FACTOR = 0.01           # F1分數距離閾值因子

# --- 1. 輔助函數 ---

def normalize_mesh(mesh):
    """
    對mesh做歸一化處理：
    - 將模型中心平移至原點
    - 縮放使模型最大邊長為1
    """
    # 計算邊界框大小和中心
    bounds = mesh.bounds  # (min_xyz, max_xyz)
    center = (bounds[0] + bounds[1]) / 2.0
    scale = np.max(bounds[1] - bounds[0])
    
    # 平移到中心點原點
    mesh.apply_translation(-center)
    # 縮放至邊長最大為1
    if scale > 0:
        mesh.apply_scale(1.0 / scale)
    
    return mesh, scale, center

def sample_points_and_normals(mesh, n_points):
    points, face_indices = trimesh.sample.sample_surface(mesh, n_points)
    normals = mesh.face_normals[face_indices]
    return points, normals

def calculate_chamfer_distance_l2(points_gt, points_pred):
    tree_gt = ScipyKDTree(points_gt)
    dist_pred_to_gt, _ = tree_gt.query(points_pred, k=1)

    tree_pred = ScipyKDTree(points_pred)
    dist_gt_to_pred, _ = tree_pred.query(points_gt, k=1)

    chamfer_dist = np.mean(dist_gt_to_pred**2) + np.mean(dist_pred_to_gt**2)
    return chamfer_dist

def calculate_f1_score(points_gt, points_pred, threshold):
    tree_pred = ScipyKDTree(points_pred)
    dist_gt_to_pred, _ = tree_pred.query(points_gt, k=1)
    recall_count = np.sum(dist_gt_to_pred < threshold)
    recall = recall_count / len(points_gt) if len(points_gt) > 0 else 0

    tree_gt = ScipyKDTree(points_gt)
    dist_pred_to_gt, _ = tree_gt.query(points_pred, k=1)
    precision_count = np.sum(dist_pred_to_gt < threshold)
    precision = precision_count / len(points_pred) if len(points_pred) > 0 else 0

    f1 = 0
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    return f1, precision, recall

def calculate_normal_consistency(points_src, normals_src, points_tgt, normals_tgt):
    tree_tgt = ScipyKDTree(points_tgt)
    distances, indices = tree_tgt.query(points_src, k=1)
    corresponding_normals_tgt = normals_tgt[indices]
    nc = np.mean(np.abs(np.sum(normals_src * corresponding_normals_tgt, axis=1)))
    return nc

def mesh_to_open3d_pointcloud(mesh, n_points=None):
    if n_points is None:
        points_np = mesh.vertices
    else:
        points_np, _ = trimesh.sample.sample_surface(mesh, n_points)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)
    return pcd

def align_mesh_to_gt_icp(mesh_to_align, gt_mesh, n_icp_points, icp_threshold_factor):
    print(f"Starting ICP alignment for {mesh_to_align.metadata.get('file_name', 'unknown mesh')}...")

    source_pcd = mesh_to_open3d_pointcloud(mesh_to_align, n_icp_points)
    target_pcd = mesh_to_open3d_pointcloud(gt_mesh, n_icp_points)

    threshold = icp_threshold_factor * 1.0  # 注意：這裡ICP閾值因子用1.0是因為mesh已經歸一化
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, threshold, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)
    )
    print("ICP Fitness:", reg_p2p.fitness)
    print("ICP Inlier RMSE:", reg_p2p.inlier_rmse)

    final_transformation = reg_p2p.transformation
    mesh_to_align.apply_transform(final_transformation)
    print("ICP alignment completed.")
    return mesh_to_align

# --- 2. 載入並歸一化模型 ---
print("Loading meshes...")
try:
    gt_slicer_stl_path = "pid_21/3d_slicer.stl"  # 修改為實際路徑
    ndc_obj_path = "pid_21/pid21_ndc_voxel.obj"
    nmc_obj_path = "pid_21/pid21_nmc_voxel.obj"
    vtk_stl_path = "pid_21/pid_21_vtk.stl"

    gt_mesh = trimesh.load_mesh(gt_slicer_stl_path, process=True)
    ndc_mesh_orig = trimesh.load_mesh(ndc_obj_path, process=True)
    nmc_mesh_orig = trimesh.load_mesh(nmc_obj_path, process=True)  # <-- 載入
    vtk_mesh_orig = trimesh.load_mesh(vtk_stl_path, process=True)
    print("Meshes loaded successfully.")
except Exception as e:
    print(f"Error loading meshes: {e}")
    exit()

# 保存檔名
gt_mesh.metadata['file_name'] = gt_slicer_stl_path
ndc_mesh_orig.metadata['file_name'] = ndc_obj_path
nmc_mesh_orig.metadata['file_name'] = nmc_obj_path             # <-- metadata
vtk_mesh_orig.metadata['file_name'] = vtk_stl_path

# 歸一化：移動到中心並縮放至單位大小
print("\nNormalizing meshes to unit scale...")
gt_mesh, gt_scale, gt_center = normalize_mesh(gt_mesh)
ndc_mesh_orig, ndc_scale, ndc_center = normalize_mesh(ndc_mesh_orig)
nmc_mesh_orig, nmc_scale, nmc_center = normalize_mesh(nmc_mesh_orig)          # <-- 歸一化 NMC
vtk_mesh_orig, vtk_scale, vtk_center = normalize_mesh(vtk_mesh_orig)
print(f"Ground Truth scale before normalization: {gt_scale:.6f}")
print(f"NDC mesh scale before normalization: {ndc_scale:.6f}")
print(f"NMC mesh scale before normalization: {ndc_scale:.6f}")
print(f"VTK mesh scale before normalization: {vtk_scale:.6f}")

# --- 3. ICP 對齊 ---
print("\nAligning meshes using ICP...")
ndc_mesh_aligned = ndc_mesh_orig.copy()
nmc_mesh_aligned = nmc_mesh_orig.copy()
vtk_mesh_aligned = vtk_mesh_orig.copy()

ndc_mesh_aligned = align_mesh_to_gt_icp(ndc_mesh_aligned, gt_mesh, N_SAMPLING_POINTS_ICP, ICP_THRESHOLD_FACTOR)
nmc_mesh_aligned = align_mesh_to_gt_icp(nmc_mesh_orig.copy(), gt_mesh, N_SAMPLING_POINTS_ICP, ICP_THRESHOLD_FACTOR)  # <-- 對齊 NMC
vtk_mesh_aligned = align_mesh_to_gt_icp(vtk_mesh_aligned, gt_mesh, N_SAMPLING_POINTS_ICP, ICP_THRESHOLD_FACTOR)

# --- 4. 採樣點雲 ---
print(f"\nSampling {N_SAMPLING_POINTS_METRICS} points for metric calculation...")
gt_points, gt_normals = sample_points_and_normals(gt_mesh, N_SAMPLING_POINTS_METRICS)
ndc_points, ndc_normals = sample_points_and_normals(ndc_mesh_aligned, N_SAMPLING_POINTS_METRICS)
nmc_points, nmc_normals = sample_points_and_normals(nmc_mesh_aligned, N_SAMPLING_POINTS_METRICS)  # <-- 採樣 NMC
vtk_points, vtk_normals = sample_points_and_normals(vtk_mesh_aligned, N_SAMPLING_POINTS_METRICS)
print("Sampling done.")

# --- 5. 計算評估指標 ---
results = {}
f1_actual_threshold = F1_THRESHOLD_FACTOR * 1.0  # 由於歸一化，尺度為1

models_for_evaluation = {
    "NDC_OBJ": (ndc_points, ndc_normals),
    "NMC_OBJ": (nmc_points, nmc_normals),   # <-- 新增這一行
    "VTK_STL": (vtk_points, vtk_normals)
}

for model_name, (pred_points, pred_normals) in models_for_evaluation.items():
    print(f"\n--- Evaluating {model_name} ---")

    cd_l2 = calculate_chamfer_distance_l2(gt_points, pred_points)
    print(f"Chamfer Distance (L2): {cd_l2:.6f}")

    hd_gt_to_pred = directed_hausdorff(gt_points, pred_points)[0]
    hd_pred_to_gt = directed_hausdorff(pred_points, gt_points)[0]
    hd_max = max(hd_gt_to_pred, hd_pred_to_gt)
    print(f"Hausdorff Distance (max): {hd_max:.6f}")

    f1, precision, recall = calculate_f1_score(gt_points, pred_points, f1_actual_threshold)
    print(f"F1-Score (threshold={f1_actual_threshold:.4f}): {f1:.4f} (Precision: {precision:.4f}, Recall: {recall:.4f})")

    nc_gt_to_pred = calculate_normal_consistency(gt_points, gt_normals, pred_points, pred_normals)
    nc_pred_to_gt = calculate_normal_consistency(pred_points, pred_normals, gt_points, gt_normals)
    nc_avg = (nc_gt_to_pred + nc_pred_to_gt) / 2.0
    print(f"Normal Consistency (avg): {nc_avg:.4f}")

    results[model_name] = {
        "ChamferL2": cd_l2,
        "Hausdorff_max": hd_max,
        "Hausdorff_GT_to_Pred": hd_gt_to_pred,
        "Hausdorff_Pred_to_GT": hd_pred_to_gt,
        "F1_Score": f1,
        "Precision": precision,
        "Recall": recall,
        "F1_Threshold": f1_actual_threshold,
        "NormalConsistency_avg": nc_avg,
        "NormalConsistency_GT_to_Pred": nc_gt_to_pred,
        "NormalConsistency_Pred_to_GT": nc_pred_to_gt
    }

# --- 6.
# （可選）把 results 存成 JSON
with open("quant_results.json", "w") as f:
    json.dump(results, f, indent=2)