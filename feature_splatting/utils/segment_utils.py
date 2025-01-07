from typing import Tuple

import torch
from pytorch3d.ops import knn_points
import numpy as np
from scipy.spatial.transform import Rotation as R
from .math_utils import point_to_plane_distance, vector_angle

def cluster_instance(all_xyz_n3, selected_obj_idx=None, min_sample=20, eps=0.1):
    """
    Cluster points into instances using DBSCAN.
    Return the indices of the most populated cluster.
    """
    from sklearn.cluster import DBSCAN
    if selected_obj_idx is None:
        selected_obj_idx = np.ones(all_xyz_n3.shape[0], dtype=bool)
    dbscan = DBSCAN(eps=eps, min_samples=min_sample).fit(all_xyz_n3[selected_obj_idx])
    clustered_labels = dbscan.labels_

    # Find the most populated cluster
    label_idx_list, label_count_list = np.unique(clustered_labels, return_counts=True)
    # Filter out -1
    label_count_list = label_count_list[label_idx_list != -1]
    label_idx_list = label_idx_list[label_idx_list != -1]
    max_count_label = label_idx_list[np.argmax(label_count_list)]

    clustered_idx = np.zeros_like(selected_obj_idx, dtype=bool)
    # Double assignment to make sure indices go into the right place
    arr = clustered_idx[selected_obj_idx]
    arr[clustered_labels == max_count_label] = True
    clustered_idx[selected_obj_idx] = arr
    return clustered_idx

def estimate_ground(ground_pts, distance_threshold=0.005, rotation_flip=False):
    import open3d as o3d
    point_cloud = ground_pts.copy()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)

    plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                            ransac_n=3,
                                            num_iterations=2000)
    # [a, b, c, d] = plane_model
    # print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

    origin_plane_distance = point_to_plane_distance((0, 0, 0), plane_model)

    # Calculate rotation angle between plane normal & z-axis
    plane_normal = tuple(plane_model[:3])
    plane_normal = np.array(plane_normal) / np.linalg.norm(plane_normal)

    # Taichi uses y-axis as up-axis (OpenGL convention)
    if rotation_flip:
        # Sometimes the estimated plane normal is flipped
        y_axis = np.array((0, -1, 0))
    else:
        y_axis = np.array((0, 1, 0))  # Taichi uses y-axis as up-axis
    
    rotation_angle = vector_angle(plane_normal, y_axis)

    # Calculate rotation axis
    rotation_axis = np.cross(plane_normal, y_axis)
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

    # Generate axis-angle representation
    axis_angle = tuple([x * rotation_angle for x in rotation_axis])

    # Rotate point cloud
    rotation_object = R.from_rotvec(axis_angle)
    rotation_matrix = rotation_object.as_matrix()

    return (rotation_matrix, np.array((0, origin_plane_distance, 0)), inliers)

def estimate_plane(pc: np.ndarray, distance_threshold: float = 0.005, downsample_voxel_size: float = 0.05) -> Tuple[tuple, np.ndarray]:
    import open3d as o3d
    point_cloud = pc.copy()
    print(f"RANSAC pc start size: {point_cloud.shape}")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    downsampled_pcd = pcd.voxel_down_sample(voxel_size=downsample_voxel_size)
    print(f"RANSAC ps downsampled: {len(downsampled_pcd.points)}")

    #cl, ind = downsampled_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    inlier_cloud = downsampled_pcd#.select_by_index(ind)
    print(f"RANSAC ps inlier_cloud: {len(inlier_cloud.points)}")

    plane_model, inliers = inlier_cloud.segment_plane(distance_threshold=distance_threshold,
                                                  ransac_n=3,
                                                  num_iterations=2000)
    return plane_model, np.asarray(inlier_cloud.select_by_index(inliers).points)

def get_ground_bbox_min_max(all_xyz_n3, selected_obj_idx, ground_R, ground_T):
    """
    Select points within a bounding box.
    """
    particles = all_xyz_n3 @ ground_R.T # translates and rotates points to the ground CS
    particles += ground_T
    xyz_min = np.min(particles[selected_obj_idx], axis=0)
    xyz_max = np.max(particles[selected_obj_idx], axis=0)
    return xyz_min, xyz_max

def get_bbox_min_max(particles: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [torch.min(particles, dim=0)[0],
        torch.max(particles, dim=0)[0]]
    )

def knn_infilling(all_xyz_n3: torch.Tensor, obj_idx, k=50, dilation_iters=3, positive_ratio=0.8):
    obj_idx = obj_idx.copy()
    for _ in range(dilation_iters):
        non_fg_xyz = all_xyz_n3[~obj_idx]
        dists_nk, indices_nk = knn_points(all_xyz_n3[obj_idx][None], non_fg_xyz[None], K=k)
        dists_nk = dists_nk.squeeze(0)
        indices_nk = indices_nk.squeeze(0)
        positive_cnt = obj_idx[indices_nk].sum(axis=1) # for any point - how many of its neighbors are positive (relate to the object)
        non_fg_indices = torch.arange(all_xyz_n3.shape[0])[~obj_idx] # indices of non-fg points
        non_fg_indices = non_fg_indices[positive_cnt > int(k * positive_ratio)] # select points that have enough positive neighbors
        obj_idx[non_fg_indices] = True # dilate the object
    return obj_idx



