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
    obj_idx = obj_idx.clone()
    for _ in range(dilation_iters):
        # Get non-object points
        non_fg_xyz = all_xyz_n3[~obj_idx]
        non_fg_indices = torch.nonzero(~obj_idx, as_tuple=False).squeeze(1)  # Indices of non-object points
        object_subset_indices = torch.nonzero(obj_idx, as_tuple=False).squeeze(1)
        
        # Check if there are non-object points left
        if non_fg_indices.numel() == 0 or not len(object_subset_indices):
            print("No non-object points left to process.")
            break
        
        # Perform KNN search
        dists_nk, indices_nk, _ = knn_points(all_xyz_n3[~obj_idx][None], all_xyz_n3[obj_idx][None], K=k)
        dists_nk = dists_nk.squeeze(0)
        indices_nk = indices_nk.squeeze(0)

        # Convert local indices to global indices
        global_indices_nk = object_subset_indices[indices_nk]

        # Count the number of positive neighbors
        positive_cnt = obj_idx[global_indices_nk].sum(dim=1)  # Count positive neighbors
        
        # Select non-object points with sufficient positive neighbors
        print(positive_cnt)
        eligible_mask = positive_cnt > int(k * positive_ratio)
        eligible_indices = non_fg_indices[eligible_mask]
        
        # -- Count how many are transitioning from False to True --
        old_positives = (obj_idx == True).sum().item()
        new_positives = (obj_idx[eligible_indices] == False).sum().item()
        print(f"Number of newly activated knn points this iteration: {new_positives}; percentage: {new_positives / old_positives}")
        
        # Update the object index mask
        obj_idx[eligible_indices] = True  # Dilate the object
    
    return obj_idx

def remove_ground(all_xyz_n3, selected_obj_idx, ground_R, ground_T, ground_level=0):
    """
    Remove points that are on the ground
    Should work both for numpy and torch tensors
    """
    particles = all_xyz_n3 @ ground_R.T # translates and rotates points to the ground CS
    particles += ground_T
    non_ground_idx = particles[:, 1] > ground_level
    return selected_obj_idx & non_ground_idx

def ground_bbox_filter(all_xyz_n3, selected_obj_idx, ground_R, ground_T, boundary):
    """Filters points within a bounding box on the ground
    Should be working both for numpy and torch tensors
    """
    particles = all_xyz_n3 @ ground_R.T
    particles += ground_T
    assert particles.shape[0] == selected_obj_idx.shape[0], f"{particles.shape[0]} != {selected_obj_idx.shape[0]}"
    selected_particles = particles[selected_obj_idx]
    print(f"selected_particles: {selected_particles.shape}")
    if hasattr(selected_particles, 'min'):  # PyTorch
        xyz_min = selected_particles.min(dim=0).values
        xyz_max = selected_particles.max(dim=0).values
        bbox_extent = xyz_max - xyz_min
        bbox_diagonal = torch.norm(bbox_extent)
    else:  # NumPy
        xyz_min = np.min(selected_particles, axis=0)
        xyz_max = np.max(selected_particles, axis=0)
        bbox_extent = xyz_max - xyz_min
        bbox_diagonal = np.linalg.norm(bbox_extent)
    xyz_min += (boundary * bbox_diagonal)
    xyz_max -= (boundary * bbox_diagonal)
    bbox_particles_idx = ((particles > xyz_min) & (particles < xyz_max)).all(dim=1 if hasattr(particles, 'dim') else 1)
    assert bbox_particles_idx.shape[0] == selected_obj_idx.shape[0]
    bbox_selected_particles = bbox_particles_idx | selected_obj_idx
    bbox_selected_particles = bbox_selected_particles.bool() if isinstance(bbox_selected_particles, torch.Tensor) else bbox_selected_particles.astype(bool)
    return bbox_selected_particles
