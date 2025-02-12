from typing import Tuple

import numpy as np


class MeshUtils:
    @staticmethod
    def plane_mesh(plane_model: tuple, w: float, h: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns vertices and faces of the plane
        """
        a, b, c, d = plane_model
        normal = np.array([a, b, c])
        normal = normal / np.linalg.norm(normal)

        # Find a point on the plane: Set z=0 (or another axis) and solve for x, y
        if c != 0:  # Solve for z when c != 0
            point_on_plane = np.array([0, 0, -d / c])
        elif b != 0:  # Solve for y when b != 0
            point_on_plane = np.array([0, -d / b, 0])
        else:  # Solve for x when a != 0
            point_on_plane = np.array([-d / a, 0, 0])
        
        # Create two orthogonal vectors in the plane
        u = np.cross(normal, [1, 0, 0]) if abs(normal[0]) < 1 else np.cross(normal, [0, 1, 0])
        u = u / np.linalg.norm(u)  # Normalize
        v = np.cross(normal, u)    # Second orthogonal vector

        # Scale vectors to width and height
        u = u * (w / 2)
        v = v * (h / 2)

        # Generate the 4 corners of the rectangle
        vertices = np.array([
            point_on_plane - u - v,
            point_on_plane + u - v,
            point_on_plane + u + v,
            point_on_plane - u + v,
        ])

        # Define two triangular faces
        faces = np.array([
            [0, 1, 2],  # Triangle 1
            [2, 3, 0],  # Triangle 2
        ])

        return vertices, faces
