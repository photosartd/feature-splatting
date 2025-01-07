from dataclasses import dataclass, field
from typing import Dict, Optional, Union
from collections import defaultdict
from enum import Enum
from functools import cached_property
import threading
import io

import numpy as np
from scipy.spatial.transform import Rotation
import torch
import trimesh
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.viewer.viewer import VISER_NERFSTUDIO_SCALE_RATIO
from viser import ViserServer, GlbHandle, SceneNodePointerEvent, MeshHandle
import viser.transforms as tf

from feature_splatting.db import SearchResult


@dataclass
class MeshBase:
    mesh_id: str
    transform: torch.Tensor = field(default_factory=lambda: torch.eye(4))
    visible: bool = True
    scale: float = 1.0

    @property
    def rotation(self) -> torch.Tensor:
        return self.transform[:3, :3]

    @property
    def position(self) -> torch.Tensor:
        return self.transform[:3, 3]
    
    @position.setter
    def position(self, position: Union[torch.Tensor, np.ndarray]):
        if not isinstance(position, torch.Tensor):
            position = torch.from_numpy(position)
        self.transform[:3, 3] = position
    
    @staticmethod
    def transform_from_position(position: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        transform = torch.eye(4)
        transform[:3, 3] = position
        return transform


@dataclass
class MeshSimpleData(MeshBase):
    vertices: np.ndarray = field(default_factory=lambda: np.array([]))
    faces: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class MeshData(MeshBase):
    """Stores mesh related data"""
    #mesh_id: str
    data: SearchResult = field(default_factory=SearchResult.empty())
    #transform: torch.Tensor = field(default_factory=torch.eye(4))
    #visible: bool = True
    #scale: float = 1.0

    @classmethod
    def from_search_result(cls, data: SearchResult, **kwargs):
        return cls(
            mesh_id=data.id_,
            data=data,
            **kwargs
        )
    
    def fit_bbox(self, bbox: Optional[torch.Tensor] = None) -> None:
        if bbox is None:
            return
        mesh_min, mesh_max = self.bounds / VISER_NERFSTUDIO_SCALE_RATIO
        bbox_min, bbox_max = bbox.numpy()
        # Compute the size of the mesh and the target bounding box
        mesh_size = mesh_max - mesh_min
        bbox_size = bbox_max - bbox_min
        print(f"Mesh size: {mesh_size}")
        print(f"Bbox size: {bbox_size}")

        # Calculate the scale factor (uniform scaling to fit the largest dimension)
        scale = min(bbox_size / mesh_size)

        # Calculate the center of the mesh and the target bounding box
        mesh_center = (mesh_max + mesh_min) / 2
        bbox_center = (bbox_max + bbox_min) / 2

        # TODO: 10 is hardcoded for now, will need to think on how to change it
        self.scale = scale

        # Calculate the translation to move the scaled mesh to the bounding box center
        position = bbox_center * VISER_NERFSTUDIO_SCALE_RATIO
        mesh_height = mesh_size[2] * VISER_NERFSTUDIO_SCALE_RATIO * self.scale
        z_translation_additional = torch.tensor([0.0, 0.0, -mesh_height / 2])

        print(f"Scale: {self.scale}")
        self.position = torch.from_numpy(position) + z_translation_additional
        #self.transform[:3, 3] = torch.zeros((3,))
        return

    @property
    def is_glb(self) -> bool:
        if self.data.is_root_set:
            return self.data.mesh_path.endswith(".glb")
        raise ValueError(f"self.data.root was None for {self.data.id_}")
    
    @cached_property
    def trimesh(self) -> trimesh.Trimesh:
        # TODO: fix of the file type handling
        mesh = trimesh.load(io.BytesIO(self.mesh_bytes), file_type='glb' if self.is_glb else "glb")

        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.to_geometry()
        return mesh
    
    @cached_property
    def bounds(self) -> np.ndarray:
        return self.trimesh.bounds
    
    @cached_property
    def mesh_bytes(self) -> bytes:
        if self.data.is_root_set:
            with open(self.data.mesh_path, "rb") as file_mesh:
                mesh_bytes = file_mesh.read()
            return mesh_bytes
        raise ValueError(f"self.data.root was None for {self.data.id_}")
    

class MeshSelectionMode(Enum):
    REPLACE = "replace"
    ADD = "add"


class MeshController:
    def __init__(self, viser_server: Optional[ViserServer] = None):
        self._viser_server = viser_server
        self.meshes: Dict[str, MeshData] = {}
        self.mesh_handles: Dict[str, GlbHandle] = {}
        self.same_mesh_counter: dict = defaultdict(int)
        self._lock = threading.Lock()

        self.selected_mesh_handle: Optional[GlbHandle] = None

    def get_mesh_id_key(self, mesh_id: str, add_new: bool = True) -> str:
        with self._lock:
            counter = self.same_mesh_counter[mesh_id]
            if add_new:
                self.same_mesh_counter[mesh_id] += 1
            return f"{mesh_id}_{counter}"
    
    def add_mesh_data(
            self,
            mesh: MeshBase,
            mode: MeshSelectionMode = MeshSelectionMode.REPLACE
        ) -> str:
        assert self._viser_server is not None, "ViserServer not set in MeshController"

        #with self._lock:
        if mode == MeshSelectionMode.REPLACE:
            self.remove_all()
        mesh_id_key = self.get_mesh_id_key(mesh.mesh_id)
        self.meshes[mesh_id_key] = mesh
        wxyz = np.roll(Rotation.from_matrix(mesh.transform[:3, :3].numpy()).as_quat(), shift=1)
        # TODO: hardcoded, idk why the orientations are different in viser and trimesh
        #wxyz = tf.SO3.from_x_radians(np.pi / 2).wxyz
        print(f"wxyz: {wxyz}")
        position = mesh.position.view(-1).numpy()
        print(f"Position: {position}")
        if isinstance(mesh, MeshData):
            mesh_handle: GlbHandle = self._viser_server.scene.add_glb(
                name=mesh_id_key,
                glb_data=mesh.mesh_bytes,
                scale=mesh.scale,
                wxyz=wxyz,
                position=position,
                visible=mesh.visible
            )
        elif isinstance(mesh, MeshSimpleData):
            mesh_handle: MeshHandle = self._viser_server.scene.add_mesh_simple(
                name=mesh_id_key,
                vertices=mesh.vertices,
                faces=mesh.faces,
                position=position,
                visible=mesh.visible
            )
        mesh_handle.on_click(self._on_mesh_clicked)
        self.mesh_handles[mesh_id_key] = mesh_handle

        return mesh_id_key

    def remove_by_id(self, mesh_id_key: str):
        with self._lock:
            self.meshes.pop(mesh_id_key, None)
            mesh_handle: GlbHandle = self.mesh_handles.pop(mesh_id_key, None)
            if mesh_handle is not None:
                mesh_handle.remove()
            

    def remove_all(self):
        with self._lock:
            for mesh_id_key, mesh in self.meshes.items():
                # self.viser_server.scene.remove_by_name(mesh_id_key)
                self.mesh_handles[mesh_id_key].remove()
            self.meshes.clear()
            self.same_mesh_counter.clear()
            self.mesh_handles.clear()

    def translate_position(self, translation: np.ndarray):
        with self._lock:
            for mesh_id_key, handle in self.mesh_handles.items():
                mesh = self.meshes[mesh_id_key]
                initial_position = mesh.position.numpy()
                handle.position = initial_position + np.array(translation) * VISER_NERFSTUDIO_SCALE_RATIO

    def translate_chosen_mesh(self, translation: np.ndarray):
        with self._lock:
            mesh = self.meshes[self.selected_mesh_handle._impl.name]
            initial_position = mesh.position.numpy()
            self.selected_mesh_handle.position = initial_position + np.array(translation) * VISER_NERFSTUDIO_SCALE_RATIO

    def set_position(self, new_position: np.ndarray):
        with self._lock:
            for _, handle in self.mesh_handles.items():
                handle.position = new_position

    def _on_mesh_clicked(self, event: SceneNodePointerEvent) -> None:
        self.selected_mesh_handle: GlbHandle = event.target

    def delete_selected_mesh(self):
        if self.selected_mesh_handle:
            self.remove_by_id(self.selected_mesh_handle._impl.name)
    
    @property
    def viser_server(self) -> ViserServer:
        assert self._viser_server is not None, "ViserServer not set in MeshController"
        return self._viser_server
    
    @viser_server.setter
    def viser_server(self, viser_server: ViserServer):
        self._viser_server = viser_server
