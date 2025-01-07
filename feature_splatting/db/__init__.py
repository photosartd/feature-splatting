import os
import json
from tqdm import tqdm
import tempfile
from typing import List, Optional
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import pandas as pd
from pymilvus import CollectionSchema, MilvusClient, FieldSchema, DataType
from pymilvus.milvus_client.index import IndexParams

from feature_splatting.db.feature_extractor import FeatureExtractor
from feature_splatting.db.utils import unzip, metadata


def find_files_with_partial_name(root_dir: str, partial_name: str) -> List[str]:
    root_path = Path(root_dir)
    return [str(file) for file in root_path.rglob(f"{partial_name}*")]  # Recursive glob with partial name


@dataclass
class SearchResult:
    id_: str
    save_uid: str
    distance: float
    metadata: dict
    root: Optional[str] = None

    @cached_property
    def mesh_path(self, root: Optional[str] = None) -> str:
        if root is None and self.root is None:
            raise ValueError(f"root was {root}")
        elif root is None and self.root is not None:
            root = self.root
        return os.path.join(root, find_files_with_partial_name(root, self.id_)[0])
    
    @property
    def name(self) -> str:
        return self.metadata.get("name", "")
    
    @classmethod
    def from_result(cls, result: dict) -> "SearchResult":
        id_ = result["id"]
        entity = result["entity"]
        save_uid = entity["save_uid"]
        metadata = entity["metadata"]
        distance = result["distance"]

        return cls(id_, save_uid, distance, metadata)
    
    @classmethod
    def empty(cls) -> "SearchResult":
        return cls(
            "",
            "",
            0.0,
            {},
            ""
        )
    
    def set_root(self, root: str) -> None:
        self.root = root

    @property
    def is_root_set(self) -> bool:
        return self.root is not None
    
    @staticmethod
    def from_results(results: List[dict]) -> List["SearchResult"]:
        return [SearchResult.from_result(result) for result in results]
    
    def __str__(self):
        return f"{self.name}; dist: {round(self.distance, 3)}"

class GaussianSimilaritySearchWrapper(MilvusClient):
    VECTOR_FIELD_NAME: str = "embedding"
    EMB_DIM: int = 768
    METRIC: str = "COSINE"

    def create_collection(
            self,
            collection_name: str,
            dimension: int = EMB_DIM,
            auto_id: bool = False,
            schema: Optional[CollectionSchema] = None,
            index_params: Optional[IndexParams] = None,
            **kwargs
    ):
        return super().create_collection(
            collection_name,
            dimension=dimension,
            vector_field_name=self.VECTOR_FIELD_NAME,
            metric_type=self.METRIC,
            auto_id=auto_id,
            schema=schema,
            index_params=index_params,
            enable_dynamic_field=True,
            **kwargs
        )
    
    def create_collection_from_schema(self, collection_name: str, schema: CollectionSchema):
        index_params = IndexParams(field_name=self.VECTOR_FIELD_NAME, metric_type=self.METRIC)
        return self.create_collection(collection_name, schema=schema, index_params=index_params)
    
    def search_meshes(self, collection_name: str, data: List[list], limit: int = 10, output_fields: list = ["save_uid", "metadata"]) -> List[List[SearchResult]]:
        results = self.search(collection_name, data, limit=limit, metric_type=self.METRIC, output_fields=output_fields)
        return [SearchResult.from_results(res) for res in results]
    
    @staticmethod
    def objaverse_schema() -> CollectionSchema:
        id_ = FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=256)
        emb = FieldSchema(name=GaussianSimilaritySearchWrapper.VECTOR_FIELD_NAME, dtype=DataType.FLOAT_VECTOR, dim=GaussianSimilaritySearchWrapper.EMB_DIM)
        save_uid = FieldSchema(name="save_uid", dtype=DataType.VARCHAR, max_length=256)
        metadata = FieldSchema(name="metadata", dtype=DataType.JSON)

        return CollectionSchema(fields=[id_, emb, save_uid, metadata])
    
    @staticmethod
    def embed_objaverse(
        collection_name: str,
        feature_extractor: FeatureExtractor,
        client: MilvusClient,
        paths: list,
        metadata_json: dict
    ) -> list:
        missing_ids = []

        for mesh_render_zip in tqdm(paths):
            with tempfile.TemporaryDirectory() as temp_dir:
                unzip(mesh_render_zip, temp_dir)
                folder = os.path.join(temp_dir, os.listdir(temp_dir)[0])
                emb = feature_extractor(folder)
                meta = metadata(folder)
                id_ = meta["file_identifier"].split("/")[-1]
                save_uid = meta["save_uid"]

                try:
                    client.insert(
                        collection_name,
                        {
                            "id": id_,
                            GaussianSimilaritySearchWrapper.VECTOR_FIELD_NAME: emb,
                            "save_uid": save_uid,
                            "metadata": metadata_json.get(id_, None)
                        }
                    )
                except Exception as e:
                    missing_ids.append((save_uid, str(e)))
                    print(e)
        
        return missing_ids
        

if __name__ == "__main__":
    db_path = "/vol/isy-rl/dtrofimov/data/databases/milvus"
    db_name = "objaverse-gauss.db"
    db_pathname = os.path.join(db_path, db_name)
    clip_model_name = "ViT-L/14@336px"
    collection_name = "subset_788_fsp_clip2"
    
    with open("/vol/isy-rl/dtrofimov/projects/langs/lanGS/notebooks/sampled_classes.json", "r") as f:
        sampled_classes = json.load(f)

    render_root = "/vol/isy-rl/dtrofimov/data/objaverse-renders/renders/"
    meshes_rendered_zips = list(
        map(
            lambda zipfile: os.path.join(render_root, zipfile),
            filter(lambda f: f.endswith(".zip"), os.listdir(render_root))
        )
    )

    db = GaussianSimilaritySearchWrapper(uri=db_pathname)
    db.create_collection_from_schema(collection_name, GaussianSimilaritySearchWrapper.objaverse_schema())

    feature_extractor = FeatureExtractor(clip_model_name)

    not_embedded = GaussianSimilaritySearchWrapper.embed_objaverse(
        collection_name,
        feature_extractor,
        db,
        meshes_rendered_zips,
        sampled_classes
    )

    print(f"Number of missing ids: {len(not_embedded)}")
    print(f"Stats: {db.get_collection_stats(collection_name)}")
