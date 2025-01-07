import os
import json
import zipfile

def unzip(zip_path, temp_dir) -> list:
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(temp_dir)

def metadata(temp_dir) -> str:
    with open(os.path.join(temp_dir, "metadata.json")) as f:
        return json.load(f)
    