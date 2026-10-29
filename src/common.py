"""Small helpers shared by the pipeline stages."""
from pathlib import Path

import yaml


def load_params(path="params.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def dvc_md5(dvc_file):
    """Content hash of a DVC-tracked file, read from its .dvc metadata.

    This is the link between a trained model and the exact data version it saw.
    """
    path = Path(dvc_file)
    if not path.exists():
        return "unknown"
    meta = yaml.safe_load(path.read_text())
    return meta["outs"][0]["md5"]
