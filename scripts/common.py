"""Shared helpers: axis references and the normalised Assistant score.

normalised score = (projection - role_mean) / (default - role_mean)
  1.0 = the published default Assistant vector, 0.0 = the mean of the published
  role vectors, both projected onto the axis at the target layer. Comparable
  across models whose raw projections live on very different scales.
"""

import glob
import os

import numpy as np
import torch
from huggingface_hub import snapshot_download

from assistant_axis import load_axis, project

AXIS_REPO = "lu-christina/assistant-axis-vectors"


def axis_dir(axis_folder):
    """Local path to the axis folder (default, axis, role_vectors), downloading only that folder."""
    patterns = [f"{axis_folder}/assistant_axis.pt", f"{axis_folder}/default_vector.pt",
                f"{axis_folder}/role_vectors/*"]
    try:
        root = snapshot_download(AXIS_REPO, repo_type="dataset", allow_patterns=patterns, local_files_only=True)
        if not glob.glob(os.path.join(root, axis_folder, "role_vectors", "*.pt")):
            raise FileNotFoundError
    except Exception:
        root = snapshot_download(AXIS_REPO, repo_type="dataset", allow_patterns=patterns)
    return os.path.join(root, axis_folder)


def axis_reference(axis_folder, layer, axis=None):
    """Projections of the published default vector and role vectors onto the axis at `layer`."""
    d = axis_dir(axis_folder)
    if axis is None:
        axis = load_axis(os.path.join(d, "assistant_axis.pt"))
    default = project(load_axis(os.path.join(d, "default_vector.pt")), axis, layer=layer)
    roles = []
    for f in sorted(glob.glob(os.path.join(d, "role_vectors", "*.pt"))):
        v = torch.load(f, map_location="cpu", weights_only=False)
        roles.append(project(v["vector"] if isinstance(v, dict) else v, axis, layer=layer))
    return {
        "axis_folder": axis_folder,
        "layer": layer,
        "default_projection": default,
        "role_mean_projection": float(np.mean(roles)),
        "n_roles": len(roles),
    }


def normalised_score(projection, ref):
    """1.0 = default Assistant, 0.0 = average role. Works on scalars or arrays."""
    return (np.asarray(projection, dtype=float) - ref["role_mean_projection"]) / (
        ref["default_projection"] - ref["role_mean_projection"])
