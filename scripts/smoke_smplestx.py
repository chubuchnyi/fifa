"""Smoke-test the SMPLest-X heavy recipe on the GPU box (build + one crop).

Mirrors smplestx_backend._load/_infer_crop exactly, standalone (no pitch3d import), so a
green run here proves the checkpoint remap, human-model files, and output dict the adapter
relies on. Run on the pod:  python scripts/smoke_smplestx.py
"""

import os
import sys
import tempfile

import numpy as np

REPO = os.environ.get("PITCH3D_SMPLESTX_REPO", "/workspace/repos/SMPLest-X")
CKPT = os.environ.get("PITCH3D_SMPLESTX_CKPT", "smplest_x_h")
DEVICE = os.environ.get("PITCH3D_DEVICE", "cuda")

sys.path.insert(0, REPO)
cwd = os.getcwd()
os.chdir(REPO)

import torch  # noqa: E402
import torchvision.transforms as transforms  # noqa: E402

from human_models.human_models import SMPLX  # noqa: E402
from main.base import Tester  # noqa: E402
from main.config import Config  # noqa: E402
from utils.data_utils import generate_patch_image, process_bbox  # noqa: E402

ckpt_dir = os.path.join(REPO, "pretrained_models", CKPT)
cfg = Config.load_config(os.path.join(ckpt_dir, "config_base.py"))
logdir = os.path.join(tempfile.gettempdir(), "pitch3d_smplestx_log")
cfg.update_config(
    {
        "model": {
            "pretrained_model_path": os.path.join(ckpt_dir, f"{CKPT}.pth.tar"),
            "human_model_path": os.path.join(REPO, "human_models", "human_model_files"),
        },
        "log": {
            "exp_name": "smoke",
            "output_dir": logdir,
            "model_dir": logdir,
            "log_dir": logdir,
            "result_dir": logdir,
        },
    }
)
cfg.prepare_log()
print("input_img_shape:", cfg.model.input_img_shape, "bbox_ratio:", getattr(cfg.data, "bbox_ratio", None))

SMPLX(cfg.model.human_model_path)
tester = Tester(cfg)
tester._make_model()
model = tester.model
to_tensor = transforms.ToTensor()

# synthetic 720x1280 RGB frame + a centred player-ish box
height, width = 720, 1280
frame = (np.random.rand(height, width, 3) * 255).astype(np.uint8)
x, y, w, h = 560, 200, 160, 360
bbox = process_bbox(
    bbox=np.array([x, y, w, h], dtype=np.float32),
    img_width=width,
    img_height=height,
    input_img_shape=cfg.model.input_img_shape,
    ratio=getattr(cfg.data, "bbox_ratio", 1.25),
)
print("processed bbox:", bbox)
patch, _, _ = generate_patch_image(
    cvimg=frame, bbox=bbox, scale=1.0, rot=0.0, do_flip=False, out_shape=cfg.model.input_img_shape
)
print("patch:", patch.shape, patch.dtype)
img = to_tensor(patch.astype(np.float32)) / 255
img = img.to(DEVICE)[None, :, :, :]
print("input tensor:", tuple(img.shape), img.dtype, img.device)

with torch.no_grad():
    out = model({"img": img}, {}, {}, "test")

for k in ("smplx_root_pose", "smplx_body_pose", "smplx_shape", "cam_trans", "smplx_mesh_cam"):
    v = out.get(k)
    print(f"  {k}:", None if v is None else tuple(v.shape))

print("SMOKE_OK")
os.chdir(cwd)
