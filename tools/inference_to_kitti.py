import argparse
import uuid
import copy
import os

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import load_checkpoint
from torchpack import distributed as dist
from torchpack.utils.config import configs
from tqdm import tqdm

from mmdet3d.core import LiDARInstance3DBoxes
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model

from pathlib import Path
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Any
from scipy.spatial.transform import Rotation as R
import open3d as o3d
import math
import json
import time

def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

next_detection_id = 0

def get_corners(yaw: float, width: float, length: float, position: np.ndarray) -> np.ndarray:
    # create the (normalized) perpendicular vectors
    v1 = np.array([np.cos(yaw), np.sin(yaw), 0])
    v2 = np.array([-v1[1], v1[0], 0])  # rotate by 90

    # scale them appropriately by the dimensions
    v1 *= length * 0.5
    v2 *= width * 0.5

    # flattened position
    pos = position.flatten()

    # return the corners by moving the center of the rectangle by the vectors
    return np.array(
        [
            pos + v1 + v2,
            pos - v1 + v2,
            pos - v1 - v2,
            pos + v1 - v2,
        ]
    )

def recursive_eval(obj, globals=None):
    if globals is None:
        globals = copy.deepcopy(obj)

    if isinstance(obj, dict):
        for key in obj:
            obj[key] = recursive_eval(obj[key], globals)
    elif isinstance(obj, list):
        for k, val in enumerate(obj):
            obj[k] = recursive_eval(val, globals)
    elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        obj = eval(obj[2:-1], globals)
        obj = recursive_eval(obj, globals)

    return obj

def kitti_line(cls_name, x, y, z, l, w, h, ry, score):
    # KITTI line (2D = 0, alpha = -10):
    # type trunc occl alpha  x1 y1 x2 y2  h w l  x y z  ry  score
    return (f"{cls_name} 0.00 0 -10.000000 "
            f"0.00 0.00 0.00 0.00 "
            f"{h:.6f} {w:.6f} {l:.6f} "
            f"{x:.6f} {y:.6f} {z:.6f} "
            f"{ry:.6f} {score:.6f}")

def pick_stem_from_metas(metas):
    # try to pick a reasonable stem from metas
    fn = metas.get("filename", None)
    # if isinstance(fn, (list, tuple)) and len(fn) > 0:
    #     return Path(fn[0]).stem          # '/.../L_...0001.jpg' -> 'L_...0001'
    # if isinstance(fn, str) and fn:
    #     return Path(fn).stem
    if "lidar_path" in metas:
        return Path(metas["lidar_path"]).stem  # '/.../xxx.bin' -> 'xxx'
    return "frame"

def main() -> None:
    dist.init()

    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--bbox-classes", nargs="+", type=int, default=None)
    parser.add_argument("--bbox-score", type=float, default=None)
    parser.add_argument("--out-dir", type=str, default="openlabel_out")
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)

    cfg = Config(recursive_eval(configs), filename=args.config)

    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    torch.cuda.set_device(dist.local_rank())

    # build the dataloader
    dataset = build_dataset(cfg.data[args.split])
    dataflow = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=True,
        shuffle=False,
    )

    # build the model and load checkpoint
    model = build_model(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location="cpu")

    model = MMDistributedDataParallel(
        model.cuda(),
        device_ids=[torch.cuda.current_device()],
        broadcast_buffers=False,
    )
    model.eval()

    time_sum = 0

    # class names (uppercase để khớp eval)
    CLASS_NAMES = [str(x).upper() for x in dataset.CLASSES]

    dt_annos = []

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    for data in tqdm(dataflow):

        pcd = o3d.geometry.PointCloud()
        o3dpoints = data["points"].data[0][0].cpu().numpy()
        pcd.points = o3d.utility.Vector3dVector(o3dpoints[:, 1:4])
        print(f"point cloud size: {len(pcd.points)} points")

        with torch.inference_mode():
            # torch.cuda.synchronize()
            t1 = time_synchronized()
            outputs = model(**data)
            t2 = time_synchronized()
            time_sum += t2 - t1

        out = outputs[0]

        print(out)

        boxes = out["boxes_3d"].tensor.detach().cpu().numpy()  # expect [N,7] = [x,y,z,l,w,h,ry] in CAMERA coords
        if boxes.shape[1] > 7:
            boxes = boxes[:, :7]
        scores = out["scores_3d"].detach().cpu().numpy()
        labels = out["labels_3d"].detach().cpu().numpy()

        if args.bbox_score is not None:
            keep = scores >= args.bbox_score
            boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

        # frame id
        metas = data["metas"].data[0][0]
        stem = pick_stem_from_metas(metas) 

        lines = []
        for bb, sc, lb in zip(boxes, scores, labels):

            # Filter by scores
            if sc < 0.1:
                continue
            x, y, z, l, w, h, ry = map(float, bb)
            # z += h / 2  # to center
            cls = CLASS_NAMES[int(lb)] if int(lb) < len(CLASS_NAMES) else str(int(lb))
            lines.append(kitti_line(cls, x, y, z, l, w, h, ry, float(sc)))

        out_path = out_dir / f"{stem}.txt"
        out_path.write_text("\n".join(lines))
        written.append(str(out_path))

    print(f"Average inference time: {time_sum / len(dataflow) * 1000:.2f} ms")
    print(f"Framerate: {1 / (time_sum / len(dataflow)):.2f} fps")

    (out_dir / "pred_list.txt").write_text("\n".join(written))
    print(f"[OK] wrote {len(written)} txt files to: {out_dir}")
    print(f"[OK] list saved to: {out_dir/'pred_list.txt'}")
    

if __name__ == "__main__":
    main()
