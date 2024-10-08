import argparse
import copy
import os

import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import load_checkpoint
from torchpack import distributed as dist
from torchpack.utils.config import configs
from tqdm import tqdm

from mmdet3d.core import LiDARInstance3DBoxes
from mmdet3d.core.utils import visualize_camera, visualize_camera_combo, visualize_lidar, visualize_lidar_combo, visualize_map
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model


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


def main() -> None:
    dist.init()

    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE")
    parser.add_argument("--mode", type=str, default="gt", choices=["gt", "pred", "combo"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--bbox-classes", nargs="+", type=int, default=None)
    parser.add_argument("--bbox-score", type=float, default=None)
    parser.add_argument("--map-score", type=float, default=0.5)
    parser.add_argument("--out-dir", type=str, default="viz")
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
    if args.mode == "pred" or args.mode == "combo":
        model = build_model(cfg.model)
        load_checkpoint(model, args.checkpoint, map_location="cpu")

        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
        )
        model.eval()

    obj_class_gt = [0] * 8
    obj_class_pred = [0] * 8

    for data in tqdm(dataflow):
        metas = data["metas"].data[0][0]
        # name = "{}".format(metas["timestamp"])
        # print("Lidar path: ", metas["lidar_path"])
        name = "{}".format(metas["lidar_path"].split("/")[-1].split(".")[0])
        # print("Save name:", name)

        pc_range = data["pc_range"].data[0][0][0].numpy().tolist()
        # print("visual pc_range", pc_range)

        if pc_range[1] == -60.0:
            pc_range = [0.0, -70.0, -10.0, 70.0, 0.0, -2.0]
        elif pc_range[1] == 0.0:
            pc_range = [0.0, 3.0, -10.0, 70.0, 73.0, -2.0]

        if args.mode == "pred" or args.mode == "combo":
            with torch.inference_mode():
                outputs = model(**data)

        if args.mode == "gt" and "gt_bboxes_3d" in data:
            bboxes = data["gt_bboxes_3d"].data[0][0].tensor.numpy()
            labels = data["gt_labels_3d"].data[0][0].numpy()

            if args.bbox_classes is not None:
                indices = np.isin(labels, args.bbox_classes)
                bboxes = bboxes[indices]
                labels = labels[indices]

            #bboxes[..., 2] -= bboxes[..., 5] / 2
            bboxes = LiDARInstance3DBoxes(bboxes, box_dim=9)
        elif args.mode == "pred" and "boxes_3d" in outputs[0]:
            bboxes = outputs[0]["boxes_3d"].tensor.numpy()
            scores = outputs[0]["scores_3d"].numpy()
            labels = outputs[0]["labels_3d"].numpy()

            if args.bbox_classes is not None:
                indices = np.isin(labels, args.bbox_classes)
                bboxes = bboxes[indices]
                scores = scores[indices]
                labels = labels[indices]

            if args.bbox_score is not None:
                indices = scores >= args.bbox_score
                bboxes = bboxes[indices]
                scores = scores[indices]
                labels = labels[indices]

            #bboxes[..., 2] -= bboxes[..., 5] / 2
            bboxes = LiDARInstance3DBoxes(bboxes, box_dim=9)
        elif args.mode == "combo" and "gt_bboxes_3d" in data and "boxes_3d" in outputs[0]:
            bboxes = outputs[0]["boxes_3d"].tensor.numpy()
            scores = outputs[0]["scores_3d"].numpy()
            labels = outputs[0]["labels_3d"].numpy()

            if args.bbox_classes is not None:
                indices = np.isin(labels, args.bbox_classes)
                bboxes = bboxes[indices]
                scores = scores[indices]
                labels = labels[indices]

            if args.bbox_score is not None:
                indices = scores >= args.bbox_score
                bboxes = bboxes[indices]
                scores = scores[indices]
                labels = labels[indices]
            
            #bboxes[..., 2] -= bboxes[..., 5] / 2
            bboxes = LiDARInstance3DBoxes(bboxes, box_dim=9)

            gtbboxes = data["gt_bboxes_3d"].data[0][0].tensor.numpy()
            gtlabels = data["gt_labels_3d"].data[0][0].numpy()

            if args.bbox_classes is not None:
                indices = np.isin(gtlabels, args.bbox_classes)
                gtbboxes = gtbboxes[indices]
                gtlabels = gtlabels[indices]

            #bboxes[..., 2] -= bboxes[..., 5] / 2
            gtbboxes = LiDARInstance3DBoxes(gtbboxes, box_dim=9)    
        else:
            bboxes = None
            labels = None

        if args.mode == "gt" and "gt_masks_bev" in data:
            masks = data["gt_masks_bev"].data[0].numpy()
            masks = masks.astype(bool)
        elif args.mode == "pred" and "masks_bev" in outputs[0]:
            masks = outputs[0]["masks_bev"].numpy()
            masks = masks >= args.map_score
        else:
            masks = None

        if "img" in data:
            for k, image_path in enumerate(metas["filename"]):
                image = mmcv.imread(image_path)
                if args.mode == "combo":
                    visualize_camera_combo(
                        os.path.join(args.out_dir, f"camera-{k}", f"cam_{name}.png"),
                        image,
                        gtbboxes=gtbboxes,
                        bboxes=bboxes,
                        gtlabels=gtlabels,
                        labels=labels,
                        transform=metas["lidar2image"][k],
                        classes=cfg.object_classes,
                    )
                else:
                    visualize_camera(
                        os.path.join(args.out_dir, f"camera-{k}", f"cam_{name}.png"),
                        image,
                        bboxes=bboxes,
                        labels=labels,
                        transform=metas["lidar2image"][k],
                        classes=cfg.object_classes,
                    )

        lidar = data["points"].data[0][0].numpy()

        if args.mode == "combo":
            obj_class_gt_ , obj_class_pred_ = visualize_lidar_combo(
                os.path.join(args.out_dir, "fused-lidar", f"{name}.png"),
                lidar,
                gtbboxes=gtbboxes,
                bboxes=bboxes,
                gtlabels=gtlabels,
                labels=labels,
                transform=metas["lidar2image"][0],
                xlim=[pc_range[d] for d in [0, 3]],
                ylim=[pc_range[d] for d in [1, 4]],
                classes=cfg.object_classes,
            )

            obj_class_gt[0] += obj_class_gt_[0]
            obj_class_gt[1] += obj_class_gt_[1]
            obj_class_gt[2] += obj_class_gt_[2]
            obj_class_gt[3] += obj_class_gt_[3]
            obj_class_gt[4] += obj_class_gt_[4]
            obj_class_gt[5] += obj_class_gt_[5]
            obj_class_gt[6] += obj_class_gt_[6]
            obj_class_gt[7] += obj_class_gt_[7]

            obj_class_pred[0] += obj_class_pred_[0]
            obj_class_pred[1] += obj_class_pred_[1]
            obj_class_pred[2] += obj_class_pred_[2]
            obj_class_pred[3] += obj_class_pred_[3]
            obj_class_pred[4] += obj_class_pred_[4]
            obj_class_pred[5] += obj_class_pred_[5]
            obj_class_pred[6] += obj_class_pred_[6]
            obj_class_pred[7] += obj_class_pred_[7]
            
        else:
            visualize_lidar(
                os.path.join(args.out_dir, "fused-lidar", f"{name}.png"),
                lidar,
                bboxes=bboxes,
                labels=labels,
                xlim=[pc_range[d] for d in [0, 3]],
                ylim=[pc_range[d] for d in [1, 4]],
                classes=cfg.object_classes,
            )

        if masks is not None:
            visualize_map(
                os.path.join(args.out_dir, "map", f"{name}.png"),
                masks,
                classes=cfg.map_classes,
            )

    print("GT: ", obj_class_gt)
    print("Pred: ", obj_class_pred)


if __name__ == "__main__":
    main()
