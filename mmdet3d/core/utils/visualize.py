import copy
import os
from typing import List, Optional, Tuple

import cv2
import mmcv
import numpy as np
from matplotlib import pyplot as plt

from ..bbox import LiDARInstance3DBoxes

__all__ = ["visualize_camera", "visualize_camera_combo", "visualize_lidar", "visualize_lidar_combo", "visualize_lidar_two", "visualize_lidar_two_combo", "visualize_map"]


OBJECT_PALETTE = {
    "CAR": (0, 255, 255),
    "TRAILER": (128, 128, 128),
    "TRUCK": (128, 255, 0),
    "VAN": (255, 128, 0),
    "PEDESTRIAN": (255, 0, 255),
    "BUS": (255, 0, 128),
    "MOTORCYCLE": (128, 0, 255),
    "OTHER": (199, 199, 199),
    "BICYCLE": (255, 128, 0),
    "EMERGENCY_VEHICLE": (102, 107, 250)
}

MAP_PALETTE = {
    "drivable_area": (166, 206, 227),
    "road_segment": (31, 120, 180),
    "road_block": (178, 223, 138),
    "lane": (51, 160, 44),
    "ped_crossing": (251, 154, 153),
    "walkway": (227, 26, 28),
    "stop_line": (253, 191, 111),
    "carpark_area": (255, 127, 0),
    "road_divider": (202, 178, 214),
    "lane_divider": (106, 61, 154),
    "divider": (106, 61, 154),
}

def visualize_camera(
    fpath: str,
    image: np.ndarray,
    *,
    bboxes: Optional[LiDARInstance3DBoxes] = None,
    labels: Optional[np.ndarray] = None,
    transform: Optional[np.ndarray] = None,
    classes: Optional[List[str]] = None,
    color: Optional[Tuple[int, int, int]] = None,
    thickness: float = 4,
) -> None:
    canvas = image.copy()
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

    if bboxes is not None and len(bboxes) > 0:
        corners = bboxes.corners
        num_bboxes = corners.shape[0]

        coords = np.concatenate(
            [corners.reshape(-1, 3), np.ones((num_bboxes * 8, 1))], axis=-1
        )
        
        if transform.size < 16:
            transform = np.append(transform, [[0.0, 0.0, 0.0, 1.0]], axis=0)
        
        transform = copy.deepcopy(transform).reshape(4, 4)
        coords = coords @ transform.T
        coords = coords.reshape(-1, 8, 4)

        indices = np.all(coords[..., 2] > 0, axis=1)
        coords = coords[indices]
        labels = labels[indices]

        indices = np.argsort(-np.min(coords[..., 2], axis=1))
        coords = coords[indices]
        labels = labels[indices]

        coords = coords.reshape(-1, 4)
        coords[:, 2] = np.clip(coords[:, 2], a_min=1e-5, a_max=1e5)
        coords[:, 0] /= coords[:, 2]
        coords[:, 1] /= coords[:, 2]

        coords = coords[..., :2].reshape(-1, 8, 2)
        for index in range(coords.shape[0]):
            name = classes[labels[index]]
            for start, end in [
                (0, 1),
                (0, 3),
                (0, 4),
                (1, 2),
                (1, 5),
                (3, 2),
                (3, 7),
                (4, 5),
                (4, 7),
                (2, 6),
                (5, 6),
                (6, 7),
            ]:
                cv2.line(
                    canvas,
                    coords[index, start].astype(int),
                    coords[index, end].astype(int),
                    color or OBJECT_PALETTE[name],
                    thickness,
                    cv2.LINE_AA,
                )
        canvas = canvas.astype(np.uint8)
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    mmcv.mkdir_or_exist(os.path.dirname(fpath))
    mmcv.imwrite(canvas, fpath)

def visualize_camera_combo(
    fpath: str,
    image: np.ndarray,
    *,
    gtbboxes: Optional[LiDARInstance3DBoxes] = None,
    bboxes: Optional[LiDARInstance3DBoxes] = None,
    gtlabels: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    transform: Optional[np.ndarray] = None,
    classes: Optional[List[str]] = None,
    color: Optional[Tuple[int, int, int]] = None,
    thickness: float = 1,
) -> None:
    canvas = image.copy()
    # canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

    if gtbboxes is not None and len(gtbboxes) > 0:
        corners = gtbboxes.corners
        num_bboxes = corners.shape[0]

        coords = np.concatenate(
            [corners.reshape(-1, 3), np.ones((num_bboxes * 8, 1))], axis=-1
        )
        
        if transform.size < 16:
            transform = np.append(transform, [[0.0, 0.0, 0.0, 1.0]], axis=0)
        
        transform = copy.deepcopy(transform).reshape(4, 4)
        coords = coords @ transform.T
        coords = coords.reshape(-1, 8, 4)

        indices = np.all(coords[..., 2] > 0, axis=1)
        coords = coords[indices]
        gtlabels = gtlabels[indices]

        indices = np.argsort(-np.min(coords[..., 2], axis=1))
        coords = coords[indices]
        gtlabels = gtlabels[indices]

        coords = coords.reshape(-1, 4)
        coords[:, 2] = np.clip(coords[:, 2], a_min=1e-5, a_max=1e5)
        coords[:, 0] /= coords[:, 2]
        coords[:, 1] /= coords[:, 2]

        coords = coords[..., :2].reshape(-1, 8, 2)
        
        for index in range(coords.shape[0]):
            name = classes[gtlabels[index]]

            # print("coords", coords[index])

            count = 0
            for coord in coords[index]:
                if not( 0 <= coord[0] <= 1920 and 0 <= coord[1] <= 1200):
                    count += 1
            
            # print("count", count)
            
            if count < 2:
                for start, end in [
                    (0, 1),
                    (0, 3),
                    (0, 4),
                    (1, 2),
                    (1, 5),
                    (3, 2),
                    (3, 7),
                    (4, 5),
                    (4, 7),
                    (2, 6),
                    (5, 6),
                    (6, 7),
                ]:
                    if start == 4 and end == 7:
                        cv2.line(
                            canvas,
                            coords[index, start].astype(int),
                            coords[index, end].astype(int),
                            (0, 0, 255),
                            thickness,
                            cv2.LINE_AA,
                        )
                    else:
                        cv2.line(
                            canvas,
                            coords[index, start].astype(int),
                            coords[index, end].astype(int),
                            (255, 255, 255),
                            thickness,
                            cv2.LINE_AA,
                        )
        canvas = canvas.astype(np.uint8)

    if bboxes is not None and len(bboxes) > 0:
        corners = bboxes.corners
        num_bboxes = corners.shape[0]

        coords = np.concatenate(
            [corners.reshape(-1, 3), np.ones((num_bboxes * 8, 1))], axis=-1
        )
        
        if transform.size < 16:
            transform = np.append(transform, [[0.0, 0.0, 0.0, 1.0]], axis=0)
        
        transform = copy.deepcopy(transform).reshape(4, 4)
        coords = coords @ transform.T
        coords = coords.reshape(-1, 8, 4)

        # print("coords image 1", coords)

        indices = np.all(coords[..., 2] > 0, axis=1)
        coords = coords[indices]
        labels = labels[indices]

        # print("coords image 2", coords)

        indices = np.argsort(-np.min(coords[..., 2], axis=1))
        coords = coords[indices]
        labels = labels[indices]

        # print("coords image 3", coords)

        coords = coords.reshape(-1, 4)
        coords[:, 2] = np.clip(coords[:, 2], a_min=1e-5, a_max=1e5)
        coords[:, 0] /= coords[:, 2]
        coords[:, 1] /= coords[:, 2]

        coords = coords[..., :2].reshape(-1, 8, 2)
    
        for index in range(coords.shape[0]):
            name = classes[labels[index]]
            # print("coords", coords[index])

            count = 0
            for coord in coords[index]:
                if not( 0 <= coord[0] <= 1920 and 0 <= coord[1] <= 1200):
                    count += 1
            
            # print("count", count)
            if count < 2:
                for start, end in [
                    (0, 1),
                    (0, 3),
                    (0, 4),
                    (1, 2),
                    (1, 5),
                    (3, 2),
                    (3, 7),
                    (4, 5),
                    (4, 7),
                    (2, 6),
                    (5, 6),
                    (6, 7),
                ]:
                    if start == 4 and end == 7:
                        cv2.line(
                            canvas,
                            coords[index, start].astype(int),
                            coords[index, end].astype(int),
                            (0, 0, 255),
                            thickness,
                            cv2.LINE_AA,
                        )
                    else:
                        cv2.line(
                            canvas,
                            coords[index, start].astype(int),
                            coords[index, end].astype(int),
                            color or OBJECT_PALETTE[name],
                            thickness,
                            cv2.LINE_AA,
                        )
        canvas = canvas.astype(np.uint8)
    # canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    mmcv.mkdir_or_exist(os.path.dirname(fpath))
    mmcv.imwrite(canvas, fpath)


def visualize_lidar(
    fpath: str,
    lidar: Optional[np.ndarray] = None,
    *,
    bboxes: Optional[LiDARInstance3DBoxes] = None,
    labels: Optional[np.ndarray] = None,
    classes: Optional[List[str]] = None,
    xlim: Tuple[float, float] = (-50, 50),
    ylim: Tuple[float, float] = (-50, 50),
    color: Optional[Tuple[int, int, int]] = None,
    radius: float = 15,
    thickness: float = 25,
) -> None:
    fig = plt.figure(figsize=(xlim[1] - xlim[0], ylim[1] - ylim[0]))

    ax = plt.gca()
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect(1)
    ax.set_axis_off()

    if lidar is not None:
        plt.scatter(
            lidar[:, 0],
            lidar[:, 1],
            s=radius,
            c="white",
        )

    if bboxes is not None and len(bboxes) > 0:
        coords = bboxes.corners[:, [0, 3, 7, 4, 0], :2]
        for index in range(coords.shape[0]):
            name = classes[labels[index]]
            front_x = (coords[index, 2, 0] + coords[index, 3, 0]) / 2
            front_y = (coords[index, 2, 1] + coords[index, 3, 1]) / 2
            plt.plot(
                coords[index, :, 0],
                coords[index, :, 1],
                linewidth=thickness,
                color=np.array(color or OBJECT_PALETTE[name]) / 255,
            )
            plt.plot(int(front_x), int(front_y), 'ro')

    mmcv.mkdir_or_exist(os.path.dirname(fpath))
    fig.savefig(
        fpath,
        dpi=10,
        facecolor="black",
        format="png",
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()

def visualize_lidar_combo(
    fpath: str,
    lidar: Optional[np.ndarray] = None,
    *,
    gtbboxes: Optional[LiDARInstance3DBoxes] = None,
    bboxes: Optional[LiDARInstance3DBoxes] = None,
    gtlabels: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    transform: Optional[np.ndarray] = None,
    classes: Optional[List[str]] = None,
    xlim: Tuple[float, float] = (-50, 50),
    ylim: Tuple[float, float] = (-50, 50),
    color: Optional[Tuple[int, int, int]] = None,
    radius: float = 15,
    thickness: float = 25,
) -> None:
    fig = plt.figure(figsize=(xlim[1] - xlim[0], ylim[1] - ylim[0]))

    ax = plt.gca()
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect(1)
    ax.set_axis_off()

    if lidar is not None:
        plt.scatter(
            lidar[:, 0],
            lidar[:, 1],
            s=radius,
            c="white",
        )

    obj_class_gt = [0] * 8
    obj_class_pred = [0] * 8
    
    if gtbboxes is not None and len(gtbboxes) > 0:

        coords = gtbboxes.corners[:, [0, 3, 7, 4, 0], :2]
        for index in range(coords.shape[0]):

            if transform.size < 16:
                transform = np.append(transform, [[0.0, 0.0, 0.0, 1.0]], axis=0)
        
            transform = copy.deepcopy(transform).reshape(4, 4)
            # print("transform", transform)
            # print("coords", gtbboxes.corners[index].reshape(-1, 3))
            coords_ = np.concatenate([gtbboxes.corners[index].reshape(-1, 3), np.ones((1 * 8, 1))], axis=-1) @ transform.T
            coords_ = coords_.reshape(-1, 8, 4)
            # print("coords_", coords_)

            # print("coords image 1", coords_)

            indices = np.all(coords_[..., 2] > 0, axis=1)
            coords_ = coords_[indices]
            # labels = labels[indices]

            # print("coords image 2", coords_)

            indices = np.argsort(-np.min(coords_[..., 2], axis=1))
            coords_ = coords_[indices]
            # labels = labels[indices]

            # print("coords image 3", coords_)

            coords_ = coords_.reshape(-1, 4)
            coords_[:, 2] = np.clip(coords_[:, 2], a_min=1e-5, a_max=1e5)
            coords_[:, 0] /= coords_[:, 2]
            coords_[:, 1] /= coords_[:, 2]

            coords_ = coords_[..., :2].reshape(-1, 8, 2)

            count = 0
            for coord in coords_:
                if not( 0 <= coord[0][0] <= 1920 and 0 <= coord[0][1] <= 1200):
                    count += 1
            
            # print("count", count)

            if count < 3:
                name = classes[gtlabels[index]]

                plt.plot(
                    coords[index, :, 0],
                    coords[index, :, 1],
                    linewidth=thickness,
                    color=np.array((255, 255, 255)) / 255,
                )
                plt.plot([coords[index, 2, 0], coords[index, 3, 0]],
                            [coords[index, 2, 1], coords[index, 3, 1]],
                            linewidth=thickness, 
                            color=np.array((255, 0, 0)) / 255)

                if name == "CAR":
                    obj_class_gt[0] += 1
                elif name == "TRUCK":
                    obj_class_gt[1] += 1
                elif name == "TRAILER":
                    obj_class_gt[2] += 1
                elif name == "VAN":
                    obj_class_gt[3] += 1
                elif name == "BUS":
                    obj_class_gt[4] += 1
                elif name == "MOTORCYCLE":
                    obj_class_gt[5] += 1
                elif name == "PEDESTRIAN":
                    obj_class_gt[6] += 1
                elif name == "BICYCLE":
                    obj_class_gt[7] += 1


    if bboxes is not None and len(bboxes) > 0:
        coords = bboxes.corners[:, [0, 3, 7, 4, 0], :2]
        for index in range(coords.shape[0]):

            if transform.size < 16:
                transform = np.append(transform, [[0.0, 0.0, 0.0, 1.0]], axis=0)
        
            transform = copy.deepcopy(transform).reshape(4, 4)
            # print("transform", transform)
            # print("coords", bboxes.corners[index].reshape(-1, 3))
            coords_ = np.concatenate([bboxes.corners[index].reshape(-1, 3), np.ones((1 * 8, 1))], axis=-1) @ transform.T
            coords_ = coords_.reshape(-1, 8, 4)
            # print("coords_", coords_)

            # print("coords image 1", coords_)

            indices = np.all(coords_[..., 2] > 0, axis=1)
            coords_ = coords_[indices]
            # labels = labels[indices]

            # print("coords image 2", coords_)

            indices = np.argsort(-np.min(coords_[..., 2], axis=1))
            coords_ = coords_[indices]
            # labels = labels[indices]

            # print("coords image 3", coords_)

            coords_ = coords_.reshape(-1, 4)
            coords_[:, 2] = np.clip(coords_[:, 2], a_min=1e-5, a_max=1e5)
            coords_[:, 0] /= coords_[:, 2]
            coords_[:, 1] /= coords_[:, 2]

            coords_ = coords_[..., :2].reshape(-1, 8, 2)

            count = 0
            for coord in coords_:
                if not( 0 <= coord[0][0] <= 1920 and 0 <= coord[0][1] <= 1200):
                    count += 1
            
            # print("count", count)

            if count < 3:
                name = classes[labels[index]]

                # print("coords", coords[index])
                plt.plot(
                    coords[index, :, 0],
                    coords[index, :, 1],
                    linewidth=thickness,
                    color=np.array(color or OBJECT_PALETTE[name]) / 255,
                )
                plt.plot([coords[index, 2, 0], coords[index, 3, 0]],
                         [coords[index, 2, 1], coords[index, 3, 1]],
                         linewidth=thickness, 
                         color=np.array((255, 0, 0)) / 255)

                if name == "CAR":
                    obj_class_pred[0] += 1
                elif name == "TRUCK":
                    obj_class_pred[1] += 1
                elif name == "TRAILER":
                    obj_class_pred[2] += 1
                elif name == "VAN":
                    obj_class_pred[3] += 1
                elif name == "BUS":
                    obj_class_pred[4] += 1
                elif name == "MOTORCYCLE":
                    obj_class_pred[5] += 1
                elif name == "PEDESTRIAN":
                    obj_class_pred[6] += 1
                elif name == "BICYCLE":
                    obj_class_pred[7] += 1

            # name = classes[labels[index]]
            # plt.plot(
            #     coords[index, :, 0],
            #     coords[index, :, 1],
            #     linewidth=thickness,
            #     color=np.array(color or OBJECT_PALETTE[name]) / 255,
            # )

    # print("obj_class_gt", obj_class_gt)
    # print("obj_class_pred", obj_class_pred)
    
    mmcv.mkdir_or_exist(os.path.dirname(fpath))
    fig.savefig(
        fpath,
        dpi=10,
        facecolor="black",
        format="png",
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()

    return obj_class_gt, obj_class_pred

def visualize_lidar_two(
    fpath: str,
    v_lidar: Optional[np.ndarray] = None,
    i_lidar: Optional[np.ndarray] = None,
    *,
    bboxes: Optional[LiDARInstance3DBoxes] = None,
    labels: Optional[np.ndarray] = None,
    classes: Optional[List[str]] = None,
    xlim: Tuple[float, float] = (-50, 50),
    ylim: Tuple[float, float] = (-50, 50),
    color: Optional[Tuple[int, int, int]] = None,
    radius: float = 15,
    thickness: float = 25,
) -> None:
    fig = plt.figure(figsize=(xlim[1] - xlim[0], ylim[1] - ylim[0]))

    ax = plt.gca()
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect(1)
    ax.set_axis_off()

    if v_lidar is not None:
        plt.scatter(
            v_lidar[:, 0],
            v_lidar[:, 1],
            s=radius,
            c="white",
        )

    if i_lidar is not None:
        plt.scatter(
            i_lidar[:, 0],
            i_lidar[:, 1],
            s=radius,
            c="red",
        )

    if bboxes is not None and len(bboxes) > 0:
        coords = bboxes.corners[:, [0, 3, 7, 4, 0], :2]
        for index in range(coords.shape[0]):
            name = classes[labels[index]]
            plt.plot(
                coords[index, :, 0],
                coords[index, :, 1],
                linewidth=thickness,
                color=np.array(color or OBJECT_PALETTE[name]) / 255,
            )

    mmcv.mkdir_or_exist(os.path.dirname(fpath))
    fig.savefig(
        fpath,
        dpi=10,
        facecolor="black",
        format="png",
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()

def visualize_lidar_two_combo(
    fpath: str,
    v_lidar: Optional[np.ndarray] = None,
    i_lidar: Optional[np.ndarray] = None,
    *,
    gtbboxes: Optional[LiDARInstance3DBoxes] = None,
    bboxes: Optional[LiDARInstance3DBoxes] = None,
    gtlabels: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    classes: Optional[List[str]] = None,
    xlim: Tuple[float, float] = (-50, 50),
    ylim: Tuple[float, float] = (-50, 50),
    color: Optional[Tuple[int, int, int]] = None,
    radius: float = 15,
    thickness: float = 25,
) -> None:
    fig = plt.figure(figsize=(xlim[1] - xlim[0], ylim[1] - ylim[0]))

    ax = plt.gca()
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect(1)
    ax.set_axis_off()

    if v_lidar is not None:
        plt.scatter(
            v_lidar[:, 0],
            v_lidar[:, 1],
            s=radius,
            c="white",
        )

    if i_lidar is not None:
        plt.scatter(
            i_lidar[:, 0],
            i_lidar[:, 1],
            s=radius,
            c="red",
        )

    if bboxes is not None and len(bboxes) > 0:

        coords = bboxes.corners[:, [0, 3, 7, 4, 0], :2]
        for index in range(coords.shape[0]):
            name = classes[labels[index]]
            plt.plot(
                coords[index, :, 0],
                coords[index, :, 1],
                linewidth=thickness,
                color=np.array(color or OBJECT_PALETTE[name]) / 255,
            )

    if gtbboxes is not None and len(gtbboxes) > 0:

        if transform.size < 16:
            transform = np.append(transform, [[0.0, 0.0, 0.0, 1.0]], axis=0)
        
        transform = copy.deepcopy(transform).reshape(4, 4)
        coords_ = coords @ transform.T
        coords_ = coords.reshape(-1, 8, 4)

        indices = np.all(coords_[..., 2] > 0, axis=1)
        coords = coords[indices]
        labels = labels[indices]

        coords = gtbboxes.corners[:, [0, 3, 7, 4, 0], :2]
        for index in range(coords.shape[0]):
            name = classes[gtlabels[index]]
            plt.plot(
                coords[index, :, 0],
                coords[index, :, 1],
                linewidth=thickness,
                color=np.array((57, 255, 20)) / 255,
            )

    mmcv.mkdir_or_exist(os.path.dirname(fpath))
    fig.savefig(
        fpath,
        dpi=10,
        facecolor="black",
        format="png",
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close()

def visualize_map(
    fpath: str,
    masks: np.ndarray,
    *,
    classes: List[str],
    background: Tuple[int, int, int] = (240, 240, 240),
) -> None:
    assert masks.dtype == bool, masks.dtype

    canvas = np.zeros((*masks.shape[-2:], 3), dtype=np.uint8)
    canvas[:] = background

    for k, name in enumerate(classes):
        if name in MAP_PALETTE:
            canvas[masks[k], :] = MAP_PALETTE[name]
    canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

    mmcv.mkdir_or_exist(os.path.dirname(fpath))
    mmcv.imwrite(canvas, fpath)
