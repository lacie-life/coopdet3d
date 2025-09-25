import time

import fire

import kitti_common as kitti
from eval_aihub import get_coco_eval_result, get_official_eval_result


def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()

    # Only get the file names without extensions
    count = 0
    out_lines = []
    for i in range(len(lines)):
        lines[i] = lines[i].split('/')[-1]  # in case that path is included
        lines[i] = lines[i].split('.')[0]
        lines[i] = lines[i].strip()  # in case that \n is included

        # if i < 1:
        #     out_lines.append(lines[i])
        #     count += 1

        out_lines.append(lines[i])
    return out_lines


def evaluate(label_path,
             result_path,
             label_split_file,
             current_class=0,
             coco=False,
             score_thresh=-1):
    
    # if score_thresh > 0:
    #     dt_annos = kitti.filter_annos_low_score(dt_annos, score_thresh)
    val_image_ids = _read_imageset_file(label_split_file)
    dt_annos = kitti.get_label_annos(result_path, val_image_ids)
    gt_annos = kitti.get_label_annos(label_path, val_image_ids)

    # for i in range(10):
    #     print("GT annos example" + str(gt_annos[i]))
    #     print("DT annos example" + str(dt_annos[i]))

    print('-------------------')
    print("len(gt_annos): ", len(gt_annos))
    print("len(dt_annos): ", len(dt_annos))

    if coco:
        return get_coco_eval_result(gt_annos, dt_annos, current_class)
    else:
        return get_official_eval_result(gt_annos, dt_annos, current_class)


if __name__ == '__main__':
    label_path = '/home/lacie/Github/coopdet3d/data/AIHub_KITTI_format_fusion_refined_v2/training/label_2/'
    result_path = '/home/lacie/Github/coopdet3d/kitti_output/lidar_only_aihub_lidar_list_2/'
    label_split_file = '/home/lacie/Github/coopdet3d/kitti_output/lidar_only_aihub_lidar_list_2/pred_list.txt'
    result, detail = evaluate(label_path, result_path, label_split_file, current_class=[0, 1, 2], coco=False, score_thresh=-1)
    
    print(result)
    print(detail)