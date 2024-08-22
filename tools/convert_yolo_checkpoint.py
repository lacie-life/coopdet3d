import torch

cp_old1 = torch.load('/home/lacie/Github/coopdet3d/weights/yolov8_s_syncbn_fast_8xb16-500e_coco_20230117_180101-5aa5f0f1.pth')

model = cp_old1["state_dict"]
new_model = dict()

for key in model:
    if "backbone" in key:
        new_key = key.replace("backbone.", "")
        new_model[new_key] = model[key]

cp_old1["state_dict"] = new_model
torch.save(cp_old1, '/home/lacie/Github/coopdet3d/weights/camera_pretrain.pth')