torchpack dist-run -np 1 python tools/train.py configs/tumtraf_i/det/transfusion/secfpn/camera+lidar/yolov8/pointpillars.yaml  --load_from /home/ivpg/Lacie/Github/coopdet3d/weights/epoch_20_lidar_only.pth 

python ./tools/create_tumtraf_data.py --root-path /home/ivpg/Lacie/Github/coopdet3d/data/tumtraf_i --out-dir /home/ivpg/Lacie/Github/coopdet3d/data/tumtraf_i_môn_processed --splits training,validation

torchpack dist-run -np 1 python tools/visualize.py configs/tumtraf_i/det/transfusion/secfpn/camera+lidar/yolov8/pointpillars.yaml --checkpoint weights/latest.pth --split val --mode pred --out-dir viz_tumtraf 

configs/tumtraf_i/det/centerhead/lssfpn/camera/256x704/yolov8/default.yaml

torchpack dist-run -np 1 python tools/visualize.py configs/tumtraf_i/det/centerhead/lssfpn/camera/256x704/yolov8/default.yaml --checkpoint weights/epoch_20.pth --split val --mode pred --out-dir viz_tumtraf_rgb
