## Getting Started

### 1. Environment Setup
```shell
conda create -n sam6d python=3.10
conda activate sam6d
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
cd Pose_Estimation_Model/model/pointnet2
python setup.py install
```

### 2. Evaluation on the IPD&XYZ dataset

#### Your data should follow the structure
```
SAM-6D
|---Data/
|   |---IPD/
|       |---models/
|       |---camera.json
|       |---ipd_mask_sam6d.json
|       |---test_targets_multiview_bop25.json
|       |---test/
|           |--000000/
|                |--gray
|                |--depth
|                |--scene_camera.json
|           |--000001/
|           |--...
|           |--0000014/
|   |---XYZ/
|       |---models/
|       |---camera.json
|       |---xyz_mask_sam6d.json
|       |---test_targets_multiview_bop25.json
|       |---test/
|           |--000001/
|               |--gray
|               |--depth
|               |--scene_camera.json
|           |--000002/
|           |--...
|           |--0000074/
```

#### Run the template render
```shell
blenderproc run render_templates.py --dataset_name IPD
```

#### Run the instance segentation model
```shell
cd Instance_Segmentation_Model
python run_inference_ism.py --output_dir ../Data/IPD/sam6d_outputs --input_dir ../Data/IPD/val --template_dir ../Data/IPD/templates --cad_dir ../Data/IPD/models --targets_path ../Data/IPD/test_targets_bop19.json
```

#### Run the pose estimation model
```shell
python Pose_Estimation_Model/run_inference_detections.py --dataset_name IPD --output_dir Data/IPD/sam6d_outputs --input_dir Data/IPD/val --template_dir Data/IPD/templates --cad_dir Data/IPD/models --detection_path Data/IPD/ipd_mask_sam6d.json --det_score_thresh 0.4
```