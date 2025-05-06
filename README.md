## 6D detection on IPD (Industrial Plenoptic Dataset) with SAM-6D
<p align="center">
  <img src="media/ipd-objects.png" width="960px">
</p>

### 1. Environment Setup
CUDA versions should match between GPU Driver, CUDA Toolkit and PyTorch.
```shell
conda create -n sam6d python=3.10
conda activate sam6d
conda install cuda=12.4
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```
Build Pointnet++ extensions for PEM.
```shell
cd Pose_Estimation_Model/model/pointnet2
python setup.py install
```
Data should follow the structure:
```
SAM-6D-IPD
|---Data/
|   |---IPD/
|       |---models/
|       |---camera_[cam1, cam2, cam3, photoneo].json
|       |---camera_phononeo.json
|       |---ism_mask_sam6d.json
|       |---test_targets_bop19.json
|       |---test/
|           |---000000/
|                |---aolp_[cam1, cam2, cam3]/
|                |---depth_[cam1, cam2, cam3, photoneo]/
|                |---dolp_[cam1, cam2, cam3]/
|                |---rgb_[cam1, cam2, cam3, photoneo]/
|                |---scene_camera_[cam1, cam2, cam3, photoneo].json
|           |---000001/
|           |---...
|           |---0000014/
```

### 2. Evaluation on the IPD dataset

#### Run the template render
```shell
blenderproc run render_templates.py --dataset_name IPD
```

#### Run the instance segentation model
```shell
cd Instance_Segmentation_Model
python run_inference_ism.py --segmentor_model sam --output_dir ../Data/IPD/sam6d_outputs --input_dir ../Data/IPD/val --template_dir ../Data/IPD/templates --cad_dir ../Data/IPD/models --targets_path ../Data/IPD/test_targets_bop19.json
```

#### Run the pose estimation model
```shell
cd ../Pose_Estimation_Model
python run_inference_pem.py  --output_dir ../Data/IPD/sam6d_outputs --input_dir ../Data/IPD/val --template_dir ../Data/IPD/templates --cad_dir ../Data/IPD/models --targets_path ../Data/IPD/test_targets_bop19.json --detection_path ../Data/IPD/sam6d_outputs --det_score_thresh 0.4
```

### 3. Results
Here is an example detection with ISM segmentations and PEM detections concatenated with the original RGB image:
<p align="center">
  <img src="media/03_vis_ism_img1_obj11.png" width="960px">
</p>
<p align="center">
  <img src="media/03_vis_pem_img1_obj11.png" width="960px">
</p>