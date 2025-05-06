import os
import gc
import numpy as np
from tqdm import tqdm
import torch
from PIL import Image
import logging
from hydra import initialize, compose
# set level logging
logging.basicConfig(level=logging.INFO)
import logging
import trimesh
import numpy as np
from hydra.utils import instantiate
import argparse
import glob
from omegaconf import DictConfig, OmegaConf
from torchvision.utils import save_image
import torchvision.transforms as T
import cv2
import imageio.v2 as imageio
import distinctipy
from skimage.feature import canny
from skimage.morphology import binary_dilation
from segment_anything.utils.amg import rle_to_mask

from utils.poses.pose_utils import get_obj_poses_from_template_level, load_index_level_in_level2
from utils.bbox_utils import CropResizePad
from model.utils import Detections, convert_npz_to_json
from utils.inout import load_json, save_json_bop23
from rich.progress import Progress
import json

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
gc.collect()
torch.cuda.empty_cache()

def batch_input_data(depth_path, cam_path, device):
    batch = {}
    cam_info = load_json(cam_path)
    depth = np.array(imageio.imread(depth_path)).astype(np.int32)
    cam = cam_info[next(iter(cam_info))]
    cam_K = np.array(cam['cam_K']).reshape((3, 3))
    depth_scale = np.array(cam['depth_scale'])
    batch["depth"] = torch.from_numpy(depth).unsqueeze(0).to(device)
    batch["cam_intrinsic"] = torch.from_numpy(cam_K).unsqueeze(0).to(device)
    batch['depth_scale'] = torch.from_numpy(depth_scale).unsqueeze(0).to(device)
    return batch

def run_inference(model, test_targets_path, output_dir, input_dir, template_folder, cad_folder):
    logging.info("Initializing template")
    
    with open(test_targets_path, "r") as f:
        test_target = json.load(f)
    scene_id = int(os.path.basename(input_dir))
    im_idx = [item['im_id'] for item in test_target if item['scene_id']==scene_id]
    obj_idx = [item['obj_id'] for item in test_target if item['scene_id']==scene_id]

    for im_id, obj_id in zip(im_idx, obj_idx):
        rgb_path = os.path.join(input_dir, "rgb_cam1", f"{im_id:06d}.png")
        depth_path = os.path.join(input_dir, "depth_cam1", f"{im_id:06d}.png")
        cam_path = next(iter(glob.glob(os.path.join(input_dir, "scene_camera_cam1.json"))), None)
        cad_path = os.path.join(cad_folder, f"obj_{obj_id:06d}.ply")

        template_dir = os.path.join(template_folder, f"obj_{obj_id:06d}")
        num_templates = len(glob.glob(f"{template_dir}/*.npy"))
        boxes, masks, templates = [], [], []
        for idx in range(num_templates):
            image = Image.open(os.path.join(template_dir, 'rgb_'+str(idx)+'.png'))
            mask = Image.open(os.path.join(template_dir, 'mask_'+str(idx)+'.png'))
            boxes.append(mask.getbbox())
            image = torch.from_numpy(np.array(image.convert("RGB")) / 255).float()
            mask = torch.from_numpy(np.array(mask.convert("L")) / 255).float()
            image = image * mask[:, :, None]
            templates.append(image)
            masks.append(mask.unsqueeze(-1))
            
        templates = torch.stack(templates).permute(0, 3, 1, 2)
        masks = torch.stack(masks).permute(0, 3, 1, 2)
        boxes = [box for box in boxes if box is not None]
        boxes = torch.tensor(np.array(boxes))
        
        processing_config = OmegaConf.create(
            {
                "image_size": 224,
            }
        )
        proposal_processor = CropResizePad(processing_config.image_size)
        templates = proposal_processor(images=templates, boxes=boxes).to(device)
        masks_cropped = proposal_processor(images=masks, boxes=boxes).to(device)

        model.ref_data = {}
        model.ref_data["descriptors"] = model.descriptor_model.compute_features(
                        templates, token_name="x_norm_clstoken"
                    ).unsqueeze(0).data
        model.ref_data["appe_descriptors"] = model.descriptor_model.compute_masked_patch_feature(
                        templates, masks_cropped[:, 0, :, :]
                    ).unsqueeze(0).data
        
        # run inference
        rgb = Image.open(rgb_path).convert("RGB")
        detections = model.segmentor_model.generate_masks(np.array(rgb))
        detections = Detections(detections)
        query_decriptors, query_appe_descriptors = model.descriptor_model.forward(np.array(rgb), detections)

        # matching descriptors
        (
            idx_selected_proposals,
            pred_idx_objects,
            semantic_score,
            best_template,
        ) = model.compute_semantic_score(query_decriptors)

        # update detections
        detections.filter(idx_selected_proposals)
        query_appe_descriptors = query_appe_descriptors[idx_selected_proposals, :]

        # compute the appearance score
        appe_scores, ref_aux_descriptor= model.compute_appearance_score(best_template, pred_idx_objects, query_appe_descriptors)

        # compute the geometric score
        batch = batch_input_data(depth_path, cam_path, device)
        template_poses = get_obj_poses_from_template_level(level=2, pose_distribution="all")
        template_poses[:, :3, 3] *= 0.4
        poses = torch.tensor(template_poses).to(torch.float32).to(device)
        model.ref_data["poses"] =  poses[load_index_level_in_level2(0, "all"), :, :]

        mesh = trimesh.load_mesh(cad_path)
        model_points = mesh.sample(2048).astype(np.float32) / 1000.0
        model.ref_data["pointcloud"] = torch.tensor(model_points).unsqueeze(0).data.to(device)
        
        image_uv = model.project_template_to_image(best_template, pred_idx_objects, batch, detections.masks)

        geometric_score, visible_ratio = model.compute_geometric_score(
            image_uv, detections, query_appe_descriptors, ref_aux_descriptor, visible_thred=model.visible_thred
            )

        # final score
        final_score = (semantic_score + appe_scores + geometric_score*visible_ratio) / (1 + 1 + visible_ratio)

        detections.add_attribute("scores", final_score)
        detections.add_attribute("object_ids", torch.tensor(obj_id))   
            
        detections.to_numpy()
        save_path = f"{output_dir}/detection_ism_img{im_id}_obj{obj_id}"
        detections.save_to_file(scene_id, im_id, 0, save_path, "Custom", return_results=False)
        detections = convert_npz_to_json(idx=0, list_npz_paths=[save_path+".npz"])
        save_json_bop23(save_path+".json", detections)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--segmentor_model", default='sam', help="The segmentor model in ISM")
    parser.add_argument("--output_dir", nargs="?", help="Path to root directory of the output")
    parser.add_argument("--input_dir", nargs="?", help="Path to input data")
    parser.add_argument("--template_dir", nargs="?", help="Path to templates")
    parser.add_argument("--cad_dir", nargs="?", help="Path to CAD(mm)")
    parser.add_argument("--targets_path", default='?', nargs="?", help="Path to test/val targets json")
    parser.add_argument("--stability_score_thresh", default=0.97, type=float, help="stability_score_thresh of SAM")
    args = parser.parse_args()

    with initialize(version_base=None, config_path="configs"):
        cfg = compose(config_name='run_inference.yaml')

    segmentor_model = args.segmentor_model
    if segmentor_model == "sam":
        with initialize(version_base=None, config_path="configs/model"):
            cfg.model = compose(config_name='ISM_sam.yaml')
        cfg.model.segmentor_model.stability_score_thresh = args.stability_score_thresh
    elif segmentor_model == "fastsam":
        with initialize(version_base=None, config_path="configs/model"):
            cfg.model = compose(config_name='ISM_fastsam.yaml')
    else:
        raise ValueError("The segmentor_model {} is not supported now!".format(segmentor_model))

    logging.info("Initializing model")
    model = instantiate(cfg.model)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.descriptor_model.model = model.descriptor_model.model.to(device)
    model.descriptor_model.model.device = device
    # if there is predictor in the model, move it to device
    if hasattr(model.segmentor_model, "predictor"):
        model.segmentor_model.predictor.model = (
            model.segmentor_model.predictor.model.to(device)
        )
    else:
        model.segmentor_model.model.setup_model(device=device, verbose=True)
    logging.info(f"Moving models to {device} done!")

    test_targets_path = args.targets_path
    input_folders = sorted(os.listdir(args.input_dir))
    with Progress() as progress:
        input_tqdm = progress.add_task('input_folders', total=len(input_folders))    
        for input_folder in input_folders:
            input_dir = os.path.join(args.input_dir, input_folder)
            output_dir = os.path.join(args.output_dir, input_folder)
            run_inference(model, test_targets_path, output_dir, input_dir, args.template_dir, args.cad_dir)
            progress.update(input_tqdm, advance=1)