# Instance Segmentation Model
cd Instance_Segmentation_Model
# SAM
wget -P ./checkpoints/segment-anything https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth --no-check-certificate
# FastSAM
mkdir -p ./checkpoints/FastSAM && gdown --no-cookies --no-check-certificate -O 'checkpoints/FastSAM/FastSAM-x.pt' 1m1sjY4ihXBU1fZXdQ-Xdj-mDltW-2Rqv
# DinoV2
wget -P ./checkpoints/dinov2 https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth --no-check-certificate

# Pose Estimation Model
cd ../Pose_Estimation_Model
mkdir -p checkpoints && gdown --no-cookies --no-check-certificate -O 'checkpoints/sam-6d-pem-base.pth' 1joW9IvwsaRJYxoUmGo68dBVg-HcFNyI7