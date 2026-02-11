import torch
from .mask import Mask
from .pytorch_fid import fid_score_incep
# from evaluation_utils.evaluate_MTM import process_sequences
# from evaluation_utils.evaluate_PLRS import main as PLRS_main
import numpy as np
import pickle
from .syncnet_python.SyncNetInstance import SyncNetInstance
import os
import argparse
from utils.render import read_obj
import shutil
from CopulaSimilarity.CSM import CopulaBasedSimilarity as CSIMSimilarity
import subprocess
import cv2
from .pytorch_fid import Emofid_score
from .beat_align_score_flame import compute_beat_alignment_score

def load_vertices(vertices_npy_path) :
    vertices = np.load(vertices_npy_path) # (t,5023,3)
    vertices = torch.from_numpy(vertices)
    return vertices

def LVE(vertices_pred_npy_path, vertices_gt_npy_path, device) :
    vertices_pred = load_vertices(vertices_pred_npy_path).to(device)
    vertices_gt = load_vertices(vertices_gt_npy_path).to(device)
    flame_mask_path = "./resources/FLAME_masks.pkl"
    with open(flame_mask_path, "rb") as f:   # "rb" = read as binary
        full_mask = pickle.load(f, encoding="latin1")
    lip_mask = full_mask['lips']  # lip:254
    min_len = min(vertices_gt.shape[0], vertices_pred.shape[0])
    vertices_gt = vertices_gt[:min_len]
    vertices_pred = vertices_pred[:min_len]
    metric_L2 = (vertices_gt[:, lip_mask] - vertices_pred[:, lip_mask])**2
    metric_L2 = metric_L2.sum(-1)
    metric_L2 = metric_L2.max(-1)[0]
    metric_L2norm = metric_L2**0.5
    return metric_L2.mean(), metric_L2norm.mean()

def LVS(vertices_pred_npy_path, vertices_gt_npy_path, device):
    # 加载预测和GT
    vertices_pred = load_vertices(vertices_pred_npy_path).to(device)
    vertices_gt = load_vertices(vertices_gt_npy_path).to(device)

    # 加载lip掩码索引
    flame_mask_path = "./resources/FLAME_masks.pkl"
    with open(flame_mask_path, "rb") as f:
        full_mask = pickle.load(f, encoding="latin1")
    lip_mask = full_mask['lips']  # lip:254个索引

    # 截断为相同帧数
    min_len = min(vertices_gt.shape[0], vertices_pred.shape[0])
    vertices_gt = vertices_gt[:min_len]
    vertices_pred = vertices_pred[:min_len]

    # 提取 lips 区域 (B, 254, 3)
    lips_gt = vertices_gt[:, lip_mask]
    lips_pred = vertices_pred[:, lip_mask]

    # 计算 cosine similarity for each frame
    lips_gt_flat = lips_gt.reshape(min_len, -1)     # (B, 254*3)
    lips_pred_flat = lips_pred.reshape(min_len, -1) # (B, 254*3)

    cosine_sim = torch.nn.functional.cosine_similarity(lips_pred_flat, lips_gt_flat, dim=-1)  # (B,)
    return cosine_sim.mean()

def MVE(vertices_pred_npy_path, vertices_gt_npy_path, device):
    vertices_pred = load_vertices(vertices_pred_npy_path).to(device)
    vertices_gt = load_vertices(vertices_gt_npy_path).to(device)
    min_len = min(vertices_gt.shape[0], vertices_pred.shape[0])
    vertices_gt = vertices_gt[:min_len]
    vertices_pred = vertices_pred[:min_len]
    MVE_score = torch.mean(torch.sqrt(torch.sum((vertices_pred - vertices_gt) ** 2, dim=-1)))
    return MVE_score

def UFVE(vertices_pred_npy_path, vertices_gt_npy_path, device):
    vertices_pred = load_vertices(vertices_pred_npy_path).to(device)
    vertices_gt = load_vertices(vertices_gt_npy_path).to(device)
    min_len = min(vertices_gt.shape[0], vertices_pred.shape[0])
    vertices_gt = vertices_gt[:min_len]
    vertices_pred = vertices_pred[:min_len]
    flame_mask_path = "./resources/FLAME_masks.pkl"
    with open(flame_mask_path, "rb") as f:   # "rb" = read as binary
        full_mask = pickle.load(f, encoding="latin1")
    upper_mask_parts = ['eye_region', 'forehead', 'nose']
    upper_mask_indices = []
    for part in upper_mask_parts:
        upper_mask_indices.extend(full_mask[part])
    upper_mask_indices = np.array(upper_mask_indices)
    metric_L2 = (vertices_gt[:, upper_mask_indices] - vertices_pred[:, upper_mask_indices])**2
    metric_L2 = metric_L2.sum(-1)
    metric_L2norm = metric_L2**0.5
    UFVE_score = torch.mean(metric_L2norm)
    return UFVE_score
    

def SyncNet_LSEC_LSED(pred_rendered_mesh_video_sync_path):
    model_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(pred_rendered_mesh_video_sync_path))))
    syncnet_cache_work_path = os.path.join("./syncnet_data", model_name, "work")
    # syncnet_cache_work_path = "./evaluation_utils/syncnet_python/data/work/pytmp"
    if os.path.exists(syncnet_cache_work_path):
        shutil.rmtree(syncnet_cache_work_path)
    sync_model = SyncNetInstance()
    sync_weight_path = "./evaluation_utils/syncnet_python/data/syncnet_v2.model"
    sync_model.loadParameters(sync_weight_path)

    opt = argparse.Namespace(
        initial_model = sync_weight_path,
        batch_size = 20,
        vshift = 15,
        data_dir = syncnet_cache_work_path,
        videofile = pred_rendered_mesh_video_sync_path,
        reference = ""
    )

    setattr(opt, 'avi_dir',  os.path.join(opt.data_dir, 'pyavi'))
    setattr(opt, 'tmp_dir',  os.path.join(opt.data_dir, 'pytmp'))
    setattr(opt, 'work_dir', os.path.join(opt.data_dir, 'pywork'))
    setattr(opt, 'crop_dir', os.path.join(opt.data_dir, 'pycrop'))
    
    offset, conf, dist = sync_model.evaluate(opt, videofile=pred_rendered_mesh_video_sync_path)

    return conf, dist

def FDD(vertices_pred_npy_path, vertices_gt_npy_path, device) :
    template_vertices_path = "./resources/head_template.obj"
    template_vertices = np.array(read_obj(template_vertices_path)[0])
    template_vertices = torch.from_numpy(template_vertices).to(device)
    flame_mask_path = "./resources/FLAME_masks.pkl"
    mask_model = Mask(flame_mask_path)
    upper_type = ['eye_region', 'forehead', 'nose', ]
    
    vertices_pred = load_vertices(vertices_pred_npy_path).to(device)
    vertices_gt = load_vertices(vertices_gt_npy_path).to(device)

    min_len = min(vertices_gt.shape[0], vertices_pred.shape[0])
    vertices_gt = vertices_gt[:min_len]
    vertices_pred = vertices_pred[:min_len]

    # calc motion
    motion_pred = vertices_pred - template_vertices.unsqueeze(0)
    motion_gt = vertices_gt - template_vertices.unsqueeze(0)
    # masking only upper part
    masked_motion_pred = mask_model.masked_vertice(upper_type, motion_pred.shape, motion_pred, device)
    masked_motion_gt = mask_model.masked_vertice(upper_type, motion_gt.shape, motion_gt, device)

    def cal_std(masked_vertice) :
        L2_dis_upper = torch.transpose(masked_vertice,0,1)
        L2_dis_upper = torch.sum(L2_dis_upper, dim=2)
        motion_std = torch.std(L2_dis_upper, dim=0)
        motion_std_mean = torch.mean(motion_std)
        # print(f'mean shape : {motion_std.shape}')

        # return motion_std.item()
        return motion_std_mean

    # calc motion std
    pred_motion_std = cal_std(masked_motion_pred)
    print(f'pred : {pred_motion_std}')
    gt_motion_std = cal_std(masked_motion_gt)
    print(f'gt : {gt_motion_std}')
    # calc FDD score
    motion_std_diff = gt_motion_std - pred_motion_std
    FDD_score = abs(motion_std_diff)
    # FDD_score = torch.sum(motion_std_diff, dim=0)/motion_std_diff.shape[0]
    
    return FDD_score

def FID(gt_rendered_video_path, pred_rendered_video_path, device) :
    FID_score = fid_score_incep.calculate_fid_given_paths(paths=[gt_rendered_video_path, pred_rendered_video_path], batch_size = 10, device=device, dims=2048, num_workers=0)
    return FID_score

# def MTM(gt_verts_npy_dir_path, pred_verts_npy_dir_path, out_dir, out_csv_path):
#     MTM_score = process_sequences(gt_verts_npy_dir_path, pred_verts_npy_dir_path, out_dir, out_csv_path)
#     return MTM_score

def PLRS(audio_path, pred_verts_npy_dir_path):

    args = argparse.Namespace(
        device = "cuda:0",
        model = "speech_mesh_rep",
        model_path =  "./evaluation_utils/perceptual3D/checkpoints/model_eval.pth",
        input_size  = 160, 
        input_size_audio = 64, 
        depth = 10,
        depth_audio = 10, 
        num_frames = 5 ,
        eval_data_path = pred_verts_npy_dir_path,
        wav_path = audio_path,
        num_mel_bins = 128,
        model_key ='model|module', 
        model_prefix = "",
    )

    plrs_score = PLRS_main(args)

    return plrs_score


def compute_CSIM_sim(image_1_path, image_2_path):
    # Default patch_size set to 8 but can be changed depending on the aimed balance between accuracy and realtime
    copula_similarity = CSIMSimilarity(patch_size=8) 

    #load your images
    image1 = cv2.imread(image_1_path)
    image2 = cv2.imread(image_2_path)

    #calculate the similarity map
    csim_map = copula_similarity.compute_local_similarity(image1, image2)
    csim = np.mean(csim_map)

    return csim

def CSIM(pred_rendered_video_path):
    video_dir = os.path.dirname(pred_rendered_video_path)
    frames_folder = os.path.join(video_dir, "frames")
    if not os.path.exists(frames_folder):
        os.makedirs(frames_folder, exist_ok=True)
        cmd = ["ffmpeg", "-i", pred_rendered_video_path, "-r", "25", os.path.join(frames_folder, "%04d.png")]
        subprocess.run(cmd, check=True)
    images = sorted(os.listdir(frames_folder))  # sorted list of filenames
    images = [os.path.join(frames_folder, f) for f in images]  # full paths
    csim_score_lst = []
    for img1, img2 in zip(images, images[1:]):
        frame_score = compute_CSIM_sim(img1, img2)
        csim_score_lst.append(frame_score)
    csim_score_arr = np.array(csim_score_lst)
    return np.mean(csim_score_arr)

def EmoFID(pred_rendered_video_path, gt_rendered_video_path):
    pred_sample_dir = os.path.dirname(pred_rendered_video_path)
    gt_sample_dir = os.path.dirname(gt_rendered_video_path)
    gt_frames_folder = os.path.join(gt_sample_dir, "frames")
    if not os.path.exists(gt_frames_folder):
        os.makedirs(gt_frames_folder, exist_ok=True)
        cmd = ["ffmpeg", "-i", gt_rendered_video_path, "-r", "25", os.path.join(gt_frames_folder, "%04d.png")]
        subprocess.run(cmd, check=True)

    pred_frames_folder = os.path.join(pred_sample_dir, "frames")
    if not os.path.exists(pred_frames_folder):
        os.makedirs(pred_frames_folder, exist_ok=True)
        cmd = ["ffmpeg", "-i", pred_rendered_video_path, "-r", "25", os.path.join(pred_frames_folder, "%04d.png")]
        subprocess.run(cmd, check=True)

    emo_fid_score = Emofid_score.calculate_fid_given_paths(paths=[gt_frames_folder, pred_frames_folder], batch_size=128, device="cuda:0", dims=1024, num_workers=8)
    return emo_fid_score

def pose_variation(pred_params_npy_path):
    params_pred =  np.load(pred_params_npy_path)
    params_pred = params_pred[:, 150:153]
    pose_var = np.var(params_pred, axis=0).mean()
    return pose_var

def Compute_BA_score(audio_path, pred_flame_path):
    ba_score = compute_beat_alignment_score(
        audio=audio_path,
        flame_parameters_path=pred_flame_path,
        fps=25,                    # 视频帧率
        audio_method='beat',      # 'onset' 或 'beat'
        pose_sigma=2,              # pose平滑参数
        variance=8                 # 对齐容忍度
    )
    return ba_score