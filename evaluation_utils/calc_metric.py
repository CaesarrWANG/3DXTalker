import os
# import torch
import numpy as np
from tqdm import tqdm
from evaluation_utils.metrics import LVS, LVE, MVE, UFVE, SyncNet_LSEC_LSED, FDD, FID, PLRS, CSIM, EmoFID, pose_variation
from evaluation_utils.beat_align_score_flame import compute_beat_alignment_score
cur_dir = os.path.dirname(os.path.abspath(__file__))



def Compute_BA_score(audio_path, pred_flame_path):
    ba_score = compute_beat_alignment_score(
        audio=audio_path,
        flame_parameters_path=pred_flame_path,
        fps=25,                    # 视频帧率
        audio_method='beat',      # 'onset' 或 'beat'
        pose_sigma=3,              # pose平滑参数
        variance=7                # 对齐容忍度
    )
    return ba_score

def main(args):
    model_testset_pred_dir_path = args.model_testset_pred_dir_path
    gt_testset_dir_path = args.gt_testset_dir_path
    model_project_dir = os.path.dirname(model_testset_pred_dir_path)
    # MTM_output_path = os.path.join(model_project_dir, "MTM_output")
    # os.makedirs(MTM_output_path, exist_ok=True)
    # MTM_csv_path = os.path.join(MTM_output_path, "mtm_results.csv")

    dataset_list = sorted(os.listdir(gt_testset_dir_path))
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    LVS_list = []
    LVE_list = []
    MVE_list = []
    UFVE_list = []
    LSEC_list = []
    LSED_list = []
    FDD_list = []
    CSIM_list = []  
    POSE_list = []
    EmoFID_list = []
    FID_list = []
    MTM_list = []
    PLRS_list = []
    BA_list = []

    for dataset in dataset_list:
        print(f"Evaluating {dataset}...")
        dataset_pred_dir_path = os.path.join(model_testset_pred_dir_path, dataset)
        dataset_gt_dir_path = os.path.join(gt_testset_dir_path, dataset)
        all_samples = sorted(os.listdir(dataset_gt_dir_path))

        for sample in tqdm(all_samples):
            sample_pred_dir_path = os.path.join(dataset_pred_dir_path, sample)
            sample_gt_dir_path = os.path.join(dataset_gt_dir_path, sample)

            sample_audio_path = os.path.join(sample_gt_dir_path, "audio.wav")
            
            pred_vertices_npy_path = os.path.join(sample_pred_dir_path, "verts_seq.npy")
            gt_vertices_npy_path = os.path.join(sample_gt_dir_path, "vertices.npy")
            
            pred_rendered_mesh_video_path = os.path.join(sample_pred_dir_path, "MeshRenderedVideo.mp4")
            gt_rendered_mesh_video_path = os.path.join(sample_gt_dir_path, "MeshRenderedVideo.mp4")
            
            pred_params_path = os.path.join(sample_pred_dir_path, "flame_params.npy")
            # pose_variation_score = pose_variation(pred_params_path)
            
            # LVS_score = LVS(pred_vertices_npy_path, gt_vertices_npy_path, device)
            # LVE_score, LVE_norm = LVE(pred_vertices_npy_path, gt_vertices_npy_path, device)
            # MVE_score = MVE(pred_vertices_npy_path, gt_vertices_npy_path, device)
            # UFVE_score = UFVE(pred_vertices_npy_path, gt_vertices_npy_path, device)
            # LSEC_score, LSED_score = SyncNet_LSEC_LSED(pred_rendered_mesh_video_path)
            # FDD_score = FDD(pred_vertices_npy_path, gt_vertices_npy_path, device)
            # CSIM_score = CSIM(pred_rendered_mesh_video_path)
            # EmoFID_score = EmoFID(pred_rendered_mesh_video_path, gt_rendered_mesh_video_path)
            # FID_score = FID(gt_rendered_mesh_video_path, pred_rendered_mesh_video_path, device)
            # MTM_score = MTM(sample_gt_dir_path, sample_pred_dir_path, MTM_output_path, MTM_csv_path)
            # PLRS_score = PLRS(sample_audio_path, sample_pred_dir_path)

            BA_score = Compute_BA_score(sample_audio_path, pred_params_path)
            BA_list.append(BA_score)
            # LVS_list.append(LVS_score.item())
            # LVE_list.append(LVE_score.item())
            # MVE_list.append(MVE_score.item())
            # UFVE_list.append(UFVE_score.item())
            # LSEC_list.append(LSEC_score)
            # LSED_list.append(np.mean(LSED_score))
            # FDD_list.append(FDD_score.item())
            # EmoFID_list.append(EmoFID_score.item())
            # CSIM_list.append(CSIM_score)
            # FID_list.append(FID_score.item())
            # MTM_list.append(MTM_score)
            # PLRS_list.append(PLRS_score)
            # POSE_list.append(pose_variation_score)

    print("==== Final Evaluation Results ====")
    # print(f"LVS: {np.mean(LVS_list)}")
    # print(f"LVE: {np.mean(LVE_list)}")
    # print(f"MVE: {np.mean(MVE_list)}")
    # print(f"UFVE: {np.mean(UFVE_list)}")
    # print(f"LSEC: {np.mean(LSEC_list)}")
    # print(f"LSED: {np.mean(LSED_list)}")
    # print(f"FDD: {np.mean(FDD_list)}")
    # print(f"CSIM: {np.mean(CSIM_list)}")
    # print(f"EmoFID: {np.mean(EmoFID_list)}")
    # print(f"FID: {np.mean(FID_list)}")
#    print(f"MTM: {np.mean(MTM_list)}")
#    print(f"PLRS: {np.mean(PLRS_list)}")
    # print(f"POSE: {np.mean(POSE_list)}")
    print(f"BA: {np.mean(BA_list)}")

if __name__ == "__main__":
    

    # model_testset_pred_dir_path = "/scratch3/wan451/3DTalk/The-Sound-of-Motion/3DXTalker_preds"
    # model_testset_pred_dir_path = "/scratch3/wan451/3DTalk/inferno/EMOTE_preds"
    # model_testset_pred_dir_path = "/scratch3/wan451/3DTalk/DEEPTalk/DEEPTalk_preds"
    # gt_testset_dir_path = '/scratch3/wan451/3DTalk/emoca/video_output/EMOCA_v2_lr_mse_20/testset'

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_testset_pred_dir_path", type=str, default= "./inference_results",
                        help="Path to the model's predicted testset directory.")
    parser.add_argument("--gt_testset_dir_path", type=str, default="./testset",
                        help="Path to the ground truth testset directory.")
    args = parser.parse_args()

    # model_preds_path_array=[
    # "/scratch3/wan451/3DTalk/The-Sound-of-Motion/3DXTalker_Predictions/TrainLip1-ModelName-TalkingHead3D_Evlp_v6-NoiseType-seg_interp_25-use_envelope_scale-True-LipScale-1.0-UseAbsFrameTraining-False",
    # "/scratch3/wan451/3DTalk/The-Sound-of-Motion/3DXTalker_Predictions/TrainLip1.5-ModelName-TalkingHead3D_Evlp_v6-NoiseType-seg_interp_25-use_envelope_scale-True-LipScale-1.0-UseAbsFrameTraining-False",
    # "/scratch3/wan451/3DTalk/The-Sound-of-Motion/3DXTalker_Predictions/TrainLip1.5-ModelName-TalkingHead3D_Evlp_v6-NoiseType-seg_interp_25-use_envelope_scale-True-LipScale-1.5-UseAbsFrameTraining-False",
    # "/scratch3/wan451/3DTalk/The-Sound-of-Motion/3DXTalker_Predictions/TrainLip2-ModelName-TalkingHead3D_Evlp_v6_wo_emo2vec-NoiseType-seg_interp_25-use_envelope_scale-True-LipScale-1.0-UseAbsFrameTraining-False",
    # "/scratch3/wan451/3DTalk/The-Sound-of-Motion/3DXTalker_Predictions/TrainLip2-ModelName-TalkingHead3D_Evlp_v6_wo_envelope-NoiseType-seg_interp_25-use_envelope_scale-False-LipScale-1.0-UseAbsFrameTraining-False",
    # "/scratch3/wan451/3DTalk/The-Sound-of-Motion/3DXTalker_Predictions/TrainLip2-ModelName-TalkingHead3D_Evlp_v6-NoiseType-random-use_envelope_scale-True-LipScale-1.0-UseAbsFrameTraining-False",
    # "/scratch3/wan451/3DTalk/The-Sound-of-Motion/3DXTalker_Predictions/TrainLip2-ModelName-TalkingHead3D_Evlp_v6-NoiseType-seg_interp_25-use_envelope_scale-True-LipScale-1.0-UseAbsFrameTraining-False",
    # "/scratch3/wan451/3DTalk/The-Sound-of-Motion/3DXTalker_Predictions/TrainLip2-ModelName-TalkingHead3D_Evlp_v6-NoiseType-seg_interp_25-use_envelope_scale-True-LipScale-1.0-UseAbsFrameTraining-True",
    # "/scratch3/wan451/3DTalk/The-Sound-of-Motion/3DXTalker_Predictions/TrainLip2-ModelName-TalkingHead3D_Evlp_v6-NoiseType-seg_interp_25-use_envelope_scale-True-LipScale-2.0-UseAbsFrameTraining-False",
    # "/scratch3/wan451/3DTalk/The-Sound-of-Motion/3DXTalker_Predictions/TrainLip2-ModelName-TalkingHead3D_Evlp_v6-NoiseType-seg_interp_250-use_envelope_scale-True-LipScale-1.0-UseAbsFrameTraining-False",
    # "/scratch3/wan451/3DTalk/The-Sound-of-Motion/3DXTalker_Predictions/TrainLip2-TestLip2-3DXTalker_preds_v6" 
    # "/scratch3/wan451/3DTalk/The-Sound-of-Motion/3DXTalker_Predictions/ModelName-TalkingHead3D_Evlp_v6_wo_envelope-NoiseType-seg_interp_25-use_envelope_scale-True-LipScale-2.0-UseAbsFrameTraining-False",
    # "/scratch3/wan451/3DTalk/The-Sound-of-Motion/3DXTalker_Predictions/ModelName-TalkingHead3D_Evlp_v6-NoiseType-random-use_envelope_scale-True-LipScale-2.0-UseAbsFrameTraining-False",
    # "/scratch3/wan451/3DTalk/The-Sound-of-Motion/3DXTalker_Predictions/ModelName-TalkingHead3D_Evlp_v6-NoiseType-seg_interp_25-use_envelope_scale-True-LipScale-2.0-UseAbsFrameTraining-True",
    # "/scratch3/wan451/3DTalk/The-Sound-of-Motion/3DXTalker_Predictions/ModelName-TalkingHead3D_Evlp_v6-NoiseType-seg_interp_250-use_envelope_scale-True-LipScale-2.0-UseAbsFrameTraining-False",
    # ]

    # model_preds_path_array = [
    #     # "/scratch3/wan451/3DTalk/The-Sound-of-Motion/3DXTalker_Pred_ablations/ModelName-TalkingHead3D_Evlp_v6-NoiseType-seg_interp_25-use_envelope_scale-True-LipScale-0.5-UseAbsFrameTraining-False",
    #     # "/scratch3/wan451/3DTalk/The-Sound-of-Motion/3DXTalker_Pred_ablations/ModelName-TalkingHead3D_Evlp_v6-NoiseType-seg_interp_25-use_envelope_scale-True-LipScale-0.8-UseAbsFrameTraining-False",
    #     "/scratch3/wan451/3DTalk/The-Sound-of-Motion/3DXTalker_Pred_UnLockPose_2/ModelName-TalkingHead3D_Evlp_v6-NoiseType-seg_interp_25-use_envelope_scale-True-LipScale-0.5-UseAbsFrameTraining-False"
    #     ]

    # for pred_path in model_preds_path_array:
    #     args.model_testset_pred_dir_path=pred_path
    #     print(f"Evaluating model predictions at: {args.model_testset_pred_dir_path}")
    #     main(args)
    main(args)

