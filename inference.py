import librosa
import numpy as np
import os
import re
import yaml
import torch
from transformers import Wav2Vec2FeatureExtractor
import argparse
from models._3DXTalker import MyModel
from decalib.deca import DECA
from decalib.utils.config import cfg as deca_cfg
from utils.deca_encoder import deca_encode, deca_decode
import time
from tqdm import tqdm
import numpy as np
import torch
from utils.render import vis_model_out_proxy, vis_render_flmae_camera
from utils.utils import seed_everything
import matplotlib.pyplot as plt
from demo_pred_detail import split_long_audio, wav_interpolate, envelope_hilbert, init_smoothed_noise, merge_out_list
from data_provider.data_loader import all_in_flame_code
from decalib.utils import util
from utils.pose_control import generate_flame_global_pose_from_audio
import warnings
warnings.filterwarnings("ignore")

from gdl_apps.EMOCA.utils.load import load_model

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def evaluation_pipeline(args):
    checkpoint = torch.load(args.weight_path, map_location='cpu')
    with open(args.model_cfg_path) as fid:
        cfg = yaml.load(fid, Loader=yaml.Loader)

    cfg['device'] = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
    cfg["weight_dtype"] = torch.float32  

    preds_out_path = args.preds_out_path
    os.makedirs(preds_out_path, exist_ok=True)

    processor = Wav2Vec2FeatureExtractor.from_pretrained(cfg['audio_encoder_repo'])

    model = args.model_class(cfg=cfg)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model total parameters: {total_params/1e6:.2f}M")
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model trainable parameters: {trainable_params/1e6:.2f}M")

    drop_ddp_ckpts = {}
    for k, v in checkpoint.items():
        new_k = k.replace("module.", "") if k.startswith("module.") else k
        drop_ddp_ckpts[new_k] = v
    
    model.load_state_dict(drop_ddp_ckpts)
    print("Load model successfully!")
    model.cuda()
    model.eval()

    emoca_model, emoca_cfg = load_model(cfg['emoca_model_path'], cfg['emoca_model_name'], "detail")
    emoca_model.cuda()
    emoca_model.eval()

    if args.use_abs_frame_training:
        data_mean_std = np.load('./data_provider/absframe_data_mean_std.npz')
    else:
        data_mean_std = np.load('./data_provider/data_mean_std.npz')

    data_mean = torch.from_numpy(data_mean_std['data_mean'])[None,...].to(cfg['device'])
    data_std =  torch.from_numpy(data_mean_std['data_std'])[None,...].to(cfg['device'])

    testset_path = cfg['test_data_root_path']
    if args.subdataset is not None:
        if args.subdataset not in ['V0-GRID', 'V1-RAVDESS', 'V2-MEAD', 'V3-VoxCeleb2', 'V4-HDTF', 'V5-CelebV-HQ']:
            raise ValueError("subdataset must be one of ['V0-GRID', 'V1-RAVDESS', 'V2-MEAD', 'V3-VoxCeleb2', 'V4-HDTF', 'V5-CelebV-HQ']")
        else:
            dataset_list = [args.subdataset]
    else:
        dataset_list = sorted(os.listdir(testset_path))
    print(dataset_list)
    
    for dataset in dataset_list:
        data_path = os.path.join(testset_path, dataset)
        dataset_preds_out_path = os.path.join(preds_out_path, dataset)
        os.makedirs(dataset_preds_out_path, exist_ok=True)
        if args.split is None:
            all_samples = sorted(os.listdir(data_path))
        elif args.split == 'prehalf':
            all_samples = sorted(os.listdir(data_path))[:len(os.listdir(data_path))//2]
        elif args.split == 'posthalf':
            all_samples = sorted(os.listdir(data_path))[len(os.listdir(data_path))//2:]
        else:
            raise ValueError("split must be one of [None, 'prehalf', 'posthalf']")
            
        print(f"Evaluating {dataset}... with {len(all_samples)} samples")
        # all_samples = all_samples[-1:]
        out_dict = {}

        for sample in tqdm(all_samples):
            print(f"Processing sample: {sample}")
            sample_preds_out_path = os.path.join(dataset_preds_out_path, sample)
            os.makedirs(sample_preds_out_path, exist_ok=True)

            sample_path = os.path.join(data_path, sample)
            audio_path = os.path.join(sample_path, "audio.wav")
            audio_data, sr = librosa.load(audio_path, sr=16000)
            audio_data = audio_data[:16000*10] # limit to first 10s          
            audio_data = np.squeeze(processor(audio_data, sampling_rate=sr).input_values)
            audio_data_splits = split_long_audio(audio_data, processor, n_frames=cfg['num_frames'], fps=cfg['fps'])

            shapecode_path = os.path.join(sample_path, 'shapecode.npy')
            expcode_path = os.path.join(sample_path, 'expcode.npy')
            posecode_path = os.path.join(sample_path, 'posecode.npy')
            detailcode_path = os.path.join(sample_path, 'detailcode.npy')
            annot_data = torch.from_numpy(all_in_flame_code(shapecode_path, expcode_path, posecode_path, detailcode_path))
            refer_frame_codes = annot_data[0:1, :]
            refer_shape = refer_frame_codes[:, :100]
            refer_exp = refer_frame_codes[:, 100:150]
            refer_pose = refer_frame_codes[:, 150:156]
            refer_detail = refer_frame_codes[:, 156:]

            if args.control_emo is None:
                refer_frame_codes = torch.concat([refer_shape, refer_exp, refer_pose, refer_detail], dim=-1)
            else:
                emotion_list = ["natural", "angry", "contempt", "disgusted", "fear", "happy", "sad", "surprised"]
                if args.control_emo in emotion_list:
                    emo_exp_path = f"./data/exp_{args.control_emo}.npy"
                    emo_exp = torch.from_numpy(np.load(emo_exp_path))[None,:].to(refer_exp.device)
                    fused_weight = 0.75
                    emo_exp = (emo_exp * args.emo_level)*fused_weight + refer_exp*(1 - fused_weight)
                else:
                    raise ValueError(f"Unsupported emotion type: {args.control_emo}. Supported emotions: {emotion_list}")
                refer_frame_codes = torch.concat([refer_shape, emo_exp, refer_pose, refer_detail], dim=-1)

            refer_frame_list = []
            refer_frame_list.append(refer_frame_codes)

            out_list = []
            dense_out_list = []
            for split_idx, audio_data in enumerate(audio_data_splits):
                x0 = refer_frame_list[0].unsqueeze(0).to(cfg["device"])
                audio_data = audio_data.squeeze()
                audio_data = wav_interpolate(audio_data, sr=16000, fps=25)
                envelope_info = envelope_hilbert(audio_data)

                audio_data = torch.from_numpy(audio_data)
                input_audio_data = audio_data.unsqueeze(0).cuda()

                envelope_tensor = torch.from_numpy(envelope_info)
                envelope_input = envelope_tensor.unsqueeze(0).cuda()

                noisy_latents, last_frame_latent = init_smoothed_noise(batch_size=1,
                                                    n_motion_frames=int(input_audio_data.shape[-1]/16000 * cfg['fps']),
                                                    latent_dim=cfg['flame_dim'], device=cfg["device"],
                                                    dtype=cfg["weight_dtype"],
                                                    envelope=envelope_input,
                                                    lip_cof=args.lip_cof,
                                                    noise_type=args.noise_type,
                                                    use_envelope_scale=args.use_envelope_scale,
                                                    )
                delta_t = 1.0 / cfg["n_diffusion_inference_steps"]
                for tau in range(cfg["n_diffusion_inference_steps"]):
                    timesteps = int(tau / cfg["n_diffusion_inference_steps"] * cfg["n_diffusion_train_steps"])
                    timesteps = torch.tensor([timesteps,], device=cfg["device"])
                    timesteps = timesteps.repeat([noisy_latents.shape[0]])
                    with torch.no_grad():
                        pred_velocity = model(latent=noisy_latents,
                                            x0 = x0,
                                            audio=input_audio_data,
                                            envelope= envelope_input, 
                                            timestep=timesteps,
                                            )
                        noisy_latents = noisy_latents + pred_velocity * delta_t
                
                # denorm
                pred_delta_annot = (noisy_latents*data_std) + data_mean

                if args.use_abs_frame_training:
                    abs_flame_params = pred_delta_annot
                else:
                    abs_flame_params = x0 + pred_delta_annot
                # abs_flame_params[:, 0:1, :] = x0

                abs_flame_params[:, :, 153].clamp_(min=0, max=0.3) # 0.3
                abs_flame_params[:, :, 154].clamp_(min=-0.1, max=0.1)
                abs_flame_params[:, :, 155].clamp_(min=-0.1, max=0.1)
                
                if args.fix_head:
                    abs_flame_params[:, :, 150:153] = 0
                elif args.head_pose_control:
                    pose_traje = generate_flame_global_pose_from_audio(
                            audio_data.cpu().numpy(),
                            fps=25,
                            yaw_max_deg=16, pitch_max_deg=10, roll_max_deg=6,
                            seed=0, return_features=False
                        )
                    head_pose_tensor = torch.from_numpy(pose_traje)[None, ...].to(cfg['device'])
                    abs_flame_params[:, :, 150:153] = head_pose_tensor
            
                pred_flame_params = abs_flame_params[0]           # [1, f, 284]  -> [f, 284]

                if args.save_flame_params:
                    flame_params_path = os.path.join(sample_preds_out_path, f"flame_params.npy")
                    np.save(flame_params_path, pred_flame_params.detach().cpu().numpy())
                verts, detail_verts = deca_decode(emoca_model, pred_flame_params.float())   #[f, 5023, 3]  [f, 59315, 3]  
                # save out verts
                sf, n, v =  verts.shape
                verts = verts.reshape((sf, -1))
                out_list.append(verts.detach().cpu().numpy())
                if args.use_details:
                    dense_sf, dense_n, dense_v = detail_verts.shape
                    detail_verts_merge = detail_verts.reshape((dense_sf, -1))
                    dense_out_list.append(detail_verts_merge.detach().cpu().numpy())
            out = merge_out_list(out_list, n_frames=cfg['num_frames'], fps=cfg['fps'])    # (f,  5023*3)
            out = out.reshape((out.shape[0], n, v))
            verts_path = os.path.join(sample_preds_out_path, f"verts_seq.npy")
            np.save(verts_path, out)
            out_dict[sample] = out
            if args.use_details:
                dense_out = merge_out_list(dense_out_list, n_frames=cfg['num_frames'], fps=cfg['fps'])
                dense_out = dense_out.reshape((dense_out.shape[0], detail_verts.shape[1], detail_verts.shape[2]))
                dense_verts_path = os.path.join(sample_preds_out_path, f"detail_verts_seq.npy")
                np.save(dense_verts_path, dense_out)
                out_dict[f"{sample}-details"] = dense_out
            
            if args.save_video:
                npz_path = os.path.join(sample_preds_out_path, f"preds_dict.npz")
                print(f"save results to {npz_path}")
                np.savez(npz_path, **out_dict)
                vis_model_out_proxy(
                                    npz_path = npz_path,
                                    audio_dir =  "",
                                    out_dir = sample_preds_out_path,
                                    test_sample_dir = sample_path,
                                    fix_cam=args.fix_cam
                                )            
        
def main():  
    
    parser = argparse.ArgumentParser(description="3D Talking training.")
    parser.add_argument("--weight_path", type=str, default="./checkpoints_output/checkpoints_v6/epoch_100/model.pth")
    parser.add_argument("--preds_out_path", type=str, default="./inference_results")
    parser.add_argument("--model_cfg_path", type=str, default="./config/default_config.yaml")
    parser.add_argument('--use_details', default=False, type=lambda x: x.lower() in ['true', '1'], help='whether to use details')
    parser.add_argument('--fix_head', default=False, type=str2bool, help='whether to fix head')
    parser.add_argument('--fix_cam',  default=True, type=str2bool, help='whether to fix camera')
    parser.add_argument('--save_video',  default=True, type=str2bool, help='whether to fix camera')
    parser.add_argument('--gpu_number', default=3, type=int, help='gpu_number')
    parser.add_argument("--subdataset", type=str, default='V4-HDTF', help="['V0-GRID', 'V1-RAVDESS', 'V2-MEAD', 'V3-VoxCeleb2', 'V4-HDTF', 'V5-CelebV-HQ']")
    parser.add_argument("--split", type=str, default=None, help="['prehalf', 'posthalf']")
    parser.add_argument("--save_flame_params", default=True, type=lambda x: x.lower() in ['true', '1'], help="Whether to save FLAME parameters")
    parser.add_argument('--control_emo', default=None, type=str, help='emotion type:  ["neutral",  "happy", "angry", "contempt", "disgusted", "fear", "sad", "surprised"]')
    parser.add_argument('--emo_level', default=1.0, type=float, help='emotion level: 1.0 to 2.0')
    parser.add_argument('--head_pose_control', default=True, type=lambda x: x.lower() in ['true', '1'], help='whether to use audio-based head pose control')

    parser.add_argument(
        "--noise_type",
        type=str,
        default="seg_interp_25",
        help="Type of noise to use: random, seg_interp_25, seg_interp_250",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="TalkingHead3D_Evlp_v6",
        help="Model name to use",
    )

    parser.add_argument(
        "--lip_cof",
        type=float,
        default=1.0,
        help="lip scale coefficient during noise initialization",
    )

    parser.add_argument(
        "--use_envelope_scale",
        action="store_false",
        help="Whether to use envelope scaling in noise initialization",
    )


    parser.add_argument(
        "--use_abs_frame_training",
        action="store_true",
        help="Whether to use absolute frame training",
    )

    args, _ = parser.parse_known_args()

    settings = f"ModelName-{args.model_name}-NoiseType-{args.noise_type}-use_envelope_scale-{str(args.use_envelope_scale)}-LipScale-{args.lip_cof}-UseAbsFrameTraining-{str(args.use_abs_frame_training)}"
    import importlib
    model_name = args.model_name
    if model_name is not None:
        module_path = f"models.{model_name}"
        model_module = importlib.import_module(module_path)
        MyModel = getattr(model_module, "MyModel")
        args.model_class = MyModel
        
    torch.cuda.set_device(args.gpu_number)
    seed_everything(42)
    evaluation_pipeline(args)

if __name__ == "__main__":
    main()