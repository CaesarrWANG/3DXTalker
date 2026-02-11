
import os, sys

import torch.distributed
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
os.environ["TORCH_DISTRIBUTED_TIMEOUT"] = "1200"
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:False"

import argparse
import numpy as np
import yaml, json
import torch
import glob
import shutil
import time
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader                            
from accelerate import Accelerator
from data_provider.data_loader import TalkingDataset
from models import _3DXTalker
from utils.utils import dump_jsonl, get_average_meter_dict, get_logger, log_datasetloss, write_to_tensorboard, seed_everything
from tqdm import tqdm
from tensorboardX import SummaryWriter
from utils.utils import init_smoothed_noise

def main(cfg):
    # Initialize accelerator
    # dataloader_config = DataLoaderConfiguration(use_stateful_dataloader=cfg.use_stateful_dataloader)

    seed_everything(cfg["seed"])
    # accelerator = Accelerator(
    #     cpu=cfg["cpu"], mixed_precision=cfg["mixed_precision"], dataloader_config=None, kwargs_handlers=[ddp_kwargs])
    
    accelerator = Accelerator(
        cpu=cfg["cpu"], mixed_precision=cfg["mixed_precision"], dataloader_config=None)
    
    accelerator.print("Accelerator processes:", accelerator.num_processes)

    log_file = os.path.join(cfg['log_path'], 'train.log')
    logger = get_logger(log_file)

    cfg['writer'] = SummaryWriter(cfg["events_path"])

    device = accelerator.device    
    cfg["machine_rank"] = accelerator.process_index
    cfg["world_size"] = accelerator.num_processes
    cfg["device"] = device
    if cfg["dtype"] == "fp16":
        weight_dtype = torch.half
    if cfg["dtype"] == "bf16":
        weight_dtype = torch.bfloat16
    else:
        assert cfg["dtype"] == "fp32"
        weight_dtype = torch.float32                            
    cfg["weight_dtype"] = weight_dtype

    lr = cfg["lr"]
    num_epochs = int(cfg["num_epochs"])
    checkpointing_steps = cfg["checkpointing_steps"]

    train_dataset = TalkingDataset(cfg=cfg, flag='train')
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=cfg["batch_size"], num_workers=cfg["num_workers"])

    if cfg.get('use_abs_frame_training', False):
        data_mean_std = np.load('./data_provider/absframe_data_mean_std.npz')
    else:
        data_mean_std = np.load('./data_provider/data_mean_std.npz')
    data_mean = torch.from_numpy(data_mean_std['data_mean'])[None,...].to(device) # [1, 284]
    data_std =  torch.from_numpy(data_mean_std['data_std'])[None,...].to(device) # [1, 284]
    
    logger.info('=> creating model ...')

    cfg['logger'] = logger
    model = cfg["model_class"](cfg=cfg)

    cfg['train_meter'], cfg['train_metric_name_list'] = get_average_meter_dict(phase='train')
    cfg['val_meter'], cfg['val_metric_name_list'] = get_average_meter_dict(phase='val')
    cfg['test_meter'], cfg['test_metric_name_list'] = get_average_meter_dict(phase='test')


    if hasattr(model, "module"):
        model.module.audio_encoder._freeze_wav2vec2_parameters(True)
    else:
        model.audio_encoder._freeze_wav2vec2_parameters(True)
    
    # Instantiate optimizer
    lr = float(cfg["lr"])
    if cfg["optimizer"] == "adam":
        optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    elif cfg["optimizer"] == "adamW":
        optimizer = torch.optim.AdamW(params=filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=0.01)
    elif cfg["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(params=filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    else:
        raise RuntimeError(" Unknown optimizer: " + cfg["optimizer"])

    # Instantiate learning rate scheduler
    lr_scheduler = OneCycleLR(optimizer=optimizer, max_lr=lr, epochs=num_epochs, steps_per_epoch=len(train_loader), pct_start=0.0)
    
    # Prepare everything
    model, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, lr_scheduler
    )
    overall_step = 0
    starting_epoch = 0
    
    is_found_resume_checkpoint = False
    resume_global_step = 0
    resume_step = 0
    if cfg["resume_from_checkpoint"] != "None" and cfg["load_from_checkpoint"] is None:
        if cfg["resume_from_checkpoint"] != "latest":
            path = os.path.basename(cfg["resume_from_checkpoint"])
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(cfg["resume_from_checkpoint"])
            is_found_resume_checkpoint = True
        else:
            # Get the most recent checkpoint
            if os.path.exists(os.path.join(cfg["output_dir"], "checkpoints")):
                dirs = os.listdir(os.path.join(cfg["output_dir"], "checkpoints"))                
                dirs = [d for d in dirs if d.startswith("step_")]
                if len(dirs) >= 1:
                    dirs = sorted(dirs, key=lambda x: int(x.split("_")[1]))
                    path = dirs[-1] if len(dirs) > 0 else None
                    accelerator.print(f"Resuming from checkpoint {path}")
                    accelerator.load_state(os.path.join(cfg["output_dir"], "checkpoints", path))
                    is_found_resume_checkpoint = True
            
        if is_found_resume_checkpoint:
            global_step = int(path.split("_")[1])
            num_update_steps_per_epoch = len(train_loader)
            resume_global_step = global_step
            starting_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch)
            overall_step = resume_global_step
        else:
            resume_global_step = 0
            resume_step = 0
            
    if cfg["load_from_checkpoint"] is not None:
        accelerator.print(f"Load model weights from checkpoint {cfg['load_from_checkpoint']}")
        accelerator.load_state(cfg["load_from_checkpoint"])
    

    for epoch in range(starting_epoch, num_epochs):
        epoch_plus = epoch + 1       
        model.train()
        train_phbar = tqdm(train_loader, disable= not accelerator.is_main_process, desc=f'Epoch: {epoch} train')

        for batch in train_phbar:
            with accelerator.accumulate(model):
                
                batch = {k: v.to(accelerator.device, dtype=weight_dtype) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                audio_data = batch["audio_data"]     
                annot_data = batch["annot_data"]  # (b, f, d)
                envelope_data = batch["envelope_data"]
                ref_frame_annot = batch["ref_frame_annot"]
                if cfg.get('use_abs_frame_training', False):
                    delta_annot = annot_data
                else:
                    delta_annot = annot_data - ref_frame_annot
        
                bs, f , d = batch["annot_data"].shape     #[b, n_frames, 284]

                if cfg["n_diffusion_train_steps"] == 1:
                        timesteps = torch.zeros((bs,), device=device, dtype=torch.long)
                else:
                    timesteps = torch.randint(0, cfg["n_diffusion_train_steps"], (bs,), device=device, dtype=torch.long)
                
                data_mean = data_mean.to(dtype=weight_dtype)
                data_std = data_std.to(dtype=weight_dtype)
                # data normalization
                normalized_delta_annot = (delta_annot - data_mean)/data_std
                first_frame_delta_annot = normalized_delta_annot[:, 0:1, :]  # (b, 1, d)

                # Initial noise
                initial_noise, _ = init_smoothed_noise(batch_size=bs,
                                                    n_motion_frames=f,
                                                    latent_dim=d,
                                                    device=device,
                                                    dtype=weight_dtype,
                                                    envelope=envelope_data,
                                                    lip_cof=cfg['lip_cof'],
                                                    noise_type=cfg["noise_type"],
                                                    use_envelope=cfg["use_envelope"],
                                                )

                t_float = timesteps.reshape([bs, 1, 1]) / cfg["n_diffusion_train_steps"]

                noisy_latents = t_float * normalized_delta_annot + (1-t_float) * initial_noise

                pred_velocity = model(
                    latent=noisy_latents,
                    x0 = ref_frame_annot,
                    audio= audio_data,
                    envelope= envelope_data,
                    timestep=timesteps,
                    )

                pred_velocity = pred_velocity.to(dtype=cfg["weight_dtype"])            
                gt_velocity = normalized_delta_annot - initial_noise

                loss = torch.nn.functional.mse_loss(pred_velocity.float(), gt_velocity.float())
                
                for k, v in zip(cfg['train_metric_name_list'], [loss]):
                    cfg['train_meter'][k].update(v.item(), n =len(annot_data))
                
                accelerator.backward(loss)

                if cfg["clip_grad_norm"] is not None and cfg["clip_grad_norm"] > 1e-6:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip_grad_norm"])

                print_noGrad_param = False
                if print_noGrad_param:
                    for name, param in model.named_parameters():
                        if param.grad is None:
                            print(name)
            
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                overall_step += 1
                        
                if overall_step in (1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000) or overall_step % 1000 == 0:
                    with torch.no_grad():                    
                        train_metrics = {}
                        train_metrics["loss"] = float(loss.detach().item())
                        train_metrics["overall_step"] = overall_step
                        train_metrics["lr"] = lr_scheduler.get_last_lr()[0]                    
                        if accelerator.is_main_process:
                            dump_jsonl(
                                os.path.join(cfg["output_dir"], "train_metrics.log"),
                                [train_metrics,],
                            )
                            accelerator.print(f"[{overall_step}]" + str(train_metrics))

        log_datasetloss(cfg['logger'], epoch, 'train', cfg['train_meter'])
        write_to_tensorboard(cfg['writer'], epoch, 'train', cfg['train_meter'])

        for loss_type in cfg['train_meter'].keys():
                cfg['train_meter'][loss_type].reset()

        if (epoch_plus in [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]) and accelerator.is_main_process:
            save_dir = os.path.join(cfg["output_dir"], f"checkpoints", f"epoch_{epoch_plus}")
            accelerator.save_state(save_dir)
            torch.save(model.state_dict(), os.path.join(save_dir, "model.pth"))
            # clean checkpoints
            all_saved_model_paths = glob.glob(os.path.join(cfg["output_dir"], "checkpoints", "epoch_*"))
            if len(all_saved_model_paths) > 10:
                all_saved_model_paths.sort(key=lambda x: int(os.path.basename(x).split("_")[-1]))                
                for a_path in all_saved_model_paths[0:-10]:
                    shutil.rmtree(a_path)


        accelerator.wait_for_everyone()

    accelerator.end_training()

if __name__ == "__main__":
    print("~~~~~"*20)
    print("Starting training script at ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    parser = argparse.ArgumentParser(description="3DXTalker training pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="./config/default_config.yaml",
    )
    parser.add_argument(
        "--noise_type",
        type=str,
        default="seg_interp_25",
        help="Type of noise to use: random, seg_interp_25, seg_interp_250",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="_3DXTalker",
        help="Model name to use",
    )

    parser.add_argument(
        "--lip_cof",
        type=float,
        default="1.0",
        help="lip scale coefficient during noise initialization",
    )

    parser.add_argument(
        "--use_envelope",
        action="store_false",
        help="Whether to use envelope scaling in noise initialization",
    )


    parser.add_argument(
        "--use_abs_frame_training",
        action="store_true",
        help="Whether to use absolute frame training",
    )


    args, _ = parser.parse_known_args()
    with open(args.config) as fid:
        cfg = yaml.load(fid, Loader=yaml.Loader)

    args_dict = vars(args)

    for k, v in args_dict.items():
        if k in cfg and v is not None:
            cfg[k] = v
        if k not in cfg:
            cfg[k] = v

    import importlib
    model_name = cfg.get("model_name", None)
    if model_name is not None:
        module_path = f"models.{model_name}"
        model_module = importlib.import_module(module_path)
        MyModel = getattr(model_module, "MyModel")
        cfg["model_class"] = MyModel

    exp_settings = f"ModelName-{args.model_name}-use_envelope{str(args.use_envelope)}-LipScale-{args.lip_cof}-UseAbsFrameTraining-{str(args.use_abs_frame_training)}-lr-{cfg['lr']}-BatchSize-{cfg['batch_size']}"
    save_output_dir = os.path.join("./checkpoints_output/", exp_settings)
    cfg.update({"output_dir": save_output_dir})

    # save a copy of cfg
    os.makedirs(cfg["output_dir"], exist_ok=True)
    with open(os.path.join(cfg["output_dir"], "cfg.yaml"), "w") as fid:
        yaml.dump(cfg, fid)
        
    main(cfg)
    print("~~~~~"*20)
    print("Ending training script at ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))