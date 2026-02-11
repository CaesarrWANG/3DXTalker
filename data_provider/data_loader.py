import librosa
import numpy as np
from typing import Any
from torch.utils.data import Dataset
import os
from transformers import Wav2Vec2FeatureExtractor
from scipy import signal
from scipy.signal import hilbert
import torch
from tqdm import tqdm


def get_envelope_from_audio(data_root='/scratch3/wan451/3DTalk/emoca/video_output/EMOCA_v2_lr_mse_20/trainset'):
    data_files = sorted(os.listdir(data_root))
    def envelope_hilbert(signal):
        analytic_signal = hilbert(signal)
        envelope = np.abs(analytic_signal)
        return envelope
    for data_name in tqdm(data_files):
        audio_path = os.path.join(data_root, data_name, data_name+'.wav')
        audio_data, sr = librosa.load(audio_path, sr=16000)
        envelope_info = envelope_hilbert(audio_data)
        out_path = os.path.join(data_root, data_name, data_name+'-evlp.npy')
        np.save(out_path, envelope_info)

def distribution_analysis(data_root='/scratch3/wan451/3DTalk/emoca/video_output/EMOCA_v2_lr_mse_20/trainset'):
    data_files = sorted(os.listdir(data_root))
    hdtf_global_statics_info = {}
    lst_collect = []
    for data_name in tqdm(data_files):
        frames_path = os.path.join(data_root, data_name, data_name+'-AllInOne.npy')
        frames_seq = np.load(frames_path)   #[f, d]
        delta_seq = frames_seq - frames_seq[0:1, :]
        lst_collect.append(delta_seq)
    out = np.vstack(lst_collect) #[n*f, d]
    hdtf_global_statics_info['data_mean'] = np.mean(out, axis=0) 
    hdtf_global_statics_info['data_std'] = np.std(out, axis=0)     
    out_path = '/scratch3/wan451/3DTalk/The-Sound-of-Motion/data_provider/hdtf_global_statics_info.npz'
    np.savez(out_path, **hdtf_global_statics_info)



def compute_global_mean_std(root_dir, key='shape', use_absolute_frame=False):
    mean = None
    M2 = None
    count = 0

    for dataset in sorted(os.listdir(root_dir)):
        dataset_path = os.path.join(root_dir, dataset)
        if not os.path.isdir(dataset_path):
            continue

        print(f"Processing {dataset} ...")
        for sample in os.listdir(dataset_path):
            sample_path = os.path.join(dataset_path, sample)
            npy_path = os.path.join(sample_path, f"{key}code.npy")
            if not os.path.exists(npy_path):
                continue

            data = np.load(npy_path)  # shape: (T, D)
            if use_absolute_frame:
                delta_data = data 
            else:
                delta_data = data - data[0:1, :]  # (T, D)

            # Flatten time dimension → (T, D)
            batch_mean = np.mean(delta_data, axis=0)
            batch_M2 = np.var(delta_data, axis=0) * delta_data.shape[0]
            batch_count = delta_data.shape[0]

            if mean is None:
                mean = batch_mean
                M2 = batch_M2
                count = batch_count
            else:
                # Combine batch statistics with Welford merge
                delta = batch_mean - mean
                total_count = count + batch_count

                mean = mean + delta * batch_count / total_count
                M2 = M2 + batch_M2 + delta**2 * (count * batch_count / total_count)
                count = total_count

    std = np.sqrt(M2 / count)
    return mean, std


def summarize_Data_path(data_root_path):
    gt_data_path = {}
    dataset_list = sorted(os.listdir(data_root_path))
    for dataset in dataset_list:
        dataset_path = os.path.join(data_root_path, dataset)
        if not os.path.exists(dataset_path):
            print(f"[WARNING] {dataset_path} not found, skipping...")
            continue
        video_list = []
        for subject in sorted(os.listdir(dataset_path)):
            subject_dir = os.path.join(dataset_path, subject)
            if not os.path.isdir(subject_dir):
                continue
            video_list.append(subject_dir)
        gt_data_path[dataset] = video_list
        print(f"[INFO] Found {len(video_list)} samples in {dataset}")
    return gt_data_path

def all_in_flame_code(shape_path, exp_path, pose_path, detail_path):
    shape = np.load(shape_path)  # [1, 100]
    exp = np.load(exp_path)     # [1, 50]
    pose = np.load(pose_path)   # [1, 6]
    detail = np.load(detail_path) # [1, 128]
    return np.concatenate([shape, exp, pose, detail], axis=-1) # 284

class TalkingDataset(Dataset):

    def __init__(self, cfg, flag="train"):

        self.train_data_root_path = cfg['train_data_root_path']
        self.test_data_root_path = cfg['test_data_root_path']
        self.data_files = []
        self.fps = cfg['fps']
        self.sr = cfg['sample_rate']
        self.n_frames = cfg['num_frames']
        self.flag = flag
        # self.batch_sample =cfg['batch_sample']
        assert flag in ['train', 'test']
        if flag == "train":
            dict_train_data = summarize_Data_path(self.train_data_root_path)
            if "HDTF" in dict_train_data:
                dict_train_data["HDTF"] = dict_train_data["HDTF"]*4
            for dataset, file_list in dict_train_data.items():
                self.data_files.extend(file_list)
            print(f"Total {len(self.data_files)} training set")
        elif flag == "test":
            dict_test_data = summarize_Data_path(self.test_data_root_path)
            for dataset, file_list in dict_test_data.items():
                self.data_files.extend(file_list)
            print(f"Total {len(self.data_files)} testing set")
        else:
            raise RuntimeError(f"Unsupported flag: {flag}")

        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(cfg['audio_encoder_repo'])

    def truncate(self, audio_array: np.ndarray, sr: int, annot_data: np.ndarray, fps: int):
        audio_duration = int(len(audio_array) / sr)
        annot_duration = int(len(annot_data) / fps)
        duration = min(audio_duration, annot_duration)
        
        audio_length = int(duration * sr)
        annot_length = int(duration * fps)
        # self.duration = duration
        return audio_array[:audio_length], annot_data[:annot_length]
    
    def __getitem__(self, index):
        is_success = False
        while not is_success:
            try:
                sample = self._prepare_sample(index=index)
                is_success = True
            except Exception as e:
                # print(e)
                index = np.random.randint(0, len(self.data_files)-1)
                continue
        return sample
    
    def _prepare_sample(self, index):
        fps= self.fps
        audio_path = os.path.join(self.data_files[index], 'audio.wav')
        audio_data, sr = librosa.load(audio_path, sr=self.sr)

        shapecode_path = os.path.join(self.data_files[index], 'shapecode.npy')
        expcode_path = os.path.join(self.data_files[index], 'expcode.npy')
        posecode_path = os.path.join(self.data_files[index], 'posecode.npy')
        detailcode_path = os.path.join(self.data_files[index], 'detailcode.npy')
        annot_data = all_in_flame_code(shapecode_path, expcode_path, posecode_path, detailcode_path)
        ref_frame_annot = annot_data[0:1, :]
        # delta_annot_data = annot_data - ref_frame_annot  # delta

        evlp_path = os.path.join(self.data_files[index], 'envelope.npy')
        evlp_data = np.load(evlp_path)

        if (len(annot_data)-self.n_frames) <=0:
            raise RuntimeError(f"Video {self.data_files[index]} is too short, pass it")
        
        audio_data, annot_data = self.truncate(audio_data, sr, annot_data, fps)
        evlp_data, _ =  self.truncate(evlp_data, sr, annot_data, fps)

        assert len(audio_data)/(sr/fps) == len(annot_data)
        assert len(evlp_data)/(sr/fps) == len(annot_data)

        start_idx = np.random.randint(0, len(annot_data)-self.n_frames, size=1)
        end_idx = start_idx + self.n_frames
        
        frame_idx = np.array([np.linspace(s, e, self.n_frames, dtype=int) for s, e in zip(start_idx, end_idx)])
        sample_annota = np.take(annot_data, frame_idx, axis=0)
 
        samples_per_frame = int(sr/fps)

        sample_base = frame_idx * samples_per_frame 
        offset = np.arange(samples_per_frame)[None, None, :]
        audio_indices = sample_base[..., None] + offset
        sample_audio = np.take(audio_data, audio_indices)
        sample_audio = sample_audio.reshape((1, -1))

        sample_evlp = np.take(evlp_data, audio_indices)
        sample_evlp = sample_evlp.reshape((1, -1))
        
        sample_audio = sample_audio.squeeze(0)
        sample_evlp = sample_evlp.squeeze(0)
        # ref_frame_annot = ref_frame_annot.squeeze(0)
        sample_annota = sample_annota.squeeze(0)

        item = {
            'audio_data': sample_audio,
            'envelope_data': sample_evlp,
            'ref_frame_annot': ref_frame_annot,
            'annot_data': sample_annota,
        }
        return item


    def __len__(self, ):
        return len(self.data_files)

    def scale_and_offset(self,
                         data: np.ndarray,
                         scale: float = 1.0,
                         offset: np.ndarray = 0.0):
        return data * scale + offset



if __name__ == "__main__":
    import yaml
    with open("./config/default_config.yaml", "r") as fid:
        cfg = yaml.load(fid, Loader=yaml.Loader)
    
    cfg['train_data_root_path'] = '/scratch3/wan451/3DTalk/emoca/video_output/EMOCA_v2_lr_mse_20/trainset'
    cfg['test_data_root_path'] = '/scratch3/wan451/3DTalk/emoca/video_output/EMOCA_v2_lr_mse_20/testset'

    # distribution_analysis(data_root='/scratch3/wan451/3DTalk/The-Sound-of-Motion/dataset/hdtf400_processed')
    use_absolute_frame = True

    shape_mean, shape_std = compute_global_mean_std(cfg['train_data_root_path'], key="shape", use_absolute_frame=use_absolute_frame)
    exp_mean, exp_std = compute_global_mean_std(cfg['train_data_root_path'], key="exp",use_absolute_frame=use_absolute_frame)
    pose_mean, pose_std = compute_global_mean_std(cfg['train_data_root_path'], key="pose", use_absolute_frame=use_absolute_frame)
    detail_mean, detail_std = compute_global_mean_std(cfg['train_data_root_path'], key="detail", use_absolute_frame=use_absolute_frame)
    
    data_mean = np.concatenate([shape_mean, exp_mean, pose_mean, detail_mean], axis=-1)
    data_std = np.concatenate([shape_std, exp_std, pose_std, detail_std], axis=-1)
    print(data_mean.shape, data_std.shape) #(284,) (284,)
    out_path = '/scratch3/wan451/3DTalk/The-Sound-of-Motion/data_provider/absframe_data_mean_std.npz'
    np.savez(out_path, data_mean=data_mean, data_std=data_std)
    print(f"Save global mean and std to {out_path}")

    dataset = TalkingDataset(        
        cfg=cfg, flag='train'
    )
    
    for ii, batch in enumerate(dataset):
        print(f"audio_data shape: {batch['audio_data'].shape}")   # (320000, )   # 320000/640 = 500
        print(f"envelope_data shape:  {batch['envelope_data'].shape}") # (320000, )   # 320000/640 = 500
        print(f"ref_frame_annot shape: {batch['ref_frame_annot'].shape}")   # (1, 334)   #100+50+6+128+50
        print(f"annot_data shape: {batch['annot_data'].shape}")   # (500, 334)   #100+50+6+128+50
        # print(f"delta_annot_data shape: {batch['delta_annot_data'].shape}")   # (500, 334)   #100+50+6+128+50
        if ii >= 2:
            break
    print("done")
