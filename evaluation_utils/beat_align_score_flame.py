import numpy as np
import librosa
from scipy.ndimage import gaussian_filter as G
from scipy.signal import argrelextrema
import os


def compute_beat_alignment_score(audio, flame_parameters_path, fps=30, audio_method='onset', 
                                  pose_sigma=5, variance=9):
    """
    计算FLAME参数与音频的beat alignment score
    
    Args:
        audio: 音频数据，可以是:
               - str: 音频文件路径 (如 'audio.wav')
               - np.ndarray: 音频波形数组 [samples,] 或 [samples, channels]
               - tuple: (audio_array, sample_rate)
        flame_parameters: FLAME pose参数
                         - np.ndarray: shape [frames, pose_dim]
                           例如 head_pose (3,) 或 head_pose + jaw_pose (6,)
        fps: 视频帧率，默认30
        audio_method: 'onset' (语音起始点，推荐用于talking head) 或 'beat' (音乐节拍)
        pose_sigma: pose速度平滑的高斯核标准差，默认5
        variance: 对齐评分的容忍度参数，默认9 (标准差=3帧)
    
    Returns:
        float: beat alignment score (0-1之间，越高表示对齐越好)
    
    Example:
        >>> # 从文件路径
        >>> score = compute_beat_alignment_score('audio.wav', pose_params)
        >>> 
        >>> # 从音频数组
        >>> audio_array, sr = librosa.load('audio.wav')
        >>> score = compute_beat_alignment_score((audio_array, sr), pose_params)
        >>> 
        >>> # 或直接传数组（默认sr=22050）
        >>> score = compute_beat_alignment_score(audio_array, pose_params)
    """
    # 处理音频输入
    if isinstance(audio, str):
        # 文件路径
        y, sr = librosa.load(audio, sr=16000)
        y = y[:16000 * 60]  # 最多加载60秒
    elif isinstance(audio, tuple):
        # (audio_array, sample_rate)
        y, sr = audio
        y = y[:16000 * 60]  # 最多加载60秒
    elif isinstance(audio, np.ndarray):
        # 只有音频数组，使用默认采样率
        y = audio
        y = y[:16000 * 60]  # 最多加载60秒
        sr = 16000
    else:
        raise ValueError(f"Unsupported audio type: {type(audio)}")
    flame_parameters = np.load(flame_parameters_path)
    pose_para = flame_parameters[:, 150:153]
    # 提取音频节拍
    audio_beats = _extract_audio_beats_from_array(y, sr, fps=fps, method=audio_method)
    
    # 提取动作节拍
    motion_beats, length = _calc_pose_beats(pose_para, sigma=pose_sigma)
    
    # 过滤超出范围的音频节拍
    audio_beats = audio_beats[audio_beats < length]
    
    # 计算对齐分数
    score = _beat_align_score(audio_beats, motion_beats, variance=variance)
    
    return score


def _extract_audio_beats_from_array(y, sr, fps=30, method='onset'):
    """从音频数组中提取节拍点（内部函数）"""
    if method == 'onset':
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env, 
            sr=sr,
            backtrack=True
        )
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    elif method == 'beat':
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        onset_times = librosa.frames_to_time(beat_frames, sr=sr)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    beat_frames = (onset_times * fps).astype(int)
    return beat_frames


def _calc_pose_beats(pose_params, sigma=2):
    """从FLAME pose参数中提取动作节拍点（内部函数）"""
    pose_params = np.array(pose_params)
    pose_velocity = np.sqrt(np.sum((pose_params[1:] - pose_params[:-1]) ** 2, axis=1))
    pose_velocity = G(pose_velocity, sigma)
    motion_beats = argrelextrema(pose_velocity, np.less)
    return motion_beats, len(pose_velocity)


def _beat_align_score(audio_beats, motion_beats, variance=2):
    """计算音频节拍与动作节拍的对齐分数（内部函数）"""
    if len(audio_beats) == 0:
        return 0.0
    
    ba = 0
    for audio_beat in audio_beats:
        distances = (motion_beats[0] - audio_beat) ** 2
        min_distance = np.min(distances)
        ba += np.exp(-min_distance / (2 * variance))
    
    return ba / len(audio_beats)


def extract_audio_beats(audio_path, fps=25, method='onset'):
    """
    从音频文件中提取节拍点
    
    Args:
        audio_path: 音频文件路径
        fps: 视频帧率，用于将时间转换为帧索引
        method: 'onset' (语音起始点) 或 'beat' (音乐节拍)
    
    Returns:
        beat_frames: 节拍对应的帧索引数组
    """
    # 加载音频
    y, sr = librosa.load(audio_path, sr=None)
    
    if method == 'onset':
        # 检测语音起始点（适合talking head）
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env, 
            sr=sr,
            backtrack=True
        )
        # 转换为时间
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    elif method == 'beat':
        # 检测音乐节拍
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        onset_times = librosa.frames_to_time(beat_frames, sr=sr)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # 转换为视频帧索引
    beat_frames = (onset_times * fps).astype(int)
    
    return beat_frames


def calc_pose_beats(pose_params, sigma=5):
    """
    从FLAME pose参数中提取动作节拍点
    
    Args:
        pose_params: [frames, pose_dim] FLAME pose参数
                    例如: head_pose (3,) 或 head_pose + jaw_pose (6,)
        sigma: 高斯平滑的标准差
    
    Returns:
        motion_beats: 动作节拍的帧索引
        length: 序列总长度
    """
    pose_params = np.array(pose_params)
    
    # 计算姿态参数的变化速度（欧氏距离）
    pose_velocity = np.sqrt(np.sum((pose_params[1:] - pose_params[:-1]) ** 2, axis=1))
    
    # 高斯平滑
    pose_velocity = G(pose_velocity, sigma)
    
    # 找局部最小值点（动作的"停顿点"或"重拍点"）
    motion_beats = argrelextrema(pose_velocity, np.less)
    
    return motion_beats, len(pose_velocity)


def beat_align_score(audio_beats, motion_beats, variance=9):
    """
    计算音频节拍与动作节拍的对齐分数
    
    Args:
        audio_beats: 音频节拍的帧索引数组
        motion_beats: 动作节拍的帧索引（tuple格式，来自argrelextrema）
        variance: 高斯核的方差参数，控制容忍度
    
    Returns:
        ba_score: 对齐分数 (0-1之间，越高越好)
    """
    if len(audio_beats) == 0:
        return 0.0
    
    ba = 0
    for audio_beat in audio_beats:
        # 计算当前音频节拍到所有动作节拍的距离
        distances = (motion_beats[0] - audio_beat) ** 2
        min_distance = np.min(distances)
        
        # 使用高斯核函数计算相似度
        ba += np.exp(-min_distance / (2 * variance))
    
    # 归一化
    return ba / len(audio_beats)


def evaluate_flame_pose_alignment(pose_params, audio_path, fps=30, 
                                   audio_method='onset', 
                                   pose_sigma=5, 
                                   variance=9):
    """
    评估FLAME pose与音频的对齐程度
    
    Args:
        pose_params: [frames, pose_dim] FLAME pose参数
        audio_path: 音频文件路径
        fps: 视频帧率
        audio_method: 'onset' 或 'beat'
        pose_sigma: pose速度平滑参数
        variance: 对齐评分的容忍度参数
    
    Returns:
        score: beat alignment score
        audio_beats: 音频节拍帧索引
        motion_beats: 动作节拍帧索引
    """
    # 提取音频节拍
    audio_beats = extract_audio_beats(audio_path, fps=fps, method=audio_method)
    
    # 提取动作节拍
    motion_beats, length = calc_pose_beats(pose_params, sigma=pose_sigma)
    
    # 过滤超出范围的音频节拍
    audio_beats = audio_beats[audio_beats < length]
    
    # 计算对齐分数
    score = beat_align_score(audio_beats, motion_beats, variance=variance)
    
    return score, audio_beats, motion_beats


def batch_evaluate(pose_dir, audio_dir, fps=30, audio_method='onset'):
    """
    批量评估多个样本
    
    Args:
        pose_dir: pose参数文件目录 (.npy或.npz格式)
        audio_dir: 音频文件目录
        fps: 视频帧率
        audio_method: 'onset' 或 'beat'
    
    Returns:
        mean_score: 平均对齐分数
        scores: 每个样本的分数列表
    """
    scores = []
    
    for pose_file in os.listdir(pose_dir):
        if not (pose_file.endswith('.npy') or pose_file.endswith('.npz')):
            continue
        
        # 加载pose参数
        pose_path = os.path.join(pose_dir, pose_file)
        if pose_file.endswith('.npy'):
            pose_params = np.load(pose_path)
        else:
            pose_data = np.load(pose_path)
            # 假设pose参数在'pose'键中，根据实际情况调整
            pose_params = pose_data['pose']
        
        # 找对应的音频文件
        base_name = os.path.splitext(pose_file)[0]
        audio_file = None
        for ext in ['.wav', '.mp3', '.flac']:
            candidate = os.path.join(audio_dir, base_name + ext)
            if os.path.exists(candidate):
                audio_file = candidate
                break
        
        if audio_file is None:
            print(f"Warning: No audio file found for {pose_file}")
            continue
        
        # 计算分数
        try:
            score, _, _ = evaluate_flame_pose_alignment(
                pose_params, audio_file, fps=fps, audio_method=audio_method
            )
            scores.append(score)
            print(f"{pose_file}: {score:.4f}")
        except Exception as e:
            print(f"Error processing {pose_file}: {e}")
            continue
    
    mean_score = np.mean(scores) if scores else 0.0
    return mean_score, scores


if __name__ == '__main__':
    # ============ 推荐使用：简洁API ============
    
    # 方式1: 从文件路径
    # pose_params = np.load('path/to/pose.npy')  # shape: [frames, 3] 或 [frames, 6]
    # score = compute_beat_alignment_score('path/to/audio.wav', pose_params)
    # print(f"Beat Alignment Score: {score:.4f}")
    
    # 方式2: 从音频数组
    # audio_array, sr = librosa.load('path/to/audio.wav')
    # score = compute_beat_alignment_score((audio_array, sr), pose_params)
    # print(f"Beat Alignment Score: {score:.4f}")
    test_root_path = './testset/'
    pred_rootpath = './inference_results'
    dataset_name = 'V4-HDTF'
    sub_sample_name = 'V4-00092'
    audio_path = os.path.join(test_root_path, dataset_name, sub_sample_name, 'audio.wav')
    flame_parameters_path = os.path.join(pred_rootpath, dataset_name, sub_sample_name, 'flame_params.npy')
    # 方式3: 自定义参数
    score = compute_beat_alignment_score(
        audio=audio_path,
        flame_parameters_path=flame_parameters_path,
        fps=25,                    # 视频帧率
        audio_method='beat',      # 'onset' 或 'beat'
        pose_sigma=2,              # pose平滑参数
        variance=8                 # 对齐容忍度
    )
    print(f"Beat Alignment Score: {score:.4f}")
    # ============ 高级用法：批量评估 ============
    
    # 批量评估多个样本
    # pose_dir = '/path/to/pose/directory'
    # audio_dir = '/path/to/audio/directory'
    # mean_score, all_scores = batch_evaluate(pose_dir, audio_dir, fps=30, audio_method='onset')
    # print(f"\nMean Beat Alignment Score: {mean_score:.4f}")
    # print(f"Std: {np.std(all_scores):.4f}")
    
 
