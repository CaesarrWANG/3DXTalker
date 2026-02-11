import os
import numpy as np
import subprocess
from tqdm import auto
from tqdm import tqdm
import librosa
import shutil
import whisper


cur_dir = os.path.dirname(os.path.abspath(__file__))

def get_videos(raw_video_dir):
    all_videos = sorted(os.listdir(raw_video_dir))
    print(f"Total videos: {len(all_videos)}")
    return all_videos

def filter_by_duration(all_videos, input_video_dir, min_duration=10.0, re_run=False):
    if re_run or not os.path.exists(os.path.join(cur_dir, 'duration_filter_results.npy')):
        duration_filter_results = []
        for video in auto.tqdm(all_videos):
            video_path = os.path.join(input_video_dir, video)
            video_parse = subprocess.run(
                ['ffprobe', '-v', 'error', '-show_entries',
                'format=duration', '-of',
                'csv=p=0', video_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            duration = float(video_parse.stdout.strip())
            if min_duration <= duration:
                duration_filter_results.append(video)
        np.save(os.path.join(cur_dir, 'duration_filter_results.npy'), np.array(duration_filter_results))
    else:
        duration_filter_results = np.load(os.path.join(cur_dir, 'duration_filter_results.npy'), allow_pickle=True).tolist()
    return duration_filter_results



def resave_filtered_videos(filtered_videos, input_video_dir, output_video_dir):
    os.makedirs(output_video_dir, exist_ok=True)
    for idx, video in tqdm(enumerate(filtered_videos)):
        input_path = os.path.join(input_video_dir, video)
        output_path = os.path.join(output_video_dir, f"{idx:05d}.mp4")
        os.system(f"cp {input_path} {output_path}")



def estimate_snr(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=None)
        
        frame_length = int(0.025 * sr)  # 25ms
        hop_length = int(0.010 * sr)    # 10ms
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        noise_level = np.percentile(rms, 20)
        speech_level = np.percentile(rms, 80)
        snr = 10 * np.log10(speech_level / (noise_level + 1e-6))
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        snr = 0
    return snr

def filter_noisy_videos(video_list, input_video_dir, tmp_dir="./tmp_audio", snr_threshold=8):
    os.makedirs(tmp_dir, exist_ok=True)
    clean_videos = []

    for video in tqdm(video_list):
        video_path = os.path.join(input_video_dir, video)
        audio_path = os.path.join(tmp_dir, os.path.splitext(video)[0] + ".wav")
        # 提取音频
        subprocess.run([
            "ffmpeg", "-y", "-i", video_path, "-q:a", "0", "-map", "a", audio_path
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if os.path.exists(audio_path):
            # 计算SNR
            snr = estimate_snr(audio_path)
            if snr >= snr_threshold:
                clean_videos.append(video)
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return clean_videos


def filter_by_language(video_list, input_video_dir, tmp_dir="./tmp_audio", 
                       target_language="en", confidence_threshold=0.8):
    """
    Step 3: Language Filtering using Whisper
    Filters videos to keep only English (or target language) samples with high confidence.
    """
    os.makedirs(tmp_dir, exist_ok=True)
    print("Loading Whisper model...")
    model = whisper.load_model("base")  # Can use "small", "medium", "large" for better accuracy
    
    english_videos = []
    
    for video in tqdm(video_list, desc="Language filtering"):
        video_path = os.path.join(input_video_dir, video)
        audio_path = os.path.join(tmp_dir, os.path.splitext(video)[0] + ".wav")
        
        # Extract audio
        subprocess.run([
            "ffmpeg", "-y", "-i", video_path, "-q:a", "0", "-map", "a", 
            "-ar", "16000", audio_path  # Whisper expects 16kHz
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        if os.path.exists(audio_path):
            try:
                # Detect language
                audio = whisper.load_audio(audio_path)
                audio = whisper.pad_or_trim(audio)
                mel = whisper.log_mel_spectrogram(audio).to(model.device)
                _, probs = model.detect_language(mel)
                detected_lang = max(probs, key=probs.get)
                confidence = probs[detected_lang]
                
                # Keep if target language with sufficient confidence
                if detected_lang == target_language and confidence >= confidence_threshold:
                    english_videos.append(video)
            except Exception as e:
                print(f"Error processing {video}: {e}")
                continue
    
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return english_videos


def filter_by_syncnet(video_list, input_video_dir, syncnet_threshold=5.0):
    """
    Step 4: Audio-Visual Sync Filtering using SyncNet
    Filters out videos with poor lip synchronization, scene cuts, or off-screen speakers.
    
    Requires SyncNet to be installed in ./syncnet_python/
    Download from: https://github.com/joonson/syncnet_python
    """
    syncnet_dir = os.path.join(cur_dir, "syncnet_python")
    
    if not os.path.exists(syncnet_dir):
        print(f"Warning: SyncNet not found at {syncnet_dir}")
        print("Skipping SyncNet filtering. To enable:")
        print("1. git clone https://github.com/joonson/syncnet_python")
        print("2. Download pretrained models as per their instructions")
        return video_list
    
    synced_videos = []
    
    for video in tqdm(video_list, desc="SyncNet filtering"):
        video_path = os.path.join(input_video_dir, video)
        
        try:
            # Run SyncNet evaluation
            result = subprocess.run([
                "python", os.path.join(syncnet_dir, "run_syncnet.py"),
                "--videofile", video_path,
                "--tmp_dir", "./tmp_syncnet"
            ], capture_output=True, text=True, timeout=60)
            
            # Parse SyncNet confidence score from output
            # Format: "AV offset: X (conf: Y)"
            output = result.stdout
            if "conf:" in output:
                conf_str = output.split("conf:")[1].split(")")[0].strip()
                confidence = float(conf_str)
                
                # Keep videos with good sync (higher confidence = better sync)
                if confidence >= syncnet_threshold:
                    synced_videos.append(video)
            
        except subprocess.TimeoutExpired:
            print(f"Timeout processing {video}")
            continue
        except Exception as e:
            print(f"Error processing {video}: {e}")
            continue
    
    # Cleanup
    shutil.rmtree("./tmp_syncnet", ignore_errors=True)
    return synced_videos


def normalize_resolution(video_list, input_video_dir, output_dir, 
                         size=512, fps=25, re_encode=True):
    """
    Step 5: Resolution Normalization
    Resize and center-crop all videos to 512×512 at 25 FPS with standardized RGB encoding.
    """
    os.makedirs(output_dir, exist_ok=True)
    normalized_videos = []
    
    for idx, video in enumerate(tqdm(video_list, desc="Normalizing resolution")):
        input_path = os.path.join(input_video_dir, video)
        output_filename = f"{idx:05d}.mp4"
        output_path = os.path.join(output_dir, output_filename)
        
        if os.path.exists(output_path):
            normalized_videos.append(output_filename)
            continue
        
        # ffmpeg filter: scale short side to 512, then center crop to 512x512
        filter_str = (
            f"scale='if(gt(iw,ih),-1,{size})':'if(gt(iw,ih),{size},-1)',"
            f"crop={size}:{size}"
        )
        
        cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-vf", filter_str,
            "-r", str(fps),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",  # Standardized RGB encoding
            "-preset", "medium",
            "-crf", "23",
            "-c:a", "aac", 
            "-b:a", "128k",
            "-ar", "16000",  # Audio sample rate
            output_path
        ]
        
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, 
                          timeout=120, check=True)
            normalized_videos.append(output_filename)
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
            print(f"Error processing {video}: {e}")
            continue
    
    return normalized_videos

if __name__ == "__main__":
    raw_video_dir = "./path/to/raw_videos" 
    output_dir = os.path.join(cur_dir, 'filtered_videos')
    final_output_dir = os.path.join(cur_dir, 'final_curated_videos')
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(final_output_dir, exist_ok=True)
    
    all_videos = get_videos(raw_video_dir)
    
    # Step 1: Duration filter
    duration_filter = filter_by_duration(all_videos, raw_video_dir, min_duration=10.0)
    print(f"Videos after duration filter: {len(duration_filter)}")
    
    # Step 2: Noise filter
    clean_videos = filter_noisy_videos(duration_filter, raw_video_dir)
    print(f"Videos after noise filter: {len(clean_videos)}")
    
    # Step 3: Language filter (English only with high confidence)
    english_videos = filter_by_language(clean_videos, raw_video_dir, 
                                        target_language="en", 
                                        confidence_threshold=0.8)
    print(f"Videos after language filter: {len(english_videos)}")
    
    # Step 4: SyncNet filter (audio-visual synchronization)
    synced_videos = filter_by_syncnet(english_videos, raw_video_dir, 
                                      syncnet_threshold=5.0)
    print(f"Videos after SyncNet filter: {len(synced_videos)}")
    
    # Step 5: Resolution normalization (512×512 @ 25 FPS)
    print("Normalizing resolution to 512×512 @ 25 FPS...")
    normalized_videos = normalize_resolution(synced_videos, raw_video_dir, 
                                            final_output_dir, size=512, fps=25)
    print(f"Final curated videos: {len(normalized_videos)}")
    
    print(f"\n✅ Data curation pipeline complete!")
    print(f"   Total input videos: {len(all_videos)}")
    print(f"   After duration filter: {len(duration_filter)}")
    print(f"   After noise filter: {len(clean_videos)}")
    print(f"   After language filter: {len(english_videos)}")
    print(f"   After SyncNet filter: {len(synced_videos)}")
    print(f"   Final normalized videos: {len(normalized_videos)}")
    print(f"   Output directory: {final_output_dir}")


