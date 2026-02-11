# Data Curation Pipeline

Complete 5-step pipeline for curating high-quality video datasets for 3D talking head generation.

## Pipeline Overview

```
Raw Videos → Duration Filter → Noise Filter → Language Filter → SyncNet Filter → Resolution Normalization → Final Dataset
```

## Steps

### Step 1: Duration Filtering
- **Purpose**: Remove videos shorter than minimum duration
- **Default threshold**: 10 seconds
- **Implementation**: Uses ffprobe to check video duration

### Step 2: Noise Filtering  
- **Purpose**: Remove videos with low audio quality
- **Method**: Estimates SNR (Signal-to-Noise Ratio) from audio
- **Default threshold**: SNR ≥ 8 dB

### Step 3: Language Filtering (NEW)
- **Purpose**: Keep only English samples with high detection confidence
- **Method**: Uses OpenAI Whisper for language detection
- **Default threshold**: 80% confidence for English
- **Whisper model**: `base` (can upgrade to `small`, `medium`, `large` for better accuracy)

### Step 4: Audio-Visual Sync Filtering (NEW)
- **Purpose**: Eliminate videos with poor lip synchronization, scene cuts, or off-screen speakers
- **Method**: Uses SyncNet to compute audio-visual synchronization confidence
- **Default threshold**: SyncNet confidence ≥ 5.0
- **Requirements**: 
  - Clone SyncNet: `git clone https://github.com/joonson/syncnet_python`
  - Download pretrained models as per SyncNet instructions
  - Place in `./data_prepare/syncnet_python/`

### Step 5: Resolution Normalization (NEW)
- **Purpose**: Standardize all videos to consistent format
- **Output specs**:
  - Resolution: 512×512 (center-cropped)
  - Frame rate: 25 FPS
  - Video codec: H.264 (libx264)
  - Pixel format: yuv420p (standardized RGB)
  - Audio: AAC 128kbps @ 16kHz
- **Method**: Uses ffmpeg with smart scaling (short side → 512) + center crop

## Usage

### Prerequisites
```bash
# Install dependencies
pip install openai-whisper torch librosa tqdm numpy

# For SyncNet (optional but recommended)
cd data_prepare
git clone https://github.com/joonson/syncnet_python
# Follow SyncNet setup instructions to download pretrained models
```

### Run Pipeline
```bash
cd data_prepare

# Edit the script to set your input directory
# Change line: raw_video_dir = "./path/to/raw_videos"

python data_curation_pipeline.py
```

### Output Structure
```
data_prepare/
├── duration_filter_results.npy    # Cached results from step 1
├── final_curated_videos/          # Final output (step 5)
│   ├── 00000.mp4
│   ├── 00001.mp4
│   └── ...
└── tmp_audio/                     # Temporary (auto-cleaned)
```

## Configuration Parameters

Edit these in the `__main__` section:

```python
# Step 1: Duration threshold
min_duration = 10.0  # seconds

# Step 2: Audio quality threshold  
snr_threshold = 8  # dB

# Step 3: Language detection
target_language = "en"  # Language code
confidence_threshold = 0.8  # 0.0 to 1.0

# Step 4: SyncNet threshold
syncnet_threshold = 5.0  # Higher = stricter sync requirements

# Step 5: Resolution settings
size = 512  # Output resolution (512x512)
fps = 25    # Output frame rate
```

## Performance Notes

- **Step 3 (Whisper)**: ~1-2 seconds per video (GPU recommended)
- **Step 4 (SyncNet)**: ~5-10 seconds per video (requires GPU)
- **Step 5 (ffmpeg)**: ~5-15 seconds per video (CPU)

### Speed Optimization Tips
- Use GPU for Whisper: Model will auto-detect CUDA
- For SyncNet: Ensure it's using GPU (check syncnet_python config)
- Parallel processing: Modify code to use multiprocessing.Pool for steps 3-4

## Troubleshooting

### SyncNet not installed
If SyncNet directory is missing, the pipeline will **skip step 4** and print a warning. This allows the pipeline to continue without SyncNet if unavailable.

### Whisper out of memory
Use smaller model: Change `whisper.load_model("base")` to `whisper.load_model("tiny")`

### FFmpeg errors in step 5
- Ensure ffmpeg is installed: `ffmpeg -version`
- Check input video codecs: Some formats may need pre-conversion

## Expected Results

For a typical dataset:
- **Input**: 1000 videos
- **After duration filter**: ~900 videos (10% too short)
- **After noise filter**: ~800 videos (11% noisy audio)
- **After language filter**: ~650 videos (19% non-English or low confidence)
- **After SyncNet filter**: ~550 videos (15% poor sync)
- **Final output**: ~550 videos (55% retention rate)

Actual retention depends on source data quality.
