# Dataset Structure

This dataset contains processed audio-visual sequences with EMOCA-generated facial parameters from multiple source datasets. The data is organized for 3D talking face generation and audio-driven facial animation tasks.

## Overview

- **Total Sequences**: 11,506 sequences across 6 datasets
- **Format**: Each sequence contains synchronized audio and facial parameter codes
- **Use Case**: Audio-driven 3D facial animation, speech-driven talking head generation

## Directory Structure

```
trainset/
├── V0-GRID/                      # 6,570 sequences from GRID corpus
│   ├── V0-s1-00001/
│   │   ├── audio.wav             # (N,) audio data
│   │   ├── cam.npy               # (T, 3) camera parameters
│   │   ├── detailcode.npy        # (T, 128) facial details
│   │   ├── envelope.npy          # (N,) audio envelope
│   │   ├── expcode.npy           # (T, 50) expression codes
│   │   ├── lightcode.npy         # (T, 9, 3) lighting
│   │   ├── metadata.pkl          # Sequence metadata 
│   │   ├── posecode.npy          # (376, 6) head pose 
│   │   ├── refimg.npy            # (3, 224, 224) reference image 
│   │   ├── shapecode.npy         # (376, 100) shape codes
│   │   └── texcode.npy           # (376, 50) texture codes
│   ├── V0-s1-00002/
│   │   └── ... (same 11 files)
│   ├── V0-s1-00003/
│   └── ... (6,570 total sequences)
│
├── V1-RAVDESS/                   # 583 sequences from RAVDESS dataset
│   ├── V1-Song-Actor_01-00001/
│   │   └── ... (same 11 files)
│   ├── V1-Song-Actor_01-00002/
│   ├── V1-Speech-Actor_01-00001/
│   ├── V1-Speech-Actor_02-00001/
│   └── ... (583 total sequences)
│
├── V2-MEAD/                      # 1,939 sequences from MEAD dataset
│   ├── V2-M003-angry-00001/
│   │   └── ... (same 11 files)
│   ├── V2-M003-angry-00002/
│   ├── V2-M003-happy-00001/
│   ├── V2-W009-sad-00001/
│   └── ... (1,939 total sequences)
│
├── V3-VoxCeleb2/                 # 1,296 sequences from VoxCeleb2
│   ├── {sequence_id}/
│   │   └── ... (same 11 files)
│   └── ... (1,296 total sequences)
│
├── V4-HDTF/                      # 350 sequences from HDTF dataset
│   ├── {sequence_id}/
│   │   └── ... (same 11 files)
│   └── ... (350 total sequences)
│
└── V5-CelebV-HQ/                 # 768 sequences from CelebV-HQ dataset
    ├── {sequence_id}/
    │   └── ... (same 11 files)
    └── ... (768 total sequences)
```

### File Overview

| File | Type | Shape | Description |
|------|------|-------|-------------|
| `audio.wav` | Audio | (N_samples,) | Original audio waveform|
| `cam.npy` | Parameters | (N_frames, 3) | Camera parameters (position/scale) |
| `detailcode.npy` | Parameters | (N_frames, 128) | Facial detail codes (wrinkles, fine features) |
| `envelope.npy` | Parameters | (N_audio_samples,) | Audio envelope/amplitude over time |
| `expcode.npy` | Parameters | (N_frames, 50) | FLAME expression parameters (50-dim) |
| `lightcode.npy` | Parameters | (N_frames, 9, 3) | Spherical harmonics lighting (9 bands × RGB) |
| `metadata.pkl` | Metadata | N/A | Sequence metadata (integer or dict) |
| `posecode.npy` | Parameters | (N_frames, 6) | 3 head pose + 3 jaw pose |
| `refimg.npy` | Image | (3, 224, 224) | Reference image (RGB, 224×224 pixels) |
| `shapecode.npy` | Parameters | (N_frames, 100) | FLAME shape parameters (100-dim) |
| `texcode.npy` | Parameters | (N_frames, 50) | Texture codes (50-dim) |

## Data Format Details

### Coordinate Systems and Conventions
- **FLAME model**: 3D Morphable Face Model with 5023 vertices
- **Expression space**: 50-dimensional linear basis
- **Shape space**: 100-dimensional PCA space
- **Pose representation**: 3 head pose + 3 jaw pose
- **Lighting**: 2nd-order spherical harmonics (9 bands)

### Temporal Synchronization
- **Video frames**: 25 FPS (frames per second)
- **Audio samples**: 16,000 samples per second
- All video parameters (`expcode`, `shapecode`, `detailcode`, `posecode`, `cam`, `lightcode`, `texcode`) share the same `N_frames` dimension
- Audio and video are temporally aligned (frame 0 corresponds to start of audio)


## Citation

If you use this dataset, please cite the original source datasets:

- **GRID**: Cooke, M., et al. (2006). An audio-visual corpus for speech perception and automatic speech recognition.
- **RAVDESS**: Livingstone, S. R., & Russo, F. A. (2018). The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS).
- **MEAD**: Wang, K., et al. (2020). MEAD: A Large-scale Audio-visual Dataset for Emotional Talking-face Generation.
- **VoxCeleb2**: Chung, J. S., et al. (2018). VoxCeleb2: Deep Speaker Recognition.
- **HDTF**: Zhang, Z., et al. (2021). Flow-guided One-shot Talking Face Generation with a High-resolution Audio-visual Dataset.
- **CelebV-HQ**: Zhu, H., et al. (2022). CelebV-HQ: A Large-Scale Video Facial Attributes Dataset.

And the EMOCA model used for parameter extraction:
- **EMOCA**: Danecek, R., et al. (2022). EMOCA: Emotion Driven Monocular Face Capture and Animation.

## License

Please refer to the original dataset licenses:
- GRID: Research use only
- RAVDESS: CC BY-NA-SC 4.0
- MEAD, VoxCeleb2, HDTF, CelebV-HQ: Check respective dataset licenses

## Notes

- Not all sequence numbers are contiguous (some sequences may be missing due to quality filtering or processing failures)
- File counts per sequence are consistent (11 files per sequence)
- This is a processed/derived dataset - original videos are not included, only extracted parameters
