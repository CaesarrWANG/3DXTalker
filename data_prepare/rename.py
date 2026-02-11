import os
from tqdm import tqdm

# =====================
# 配置
# =====================

# 数据集编号映射
dict_dataset_id_map = {
    "GRID": "V0",
    "RAVDESS": "V1",
    "MEAD": "V2",
    "VoxCeleb2": "V3",
    "HDTF": "V4",
    "CelebV-HQ": "V5"
}

cur_dir = os.path.dirname(os.path.abspath(__file__))
dataset_name = "VoxCeleb2"
dataset_id = dict_dataset_id_map[dataset_name]

input_dir = os.path.join(cur_dir, "Scaled_videos")
output_dir = os.path.join(cur_dir, "Renamed_videos")
os.makedirs(output_dir, exist_ok=True)

all_videos = sorted([v for v in os.listdir(input_dir) if v.endswith(".mp4")])

for idx, video in enumerate(tqdm(all_videos, desc=f"Renaming {dataset_name}")):
    src_path = os.path.join(input_dir, video)
    new_name = f"{dataset_id}-{idx:05d}.mp4"
    dst_path = os.path.join(output_dir, new_name)
    os.rename(src_path, dst_path)

print(f"\n✅ Done! Renamed {len(all_videos)} videos.")
print(f"📁 Output directory: {output_dir}")