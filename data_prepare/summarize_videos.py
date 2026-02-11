import os
from tqdm import tqdm

# 定义要扫描的两个目录
cur_dir = os.path.dirname(os.path.abspath(__file__))
scaled_dirs = [
    os.path.join(cur_dir, "Scaled_videos"),
]

# 输出文件
out_txt = os.path.join(cur_dir, "all_videos_path.txt")

all_video_paths = []

 
for scaled_dir in scaled_dirs:
    if not os.path.exists(scaled_dir):
        print(f"[SKIP] {scaled_dir} not found.")
        continue

    print(f"[Scanning] {scaled_dir}")
    for root, dirs, files in os.walk(scaled_dir):
        dirs.sort()
        files.sort()
        for file in files:
            if file.endswith(".mp4"):
                abs_path = os.path.abspath(os.path.join(root, file))
                all_video_paths.append(abs_path)

# 写入txt文件
with open(out_txt, "w") as f:
    for path in all_video_paths:
        f.write(path + "\n")

print(f"[DONE] Found {len(all_video_paths)} videos in total.")
print(f"[Saved to] {out_txt}")