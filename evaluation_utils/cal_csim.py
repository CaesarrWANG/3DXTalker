from CopulaSimilarity.CSM import CopulaBasedSimilarity as CSIMSimilarity
import matplotlib.pyplot as plt
import numpy as np
import os
import subprocess
import cv2
from tqdm import tqdm


def CSIM(image_1_path, image_2_path):
    # Default patch_size set to 8 but can be changed depending on the aimed balance between accuracy and realtime
    copula_similarity = CSIMSimilarity(patch_size=8) 

    #load your images
    image1 = cv2.imread(image_1_path)
    image2 = cv2.imread(image_2_path)

    #calculate the similarity map
    csim_map = copula_similarity.compute_local_similarity(image1, image2)
    csim = np.mean(csim_map)

    return csim


def main(args):
    result_path = args.model_testset_pred_dir_path
    CSIM_list = []
    for dataset in sorted(os.listdir(result_path)):
        print(f"Processing dataset: {dataset}")
        dataset_path = os.path.join(result_path, dataset) 
        for data in tqdm(sorted(os.listdir(dataset_path))):
            frames_folder = os.path.join(dataset_path, data, "frames")
            if not os.path.exists(frames_folder):
                video_path = os.path.join(dataset_path, data, "MeshRenderedVideo.mp4")
                os.makedirs(frames_folder, exist_ok=True)
                cmd = ["ffmpeg", "-i", video_path, "-r", "25", os.path.join(frames_folder, "%04d.png")]
                subprocess.run(cmd, check=True)
            images = sorted(os.listdir(frames_folder))  # sorted list of filenames
            images = [os.path.join(frames_folder, f) for f in images]  # full paths
            for img1, img2 in zip(images, images[1:]):
                csim_score = CSIM(img1, img2)
                CSIM_list.append(csim_score)
    print(f"CSIM: {np.mean(CSIM_list)}")


if __name__ == "__main__":
    
    # model_testset_pred_dir_path = "/scratch3/wan451/3DTalk/The-Sound-of-Motion/3DXTalker_preds"
    # model_testset_pred_dir_path = "/scratch3/wan451/3DTalk/inferno/EMOTE_preds"
    # model_testset_pred_dir_path = "/scratch3/wan451/3DTalk/DEEPTalk/DEEPTalk_preds"
    # gt_testset_dir_path = '/scratch3/wan451/3DTalk/emoca/video_output/EMOCA_v2_lr_mse_20/testset'

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_testset_pred_dir_path", type=str, default= "./inference_results",
                        help="Path to the model's predicted testset directory.")
    args = parser.parse_args()
    main(args)