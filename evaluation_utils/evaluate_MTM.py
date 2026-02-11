#!/usr/bin/env python3
"""Lip-Sync Evaluation

This script evaluates temporal alignment between predicted and ground-truth
3-D lip movements. It measures the inter-lip distance for each frame, smooths
the trajectories, aligns them with derivative Dynamic Time Warping (DDTW),
and computes the frame offset (Δt) for corresponding extrema.  Per-sequence
plots and a CSV summary are produced.


Dependencies
------------
numpy, matplotlib, scipy, fastdtw
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d


def ddtw(x: np.ndarray, y: np.ndarray,
         dist=lambda a, b: abs(a - b)) -> Tuple[float, np.ndarray]:
    """Derivative Dynamic Time Warping (DDTW).

    Parameters
    ----------
    x, y : np.ndarray
        One-dimensional sequences to be aligned.
    dist : callable
        Point-wise distance function.

    Returns
    -------
    distance : float
        The cumulative alignment cost.
    path : np.ndarray of shape (k, 2)
        Optimal warping path (index pairs).
    """
    # import ipdb; ipdb.set_trace()
    # First-order derivatives
    dx = np.diff(x, prepend=x[0])
    dy = np.diff(y, prepend=y[0])

    n, m = len(dx), len(dy)
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # import ipdb; ipdb.set_trace()
            # cost = dist(dx[i - 1], dy[j - 1])
            cost = abs(dx[i - 1]- dy[j - 1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],      # insertion
                dtw_matrix[i, j - 1],      # deletion
                dtw_matrix[i - 1, j - 1],  # match
            )

    # Trace optimal path
    path: List[Tuple[int, int]] = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        steps = [
            dtw_matrix[i - 1, j - 1],
            dtw_matrix[i - 1, j],
            dtw_matrix[i, j - 1],
        ]
        argmin = int(np.argmin(steps))
        if argmin == 0:
            i -= 1
            j -= 1
        elif argmin == 1:
            i -= 1
        else:
            j -= 1

    path.reverse()
    return float(dtw_matrix[n, m]), np.array(path)


def calculate_upper_lower_distance(
        gt_vertice_path: Path,
        pred_vertice_path: Path,
        output_path: Path,
        sigma: float = 2.0
) -> Tuple[Optional[float], List[int]]:
    """Compute Δt for a single sequence and save the plot."""
    
    # --- Load & reshape data --------------------------------------------------
    gt_keypoints = np.load(gt_vertice_path, allow_pickle=True).reshape(-1, 5023, 3)
    # import ipdb; ipdb.set_trace()
    pred_keypoints = np.load(pred_vertice_path, allow_pickle=True).reshape(-1, 5023, 3)

    min_len = min(gt_keypoints.shape[0], pred_keypoints.shape[0])
    gt_keypoints = gt_keypoints[:min_len]
    pred_keypoints = pred_keypoints[:min_len]

    # Optional down-sampling of GT (every second frame)
    # gt_keypoints = gt_keypoints[::2]

    # Central outer-lip vertices (Blanz & Vetter index convention)
    upper_idx, lower_idx = 3531, 3504
    gt_disp = gt_keypoints[:, upper_idx, :] - gt_keypoints[:, lower_idx, :]
    pred_disp = pred_keypoints[:, upper_idx, :] - pred_keypoints[:, lower_idx, :]

    gt_mag = np.linalg.norm(gt_disp, axis=1)
    pred_mag = np.linalg.norm(pred_disp, axis=1)
    
    # --- Smoothing ------------------------------------------------------------
    gt_smooth = gaussian_filter1d(gt_mag, sigma=sigma)
    pred_smooth = gaussian_filter1d(pred_mag, sigma=sigma)

    # --- Alignment ------------------------------------------------------------
    _, path = ddtw(gt_smooth, pred_smooth, dist=euclidean)

    # --- Detect extrema -------------------------------------------------------
    peaks_gt, _ = find_peaks(gt_smooth)
    valleys_gt, _ = find_peaks(-gt_smooth)
    peaks_pred, _ = find_peaks(pred_smooth)
    valleys_pred, _ = find_peaks(-pred_smooth)

    extrema_gt: Dict[int, str] = {idx: 'max' for idx in peaks_gt}
    extrema_gt.update({idx: 'min' for idx in valleys_gt})
    extrema_pred: Dict[int, str] = {idx: 'max' for idx in peaks_pred}
    extrema_pred.update({idx: 'min' for idx in valleys_pred})

    extrema_indices_gt = sorted(extrema_gt)

    # --- Compute Δt under matching conditions ---------------------------------
    delta_t_list: List[int] = []

    # Prepare plot
    plt.figure(figsize=(12, 6))
    plt.plot(gt_smooth, label='Ground Truth', color='blue')
    plt.plot(pred_smooth, label='Predicted', color='red', alpha=0.7)

    for idx in peaks_gt:
        plt.plot(idx, gt_smooth[idx], 'b^', label='GT Maxima' if idx == peaks_gt[0] else '')
    for idx in valleys_gt:
        plt.plot(idx, gt_smooth[idx], 'bo', label='GT Minima' if idx == valleys_gt[0] else '')
    for idx in peaks_pred:
        plt.plot(idx, pred_smooth[idx], 'r^', label='Pred Maxima' if idx == peaks_pred[0] else '')
    for idx in valleys_pred:
        plt.plot(idx, pred_smooth[idx], 'ro', label='Pred Minima' if idx == valleys_pred[0] else '')

    # Evaluate warping path
    for idx_gt, idx_pred in path:
        if idx_gt in extrema_gt and idx_pred in extrema_pred:
            if extrema_gt[idx_gt] == extrema_pred[idx_pred]:
                # Locate neighbouring extrema in GT sequence
                pos = extrema_indices_gt.index(idx_gt)
                left_gt = extrema_indices_gt[pos - 1] if pos > 0 else 0
                right_gt = extrema_indices_gt[pos + 1] if pos < len(extrema_indices_gt) - 1 else len(gt_smooth) - 1
                if left_gt <= idx_pred <= right_gt:
                    delta = abs(idx_gt - idx_pred)
                    delta_t_list.append(delta)
                    plt.plot([idx_gt, idx_pred], [gt_smooth[idx_gt], pred_smooth[idx_pred]], color='green')

    plt.xlabel('Frame')
    plt.ylabel('Lip Distance')
    plt.title('Upper and Lower Lip Distance with DDTW Alignment')
    plt.legend()
    plt.grid(True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()

    mean_delta_t: Optional[float] = float(np.mean(delta_t_list)) if delta_t_list else 0
    return mean_delta_t, delta_t_list


def process_sequences(
        gt_dir: Path,
        pred_dir: Path,
        output_dir: Path,
        csv_path: Path
) -> None:
    """Evaluate all predicted sequences in *pred_dir* against their GT counterparts."""
    if isinstance(gt_dir, str):
        gt_dir = Path(gt_dir)
    if isinstance(pred_dir, str):
        pred_dir = Path(pred_dir)
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    if isinstance(csv_path, str):
        csv_path = Path(csv_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_files =  [pred_dir / 'verts_seq.npy']

    mean_delta_t_values: List[float] = []
    # import ipdb; ipdb.set_trace()
    with csv_path.open('w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Sequence', 'Mean Δt (frames)', '#Matching Points', 'Δt Values'])

        for pred_file in sorted(pred_files):
            # if '_condition_' not in pred_file.name:
            #     print(f'[WARN] Skipping invalid name: {pred_file.name}')
            #     writer.writerow([pred_file.name, 'Invalid filename', 0, []])
            #     continue
            
            gt_name = 'vertices.npy'
            gt_file = gt_dir / gt_name

            if not gt_file.exists():
                print(f'[WARN] Ground-truth file missing: {gt_file.name}')
                writer.writerow([pred_file.name, 'GT not found', 0, []])
                continue

            # print(f'[INFO] Processing {pred_file}')
            plot_path = output_dir / f'dtw_{pred_file.stem}.png'
            mean_delta_t, delta_ts = calculate_upper_lower_distance(
                gt_file, pred_file, plot_path
            )

            if mean_delta_t is not None:
                mean_delta_t_values.append(mean_delta_t)
                writer.writerow([pred_file.name, f'{mean_delta_t:.2f}', len(delta_ts), ';'.join(map(str, delta_ts))])
            else:
                writer.writerow([pred_file.name, 'No matching points', 0, []])

        writer.writerow([])
        # if mean_delta_t_values is None:
        #     print("None")
        #     raise ValueError(f"Mean Δt values not found for the sample in {pred_dir} & {gt_dir}")
        # else:
        #     print(f"Mean Δt values: {mean_delta_t_values}")
        overall = float(np.mean(mean_delta_t_values)) if mean_delta_t_values else 0
        writer.writerow(['Overall Mean Δt', f'{overall:.2f}' if overall is not None else 'N/A'])
    return overall


if __name__ == '__main__':
    # GT_DIR = Path('data_MTM/gt_example')
    # PRED_DIR = Path('data_MTM/pred_example')
    # OUT_DIR = Path('data_MTM/output_example')
    # CSV_PATH = OUT_DIR / 'delta_t_results.csv'
    # import os
    # gt_lips_path = "/scratch/jc02/zw6060/workSpace/DEEPTalk/gt_lips_verts"
    # pred_lips_path = "/scratch/jc02/zw6060/workSpace/DEEPTalk/pred_lips_verts"
    # for gt_files, pred_files in zip(sorted(os.listdir(gt_lips_path)), sorted(os.listdir(pred_lips_path))): # sort to ensure they load the same file 
    #     # correct the current file path 
    #     gt_file_path = os.path.join(gt_lips_path, gt_files)
    #     pred_file_path = os.path.join(pred_lips_path, pred_files)
    GT_DIR = Path("/scratch/jc02/zw6060/workSpace/DEEPTalk/pseudo_gt")
    PRED_DIR = Path("/scratch/jc02/zw6060/workSpace/DEEPTalk/pred_full_npy")
    OUT_DIR = Path("/scratch/jc02/zw6060/workSpace/DEEPTalk/MTM_output")
    CSV_PATH = Path("/scratch/jc02/zw6060/workSpace/DEEPTalk/MTM_output/mtm_results.csv")
    process_sequences(GT_DIR, PRED_DIR, OUT_DIR, CSV_PATH)
    print('[DONE] Evaluation finished. Results saved to', CSV_PATH)
