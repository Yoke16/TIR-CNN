import glob
import os
import re

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import HimawariCloudDataset
from model import TIR_CNN


def extract_year_month(path: str):
    filename = os.path.basename(path)
    m = re.search(r"(\d{8})_(\d{4})Z", filename)
    if m:
        datestr = m.group(1)
        return int(datestr[:4]), int(datestr[4:6])
    return None, None


def filter_complete_files(file_list, expected_channels: int):
    kept = []
    for path in tqdm(file_list, desc="检查 2024 文件通道数", ncols=100):
        try:
            data = np.load(path)
            if "data" in data:
                cube = data["data"]
            else:
                keys = list(data.keys())
                cube = data[keys[0]]
        except Exception:
            continue

        if cube.ndim == 3 and cube.shape[0] >= expected_channels:
            kept.append(path)
    print(f"  2024 年完整(>=15通道)文件数: {len(kept)}")
    return kept


def collect_2024_files(root_dirs, expected_channels: int):
    all_2024 = []
    for root in root_dirs:
        pattern = os.path.join(root, "**", "*.npz")
        files = glob.glob(pattern, recursive=True)
        for f in files:
            year, _ = extract_year_month(f)
            if year == 2024:
                all_2024.append(f)
    if not all_2024:
        raise ValueError("未找到 2024 年的数据文件，请检查路径/文件名格式。")
    print(f"找到 2024 年文件总数: {len(all_2024)}")
    return filter_complete_files(all_2024, expected_channels)


def evaluate_2024():
    # 1. 配置 & 设备
    with open("configs/base/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device(config["training"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    input_channels = config["model"]["input_channels"]
    num_total_labels = 10  # 10个标签
    expected_C = 8 + 1 + num_total_labels  # 8 TIR + SZA + 10 labels = 19
    reg_label_indices = [3, 4, 5, 2, 0, 6, 7]  # 7个回归标签索引

    # 2. 收集 2024 年的 19 通道文件
    test_files = collect_2024_files(config["data"]["root_dirs"], expected_C)

    # 3. 加载归一化统计量
    stats_dir = config["data"]["stats_dir"]
    input_mean = np.load(os.path.join(stats_dir, "input_mean.npy")).astype(np.float32)
    input_std = np.load(os.path.join(stats_dir, "input_std.npy")).astype(np.float32)
    target_mean = np.load(os.path.join(stats_dir, "target_mean.npy")).astype(np.float32)
    target_std = np.load(os.path.join(stats_dir, "target_std.npy")).astype(np.float32)

    # 4. Dataset / DataLoader（注意：这里不区分训练/验证，只是测试）
    test_dataset = HimawariCloudDataset(
        file_list=test_files,
        input_mean=input_mean,
        input_std=input_std,
        target_mean=target_mean,
        target_std=target_std,
        input_channels=input_channels,
        num_total_labels=num_total_labels,
        sza_index=8,
        cldtype_label_index=8,
        reg_label_indices=reg_label_indices,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    # 5. 加载最佳模型
    ckpt_path = os.path.join(config["training"]["output_dir"], "checkpoints", "best_model.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"找不到模型权重: {ckpt_path}，请先运行 train.py。")

    model = TIR_CNN(
        input_channels=input_channels,
        output_channels_cls=10,
        output_channels_reg=7,
        base_channels=config["model"]["base_channels"],
    ).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"已加载模型权重: {ckpt_path}")

    # 6. 为统计物理量上的 RMSE/MBE/R 做准备（回归和分类）
    C = len(reg_label_indices)  # 7个回归输出
    n = np.zeros(C, dtype=np.int64)
    sum_t = np.zeros(C, dtype=np.float64)
    sum_p = np.zeros(C, dtype=np.float64)
    sum_t2 = np.zeros(C, dtype=np.float64)
    sum_p2 = np.zeros(C, dtype=np.float64)
    sum_tp = np.zeros(C, dtype=np.float64)

    # 分类指标（CldType）
    cldtype_correct = 0
    cldtype_total = 0
    cldtype_confusion = np.zeros((10, 10), dtype=np.int64)  # 10类混淆矩阵

    mean_t = torch.from_numpy(target_mean).to(device).view(1, C, 1, 1)
    std_t = torch.from_numpy(target_std).to(device).view(1, C, 1, 1)

    with torch.no_grad():
        for inputs, cldtype_targets, targets_norm in tqdm(test_loader, desc="Evaluating 2024", ncols=120):
            inputs = inputs.to(device, non_blocking=True)
            cldtype_targets = cldtype_targets.to(device, non_blocking=True)
            targets_norm = targets_norm.to(device, non_blocking=True)

            cldtype_logits, preds_norm = model(inputs)

            # 1. 分类评估：CldType
            cldtype_pred = torch.argmax(cldtype_logits, dim=1)  # [B, H, W]
            
            # 只统计有效像元（target不为0）
            valid_mask = cldtype_targets > 0
            if torch.any(valid_mask):
                cldtype_correct += (cldtype_pred[valid_mask] == cldtype_targets[valid_mask]).sum().item()
                cldtype_total += valid_mask.sum().item()
                
                # 更新混淆矩阵
                for i in range(10):
                    for j in range(10):
                        mask_ij = (cldtype_targets == i) & (cldtype_pred == j) & valid_mask
                        cldtype_confusion[i, j] += mask_ij.sum().item()

            # 2. 回归评估：7个云参数（反归一化到物理量空间）
            preds = preds_norm * std_t + mean_t
            targets = targets_norm * std_t + mean_t

            for c in range(C):
                t = targets[:, c, :, :]
                p = preds[:, c, :, :]

                mask = torch.isfinite(t)
                if not torch.any(mask):
                    continue

                t_valid = t[mask].view(-1).double().cpu().numpy()
                p_valid = p[mask].view(-1).double().cpu().numpy()

                n_c = t_valid.size
                if n_c == 0:
                    continue

                n[c] += n_c
                sum_t[c] += t_valid.sum()
                sum_p[c] += p_valid.sum()
                sum_t2[c] += (t_valid ** 2).sum()
                sum_p2[c] += (p_valid ** 2).sum()
                sum_tp[c] += (t_valid * p_valid).sum()

    # 7. 计算指标
    # 7.1 分类指标（CldType）
    print("\n=== 2024 年 CldType 分类评估结果 ===")
    if cldtype_total > 0:
        accuracy = cldtype_correct / cldtype_total
        print(f"整体准确率: {accuracy:.4f} ({cldtype_correct}/{cldtype_total})")
        
        # 每类的精确率、召回率、F1
        print("\n每类别性能:")
        print(f"{'类别':<8} {'精确率':>8} {'召回率':>8} {'F1':>8} {'样本数':>10}")
        print("-" * 50)
        for i in range(2, 10):  # 只统计云类型(2-9)
            tp = cldtype_confusion[i, i]
            fp = cldtype_confusion[:, i].sum() - tp
            fn = cldtype_confusion[i, :].sum() - tp
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            print(f"Type {i:<3} {precision:>8.4f} {recall:>8.4f} {f1:>8.4f} {cldtype_confusion[i, :].sum():>10d}")
    else:
        print("没有有效的CldType样本")

    # 7.2 回归指标（7个云参数）
    reg_label_names = [
        "DCOMP35_CPS",
        "DCOMP36_CPS",
        "DCOMP37_CPS",
        "DCOMP35_COD",
        "CldPressure",
        "CldHeight",
        "CldTemperature",
    ]

    print("\n=== 2024 年云参数回归评估结果（物理量空间）===")
    for i, name in enumerate(reg_label_names):
        if n[i] == 0:
            print(f"{name}: 没有有效样本。")
            continue

        N = float(n[i])
        mean_t_i = sum_t[i] / N
        mean_p_i = sum_p[i] / N

        mse = (sum_p2[i] + sum_t2[i] - 2 * sum_tp[i]) / N
        rmse = np.sqrt(max(mse, 0.0))
        mbe = (sum_p[i] - sum_t[i]) / N

        var_t = sum_t2[i] / N - mean_t_i ** 2
        var_p = sum_p2[i] / N - mean_p_i ** 2
        cov_tp = sum_tp[i] / N - mean_t_i * mean_p_i
        if var_t > 0 and var_p > 0:
            r = cov_tp / np.sqrt(var_t * var_p)
        else:
            r = np.nan

        print(
            f"{name:12s} | N={n[i]:10d} | RMSE={rmse:8.3f} | MBE={mbe:8.3f} | R={r:6.3f}"
        )


if __name__ == "__main__":
    evaluate_2024()
