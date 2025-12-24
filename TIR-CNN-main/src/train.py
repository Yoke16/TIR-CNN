import glob
import gc
import os
import re
from datetime import datetime

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import HimawariCloudDataset
from loss import TwoStageLoss
from model import TIR_CNN


# ---------------- 工具函数 ---------------- #
def extract_year_month(path: str):
    """从文件名中解析 YYYYMMDD_HHMMZ，返回 (year, month)；解析失败返回 (None, None)"""
    filename = os.path.basename(path)
    m = re.search(r"(\d{8})_(\d{4})Z", filename)
    if m:
        datestr = m.group(1)
        return int(datestr[:4]), int(datestr[4:6])
    return None, None


def filter_complete_files(file_list, expected_channels: int):
    """只保留通道数 >= expected_channels 的文件（旧格式15通道文件会被剔除，新格式需要19通道）"""
    kept = []
    dropped = 0
    for path in tqdm(file_list, desc=f"检查通道数 (只保留 >= {expected_channels} 通道)", ncols=100):
        try:
            data = np.load(path)
            if "data" in data:
                cube = data["data"]
            else:
                keys = list(data.keys())
                cube = data[keys[0]]
        except Exception:
            dropped += 1
            continue

        if cube.ndim != 3:
            dropped += 1
            continue

        if cube.shape[0] >= expected_channels:
            kept.append(path)
        else:
            dropped += 1

    print(f"  共 {len(file_list)} 个文件，保留 {len(kept)} 个完整(>={expected_channels}通道)，丢弃 {dropped} 个。")
    return kept


def split_train_val(files):
    """
    按时间划分训练/验证：
      - 训练: 2022 全年 + 2023年 1–5, 9–12 月
      - 验证: 2023 年 6–8 月
      - 2024 全部为测试，不在这个函数里返回
    """
    train_files, val_files = [], []
    for path in files:
        year, month = extract_year_month(path)
        if year is None or month is None:
            train_files.append(path)
            continue

        if year == 2022:
            train_files.append(path)
        elif year == 2023:
            if month in (1, 2, 3, 4, 5, 9, 10, 11, 12):
                train_files.append(path)
            elif month in (6, 7, 8):
                val_files.append(path)

    print(f"  训练文件: {len(train_files)}, 验证文件(2023 JJA): {len(val_files)}")
    return train_files, val_files


def load_or_compute_stats(
    stats_dir: str,
    train_files,
    input_channels: int,
    num_total_targets: int,
    reg_indices,
    sza_index: int = 8,
    cloud_phase_index: int = 1,
    chunk_size: int = 200,
):
    """
    只用“训练文件”计算输入和 5 个回归标签的 mean/std：
      - 条件：
          * CldPhase>0 且 ≤4
          * 所有 15 通道非负
          * SZA ∈ [0, 180] 且有限（白天 + 夜间）
          * 5 个回归标签均为有限值
      - 逐文件在线累加 sum/sumsq/count，不占大量内存
    """
    os.makedirs(stats_dir, exist_ok=True)
    in_mean_path = os.path.join(stats_dir, "input_mean.npy")
    in_std_path = os.path.join(stats_dir, "input_std.npy")
    tgt_mean_path = os.path.join(stats_dir, "target_mean.npy")
    tgt_std_path = os.path.join(stats_dir, "target_std.npy")

    if (
        os.path.exists(in_mean_path)
        and os.path.exists(in_std_path)
        and os.path.exists(tgt_mean_path)
        and os.path.exists(tgt_std_path)
    ):
        print(f"从 {stats_dir} 读取已有的归一化统计量。")
        input_mean = np.load(in_mean_path).astype(np.float32)
        input_std = np.load(in_std_path).astype(np.float32)
        target_mean = np.load(tgt_mean_path).astype(np.float32)
        target_std = np.load(tgt_std_path).astype(np.float32)
        return input_mean, input_std, target_mean, target_std

    print("未发现已有统计量，将在训练集上在线计算 input/target 的 mean/std ...")
    n_in = input_channels
    n_tgt = len(reg_indices)
    expected_C = input_channels + num_total_targets

    sum_in = np.zeros(n_in, dtype=np.float64)
    sumsq_in = np.zeros(n_in, dtype=np.float64)
    cnt_in = np.zeros(n_in, dtype=np.int64)

    sum_t = np.zeros(n_tgt, dtype=np.float64)
    sumsq_t = np.zeros(n_tgt, dtype=np.float64)
    cnt_t = np.zeros(n_tgt, dtype=np.int64)

    total = len(train_files)
    for i, path in enumerate(train_files):
        try:
            data = np.load(path)
            if "data" in data:
                cube = data["data"]
            else:
                keys = list(data.keys())
                cube = data[keys[0]]
        except Exception:
            continue

        if cube.ndim != 3 or cube.shape[0] < expected_C:
            continue

        cube = cube[:expected_C].astype(np.float64)
        
        # 提取各部分：8 TIR + SZA + 10 labels
        tir_channels = cube[:8]                    # [8,H,W]
        sza = cube[sza_index]                      # [H,W]
        labels = cube[9:]                          # [10,H,W]
        
        # 提取VZA（labels的最后一个，索引9）
        vza = labels[9]                            # [H,W]
        
        # 提取CldType（labels索引8）
        cldtype = labels[cldtype_label_index]      # [H,W]
        
        # 提取7个回归标签
        reg_labels = labels[reg_label_indices]     # [7,H,W]

        # 筛选条件
        # 1) CldType > 1（云区域）
        mask = (cldtype > 1) & (cldtype <= 9) & np.isfinite(cldtype)

        # 2) SZA < 60度（白天数据）
        mask &= (sza >= 0.0) & (sza < 60.0) & np.isfinite(sza)

        # 3) 负值 → NaN
        tir_channels[tir_channels < 0] = np.nan
        vza_copy = vza.copy()
        vza_copy[vza_copy < 0] = np.nan
        reg_labels[reg_labels < 0] = np.nan

        # 4) TIR/VZA任一通道NaN → 无效
        nan_tir = np.any(~np.isfinite(tir_channels), axis=0)
        mask &= ~nan_tir
        mask &= np.isfinite(vza_copy)

        # 5) 回归标签任一通道NaN → 无效
        nan_reg = np.any(~np.isfinite(reg_labels), axis=0)
        mask &= ~nan_reg

        if not np.any(mask):
            continue

        flat_mask = mask.reshape(-1)

        # 输入统计：16通道 = [8 TIR] + [VZA] + [7 cloud params]
        # 统计TIR通道 (前8个)
        tir_flat = tir_channels.reshape(8, -1)
        for c in range(8):
            vals = tir_flat[c, flat_mask]
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            sum_in[c] += vals.sum()
            sumsq_in[c] += (vals ** 2).sum()
            cnt_in[c] += vals.size

        # 统计VZA（第9个）
        vza_flat = vza_copy.reshape(-1)
        vals = vza_flat[flat_mask]
        vals = vals[np.isfinite(vals)]
        if vals.size > 0:
            sum_in[8] += vals.sum()
            sumsq_in[8] += (vals ** 2).sum()
            cnt_in[8] += vals.size

        # 统计7个云参数（第10-16个）
        reg_flat = reg_labels.reshape(n_tgt, -1)
        for j in range(n_tgt):
            vals = reg_flat[j, flat_mask]
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            sum_in[9 + j] += vals.sum()
            sumsq_in[9 + j] += (vals ** 2).sum()
            cnt_in[9 + j] += vals.size
            
            # 同时也为回归目标统计量累加
            sum_t[j] += vals.sum()
            sumsq_t[j] += (vals ** 2).sum()
            cnt_t[j] += vals.size

        if (i + 1) % chunk_size == 0 or (i + 1) == total:
            print(f"  stats 进度: {i + 1}/{total} ({(i + 1) / total * 100:.1f}%)")
            gc.collect()

    eps = 1e-12
    input_mean = sum_in / np.maximum(cnt_in, 1)
    input_var = sumsq_in / np.maximum(cnt_in, 1) - input_mean ** 2
    input_std = np.sqrt(np.maximum(input_var, eps))

    target_mean = sum_t / np.maximum(cnt_t, 1)
    target_var = sumsq_t / np.maximum(cnt_t, 1) - target_mean ** 2
    target_std = np.sqrt(np.maximum(target_var, eps))

    input_mean = input_mean.astype(np.float32)
    input_std = input_std.astype(np.float32)
    target_mean = target_mean.astype(np.float32)
    target_std = target_std.astype(np.float32)

    np.save(in_mean_path, input_mean)
    np.save(in_std_path, input_std)
    np.save(tgt_mean_path, target_mean)
    np.save(tgt_std_path, target_std)

    print(f"统计量已保存到 {stats_dir}")
    return input_mean, input_std, target_mean, target_std


# ---------------- 主训练流程 ---------------- #
def train():
    with open("configs/base/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = torch.device(config["training"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    output_dir = config["training"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    tb_dir = os.path.join(output_dir, "tensorboard")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)

    print("\n搜索数据文件:")
    all_files = []
    for root in config["data"]["root_dirs"]:
        pattern = os.path.join(root, "**", "*.npz")
        files = glob.glob(pattern, recursive=True)
        print(f"  {root}: 找到 {len(files)} 个文件")
        all_files.extend(files)

    if not all_files:
        raise ValueError("没有找到任何 .npz 文件，请检查 data.root_dirs。")

    input_channels = config["model"]["input_channels"]
    num_total_labels = 10  # 10个标签
    expected_C = 8 + 1 + num_total_labels  # 8 TIR + SZA + 10 labels = 19
    all_files = filter_complete_files(all_files, expected_C)

    train_files, val_files = split_train_val(all_files)
    if not train_files:
        raise ValueError("训练文件列表为空，请检查时间划分规则。")
    if not val_files:
        print("警告: 验证集为空，将从训练集中随机划分 10% 作为验证。")
        rng = np.random.RandomState(42)
        rng.shuffle(train_files)
        split_idx = int(0.9 * len(train_files))
        train_files, val_files = train_files[:split_idx], train_files[split_idx:]
        print(f"重新划分后: 训练 {len(train_files)}, 验证 {len(val_files)}")

    stats_dir = config["data"]["stats_dir"]
    # 7个回归标签在10个标签中的索引: DCOMP35_CPS, DCOMP36_CPS, DCOMP37_CPS, DCOMP35_COD, CldPressure, CldHeight, CldTemperature
    reg_label_indices = [3, 4, 5, 2, 0, 6, 7]
    input_mean, input_std, target_mean, target_std = load_or_compute_stats(
        stats_dir=stats_dir,
        train_files=train_files,
        input_channels=input_channels,
        num_total_labels=num_total_labels,
        reg_label_indices=reg_label_indices,
        sza_index=8,
        cldtype_label_index=8,
        chunk_size=config["data"].get("stats_chunk_size", 200),
    )

    train_dataset = HimawariCloudDataset(
        file_list=train_files,
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
    val_dataset = HimawariCloudDataset(
        file_list=val_files,
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

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    model = TIR_CNN(
        input_channels=input_channels,
        output_channels_cls=10,  # CldType分类，10类
        output_channels_reg=7,   # 7个云参数回归
        base_channels=config["model"]["base_channels"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n模型参数总数: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["l2_lambda"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )
    criterion = TwoStageLoss(num_outputs=7)

    writer = SummaryWriter(os.path.join(tb_dir, datetime.now().strftime("%Y%m%d-%H%M%S")))

    best_val_loss = float("inf")
    best_ckpt_path = None
    patience = config["early_stopping"]["patience"]
    min_delta = config["early_stopping"]["min_delta"]
    patience_counter = 0

    reg_label_names = [
        "DCOMP35_CPS",
        "DCOMP36_CPS",
        "DCOMP37_CPS",
        "DCOMP35_COD",
        "CldPressure",
        "CldHeight",
        "CldTemperature",
    ]
    num_labels = len(reg_label_names)
    epochs = config["training"]["epochs"]

    print("\n开始训练:")
    for epoch in range(epochs):
        t0 = datetime.now()
        # --- train ---
        model.train()
        train_loss_sum = 0.0
        train_cls_loss_sum = 0.0
        train_reg_loss_sum = 0.0
        train_label_loss_sum = np.zeros(num_labels, dtype=np.float64)
        n_train_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [train]", ncols=120)
        for inputs, cldtype_targets, reg_targets in pbar:
            inputs = inputs.to(device, non_blocking=True)
            cldtype_targets = cldtype_targets.to(device, non_blocking=True)
            reg_targets = reg_targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            cldtype_logits, reg_preds = model(inputs)
            loss, cls_loss, reg_loss, per_label = criterion(cldtype_logits, reg_preds, cldtype_targets, reg_targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss_sum += loss.item()
            train_cls_loss_sum += cls_loss.item()
            train_reg_loss_sum += reg_loss.item()
            if per_label.numel() == num_labels:
                train_label_loss_sum += per_label.detach().cpu().numpy()
            n_train_batches += 1

            avg_train_loss = train_loss_sum / n_train_batches
            avg_cls_loss = train_cls_loss_sum / n_train_batches
            avg_reg_loss = train_reg_loss_sum / n_train_batches
            postfix = {"total": f"{avg_train_loss:.3f}", "cls": f"{avg_cls_loss:.3f}", "reg": f"{avg_reg_loss:.3f}"}
            if per_label.numel() == num_labels:
                avg_label = train_label_loss_sum / n_train_batches
                for i, name in enumerate(reg_label_names[:3]):  # 只显示前3个
                    postfix[name] = f"{avg_label[i]:.3f}"
            pbar.set_postfix(postfix)

        avg_train_loss = train_loss_sum / max(1, n_train_batches)
        avg_train_cls_loss = train_cls_loss_sum / max(1, n_train_batches)
        avg_train_reg_loss = train_reg_loss_sum / max(1, n_train_batches)
        avg_train_label = train_label_loss_sum / max(1, n_train_batches)
        writer.add_scalar("Loss/train_total", avg_train_loss, epoch)
        writer.add_scalar("Loss/train_cls", avg_train_cls_loss, epoch)
        writer.add_scalar("Loss/train_reg", avg_train_reg_loss, epoch)
        for i, name in enumerate(reg_label_names):
            writer.add_scalar(f"Loss/train_{name}", avg_train_label[i], epoch)

        # --- val ---
        model.eval()
        val_loss_sum = 0.0
        n_val_batches = 0
        with torch.no_grad():
            for inputs, cldtype_targets, reg_targets in tqdm(
                val_loader, desc=f"Epoch {epoch+1}/{epochs} [val]", ncols=120
            ):
                inputs = inputs.to(device, non_blocking=True)
                cldtype_targets = cldtype_targets.to(device, non_blocking=True)
                reg_targets = reg_targets.to(device, non_blocking=True)
                cldtype_logits, reg_preds = model(inputs)
                loss, _, _, _ = criterion(cldtype_logits, reg_preds, cldtype_targets, reg_targets)
                val_loss_sum += loss.item()
                n_val_batches += 1

        avg_val_loss = val_loss_sum / max(1, n_val_batches)
        writer.add_scalar("Loss/val_total", avg_val_loss, epoch)
        scheduler.step(avg_val_loss)

        dt = (datetime.now() - t0).total_seconds() / 60.0
        print(
            f"Epoch {epoch+1}/{epochs} 结束: "
            f"耗时 {dt:.2f} min | train={avg_train_loss:.4f}, val={avg_val_loss:.4f}"
        )

        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_ckpt_path = os.path.join(ckpt_dir, "best_model.pth")
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"  -> 新的最优模型已保存: {best_ckpt_path}")
        else:
            patience_counter += 1
            print(f"  -> 验证集未提升: patience {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("早停触发，结束训练。")
            break

    writer.close()
    if best_ckpt_path:
        print(f"\n训练完成，最佳模型保存在: {best_ckpt_path}")
    else:
        print("\n训练完成，但未生成最佳模型（请检查数据/损失是否异常）。")


if __name__ == "__main__":
    train()
