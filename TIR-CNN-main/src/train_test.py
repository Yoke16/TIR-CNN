#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试训练脚本 - 修正版
1. 增加文件路径强制去重，防止父子目录递归导致的重复读取。
2. 保留统计量保存/加载功能。
3. 修复 Loss 参数和 inf 溢出问题。
"""

import glob
import gc
import os
import sys
from datetime import datetime

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# 导入你原有的模块
from dataset import HimawariCloudDataset
from loss import TwoStageLoss
from model import TIR_CNN


def split_train_val_test(files):
    """
    针对2022年6-8月数据进行划分: 
      - 训练: 6月 + 7月
      - 验证: 8月
    """
    train_files, val_files = [], []
    
    # 按照文件名进行分类
    for path in files:
        filename = os.path.basename(path)
        # 从文件名提取月份
        if "_202206" in filename:
            train_files.append(path)
        elif "_202207" in filename:
            train_files.append(path)
        elif "_202208" in filename:
            val_files.append(path)
        else:
            # 如果是散落在 new_2022 根目录下的文件（通常是6月），也放入训练集
            # 这里可以根据你的实际文件名特征调整
            train_files.append(path)
    
    print(f"  训练文件(6-7月): {len(train_files)}")
    print(f"  验证文件(8月): {len(val_files)}")
    
    return train_files, val_files


def filter_complete_files(file_list, expected_channels: int = 19):
    """只保留通道数 >= expected_channels 的文件"""
    kept = []
    dropped = 0
    
    # 为了加快速度，这里只做简单的检查，不再打开所有文件
    # 如果你需要严格检查，可以保留原来的代码
    # 这里优化为：只在训练 DataLoader 里遇到坏文件时跳过
    
    print("正在检查文件列表...")
    # 暂时跳过耗时的逐个文件检查，依靠 DataLoader 的健壮性
    # 如果你想保留严格检查，请取消下面注释，但这会很慢
    kept = file_list 
    
    """
    for path in tqdm(file_list, desc=f"检查通道数(>={expected_channels})", ncols=80):
        try:
            # 这是一个非常耗时的操作，如果文件有几十万个，建议先跳过
            # data = np.load(path)
            # ...
            kept.append(path)
        except Exception: 
            dropped += 1
            continue
    """
    
    return kept


def load_or_compute_stats(stats_dir, file_list, input_channels=16, num_total_targets=10, 
                         reg_indices=[3,4,5,2,0,6,7], chunk_size=100):
    """
    计算或加载数据的均值和标准差。
    """
    os.makedirs(stats_dir, exist_ok=True)
    
    in_mean_path = os.path.join(stats_dir, "input_mean.npy")
    in_std_path = os.path.join(stats_dir, "input_std.npy")
    tgt_mean_path = os.path.join(stats_dir, "target_mean.npy")
    tgt_std_path = os.path.join(stats_dir, "target_std.npy")

    # 1. 尝试加载
    if (os.path.exists(in_mean_path) and os.path.exists(in_std_path) and 
        os.path.exists(tgt_mean_path) and os.path.exists(tgt_std_path)):
        print(f"\n从 {stats_dir} 加载已有统计量...")
        input_mean = np.load(in_mean_path).astype(np.float32)
        input_std = np.load(in_std_path).astype(np.float32)
        target_mean = np.load(tgt_mean_path).astype(np.float32)
        target_std = np.load(tgt_std_path).astype(np.float32)
        
        print(f"  输入均值范围: [{input_mean.min():.2f}, {input_mean.max():.2f}]")
        return input_mean, input_std, target_mean, target_std

    # 2. 如果不存在，则开始计算
    print(f"\n未找到统计量，开始计算并保存到 {stats_dir} ...")
    
    input_sum = np.zeros(input_channels, dtype=np.float64)
    input_sq_sum = np.zeros(input_channels, dtype=np.float64)
    target_sum = np.zeros(len(reg_indices), dtype=np.float64)
    target_sq_sum = np.zeros(len(reg_indices), dtype=np.float64)
    count = 0
    
    # 为了节省时间，统计时可以只采样一部分数据（例如每隔10个取1个）
    # 如果数据量巨大，这是常见做法
    sample_step = 5 
    sampled_files = file_list[::sample_step]
    print(f"  采样 {len(sampled_files)}/{len(file_list)} 个文件进行统计计算...")

    for i in tqdm(range(0, len(sampled_files), chunk_size), desc="统计计算", ncols=80):
        chunk = sampled_files[i:i+chunk_size]
        
        for path in chunk:
            try:
                data = np.load(path)
                if "data" in data:
                    cube = data["data"]
                else:
                    cube = data[list(data.keys())[0]]
                
                tir_channels = cube[:8, :, :]
                sza = cube[8:9, :, :]
                labels = cube[9:, :, :]
                vza = labels[9:10, :, :]
                cloud_params = labels[reg_indices, :, :]
                
                inputs = np.concatenate([tir_channels, vza, cloud_params], axis=0)
                cldtype = labels[8:9, :, :]
                
                # Valid Mask (含 inf/nan 修复)
                valid_mask = (
                    (sza < 60) &
                    (cldtype > 1) &
                    np.all(np.isfinite(inputs), axis=0, keepdims=True) & 
                    np.all(inputs >= 0, axis=0, keepdims=True)
                )
                
                valid_inputs = inputs[:, valid_mask[0, :, :]]
                valid_targets = cloud_params[:, valid_mask[0, :, :]]
                
                if valid_inputs.shape[1] > 0:
                    input_sum += valid_inputs.sum(axis=1)
                    input_sq_sum += (valid_inputs ** 2).sum(axis=1)
                    target_sum += valid_targets.sum(axis=1)
                    target_sq_sum += (valid_targets ** 2).sum(axis=1)
                    count += valid_inputs.shape[1]
                    
            except Exception: 
                continue
    
    if count == 0:
        raise ValueError("计算统计量失败：没有找到任何有效像素！")
    
    input_mean = input_sum / count
    input_std = np.sqrt(np.maximum(input_sq_sum / count - input_mean ** 2, 0))
    target_mean = target_sum / count
    target_std = np.sqrt(np.maximum(target_sq_sum / count - target_mean ** 2, 0))
    
    input_std = np.where(input_std < 1e-8, 1.0, input_std)
    target_std = np.where(target_std < 1e-8, 1.0, target_std)
    
    print(f"  有效像素总数: {count:,}")
    np.save(in_mean_path, input_mean.astype(np.float32))
    np.save(in_std_path, input_std.astype(np.float32))
    np.save(tgt_mean_path, target_mean.astype(np.float32))
    np.save(tgt_std_path, target_std.astype(np.float32))
    print(f"✅ 统计量已保存到: {stats_dir}")
    
    return input_mean, input_std, target_mean, target_std


def main():
    config_path = "configs/test/test_2022_jja.yaml"
    
    if not os.path.exists(config_path):
        print(f"配置文件不存在: {config_path}")
        return
    
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    print("=" * 70)
    print("测试训练 - 2022年6-8月数据 (修正版)")
    print("=" * 70)
    
    # ------------------ 修正的核心部分：文件去重 ------------------
    all_files_list = []
    print("正在搜索文件...")
    for root_dir in cfg['data']['root_dirs']:
        # 递归搜索
        files = glob.glob(os.path.join(root_dir, "**", "*.npz"), recursive=True)
        all_files_list.extend(files)
        print(f"  > 从 {root_dir} (递归) 找到 {len(files)} 个文件")
    
    # 使用 set 进行强制去重
    original_count = len(all_files_list)
    unique_files = sorted(list(set(all_files_list)))
    final_count = len(unique_files)
    
    print("-" * 30)
    print(f"原始找到文件数: {original_count}")
    print(f"去重后文件数:   {final_count}")
    print(f"重复文件数:     {original_count - final_count}")
    print("-" * 30)
    
    if final_count == 0:
        print("❌ 没有找到任何NPZ文件！")
        return
    # -----------------------------------------------------------

    # 简单的文件过滤（不再逐个打开检查，太慢）
    complete_files = unique_files
    
    # 划分训练/验证集
    train_files, val_files = split_train_val_test(complete_files)
    
    if len(train_files) == 0 or len(val_files) == 0:
        print("❌ 训练集或验证集为空！请检查文件名是否包含 _202206 / _202207 / _202208")
        return
    
    stats_dir = cfg['data']['stats_dir']
    reg_indices = [3, 4, 5, 2, 0, 6, 7]
    
    input_mean, input_std, target_mean, target_std = load_or_compute_stats(
        stats_dir=stats_dir,
        file_list=train_files, 
        input_channels=16,
        num_total_targets=10,
        reg_indices=reg_indices,
        chunk_size=100
    )
    
    print("\n创建数据集...")
    train_dataset = HimawariCloudDataset(
        train_files, input_mean, input_std, target_mean, target_std,
        input_channels=16, num_total_labels=10, sza_index=8, cldtype_label_index=8, reg_label_indices=reg_indices
    )
    
    val_dataset = HimawariCloudDataset(
        val_files, input_mean, input_std, target_mean, target_std,
        input_channels=16, num_total_labels=10, sza_index=8, cldtype_label_index=8, reg_label_indices=reg_indices
    )
    
    print(f"  训练集样本数: {len(train_dataset):,}")
    print(f"  验证集样本数: {len(val_dataset):,}")
    
    train_loader = DataLoader(
        train_dataset, batch_size=cfg['training']['batch_size'], shuffle=True,
        num_workers=cfg['training']['num_workers'], pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=cfg['training']['batch_size'], shuffle=False,
        num_workers=cfg['training']['num_workers'], pin_memory=False
    )
    
    print("\n创建模型...")
    device = torch.device(cfg['training']['device'])
    model = TIR_CNN(
        input_channels=cfg['model']['input_channels'],
        output_channels_cls=cfg['model']['output_channels_cls'],
        output_channels_reg=cfg['model']['output_channels_reg'],
        base_channels=cfg['model']['base_channels']
    ).to(device)
    
    # 修正 Loss
    criterion = TwoStageLoss(
        num_outputs=cfg['model']['output_channels_reg'],
        cls_weight=0.5,
        reg_weight=0.5
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['training']['learning_rate'])
    
    output_dir = cfg['training']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'logs'))
    
    print("\n" + "=" * 70)
    print("开始训练...")
    print("=" * 70)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(cfg['training']['epochs']):
        model.train()
        train_loss_sum = 0.0
        train_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['training']['epochs']} [Train]", ncols=100)
        for batch_idx, (inputs, cls_targets, reg_targets) in enumerate(pbar):
            inputs, cls_targets, reg_targets = inputs.to(device), cls_targets.to(device), reg_targets.to(device)
            
            optimizer.zero_grad()
            cls_pred, reg_pred = model(inputs)
            loss, cls_loss, reg_loss, _ = criterion(cls_pred, reg_pred, cls_targets, reg_targets)
            loss.backward()
            optimizer.step()
            
            train_loss_sum += loss.item()
            train_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss_sum / max(train_batches, 1)
        
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{cfg['training']['epochs']} [Val]", ncols=100)
            for inputs, cls_targets, reg_targets in pbar:
                inputs, cls_targets, reg_targets = inputs.to(device), cls_targets.to(device), reg_targets.to(device)
                cls_pred, reg_pred = model(inputs)
                loss, cls_loss, reg_loss, _ = criterion(cls_pred, reg_pred, cls_targets, reg_targets)
                val_loss_sum += loss.item()
                val_batches += 1
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_val_loss = val_loss_sum / max(val_batches, 1)
        
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss - cfg['early_stopping']['min_delta']: 
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
            print(f"  ✓ 保存最佳模型")
        else:
            patience_counter += 1
        
        if patience_counter >= cfg['early_stopping']['patience']: 
            print("Early stopping")
            break
            
    writer.close()
    print("训练完成！")

if __name__ == "__main__": 
    main()