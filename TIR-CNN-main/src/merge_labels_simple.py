#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复版：合并旧数据NPZ和新标签NPZ
"""

import os
import glob
import re
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm


def extract_file_key(filename):
    """提取文件匹配键：时间_位置"""
    # 匹配:  20220601_0000Z_X0_Y0
    pattern = r'(\d{8}_\d{4}Z_X\d+_Y\d+)'
    match = re.search(pattern, filename)
    if match:
        return match.group(1)
    return None


def merge_single_month(old_dir, new_dir, output_dir, overwrite=False):
    """合并单个月份的数据"""
    
    print(f"\n{'='*70}")
    print(f"旧数据目录:  {old_dir}")
    print(f"新标签目录: {new_dir}")
    print(f"输出目录:    {output_dir}")
    print(f"{'='*70}")
    
    # 检查目录
    if not os.path.exists(old_dir):
        print(f"❌ 旧数据目录不存在: {old_dir}")
        return 0, 0, 0
    
    if not os.path.exists(new_dir):
        print(f"❌ 新标签目录不存在: {new_dir}")
        return 0, 0, 0
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 收集文件 - 修复：使用 ** 匹配所有 npz 然后过滤
    old_files = glob.glob(os.path. join(old_dir, "*.npz"))
    
    # 修复：先获取所有npz，再过滤NewLabels开头的
    all_npz_in_new_dir = glob.glob(os. path.join(new_dir, "*.npz"))
    new_files = [f for f in all_npz_in_new_dir if os.path.basename(f).startswith("NewLabels")]
    
    print(f"\n找到文件:")
    print(f"  旧数据: {len(old_files)} 个")
    print(f"  新目录所有NPZ: {len(all_npz_in_new_dir)} 个")
    print(f"  新标签(NewLabels开头): {len(new_files)} 个")
    
    if len(old_files) == 0:
        print("❌ 旧数据目录中没有NPZ文件")
        return 0, 0, 0
    
    if len(new_files) == 0:
        print("❌ 新标签目录中没有以NewLabels开头的NPZ文件")
        print(f"\n该目录中的所有NPZ文件（前10个）:")
        for f in all_npz_in_new_dir[:10]:
            print(f"  - {os.path.basename(f)}")
        return 0, 0, 0
    
    # 建立新标签文件的索引
    new_label_dict = {}
    for new_file in new_files: 
        filename = os.path.basename(new_file)
        key = extract_file_key(filename)
        if key: 
            new_label_dict[key] = new_file
    
    print(f"\n可匹配的新标签:  {len(new_label_dict)} 个")
    
    # 验证数据
    print("\n验证数据结构...")
    try:
        sample_old = np.load(old_files[0])
        sample_new = np.load(new_files[0])
        
        old_shape = sample_old["data"].shape if "data" in sample_old else "未知"
        new_shape = sample_new["data"].shape if "data" in sample_new else "未知"
        
        print(f"  旧数据示例: {os.path.basename(old_files[0])}")
        print(f"    shape: {old_shape}")
        print(f"  新标签示例: {os.path.basename(new_files[0])}")
        print(f"    shape: {new_shape}")
    except Exception as e:
        print(f"  ⚠️ 无法加载示例文件: {e}")
    
    # 合并文件
    print("\n开始合并...")
    success = 0
    fail = 0
    skip = 0
    no_match = 0
    
    for old_file in tqdm(old_files, desc="合并", ncols=100):
        old_filename = os.path.basename(old_file)
        key = extract_file_key(old_filename)
        
        if not key:
            fail += 1
            continue
        
        # 查找匹配的新标签
        if key not in new_label_dict: 
            no_match += 1
            continue
        
        new_file = new_label_dict[key]
        output_file = os.path.join(output_dir, old_filename)
        
        # 检查是否已存在
        if os.path. exists(output_file) and not overwrite:
            skip += 1
            continue
        
        try:
            # 加载数据
            old_data = np.load(old_file)
            new_data = np.load(new_file)
            
            if "data" not in old_data or "data" not in new_data:
                fail += 1
                continue
            
            old_cube = old_data["data"]
            new_cube = new_data["data"]
            
            # 检查空间维度
            if old_cube. shape[1: ] != new_cube.shape[1:]:
                print(f"\n  ✗ 形状不匹配: {old_filename}")
                print(f"    旧:  {old_cube.shape}, 新: {new_cube. shape}")
                fail += 1
                continue
            
            # 拼接
            merged = np.concatenate([old_cube, new_cube], axis=0)
            
            # 保存
            np.savez_compressed(output_file, data=merged)
            success += 1
            
        except Exception as e:
            print(f"\n  ✗ 错误:  {old_filename} - {e}")
            fail += 1
    
    print(f"\n详细统计:")
    print(f"  成功匹配并合并: {success}")
    print(f"  找不到对应新标签: {no_match}")
    print(f"  处理失败: {fail}")
    print(f"  已存在跳过: {skip}")
    
    return success, fail, skip


def main():
    parser = argparse.ArgumentParser(description='合并旧数据和新标签NPZ文件')
    parser.add_argument('--old_dir', type=str, required=True,
                        help='旧数据目录（15通道）')
    parser.add_argument('--new_dir', type=str, required=True,
                        help='新标签目录（4通道，包含NewLabels*. npz）')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='输出目录（19通道）')
    parser.add_argument('--overwrite', action='store_true',
                        help='覆盖已存在的文件')
    
    args = parser.parse_args()
    
    success, fail, skip = merge_single_month(
        args.old_dir,
        args.new_dir,
        args.output_dir,
        args.overwrite
    )
    
    print(f"\n{'='*70}")
    print("合并完成!")
    print(f"  成功: {success}")
    print(f"  失败:  {fail}")
    print(f"  跳过: {skip}")
    print(f"{'='*70}")
    
    # 验证输出
    if success > 0:
        print("\n验证输出（前5个文件）...")
        output_files = sorted(Path(args.output_dir).glob("*.npz"))[:5]
        for f in output_files:
            try:
                data = np.load(f)
                if "data" in data:
                    shape = data["data"].shape
                    size_mb = f.stat().st_size / (1024**2)
                    print(f"  ✓ {f.name}: shape={shape}, size={size_mb:.2f}MB")
            except Exception as e:
                print(f"  ✗ {f.name}: {e}")
        
        # 生成feature_names. txt
        feature_names_file = os.path.join(args.output_dir, 'feature_names.txt')
        all_features = [
            'tbb_08', 'tbb_09', 'tbb_10', 'tbb_11',
            'tbb_13', 'tbb_14', 'tbb_15', 'tbb_16',
            'solar_zenith_angle',
            'CldPressure', 'CldPhase', 'DCOMP35_COD',
            'DCOMP35_CPS', 'DCOMP36_CPS', 'DCOMP37_CPS',
            'CldHeight', 'CldTemperature', 'CldType', 'VZA'
        ]
        
        with open(feature_names_file, 'w') as f:
            for i, name in enumerate(all_features):
                f.write(f"{i}:  {name}\n")
        
        print(f"\n✅ 已生成 feature_names.txt: {feature_names_file}")


if __name__ == '__main__':
    main()