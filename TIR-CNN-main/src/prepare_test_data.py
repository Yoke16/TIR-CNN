#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将2022年6、7、8月的NPZ数据拷贝到目标目录用于训练测试
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm

def prepare_test_data():
    # 源数据路径
    SOURCE_BASE = "/array1/satellite/lujc/output_2022"
    
    # 目标路径
    TARGET_BASE = "/array1/satellite/lujc/npy_dir_2022"
    
    # 要拷贝的月份
    MONTHS = ["06", "07", "08"]
    YEAR = 2022
    
    # 创建目标目录
    Path(TARGET_BASE).mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("准备2022年6-8月测试数据")
    print(f"源目录: {SOURCE_BASE}")
    print(f"目标目录: {TARGET_BASE}")
    print("=" * 70)
    
    total_files = 0
    total_size = 0
    
    for month in MONTHS:
        source_dir = f"{SOURCE_BASE}/month_{month}"
        
        if not os.path.isdir(source_dir):
            print(f"✗ 目录不存在: {source_dir}")
            continue
        
        # 获取该月所有npz文件
        npz_files = sorted([f for f in os.listdir(source_dir) if f.endswith('.npz')])
        
        print(f"\n处理 {YEAR}年{month}月:  共 {len(npz_files)} 个文件")
        print(f"  源:  {source_dir}")
        
        # 拷贝文件
        for filename in tqdm(npz_files, desc=f"month_{month}", ncols=80):
            source_file = os.path.join(source_dir, filename)
            target_file = os.path.join(TARGET_BASE, filename)
            
            # 检查目标文件是否已存在
            if os.path.exists(target_file):
                # 文件已存在，跳过
                file_size = os.path.getsize(source_file)
                total_size += file_size
                total_files += 1
                continue
            
            try:
                # 使用硬链接（更快，不占用额外空间）
                # 如果跨文件系统则使用拷贝
                try:
                    os.link(source_file, target_file)
                except OSError: 
                    # 硬链接失败，使用拷贝
                    shutil.copy2(source_file, target_file)
                
                file_size = os.path.getsize(source_file)
                total_size += file_size
                total_files += 1
                
            except Exception as e: 
                print(f"\n  ✗ 复制失败: {filename} - {e}")
    
    print("\n" + "=" * 70)
    print(f"✓ 完成！共处理 {total_files} 个文件")
    print(f"  总大小: {total_size / (1024**3):.2f} GB")
    print(f"  目标目录: {TARGET_BASE}")
    print("=" * 70)
    
    # 验证数据
    print("\n验证数据完整性（检查前5个文件）...")
    import numpy as np
    
    sample_files = sorted(Path(TARGET_BASE).glob("*.npz"))[:5]
    
    if len(sample_files) == 0:
        print("  ✗ 目标目录中没有找到NPZ文件！")
        return
    
    for f in sample_files:
        try:
            data = np.load(f)
            if "data" in data: 
                shape = data["data"].shape
            else:
                shape = data[list(data.keys())[0]].shape
            print(f"  ✓ {f.name}:  shape={shape}")
        except Exception as e:
            print(f"  ✗ {f. name}: {e}")
    
    # 统计目标目录文件总数
    all_files = list(Path(TARGET_BASE).glob("*.npz"))
    print(f"\n目标目录总文件数: {len(all_files)}")

if __name__ == "__main__":
    prepare_test_data()