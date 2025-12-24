"""
计算新增标签的统计量 (mean/std)

新增标签包括:
- VZA (观测天顶角)
- 7个云参数: DCOMP35_CPS, DCOMP36_CPS, DCOMP37_CPS, DCOMP35_COD, CldPressure, CldHeight, CldTemperature

注意:
- 这个脚本假设npz文件已经包含19个通道: [8 TIR] + [SZA] + [10 labels]
- 10个labels: CldPressure, CldPhase, DCOMP35_COD, DCOMP35_CPS, DCOMP36_CPS, DCOMP37_CPS, CldHeight, CldTemperature, CldType, VZA
- 只使用训练集数据计算统计量
- 筛选条件: SZA < 60 度 (白天数据), CldType > 1 (云区域), 所有通道无负值/NaN/0值

使用方法:
    python compute_new_stats.py --npz_dirs /path/to/train/data1 /path/to/train/data2 --output_dir ./stats
"""
import os
import glob
import re
import argparse
import numpy as np
from tqdm import tqdm
import gc


def extract_year_month(path):
    """从文件名中解析 YYYYMMDD_HHMMZ，返回 (year, month)"""
    filename = os.path.basename(path)
    m = re.search(r"(\d{8})_(\d{4})Z", filename)
    if m:
        datestr = m.group(1)
        return int(datestr[:4]), int(datestr[4:6])
    return None, None


def filter_train_files(file_list):
    """
    筛选训练文件: 2022全年 + 2023年1-5, 9-12月
    """
    train_files = []
    for path in file_list:
        year, month = extract_year_month(path)
        if year is None or month is None:
            train_files.append(path)
            continue
        
        if year == 2022:
            train_files.append(path)
        elif year == 2023:
            if month in (1, 2, 3, 4, 5, 9, 10, 11, 12):
                train_files.append(path)
    
    return train_files


def compute_new_stats(npz_dirs, output_dir, chunk_size=200):
    """
    计算VZA和7个云参数的mean/std
    
    参数:
        npz_dirs: npz文件目录列表
        output_dir: 输出目录
        chunk_size: 每处理多少个文件显示一次进度
    
    通道索引（假设19通道）:
        0-7: tbb_08 ~ tbb_16 (8个TIR通道)
        8: SolarZenithAngle
        9: CldPressure
        10: CldPhase
        11: DCOMP35_COD
        12: DCOMP35_CPS
        13: DCOMP36_CPS
        14: DCOMP37_CPS
        15: CldHeight
        16: CldTemperature
        17: CldType
        18: VZA
    
    需要计算的通道:
        - VZA (索引18)
        - 7个云参数: DCOMP35_CPS(12), DCOMP36_CPS(13), DCOMP37_CPS(14), 
                     DCOMP35_COD(11), CldPressure(9), CldHeight(15), CldTemperature(16)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 收集所有npz文件
    all_files = []
    for npz_dir in npz_dirs:
        pattern = os.path.join(npz_dir, "**", "*.npz")
        files = glob.glob(pattern, recursive=True)
        all_files.extend(files)
    
    if not all_files:
        print(f"错误: 未找到任何npz文件")
        return
    
    print(f"找到 {len(all_files)} 个文件")
    
    # 筛选训练文件
    train_files = filter_train_files(all_files)
    print(f"训练文件数: {len(train_files)}")
    
    if not train_files:
        print("错误: 没有训练文件")
        return
    
    # 定义通道索引
    sza_index = 8
    cldtype_index = 17
    vza_index = 18
    
    # 需要计算统计量的通道（按输入顺序）
    # 输入顺序: [8 TIR] + [VZA] + [7 cloud params]
    # 在npz中的索引:
    stat_indices = [
        18,  # VZA
        12,  # DCOMP35_CPS
        13,  # DCOMP36_CPS
        14,  # DCOMP37_CPS
        11,  # DCOMP35_COD
        9,   # CldPressure
        15,  # CldHeight
        16,  # CldTemperature
    ]
    
    n_stats = len(stat_indices)
    
    # 初始化累加器
    sum_vals = np.zeros(n_stats, dtype=np.float64)
    sumsq_vals = np.zeros(n_stats, dtype=np.float64)
    cnt_vals = np.zeros(n_stats, dtype=np.int64)
    
    print("\n开始计算统计量...")
    total = len(train_files)
    
    for i, path in enumerate(train_files):
        try:
            data = np.load(path)
            if "data" not in data:
                continue
            
            cube = data["data"]
        except Exception:
            continue
        
        if cube.ndim != 3 or cube.shape[0] < 19:
            continue
        
        cube = cube[:19].astype(np.float64)
        
        # 获取筛选条件
        sza = cube[sza_index]
        cldtype = cube[cldtype_index]
        
        # 构建mask
        # 1. SZA < 60度 (白天数据)
        mask = (sza < 60.0) & (sza >= 0.0) & np.isfinite(sza)
        
        # 2. CldType > 1 (云区域, 值为2-9)
        mask &= (cldtype > 1) & (cldtype <= 9) & np.isfinite(cldtype)
        
        # 3. 所有通道无负值、NaN
        # 检查前15个通道 (8 TIR + SZA + 6 labels)
        for c in range(15):
            channel_data = cube[c]
            mask &= (channel_data >= 0) & np.isfinite(channel_data)
        
        # 4. 新增的4个标签也需要检查
        for c in [15, 16, 17, 18]:  # CldHeight, CldTemperature, CldType, VZA
            channel_data = cube[c]
            mask &= (channel_data >= 0) & np.isfinite(channel_data)
        
        if not np.any(mask):
            continue
        
        flat_mask = mask.reshape(-1)
        
        # 对每个需要计算统计量的通道累加
        for j, idx in enumerate(stat_indices):
            channel_flat = cube[idx].reshape(-1)
            vals = channel_flat[flat_mask]
            vals = vals[np.isfinite(vals)]
            
            if vals.size == 0:
                continue
            
            sum_vals[j] += vals.sum()
            sumsq_vals[j] += (vals ** 2).sum()
            cnt_vals[j] += vals.size
        
        if (i + 1) % chunk_size == 0 or (i + 1) == total:
            print(f"  进度: {i + 1}/{total} ({(i + 1) / total * 100:.1f}%)")
            gc.collect()
    
    # 计算mean和std
    eps = 1e-12
    mean_vals = sum_vals / np.maximum(cnt_vals, 1)
    var_vals = sumsq_vals / np.maximum(cnt_vals, 1) - mean_vals ** 2
    std_vals = np.sqrt(np.maximum(var_vals, eps))
    
    mean_vals = mean_vals.astype(np.float32)
    std_vals = std_vals.astype(np.float32)
    
    # 保存统计量
    mean_file = os.path.join(output_dir, 'new_labels_mean.npy')
    std_file = os.path.join(output_dir, 'new_labels_std.npy')
    
    np.save(mean_file, mean_vals)
    np.save(std_file, std_vals)
    
    print(f"\n✅ 统计量已保存:")
    print(f"   Mean: {mean_file}")
    print(f"   Std: {std_file}")
    
    # 打印统计信息
    label_names = ['VZA', 'DCOMP35_CPS', 'DCOMP36_CPS', 'DCOMP37_CPS', 
                   'DCOMP35_COD', 'CldPressure', 'CldHeight', 'CldTemperature']
    
    print("\n统计结果:")
    print(f"{'Label':<15} {'Count':>12} {'Mean':>12} {'Std':>12}")
    print("-" * 55)
    for i, name in enumerate(label_names):
        print(f"{name:<15} {cnt_vals[i]:>12d} {mean_vals[i]:>12.4f} {std_vals[i]:>12.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='计算新增标签的统计量')
    parser.add_argument('--npz_dirs', type=str, nargs='+', required=True,
                        help='npz文件目录列表（可以包含子目录）')
    parser.add_argument('--output_dir', type=str, default='./stats',
                        help='输出目录 (默认: ./stats)')
    parser.add_argument('--chunk_size', type=int, default=200,
                        help='每处理多少个文件显示一次进度 (默认: 200)')
    
    args = parser.parse_args()
    
    for npz_dir in args.npz_dirs:
        if not os.path.exists(npz_dir):
            print(f"警告: 目录 {npz_dir} 不存在，将被忽略")
    
    compute_new_stats(args.npz_dirs, args.output_dir, args.chunk_size)
