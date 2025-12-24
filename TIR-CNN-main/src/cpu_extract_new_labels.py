"""
CPU服务器端：从nc.gz文件中提取新标签（CldHeight, CldTemperature, CldType）和VZA，
进行250x250切片，保存为npz文件

使用方法:
    python cpu_extract_new_labels.py \
        --hcfd_dir /home/lujc/NJIAS_H/NJIAS_HCFD_v1/0.04Deg/2022 \
        --vza_file /home/lujc/NJIAS_HCFD_0.04deg_SatZenAngle.nc \
        --output_dir /path/to/output \
        --month 1

数据源:
- CldHeight, CldTemperature, CldType: /home/lujc/NJIAS_H/NJIAS_HCFD_v1/0.04Deg/YYYY/YYYYMMDD/NJIAS_HCFD_0.04deg_YYYYMMDD_HHMMZ.nc.gz
- VZA (卫星天顶角): /home/lujc/NJIAS_HCFD_0.04deg_SatZenAngle.nc (无时间维度)

输出:
- 4通道npz文件: [CldHeight, CldTemperature, CldType, VZA]
- 文件命名: NewLabels_0.04deg_YYYYMMDD_HHMMZ_X###_Y###.npz
"""
import os
import glob
import gzip
import io
import re
import argparse
import xarray as xr
import numpy as np
from tqdm import tqdm


def compress_data(data, scale_factor=1000, dtype=np.float16):
    """压缩数据为float16以节省空间"""
    if data is not None:
        nan_mask = np.isnan(data)
        compressed_data = np.round(data * scale_factor) / scale_factor
        compressed_data = compressed_data.astype(dtype)
        compressed_data[nan_mask] = np.nan
        return compressed_data
    return data


def extract_time_from_filename(filename):
    """从文件名中提取时间信息"""
    pattern = r'(\d{8})_(\d{4})Z'
    match = re.search(pattern, os.path.basename(filename))
    if match:
        date_str = match.group(1)
        time_str = match.group(2)
        return date_str, time_str
    return None, None


def process_hcfd_file(hcfd_file_gz, vza_data, output_dir):
    """
    处理单个HCFD nc.gz文件，提取新标签并切片保存
    
    参数:
        hcfd_file_gz: HCFD nc.gz文件路径
        vza_data: VZA数据数组 [lat, lon]
        output_dir: 输出目录
    
    返回:
        成功处理的切片数量
    """
    try:
        # 1. 读取HCFD nc.gz文件
        with open(hcfd_file_gz, 'rb') as f:
            gz_data = f.read()
        
        with gzip.GzipFile(fileobj=io.BytesIO(gz_data)) as gz_file:
            netcdf_bytes = gz_file.read()
        netcdf_file = io.BytesIO(netcdf_bytes)
        
        hcfd_ds = xr.open_dataset(netcdf_file)
        
        # 2. 检查是否包含需要的标签
        required_labels = ['CldHeight', 'CldTemperature', 'CldType']
        for label in required_labels:
            if label not in hcfd_ds:
                print(f"  警告: {os.path.basename(hcfd_file_gz)} 缺少 {label}，跳过")
                return 0
        
        # 3. 提取3个标签
        cld_height = hcfd_ds['CldHeight'].values  # [lat, lon]
        cld_temperature = hcfd_ds['CldTemperature'].values
        cld_type = hcfd_ds['CldType'].values
        
        # 4. 获取VZA数据（需要确保维度匹配）
        if vza_data.shape != cld_height.shape:
            print(f"  警告: VZA维度 {vza_data.shape} 与标签维度 {cld_height.shape} 不匹配，跳过")
            return 0
        
        # 5. 构建4通道数据立方体: [CldHeight, CldTemperature, CldType, VZA]
        data_cube = np.stack([cld_height, cld_temperature, cld_type, vza_data], axis=0)  # [4, H, W]
        
        # 6. 压缩数据
        SCALE_FACTOR = 1000.0
        DATA_TYPE = np.float16
        compressed_data_cube = compress_data(data_cube, SCALE_FACTOR, DATA_TYPE)
        
        # 7. 提取时间信息
        date_str, time_str = extract_time_from_filename(hcfd_file_gz)
        if not date_str or not time_str:
            print(f"  警告: 无法从文件名提取时间信息: {hcfd_file_gz}")
            return 0
        
        # 8. 按250x250切片保存
        height, width = compressed_data_cube.shape[1], compressed_data_cube.shape[2]
        slice_count = 0
        
        for xindex in range(0, height - 2, 250):
            for yindex in range(0, width - 2, 250):
                x_end = min(xindex + 250, height)
                y_end = min(yindex + 250, width)
                
                # 检查是否有足够的格点
                if x_end - xindex < 250 or y_end - yindex < 250:
                    continue
                
                data_slice = compressed_data_cube[:, xindex:x_end, yindex:y_end]  # [4, 250, 250]
                
                # 检查数据有效性：4个通道都存在且无负值、NaN
                has_valid_data = True
                for c in range(4):
                    channel_data = data_slice[c, :, :]
                    # 至少要有一些有效数据点
                    valid_mask = (channel_data >= 0) & np.isfinite(channel_data)
                    if not np.any(valid_mask):
                        has_valid_data = False
                        break
                
                if not has_valid_data:
                    continue
                
                # 使用新的文件名格式: NewLabels_0.04deg_YYYYMMDD_HHMMZ_X###_Y###.npz
                slice_filename = f"NewLabels_0.04deg_{date_str}_{time_str}Z_X{xindex}_Y{yindex}.npz"
                slice_file_path = os.path.join(output_dir, slice_filename)
                
                np.savez_compressed(slice_file_path, data=data_slice)
                slice_count += 1
        
        return slice_count
        
    except Exception as e:
        print(f"  错误处理文件 {os.path.basename(hcfd_file_gz)}: {e}")
        return 0


def main():
    parser = argparse.ArgumentParser(description='CPU服务器：提取新标签并切片保存')
    parser.add_argument('--hcfd_dir', type=str, required=True,
                        help='HCFD nc文件根目录 (例如: /home/lujc/NJIAS_H/NJIAS_HCFD_v1/0.04Deg/2022)')
    parser.add_argument('--vza_file', type=str, required=True,
                        help='VZA nc文件路径 (例如: /home/lujc/NJIAS_HCFD_0.04deg_SatZenAngle.nc)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='输出目录')
    parser.add_argument('--month', type=int, choices=range(1, 13),
                        help='仅处理指定月份的数据 (1-12)')
    
    args = parser.parse_args()
    
    # 1. 加载VZA数据（卫星天顶角，无时间维度）
    print(f"加载VZA数据: {args.vza_file}")
    if not os.path.exists(args.vza_file):
        print(f"错误: VZA文件不存在: {args.vza_file}")
        return
    
    try:
        vza_ds = xr.open_dataset(args.vza_file)
        # 假设VZA变量名为 'SatZenAngle' 或 'VZA'
        if 'SatZenAngle' in vza_ds:
            vza_data = vza_ds['SatZenAngle'].values
        elif 'VZA' in vza_ds:
            vza_data = vza_ds['VZA'].values
        else:
            print(f"错误: VZA文件中未找到 'SatZenAngle' 或 'VZA' 变量")
            print(f"可用变量: {list(vza_ds.variables.keys())}")
            return
        
        print(f"  VZA数据维度: {vza_data.shape}")
    except Exception as e:
        print(f"错误: 无法加载VZA文件: {e}")
        return
    
    # 2. 查找所有HCFD文件
    if not os.path.exists(args.hcfd_dir):
        print(f"错误: HCFD目录不存在: {args.hcfd_dir}")
        return
    
    # 获取所有日期目录
    date_dirs = sorted([d for d in os.listdir(args.hcfd_dir)
                       if os.path.isdir(os.path.join(args.hcfd_dir, d))
                       and re.match(r'^\d{8}$', d)])
    
    # 如果指定了月份，则只处理该月份的数据
    if args.month:
        target_month = f"{args.month:02d}"
        date_dirs = [d for d in date_dirs if d[4:6] == target_month]
    
    if not date_dirs:
        print(f"错误: 在 {args.hcfd_dir} 中未找到符合条件的日期子目录")
        return
    
    print(f"发现 {len(date_dirs)} 个日期目录")
    
    # 3. 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 4. 处理所有文件
    total_files = 0
    total_slices = 0
    
    for date_dir in date_dirs:
        date_path = os.path.join(args.hcfd_dir, date_dir)
        hcfd_files = glob.glob(os.path.join(date_path, '*.nc.gz'))
        
        if not hcfd_files:
            continue
        
        print(f"\n处理日期 {date_dir}: 找到 {len(hcfd_files)} 个文件")
        
        for hcfd_file in tqdm(hcfd_files, desc=f"  {date_dir}", ncols=100):
            slice_count = process_hcfd_file(hcfd_file, vza_data, args.output_dir)
            if slice_count > 0:
                total_files += 1
                total_slices += slice_count
    
    print(f"\n✅ 处理完成!")
    print(f"  成功处理文件数: {total_files}")
    print(f"  生成切片数: {total_slices}")
    
    # 5. 保存标签名称信息
    info_file = os.path.join(args.output_dir, 'new_labels_info.txt')
    with open(info_file, 'w') as f:
        f.write("新标签通道信息:\n")
        f.write("0: CldHeight\n")
        f.write("1: CldTemperature\n")
        f.write("2: CldType\n")
        f.write("3: VZA\n")
    
    print(f"\n标签信息已保存到: {info_file}")


if __name__ == '__main__':
    main()
