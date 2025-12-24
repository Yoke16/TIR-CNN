from cProfile import label
import os
import glob
import gzip
import io
import xarray as xr
import numpy as np
import re
import argparse

def sunFix(gmt, tet, xlat, xlon):
    # 天文计算常数
    a1, a2, a3, a4, a5 = 0.000075, 0.001868, 0.032077, 0.014615, 0.04089
    b1, b2, b3, b4, b5, b6, b7 = 0.006918, 0.399912, 0.070257, 0.006758, 0.000907, 0.002697, 0.001480

    # 时间方程（校正真太阳时）
    et = (a1 + a2*np.cos(tet) - a3*np.sin(tet) - a4*np.cos(2*tet) - a5*np.sin(2*tet)) * 180.0 / np.pi / 15.0

    # 真太阳时
    tst = gmt + (xlon/15) + et

    # 时角（弧度）
    ah = (tst - 12) * 15.0 * np.pi / 180.0

    # 太阳赤纬（弧度）
    delta = b1 - b2*np.cos(tet) + b3*np.sin(tet) - b4*np.cos(2*tet) + b5*np.sin(2*tet) - b6*np.cos(3*tet) + b7*np.sin(3*tet)

    # 计算天顶角
    xlat_rad = xlat * np.pi / 180.0
    cos_zenith = np.sin(xlat_rad)*np.sin(delta) + np.cos(xlat_rad)*np.cos(delta)*np.cos(ah)

    # 防止浮点误差
    cos_zenith = np.clip(cos_zenith, -1.0, 1.0)

    # 返回角度制天顶角
    return np.degrees(np.arccos(cos_zenith))

def extract_time_from_filename(filename):
    """
    从文件名中提取时间信息（格式如 NJIAS_HObs_0.02deg_20220101_0700Z.nc.gz）
    返回年、月、日、小时、分钟
    """
    pattern = r'(\d{8})_(\d{4})Z'
    match = re.search(pattern, os.path.basename(filename))

    if not match:
        raise ValueError(f"无法从文件名中解析时间: {filename}")

    date_str = match.group(1)
    time_str = match.group(2)

    year = int(date_str[0:4])
    month = int(date_str[4:6])
    day = int(date_str[6:8])

    hour = int(time_str[0:2])
    minute = int(time_str[2:4])

    return year, month, day, hour, minute

def calculate_day_of_year(year, month, day):
    """计算年积日（一年中的第几天）"""
    days_in_month = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    day_of_year = sum(days_in_month[:month]) + day

    # 闰年处理
    if month > 2 and ((year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)):
        day_of_year += 1

    return day_of_year

def find_corresponding_label_file(X_nc_file_gz, Y_nc_file_dir):
    date_match = re.search(r'(\d{8})', os.path.basename(X_nc_file_gz))
    if not date_match:
        return None

    date_str = date_match.group(1)
    X_base_name = os.path.basename(X_nc_file_gz)
    Y_base_name = X_base_name.replace('HObs_0.02', 'HCFD_0.04')
    Y_file_path = os.path.join(Y_nc_file_dir, date_str, Y_base_name)

    if os.path.exists(Y_file_path):
        return Y_file_path
    else:
        Y_file_path_alt = os.path.join(Y_nc_file_dir, Y_base_name)
        if os.path.exists(Y_file_path_alt):
            return Y_file_path_alt
        else:
            return None

def compress_data(data, scale_factor=1000, dtype=np.float16):
    if data is not None:
        nan_mask = np.isnan(data)
        compressed_data = np.round(data * scale_factor) / scale_factor
        compressed_data = compressed_data.astype(dtype)
        compressed_data[nan_mask] = np.nan
        return compressed_data
    return data

def downsample_002_to_004(data_array):
    if len(data_array.shape) >= 2:
        return data_array[..., ::2, ::2]
    return data_array

def nc2npy(X_nc_file_dir, Y_nc_file_dir, npy_file_dir):
    os.makedirs(npy_file_dir, exist_ok=True)
    X_nc_files_gz = glob.glob(os.path.join(X_nc_file_dir, '*.nc.gz'))

    saved_lat_lon_features = False
    feature_names_list = []

    # 更新标签列表，添加新标签: CldHeight, CldTemperature, CldType, VZA
    label_list = ['CldPressure', 'CldPhase', 'DCOMP35_COD', 'DCOMP35_CPS', 
                  'DCOMP36_CPS', 'DCOMP37_CPS', 'CldHeight', 'CldTemperature', 
                  'CldType', 'VZA']

    for X_nc_file_gz in X_nc_files_gz:
        try:
            Y_nc_file_gz = find_corresponding_label_file(X_nc_file_gz, Y_nc_file_dir)
            if not Y_nc_file_gz:
                continue

            with open(X_nc_file_gz, 'rb') as f:
                X_gz_data = f.read()
            with open(Y_nc_file_gz, 'rb') as f:
                Y_gz_data = f.read()

            with gzip.GzipFile(fileobj=io.BytesIO(X_gz_data)) as gz_file:
                netcdf_bytes = gz_file.read()
            X_netcdf_file = io.BytesIO(netcdf_bytes)

            with gzip.GzipFile(fileobj=io.BytesIO(Y_gz_data)) as gz_file:
                netcdf_bytes = gz_file.read()
            Y_netcdf_file = io.BytesIO(netcdf_bytes)

            X_ds = xr.open_dataset(X_netcdf_file)
            Y_ds = xr.open_dataset(Y_netcdf_file)

            feature_list = ['tbb_08', 'tbb_09', 'tbb_10', 'tbb_11', 'tbb_13', 'tbb_14', 'tbb_15', 'tbb_16']
            available_features = [feat for feat in feature_list if feat in X_ds]
            if not available_features:
                continue

            available_labels = [label for label in label_list if label in Y_ds]
            if not available_labels:
                continue

            features = X_ds[available_features].to_array().values
            features_downsampled = downsample_002_to_004(features)

            target_lat = Y_ds['latitude'].values
            target_lon = Y_ds['longitude'].values

            year, month, day, hour, minute = extract_time_from_filename(X_nc_file_gz)
            gmt = hour + minute / 60.0
            day_of_year = calculate_day_of_year(year, month, day)
            tet = 2.0 * np.pi * (day_of_year - 1) / 365.0
            xlon, xlat = np.meshgrid(target_lon, target_lat)

            solar_zenith = sunFix(gmt, tet, xlat, xlon)

            # 构建完整的数据立方体：features + solar_zenith + labels
            data_cube = features_downsampled
            
            # 添加太阳天顶角
            data_cube = np.concatenate([data_cube, solar_zenith[np.newaxis, :, :]], axis=0)
            
            # 添加标签数据
            for label_name in available_labels:
                label_data = Y_ds[label_name].values
                # 如果标签是0.04度，直接使用；否则降采样
                if len(label_data.shape) == 2:
                    # 检查是否需要降采样
                    if label_data.shape != features_downsampled.shape[1:]:
                        label_data = downsample_002_to_004(label_data)
                    data_cube = np.concatenate([data_cube, label_data[np.newaxis, :, :]], axis=0)

            # 压缩数据，使用float16类型
            SCALE_FACTOR = 1000.0
            DATA_TYPE = np.float16
            compressed_data_cube = compress_data(data_cube, SCALE_FACTOR, DATA_TYPE)

            filename = os.path.basename(X_nc_file_gz)[:-6]
            
            # 按250x250切片保存
            height, width = compressed_data_cube.shape[1], compressed_data_cube.shape[2]
            
            # 提取时间信息用于新文件名
            time_str = f"{year}{month:02d}{day:02d}_{hour:02d}{minute:02d}Z"
            
            # 获取SZA和CldType的索引（用于筛选）
            # 数据结构: [8 TIR] + [SZA] + [10 labels]
            # SZA索引: 8
            # CldType索引: 8 (SZA) + 标签索引
            # 标签顺序: CldPressure, CldPhase, DCOMP35_COD, DCOMP35_CPS, DCOMP36_CPS, DCOMP37_CPS, CldHeight, CldTemperature, CldType, VZA
            sza_channel = 8
            # CldType在标签列表中是第8个（0-based: 索引8），所以在总通道中是 8 + 1 + 8 = 17
            cldtype_channel = 8 + 1 + label_list.index('CldType')
            
            for xindex in range(0, height-2, 250):
                for yindex in range(0, width-2, 250):
                    x_end = min(xindex + 250, height)
                    y_end = min(yindex + 250, width)
                    
                    # 检查是否有足够的格点
                    if x_end - xindex < 250 or y_end - yindex < 250:
                        continue
                    
                    data_slice = compressed_data_cube[:, xindex:x_end, yindex:y_end]
                    
                    # 筛选条件1: SZA < 60度（白天数据）
                    sza_slice = data_slice[sza_channel, :, :]
                    has_daytime = np.any((sza_slice < 60.0) & (sza_slice >= 0.0) & np.isfinite(sza_slice))
                    
                    if not has_daytime:
                        continue  # 跳过完全没有白天数据的切片
                    
                    # 筛选条件2: CldType > 1（云区域，值为2-9）
                    if cldtype_channel < data_slice.shape[0]:
                        cldtype_slice = data_slice[cldtype_channel, :, :]
                        has_cloud = np.any((cldtype_slice > 1) & (cldtype_slice <= 9) & np.isfinite(cldtype_slice))
                        
                        if not has_cloud:
                            continue  # 跳过完全没有云的切片
                    
                    # 筛选条件3: 15个通道（8 TIR + SZA + 6 原始标签）都存在且无负值、NaN、0值
                    first_15_channels = data_slice[:15, :, :]
                    has_valid_data = np.any(
                        np.all(first_15_channels > 0, axis=0) & 
                        np.all(np.isfinite(first_15_channels), axis=0)
                    )
                    
                    if not has_valid_data:
                        continue  # 跳过没有有效数据的切片
                    
                    # 使用新的文件名格式
                    slice_filename = f"NJIAS_ForAI_0.04deg_{time_str}_X{xindex}_Y{yindex}.npz"
                    slice_file_path = os.path.join(npy_file_dir, slice_filename)
                    
                    np.savez_compressed(slice_file_path, data=data_slice)

            if not saved_lat_lon_features:
                feature_names_list = available_features + ['solar_zenith_angle']
                feature_names_list.extend(available_labels)
                
                saved_lat_lon_features = True
                info_file_path = os.path.join(npy_file_dir, 'feature_names.txt')
                with open(info_file_path, 'w') as f:
                    for i, name in enumerate(feature_names_list):
                        f.write(f"{i}: {name}\n")

        except Exception as e:
            continue

def stat_mean_std(npy_file_dir):
    npz_file_list = glob.glob(os.path.join(npy_file_dir, '*.npz'))

    if not npz_file_list:
        return

    all_data = []
    sample_fraction = 0.1
    first_shape = None

    for npz_file in npz_file_list:
        try:
            data = np.load(npz_file)
            if 'data' not in data:
                continue

            features_data = data['data']

            if first_shape is None:
                first_shape = features_data.shape
            elif features_data.shape != first_shape:
                continue

            total_elements = features_data.shape[1] * features_data.shape[2]
            sample_size = max(1, int(total_elements * sample_fraction))
            flat_indices = np.random.choice(total_elements, size=sample_size, replace=False)
            row_indices = flat_indices // features_data.shape[2]
            col_indices = flat_indices % features_data.shape[2]

            sampled_data_list = []
            for feat_idx in range(features_data.shape[0]):
                sampled_feat = features_data[feat_idx, row_indices, col_indices]
                sampled_data_list.append(sampled_feat[None, :])

            sampled_data = np.concatenate(sampled_data_list, axis=0)
            all_data.append(sampled_data)

        except Exception as e:
            continue

    if not all_data:
        return

    combined_data = np.concatenate(all_data, axis=1)

    mean = np.mean(combined_data, axis=1)
    std = np.std(combined_data, axis=1)

    mean_file = os.path.join(npy_file_dir, 'mean.npy')
    std_file = os.path.join(npy_file_dir, 'std.npy')
    np.save(mean_file, mean)
    np.save(std_file, std)

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Himawari AHI数据预处理 - 处理2023全年数据')
    parser.add_argument('--base_nc_dir', type=str, 
                        default='/home/lujc/NJIAS_H/NJIAS_HObs_v0/0.02Deg/2023',
                        help='nc文件的根目录，包含按日期划分的子目录 (默认: 2023年根目录)')
    parser.add_argument('--Y_nc_dir', type=str, 
                        default='/home/lujc/NJIAS_H/NJIAS_HCFD_v1/0.04Deg/2023',
                        help='标签nc文件的根目录，包含按日期划分的子目录 (默认: 2023年根目录)')
    parser.add_argument('--npy_dir', type=str, 
                        default='/array1/satellite/lujc/npy_dir_2023', 
                        help='npy文件输出目录 (默认: /array1/satellite/lujc/npy_dir_2023)')
    parser.add_argument('--month', type=int, choices=range(1,13), 
                        help='仅处理指定月份的数据 (1-12)')

    args = parser.parse_args()

    base_nc_dir = args.base_nc_dir
    main_npy_dir = args.npy_dir
    Y_nc_dir = args.Y_nc_dir

    if not os.path.exists(base_nc_dir):
        print(f"错误：根目录 {base_nc_dir} 不存在！")
        exit(1)

    # 获取所有日期目录
    date_dirs = sorted([d for d in os.listdir(base_nc_dir) 
                        if os.path.isdir(os.path.join(base_nc_dir, d)) 
                        and re.match(r'^\d{8}$', d)])

    # 如果指定了月份，则只处理该月份的数据
    if args.month:
        target_month = f"{args.month:02d}"
        date_dirs = [d for d in date_dirs if d[4:6] == target_month]
    
    if not date_dirs:
        print(f"错误：在 {base_nc_dir} 中未找到符合条件的日期子目录")
        exit(1)

    print(f"发现 {len(date_dirs)} 个日期目录，开始处理...")

    os.makedirs(main_npy_dir, exist_ok=True)

    total_files = 0

    for date_dir in date_dirs:
        # 从日期目录名中提取月份
        month_str = date_dir[4:6]  # 日期格式为YYYYMMDD，提取第5-6位作为月份
        
        # 创建月份子目录
        month_npy_dir = os.path.join(main_npy_dir, f"month_{month_str}")
        os.makedirs(month_npy_dir, exist_ok=True)
        
        nc_file_dir = os.path.join(base_nc_dir, date_dir)
        print(f"\n处理日期目录: {nc_file_dir}")

        nc_files_gz = glob.glob(os.path.join(nc_file_dir, '*.nc.gz'))

        if not nc_files_gz:
            print(f"  警告：该目录中没有找到 .nc.gz 文件，跳过。")
            continue

        print(f"  找到 {len(nc_files_gz)} 个文件，开始转换...")
        nc2npy(nc_file_dir, Y_nc_dir, month_npy_dir)  # 使用月份子目录
        total_files += len(nc_files_gz)

    print(f"\n✅ 数据处理完成！共处理 {total_files} 个文件。")

    # 计算每个月份的统计量
    if args.month:
        month_str = f"{args.month:02d}"
        month_npy_dir = os.path.join(main_npy_dir, f"month_{month_str}")
        print(f"正在计算 {month_str} 月的统计量...")
        stat_mean_std(month_npy_dir)
        print(f"{month_str} 月的预处理和统计计算完成！")
    else:
        # 处理完整年数据，计算每个月的统计量
        for month in range(1, 13):
            month_str = f"{month:02d}"
            month_npy_dir = os.path.join(main_npy_dir, f"month_{month_str}")
            if os.path.exists(month_npy_dir):
                print(f"正在计算 {month_str} 月的统计量...")
                stat_mean_std(month_npy_dir)
        print("所有月份的预处理和统计计算完成！")
