# 两阶段数据准备工作流程

## 概述

本文档说明如何使用新的两阶段脚本在CPU服务器和GPU服务器之间准备训练数据。

## 工作流程图

```
CPU服务器:
  HCFD nc.gz文件 (CldHeight, CldTemperature, CldType)
         +
  VZA nc文件 (卫星天顶角)
         ↓
  cpu_extract_new_labels.py
         ↓
  4通道npz文件: NewLabels_*.npz
         ↓
  [通过硬盘传输]
         ↓
GPU服务器:
  现有15通道npz文件
         +
  新标签4通道npz文件
         ↓
  gpu_merge_labels.py
         ↓
  最终19通道npz文件
         ↓
  训练和评估
```

## 详细步骤

### 步骤1: CPU服务器 - 提取新标签

在CPU服务器上运行:

```bash
python src/cpu_extract_new_labels.py \
    --hcfd_dir /home/lujc/NJIAS_H/NJIAS_HCFD_v1/0.04Deg/2022 \
    --vza_file /home/lujc/NJIAS_HCFD_0.04deg_SatZenAngle.nc \
    --output_dir /path/to/new_labels_output \
    --month 7
```

**参数说明**:
- `--hcfd_dir`: HCFD数据根目录（包含日期子目录的年份目录）
- `--vza_file`: VZA nc文件完整路径
- `--output_dir`: 输出目录（保存4通道npz文件）
- `--month`: （可选）只处理指定月份，1-12

**输出**:
- 在output_dir中生成多个4通道npz文件
- 文件名格式: `NewLabels_0.04deg_YYYYMMDD_HHMMZ_X###_Y###.npz`
- 每个文件包含: [CldHeight, CldTemperature, CldType, VZA]
- 数据格式: float16压缩，[4, 250, 250]

**处理的数据源**:
- CldHeight, CldTemperature, CldType 来自: `/home/lujc/NJIAS_H/NJIAS_HCFD_v1/0.04Deg/YYYY/YYYYMMDD/NJIAS_HCFD_0.04deg_YYYYMMDD_HHMMZ.nc.gz`
- VZA 来自: `/home/lujc/NJIAS_HCFD_0.04deg_SatZenAngle.nc` (所有时刻共用同一个文件)

### 步骤2: 数据传输

将CPU服务器生成的新标签npz文件传输到GPU服务器:

```bash
# 方法1: 直接复制到硬盘
cp /path/to/new_labels_output/*.npz /mnt/external_drive/

# 方法2: 打包后传输
tar -czf new_labels.tar.gz /path/to/new_labels_output/*.npz
# 然后将tar.gz文件通过硬盘传输

# 在GPU服务器上解压
tar -xzf new_labels.tar.gz -C /path/to/destination/
```

### 步骤3: GPU服务器 - 合并标签

在GPU服务器上运行:

```bash
python src/gpu_merge_labels.py \
    --existing_npz_dir /path/to/existing/15ch/npz \
    --new_labels_dir /path/to/new_labels \
    --output_dir /path/to/output/19ch/npz
```

**参数说明**:
- `--existing_npz_dir`: 现有15通道npz文件目录（可以包含子目录）
- `--new_labels_dir`: 新标签4通道npz文件目录（从CPU服务器传输过来）
- `--output_dir`: 输出目录（保存合并后的19通道npz文件）

**匹配逻辑**:
- 脚本自动根据文件名中的日期、时间、位置信息匹配对应的文件
- 15通道文件名: `NJIAS_ForAI_0.04deg_YYYYMMDD_HHMMZ_X###_Y###.npz`
- 4通道文件名: `NewLabels_0.04deg_YYYYMMDD_HHMMZ_X###_Y###.npz`
- 只有完全匹配的文件对才会被合并

**输出**:
- 19通道npz文件，文件名与原15通道文件相同
- 数据结构: [19, 250, 250]
  - 0-7: TIR通道 (tbb_08 ~ tbb_16)
  - 8: SolarZenithAngle
  - 9-14: 原有标签 (CldPressure, CldPhase, DCOMP35_COD, DCOMP35_CPS, DCOMP36_CPS, DCOMP37_CPS)
  - 15-18: 新标签 (CldHeight, CldTemperature, CldType, VZA)

### 步骤4: 后续处理

合并完成后，可以继续进行：

```bash
# 1. 计算统计量
python src/compute_new_stats.py \
    --npz_dirs /path/to/output/19ch/npz \
    --output_dir ./stats

# 2. 训练模型
python src/train.py

# 3. 评估模型
python src/evaluate_2024.py
```

## 数据验证

### 有效性检查

**cpu_extract_new_labels.py**:
- 检查每个切片至少有一些有效数据点（非负、有限值）
- 跳过完全无效的切片

**gpu_merge_labels.py**:
- 验证15通道和4通道文件的空间维度匹配
- 合并后生成19通道文件
- 不强制要求所有像元都有效（训练时会进行像元级筛选）

**dataset.py** (训练时):
- 加载时检查每个通道至少有部分有效数据
- 像元级筛选: SZA < 60度, CldType > 1, 无负值/NaN
- 无效像元设为背景类（分类）或NaN（回归）

## 注意事项

1. **VZA数据**: VZA文件没有时间维度，所有时刻使用同一个VZA数据（卫星天顶角是固定的）

2. **文件匹配**: 确保15通道npz文件和新标签npz文件的时间、位置信息完全一致才能成功匹配

3. **存储空间**: 
   - 4通道npz约为15通道的1/4大小
   - 19通道npz约为15通道的1.27倍大小
   - 建议预留足够的磁盘空间

4. **处理时间**: 
   - CPU服务器提取: 取决于nc.gz文件数量，通常几小时到一天
   - GPU服务器合并: 相对较快，通常几分钟到几十分钟

5. **错误处理**: 
   - 如果某些文件处理失败，脚本会继续处理其他文件
   - 查看输出日志了解处理统计信息

## 常见问题

### Q1: VZA文件中的变量名是什么？

脚本会自动查找 `SatZenAngle` 或 `VZA` 变量。如果变量名不同，脚本会列出所有可用变量供参考。

### Q2: 如何处理多个年份的数据？

对每个年份分别运行cpu_extract_new_labels.py:

```bash
for year in 2022 2023 2024; do
    python src/cpu_extract_new_labels.py \
        --hcfd_dir /home/lujc/NJIAS_H/NJIAS_HCFD_v1/0.04Deg/$year \
        --vza_file /home/lujc/NJIAS_HCFD_0.04deg_SatZenAngle.nc \
        --output_dir /path/to/new_labels_$year
done
```

### Q3: 合并时如果找不到匹配的文件怎么办？

检查：
1. 文件名格式是否正确
2. 时间和位置信息是否一致
3. 15通道文件是否在指定目录或子目录中

脚本会输出匹配成功的文件对数量，如果为0则需要检查上述问题。

### Q4: 如何验证生成的19通道文件是否正确？

```python
import numpy as np

# 加载文件
data = np.load('output_file.npz')['data']

# 检查维度
print(f"Shape: {data.shape}")  # 应该是 (19, 250, 250)

# 检查每个通道
for i in range(19):
    channel = data[i]
    valid_count = np.sum((channel >= 0) & np.isfinite(channel))
    print(f"Channel {i}: {valid_count} valid pixels")
```

## 技术支持

如有问题，请检查：
1. 数据路径是否正确
2. 文件权限是否足够
3. 磁盘空间是否充足
4. Python环境是否安装了所需依赖（numpy, xarray, tqdm）
