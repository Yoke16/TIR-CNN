# 云反演模型重构实施文档

## 概述

本文档详细说明了Himawari-8云反演模型的重构实现，包括新增标签、两阶段预测、数据筛选等功能。

## 实施步骤

### 1. 数据准备阶段

#### 1.1 两阶段标签拼接流程

**方案说明**: 由于需要在CPU服务器和GPU服务器之间传输数据，我们采用两阶段方案

**阶段1: CPU服务器 - 提取新标签** (`src/cpu_extract_new_labels.py`)

**目的**: 从HCFD nc.gz文件和VZA nc文件中提取4个新标签，进行250x250切片，保存为独立的npz文件

**数据源**:
- CldHeight, CldTemperature, CldType: `/home/lujc/NJIAS_H/NJIAS_HCFD_v1/0.04Deg/YYYY/YYYYMMDD/NJIAS_HCFD_0.04deg_YYYYMMDD_HHMMZ.nc.gz`
- VZA (卫星天顶角): `/home/lujc/NJIAS_HCFD_0.04deg_SatZenAngle.nc` (无时间维度)

**使用方法**:
```bash
python src/cpu_extract_new_labels.py \
    --hcfd_dir /home/lujc/NJIAS_H/NJIAS_HCFD_v1/0.04Deg/2022 \
    --vza_file /home/lujc/NJIAS_HCFD_0.04deg_SatZenAngle.nc \
    --output_dir /path/to/new_labels_output \
    --month 7
```

**输出**:
- 4通道npz文件: [CldHeight, CldTemperature, CldType, VZA]
- 文件命名: `NewLabels_0.04deg_YYYYMMDD_HHMMZ_X###_Y###.npz`
- 数据格式: float16压缩
- 切片大小: 250x250

**阶段2: GPU服务器 - 合并标签** (`src/gpu_merge_labels.py`)

**目的**: 将CPU服务器生成的新标签npz文件与现有15通道npz文件合并

**输入**:
- 现有15通道npz: [8 TIR] + [SZA] + [6 labels]
- 新标签4通道npz: [CldHeight, CldTemperature, CldType, VZA]

**使用方法**:
```bash
python src/gpu_merge_labels.py \
    --existing_npz_dir /path/to/existing/15ch/npz \
    --new_labels_dir /path/to/new_labels \
    --output_dir /path/to/output/19ch/npz
```

**输出**:
- 19通道npz: [8 TIR] + [SZA] + [10 labels]
- 文件命名: 保持与原15通道文件相同
- 数据有效性: 自动检查所有19个通道是否存在且无负值、NaN

**数据结构**: 19通道 = [8 TIR] + [SZA] + [10 labels]
- 0-7: tbb_08 ~ tbb_16 (8个TIR通道)
- 8: SolarZenithAngle
- 9-18: CldPressure, CldPhase, DCOMP35_COD, DCOMP35_CPS, DCOMP36_CPS, DCOMP37_CPS, CldHeight, CldTemperature, CldType, VZA

#### 1.2 计算新标签统计量 (`src/compute_new_stats.py`)

**目的**: 计算VZA和7个云参数的mean/std用于归一化

**使用方法**:
```bash
python src/compute_new_stats.py \
    --npz_dirs /path/to/train/data1 /path/to/train/data2 \
    --output_dir ./stats \
    --chunk_size 200
```

**统计的8个通道**:
1. VZA
2. DCOMP35_CPS
3. DCOMP36_CPS
4. DCOMP37_CPS
5. DCOMP35_COD
6. CldPressure
7. CldHeight
8. CldTemperature

**筛选条件**:
- SZA < 60度 (白天数据)
- CldType > 1 (云区域，值为2-9)
- 所有通道无负值、NaN

**输出文件**:
- `new_labels_mean.npy`: [8] 均值
- `new_labels_std.npy`: [8] 标准差

### 2. 数据预处理阶段 (`src/preprocess.py`)

**主要更新**:

1. **标签列表扩展**:
   ```python
   label_list = ['CldPressure', 'CldPhase', 'DCOMP35_COD', 'DCOMP35_CPS', 
                 'DCOMP36_CPS', 'DCOMP37_CPS', 'CldHeight', 'CldTemperature', 
                 'CldType', 'VZA']
   ```

2. **数据筛选条件**:
   - SZA < 60度 (白天数据筛选)
   - CldType > 1 (云区域筛选，值为2-9)
   - 15个通道（8 TIR + SZA + 6原始标签）都存在且无负值、NaN、0值

3. **输出数据结构**: 19通道npz文件

### 3. 数据加载阶段 (`src/dataset.py`)

**HimawariCloudDataset 类更新**:

**输入参数**:
- `input_channels`: 16 = [8 TIR] + [VZA] + [7 cloud params]
- `num_total_labels`: 10个标签
- `reg_label_indices`: [3, 4, 5, 2, 0, 6, 7] (7个回归标签索引)
- `cldtype_label_index`: 8 (CldType在标签中的索引)

**返回值**:
- `input_tensor`: [16, H, W] - 模型输入（已归一化）
- `cldtype_tensor`: [H, W] - CldType分类目标（0为背景/无效像元）
- `reg_target_tensor`: [7, H, W] - 7个云参数回归目标（已归一化）

**筛选逻辑**:
1. CldType > 1（云区域）
2. SZA < 60度（白天数据）
3. 所有TIR、VZA、回归标签无负值、NaN

**归一化策略**:
- **输入**: 16通道分别归一化（使用input_mean/std）
  - 前8个：TIR通道
  - 第9个：VZA
  - 后7个：7个云参数
- **输出**: 7个回归目标使用target_mean/std归一化

### 4. 模型架构 (`src/model.py`)

**TIR_CNN 类更新**:

**输入输出**:
- 输入: [B, 16, H, W]
- 输出1: cldtype_logits [B, 10, H, W] - CldType分类logits
- 输出2: regression_output [B, 7, H, W] - 7个云参数回归预测

**网络结构**:
- 共享编码器（Encoder）
- 共享解码器（Decoder with Attention Gates）
- 两个独立输出头：
  - `classification_head`: Conv2d(base_channels, 10, kernel_size=1)
  - `regression_head`: Conv2d(base_channels, 7, kernel_size=1)

**关键特性**:
- 使用CBAM注意力机制
- Attention Gate用于解码器跳跃连接
- Kaiming初始化

### 5. 损失函数 (`src/loss.py`)

**TwoStageLoss 类**:

**组成**:
1. **分类损失**: CrossEntropyLoss (ignore_index=0)
   - 用于CldType分类
   - 忽略背景类（0）

2. **回归损失**: MultiTaskLogCoshLoss
   - 7个云参数的多任务回归
   - 只计算CldType > 1的云区域
   - 支持NaN掩膜

**前向传播**:
```python
loss, cls_loss, reg_loss, per_channel = criterion(
    cldtype_logits,  # [B, 10, H, W]
    reg_preds,       # [B, 7, H, W]
    cldtype_target,  # [B, H, W]
    reg_targets      # [B, 7, H, W]
)
```

**返回值**:
- `loss`: 总损失 = cls_weight * cls_loss + reg_weight * reg_loss
- `cls_loss`: 分类损失
- `reg_loss`: 回归损失
- `per_channel`: [7] 每个回归通道的损失

### 6. 训练流程 (`src/train.py`)

**主要更新**:

1. **统计量计算** (`load_or_compute_stats`):
   - 计算16个输入通道的mean/std
   - 计算7个回归目标的mean/std
   - 使用在线累加方式，内存高效
   - 筛选条件：SZA < 60度，CldType > 1

2. **数据集初始化**:
   ```python
   train_dataset = HimawariCloudDataset(
       file_list=train_files,
       input_mean=input_mean,        # [16]
       input_std=input_std,          # [16]
       target_mean=target_mean,      # [7]
       target_std=target_std,        # [7]
       input_channels=16,
       num_total_labels=10,
       reg_label_indices=[3, 4, 5, 2, 0, 6, 7],
       sza_index=8,
       cldtype_label_index=8,
   )
   ```

3. **训练循环**:
   ```python
   for inputs, cldtype_targets, reg_targets in train_loader:
       cldtype_logits, reg_preds = model(inputs)
       loss, cls_loss, reg_loss, per_label = criterion(
           cldtype_logits, reg_preds, 
           cldtype_targets, reg_targets
       )
       loss.backward()
       optimizer.step()
   ```

4. **TensorBoard日志**:
   - Loss/train_total
   - Loss/train_cls
   - Loss/train_reg
   - Loss/train_{label_name} (7个)
   - Loss/val_total

### 7. 评估流程 (`src/evaluate_2024.py`)

**评估指标**:

1. **CldType分类指标**:
   - 整体准确率
   - 每类精确率、召回率、F1
   - 混淆矩阵（10x10）

2. **云参数回归指标** (物理量空间):
   - RMSE (Root Mean Square Error)
   - MBE (Mean Bias Error)
   - R (Correlation Coefficient)

**输出示例**:
```
=== 2024 年 CldType 分类评估结果 ===
整体准确率: 0.8523 (12345678/14567890)

每类别性能:
类别     精确率     召回率       F1     样本数
--------------------------------------------------
Type 2   0.8234  0.8567  0.8398    1234567
Type 3   0.7891  0.8123  0.8005     987654
...

=== 2024 年云参数回归评估结果（物理量空间）===
DCOMP35_CPS  | N=  12345678 | RMSE=   2.345 | MBE=   0.123 | R= 0.923
DCOMP36_CPS  | N=  12345678 | RMSE=   2.456 | MBE=  -0.234 | R= 0.915
...
```

### 8. 配置文件 (`configs/base/config.yaml`)

**模型配置更新**:
```yaml
model:
  input_channels: 16        # [8 TIR] + [VZA] + [7 cloud params]
  output_channels_cls: 10   # CldType分类，10类 (0-9)
  output_channels_reg: 7    # 7个云参数回归
  base_channels: 24
```

## 完整工作流程

### 背景说明

由于数据处理涉及CPU服务器和GPU服务器之间的数据传输，我们设计了两阶段的标签拼接流程：

1. **CPU服务器阶段**: 从原始nc.gz文件中提取新标签，进行切片，生成独立的新标签npz文件
2. **数据传输阶段**: 通过硬盘将新标签npz文件从CPU服务器传输到GPU服务器
3. **GPU服务器阶段**: 将新标签npz文件与现有的15通道npz文件合并，生成19通道npz文件

### 数据源位置

- **CldHeight, CldTemperature, CldType**: `/home/lujc/NJIAS_H/NJIAS_HCFD_v1/0.04Deg/YYYY/YYYYMMDD/NJIAS_HCFD_0.04deg_YYYYMMDD_HHMMZ.nc.gz`
- **VZA (卫星天顶角)**: `/home/lujc/NJIAS_HCFD_0.04deg_SatZenAngle.nc` (无时间维度，所有时刻共用)

### 工作流程1: 使用现有15通道npz文件（推荐）

**适用场景**: 已有15通道npz文件，需要添加4个新标签

```bash
# 步骤1: CPU服务器 - 提取新标签并切片
python src/cpu_extract_new_labels.py \
    --hcfd_dir /home/lujc/NJIAS_H/NJIAS_HCFD_v1/0.04Deg/2022 \
    --vza_file /home/lujc/NJIAS_HCFD_0.04deg_SatZenAngle.nc \
    --output_dir /path/to/new_labels_output \
    --month 7

# 步骤2: 将生成的新标签npz文件传输到GPU服务器（通过硬盘或网络）
# 例如: scp /path/to/new_labels_output/*.npz gpu_server:/path/to/new_labels/

# 步骤3: GPU服务器 - 合并新标签与现有15通道数据
python src/gpu_merge_labels.py \
    --existing_npz_dir /path/to/existing/15ch/npz \
    --new_labels_dir /path/to/new_labels \
    --output_dir /path/to/output/19ch/npz

# 步骤4: 计算统计量（会自动在训练时计算，或手动计算）
python src/compute_new_stats.py \
    --npz_dirs /path/to/output/19ch/npz \
    --output_dir ./stats

# 步骤5: 训练模型
python src/train.py

# 步骤6: 评估模型
python src/evaluate_2024.py
```

### 工作流程2: 从原始数据重新预处理（完整流程）

```bash
# 1. 预处理原始nc.gz文件
python src/preprocess.py --month 7

# 2. 计算统计量（会自动在训练时计算，或手动计算）
python src/compute_new_stats.py --npz_dirs /path/to/data --output_dir ./stats

# 3. 训练模型
python src/train.py

# 4. 评估模型
python src/evaluate_2024.py
```

## 关键技术细节

### 数据筛选条件总结

| 阶段 | SZA条件 | 云筛选条件 | 通道条件 |
|------|---------|-----------|----------|
| 预处理（切片保存） | < 60度 | CldType > 1 | 15通道无负值/NaN/0 |
| 统计量计算 | < 60度 | CldType > 1 | 19通道无负值/NaN |
| Dataset加载 | < 60度 | CldType > 1 | TIR/VZA/回归标签无负值/NaN |

### 通道索引映射

**NPZ文件通道（19通道）**:
- 0-7: TIR (tbb_08 ~ tbb_16)
- 8: SolarZenithAngle
- 9-18: 10个标签

**10个标签的索引**:
- 0: CldPressure
- 1: CldPhase (不再使用)
- 2: DCOMP35_COD
- 3: DCOMP35_CPS
- 4: DCOMP36_CPS
- 5: DCOMP37_CPS
- 6: CldHeight
- 7: CldTemperature
- 8: CldType
- 9: VZA

**模型输入（16通道）**:
- 0-7: TIR (归一化)
- 8: VZA (归一化)
- 9-15: 7个云参数 (归一化，顺序: DCOMP35_CPS, DCOMP36_CPS, DCOMP37_CPS, DCOMP35_COD, CldPressure, CldHeight, CldTemperature)

### 内存优化策略

1. **float16存储**: 所有npz文件使用float16类型
2. **在线统计**: 统计量计算采用在线累加，不需要加载全部数据
3. **批处理**: 数据加载和训练使用DataLoader批处理
4. **显存优化**: 使用pin_memory和non_blocking加速GPU传输

## 常见问题

### Q1: 如何验证数据是否正确拼接？

查看feature_names.txt文件，应该包含19个特征名称。使用numpy加载一个npz文件检查shape：
```python
import numpy as np
data = np.load('sample.npz')['data']
print(data.shape)  # 应该是 (19, 250, 250)
```

### Q2: 统计量文件存放在哪里？

默认存放在`./stats/`目录：
- `input_mean.npy`: 16个输入通道的均值
- `input_std.npy`: 16个输入通道的标准差
- `target_mean.npy`: 7个回归目标的均值
- `target_std.npy`: 7个回归目标的标准差

### Q3: 如何处理旧格式（15通道）文件？

旧格式文件会被自动过滤。如果需要使用，请先运行`append_new_labels.py`将其转换为19通道格式。

### Q4: 训练时显存不足怎么办？

1. 减小`batch_size`（在config.yaml中）
2. 减小`base_channels`（在config.yaml中）
3. 减少`num_workers`（在config.yaml中）

### Q5: 如何调整分类和回归损失的权重？

修改`TwoStageLoss`的初始化参数：
```python
criterion = TwoStageLoss(
    num_outputs=7,
    cls_weight=1.0,    # 分类损失权重
    reg_weight=1.0,    # 回归损失权重
)
```

## 性能优化建议

1. **预处理阶段**: 使用多进程并行处理多个日期的数据
2. **训练阶段**: 增加`num_workers`以加速数据加载（注意内存）
3. **评估阶段**: 可以使用更大的`batch_size`因为不需要梯度计算
4. **统计量计算**: 增加`chunk_size`以减少打印频率

## 后续扩展建议

1. **多GPU训练**: 使用DistributedDataParallel进行分布式训练
2. **混合精度训练**: 使用torch.cuda.amp加速训练
3. **数据增强**: 添加随机翻转、旋转等增强策略
4. **损失函数优化**: 尝试Focal Loss处理类别不平衡
5. **模型集成**: 训练多个模型进行集成预测
6. **可视化工具**: 添加预测结果可视化脚本

## 版本信息

- **实施日期**: 2025-12-16
- **Python版本**: 3.8+
- **PyTorch版本**: 1.10+
- **主要依赖**: torch, numpy, xarray, tqdm, PyYAML

## 联系方式

如有问题或建议，请联系项目维护者或在GitHub上提出Issue。
