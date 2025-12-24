import numpy as np
import torch
from torch.utils.data import Dataset


class HimawariCloudDataset(Dataset):
    """
    从 0.04° 网格 .npz 在线加载样本（适用于两阶段云反演模型）

    - 假定每个文件是一个 3D 数组 [C, H, W]
    - C >= 19 时才会被送进这个 Dataset：
        前 8 通道: 8个TIR通道 (tbb_08 ~ tbb_16)
        第 9 通道: SolarZenithAngle (仅用于筛选，不作为输入)
        后 10 通道: 标签
          0: CldPressure
          1: CldPhase (不再使用)
          2: DCOMP35_COD
          3: DCOMP35_CPS
          4: DCOMP36_CPS
          5: DCOMP37_CPS
          6: CldHeight
          7: CldTemperature
          8: CldType (新的分类目标)
          9: VZA
    
    - 模型输入: 16通道 = [8 TIR] + [VZA] + [7 cloud params]
    - 分类目标: CldType (10类: 0-9)
    - 回归目标: 7个云参数 (DCOMP35_CPS, DCOMP36_CPS, DCOMP37_CPS, DCOMP35_COD, CldPressure, CldHeight, CldTemperature)
    
    - 筛选条件:
        * SZA < 60度（白天数据）
        * CldType > 1（云区域，值为2-9）
        * 所有15个通道（8 TIR + SZA + 6原始标签）都存在且无负值、NaN、0值
    """

    def __init__(
        self,
        file_list,
        input_mean,
        input_std,
        target_mean,
        target_std,
        input_channels: int = 16,  # [8 TIR] + [VZA] + [7 cloud params]
        num_total_labels: int = 10,  # 10个标签
        sza_index: int = 8,
        cldtype_label_index: int = 8,  # CldType在标签中的索引
        reg_label_indices=None,  # 7个回归标签在10个标签中的索引
    ):
        super().__init__()

        self.file_list = list(file_list)
        if not self.file_list:
            raise ValueError("HimawariCloudDataset: file_list 为空，请检查数据路径。")

        self.input_channels = int(input_channels)
        self.num_total_labels = int(num_total_labels)
        self.sza_index = int(sza_index)
        self.cldtype_label_index = int(cldtype_label_index)
        
        # 默认的7个回归标签索引：DCOMP35_CPS, DCOMP36_CPS, DCOMP37_CPS, DCOMP35_COD, CldPressure, CldHeight, CldTemperature
        # 在标签列表中的索引: [3, 4, 5, 2, 0, 6, 7]
        self.reg_label_indices = list(reg_label_indices) if reg_label_indices is not None else [3, 4, 5, 2, 0, 6, 7]
        self.eps = 1e-6

        # 归一化统计量（来自训练集）
        # input_mean/std: [16] = [8 TIR] + [VZA] + [7 cloud params]
        # target_mean/std: [7] = 7个回归目标
        self.input_mean = np.asarray(input_mean, dtype=np.float32)
        self.input_std = np.asarray(input_std, dtype=np.float32)
        self.target_mean = np.asarray(target_mean, dtype=np.float32)
        self.target_std = np.asarray(target_std, dtype=np.float32)

        if self.input_mean.shape[0] != self.input_channels:
            raise ValueError(f"input_mean 维度 {self.input_mean.shape[0]} 与 input_channels {self.input_channels} 不一致")
        if self.target_mean.shape[0] != len(self.reg_label_indices):
            raise ValueError(f"target_mean 维度 {self.target_mean.shape[0]} 与 reg_label_indices 长度 {len(self.reg_label_indices)} 不一致")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx: int):
        file_path = self.file_list[idx]
        try:
            data = np.load(file_path)
            if "data" in data:
                cube = data["data"]
            else:
                keys = list(data.keys())
                cube = data[keys[0]]
        except Exception as e:
            raise RuntimeError(f"加载文件失败: {file_path}, 错误: {e}")

        cube = cube.astype(np.float32)
        if cube.ndim != 3:
            raise ValueError(f"{file_path} 不是 [C,H,W] 结构。")

        C, H, W = cube.shape
        expected_C = 8 + 1 + self.num_total_labels  # 8 TIR + SZA + 10 labels = 19
        if C < expected_C:
            raise ValueError(f"{file_path} 通道数 {C} < {expected_C}，请确认预处理是否一致。")
        cube = cube[:expected_C, :, :]

        # ---------- 全局数据有效性检查：所有19通道都无负值、NaN ----------
        # 用户要求：所有19个通道都存在且无负值、NaN值
        # 这里只检查是否有足够的有效像元，无效像元会在后续被mask掉
        for c in range(expected_C):
            channel_data = cube[c, :, :]
            # 至少要有一些有效数据点
            valid_mask = (channel_data >= 0) & np.isfinite(channel_data)
            if not np.any(valid_mask):
                raise ValueError(f"{file_path} 通道 {c} 没有有效数据（全是负值或NaN）")

        # 提取各部分
        tir_channels = cube[:8, :, :]  # [8, H, W]
        sza = cube[self.sza_index, :, :]  # [H, W]
        labels = cube[9:, :, :]  # [10, H, W]

        # 提取VZA（标签的最后一个，索引9）
        vza = labels[9, :, :]  # [H, W]

        # 提取CldType（分类目标）
        cldtype = labels[self.cldtype_label_index, :, :]  # [H, W]

        # 提取7个回归标签
        reg_labels = labels[self.reg_label_indices, :, :]  # [7, H, W]

        # ---------- 像元级筛选 ----------
        # 1) CldType > 1（云区域，值为2-9）
        mask = (cldtype > 1) & (cldtype <= 9) & np.isfinite(cldtype)

        # 2) SZA < 60度（白天数据）
        mask &= (sza >= 0.0) & (sza < 60.0) & np.isfinite(sza)

        # 3) 所有TIR通道、VZA无负值、NaN、0值
        tir_channels[tir_channels <= 0] = np.nan
        vza_copy = vza.copy()
        vza_copy[vza_copy < 0] = np.nan
        
        nan_tir = np.any(~np.isfinite(tir_channels), axis=0)
        mask &= ~nan_tir
        mask &= np.isfinite(vza_copy)

        # 4) 7个回归标签无负值、NaN
        reg_labels[reg_labels < 0] = np.nan
        nan_reg = np.any(~np.isfinite(reg_labels), axis=0)
        mask &= ~nan_reg

        # 5) 无效像元标记
        invalid = ~mask

        # 将无效像元的标签设为NaN或特殊值
        cldtype_target = cldtype.copy()
        cldtype_target[invalid] = 0  # 无效像元的CldType设为0（背景类，在loss中会被ignore）

        reg_labels[:, invalid] = np.nan

        # ---------- 构建模型输入: [8 TIR] + [VZA] + [7 cloud params] = 16通道 ----------
        # 首先对各部分进行归一化
        # TIR通道归一化 (前8个统计量)
        tir_mean = self.input_mean[:8].reshape(-1, 1, 1)
        tir_std = self.input_std[:8].reshape(-1, 1, 1)
        tir_std = np.where(tir_std < self.eps, 1.0, tir_std)
        tir_norm = (tir_channels - tir_mean) / tir_std
        tir_norm = np.nan_to_num(tir_norm, nan=0.0, posinf=0.0, neginf=0.0)

        # VZA归一化 (第9个统计量)
        vza_mean = self.input_mean[8]
        vza_std = self.input_std[8] if self.input_std[8] >= self.eps else 1.0
        vza_norm = (vza_copy - vza_mean) / vza_std
        vza_norm = np.nan_to_num(vza_norm, nan=0.0, posinf=0.0, neginf=0.0)

        # 7个云参数归一化 (第10-16个统计量)
        cloud_params_mean = self.input_mean[9:16].reshape(-1, 1, 1)
        cloud_params_std = self.input_std[9:16].reshape(-1, 1, 1)
        cloud_params_std = np.where(cloud_params_std < self.eps, 1.0, cloud_params_std)
        cloud_params_norm = (reg_labels - cloud_params_mean) / cloud_params_std
        
        # 保留NaN（用于loss中的mask）
        for c in range(cloud_params_norm.shape[0]):
            m = np.isfinite(cloud_params_norm[c])
            if not np.any(m):
                cloud_params_norm[c] = 0.0

        # 拼接输入: [8, H, W] + [1, H, W] + [7, H, W] = [16, H, W]
        inputs = np.concatenate([
            tir_norm,
            vza_norm[np.newaxis, :, :],
            cloud_params_norm
        ], axis=0)

        # ---------- 归一化回归目标（使用target统计量）----------
        reg_targets_norm = reg_labels.copy()
        t_mean = self.target_mean.reshape(-1, 1, 1)
        t_std = self.target_std.reshape(-1, 1, 1)
        t_std = np.where(t_std < self.eps, 1.0, t_std)

        for c in range(reg_targets_norm.shape[0]):
            m = np.isfinite(reg_targets_norm[c])
            if np.any(m):
                reg_targets_norm[c, m] = (reg_targets_norm[c, m] - t_mean[c, 0, 0]) / t_std[c, 0, 0]

        # 转换为Tensor
        input_tensor = torch.from_numpy(inputs.astype(np.float32))
        cldtype_tensor = torch.from_numpy(cldtype_target.astype(np.int64))  # 分类目标
        reg_target_tensor = torch.from_numpy(reg_targets_norm.astype(np.float32))  # 回归目标

        return input_tensor, cldtype_tensor, reg_target_tensor
