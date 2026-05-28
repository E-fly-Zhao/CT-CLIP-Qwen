import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from functools import partial
import torch.nn.functional as F
import nibabel as nib

def resize_array_global(array, current_xy_spacing, target_xy_spacing, target_z_dim):
    """
    自适应全局重采样引擎 (专为全部位大模型设计)
    array shape 预期: (1, 1, Z, X, Y)
    """
    original_shape = array.shape[2:] 
    orig_z, orig_x, orig_y = original_shape

    # 1. 对于 X 和 Y，维持绝对物理空间缩放 (保证器官横截面积不失真)
    new_x = max(1, int(orig_x * (current_xy_spacing / target_xy_spacing)))
    new_y = max(1, int(orig_y * (current_xy_spacing / target_xy_spacing)))

    # 2. 核心：对于 Z 轴，彻底放弃绝对间距，强行三线性拉伸/压扁到 TARGET_D (全局保留)
    new_shape = [target_z_dim, new_x, new_y]

    resized_array = F.interpolate(array, size=new_shape, mode='trilinear', align_corners=False).cpu().numpy()
    return resized_array

class CTReportDataset(Dataset):
    def __init__(self, filtered_csv_path, reports_file, meta_file, is_train=True):
        # 直接通过过滤后的 CSV 读取高质量样本路径
        self.filtered_csv_path = filtered_csv_path
        
        # 1. 加载并拼接中文报告 (纯文本处理)
        # self.accession_to_text = self.load_accession_text(reports_file)

        # 🚨 新增：定义 10 个基础部位列表
        self.body_parts_list = ["头部", "颈部", "胸部", "腹部", "盆腔", "脊柱", "骨骼", "软组织", "全身", "其他"]
        
        self.accession_to_data = self.load_accession_data(reports_file) # 函数改名
        
        # 2. 匹配图像文件和报告
        self.samples = self.prepare_samples_from_filtered_csv()

        # 3. FSDP 多机多卡防死锁/防梯度污染机制
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        local_batch_size = 2 if is_train else 1
        global_batch_size = world_size * local_batch_size  
        
        if global_batch_size > 0:
            remainder = len(self.samples) % global_batch_size
            if remainder != 0:
                if is_train:
                    # ⚔️ 训练集：物理截断！绝不让重复样本污染梯度！
                    valid_length = (len(self.samples) // global_batch_size) * global_batch_size
                    self.samples = self.samples[:valid_length]
                    print(f"⚔️ [Train] 截断至 {len(self.samples)} 个样本 (全局BS: {global_batch_size})。")
                else:
                    # 🛡️ 验证集：复制补齐，防止 NCCL 通信崩溃
                    pad_count = global_batch_size - remainder
                    self.samples.extend(self.samples[:pad_count])
                    print(f"🛡️ [Valid] 补齐至 {len(self.samples)} 个样本 (全局BS: {global_batch_size})。")

        df = pd.read_csv(meta_file) 
        self.nii_to_tensor = partial(self.nii_img_to_tensor, df=df)

    # def load_accession_text(self, reports_file):
    #     """读取中文报告并拼接"""
    #     df = pd.read_csv(reports_file)
    #     for col in ['临床诊断', '影像所见', '影像所得']:
    #         if col in df.columns:
    #             df[col] = df[col].fillna('')
                
    #     accession_to_text = {}
    #     for index, row in df.iterrows():
    #         vol_name = str(row['VolumeName']).strip()
            
    #         clin_diag = str(row['临床诊断']).strip() if '临床诊断' in df.columns else ""
    #         findings = str(row['影像所见']).strip() if '影像所见' in df.columns else ""
    #         impression = str(row['影像所得']).strip() if '影像所得' in df.columns else ""
            
    #         combined_text = "".join([f"{k}：{v}。" for k, v in zip(["临床诊断", "影像所见", "影像所得"], [clin_diag, findings, impression]) if v])
    #         if not combined_text: 
    #             combined_text = "未提供有效诊断信息。"
                
    #         accession_to_text[vol_name] = combined_text
    #     return accession_to_text

    # def prepare_samples_from_filtered_csv(self):
    #     """直接从高纯度清洗表中精准挂载数据"""
    #     samples = []
    #     df_filtered = pd.read_csv(self.filtered_csv_path)
        
    #     for _, row in df_filtered.iterrows():
    #         nii_file = row['nii_path']
    #         file_name = nii_file.split("/")[-1]
    #         file_name_no_ext = file_name.replace(".nii.gz", "")
            
    #         if file_name in self.accession_to_text:
    #             matched_key = file_name
    #         elif file_name_no_ext in self.accession_to_text:
    #             matched_key = file_name_no_ext
    #         else:
    #             continue 

    #         text = self.accession_to_text[matched_key]
    #         samples.append((nii_file, text))
            
    #     print(f"✅ 从清洗表中成功映射了 {len(samples)} 个图文配对样本。")
    #     return samples
        
    def load_accession_data(self, reports_file):
        """🚨 升级：同时读取诊断文本和部位标签"""
        df = pd.read_csv(reports_file)
        for col in ['临床诊断', '影像所见', '影像所得', '部位']:
            if col in df.columns:
                df[col] = df[col].fillna('')
                
        accession_data = {}
        for index, row in df.iterrows():
            vol_name = str(row['VolumeName']).strip()
            
            clin_diag = str(row.get('临床诊断', '')).strip()
            findings = str(row.get('影像所见', '')).strip()
            impression = str(row.get('影像所得', '')).strip()
            body_str = str(row.get('部位', '')).strip()
            
            combined_text = "".join([f"{k}：{v}。" for k, v in zip(["临床诊断", "影像所见", "影像所得"], [clin_diag, findings, impression]) if v])
            if not combined_text: 
                combined_text = "未提供有效诊断信息。"
            
            # 🎯 生成部位的多标签 (Multi-hot) 张量
            label_tensor = torch.zeros(len(self.body_parts_list), dtype=torch.float32)
            for i, part in enumerate(self.body_parts_list):
                if part in body_str:
                    label_tensor[i] = 1.0
            
            # 如果什么都没匹配上，归为“其他”
            if label_tensor.sum() == 0:
                label_tensor[-1] = 1.0
                
            accession_data[vol_name] = {'text': combined_text, 'body_label': label_tensor}
        return accession_data

    def prepare_samples_from_filtered_csv(self):
        samples = []
        df_filtered = pd.read_csv(self.filtered_csv_path)
        for _, row in df_filtered.iterrows():
            nii_file = row['nii_path']
            file_name = nii_file.split("/")[-1]
            file_name_no_ext = file_name.replace(".nii.gz", "")
            
            if file_name in self.accession_to_data:
                matched_key = file_name
            elif file_name_no_ext in self.accession_to_data:
                matched_key = file_name_no_ext
            else:
                continue 

            data_dict = self.accession_to_data[matched_key]
            # 🚨 挂载部位标签
            samples.append((nii_file, data_dict['text'], data_dict['body_label']))
            
        print(f"✅ 从清洗表中成功映射了 {len(samples)} 个图文+部位配对样本。")
        return samples

    def __len__(self):
        return len(self.samples)

    def nii_img_to_tensor(self, path, df):
        nii_img = nib.load(str(path))
        img_data = nii_img.get_fdata()

        file_name = path.split("/")[-1]
        file_name_no_ext = file_name.replace(".nii.gz", "")
        row = df[df['VolumeName'] == file_name]
        if row.empty: row = df[df['VolumeName'] == file_name_no_ext]
        if row.empty: raise ValueError(f"----------------目标缺失: {file_name}")

        slope = float(row["RescaleSlope"].iloc[0])
        intercept = float(row["RescaleIntercept"].iloc[0])
        xy_spacing_str = str(row["XYSpacing"].iloc[0])
        xy_spacing = float(xy_spacing_str[1:][:-2].split(",")[0]) if "[" in xy_spacing_str else float(xy_spacing_str.split(",")[0])

        # ==========================================
        # 🔥 全部位模型参数配置 🔥
        # ==========================================
        TARGET_D = 320       # 从原版的 240 上调至 320，确保全身特征不严重失真
        TARGET_XY_SPACING = 0.75
        TARGET_H, TARGET_W = 480, 480

        img_data = slope * img_data + intercept

        # 降维结界 (防御异常 4D/5D 数据)
        while img_data.ndim > 3: img_data = img_data[..., 0]
        if img_data.ndim == 2: img_data = img_data[:, :, np.newaxis]
        if img_data.ndim < 2: raise ValueError(f"严重畸形数据")

        img_data = img_data.transpose(2, 0, 1)
        tensor = torch.tensor(img_data.copy()).unsqueeze(0).unsqueeze(0)

        # 全局压扁/拉伸核心 (任何尺寸切片全部统一至 TARGET_D)
        img_data = resize_array_global(tensor, xy_spacing, TARGET_XY_SPACING, TARGET_D)[0][0]
        img_data = np.transpose(img_data, (1, 2, 0))

        # HU 归一化
        # img_data = np.clip(img_data, -1000, 1000)
        # img_data = (img_data / 1000).astype(np.float32)
        # 将原本的 -1000 到 1000，改为 -1024 到 2048 (或3000)，以保留骨皮质细节
        img_data = np.clip(img_data, -1024, 3000)
        # 归一化也相应调整到 0~1 或 -1~1 范围
        img_data = (img_data - (-1024)) / (3000 - (-1024))

        tensor = torch.tensor(img_data)
        h, w, d = tensor.shape # 注意此时的 d 已经是 320 了

        # 废弃 Z 轴裁剪，仅裁剪 X 和 Y 轴
        h_start = max((h - TARGET_H) // 2, 0)
        h_end = min(h_start + TARGET_H, h)
        w_start = max((w - TARGET_W) // 2, 0)
        w_end = min(w_start + TARGET_W, w)

        tensor = tensor[h_start:h_end, w_start:w_end, :] # Z 轴保留 100% 视野！

        # X 和 Y 轴的 Padding
        pad_h_before = (TARGET_H - tensor.size(0)) // 2
        pad_h_after = TARGET_H - tensor.size(0) - pad_h_before
        pad_w_before = (TARGET_W - tensor.size(1)) // 2
        pad_w_after = TARGET_W - tensor.size(1) - pad_w_before

        tensor = torch.nn.functional.pad(tensor, (0, 0, pad_w_before, pad_w_after, pad_h_before, pad_h_after), value=-1)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)

        return tensor
    
    # def __getitem__(self, index):
    #     try:
    #         # 返回项已恢复为原版的 2 项：图像，文本
    #         nii_file, input_text = self.samples[index]
    #         video_tensor = self.nii_to_tensor(nii_file)
    #         input_text = str(input_text).replace('"', '').replace('\'', '').replace('(', '').replace(')', '')
    #         return video_tensor, input_text
    #     except Exception as e:
    #         # 替身使者容错机制
    #         import random
    #         return self.__getitem__(random.randint(0, len(self.samples) - 1))
    def __getitem__(self, index):
        try:
            # 🚨 返回 3 个元素：图像，文本，部位标签
            nii_file, input_text, body_label = self.samples[index]
            video_tensor = self.nii_to_tensor(nii_file)
            input_text = str(input_text).replace('"', '').replace('\'', '').replace('(', '').replace(')', '')
            return video_tensor, input_text, body_label
        except Exception as e:
            import random
            return self.__getitem__(random.randint(0, len(self.samples) - 1))
