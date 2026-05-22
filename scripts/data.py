import os
import glob
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from functools import partial
import torch.nn.functional as F
import nibabel as nib
import tqdm

def resize_array(array, current_spacing, target_spacing):
    """
    Resize the array to match the target spacing.
    """
    # Calculate new dimensions
    original_shape = array.shape[2:]
    scaling_factors = [
        current_spacing[i] / target_spacing[i] for i in range(len(original_shape))
    ]
    # new_shape = [
    #     int(original_shape[i] * scaling_factors[i]) for i in range(len(original_shape))
    # ]
    # 🚨 增加 max(1, ...) 保底机制，防止单层定位像在缩放后维度变为 0 导致 PyTorch 崩溃
    new_shape = [
        max(1, int(original_shape[i] * scaling_factors[i])) for i in range(len(original_shape))
    ]
    # Resize the array
    resized_array = F.interpolate(array, size=new_shape, mode='trilinear', align_corners=False).cpu().numpy()
    return resized_array

class CTReportDataset(Dataset):
    def __init__(self, data_folder, reports_file, meta_file, min_slices=20, resize_dim=500, force_num_frames=True):
        self.data_folder = data_folder
        self.min_slices = min_slices
        
        # 1. 加载并拼接中文报告
        self.accession_to_text = self.load_accession_text(reports_file)
        self.paths = []
        
        # 2. 匹配图像文件和报告
        self.samples = self.prepare_samples()
        
        # 确保跑 100% 的数据
        percent = 100 
        num_files = int((len(self.samples) * percent) / 100)
        self.samples = self.samples[:num_files]
        # # 自适应多机扩展截断
        # import os

        # # 1. 动态获取集群总显卡数（如果是单机单卡调试，默认为 1）
        # world_size = int(os.environ.get("WORLD_SIZE", 1))

        # # 2. 这里的 local_batch_size 必须和你 run_train.py 里的 batch_size 保持一致
        # # 如果你未来修改了单卡 batch_size，请记得同步修改这里（或者把它作为 __init__ 的参数传进来）
        # local_batch_size = 2

        # global_batch_size = world_size * local_batch_size

        # # 3. 动态截断
        # if global_batch_size > 0:
        #     valid_length = (len(self.samples) // global_batch_size) * global_batch_size
        #     self.samples = self.samples[:valid_length]
        # # ------------------------------------
        # print(f"✅ Successfully loaded {len(self.samples)} samples.")
        # self.count = 0
        import os
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        
        # 通过数据量大小，智能判断当前是 Train 还是 Valid
        # 如果样本数大于 50 就是训练集，单卡 BS=2；否则是验证集，单卡 BS=1
        is_train = len(self.samples) > 50  
        local_batch_size = 2 if is_train else 1
        global_batch_size = world_size * local_batch_size  
        
        if global_batch_size > 0:
            remainder = len(self.samples) % global_batch_size
            if remainder != 0:
                if is_train:
                    # ⚔️ 训练集：物理截断！绝不让重复样本(False Negatives)污染梯度！
                    valid_length = (len(self.samples) // global_batch_size) * global_batch_size
                    self.samples = self.samples[:valid_length]
                    print(f"⚔️ [Train] 截断至 {len(self.samples)} 个样本 (全局BS: {global_batch_size}) 以防梯度污染。")
                else:
                    # 🛡️ 验证集：不更新梯度，复制前面的样本补齐，防止 NCCL 崩溃！
                    pad_count = global_batch_size - remainder
                    self.samples.extend(self.samples[:pad_count])
                    print(f"🛡️ [Valid] 补齐至 {len(self.samples)} 个样本 (全局BS: {global_batch_size}) 以防通信死锁。")

        self.count = 0

        # 保留了对 meta_file 的读取，用于 nii_img_to_tensor 中的 HU 换算
        df = pd.read_csv(meta_file) 
        self.nii_to_tensor = partial(self.nii_img_to_tensor, df=df)

    def load_accession_text(self, reports_file):
        """读取中文 CSV 报告并拼接"""
        df = pd.read_csv(reports_file)
        
        if '临床诊断' in df.columns:
            df['临床诊断'] = df['临床诊断'].fillna('')
        if '影像所见' in df.columns:
            df['影像所见'] = df['影像所见'].fillna('')
        if '影像所得' in df.columns:
            df['影像所得'] = df['影像所得'].fillna('')
            
        accession_to_text = {}
        for index, row in df.iterrows():
            vol_name = str(row['VolumeName']).strip()
            
            clin_diag = str(row['临床诊断']).strip() if '临床诊断' in df.columns else ""
            findings = str(row['影像所见']).strip() if '影像所见' in df.columns else ""
            impression = str(row['影像所得']).strip() if '影像所得' in df.columns else ""
            
            combined_text = ""
            if clin_diag:
                combined_text += f"临床诊断：{clin_diag}。"
            if findings:
                combined_text += f"影像所见：{findings}。"
            if impression:
                combined_text += f"影像所得：{impression}。"
            
            if not combined_text:
                combined_text = "未提供有效诊断信息。"
                
            accession_to_text[vol_name] = combined_text

        return accession_to_text

    def prepare_samples(self):
        """遍历文件夹，匹配影像和文本"""
        samples = []
        search_path = os.path.join(self.data_folder, '**', '*.nii.gz')
        file_list = glob.glob(search_path, recursive=True)
        
        if len(file_list) == 0: 
            for patient_folder in glob.glob(os.path.join(self.data_folder, '*')):
                for accession_folder in glob.glob(os.path.join(patient_folder, '*')):
                    file_list.extend(glob.glob(os.path.join(accession_folder, '*.nii.gz')))

        for nii_file in tqdm.tqdm(file_list):
            file_name = nii_file.split("/")[-1]
            file_name_no_ext = file_name.replace(".nii.gz", "")
            
            if file_name in self.accession_to_text:
                matched_key = file_name
            elif file_name_no_ext in self.accession_to_text:
                matched_key = file_name_no_ext
            else:
                continue 

            input_text_concat = self.accession_to_text[matched_key]
            
            samples.append((nii_file, input_text_concat))
            self.paths.append(nii_file)
            
        return samples

    def __len__(self):
        return len(self.samples)

    def nii_img_to_tensor(self, path, df):
        """保留原版的在线标准化预处理逻辑，并加入了维度自适应结界"""
        import numpy as np  # 确保内部能正常使用 np
        
        nii_img = nib.load(str(path))
        img_data = nii_img.get_fdata()

        file_name = path.split("/")[-1]
        file_name_no_ext = file_name.replace(".nii.gz", "")
        
        # 🚨 先尝试带后缀匹配，如果不匹配再尝试不带后缀匹配
        row = df[df['VolumeName'] == file_name]
        if row.empty:
            row = df[df['VolumeName'] == file_name_no_ext]

        # 如果两种名字都查不到，才触发捕鼠夹
        if row.empty:
            error_msg = (
                f"\n\n{'='*60}\n"
                f"🚨 [致命数据不匹配] 抓到嫌疑犯了！\n"
                f"🚨 当前正在处理的图像文件: {nii_img}\n"
                f"🚨 提取出来的查找 ID: '{file_name}' 或 '{file_name_no_ext}'\n"
                f"🚨 目标 CSV 表格的总行数: {len(df)}\n"
                f"🚨 目标 CSV 表格拥有的列名: {df.columns.tolist()}\n"
                f"🚨 表格第一列的前 3 个值: {df.iloc[:, 0].head(3).tolist()}\n"
                f"{'='*60}\n"
            )
            raise ValueError(error_msg)

        slope = float(row["RescaleSlope"].iloc[0])
        intercept = float(row["RescaleIntercept"].iloc[0])
        
        xy_spacing_str = str(row["XYSpacing"].iloc[0])
        if "[" in xy_spacing_str:
            xy_spacing = float(xy_spacing_str[1:][:-2].split(",")[0])
        else:
            xy_spacing = float(xy_spacing_str.split(",")[0])
            
        z_spacing = float(row["ZSpacing"].iloc[0])

        target_x_spacing = 0.75
        target_y_spacing = 0.75
        target_z_spacing = 1.5

        current = (z_spacing, xy_spacing, xy_spacing)
        target = (target_z_spacing, target_x_spacing, target_y_spacing)

        # ==========================================
        # 转换为 HU 值
        # ==========================================
        img_data = slope * img_data + intercept

        # 👇 🚨 维度降维打击结界：强制确保图像严格为 3D 🚨 👇
        # 处理 4D 甚至 5D 的畸形 CT
        while img_data.ndim > 3:
            img_data = img_data[..., 0]
            
        # 处理只有单层切片被读取为 2D 的情况 (H, W) -> (H, W, 1)
        if img_data.ndim == 2:
            img_data = img_data[:, :, np.newaxis]
            
        # 防御极度损坏的文件
        if img_data.ndim < 2:
            raise ValueError(f"严重畸形数据，维度过低: {img_data.shape}")
        # 👆 🚨 结界结束 🚨 👆

        # 此时 img_data 绝对是完美的 3D，transpose 绝不会再报错！
        img_data = img_data.transpose(2, 0, 1)

        # 务必加上 .copy()，防止 numpy 数组内存不连续导致 PyTorch 底层报错
        tensor = torch.tensor(img_data.copy())
        tensor = tensor.unsqueeze(0).unsqueeze(0)

        # 重采样间距
        img_data = resize_array(tensor, current, target)
        img_data = img_data[0][0]
        
        img_data = np.transpose(img_data, (1, 2, 0))

        # 截断与归一化
        hu_min, hu_max = -1000, 1000
        img_data = np.clip(img_data, hu_min, hu_max)
        img_data = (((img_data ) / 1000)).astype(np.float32)

        tensor = torch.tensor(img_data)
        
        target_shape = (480,480,240)
        h, w, d = tensor.shape

        dh, dw, dd = target_shape
        h_start = max((h - dh) // 2, 0)
        h_end = min(h_start + dh, h)
        w_start = max((w - dw) // 2, 0)
        w_end = min(w_start + dw, w)
        d_start = max((d - dd) // 2, 0)
        d_end = min(d_start + dd, d)

        tensor = tensor[h_start:h_end, w_start:w_end, d_start:d_end]

        pad_h_before = (dh - tensor.size(0)) // 2
        pad_h_after = dh - tensor.size(0) - pad_h_before

        pad_w_before = (dw - tensor.size(1)) // 2
        pad_w_after = dw - tensor.size(1) - pad_w_before

        pad_d_before = (dd - tensor.size(2)) // 2
        pad_d_after = dd - tensor.size(2) - pad_d_before

        tensor = torch.nn.functional.pad(tensor, (pad_d_before, pad_d_after, pad_w_before, pad_w_after, pad_h_before, pad_h_after), value=-1)
        tensor = tensor.permute(2, 0, 1)
        tensor = tensor.unsqueeze(0)

        return tensor
    
    def __getitem__(self, index):
        # 👇 🚨 无尽容错结界：遇到任何脏数据，直接捕获并替换 🚨 👇
        try:
            nii_file, input_text = self.samples[index]
            video_tensor = self.nii_to_tensor(nii_file)
            
            # 简单清洗文本标点
            input_text = str(input_text)
            input_text = input_text.replace('"', '').replace('\'', '').replace('(', '').replace(')', '')

            return video_tensor, input_text
            
        except Exception as e:
            # 当遇到 RGB 图片、维度畸形、或 CSV 缺失等任何报错时，捕获异常
            print(f"⚠️ 跳过损坏的样本: {nii_file}, 原因: {e}") # 取消注释可查看具体哪些图坏了
            
            import random
            # 替身使者机制：随机再抽一张新的顶替它的位置，保证 DataLoader 永远返回正常数据！
            new_index = random.randint(0, len(self.samples) - 1)
            return self.__getitem__(new_index)
        # 👆 🚨 结界结束 🚨 👆

