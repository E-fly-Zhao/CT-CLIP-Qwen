# import os
# import glob
# import torch
# import pandas as pd
# import numpy as np
# from torch.utils.data import Dataset
# from functools import partial
# import torch.nn.functional as F
# import nibabel as nib
# import tqdm

# def resize_array(array, current_spacing, target_spacing):
#     """
#     Resize the array to match the target spacing.

#     Args:
#     array (torch.Tensor): Input array to be resized.
#     current_spacing (tuple): Current voxel spacing (z_spacing, xy_spacing, xy_spacing).
#     target_spacing (tuple): Target voxel spacing (target_z_spacing, target_x_spacing, target_y_spacing).

#     Returns:
#     np.ndarray: Resized array.
#     """
#     # Calculate new dimensions
#     original_shape = array.shape[2:]
#     scaling_factors = [
#         current_spacing[i] / target_spacing[i] for i in range(len(original_shape))
#     ]
#     new_shape = [
#         int(original_shape[i] * scaling_factors[i]) for i in range(len(original_shape))
#     ]
#     # Resize the array
#     resized_array = F.interpolate(array, size=new_shape, mode='trilinear', align_corners=False).cpu().numpy()
#     return resized_array

# class CTReportDataset(Dataset):
#     def __init__(self, data_folder, reports_file, meta_file, min_slices=20, resize_dim=500, force_num_frames=True):
#         self.data_folder = data_folder
#         self.min_slices = min_slices
#         self.accession_to_text = self.load_accession_text(reports_file)
#         self.paths=[]
#         self.samples = self.prepare_samples()
#         percent = 80
#         num_files = int((len(self.samples) * percent) / 100)
#         #num_files = 2286
#         self.samples = self.samples[:num_files]
#         print(len(self.samples))
#         self.count = 0

#         df = pd.read_csv(meta_file) #select the metadata
#         self.nii_to_tensor = partial(self.nii_img_to_tensor, df = df)

#     def load_accession_text(self, reports_file):
#         df = pd.read_csv(reports_file)
#         accession_to_text = {}
#         for index, row in df.iterrows():
#             accession_to_text[row['VolumeName']] = row["Findings_EN"],row['Impressions_EN']

#         return accession_to_text


#     def prepare_samples(self):
#         samples = []
#         for patient_folder in tqdm.tqdm(glob.glob(os.path.join(self.data_folder, '*'))):
#             for accession_folder in glob.glob(os.path.join(patient_folder, '*')):

#                 for nii_file in glob.glob(os.path.join(accession_folder, '*.nii.gz')):
#                     accession_number = nii_file.split("/")[-1]
#                     #accession_number = accession_number.replace(".npz", ".nii.gz")
#                     if accession_number not in self.accession_to_text:
#                         continue

#                     impression_text = self.accession_to_text[accession_number]

#                     if impression_text == "Not given.":
#                         impression_text=""

#                     input_text_concat = ""
#                     for text in impression_text:
#                         input_text_concat = input_text_concat + str(text)
#                     input_text_concat = impression_text[0]
#                     input_text = f'{impression_text}'
#                     samples.append((nii_file, input_text_concat))
#                     self.paths.append(nii_file)
#         return samples

#     def __len__(self):
#         return len(self.samples)



#     def nii_img_to_tensor(self, path, df):
#         nii_img = nib.load(str(path))
#         img_data = nii_img.get_fdata()

#         file_name = path.split("/")[-1]
#         row = df[df['VolumeName'] == file_name]
#         slope = float(row["RescaleSlope"].iloc[0])
#         intercept = float(row["RescaleIntercept"].iloc[0])
#         xy_spacing = float(row["XYSpacing"].iloc[0][1:][:-2].split(",")[0])
#         z_spacing = float(row["ZSpacing"].iloc[0])

#         # Define the target spacing values
#         target_x_spacing = 0.75
#         target_y_spacing = 0.75
#         target_z_spacing = 1.5

#         current = (z_spacing, xy_spacing, xy_spacing)
#         target = (target_z_spacing, target_x_spacing, target_y_spacing)

#         img_data = slope * img_data + intercept

#         img_data = img_data.transpose(2, 0, 1)

#         tensor = torch.tensor(img_data)
#         tensor = tensor.unsqueeze(0).unsqueeze(0)

#         img_data = resize_array(tensor, current, target)
#         img_data = img_data[0][0]
#         img_data= np.transpose(img_data, (1, 2, 0))

#         hu_min, hu_max = -1000, 1000
#         img_data = np.clip(img_data, hu_min, hu_max)

#         img_data = (((img_data ) / 1000)).astype(np.float32)
#         slices=[]

#         tensor = torch.tensor(img_data)
#         # Get the dimensions of the input tensor
#         target_shape = (480,480,240)

#         # Extract dimensions
#         h, w, d = tensor.shape

#         # Calculate cropping/padding values for height, width, and depth
#         dh, dw, dd = target_shape
#         h_start = max((h - dh) // 2, 0)
#         h_end = min(h_start + dh, h)
#         w_start = max((w - dw) // 2, 0)
#         w_end = min(w_start + dw, w)
#         d_start = max((d - dd) // 2, 0)
#         d_end = min(d_start + dd, d)

#         # Crop or pad the tensor
#         tensor = tensor[h_start:h_end, w_start:w_end, d_start:d_end]

#         pad_h_before = (dh - tensor.size(0)) // 2
#         pad_h_after = dh - tensor.size(0) - pad_h_before

#         pad_w_before = (dw - tensor.size(1)) // 2
#         pad_w_after = dw - tensor.size(1) - pad_w_before

#         pad_d_before = (dd - tensor.size(2)) // 2
#         pad_d_after = dd - tensor.size(2) - pad_d_before

#         tensor = torch.nn.functional.pad(tensor, (pad_d_before, pad_d_after, pad_w_before, pad_w_after, pad_h_before, pad_h_after), value=-1)

#         tensor = tensor.permute(2, 0, 1)

#         tensor = tensor.unsqueeze(0)

#         return tensor


#     def __getitem__(self, index):
#         nii_file, input_text = self.samples[index]
#         video_tensor = self.nii_to_tensor(nii_file)
#         input_text = str(input_text)
#         input_text = input_text.replace('"', '')
#         input_text = input_text.replace('\'', '')
#         input_text = input_text.replace('(', '')
#         input_text = input_text.replace(')', '')

#         return video_tensor, input_text

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
    new_shape = [
        int(original_shape[i] * scaling_factors[i]) for i in range(len(original_shape))
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
        
        # 🚨 [关键修复] 原版代码这里是 percent = 80，会导致你辛辛苦苦准备的数据被默默丢弃 20%！
        # 既然你已经分好了 train_fixed 和 valid_fixed，这里必须改为 100
        percent = 100 
        num_files = int((len(self.samples) * percent) / 100)
        self.samples = self.samples[:num_files]
        print(f"Successfully loaded {len(self.samples)} samples.")
        self.count = 0

        df = pd.read_csv(meta_file) # select the metadata
        self.nii_to_tensor = partial(self.nii_img_to_tensor, df=df)

    def load_accession_text(self, reports_file):
        """读取中文 CSV 报告并拼接"""
        df = pd.read_csv(reports_file)
        
        # 填充空值，防止字符串拼接时出现 NaN
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
            
            # 拼接中文文本
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
        # 注意：原版代码这里用的是三层嵌套 glob.glob(os.path.join(self.data_folder, '*/*/*.nii.gz'))
        # 如果你的影像直接放在 train_fixed 目录下，可能需要将循环改为一或两层。这里保留原版的容错写法：
        search_path = os.path.join(self.data_folder, '**', '*.nii.gz')
        file_list = glob.glob(search_path, recursive=True)
        
        if len(file_list) == 0: # 兼容原版文件夹层级
            for patient_folder in glob.glob(os.path.join(self.data_folder, '*')):
                for accession_folder in glob.glob(os.path.join(patient_folder, '*')):
                    file_list.extend(glob.glob(os.path.join(accession_folder, '*.nii.gz')))

        for nii_file in tqdm.tqdm(file_list):
            file_name = nii_file.split("/")[-1]
            file_name_no_ext = file_name.replace(".nii.gz", "")
            
            # 兼容 VolumeName 带不带 .nii.gz 后缀的两种情况
            if file_name in self.accession_to_text:
                matched_key = file_name
            elif file_name_no_ext in self.accession_to_text:
                matched_key = file_name_no_ext
            else:
                continue # 如果 CSV 里没这个患者，就跳过

            # 🚨 [关键修复] 清理了原版混乱的 for text in impression_text 的错误拼写逻辑
            input_text_concat = self.accession_to_text[matched_key]
            
            samples.append((nii_file, input_text_concat))
            self.paths.append(nii_file)
            
        return samples

    def __len__(self):
        return len(self.samples)


    def nii_img_to_tensor(self, path, df):
        """读取 NIfTI 图像并进行标准化预处理"""
        nii_img = nib.load(str(path))
        img_data = nii_img.get_fdata()

        file_name = path.split("/")[-1]
        file_name_no_ext = file_name.replace(".nii.gz", "")
        
        # 兼容 metadata 里的 VolumeName
        row = df[df['VolumeName'] == file_name]
        if row.empty:
            row = df[df['VolumeName'] == file_name_no_ext]
            
        slope = float(row["RescaleSlope"].iloc[0])
        intercept = float(row["RescaleIntercept"].iloc[0])
        
        # 提取间距
        xy_spacing_str = str(row["XYSpacing"].iloc[0])
        if "[" in xy_spacing_str:
            xy_spacing = float(xy_spacing_str[1:][:-2].split(",")[0])
        else:
            xy_spacing = float(xy_spacing_str.split(",")[0])
            
        z_spacing = float(row["ZSpacing"].iloc[0])

        # Define the target spacing values
        target_x_spacing = 0.75
        target_y_spacing = 0.75
        target_z_spacing = 1.5

        current = (z_spacing, xy_spacing, xy_spacing)
        target = (target_z_spacing, target_x_spacing, target_y_spacing)

        # 转换为 HU 值
        img_data = slope * img_data + intercept
        img_data = img_data.transpose(2, 0, 1)

        tensor = torch.tensor(img_data)
        tensor = tensor.unsqueeze(0).unsqueeze(0)

        # 重采样间距
        img_data = resize_array(tensor, current, target)
        img_data = img_data[0][0]
        img_data= np.transpose(img_data, (1, 2, 0))

        # 截断与归一化
        hu_min, hu_max = -1000, 1000
        img_data = np.clip(img_data, hu_min, hu_max)
        img_data = (((img_data ) / 1000)).astype(np.float32)

        tensor = torch.tensor(img_data)
        
        # Get the dimensions of the input tensor
        target_shape = (480,480,240)

        # Extract dimensions
        h, w, d = tensor.shape

        # Calculate cropping/padding values for height, width, and depth
        dh, dw, dd = target_shape
        h_start = max((h - dh) // 2, 0)
        h_end = min(h_start + dh, h)
        w_start = max((w - dw) // 2, 0)
        w_end = min(w_start + dw, w)
        d_start = max((d - dd) // 2, 0)
        d_end = min(d_start + dd, d)

        # Crop or pad the tensor
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
        nii_file, input_text = self.samples[index]
        video_tensor = self.nii_to_tensor(nii_file)
        
        # 简单清洗文本标点
        input_text = str(input_text)
        input_text = input_text.replace('"', '')
        input_text = input_text.replace('\'', '')
        input_text = input_text.replace('(', '')
        input_text = input_text.replace(')', '')

        return video_tensor, input_text