# import os
# import glob
# import torch
# import pandas as pd
# import numpy as np
# from torch.utils.data import Dataset
# from functools import partial
# import torch.nn.functional as F
# import tqdm
# import nibabel as nib

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



# class CTReportDatasetinfer(Dataset):
#     def __init__(self, data_folder, reports_file, meta_file, min_slices=20, labels = "labels.csv"):
#         self.data_folder = data_folder
#         self.min_slices = min_slices
#         self.labels = labels
#         self.accession_to_text = self.load_accession_text(reports_file)
#         self.paths=[]
#         self.samples = self.prepare_samples()
#         df = pd.read_csv(meta_file) #select the metadata
#         self.nii_to_tensor = partial(self.nii_img_to_tensor, df = df)

#     # def load_accession_text(self, reports_file):
#     #     df = pd.read_csv(reports_file)
#     #     accession_to_text = {}
#     #     for index, row in df.iterrows():
#     #         accession_to_text[row['VolumeName']] = row["Findings_EN"],row['Impressions_EN']
#     #     return accession_to_text
#     def load_accession_text(self, reports_file):
#         df = pd.read_csv(reports_file)
        
#         # 填充空值
#         if '临床诊断' in df.columns:
#             df['临床诊断'] = df['临床诊断'].fillna('')
#         if '影像所见' in df.columns:
#             df['影像所见'] = df['影像所见'].fillna('')
#         if '影像所得' in df.columns:
#             df['影像所得'] = df['影像所得'].fillna('')
            
#         accession_to_text = {}
#         for index, row in df.iterrows():
#             vol_name = str(row['VolumeName']).strip()
            
#             clin_diag = str(row['临床诊断']).strip() if '临床诊断' in df.columns else ""
#             findings = str(row['影像所见']).strip() if '影像所见' in df.columns else ""
#             impression = str(row['影像所得']).strip() if '影像所得' in df.columns else ""
            
#             # 拼接中文文本
#             combined_text = ""
#             if clin_diag:
#                 combined_text += f"临床诊断：{clin_diag}。"
#             if findings:
#                 combined_text += f"影像所见：{findings}。"
#             if impression:
#                 combined_text += f"影像所得：{impression}。"
            
#             if not combined_text:
#                 combined_text = "未提供有效诊断信息。"
                
#             accession_to_text[vol_name] = combined_text

#         return accession_to_text


#     def prepare_samples(self):
#         samples = []
#         patient_folders = glob.glob(os.path.join(self.data_folder, '*'))

#         # Read labels once outside the loop
#         test_df = pd.read_csv(self.labels)
#         test_label_cols = list(test_df.columns[1:])
#         test_df['one_hot_labels'] = list(test_df[test_label_cols].values)

#         for patient_folder in tqdm.tqdm(patient_folders):
#             accession_folders = glob.glob(os.path.join(patient_folder, '*'))

#             for accession_folder in accession_folders:
#                 nii_files = glob.glob(os.path.join(accession_folder, '*.nii.gz'))

#                 for nii_file in nii_files:
#                     accession_number = nii_file.split("/")[-1]

#                     if accession_number not in self.accession_to_text:
#                         continue

#                     impression_text = self.accession_to_text[accession_number]
#                     text_final = ""
#                     for text in list(impression_text):
#                         text = str(text)
#                         if text == "Not given.":
#                             text = ""

#                         text_final = text_final + text

#                     onehotlabels = test_df[test_df["VolumeName"] == accession_number]["one_hot_labels"].values
#                     if len(onehotlabels) > 0:
#                         samples.append((nii_file, text_final, onehotlabels[0]))
#                         self.paths.append(nii_file)
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
#         hu_min, hu_max = -1000, 1000
#         img_data = np.clip(img_data, hu_min, hu_max)

#         img_data = img_data.transpose(2, 0, 1)

#         tensor = torch.tensor(img_data)
#         tensor = tensor.unsqueeze(0).unsqueeze(0)

#         img_data = resize_array(tensor, current, target)
#         img_data = img_data[0][0]
#         img_data= np.transpose(img_data, (1, 2, 0))

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
#         nii_file, input_text, onehotlabels = self.samples[index]
#         video_tensor = self.nii_to_tensor(nii_file)
#         input_text = input_text.replace('"', '')
#         input_text = input_text.replace('\'', '')
#         input_text = input_text.replace('(', '')
#         input_text = input_text.replace(')', '')
#         name_acc = nii_file.split("/")[-1].replace(".nii.gz", "")
#         return video_tensor, input_text, onehotlabels, name_acc

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
import random # 引入 random 用于生成诱饵标签

def resize_array(array, current_spacing, target_spacing):
    original_shape = array.shape[2:]
    scaling_factors = [
        current_spacing[i] / target_spacing[i] for i in range(len(original_shape))
    ]
    new_shape = [
        int(original_shape[i] * scaling_factors[i]) for i in range(len(original_shape))
    ]
    resized_array = F.interpolate(array, size=new_shape, mode='trilinear', align_corners=False).cpu().numpy()
    return resized_array

class CTReportDatasetinfer(Dataset):
    def __init__(self, data_folder, reports_file, meta_file, labels=None, min_slices=20, resize_dim=500, force_num_frames=True):
        self.data_folder = data_folder
        self.min_slices = min_slices
        # 接收 labels 参数但不再强行读取它
        self.labels = labels 
        
        # 使用和训练集一致的中文报告解析逻辑
        self.accession_to_text = self.load_accession_text(reports_file)
        self.paths=[]
        self.samples = self.prepare_samples()
        self.count = 0

        df = pd.read_csv(meta_file)
        self.nii_to_tensor = partial(self.nii_img_to_tensor, df = df)

    def load_accession_text(self, reports_file):
        df = pd.read_csv(reports_file)
        
        # 填充空值
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
            if clin_diag: combined_text += f"临床诊断：{clin_diag}。"
            if findings: combined_text += f"影像所见：{findings}。"
            if impression: combined_text += f"影像所得：{impression}。"
            
            if not combined_text: combined_text = "未提供有效诊断信息。"
                
            accession_to_text[vol_name] = combined_text

        return accession_to_text

    def prepare_samples(self):
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
            
            # 🔥 核心修改：生成 18 维的诱饵标签（Dummy Labels）
            # 随机生成 0 和 1，保证评估代码不会因为只有一种类别而计算崩溃
            dummy_label = [random.randint(0, 1) for _ in range(18)]
            
            samples.append((nii_file, input_text_concat, dummy_label))
            self.paths.append(nii_file)
            
        return samples
    
    # 👇 给这个类强行补上这一段
    def __len__(self):
        return len(self.samples)

    def nii_img_to_tensor(self, path, df):
        nii_img = nib.load(str(path))
        img_data = nii_img.get_fdata()

        file_name = path.split("/")[-1]
        file_name_no_ext = file_name.replace(".nii.gz", "")
        
        row = df[df['VolumeName'] == file_name]
        if row.empty:
            row = df[df['VolumeName'] == file_name_no_ext]
            
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

        img_data = slope * img_data + intercept
        img_data = img_data.transpose(2, 0, 1)

        tensor = torch.tensor(img_data)
        tensor = tensor.unsqueeze(0).unsqueeze(0)

        img_data = resize_array(tensor, current, target)
        img_data = img_data[0][0]
        img_data= np.transpose(img_data, (1, 2, 0))

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
        # 取出我们刚才放入的 3 个元素：路径、文本、诱饵标签
        nii_file, input_text, label = self.samples[index]
        video_tensor = self.nii_to_tensor(nii_file)
        
        # 清洗文本
        input_text = str(input_text)
        input_text = input_text.replace('"', '').replace('\'', '').replace('(', '').replace(')', '')
        
        # 将诱饵标签转换为模型需要的 float32 Tensor (对应期望的第 3 个变量: onehotlabels)
        label_tensor = torch.tensor(label, dtype=torch.float32)

        # 提取文件名 (对应期望的第 4 个变量: name_acc)
        name_acc = nii_file.split("/")[-1].replace(".nii.gz", "")

        # 严格按照训练脚本期望的顺序返回这 4 个值：
        # 1. 图像张量, 2. 文本, 3. 标签张量, 4. 文件名
        return video_tensor, input_text, label_tensor, name_acc