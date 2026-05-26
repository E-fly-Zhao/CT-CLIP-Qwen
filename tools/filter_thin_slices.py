import os
import glob
import pandas as pd
import nibabel as nib
from tqdm import tqdm

def find_all_valid_slices_for_patient(patient_dir, min_slices_threshold=50):
    nii_files = glob.glob(os.path.join(patient_dir, "*.nii.gz"))
    valid_files_for_this_patient = []

    for nii_path in nii_files:
        filename = os.path.basename(nii_path).lower()

        # 过滤掩膜
        if "roi" in filename or "mask" in filename or "label" in filename:
            continue

        try:
            # 提取 Header
            img = nib.load(nii_path)
            shape = img.header.get_data_shape()
            
            # 👇 🚨 新增核心：提取物理体素间距 (Voxel Spacing)
            zooms = img.header.get_zooms()
            
            if len(shape) < 3 or len(zooms) < 3:
                continue
            if shape[0] < 100 or shape[1] < 100:
                continue

            num_slices = shape[2]
            
            # 👇 🚨 物理计算：Z 轴体素间距 和 真实物理长度
            z_spacing = round(float(zooms[2]), 4)
            z_length_mm = round(num_slices * z_spacing, 2)
            
            if num_slices >= min_slices_threshold:
                valid_files_for_this_patient.append({
                    "nii_path": nii_path,
                    "slice_count": num_slices,
                    "full_shape": str(shape),
                    "z_spacing_mm": z_spacing,
                    "z_physical_length_mm": z_length_mm  # 真实扫描长度
                })
            
        except Exception as e:
            continue

    return valid_files_for_this_patient

def process_dataset(base_data_dir, output_csv, min_slices_threshold=50):
    print(f"\n📂 正在扫描并计算物理长度: {base_data_dir}")
    patient_dirs = [os.path.join(base_data_dir, d) for d in os.listdir(base_data_dir) 
                    if os.path.isdir(os.path.join(base_data_dir, d))]
    
    all_valid_results = []
    
    for patient_dir in tqdm(patient_dirs, desc="Processing Patients"):
        patient_id = os.path.basename(patient_dir)
        valid_files = find_all_valid_slices_for_patient(patient_dir, min_slices_threshold)
        for vf in valid_files:
            vf["patient_id"] = patient_id
            all_valid_results.append(vf)

    df = pd.DataFrame(all_valid_results)
    
    if not df.empty:
        df = df.sort_values(by=["slice_count"], ascending=[False])
        df = df[["patient_id", "slice_count", "z_spacing_mm", "z_physical_length_mm", "full_shape", "nii_path"]]
        
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"\n✅ 扫描完成！物理验证表格已保存至: {output_csv}")

if __name__ == "__main__":
    TRAIN_DIR = "/mnt/huali/ct_dataset_10000/pretrain_processed_train_data/"
    VALID_DIR = "/mnt/huali/ct_dataset_10000/pretrain_processed_valid_data/"

    # 输出表格
    TRAIN_CSV_OUT = "/home/huali/code/CT-CLIP-main/CT_CLIP/dataset_10000/filtered_thin_slices_train.csv"
    VALID_CSV_OUT = "/home/huali/code/CT-CLIP-main/CT_CLIP/dataset_10000/filtered_thin_slices_valid.csv"
    
    if os.path.exists(TRAIN_DIR):
        process_dataset(TRAIN_DIR, TRAIN_CSV_OUT, min_slices_threshold=50)

    if os.path.exists(VALID_DIR):
        process_dataset(VALID_DIR, VALID_CSV_OUT, min_slices_threshold=50)
