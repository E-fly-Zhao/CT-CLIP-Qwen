import os
import pandas as pd

def filter_metadata_and_reports(filtered_nii_csv, original_csv, output_csv):
    print(f"📂 正在根据清洗后的薄层数据筛选: {original_csv}")
    
    # 1. 获取所有有效的 VolumeName (有后缀和无后缀都作为合法 Key)
    df_nii = pd.read_csv(filtered_nii_csv)
    valid_keys = set()
    
    for path in df_nii['nii_path'].dropna():
        # 提取文件名
        filename = str(path).split("/")[-1]
        filename_no_ext = filename.replace(".nii.gz", "")
        # 将两种形式都加入白名单，防止原表中后缀不统一
        valid_keys.add(filename)
        valid_keys.add(filename_no_ext)
        
    # 2. 读取原始的 reports 或 metadata
    df_orig = pd.read_csv(original_csv)
    
    # 3. 筛选 VolumeName 存在于白名单中的行
    # 清理首尾空格，增强鲁棒性
    df_orig['VolumeName_clean'] = df_orig['VolumeName'].astype(str).str.strip()
    df_filtered = df_orig[df_orig['VolumeName_clean'].isin(valid_keys)].copy()
    
    # 丢弃临时列
    df_filtered = df_filtered.drop(columns=['VolumeName_clean'])
    
    # 4. 保存为清洗后的全新文件
    df_filtered.to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"✅ 筛选完成! 原始行数: {len(df_orig)} -> 清洗后行数: {len(df_filtered)}")
    print(f"💾 结果保存至: {output_csv}\n")

if __name__ == "__main__":
    base_dir = "/home/huali/code/CT-CLIP-main/CT_CLIP/dataset_10000"
    
    # -------- 训练集处理 --------
    train_nii = f"{base_dir}/filtered_thin_slices_train.csv"
    
    filter_metadata_and_reports(
        train_nii, 
        f"{base_dir}/train_reports.csv", 
        f"{base_dir}/filtered_train_reports.csv" # 产生新的 reports
    )
    filter_metadata_and_reports(
        train_nii, 
        f"{base_dir}/train_metadata.csv", 
        f"{base_dir}/filtered_train_metadata.csv" # 产生新的 metadata
    )

    # -------- 验证集处理 --------
    valid_nii = f"{base_dir}/filtered_thin_slices_valid.csv"
    
    filter_metadata_and_reports(
        valid_nii, 
        f"{base_dir}/valid_reports.csv", 
        f"{base_dir}/filtered_valid_reports.csv" # 产生新的 reports
    )
    filter_metadata_and_reports(
        valid_nii, 
        f"{base_dir}/valid_metadata.csv", 
        f"{base_dir}/filtered_valid_metadata.csv" # 产生新的 metadata
    )
