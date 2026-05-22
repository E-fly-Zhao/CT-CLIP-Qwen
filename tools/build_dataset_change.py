import os
import subprocess
import json
import pandas as pd
import shutil
from pathlib import Path
from tqdm import tqdm

# ================= 1. 核心配置文件路径 =================
INPUT_DIR = "/mnt/share_data/CT/ct_dataset_base_260316/lz2nodesk_ct_chest_1000"
# OUTPUT_DIR = "/mnt/share_data/CT/ct_dataset_base_260316/pretrain_processed_data"
OUTPUT_DIR = "/data2/ct_chest_1000"
EXCEL_PATH = "/mnt/share_data/CT/ct_dataset_base_260316/0316_nodesk_ct_chest_1000.xlsx"

# 🌟 修改此处以切换 Train / Valid 的输出文件名
CSV_OUTPUT_PATH = "/home/huali/code/CT-CLIP-main/CT_CLIP/dataset/train_reports.csv"

# ================= 2. 绝对路径映射前缀 =================
LOCAL_PREFIX = "/mnt/share_data/CT/ct_dataset_base_260316/lz2nodesk_ct_chest_1000"
EXCEL_PREFIX = "/data/nodesk-aliyun-oss/mnt/lz2nodesk_ct_chest_1000"

# ================= 3. 数据集切片配置 (0-based) =================
START_IDX = 0
END_IDX = 1000

def load_excel_database():
    """读取 Excel，构建基于绝对路径的精准文本查询字典"""
    print("⏳ 正在安全加载并索引 Excel 诊断数据 (绝对路径匹配模式)...")
    df = pd.read_excel(EXCEL_PATH)
    
    df['临床诊断'] = df['临床诊断'].fillna('未提及')
    df.loc[df['临床诊断'].astype(str).str.strip() == '', '临床诊断'] = '未提及'
    df['影像所见'] = df['影像所见'].fillna('缺失')
    df['影像所得'] = df['影像所得'].fillna('缺失')
    
    lookup_dict = {}
    for _, row in df.iterrows():
        d_path = str(row['dicom_path']).strip().rstrip('/')
        lookup_dict[d_path] = {
            '临床诊断': str(row['临床诊断']).strip(),
            '影像所见': str(row['影像所见']).strip(),
            '影像所得': str(row['影像所得']).strip()
        }
    print(f"✅ Excel 加载完成！成功索引 {len(lookup_dict)} 条记录。\n")
    return lookup_dict

def main():
    if not shutil.which("dcm2niix"):
        print("❌ 致命错误: 当前 Python 环境中未安装 dcm2niix！")
        return

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    csv_out_dir = os.path.dirname(CSV_OUTPUT_PATH)
    if csv_out_dir:
        Path(csv_out_dir).mkdir(parents=True, exist_ok=True)

    text_db = load_excel_database()
    all_subdirs = sorted([d for d in Path(INPUT_DIR).iterdir() if d.is_dir()])
    target_subdirs = all_subdirs[START_IDX:END_IDX]
    
    print(f"🚀 开始处理第 {START_IDX+1} 到 {END_IDX} 个文件夹 (共 {len(target_subdirs)} 个)...\n" + "="*50)

    csv_data = []

    for folder in tqdm(target_subdirs, desc="DICOM 转换与图文匹配", unit="case"):
        folder_name = folder.name
        expected_excel_path = f"{EXCEL_PREFIX}/{folder_name}"

        patient_output_dir = Path(OUTPUT_DIR) / folder_name
        patient_output_dir.mkdir(parents=True, exist_ok=True)

        command = [
            'dcm2niix', '-z', 'y', '-f', '%i_%s_%p', '-o', str(patient_output_dir), str(folder)
        ]
        
        try:
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            # 如果转换失败且目录为空，才跳过
            if result.returncode != 0 and not os.listdir(patient_output_dir):
                continue
        except Exception:
            continue

        # 获取该患者目录下所有转换出来的 NIfTI 文件
        nii_files = list(patient_output_dir.glob("*.nii.gz"))
        
        # 🌟 核心修改：不再做任何过滤，直接获取所有文件名
        all_slice_names = [nii_path.name for nii_path in nii_files]
                
        if all_slice_names:
            report = text_db.get(expected_excel_path, {
                '临床诊断': '未提及', '影像所见': '缺失', '影像所得': '缺失'
            })
            
            # 🌟 无条件将所有文件登记进 CSV
            for slice_name in all_slice_names:
                csv_data.append({
                    'VolumeName': slice_name,
                    'SourceFolder': folder_name,
                    '临床诊断': report['临床诊断'],
                    '影像所见': report['影像所见'],
                    '影像所得': report['影像所得']
                })

    print("\n" + "="*50)
    print("⏳ 正在保存图文对齐 CSV 文件...")
    df_out = pd.DataFrame(csv_data)
    
    if not df_out.empty:
        # 清理文本中的换行符，防止 CSV 格式错乱
        for col in ['临床诊断', '影像所见', '影像所得']:
            if col in df_out.columns:
                df_out[col] = df_out[col].astype(str).str.replace('\n', ' ', regex=False).str.replace('\r', '', regex=False)

        df_out.to_csv(CSV_OUTPUT_PATH, index=False, encoding='utf-8-sig')
        print(f"🎉 纯净转换完毕！成功生成 {len(csv_data)} 组数据（含所有序列与定位图）。")
        print(f"📄 标注 CSV 存放于: {CSV_OUTPUT_PATH}")
    else:
        print("⚠️ 未能提取到任何数据，未能生成 CSV。")

if __name__ == "__main__":
    main()
