import os
import subprocess
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ================= 配置文件路径 =================
INPUT_DIR = "/mnt/share_data/CT/ct_dataset_base_260316/lz2nodesk_ct_chest_1000"
# OUTPUT_DIR = "/home/huali/code/CT-CLIP-main/dataset/pretrain_processed_train_data"
OUTPUT_DIR = "/home/huali/code/CT-CLIP-main/dataset/pretrain_processed_valid_data"
EXCEL_PATH = "/mnt/share_data/CT/ct_dataset_base_260316/0316_nodesk_ct_chest_1000.xlsx"
# CSV_OUTPUT_PATH = "/home/huali/code/CT-CLIP-main/CT_CLIP/dataset/train_reports.csv"
CSV_OUTPUT_PATH = "/home/huali/code/CT-CLIP-main/CT_CLIP/dataset/valid_reports.csv"

# ================= 绝对路径映射前缀 =================
# 本地处理时的文件夹根目录
LOCAL_PREFIX = "/mnt/share_data/CT/ct_dataset_base_260316/lz2nodesk_ct_chest_1000"
# Excel表 dicom_path 列中对应的云端/OSS挂载前缀
EXCEL_PREFIX = "/data/nodesk-aliyun-oss/mnt/lz2nodesk_ct_chest_1000"

# 设置要处理的文件夹数量
PROCESS_LIMIT = 10

def is_thin_slice(json_path, nii_path):
    """基于 JSON Metadata 和物理大小，智能判断是否为高精度薄层 (<= 3mm)"""
    if not os.path.exists(json_path):
        return False
        
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
            image_type = meta.get("ImageType", [])
            
            if "LOCALIZER" in [str(t).upper() for t in image_type]:
                return False
                
            thickness = meta.get("SliceThickness") or meta.get("SpacingBetweenSlices")
            if thickness is not None:
                t = float(thickness)
                if 0 < t <= 3.0:
                    return True
                else:
                    return False
    except Exception:
        pass 
        
    size_mb = os.path.getsize(nii_path) / (1024 * 1024)
    return size_mb > 30.0

def load_excel_database():
    """读取 Excel，构建基于绝对路径的精准文本查询字典"""
    print("⏳ 正在安全加载并索引 Excel 诊断数据 (绝对路径匹配模式)...")
    
    df = pd.read_excel(EXCEL_PATH)
    
    # 处理空值
    df['临床诊断'] = df['临床诊断'].fillna('未提及')
    df.loc[df['临床诊断'].astype(str).str.strip() == '', '临床诊断'] = '未提及'
    df['影像所见'] = df['影像所见'].fillna('缺失')
    df['影像所得'] = df['影像所得'].fillna('缺失')
    
    # 构建查询字典，直接使用清理过后的 dicom_path 作为唯一主键 Key
    lookup_dict = {}
    for _, row in df.iterrows():
        # 去除路径前后的空格和多余的尾部斜杠，保证匹配的绝对纯净
        d_path = str(row['dicom_path']).strip().rstrip('/')
        lookup_dict[d_path] = {
            '临床诊断': str(row['临床诊断']).strip(),
            '影像所见': str(row['影像所见']).strip(),
            '影像所得': str(row['影像所得']).strip()
        }
    print(f"✅ Excel 加载完成！成功索引 {len(lookup_dict)} 条记录。\n")
    return lookup_dict

def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    csv_out_dir = os.path.dirname(CSV_OUTPUT_PATH)
    if csv_out_dir:
        Path(csv_out_dir).mkdir(parents=True, exist_ok=True)

    text_db = load_excel_database()

    all_subdirs = sorted([d for d in Path(INPUT_DIR).iterdir() if d.is_dir()])
    # target_subdirs = all_subdirs[:PROCESS_LIMIT]
    target_subdirs = all_subdirs[101:101 + PROCESS_LIMIT]
    
    print(f"🚀 开始处理前 {len(target_subdirs)} 个患者文件夹...\n" + "="*50)

    csv_data = []

    for folder in tqdm(target_subdirs, desc="DICOM 转换与图文匹配", unit="case"):
        folder_name = folder.name
        
        # 🌟 【核心修改】：通过前缀替换，精准重构出该文件夹在 Excel 中应有的绝对路径
        # 例如构建出: /data/nodesk-aliyun-oss/mnt/lz2nodesk_ct_chest_1000/957298105760007
        expected_excel_path = f"{EXCEL_PREFIX}/{folder_name}"

        patient_output_dir = Path(OUTPUT_DIR) / folder_name
        patient_output_dir.mkdir(parents=True, exist_ok=True)

        command = [
            'dcm2niix',
            '-z', 'y',
            '-f', '%i_%s_%p',
            '-o', str(patient_output_dir),
            str(folder)
        ]
        
        try:
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode != 0 and not os.listdir(patient_output_dir):
                continue
        except Exception:
            continue

        nii_files = list(patient_output_dir.glob("*.nii.gz"))
        valid_thin_slices = []
        
        for nii_path in nii_files:
            json_path = str(nii_path).replace('.nii.gz', '.json')
            if is_thin_slice(json_path, str(nii_path)):
                valid_thin_slices.append(nii_path.name)
                
        if valid_thin_slices:
            # 🌟 【核心修改】：使用重构的绝对路径进行查询，杜绝任何错位可能
            report = text_db.get(expected_excel_path, {
                '临床诊断': '未提及',
                '影像所见': '缺失', 
                '影像所得': '缺失'
            })
            
            for thin_slice_name in valid_thin_slices:
                # volume_name = f"{folder_name}/{thin_slice_name}" 
                volume_name = thin_slice_name # 🌟 仅保留纯文件名
                
                csv_data.append({
                    'VolumeName': volume_name,
                    '临床诊断': report['临床诊断'],
                    '影像所见': report['影像所见'],
                    '影像所得': report['影像所得']
                })

    print("\n" + "="*50)
    print("⏳ 正在保存图文对齐 CSV 文件...")
    df_out = pd.DataFrame(csv_data)
    
    if not df_out.empty:
        for col in ['临床诊断', '影像所见', '影像所得']:
            if col in df_out.columns:
                df_out[col] = df_out[col].astype(str).str.replace('\n', ' ', regex=False).str.replace('\r', '', regex=False)

        df_out.to_csv(CSV_OUTPUT_PATH, index=False, encoding='utf-8-sig')
        print(f"🎉 处理完毕！")
        print(f"📈 统计结果: 尝试处理 {len(target_subdirs)} 个文件夹，成功生成 {len(csv_data)} 组高质量图文对。")
        print(f"📁 图像安全转换至: {OUTPUT_DIR}")
        print(f"📄 标注 CSV 存放于: {CSV_OUTPUT_PATH}")
    else:
        print("⚠️ 未能提取到任何有效的薄层图文对数据，未能生成 CSV。")

if __name__ == "__main__":
    main()