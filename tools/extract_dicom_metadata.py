import os
import pandas as pd
import pydicom
from tqdm import tqdm

# ================= 配置文件路径 =================
RAW_DICOM_DIR = "/mnt/share_data/CT/ct_dataset_base_260316/lz2nodesk_ct_chest_1000"
# REPORTS_CSV = "/home/huali/code/CT-CLIP-main/CT_CLIP/dataset/train_reports.csv"
# META_CSV_OUTPUT = "/home/huali/code/CT-CLIP-main/CT_CLIP/dataset/train_metadata.csv"
REPORTS_CSV = "/home/huali/code/CT-CLIP-main/CT_CLIP/dataset/valid_reports.csv"
META_CSV_OUTPUT = "/home/huali/code/CT-CLIP-main/CT_CLIP/dataset/valid_metadata.csv"

# ================= DICOM 标签字典映射 =================
DICOM_KEYWORD_MAP = {
    "Manufacturer": "Manufacturer",
    "SeriesDescription": "SeriesDescription",
    "ManufacturerModelName": "ManufacturerModelName",
    "PatientSex": "PatientSex",
    "PatientAge": "PatientAge",
    "ReconstructionDiameter": "ReconstructionDiameter",
    "DistanceSourceToDetector": "DistanceSourceToDetector",
    "DistanceSourceToPatient": "DistanceSourceToPatient",
    "GantryDetectorTilt": "GantryDetectorTilt",
    "TableHeight": "TableHeight",
    "RotationDirection": "RotationDirection",
    "ExposureTime": "ExposureTime",
    "XRayTubeCurrent": "XRayTubeCurrent",
    "Exposure": "Exposure",
    "FilterType": "FilterType",
    "GeneratorPower": "GeneratorPower",
    "FocalSpots": "FocalSpot",  
    "ConvolutionKernel": "ConvolutionKernel",
    "PatientPosition": "PatientPosition",
    "RevolutionTime": "RevolutionTime",
    "SingleCollimationWidth": "SingleCollimationWidth",
    "TotalCollimationWidth": "TotalCollimationWidth",
    "TableSpeed": "TableSpeed",
    "TableFeedPerRotation": "TableFeedPerRotation",
    "SpiralPitchFactor": "SpiralPitchFactor",
    "DataCollectionCenterPatient": "DataCollectionCenterPatient",
    "ReconstructionTargetCenterPatient": "ReconstructionTargetCenterPatient",
    "ExposureModulationType": "ExposureModulationType",
    "CTDIvol": "CTDIvol",
    "ImagePositionPatient": "ImagePositionPatient",
    "ImageOrientationPatient": "ImageOrientationPatient",
    "SliceLocation": "SliceLocation",
    "SamplesPerPixel": "SamplesPerPixel",
    "PhotometricInterpretation": "PhotometricInterpretation",
    "Rows": "Rows",
    "Columns": "Columns",
    "XYSpacing": "PixelSpacing", 
    "RescaleIntercept": "RescaleIntercept",
    "RescaleSlope": "RescaleSlope",
    "RescaleType": "RescaleType",
    "ZSpacing": "SliceThickness", 
    "StudyDate": "StudyDate"
}

def format_dicom_value(val):
    """严格对齐 CT-RATE 格式：列表转化为 [x, y]，数值转为纯净字符串"""
    if val is None:
        return "缺失"
    
    # 针对 XYSpacing 或 ImagePosition 这种列表格式的完美转换
    if isinstance(val, (pydicom.multival.MultiValue, list)):
        return "[" + ", ".join([str(x) for x in val]) + "]"
        
    # 去除 pydicom 解析时可能带入的冗余格式
    return str(val).strip()

def find_real_3d_slice_and_count(base_folder):
    """
    🌟 核心修复区：只抓取真正的 3D 轴状位切片，剔除定位图和报告！
    """
    valid_ds = None
    target_dir = None
    
    for root, dirs, files in os.walk(base_folder):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                ds = pydicom.dcmread(file_path, stop_before_pixels=True)
                
                # 1. 获取图像类型
                img_type = [str(x).upper() for x in getattr(ds, 'ImageType', [])]
                
                # 2. 致命过滤：如果它是定位图(LOCALIZER/SCOUT) 或 报告，直接踢掉！
                if 'LOCALIZER' in img_type or 'SCOUT' in img_type:
                    continue
                    
                # 3. 四大金刚验证：缺少任意一个核心物理字段，就说明这不是有效切片
                if not hasattr(ds, 'PixelSpacing') or not hasattr(ds, 'RescaleIntercept') \
                   or not hasattr(ds, 'RescaleSlope'):
                    continue
                    
                # 成功找到完美的 3D 代表切片！
                valid_ds = ds
                target_dir = root
                break 
                
            except Exception:
                continue
                
        # 如果在这个文件夹已经找到了，就不用继续往下挖了
        if valid_ds:
            break
            
    # 计算切片总数：基于我们找到的那个真实序列所在的文件夹
    slice_count = 0
    if valid_ds and target_dir:
        # 统计该目录下的有效文件数作为近似切片数
        slice_count = len([f for f in os.listdir(target_dir) if os.path.isfile(os.path.join(target_dir, f))])
        
    return valid_ds, slice_count

def main():
    if not os.path.exists(REPORTS_CSV):
        print(f"❌ 找不到索引文件 {REPORTS_CSV}")
        return

    print("⏳ 正在加载报告索引...")
    df_reports = pd.read_csv(REPORTS_CSV)
    # df_reports['folder_name'] = df_reports['VolumeName'].apply(lambda x: x.split('/')[0])
    df_reports['folder_name'] = df_reports['SourceFolder']
    unique_folders = df_reports['folder_name'].unique()
    
    metadata_records = []

    print(f"🚀 开始从原始 DICOM 中提取严谨 Metadata (共 {len(unique_folders)} 个患者)...")
    
    for folder_name in tqdm(unique_folders, desc="挖掘 DICOM"):
        dicom_folder_path = os.path.join(RAW_DICOM_DIR, folder_name)
        
        row_data = {col: "缺失" for col in DICOM_KEYWORD_MAP.keys()}
        row_data["folder_name"] = folder_name
        row_data["NumberofSlices"] = "缺失"
        
        if os.path.isdir(dicom_folder_path):
            # 🌟 使用全新的、极其严苛的 3D 切片寻址器
            dcm_dataset, slice_count = find_real_3d_slice_and_count(dicom_folder_path)
            
            if dcm_dataset is not None:
                row_data["NumberofSlices"] = slice_count
                
                # 遍历并提取字段
                for csv_col, dcm_keyword in DICOM_KEYWORD_MAP.items():
                    try:
                        # 核心提取
                        val = getattr(dcm_dataset, dcm_keyword, None)
                        
                        # 针对 ZSpacing (层厚) 的双重保险：有时写在 SpacingBetweenSlices 里
                        if csv_col == "ZSpacing" and val is None:
                            val = getattr(dcm_dataset, "SpacingBetweenSlices", None)
                            
                        row_data[csv_col] = format_dicom_value(val)
                    except Exception:
                        pass
                        
        metadata_records.append(row_data)

    df_meta = pd.DataFrame(metadata_records)
    
    print("\n⏳ 正在与图文对齐索引合并...")
    df_final = df_reports[['VolumeName', 'folder_name']].merge(df_meta, on='folder_name', how='left')
    df_final = df_final.drop(columns=['folder_name'])
    
    df_final.to_csv(META_CSV_OUTPUT, index=False, encoding='utf-8-sig')
    print("="*50)
    print(f"🎉 严谨版 DICOM Metadata 提取完毕！最关键的物理字段已全部就位。")
    print(f"📄 数据已保存至: {META_CSV_OUTPUT}")

if __name__ == "__main__":
    main()

# import pydicom
# # 随便指定一个你认为绝对是 CT 切片的文件路径（不管有没有后缀）
# ds = pydicom.dcmread("/mnt/share_data/CT/ct_dataset_base_260316/lz2nodesk_ct_chest_1000/90171767702100071/1.2.840.113619.2.278.3.2831188266.655.1754469756.552.277.dcm", stop_before_pixels=True)
# print(ds)