import pydicom

# 配置文件路径
# dcm_path = "/mnt/share_data/CT/ct_dataset_base_260316/lz2nodesk_ct_chest_1000/10050302800015/1.2.840.113704.9.1000.16.2.20220215115157185000100010001_02800015_100503_462847_33512492"
dcm_path = "/mnt/share_data/CT/ct_dataset_base_260316/lz2nodesk_ct_chest_1000/10050302800015/1.2.840.113704.9.1000.16.2.20220215115447233000200030001_02800015_100503_462849_33512533"


def main():
    print(f"正在解析 DICOM 文件: {dcm_path.split('/')[-1]}")
    try:
        # force=True 用于强制读取没有标准 .dcm 后缀的原始文件
        ds = pydicom.dcmread(dcm_path, force=True)

        print("\n" + "🚀" * 3 + " 核心图像尺寸与物理信息 " + "🚀" * 3)
        print("-" * 50)

        # 使用 .get() 防止某些非标准文件缺失对应 Tag 导致报错
        rows = ds.get('Rows', '未知')
        cols = ds.get('Columns', '未知')
        pixel_spacing = ds.get('PixelSpacing', ['未知', '未知'])
        slice_thickness = ds.get('SliceThickness', '未知')
        modality = ds.get('Modality', '未知')
        patient_id = ds.get('PatientID', '未知')

        print(f"🔍 检查模态 (Modality): {modality}")
        print(f"👤 患者 ID (PatientID): {patient_id}")
        print(f"📏 XY平面像素尺寸 (Rows x Cols): {rows} x {cols}")
        if isinstance(pixel_spacing, list) or hasattr(pixel_spacing, '__iter__'):
            print(f"📐 物理像素间距 (Pixel Spacing): {pixel_spacing[0]} mm x {pixel_spacing[1]} mm")
        else:
            print(f"📐 物理像素间距 (Pixel Spacing): {pixel_spacing}")
        print(f"🍔 扫描层厚 (Slice Thickness): {slice_thickness} mm")

        print("\n" + "📋" * 3 + " 完整 Metadata 信息 (已自动折叠像素矩阵) " + "📋" * 3)
        print("-" * 50)
        # pydicom 打印 dataset 时会自动省略庞大的 PixelData 数组，只展示元数据，不会刷屏
        print(ds)

    except Exception as e:
        print(f"❌ 解析 DICOM 失败: {e}")
        print("提示：文件可能损坏，或者完全不是医学图像格式（如单纯的文本配置）。")


if __name__ == "__main__":
    main()