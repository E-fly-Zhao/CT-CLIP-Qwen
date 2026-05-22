import os
import ast
import argparse
import numpy as np
import pandas as pd
import SimpleITK as sitk
from pathlib import Path
import concurrent.futures
from tqdm import tqdm

def process_row(row, file_map, output_dir):
    try:
        volume_name = row["VolumeName"]
        
        # 1. 检查文件是否存在于输入目录中
        if volume_name not in file_map:
            return False
            
        input_filepath = file_map[volume_name]
        output_filepath = os.path.join(output_dir, volume_name)

        # 2. 断点续传：如果已存在，跳过
        if os.path.exists(output_filepath):
            return True

        # 3. 读取图像
        image = sitk.ReadImage(input_filepath)

        # 4. 写入空间元数据 (Spacing, Origin, Direction)
        (x, y), z = map(float, ast.literal_eval(row["XYSpacing"])), float(row["ZSpacing"])
        image.SetSpacing((x, y, z))

        image.SetOrigin(ast.literal_eval(row["ImagePositionPatient"]))

        orientation = ast.literal_eval(row["ImageOrientationPatient"])
        row_cosine, col_cosine = orientation[:3], orientation[3:6]
        z_cosine = np.cross(row_cosine, col_cosine).tolist()
        image.SetDirection(row_cosine + col_cosine + z_cosine)

        # 5. 写入 HU 值线性转换 (HU = Pixel * Slope + Intercept)
        RescaleIntercept = float(row["RescaleIntercept"])
        RescaleSlope = float(row["RescaleSlope"])
        adjusted_hu = image * RescaleSlope + RescaleIntercept

        # 6. 强制转换为 int16 (标准医疗图像存储格式，节省空间)
        adjusted_hu = sitk.Cast(adjusted_hu, sitk.sitkInt16)

        # 7. 保存图像
        sitk.WriteImage(adjusted_hu, output_filepath)
        return True
        
    except Exception as e:
        print(f"\n[Error] 处理 {row.get('VolumeName', 'Unknown')} 失败: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CT 图像物理坐标修复与 HU 值校准脚本")
    
    parser.add_argument("--input_dir", required=True, type=str, help="原始 NIfTI 文件夹路径")
    parser.add_argument("--output_dir", required=True, type=str, help="校准后 NIfTI 文件夹路径")
    parser.add_argument("--metadata_csv", required=True, type=str, help="包含物理参数的 CSV 表格路径")
    parser.add_argument("--num_workers", default=16, type=int, help="多进程并发数")
    
    args = parser.parse_args()

    # 建立输出目录
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # 读取 CSV 并构建文件映射表，极大提升搜索速度
    print("📂 正在扫描输入目录文件...")
    file_map = {p.name: str(p) for p in Path(args.input_dir).rglob("*.nii.gz")}
    print(f"✅ 在输入目录中共找到 {len(file_map)} 个 .nii.gz 文件。")

    print(f"📊 正在读取元数据表格: {args.metadata_csv}")
    metadata = pd.read_csv(args.metadata_csv)
    
    # 将 DataFrame 转换为字典列表以便并发处理
    rows = [row for _, row in metadata.iterrows()]

    print(f"🚀 开始执行物理校验与 HU 转换，输出目录: {args.output_dir}")
    
    # 启动多进程
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        # 使用 partial 冻结固定的参数
        from functools import partial
        func = partial(process_row, file_map=file_map, output_dir=args.output_dir)
        
        list(tqdm(executor.map(func, rows), total=len(rows)))
        
    print("\n🎉 物理坐标修复与 HU 值校准完成！")
