import os
import argparse
from pathlib import Path
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import torch
from monai import transforms

def process_image(loader, out_dir, img_path):
    """
    单个图像的处理流水线函数
    """
    try:
        # 提取文件名
        filename = Path(img_path).name
        out_file = Path(out_dir) / filename
        
        # 断点续传机制：如果输出文件夹中已经存在该文件，则跳过
        if out_file.exists():
            return True
        
        # 组装输入字典并传入 MONAI 流水线
        data = {"image": img_path}
        data = loader(data)
        
        return True
    
    except Exception as e:
        print(f"\n[Error] Failed to process {img_path}: {e}")
        return None

if __name__ == "__main__":
    # ==========================================
    # 1. 命令行参数解析
    # ==========================================
    parser = argparse.ArgumentParser(description="CT-Qwen-CLIP 数据极简预处理流水线")
    
    parser.add_argument("--input_dir", required=True, type=str, 
                        help="包含原始 NIfTI (.nii.gz) 文件的输入文件夹路径")
    parser.add_argument("--output_dir", required=True, type=str, 
                        help="处理后张量文件的输出保存路径")
    parser.add_argument("--num_workers", default=16, type=int, 
                        help="多进程并发数 (请根据 CPU 核心数和硬盘 I/O 速度调整)")
    
    args = parser.parse_args()

    # 确保输出目录存在
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # 递归查找输入目录下的所有 .nii.gz 文件
    img_paths = [str(p) for p in Path(args.input_dir).rglob("*.nii.gz")]
    print(f"🔍 在 {args.input_dir} 中共发现 {len(img_paths)} 个待处理文件。")
    print(f"🚀 开始预处理，输出目录: {args.output_dir}")

    # ==========================================
    # 2. MONAI 核心处理流水线
    # ==========================================
    loader = transforms.Compose([
        # 1. 加载图像并确保 Channel First (C, H, W, D)
        transforms.LoadImaged(keys=["image"], image_only=False, ensure_channel_first=True),
        
        # 2. 空间重采样 (统一物理间距至 1.0 x 1.0 x 1.5 mm)
        transforms.Spacingd(
            keys=["image"], 
            pixdim=(1.0, 1.0, 1.5), 
            mode="bilinear"
        ),
        
        # 3. 智能身体轮廓裁剪 (自动去除 HU < -500 的外围空气)
        transforms.CropForegroundd(
            keys=["image"],
            source_key="image",
            select_fn=lambda x: x > -500,
            margin=5  # 安全边距
        ),
        
        # 4. 医疗窗宽窗位截断与归一化 (将 HU 值截断到 [-1000, 1000] 并映射到 [0, 1])
        transforms.ScaleIntensityRanged(
            keys=["image"], 
            a_min=-1000, a_max=1000, 
            b_min=0.0, b_max=1.0, 
            clip=True
        ),

        # 5. 维度转置 (适配 CTViT 的特殊排列需求)
        transforms.Transposed(keys=["image"], indices=(0, 3, 2, 1)),
        
        # 6. 强制统一下游模型输入尺寸 (480, 480, 240) -> 对应 D, H, W 排列
        transforms.Resized(
            keys=["image"],
            spatial_size=(240, 480, 480), 
            mode="trilinear"
        ),

        # 7. 存储结果，禁用单独建文件夹，保持扁平化输出
        transforms.SaveImaged(
            output_dir=args.output_dir,
            keys=["image"],
            output_postfix="",
            separate_folder=False,
            resample=False,
        )
    ])

    # ==========================================
    # 3. 启动多进程并发处理
    # ==========================================
    func = partial(process_image, loader, args.output_dir)
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        list(tqdm(executor.map(func, img_paths), total=len(img_paths)))
    
    print("\n🎉 所有数据预处理完成！")
