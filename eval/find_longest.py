import argparse
import glob
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
from tqdm import tqdm

# ==========================================
# 默认路径配置
# ==========================================
DEFAULT_DATA_DIR = "/oss/share_data/CT/ct_dataset_eval_260514/ct_dataset_eval_260514_img/"
DEFAULT_OUTPUT_FILE = "/home/huali/code/CT-CLIP/eval/longest.txt"
DEFAULT_MAX_WORKERS = 16  # 默认开启 16 线程加速 I/O 读取


def parse_args():
    parser = argparse.ArgumentParser(description="Find the longest NIfTI sequence in each patient folder.")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="患者数据的根目录")
    parser.add_argument("--output-file", default=DEFAULT_OUTPUT_FILE, help="保存最长路径的 txt 文件")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS, help="并发读取的线程数")
    return parser.parse_args()


def process_patient_folder(patient_dir):
    """
    处理单个患者文件夹，找出 Z 轴切片数最多的 .nii.gz 文件
    """
    nii_files = glob.glob(os.path.join(patient_dir, "*.nii.gz"))
    
    max_slices = -1
    longest_file = None

    for nii_path in nii_files:
        filename = os.path.basename(nii_path).lower()

        # 🚨 防错机制：严禁将掩膜 (Mask/ROI/Label) 当作原图
        if "roi" in filename or "mask" in filename or "label" in filename:
            continue

        try:
            # 💡 核心优化：nib.load 默认采用内存映射 (mmap) 且只读 Header，不加载庞大的图像矩阵，速度极快！
            img = nib.load(nii_path)
            shape = img.header.get_data_shape()

            # 确保是 3D 或以上的数据 (X, Y, Z)
            if len(shape) < 3:
                continue

            num_slices = shape[2]  # Z 轴切片数
            
            # 更新最大切片数与对应文件
            if num_slices > max_slices:
                max_slices = num_slices
                longest_file = nii_path

        except Exception as e:
            # 捕获损坏的 NIfTI 文件或无权限读取等错误，保证进程不中断
            pass

    return longest_file


def main():
    args = parse_args()

    # 1. 检查根目录
    if not os.path.exists(args.data_dir):
        print(f"❌ 错误：找不到数据根目录 {args.data_dir}")
        return

    # 2. 收集所有患者（子文件夹）路径
    patient_dirs = [
        os.path.join(args.data_dir, d) for d in os.listdir(args.data_dir)
        if os.path.isdir(os.path.join(args.data_dir, d))
    ]

    print(f"[*] 发现 {len(patient_dirs)} 个患者子文件夹。")
    print(f"[*] 启动高通量读取，分配 {args.max_workers} 个工作线程...\n")

    longest_files = []
    
    # 3. 启动多线程并发提取 (复刻 fenlei.py 的高性能范式)
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # 将所有文件夹抛入线程池
        futures = {executor.submit(process_patient_folder, pdir): pdir for pdir in patient_dirs}

        # 使用 tqdm 实时跟踪完成进度
        for future in tqdm(as_completed(futures), total=len(futures), desc="Scanning NIfTI Headers"):
            result = future.result()
            if result is not None:
                longest_files.append(result)

    # 4. 排序结果（可选，为了让 txt 文件看起来更整洁，可以按路径字典序排序）
    longest_files.sort()

    # 5. 确保输出目录存在，并写入结果
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for file_path in longest_files:
            f.write(f"{file_path}\n")

    print("\n" + "="*60)
    print("🎉 筛选大获全胜！")
    print(f"✅ 成功提取了 {len(longest_files)} 个最长序列的绝对路径。")
    print(f"💾 结果已安全保存至: {args.output_file}")
    print("="*60)


if __name__ == "__main__":
    main()
