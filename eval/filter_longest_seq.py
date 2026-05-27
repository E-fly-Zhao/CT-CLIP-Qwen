import os
import glob
import nibabel as nib
from tqdm import tqdm

def main():
    # 1. 路径配置
    data_dir = "/oss/share_data/CT/ct_dataset_eval_260514/ct_dataset_eval_260514_img/"
    output_file = "../eval/longest_2000.txt"

    # 确保输出的上一级目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 2. 获取所有的患者子文件夹
    subfolders = [f.path for f in os.scandir(data_dir) if f.is_dir()]
    print(f"🔍 扫描到 {len(subfolders)} 个患者文件夹。开始筛选最长序列...")

    longest_files = []

    # 3. 遍历每个患者文件夹
    for folder in tqdm(subfolders, desc="Processing Patients"):
        nii_files = glob.glob(os.path.join(folder, "*.nii.gz"))
        
        if not nii_files:
            continue

        max_slices = -1
        best_file = None

        # 遍历当前患者的所有序列文件
        for nii_file in nii_files:
            try:
                # 🚨 极速优化：仅读取 Header 获取形状，绝不加载图像矩阵！
                img_header = nib.load(nii_file)
                shape = img_header.shape
                
                # NIfTI 格式的标准 Shape 通常是 (X, Y, Z) 或 (X, Y, Z, T)
                # 第三维 (index 2) 即代表 Z 轴（切片层数）
                if len(shape) >= 3:
                    num_slices = shape[2]
                else:
                    num_slices = shape[-1] # 极端异常 2D 数据的保底
                    
                # 寻找最大切片数
                if num_slices > max_slices:
                    max_slices = num_slices
                    best_file = nii_file
                    
            except Exception as e:
                print(f"\n⚠️ 警告: 读取文件头失败跳过 {nii_file} | 报错: {e}")
                continue
        
        # 记录该患者的最长序列
        if best_file:
            longest_files.append(best_file)

    # 4. 保存结果
    print(f"\n✅ 筛选完成！共提取出 {len(longest_files)} 个唯一序列（最长切片）。")
    print(f"💾 正在将绝对路径写入文件...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for file_path in longest_files:
            f.write(file_path + "\n")
            
    print(f"🎉 大功告成！结果已成功保存至: {output_file}")

if __name__ == "__main__":
    main()
