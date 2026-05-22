import os
import glob
import pandas as pd

# 定义目标路径
base_dir = "/data1/sft/ct_workspace/"
train_data_dir = os.path.join(base_dir, "processed/train_data/")

def count_text_lines(filepath):
    """高效统计纯文本文件（CSV/TXT）的行数"""
    count = 0
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for _ in f:
                count += 1
    except UnicodeDecodeError:
        # 如果 utf-8 失败，尝试用 gbk 读取（兼容 Windows 传过来的文件）
        with open(filepath, 'r', encoding='gbk', errors='ignore') as f:
            for _ in f:
                count += 1
    return count

def main():
    print("="*40)
    print("开始执行统计任务...")
    print("="*40)

    # 1. 统计所有的 CSV 文件
    print("\n[1] 正在统计 CSV 文件行数 (包含表头):")
    csv_files = glob.glob(os.path.join(base_dir, "*.csv"))
    if not csv_files:
        print("  -> 未找到任何 .csv 文件")
    for f in csv_files:
        lines = count_text_lines(f)
        print(f"  - {os.path.basename(f)}: {lines} 行")

    # 2. 统计所有的 Excel 文件
    print("\n[2] 正在统计 Excel 文件数据量 (不包含表头):")
    excel_files = glob.glob(os.path.join(base_dir, "*.xls*")) # 匹配 .xlsx 和 .xls
    if not excel_files:
        print("  -> 未找到任何 Excel 文件")
    for f in excel_files:
        try:
            # 读取 Excel 并获取行数
            df = pd.read_excel(f)
            print(f"  - {os.path.basename(f)}: {len(df)} 条数据")
        except Exception as e:
            print(f"  - {os.path.basename(f)}: 读取失败，错误信息: {e}")

    # 3. 统计所有的 TXT 文件
    print("\n[3] 正在统计 TXT 文档行数:")
    txt_files = glob.glob(os.path.join(base_dir, "*.txt"))
    if not txt_files:
        print("  -> 未找到任何 .txt 文件")
    for f in txt_files:
        lines = count_text_lines(f)
        print(f"  - {os.path.basename(f)}: {lines} 行")

    # 4. 统计 processed/train_data/ 下的文件数量 (递归统计所有子文件夹)
    print("\n[4] 正在统计 train_data 目录下的文件数量:")
    if os.path.exists(train_data_dir):
        file_count = 0
        folder_count = 0
        # 使用 os.walk 穿透所有子目录
        for root, dirs, files in os.walk(train_data_dir):
            file_count += len(files)
            folder_count += len(dirs)

        print(f"  - 目录 {train_data_dir} 下共有 {folder_count} 个子文件夹")
        print(f"  - 累计包含 {file_count} 个文件")
    else:
        print(f"  - 警告: 目录不存在 ({train_data_dir})")

    print("\n" + "="*40)
    print("统计完成！")

if __name__ == "__main__":
    main()
