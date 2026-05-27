import os
import pandas as pd

def main():
    # 1. 路径配置
    txt_file = "/home/huali/code/CT-CLIP-main/eval/longest_2000.txt"
    csv_file = "/home/huali/code/CT-CLIP-main/CT-CLIP/dataset_eval_2000/reports_full.csv"
    output_csv = "/home/huali/code/CT-CLIP-main/CT-CLIP/dataset_eval_2000/reports_longest.csv"

    print("🚀 启动高质量报告筛选管线...")

    # 2. 读取 TXT 文件中的合法路径
    try:
        with open(txt_file, 'r', encoding='utf-8') as f:
            valid_paths = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"❌ 找不到 TXT 文件: {txt_file}")
        return

    # 提取特征集合，用于双重鲁棒匹配
    exact_paths = set(valid_paths)
    # 提取所在的文件夹名称 (如 '4034105740001')
    valid_folder_names = set([os.path.basename(os.path.dirname(p)) for p in valid_paths])

    print(f"📂 成功从 TXT 加载了 {len(valid_paths)} 个最长序列标识。")

    # 3. 读取完整的 CSV 文件
    try:
        df = pd.read_csv(csv_file, dtype=str)
    except FileNotFoundError:
        print(f"❌ 找不到 CSV 文件: {csv_file}")
        return

    print(f"📖 原始报告表共有 {len(df)} 条记录，开始跨表匹配...")

    # 4. 定义智能匹配逻辑
    def is_valid_match(dicom_path):
        if pd.isna(dicom_path):
            return False
        
        path_str = str(dicom_path).strip()
        
        # 策略 A: 绝对路径精确匹配
        if path_str in exact_paths:
            return True
            
        # 策略 B: 文件夹名称匹配 (解决 CSV 中可能只存了文件夹名或相对路径的问题)
        folder_name = os.path.basename(path_str)
        if folder_name in valid_folder_names:
            return True
            
        return False

    # 5. 执行筛选
    if 'dicom_path' not in df.columns:
        print("❌ 严重错误: CSV 中不存在 'dicom_path' 列！请检查文件格式。")
        return

    filtered_df = df[df['dicom_path'].apply(is_valid_match)]

    # 6. 保存结果
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    filtered_df.to_csv(output_csv, index=False, encoding='utf-8-sig')

    print("\n" + "="*50)
    print(f"✅ 筛选结束！")
    print(f"📉 过滤前数据量: {len(df)} 条")
    print(f"📈 过滤后数据量: {len(filtered_df)} 条")
    print(f"💾 纯净版报告已保存至: {output_csv}")
    print("="*50)

if __name__ == "__main__":
    main()
