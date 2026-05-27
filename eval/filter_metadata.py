import os
import pandas as pd

def main():
    # 1. 路径配置
    txt_file = "/home/huali/code/CT-CLIP-main/eval/longest_2000.txt"
    csv_file = "/oss/share_data/CT/ct_dataset_eval_260514/train_metadata.csv"
    output_csv = "/home/huali/code/CT-CLIP-main/CT-CLIP/dataset_eval_2000/metadata_1999.csv"

    print("🚀 启动元数据 (Metadata) 筛选管线...")

    # 2. 读取 TXT 文件中的合法路径
    try:
        with open(txt_file, 'r', encoding='utf-8') as f:
            valid_paths = [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"❌ 找不到 TXT 文件: {txt_file}")
        return

    # 🚨 核心逻辑：提取带后缀和不带后缀的文件名，应对 CSV 存储格式不统一的问题
    valid_filenames_with_ext = set([os.path.basename(p) for p in valid_paths])
    valid_filenames_no_ext = set([os.path.basename(p).replace(".nii.gz", "").replace(".nii", "") for p in valid_paths])

    print(f"📂 成功从 TXT 解析了 {len(valid_paths)} 个目标文件名。")

    # 3. 读取完整的 CSV 文件
    try:
        df = pd.read_csv(csv_file, dtype=str, low_memory=False)
    except FileNotFoundError:
        print(f"❌ 找不到 CSV 文件: {csv_file}")
        return

    print(f"📖 原始元数据表共有 {len(df)} 条记录，开始精准匹配 VolumeName...")

    if 'VolumeName' not in df.columns:
        print("❌ 严重错误: CSV 中不存在 'VolumeName' 列！请检查文件格式。")
        return

    # 4. 定义匹配逻辑
    def is_valid_match(vol_name):
        if pd.isna(vol_name):
            return False
        name_str = str(vol_name).strip()
        # 只要带后缀的或去后缀的名字命中其一，即视为成功匹配
        return (name_str in valid_filenames_with_ext) or (name_str in valid_filenames_no_ext)

    # 5. 执行筛选
    filtered_df = df[df['VolumeName'].apply(is_valid_match)]

    # 6. 保存结果
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    filtered_df.to_csv(output_csv, index=False, encoding='utf-8-sig')

    print("\n" + "="*50)
    print(f"✅ 元数据筛选大获全胜！")
    print(f"📉 过滤前数据量: {len(df)} 条")
    print(f"📈 过滤后数据量: {len(filtered_df)} 条 (预期应该等于你 TXT 中有效文件的行数)")
    print(f"💾 纯净版元数据已保存至: {output_csv}")
    print("="*50)

if __name__ == "__main__":
    main()
