import pandas as pd
import os

def main():
    # 定义文件路径
    reports_1000_path = "/home/huali/code/CT-CLIP-main/CT_CLIP/dataset/train_reports_1000.csv"
    metadata_path = "/home/huali/code/CT-CLIP-main/CT_CLIP/dataset/train_metadata.csv"
    output_reports_path = "/home/huali/code/CT-CLIP-main/CT_CLIP/dataset/train_reports.csv"

    print("🔍 正在加载 CSV 文件...")
    try:
        df_meta = pd.read_csv(metadata_path)
        df_reports = pd.read_csv(reports_1000_path)
    except FileNotFoundError as e:
        print(f"❌ 找不到文件: {e}")
        return

    # 1. 从 metadata 中提取出绝对合法的 VolumeName 白名单
    if 'VolumeName' not in df_meta.columns:
        print("❌ 错误：train_metadata.csv 中没有找到 'VolumeName' 列！")
        return
        
    valid_volumes = set(df_meta['VolumeName'].dropna().astype(str).str.strip().tolist())
    print(f"✅ 从 metadata 中提取了 {len(valid_volumes)} 个有效的 VolumeName。")
    print(f"📊 原始 reports 表共有 {len(df_reports)} 行数据。")

    # 2. 建立健壮的匹配逻辑（兼容带不带 .nii.gz 后缀的情况）
    def is_valid_volume(vol_name):
        vol_str = str(vol_name).strip()
        # 精确匹配
        if vol_str in valid_volumes:
            return True
        # 报告里没后缀，但 metadata 里有
        if f"{vol_str}.nii.gz" in valid_volumes:
            return True
        # 报告里有后缀，但 metadata 里没有
        if vol_str.replace(".nii.gz", "") in valid_volumes:
            return True
        return False

    # 3. 执行过滤
    if 'VolumeName' not in df_reports.columns:
        print("❌ 错误：train_reports_1000.csv 中没有找到 'VolumeName' 列！")
        return

    filtered_reports = df_reports[df_reports['VolumeName'].apply(is_valid_volume)]

    # 4. 保存为新的干净的 reports 文件
    filtered_reports.to_csv(output_reports_path, index=False)
    
    print("\n🎉 清洗完成！")
    print(f"✅ 成功过滤出 {len(filtered_reports)} 行精准匹配的报告数据。")
    print(f"💾 新的报告表已保存至: {output_reports_path}")

if __name__ == "__main__":
    main()
