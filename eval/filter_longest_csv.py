import os
import pandas as pd

def main():
    # ==========================================
    # 1. 路径配置
    # ==========================================
    #txt_path = "/home/huali/code/CT-CLIP/eval/longest.txt"
    txt_path = "/home/huali/code/CT-CLIP/eval/longest_chest.txt"
    
    meta_csv_path = "/oss/share_data/CT/ct_dataset_eval_260514/train_metadata.csv"
    reports_csv_path = "/oss/share_data/CT/ct_dataset_eval_260514/train_reports.csv"
    
    out_meta_path = "/oss/share_data/CT/ct_dataset_eval_260514/train_metadata_longest_chest.csv"
    out_reports_path = "/oss/share_data/CT/ct_dataset_eval_260514/train_reports_longest_chest.csv"

    # ==========================================
    # 2. 从 txt 提取需要保留的 VolumeName
    # ==========================================
    print(f"[*] 正在读取 txt 文件: {txt_path}")
    if not os.path.exists(txt_path):
        print(f"❌ 错误：找不到文件 {txt_path}")
        return

    with open(txt_path, 'r', encoding='utf-8') as f:
        paths = [line.strip() for line in f.readlines() if line.strip()]
        
    valid_volume_names = set()
    for p in paths:
        # 提取最后一级文件名，例如 12345.nii.gz
        basename = os.path.basename(p)
        valid_volume_names.add(basename)
        # 🚨 兼容性处理：如果 CSV 中的 VolumeName 没有 .nii.gz 后缀，也能完美匹配
        valid_volume_names.add(basename.replace('.nii.gz', ''))
        
    print(f"[*] 成功提取了 {len(paths)} 个有效的最长序列文件名作为筛选基准。")

    # ==========================================
    # 3. 筛选 train_metadata.csv
    # ==========================================
    print(f"\n[*] 正在加载并筛选 Metadata CSV: {meta_csv_path}")
    try:
        # 🚨 dtype=str 极其重要，防止长数字 ID 被 pandas 自动转成科学计数法导致匹配失败
        meta_df = pd.read_csv(meta_csv_path, dtype=str)
        if 'VolumeName' not in meta_df.columns:
            print("❌ 错误：Metadata CSV 中找不到 'VolumeName' 列！")
        else:
            meta_filtered = meta_df[meta_df['VolumeName'].isin(valid_volume_names)]
            meta_filtered.to_csv(out_meta_path, index=False, encoding='utf-8-sig')
            print(f"✅ Metadata 筛选完成！( {len(meta_df)} 行 -> 缩减为 {len(meta_filtered)} 行 )")
            print(f"💾 已保存至: {out_meta_path}")
    except Exception as e:
        print(f"❌ 处理 Metadata 时出错: {e}")

    # ==========================================
    # 4. 筛选 train_reports.csv
    # ==========================================
    print(f"\n[*] 正在加载并筛选 Reports CSV: {reports_csv_path}")
    try:
        reports_df = pd.read_csv(reports_csv_path, dtype=str)
        if 'VolumeName' not in reports_df.columns:
            print("❌ 错误：Reports CSV 中找不到 'VolumeName' 列！")
        else:
            reports_filtered = reports_df[reports_df['VolumeName'].isin(valid_volume_names)]
            # 🚨 utf-8-sig 保证中文报告随后如果在 Windows Excel 中打开时绝不乱码
            reports_filtered.to_csv(out_reports_path, index=False, encoding='utf-8-sig')
            print(f"✅ Reports 筛选完成！( {len(reports_df)} 行 -> 缩减为 {len(reports_filtered)} 行 )")
            print(f"💾 已保存至: {out_reports_path}")
    except Exception as e:
        print(f"❌ 处理 Reports 时出错: {e}")

    print("\n" + "="*60)
    print("🎉 所有表格数据对齐与筛选圆满完成！")
    print("="*60)

if __name__ == "__main__":
    main()
