import os
import pandas as pd
from tqdm import tqdm

def main():
    # ==========================================
    # 1. 路径配置
    # ==========================================
    source_excel = "/home/huali/code/CT-CLIP-main/full_reports_body.xlsx"
    target_csv = "/home/huali/code/CT-CLIP-main/CT_CLIP/dataset_10000/filtered_valid_reports.csv"
    
    # 强烈建议存为一个新的文件名，以防万一需要回滚
    output_csv = "/home/huali/code/CT-CLIP-main/CT_CLIP/dataset_10000/filtered_valid_reports_with_body.csv"

    # ==========================================
    # 2. 读取源数据并构建部位映射字典
    # ==========================================
    print(f"📥 正在加载带部位信息的源数据 (Excel): {source_excel}")
    try:
        df_body = pd.read_excel(source_excel, dtype=str)
    except Exception as e:
        print(f"❌ 读取 Excel 失败: {e}")
        return

    if 'dicom_path' not in df_body.columns or '部位' not in df_body.columns:
        print("❌ 错误：Excel文件中缺少 'dicom_path' 或 '部位' 列！")
        return

    print("🔍 正在构建 [文件夹名称 -> 部位] 的映射字典...")
    
    # 提取 dicom_path 最后一级路径作为字典的 key
    def get_last_folder(path):
        if pd.isna(path): return ""
        # 移除两端空白以及路径末尾可能带有的 '/'
        clean_path = str(path).strip().rstrip('/')
        return os.path.basename(clean_path)

    body_mapping = {}
    for _, row in tqdm(df_body.iterrows(), total=len(df_body), desc="提取部位映射"):
        folder_key = get_last_folder(row['dicom_path'])
        body_part = row['部位']
        
        if not folder_key:
            continue
            
        # 清洗空值
        if pd.isna(body_part) or str(body_part).strip().lower() in {'nan', 'none', 'null', ''}:
            body_part = ""
        else:
            body_part = str(body_part).strip()

        # 防止同一个病人有多行数据时，有内容的部位被空部位覆盖
        if folder_key in body_mapping and body_mapping[folder_key] and not body_part:
            continue
            
        body_mapping[folder_key] = body_part

    print(f"✅ 成功提取了 {len(body_mapping)} 个唯一患者/检查的部位信息。")

    # ==========================================
    # 3. 读取目标 CSV 并进行填补
    # ==========================================
    print(f"\n📥 正在加载需要补充部位列的训练数据 (CSV): {target_csv}")
    try:
        df_target = pd.read_csv(target_csv, dtype=str)
    except Exception as e:
        print(f"❌ 读取 CSV 失败: {e}")
        return

    if 'SourceFolder' not in df_target.columns:
        print("❌ 错误：CSV文件中缺少 'SourceFolder' 列！")
        return

    print("🧩 开始精准匹配并填充部位信息...")
    
    def map_body(folder_name):
        folder_str = str(folder_name).strip()
        part = body_mapping.get(folder_str, "")
        # 如果从 Excel 中实在匹配不到，或者部位为空，兜底填为“其他”
        return part if part else "其他"

    # 使用 tqdm 应用映射逻辑
    tqdm.pandas(desc="填充训练集部位")
    df_target['部位'] = df_target['SourceFolder'].progress_apply(map_body)

    # 调整一下列顺序，让 '部位' 列排到我们习惯的地方（放在最后面）
    cols = list(df_target.columns)
    if '部位' in cols:
        cols.remove('部位')
    cols.append('部位')
    df_target = df_target[cols]

    # ==========================================
    # 4. 保存与核对
    # ==========================================
    print(f"\n💾 正在保存补充完成的数据至: {output_csv}")
    df_target.to_csv(output_csv, index=False, encoding='utf-8-sig')
    
    # 打印一些统计数据方便法医级核对
    matched_count = len(df_target[df_target['部位'] != '其他'])
    unmatched_count = len(df_target) - matched_count
    
    print("\n" + "="*50)
    print("🎉 匹配填充大获全胜！")
    print(f"📊 目标训练集总行数: {len(df_target)}")
    print(f"✅ 成功匹配并填入部位的行数: {matched_count}")
    print(f"⚠️ 未匹配到或无部位数据的行数(自动设为'其他'): {unmatched_count}")
    print("="*50)

if __name__ == "__main__":
    main()
