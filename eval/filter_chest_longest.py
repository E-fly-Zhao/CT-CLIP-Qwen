import os
import pandas as pd

def main():
    # ==========================================
    # 1. 路径配置 (请核对绝对路径)
    # ==========================================
    txt_path = "/home/huali/code/CT-CLIP/eval/longest.txt"
    # 使用你最新确认的绝对路径
    excel_path = "/home/huali/code/CT-CLIP-main/eval/chest_eval.xlsx" 
    output_txt = "/home/huali/code/CT-CLIP/eval/longest_chest.txt"

    # ==========================================
    # 2. 读取 Excel 并从 dicom_path 提取真实文件夹名
    # ==========================================
    print(f"[*] 正在读取 Excel 筛选清单: {excel_path}")
    if not os.path.exists(excel_path):
        print(f"❌ 错误：找不到 Excel 文件 {excel_path}")
        return

    try:
        # dtype=str 防止任何可能的数字坍缩
        df = pd.read_excel(excel_path, dtype=str)
        if 'dicom_path' not in df.columns:
            print("❌ 错误：Excel 文件中找不到 'dicom_path' 列！请检查表头。")
            return
            
        valid_ids = set()
        for val in df['dicom_path'].dropna():
            # 🚨 核心逻辑：清理路径前后的空格和多余斜杠，并提取最后一级目录
            # 例如 "/oss/share_data/.../205117405740001/" -> "205117405740001"
            cleaned_path = str(val).strip().rstrip('/')
            folder_name = os.path.basename(cleaned_path)
            
            # 去除可能的 .0 后缀（以防万一 Excel 曾把它识别为数字）
            if folder_name.endswith('.0'):
                folder_name = folder_name[:-2]
                
            valid_ids.add(folder_name)
            
        print(f"[*] 成功从 Excel dicom_path 列中提取了 {len(valid_ids)} 个唯一的胸部文件夹 ID。")
        
        # 透视打印：展示前 3 个提取出来的基准 ID，供你核对
        sample_ids = list(valid_ids)[:3]
        print(f"🔍 [透视检查] 提取出的胸部文件夹 ID 示例: {sample_ids}")
        
    except Exception as e:
        print(f"❌ 读取 Excel 时发生错误: {e}")
        return

    # ==========================================
    # 3. 逐行读取 txt 并进行匹配
    # ==========================================
    print(f"\n[*] 正在读取全量序列路径: {txt_path}")
    matched_paths = []
    
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            path = line.strip()
            if not path:
                continue
                
            # 从 nii 路径中扒出倒数第二级文件夹名
            # 例: /oss/.../10003405330005/CT820064.nii.gz -> 10003405330005
            folder_id = os.path.basename(os.path.dirname(path))
            
            # 对撞匹配
            if folder_id in valid_ids:
                matched_paths.append(path)

    # ==========================================
    # 4. 保存结果
    # ==========================================
    print(f"\n[*] 匹配完成！共筛选出 {len(matched_paths)} 条符合条件的胸部序列路径。")
    
    if len(matched_paths) == 0:
        print("⚠️ 警告：匹配数依然为 0。请检查上方 [透视检查] 打印的 ID，是否与你 txt 里的父文件夹名一致。")
    else:
        os.makedirs(os.path.dirname(output_txt), exist_ok=True)
        with open(output_txt, 'w', encoding='utf-8') as out_f:
            for p in matched_paths:
                out_f.write(f"{p}\n")
                
        print("="*60)
        print("🎉 筛选圆满结束！")
        print(f"💾 结果已安全保存至: {output_txt}")
        print("="*60)

if __name__ == "__main__":
    main()
