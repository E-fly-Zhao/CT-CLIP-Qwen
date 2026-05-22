import pandas as pd
import os

# 把你需要修复的 CSV 路径都列在这里
CSV_PATHS = [
    "/home/huali/code/CT-CLIP-main/CT_CLIP/dataset/train_reports.csv",
    "/home/huali/code/CT-CLIP-main/CT_CLIP/dataset/valid_reports.csv",
    "/home/huali/code/CT-CLIP-main/CT_CLIP/dataset/train_metadata.csv",
    "/home/huali/code/CT-CLIP-main/CT_CLIP/dataset/valid_metadata.csv"
]

for path in CSV_PATHS:
    if os.path.exists(path):
        print(f"⏳ 正在修复: {path}")
        df = pd.read_csv(path)
        
        if 'VolumeName' in df.columns:
            # 核心逻辑：以 '/' 分割字符串，并只取最后一部分（即文件名）
            df['VolumeName'] = df['VolumeName'].apply(lambda x: str(x).split('/')[-1])
            
            # 保存覆盖原文件
            df.to_csv(path, index=False, encoding='utf-8-sig')
            print(f"   ✅ 修复成功！\n")
    else:
        print(f"   ⚠️ 未找到文件 (跳过): {path}\n")