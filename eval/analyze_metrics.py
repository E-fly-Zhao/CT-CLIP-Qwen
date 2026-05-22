import pandas as pd
import numpy as np

def extract_and_analyze(excel_path):
    # 1. 定义你关注的 10 种疾病 (确保与 Excel 中的表头名完全一致)
    target_pathologies = [
        "肺气肿", "肺不张", "肺实变", "肺结节", "支气管扩张", 
        "胸腔积液", "心包积液", "主动脉硬化", "磨玻璃影", "Kerley B线"
    ]
    
    try:
        # 2. 读取 Excel
        df = pd.read_excel(excel_path)
        
        # 假设第一列是 'Pathology'，我们需要将其设置为索引以便于检索
        # 如果你的 Excel 结构不同，请微调此处的逻辑
        if 'Pathology' not in df.columns:
            # 如果没有 'Pathology' 列，尝试将第一列设为索引
            df.set_index(df.columns[0], inplace=True)
        else:
            df.set_index('Pathology', inplace=True)
            
        print(f"📂 成功读取文件: {excel_path}")
        
        # 3. 提取目标疾病的行
        extracted_data = df.loc[df.index.intersection(target_pathologies)]
        
        if extracted_data.empty:
            print("⚠️ 警告: 在 Excel 中未找到任何目标疾病，请检查名称是否对应！")
            return

        # 4. 计算每种指标的平均值 (遇到 'N/A' 会自动排除)
        # 将所有非数值数据转为 NaN 以便统计
        numeric_df = extracted_data.apply(pd.to_numeric, errors='coerce')
        
        # 计算 10 种疾病在 AUROC, F1, Accuracy 等指标上的均值
        mean_metrics = numeric_df.mean()
        
        # 5. 打印报表
        print("\n" + "="*50)
        print("🎯 目标疾病 (Top 10) 指标报表:")
        print("-" * 50)
        print(extracted_data)
        print("="*50)
        print("📊 目标疾病均值 (Mean of Top 10):")
        print("-" * 50)
        print(mean_metrics.to_string())
        print("="*50)
        
        # 导出结果
        output_name = "top10_diseases_metrics.xlsx"
        extracted_data.to_excel(output_name)
        print(f"💾 提取结果已保存至: {output_name}")

    except Exception as e:
        print(f"❌ 处理出错: {e}")

if __name__ == "__main__":
    # 在此处填入你的 Excel 文件路径
    path = "/home/huali/code/CT-CLIP-main/qwen_zeroshot_2000/qwen_zeroshot_full_metrics.xlsx"
    extract_and_analyze(path)
