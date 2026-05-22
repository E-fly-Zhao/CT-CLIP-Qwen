import pandas as pd
import os

def filter_and_evaluate(stats_path, metrics_path, output_path, freq_threshold=100):
    print(f"📂 正在读取预训练频次统计: {stats_path}")
    if not os.path.exists(stats_path):
        print(f"❌ 找不到统计文件: {stats_path}")
        return
        
    # 1. 读取频次统计文件 (假设第一列或索引是疾病名称)
    stats_df = pd.read_excel(stats_path, index_col=0)
    
    # 2. 筛选频次大于阈值 (n) 的疾病
    valid_diseases = stats_df[stats_df['Frequency'] > freq_threshold].index.tolist()
    print(f"🔍 发现 {len(valid_diseases)} 种在预训练数据集中频次大于 {freq_threshold} 的疾病。")
    
    print(f"📂 正在读取 2000例 测评结果: {metrics_path}")
    if not os.path.exists(metrics_path):
        print(f"❌ 找不到测评文件: {metrics_path}")
        return
        
    metrics_df = pd.read_excel(metrics_path)
    
    # 过滤掉原测评表底部可能存在的 "Average (Valid Only)" 统计行
    metrics_df = metrics_df[~metrics_df['Pathology'].astype(str).str.contains('Average', na=False)]
    
    # 3. 在测评结果中筛选出属于 valid_diseases 的行
    filtered_metrics = metrics_df[metrics_df['Pathology'].isin(valid_diseases)].copy()
    print(f" 在测评结果中，成功匹配到 {len(filtered_metrics)} 种符合条件的疾病。")
    
    if filtered_metrics.empty:
        print("⚠️ 警告：未匹配到任何测评结果，请检查两份表中的疾病名称格式是否完全一致！")
        return
        
    # 4. 计算筛选后的平均指标
    # 将所有的 "N/A" 或非数字字符强制转为 NaN，pandas 在计算 .mean() 时会自动忽略 NaN
    numeric_cols = filtered_metrics.columns.drop('Pathology')
    numeric_df = filtered_metrics[numeric_cols].apply(pd.to_numeric, errors='coerce')
    mean_metrics = numeric_df.mean()
    
    # 5. 打印终端战报
    print("\n" + "="*55)
    print(f"高频疾病 (频次 > {freq_threshold}) 测评 (宏平均 Macro-Avg):")
    print("="*55)
    for col, val in mean_metrics.items():
        if pd.notna(val):
            print(f"    {col:<10}: {val:.4f}")
        else:
            print(f"    {col:<10}: N/A")
    print("="*55)
    
    # 6. 保存新的筛选结果 (在最后一行追加均值，方便用 Excel 直接查看)
    avg_row = pd.DataFrame([['Average (Freq > 100)'] + mean_metrics.tolist()], 
    #avg_row = pd.DataFrame([['Average (Freq > 500)'] + mean_metrics.tolist()],
                           columns=['Pathology'] + numeric_cols.tolist())
    final_output_df = pd.concat([filtered_metrics, avg_row], ignore_index=True)
    
    final_output_df.to_excel(output_path, index=False)
    print(f"💾 筛选后的详细报表已保存至: {output_path}")

if __name__ == "__main__":
    # 路径配置
    STATS_FILE = "/home/huali/code/CT-CLIP-main/label_statistics.xlsx"
    METRICS_FILE = "/home/huali/code/CT-CLIP-main/qwen_zeroshot_2000/qwen_zeroshot_full_metrics.xlsx"
    OUTPUT_FILE = "/home/huali/code/CT-CLIP-main/qwen_zeroshot_2000/filtered_metrics_freq_gt_100.xlsx"
    #OUTPUT_FILE = "/home/huali/code/CT-CLIP-main/qwen_zeroshot_2000/filtered_metrics_freq_gt_500.xlsx"
    
    # 执行筛选 (频次大于 n)
    filter_and_evaluate(STATS_FILE, METRICS_FILE, OUTPUT_FILE, freq_threshold=100)
