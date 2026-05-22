import pandas as pd
import os

def analyze_label_frequency(csv_path):
    print(f"📂 正在读取文件: {csv_path}")
    
    # 读取 CSV
    # 注意：如果文件中存在换行导致的格式问题，engine='python' 会更健壮
    df = pd.read_csv(csv_path)
    
    # 提取标签列（跳过第一列 ID）
    label_columns = df.columns[1:]
    
    # 计算频次 (sum) 和频率 (mean)
    # sum: 因为标签是0/1，求和即为出现次数
    # mean: 求均值即为出现频率
    freq_counts = df[label_columns].sum()
    freq_rates = df[label_columns].mean()
    
    # 合并成一个新的 DataFrame
    stats_df = pd.DataFrame({
        'Frequency': freq_counts,
        'Rate': freq_rates
    })
    
    # 按频次降序排列，方便查看哪些病最常见
    stats_df = stats_df.sort_values(by='Frequency', ascending=False)
    
    # 输出结果
    print("\n" + "="*50)
    print("📊 标签统计结果 (前 20 个最常见疾病):")
    print("="*50)
    print(stats_df.head(20).to_string())
    print("="*50)
    
    # 保存结果到 Excel
    output_path = "label_statistics.xlsx"
    stats_df.to_excel(output_path)
    print(f"💾 完整统计结果已保存至: {output_path}")

if __name__ == "__main__":
    file_path = "/mnt/huali/ct_dataset_10000/labeled_10000_label_matrix.csv"
    if os.path.exists(file_path):
        analyze_label_frequency(file_path)
    else:
        print(f"❌ 找不到文件: {file_path}")
