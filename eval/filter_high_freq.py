import pandas as pd
import os

def extract_high_freq_data():
    # 1. 定义文件路径
    csv_target_path = "/home/huali/workspace/psj/evaluation_dataset/api/eval/labeled_eval_label_matrix.csv"
    xlsx_exclude_path = "/home/huali/workspace/xly/CTModel/data/labeled_eval_matrix_only_low_freq.xlsx"
    output_path = "/home/huali/code/CT-CLIP-main/eval/label_high_freq.csv"

    print("🚀 开始处理数据...")

    try:
        # 2. 读取“黑名单”数据（低频词 Excel），提取需要剔除的 _id
        print(f"📦 正在读取排除名单 (Excel): {os.path.basename(xlsx_exclude_path)}")
        df_exclude = pd.read_excel(xlsx_exclude_path)
        
        if '_id' not in df_exclude.columns:
            print("❌ 错误：在低频Excel文件中找不到 '_id' 列，请检查表头。")
            return
            
        # 提取所有的 _id，去重并剔除空值
        exclude_ids = df_exclude['_id'].dropna().unique()
        print(f"✅ 成功提取到 {len(exclude_ids)} 个需要剔除的低频目标 _id。")

        # 3. 读取总表数据（包含所有标签矩阵的 CSV 文件）
        print(f"📦 正在读取原始总表矩阵 (CSV): {os.path.basename(csv_target_path)}")
        df_matrix = pd.read_csv(csv_target_path)

        if '_id' not in df_matrix.columns:
            print("❌ 错误：在原始矩阵CSV文件中找不到 '_id' 列，请检查表头。")
            return

        # 4. 执行反向剔除筛选
        print("🔍 正在执行反向剔除...")
        # 核心逻辑：加了 ~ 符号，表示保留 df_matrix 中 '_id' 【不在】 exclude_ids 里的行
        mask = ~df_matrix['_id'].isin(exclude_ids)
        filtered_df = df_matrix[mask].copy()

        # 5. 确保输出目录存在，并保存结果
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        print(f"💾 正在将高频矩阵保存至: {output_path}")
        filtered_df.to_csv(output_path, index=False)

        print("✅ 处理完成！")
        print(f"📊 数据对比:")
        print(f"   - 原始矩阵总计: {len(df_matrix)} 行")
        print(f"   - 剔除低频数据: {len(exclude_ids)} 行")
        print(f"   - 最终剩余高频: {len(filtered_df)} 行")

    except FileNotFoundError as e:
        print(f"❌ 找不到文件: {e.filename}，请检查路径。")
    except Exception as e:
        print(f"❌ 运行过程中发生意外错误: {e}")

if __name__ == "__main__":
    extract_high_freq_data()
