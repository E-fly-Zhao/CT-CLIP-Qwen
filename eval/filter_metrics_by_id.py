import pandas as pd
import os

def filter_label_matrix():
    # 1. 定义文件路径
    csv_path = "/home/huali/workspace/psj/evaluation_dataset/api/eval/labeled_eval_label_matrix.csv"
    xlsx_path = "/home/huali/code/CT-CLIP-main/eval/chest_eval.xlsx"
    output_path = "/home/huali/code/CT-CLIP-main/eval/labeled_eval_chest.csv"

    print("🚀 开始处理数据...")

    try:
        # 2. 读取我们上一步刚刚生成的胸部数据表，提取目标 _id
        print(f"📦 正在读取目标ID来源 (Excel): {os.path.basename(xlsx_path)}")
        df_chest = pd.read_excel(xlsx_path)
        
        if '_id' not in df_chest.columns:
            print("❌ 错误：在胸部Excel文件中找不到 '_id' 列，请检查表头。")
            return
            
        # 提取所有的 _id，去重并剔除空值，转化为集合以提高匹配速度
        target_ids = df_chest['_id'].dropna().unique()
        print(f"✅ 成功从 Excel 中提取到 {len(target_ids)} 个不重复的胸部目标 _id。")

        # 3. 读取包含所有标签矩阵的庞大 CSV 文件
        print(f"📦 正在读取原始标签矩阵 (CSV): {os.path.basename(csv_path)}")
        df_matrix = pd.read_csv(csv_path)

        if '_id' not in df_matrix.columns:
            print("❌ 错误：在标签矩阵CSV文件中找不到 '_id' 列，请检查表头。")
            return

        # 4. 执行匹配筛选
        print("🔍 正在根据 _id 进行精准匹配筛选...")
        # 核心逻辑：保留 df_matrix 中 '_id' 存在于 target_ids 列表里的行
        mask = df_matrix['_id'].isin(target_ids)
        filtered_df = df_matrix[mask].copy()

        # 5. 确保输出目录存在，并保存结果
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        print(f"💾 正在将筛选后的矩阵保存至: {output_path}")
        filtered_df.to_csv(output_path, index=False)

        print("✅ 处理完成！")
        print(f"📊 数据对比: 原始矩阵共计 {len(df_matrix)} 行 ➡️ 成功提取出 {len(filtered_df)} 行与胸部对应的矩阵数据。")

    except FileNotFoundError as e:
        print(f"❌ 找不到文件: {e.filename}，请检查路径是否拼写正确。")
    except Exception as e:
        print(f"❌ 运行过程中发生意外错误: {e}")

if __name__ == "__main__":
    filter_label_matrix()
