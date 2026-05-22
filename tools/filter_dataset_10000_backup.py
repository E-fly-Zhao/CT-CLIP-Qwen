import pandas as pd
import os

def filter_and_save():
    # 1. 定义文件路径
    #xlsx_path = "/data1/sft/ct_workspace/ct_dataset_base_260428.xlsx"
    #csv_path = "/data1/sft/ct_workspace/train_reports.csv"
    #output_path = "/home/huali/code/CT-CLIP-main/CT_CLIP/dataset_10000/reports_full.csv"
    xlsx_path = "/oss/share_data/CT/ct_dataset_eval_260514/ct_dataset_eval_260514.xlsx"
    csv_path = "/oss/share_data/CT/ct_dataset_eval_260514/train_reports.csv"
    output_path = "home/huali/code/CT-CLIP-main/CT-CLIP/dataset_eval_2000/reports_full.csv"

    print(f"🚀 开始处理数据...")

    try:
        # 2. 加载数据
        print(f"📦 正在读取 Excel: {os.path.basename(xlsx_path)}")
        df_xlsx = pd.read_excel(xlsx_path)
        
        print(f"📦 正在读取 CSV: {os.path.basename(csv_path)}")
        df_csv = pd.read_csv(csv_path)

        # 3. 提取路径最后一级
        # 使用 os.path.basename 可以自动处理末尾是否有斜杠的情况
        # strip() 确保去除可能存在的空格
        print(f"🔍 正在解析 dicom_path...")
        df_xlsx['tmp_last_level'] = df_xlsx['dicom_path'].apply(
            lambda x: os.path.basename(str(x).strip().rstrip('/'))
        )

        # 4. 筛选数据
        # 找出 xlsx 中 tmp_last_level 与 csv 中 SourceFolder 相匹配的行
        print(f"🧬 正在执行交叉匹配...")
        # 我们使用 isin 来筛选，如果你需要将 CSV 的报告内容也合并进来，可以使用 pd.merge
        mask = df_xlsx['tmp_last_level'].isin(df_csv['SourceFolder'])
        filtered_df = df_xlsx[mask].copy()

        # 5. 保存结果
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 移除为了筛选而创建的临时列
        result_df = filtered_df.drop(columns=['tmp_last_level'])
        
        result_df.to_csv(output_path, index=False)
        
        print(f"✅ 处理完成！")
        print(f"📊 匹配成功的行数: {len(result_df)}")
        print(f"💾 结果已存至: {output_path}")

    except Exception as e:
        print(f"❌ 运行出错: {e}")

if __name__ == "__main__":
    filter_and_save()
