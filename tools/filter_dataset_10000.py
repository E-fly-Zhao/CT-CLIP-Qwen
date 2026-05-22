import pandas as pd
import os

def filter_and_save():
    # 1. 定义文件路径
    #xlsx_path = "/data1/sft/ct_workspace/ct_dataset_base_260428.xlsx"
    #csv_path = "/data1/sft/ct_workspace/train_reports.csv"
    #output_path = "/home/huali/code/CT-CLIP-main/CT_CLIP/dataset_10000/reports_full.csv"
    xlsx_path = "/oss/share_data/CT/ct_dataset_eval_260514/ct_dataset_eval_260514.xlsx"
    csv_path = "/oss/share_data/CT/ct_dataset_eval_260514/train_reports.csv"
    output_path = "/home/huali/code/CT-CLIP-main/CT-CLIP/dataset_eval_2000/reports_full.csv"

    print(f"🚀 开始处理数据...")

    try:
        # 2. 加载数据
        print(f"📦 正在读取 Excel: {os.path.basename(xlsx_path)}")
        df_xlsx = pd.read_excel(xlsx_path)

        print(f"📦 正在读取 CSV: {os.path.basename(csv_path)}")
        df_csv = pd.read_csv(csv_path)

        # 3. 提取 XLSX 路径最后一级
        print(f"🔍 正在解析 Excel 中的 dicom_path...")
        df_xlsx['tmp_dicom_last'] = df_xlsx['dicom_path'].apply(
            lambda x: os.path.basename(str(x).strip().rstrip('/'))
        )

        # 4. 提取 CSV 路径最后一级 (🌟 本次新增的核心逻辑)
        print(f"🔍 正在解析 CSV 中的 SourceFolder...")
        df_csv['tmp_source_last'] = df_csv['SourceFolder'].apply(
            lambda x: os.path.basename(str(x).strip().rstrip('/'))
        )

        # 5. 筛选数据：让双方提取出来的“最后一级”进行精确匹配
        print(f"🧬 正在执行双向交叉匹配...")
        mask = df_xlsx['tmp_dicom_last'].isin(df_csv['tmp_source_last'])
        filtered_df = df_xlsx[mask].copy()

        # 6. 保存结果
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # 移除为了筛选而创建的临时列，保持原表干净
        result_df = filtered_df.drop(columns=['tmp_dicom_last'])

        result_df.to_csv(output_path, index=False)

        print(f"✅ 处理完成！")
        print(f"📊 匹配成功的行数: {len(result_df)}")
        print(f"💾 结果已存至: {output_path}")

    except Exception as e:
        print(f"❌ 运行出错: {e}")


if __name__ == "__main__":
    filter_and_save()
