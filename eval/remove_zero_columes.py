import pandas as pd
import os

def clean_zero_columns():
    input_path = "/home/huali/code/CT-CLIP-main/eval/label_high_freq.csv"
    output_path = "/home/huali/code/CT-CLIP-main/eval/label_high_freq_cleaned.csv"

    print("🚀 开始清洗数据...")

    try:
        # 加上 low_memory=False 防止混合类型读取报错
        print(f"📦 正在读取高频矩阵: {os.path.basename(input_path)}")
        df = pd.read_csv(input_path, low_memory=False)
        
        initial_col_count = len(df.columns)
        print(f"📊 初始状态: 共 {initial_col_count} 列，{len(df)} 行。")

        # 保护 '_id' 列
        if '_id' in df.columns:
            id_column = df['_id']
            label_df = df.drop(columns=['_id'])
        else:
            id_column = None
            label_df = df

        # ==================== 本次修复的核心 ====================
        print("⚙️ 正在执行强制类型转换和空值填补...")
        # 1. 强制将所有数据转为数字（如果遇到奇怪的字母，coerce 会把它变成 NaN）
        label_df_numeric = label_df.apply(pd.to_numeric, errors='coerce')
        # 2. 将所有的 NaN（空缺值）全部当成 0 处理，防止 NaN != 0 被误判为 True
        label_df_numeric = label_df_numeric.fillna(0)

        # 3. 现在的判断绝对精准：找出一列中是否存在真正意义上不等于 0 的数字
        valid_columns_mask = (label_df_numeric != 0).any(axis=0)
        # ========================================================
        
        print("🔍 正在扫描并剔除全为 0 的疾病标签列...")
        # 使用 loc 仅保留判定为 True 的列（我们保留 label_df 原本的列，确保格式不变）
        cleaned_label_df = label_df.loc[:, valid_columns_mask]

        # 将受保护的 '_id' 列重新拼接到最前面
        if id_column is not None:
            final_df = pd.concat([id_column, cleaned_label_df], axis=1)
        else:
            final_df = cleaned_label_df

        final_col_count = len(final_df.columns)
        dropped_count = initial_col_count - final_col_count

        # 保存结果
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        print(f"💾 正在保存清理后的数据至: {output_path}")
        final_df.to_csv(output_path, index=False)

        print("✅ 清洗完成！")
        print(f"📉 结果统计: 成功删除了 {dropped_count} 个全为 0 的无效疾病列，最终保留了 {final_col_count} 列。")

    except FileNotFoundError:
        print(f"❌ 错误：找不到源文件 {input_path}，请检查路径。")
    except Exception as e:
        print(f"❌ 运行过程中发生错误: {e}")

if __name__ == "__main__":
    clean_zero_columns()
