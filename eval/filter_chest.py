import pandas as pd
import os

def filter_chest_data():
    # 1. 定义文件路径（已修复目标路径中的双斜杠）
    input_path = "/home/huali/workspace/xly/CTModel/labeled_eval_body.xlsx"
    output_path = "/home/huali/code/CT-CLIP-main/eval/chest_eval.xlsx"

    print(f"🚀 开始处理数据...")

    try:
        # 2. 读取原始 Excel 文件
        print(f"📦 正在读取文件: {input_path}")
        df = pd.read_excel(input_path)

        # 检查是否存在“部位”列，防止由于表头带有空格等导致的读取失败
        if '部位' not in df.columns:
            print(f"❌ 错误：未在表格中找到精确名为 '部位' 的列。")
            print(f"💡 当前表格包含的列名为: {list(df.columns)}")
            return

        # 3. 筛选包含“胸部”的行
        print("🔍 正在筛选'部位'为胸部或包含'胸部'的数据...")
        # 核心逻辑：
        # fillna('') 防止存在空值报错
        # str.contains('胸部', na=False) 会匹配“胸部”、“胸部CT”、“全胸部”等所有包含该词的字符串
        mask = df['部位'].fillna('').str.contains('胸部', na=False)
        filtered_df = df[mask].copy()

        # 4. 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)

        # 5. 保存结果到新的 Excel
        print(f"💾 正在保存结果到: {output_path}")
        # index=False 保证输出的表格不会额外带上一列毫无意义的数字索引
        filtered_df.to_excel(output_path, index=False)

        print("✅ 处理完成！")
        print(f"📊 数据对比: 原始总计 {len(df)} 行 ➡️ 筛选得到 {len(filtered_df)} 行。")

    except FileNotFoundError:
        print(f"❌ 错误：找不到源文件 {input_path}，请检查路径。")
    except Exception as e:
        print(f"❌ 运行过程中发生错误: {e}")

if __name__ == "__main__":
    filter_chest_data()
