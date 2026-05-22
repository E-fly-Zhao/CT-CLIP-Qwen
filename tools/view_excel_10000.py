#!/usr/bin/env python3
import pandas as pd
import sys

def view_excel(file_path):
    try:
        # 读取 Excel 文件，默认读取第一个 Sheet
        print(f"Loading {file_path}...\n")
        df = pd.read_excel(file_path)
        
        # 优化终端显示效果，防止列被随意折叠
        pd.set_option('display.max_columns', None)  # 显示所有列
        pd.set_option('display.width', 1000)        # 加宽终端显示宽度
        pd.set_option('display.max_rows', 50)       # 默认最多显示50行，防止刷屏
        
        # 打印表格内容
        print(df)
        
        # 打印基础信息
        print("\n" + "="*50)
        print(f"📊 Dataset Summary: {df.shape[0]} Rows, {df.shape[1]} Columns")
        print("="*50)
        
    except ImportError:
        print("❌ Error: Missing required libraries. Please run: pip install pandas openpyxl")
    except Exception as e:
        print(f"❌ Error reading file: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_file = sys.argv[1]
    else:
        # 默认指向你提到的那个文件
        target_file = "/data1/sft/ct_workspace/ct_dataset_base_260428.xlsx"
        
    view_excel(target_file)
