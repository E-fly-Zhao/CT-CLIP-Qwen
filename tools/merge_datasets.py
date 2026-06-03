import pandas as pd
import os

def merge_and_format_datasets():
    # 1. 定义文件夹路径与前缀的映射字典 (Mapping)
    # 字典格式：{ "源文档目录": "对应的图像前缀路径" }
    data_mapping = {
        "/nas/share_data/CT/ct_dataset_base_260513/ct_dataset_base_260513_part1_doc/": "ct_dataset_base_260513_part1_img/",
        "/nas/share_data/CT/ct_dataset_base_260513/0513_lz2nodesk_ct_chest_part2_doc/": "ct_dataset_base_260513_part2_img/",
        "/data2/0513_lz2nodesk_ct_chest_part3_doc/": "ct_dataset_base_260513_part3_img/",
        "/data2/0513_lz2nodesk_ct_chest_part4_doc/": "ct_dataset_base_260513_part4_img/",
        "/oss/share_data/CT/ct_dataset_base_260513/0513_lz2nodesk_ct_chest_part5_doc/": "ct_dataset_base_260513_part5_img/",
        "/data4/0513_lz2nodesk_ct_chest_part6_doc/": "ct_dataset_base_260513_part6_img/",
        "/data1/sft/ct_workspace/": "/ct_dataset_base_260513_part0/"  # part0 的特殊前缀
    }

    # 2. 定义输出路径（你可以根据需要修改保存目录）
    output_dir = "/home/huali/code/CT-CLIP-main/dataset_merged/"
    output_reports_path = os.path.join(output_dir, "merged_train_reports.csv")
    output_metadata_path = os.path.join(output_dir, "merged_train_metadata.csv")

    # 用于存放所有处理后的 DataFrame 的列表
    reports_dfs = []
    metadata_dfs = []

    print(" 开始读取并处理数据...")

    # 3. 遍历所有的源目录
    for base_dir, prefix in data_mapping.items():
        reports_file = os.path.join(base_dir, "train_reports.csv")
        metadata_file = os.path.join(base_dir, "train_metadata.csv")

        # 处理 train_reports.csv
        if os.path.exists(reports_file):
            print(f" 正在处理: {reports_file}")
            df_rep = pd.read_csv(reports_file, low_memory=False)
            
            if 'SourceFolder' in df_rep.columns:
                # 核心逻辑：提取最后一级文件夹，并加上对应的前缀
                # strip()去空格，rstrip('/')去掉可能存在的末尾斜杠，然后取basename
                df_rep['SourceFolder'] = df_rep['SourceFolder'].apply(
                    lambda x: prefix + os.path.basename(str(x).strip().rstrip('/'))
                )
            else:
                print(f" 警告: {reports_file} 中没有找到 'SourceFolder' 列！")
                
            reports_dfs.append(df_rep)
        else:
            print(f"❌ 找不到文件: {reports_file}")

        # 处理 train_metadata.csv (直接读取即可，不需要修改路径)
        if os.path.exists(metadata_file):
            print(f" 正在读取: {metadata_file}")
            df_meta = pd.read_csv(metadata_file, low_memory=False)
            metadata_dfs.append(df_meta)
        else:
            print(f"❌ 找不到文件: {metadata_file}")

    print("\n 正在拼接所有数据...")

    # 4. 合并数据
    # ignore_index=True 确保合并后的行索引是从 0 重新开始排列的
    if reports_dfs:
        merged_reports = pd.concat(reports_dfs, ignore_index=True)
    else:
        merged_reports = pd.DataFrame()

    if metadata_dfs:
        merged_metadata = pd.concat(metadata_dfs, ignore_index=True)
    else:
        merged_metadata = pd.DataFrame()

    # 5. 保存结果
    os.makedirs(output_dir, exist_ok=True)
    
    if not merged_reports.empty:
        merged_reports.to_csv(output_reports_path, index=False)
        print(f"✅ 成功合并 reports，总行数: {len(merged_reports)}")
        print(f" 已保存至: {output_reports_path}")
        
    if not merged_metadata.empty:
        merged_metadata.to_csv(output_metadata_path, index=False)
        print(f"✅ 成功合并 metadata，总行数: {len(merged_metadata)}")
        print(f" 已保存至: {output_metadata_path}")

if __name__ == "__main__":
    merge_and_format_datasets()
