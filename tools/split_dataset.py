import os
import random
import pandas as pd

def main():
    # ==========================================
    # 路径配置
    # ==========================================
    base_dir = "/data1/sft/ct_workspace/"
    #out_dir = os.path.join(base_dir, "huali")
    out_dir = "/home/huali/code/CT-CLIP-main/tools/"
    
    # 确保输出文件夹存在
    os.makedirs(out_dir, exist_ok=True)
    print(f"[*] 检查输出目录: {out_dir}")

    # 输入文件路径
    dir_list_path = os.path.join(base_dir, "dir_list.txt")
    train_reports_src = os.path.join(base_dir, "train_reports.csv")
    train_meta_src = os.path.join(base_dir, "train_metadata.csv")

    # ==========================================
    # 任务 ①：随机划分 dir_list.txt
    # ==========================================
    print("\n>>> 开始任务 ①: 划分目录列表")
    with open(dir_list_path, 'r', encoding='utf-8') as f:
        # 读取并去掉每行末尾的换行符和空白
        all_dirs = [line.strip() for line in f if line.strip()]
    
    total_dirs = len(all_dirs)
    print(f"  - dir_list.txt 中共读取到 {total_dirs} 个文件夹")
    
    if total_dirs < 100:
        raise ValueError("文件夹总数不足100个，无法抽取100个作为验证集！")

    # 随机打乱并切片
    random.seed(42) # 固定随机种子，保证每次运行划分结果一致。如需每次不同可删掉此行
    random.shuffle(all_dirs)
    
    valid_dirs = all_dirs[:100]
    train_dirs = all_dirs[100:]
    
    # 保存划分后的 list
    valid_list_path = os.path.join(out_dir, "valid_list.txt")
    train_list_path = os.path.join(out_dir, "train_list.txt")
    
    with open(valid_list_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(valid_dirs))
    with open(train_list_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_dirs))
        
    print(f"  - 成功划分: 验证集 100 个 -> {valid_list_path}")
    print(f"  - 成功划分: 训练集 {len(train_dirs)} 个 -> {train_list_path}")


    # ==========================================
    # 任务 ②：过滤 train_reports.csv
    # ==========================================
    print("\n>>> 开始任务 ②: 过滤 Reports 数据")
    print("  - 正在读取原始 reports csv...")
    df_reports = pd.read_csv(train_reports_src)
    
    # 利用 pandas 的 isin() 函数进行高效匹配
    df_valid_reports = df_reports[df_reports['SourceFolder'].isin(valid_dirs)]
    df_train_reports = df_reports[df_reports['SourceFolder'].isin(train_dirs)]
    
    # 保存 csv
    out_valid_reports = os.path.join(out_dir, "valid_reports.csv")
    out_train_reports = os.path.join(out_dir, "train_reports.csv")
    
    df_valid_reports.to_csv(out_valid_reports, index=False)
    df_train_reports.to_csv(out_train_reports, index=False)
    
    print(f"  - 验证集 Reports: 筛选出 {len(df_valid_reports)} 行 -> {out_valid_reports}")
    print(f"  - 训练集 Reports: 筛选出 {len(df_train_reports)} 行 -> {out_train_reports}")


    # ==========================================
    # 任务 ③：过滤 train_metadata.csv
    # ==========================================
    print("\n>>> 开始任务 ③: 过滤 Metadata 数据")
    # 提取刚过滤出的 reports 中的 VolumeName，并去重
    valid_volume_names = df_valid_reports['VolumeName'].unique()
    train_volume_names = df_train_reports['VolumeName'].unique()
    
    print("  - 正在读取原始 metadata csv...")
    df_metadata = pd.read_csv(train_meta_src)
    
    # 根据 VolumeName 进行过滤匹配
    df_valid_meta = df_metadata[df_metadata['VolumeName'].isin(valid_volume_names)]
    df_train_meta = df_metadata[df_metadata['VolumeName'].isin(train_volume_names)]
    
    # 保存 csv
    out_valid_meta = os.path.join(out_dir, "valid_metadata.csv")
    out_train_meta = os.path.join(out_dir, "train_metadata.csv")
    
    df_valid_meta.to_csv(out_valid_meta, index=False)
    df_train_meta.to_csv(out_train_meta, index=False)
    
    print(f"  - 验证集 Metadata: 筛选出 {len(df_valid_meta)} 行 -> {out_valid_meta}")
    print(f"  - 训练集 Metadata: 筛选出 {len(df_train_meta)} 行 -> {out_train_meta}")

    print("\n" + "="*40)
    print("所有任务执行完毕！")

if __name__ == "__main__":
    main()
