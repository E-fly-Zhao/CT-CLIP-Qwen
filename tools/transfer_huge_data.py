import os
import shutil
import sys
import time

# ==========================================
# 1. 核心路径配置 (请核对)
# ==========================================
TRAIN_LIST = "/home/huali/code/CT-CLIP-main/tools/train_list.txt"
VALID_LIST = "/home/huali/code/CT-CLIP-main/tools/valid_list.txt"

SRC_BASE_DIR = "/data1/sft/ct_workspace/processed/train_data/"

DEST_TRAIN_DIR = "/mnt/huali/ct_dataset_10000/pretrain_processed_train_data"
DEST_VALID_DIR = "/mnt/huali/ct_dataset_10000/pretrain_processed_valid_data"

# ==========================================
# 2. 安全与性能配置
# ==========================================
MIN_FREE_SPACE_GB = 30.0  # 安全红线：剩余空间低于 30GB 时立即熔断停止
CHECK_INTERVAL = 10       # 检查频率：每拷贝 10 个数据节点，检查一次磁盘容量

def check_disk_space(target_dir):
    """获取指定路径挂载点的剩余空间(GB)"""
    # 确保目标目录存在，否则往上找父目录来查容量
    check_path = target_dir
    while not os.path.exists(check_path) and check_path != '/':
        check_path = os.path.dirname(check_path)
        
    total, used, free = shutil.disk_usage(check_path)
    free_gb = free / (1024 ** 3)
    return free_gb

def read_list(list_path):
    """读取 txt 列表"""
    if not os.path.exists(list_path):
        print(f"[错误] 找不到列表文件: {list_path}")
        return []
    with open(list_path, 'r', encoding='utf-8') as f:
        # 去除空白和换行符
        return [line.strip() for line in f if line.strip()]

def transfer_data(data_list, src_base, dest_base, task_name):
    """执行传输的核心逻辑"""
    os.makedirs(dest_base, exist_ok=True)
    total_items = len(data_list)
    
    print(f"\n[{task_name}] 开始传输，共计 {total_items} 个目标...")
    
    # 启动前先做一次基础容量检查
    initial_free_gb = check_disk_space(dest_base)
    print(f"[{task_name}] 目标磁盘初始可用空间: {initial_free_gb:.2f} GB")
    if initial_free_gb < MIN_FREE_SPACE_GB:
        print(f"[致命错误] 初始可用空间 ({initial_free_gb:.2f} GB) 已低于安全红线 ({MIN_FREE_SPACE_GB} GB)！终止任务。")
        sys.exit(1)

    success_count = 0
    skip_count = 0
    error_count = 0

    for idx, item_name in enumerate(data_list, 1):
        # 1. 触发定时容量检查
        if idx % CHECK_INTERVAL == 0:
            current_free_gb = check_disk_space(dest_base)
            if current_free_gb < MIN_FREE_SPACE_GB:
                print(f"\n[🚨 熔断触发] 剩余空间不足: 仅剩 {current_free_gb:.2f} GB！为保护系统，脚本已停止拷贝。")
                print(f"当前进度停留在: {idx}/{total_items}")
                sys.exit(1)
            else:
                print(f"  -> 容量巡检: 剩余 {current_free_gb:.2f} GB (安全)")

        src_path = os.path.join(src_base, item_name)
        dest_path = os.path.join(dest_base, item_name)

        # 2. 存在性检查 (防重复)
        if os.path.exists(dest_path):
            print(f"[{idx}/{total_items}] ⏭️ 跳过 (已存在): {item_name}")
            skip_count += 1
            continue

        # 3. 源文件/文件夹检查
        if not os.path.exists(src_path):
            print(f"[{idx}/{total_items}] ❌ 源端不存在: {src_path}")
            error_count += 1
            continue

        # 4. 执行拷贝 (区分是文件夹还是文件)
        try:
            print(f"[{idx}/{total_items}] ⏳ 正在拷贝: {item_name} ...", end="", flush=True)
            if os.path.isdir(src_path):
                shutil.copytree(src_path, dest_path)
            else:
                shutil.copy2(src_path, dest_path)
            print(" 完成!")
            success_count += 1
        except Exception as e:
            print(f" 失败! 错误信息: {str(e)}")
            error_count += 1

    print(f"\n[{task_name}] 报告:")
    print(f"  - 成功拷贝: {success_count} 个")
    print(f"  - 防重复跳过: {skip_count} 个")
    print(f"  - 发生错误: {error_count} 个")

def main():
    print("="*50)
    print(" 巨量数据传输程序初始化")
    print("="*50)
    
    # 获取列表
    train_list = read_list(TRAIN_LIST)
    valid_list = read_list(VALID_LIST)
    
    if not train_list and not valid_list:
        print("未获取到任何传输列表，程序退出。")
        return

    # 传输验证集
    if valid_list:
        transfer_data(valid_list, SRC_BASE_DIR, DEST_VALID_DIR, "验证集 (Valid)")
        
    # 传输训练集
    if train_list:
        transfer_data(train_list, SRC_BASE_DIR, DEST_TRAIN_DIR, "训练集 (Train)")
        
    print("\n🎉 所有传输队列执行完毕！")

if __name__ == "__main__":
    main()
