import os
import time
from huggingface_hub import HfApi, hf_hub_download

# 1. 强制走国内镜像站加速
HF_ENDPOINT = "https://hf-mirror.com"
os.environ["HF_ENDPOINT"] = HF_ENDPOINT
HF_TOKEN = "" # 确保这是你的真实 Token
REPO_ID = "ibrahimhamamci/CT-RATE"

# 2. 显式初始化 API 客户端
api = HfApi(token=HF_TOKEN, endpoint=HF_ENDPOINT)

print("🚀绕过全站索引，选择目标文件夹")

# 3. 精准循环你需要的前 n个患者
# for i in range(37, 301):
for i in range(301, 1001):
    folder_path = f"dataset/train_fixed/train_{i}"
    print(f"\n🔍 正在检索患者 {i} 的目录: {folder_path}")
    
    try:
        # 直接去读这 1 个患者的文件夹，绝不触发全站扫盘
        tree = api.list_repo_tree(repo_id=REPO_ID, repo_type="dataset", path_in_repo=folder_path, recursive=True)
        
        file_count = 0
        for item in tree:
            # 【核心修复】：兼容最新版 hf_hub 的对象类型检查
            if type(item).__name__ == "RepoFile" or (hasattr(item, "type") and item.type == "file"):
                
                # 进一步安全保障：只下载以 .nii.gz 或 .json 结尾的真实数据文件
                if item.path.endswith(".nii.gz") or item.path.endswith(".json"):
                    print(f"   ⬇️ 正在下载: {item.path}")
                    
                    hf_hub_download(
                        repo_id=REPO_ID,
                        repo_type="dataset",
                        filename=item.path,
                        local_dir="/home/supermicro/zyx/code/CT-CLIP-main/CT-CLIP-main/CT_CLIP/data/",
                        local_dir_use_symlinks=False,
                        token=HF_TOKEN
                    )
                    file_count += 1
        
        if file_count == 0:
            print("   ⚠️ 该患者文件夹为空或在仓库中不存在。")
            
        # 暂停 0.5 秒，防止被镜像站当成并发爬虫封禁
        time.sleep(0.5)
        
    except Exception as e:
        print(f"   ❌ 检索或下载患者 {i} 时出错: {e}")

print("\n✅ 所有指定患者的数据拉取完毕！")
