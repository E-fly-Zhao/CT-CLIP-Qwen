from torch.distributed.checkpoint import FileSystemReader

# 1. 指向你的 FSDP 碎片文件夹
checkpoint_path = "/mnt/huali/ct_dataset_10000/output/CTClip_step_20000/pytorch_model_fsdp_0"

# 2. 读取硬盘里到底存了什么名字
reader = FileSystemReader(checkpoint_path)
metadata = reader.read_metadata()

# 3. 打印出前 20 个真实的参数名字
saved_keys = list(metadata.state_dict_metadata.keys())
print("🚨 硬盘里真实的参数名字长这样：\n")
for k in saved_keys[:20]:
    print(k)
