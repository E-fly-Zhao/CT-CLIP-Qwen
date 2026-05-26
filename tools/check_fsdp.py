from torch.distributed.checkpoint import FileSystemReader

# 🚨 路径已经精准指向了真正的权重子目录
checkpoint_path = "/mnt/huali/ct_dataset_10000/output/CTClip_step_34500/pytorch_model_fsdp_0"

print(f"🔍 正在读取 FSDP 元数据: {checkpoint_path}")
try:
    reader = FileSystemReader(checkpoint_path)
    metadata = reader.read_metadata()
    saved_keys = metadata.state_dict_metadata.keys()

    print(f"📦 分片内总计包含 {len(saved_keys)} 个张量矩阵。")

    found_null = False
    found_peg = False

    for key in saved_keys:
        if 'null_kv' in key:
            found_null = True
        if 'layers.0.3.0.weight' in key:
            found_peg = True

    print("\n📊 ================= 最终判决报告 ================= 📊")
    if not found_null:
        print("❌ 铁证 1：当年跑预训练时，硬盘里的模型根本就没有 null_kv！")
    else:
        print("✅ 居然有 null_kv！")

    if not found_peg:
        print("❌ 铁证 2：当年跑预训练时，硬盘里的模型根本就没有 PEG(layers.x.3.0) 层！")
    else:
        print("✅ 居然有 PEG！")

except Exception as e:
    print(f"读取失败，请检查路径: {e}")
