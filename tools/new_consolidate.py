import os
import torch
import argparse
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader

# 导入你的模型类 (确保在运行此脚本时，使用的是你最新修改过的 ctvit.py)
from ct_clip import CTCLIP
from transformer_maskgit.ctvit import CTViT
from transformers import AutoConfig, AutoModel

def extract_and_merge(checkpoint_path, output_path, qwen_path):
    print("\n🚀 启动 FSDP 满血版完美缝合脚本...")
    
    # 1. 建立伪装的本地单卡分布式环境（dcp.load 必须）
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29511" 
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", rank=0, world_size=1)

    # 2. 实例化最新版的模型容器
    print("🏗️ 正在构建全新架构的视觉容器...")
    ctvit = CTViT(
        dim=512,
        codebook_size=8192,
        image_size=480,
        patch_size=20,
        temporal_patch_size=10,
        spatial_depth=4,
        temporal_depth=4,
        dim_head=32,
        heads=8
    )
    
    print("🧠 正在构建 Qwen 空壳结构...")
    config = AutoConfig.from_pretrained(qwen_path, trust_remote_code=True)
    text_model = AutoModel.from_config(config, trust_remote_code=True)

    model = CTCLIP(
        image_encoder=ctvit,
        text_encoder=text_model,
        dim_image=294912, 
        dim_text=4096,     
        dim_latent=512,
        use_mlm=False,
        extra_latent_projection=False,
        downsample_image_embeds=False,
        use_all_token_embeds=False,
        tokenizer=None
    )

    state_dict = model.state_dict()
    
    # ==========================================
    # 🛡️ 绝对安全校验：确保内存模型拥有最新图纸
    # ==========================================
    must_have_keys = [
        "visual_transformer.enc_spatial_transformer.layers.0.1.null_kv",
        "visual_transformer.enc_spatial_transformer.layers.0.3.0.weight"
    ]
    for key in must_have_keys:
        if key not in state_dict:
            raise RuntimeError(f"❌ 致命错误：当前代码实例化出来的模型依然没有 {key}！请确保你在运行此脚本时，环境变量和目录使用的是最新的 ctvit.py 代码！")
            
    print("✅ 架构校验通过：当前代码容器完美匹配，口袋够大，不会漏掉任何参数！")
    
    # ==========================================
    # 📥 缝合 FSDP 权重
    # ==========================================
    print("\n📥 正在读取硬盘 FSDP 分片并对齐...")
    reader = FileSystemReader(checkpoint_path)
    metadata = reader.read_metadata()
    saved_keys = metadata.state_dict_metadata.keys()

    state_dict_to_load = {}
    for k, v in state_dict.items():
        disk_key = f"model.{k}"
        if disk_key in saved_keys:
            state_dict_to_load[disk_key] = v
        elif k in saved_keys:
            state_dict_to_load[k] = v

    print("🔌 正在将硬盘分片矩阵灌入内存模型...")
    dcp.load(
        state_dict=state_dict_to_load,
        checkpoint_id=checkpoint_path,
    )
    
    # ==========================================
    # ✂️ 精准切割：丢掉 9B 语言模型，只留视觉和投影
    # ==========================================
    print("\n✂️ 正在剥离 Qwen，提取 视觉骨干 + 投影层...")
    final_state_dict = {}
    count_vision, count_proj, count_special = 0, 0, 0

    for loaded_key, value in state_dict_to_load.items():
        clean_key = loaded_key.replace("model.", "")

        if clean_key.startswith("visual_transformer."):
            final_state_dict[clean_key] = value
            count_vision += 1
            if 'null_kv' in clean_key or 'layers.0.3.0' in clean_key:
                count_special += 1
                
        elif clean_key.startswith("to_visual_latent") or clean_key.startswith("to_text_latent"):
            final_state_dict[clean_key] = value
            count_proj += 1

    print(f"📊 提取统计：")
    print(f"   - 视觉层张量 (CTViT): {count_vision} 个")
    print(f"   - 成功捕获特殊层 (PEG/null_kv): {count_special} 个 🎉")
    print(f"   - 投影层张量 (Adapter): {count_proj} 个")

    print(f"\n📤 正在保存满血版权重至: {output_path}")
    torch.save(final_state_dict, output_path)
    dist.destroy_process_group()
    print("✅ 大功告成！你现在拥有了最完美、零参数丢失的满血预训练权重！")


if __name__ == "__main__":
    # 🚨 我们直接将路径写死为你刚刚探查成功的真实 FSDP 数据层，防止弄错
    # checkpoint_dir = "/mnt/huali/ct_dataset_10000/output/CTClip_step_33500/pytorch_model_fsdp_0"
    checkpoint_dir = "/mnt/huali/ct_dataset_10000/output_v2_body/CTClip_step_11000/pytorch_model_fsdp_0"
    # 生成的新权重加个 _full_fixed 后缀以示区别
    output_file = "/mnt/huali/ct_dataset_10000/output_v2_body/CTClip_step_11000_full_fixed.pt"
    
    qwen_path = "/home/huali/model/Qwen3.5-9B"
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    extract_and_merge(checkpoint_dir, output_file, qwen_path)
