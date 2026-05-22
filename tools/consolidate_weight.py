import os
import torch
import argparse
import torch.distributed as dist
from torch.distributed.checkpoint import FileSystemReader
import torch.distributed.checkpoint as dcp

# 导入你的模型类 (确保路径能找到)
from ct_clip import CTCLIP
from transformer_maskgit.ctvit import CTViT
from transformers import AutoConfig, AutoModel

def extract_ctvit_weights(checkpoint_path, output_path, qwen_path):
    print(f"\n🚀 开始提取纯净版 CT-CLIP 下游权重...")
    print(f"📂 输入分片路径: {checkpoint_path}")
    print(f"💾 输出文件路径: {output_path}")

    print("\n🏗️ 正在构建模型架构模板...")

    # 1. 构建视觉分支 (🚨 满血复原深度为 4)
    image_size = 480
    ctvit = CTViT(
        dim=512,
        codebook_size=8192,
        image_size=image_size,
        patch_size=20,
        temporal_patch_size=10,
        spatial_depth=4,   # 👈 必须为4
        temporal_depth=4,  # 👈 必须为4
        dim_head=32,
        heads=8
    )

    # 2. 构建文本分支
    print("🧠 正在构建 Qwen 空壳结构 (不加载真实权重)...")
    config = AutoConfig.from_pretrained(qwen_path, trust_remote_code=True)
    text_model = AutoModel.from_config(config, trust_remote_code=True)

    # 3. 拼装成整体 CLIP 模型 (🚨 尺寸完美对齐预训练)
    model = CTCLIP(
        image_encoder=ctvit,
        text_encoder=text_model,
        dim_image=294912,  # 👈 必须为 294912
        dim_text=4096,     # 👈 必须为 4096
        dim_latent=512,
        use_mlm=False,
        extra_latent_projection=False,
        downsample_image_embeds=False,
        use_all_token_embeds=False,
        tokenizer=None
    )

    state_dict = model.state_dict()

    print("\n📥 正在探查 .distcp 分片元数据...")
    reader = FileSystemReader(checkpoint_path)
    metadata = reader.read_metadata()
    saved_keys = metadata.state_dict_metadata.keys()

    state_dict_to_load = {}
    print("\n🔍 ================= 投影层诊断雷达 ================= 🔍")
    proj_found = False
    
    for k, v in state_dict.items():
        disk_key = f"model.{k}"

        if disk_key in saved_keys:
            state_dict_to_load[disk_key] = v
            # 💡 诊断：如果发现是投影层，立刻大声报告！
            if 'to_visual_latent' in k or 'to_text_latent' in k:
                print(f"✅ 成功在硬盘找到投影层并准备装载: {disk_key}")
                proj_found = True
        elif k in saved_keys:
            state_dict_to_load[k] = v
            if 'to_visual_latent' in k or 'to_text_latent' in k:
                print(f"✅ 成功在硬盘找到投影层并准备装载: {k}")
                proj_found = True
        else:
            # 取消其他层找不到的打印，保持日志干净，只关注投影层
            pass
            
    if not proj_found:
        print("❌ 警告：在硬盘分片中没有找到任何投影层参数！这会导致零样本分类随机乱猜！")
    print("🔍 ==================================================== 🔍")

    print(f"\n📥 准备从磁盘缝合 {len(state_dict_to_load)} 个核心参数矩阵...")
    
    # 建立伪装群聊
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29511" 
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", rank=0, world_size=1)

    # 缝合到内存中
    dcp.load(
        state_dict=state_dict_to_load,
        checkpoint_id=checkpoint_path,
    )

    # ==========================================
    # 🎯 🚨核心修复：精准切割与前缀修正🚨
    # ==========================================
    print("\n✂️ 正在剥离冗余权重，精准提取 视觉骨干 + 投影层...")
    final_state_dict = {}

    count_vision = 0
    count_proj = 0

    for loaded_key, value in state_dict_to_load.items():
        # 1. 先把 "model." 这个 FSDP 马甲强行脱掉，还原出干净的 CTCLIP 键名
        clean_key = loaded_key.replace("model.", "")

        # 2. 我们只提取下面这三种关键部件：视觉骨干、视觉投影层、文本投影层
        if clean_key.startswith("visual_transformer."):
            final_state_dict[clean_key] = value
            count_vision += 1
        elif clean_key.startswith("to_visual_latent"):
            final_state_dict[clean_key] = value
            count_proj += 1
        elif clean_key.startswith("to_text_latent"):
            final_state_dict[clean_key] = value
            count_proj += 1

    print(f"📊 提取完成！")
    print(f"   - 视觉层张量 (CTViT): {count_vision} 个")
    print(f"   - 投影层张量 (Adapter): {count_proj} 个 (预期应为 4 个：两个 weight，两个 bias)")

    # 保存最终结果
    print(f"\n📤 正在保存最终权重至 {output_path}...")
    torch.save(final_state_dict, output_path)
    
    dist.destroy_process_group()
    print("✅ 大功告成！完美包含投影层的下游权重提取完毕！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="提取 FSDP 中的纯 CTViT 及投影层权重")
    
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--qwen_path", type=str, default="/home/huali/model/Qwen3.5-9B")

    args = parser.parse_args()

    if not os.path.exists(args.checkpoint_dir):
        print(f"❌ 严重错误: 找不到输入路径 {args.checkpoint_dir}")
    else:
        output_dir = os.path.dirname(args.output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        extract_ctvit_weights(args.checkpoint_dir, args.output_file, args.qwen_path)
