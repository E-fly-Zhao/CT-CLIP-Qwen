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
    print(f"\n🚀 开始提取纯净版 CTViT 权重...")
    print(f"📂 输入分片路径: {checkpoint_path}")
    print(f"💾 输出文件路径: {output_path}")

    print("\n🏗️ 正在构建模型架构模板...")

    # 1. 构建视觉分支
    image_size = 480
    ctvit = CTViT(
        dim=512,
        codebook_size=8192,
        image_size=image_size,
        patch_size=20,
        temporal_patch_size=10,
        #spatial_depth=4,
        #temporal_depth=4,
        spatial_depth=2,
        temporal_depth=2,
        dim_head=32,
        heads=8
    )

    # 2. 构建文本分支 (使用 Config 仅创建空壳结构，极速且省内存)
    print("🧠 正在构建 Qwen 空壳结构 (不加载真实权重)...")
    config = AutoConfig.from_pretrained(qwen_path, trust_remote_code=True)
    text_model = AutoModel.from_config(config, trust_remote_code=True)

    # 3. 拼装成整体 CLIP 模型
    model = CTCLIP(
        image_encoder=ctvit,
        text_encoder=text_model,
        dim_image=512,
        dim_text=3584,  # Qwen 9B 的 hidden_size
        dim_latent=512,
        tokenizer=None
    )

    # 👇 🚨 重点修复：必须获取到模板的 state_dict！🚨 👇
    state_dict = model.state_dict()
    # 👆 ========================================= 👆

    print("\n📥 正在探查 .distcp 分片元数据...")
    reader = FileSystemReader(checkpoint_path)

    # ==========================================
    # 🎯 智能过滤结界：只加载硬盘里真正存在的参数
    # ==========================================
    metadata = reader.read_metadata()
    saved_keys = metadata.state_dict_metadata.keys()

    state_dict_to_load = {}
    for k, v in state_dict.items():
        #if k in saved_keys:
        #    state_dict_to_load[k] = v
        #else:
        #    print(f"⚠️ 跳过分片中未保存的冗余参数 (正常现象): {k}")
        # 🚨 核心修复：给我们要找的 key 穿上 FSDP 的 "model." 马甲
        disk_key = f"model.{k}"

        if disk_key in saved_keys:
            # 告诉 DCP：把硬盘里的 disk_key 填入到我们内存的 v 里
            state_dict_to_load[disk_key] = v
        elif k in saved_keys:
            # 备用防呆设计：如果没穿马甲，直接加载
            state_dict_to_load[k] = v
        else:
            # 这里的警告才会是真正被跳过的幽灵参数（如生成解码器）
            print(f"⚠️ 跳过分片中未保存的冗余参数 (正常现象): {k}")

    print(f"\n📥 准备从磁盘缝合 {len(state_dict_to_load)} 个核心参数矩阵...")

    # ==========================================
    # 🎯 伪装结界：建立单人本地群聊，骗过 DCP 的强制检查
    # ==========================================
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29511"  # 找一个干净的端口
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", rank=0, world_size=1)

    # 缝合到内存中
    dcp.load(
        state_dict=state_dict_to_load,
        checkpoint_id=checkpoint_path,
    )

    # ==========================================
    # 🎯 精准切割：剥离大语言模型权重，仅保留 CTViT
    # ==========================================
    print("\n✂️ 正在剥离大语言模型权重，仅提取 CTViT 参数...")
    ctvit_pure_state_dict = {}

    prefix = "image_encoder."

    # 从刚刚加载好数据的 state_dict_to_load 中提取视觉层
    for key, value in state_dict_to_load.items():
        if key.startswith(prefix):
            pure_key = key.replace(prefix, "")
            ctvit_pure_state_dict[pure_key] = value

    print(f"📊 提取完成！共提取了 {len(ctvit_pure_state_dict)} 个纯视觉层参数张量。")

    # 保存最终结果
    print(f"\n📤 正在保存 CTViT 最终权重至 {output_path}...")
    torch.save(ctvit_pure_state_dict, output_path)

    # 用完记得销毁假群聊
    dist.destroy_process_group()
    print("✅ 大功告成！纯净版视觉权重提取完毕！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="提取 FSDP 中的纯 CTViT 视觉权重")

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="输入路径：包含 .distcp 文件的 pytorch_model_fsdp_0 文件夹"
    )

    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="输出路径：提取后的 CTViT .pt 文件完整路径"
    )

    parser.add_argument(
        "--qwen_path",
        type=str,
        default="/home/huali/model/Qwen3.5-9B",
        help="Qwen 模型的本地路径"
    )

    args = parser.parse_args()

    if not os.path.exists(args.checkpoint_dir):
        print(f"❌ 严重错误: 找不到输入路径 {args.checkpoint_dir}")
    else:
        output_dir = os.path.dirname(args.output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        extract_ctvit_weights(args.checkpoint_dir, args.output_file, args.qwen_path)
