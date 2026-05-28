import sys
import os

# 暴力注入绝对路径，防止 DeepSpeed 子进程迷失方向
base_path = "/home/huali/code/CT-CLIP-main"
# 在你原来的代码最开头加上这一句，防止系统级别的内存碎片
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, base_path)
sys.path.insert(0, os.path.join(base_path, "transformer_maskgit"))
sys.path.insert(0, os.path.join(base_path, "CT_CLIP"))

import torch

from transformer_maskgit import CTViT
from transformers import AutoTokenizer, AutoModel
from ct_clip import CTCLIP
from CTCLIPTrainer_body import CTClipTrainer

import torch.backends.cudnn as cudnn

# 关闭 cuDNN 的 Benchmark，防止它在超大 3D 卷积时暴力搜索导致 OOM
cudnn.benchmark = False
# 直接禁用 cuDNN，强迫 PyTorch 使用原生底层算子执行 3D 卷积，绕过底层 Bug 和 Workspace OOM
cudnn.enabled = False
# 清理可能残留的显存碎片
torch.cuda.empty_cache()


# =====================================================================
# 1. 挂载 Qwen3.5-9B (此时不进行任何参数冻结操作)
# =====================================================================
qwen_path = "/home/huali/model/Qwen3.5-9B"
tokenizer = AutoTokenizer.from_pretrained(qwen_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = '<|endoftext|>' # 强制设置 PAD token

text_encoder = AutoModel.from_pretrained(
    qwen_path, 
    trust_remote_code=True,
    torch_dtype=torch.bfloat16 # 保持加载时为 bf16
)

# 动态获取文本特征维度，兼容不同命名规范的配置文件
if hasattr(text_encoder.config, 'hidden_size'):
    dim_text = text_encoder.config.hidden_size
elif hasattr(text_encoder.config, 'd_model'):
    dim_text = text_encoder.config.d_model
else:
    # 终极保底方法：直接获取模型 Embedding 层的真实输出维度
    dim_text = text_encoder.get_input_embeddings().weight.shape[1]

print(f"✅ 成功获取文本编码器维度: {dim_text}")
print("---------")
print(f"PAD Token ID: {tokenizer.pad_token_id}")
print(f"MASK Token ID: {tokenizer.mask_token_id}")
print("-----------")


# =====================================================================
# 2. 实例化视觉编码器
# =====================================================================
image_encoder = CTViT(
    dim = 512,
    codebook_size = 8192,
    image_size = 480,
    patch_size = 20,
    temporal_patch_size = 10,
    spatial_depth = 4,
    temporal_depth = 4,
    dim_head = 32,
    heads = 8
)

# =====================================================================
# 📊 🔥 核心探针：全部位大模型 dim_image 动态特征对齐引擎 🔥
# =====================================================================
# 🚨 绕过底层 Bug 的核心魔法：在模型运行前，强制劫持 PyTorch 的默认设备！
local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)  # <--- 这句极其关键！让底层的 'cuda' 默认指向正确的卡
temp_device = torch.device(f"cuda:{local_rank}")

print(f"⏳ 正在根据全新全身视野 (D=320) 动态推导 Vision Encoder 的维度 (在设备 {temp_device} 上)...")

# 将模型放置到当前专属显卡
image_encoder.to(temp_device)
image_encoder.eval()

with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    # 创建虚拟张量，放在当前显卡
    dummy_img = torch.zeros(1, 1, 320, 480, 480).to(temp_device)

    # 此时进入原版 ctvit.py。当它执行 device=torch.device('cuda') 时，
    # 会因为上面的 set_device，自动被重定向到 temp_device！完美绕过冲突！
    dummy_enc = image_encoder(dummy_img, return_encoded_tokens=True)
    dummy_enc = torch.mean(dummy_enc, dim=1)

    dynamic_dim_image = dummy_enc.view(1, -1).shape[1]

# 探针测算完毕，把模型放回 CPU，交还给 FSDP 正常管理
image_encoder.to("cpu")
image_encoder.train()

print(f"✅ 动态对齐成功！底层 Bug 绕过！全部位维度为: {dynamic_dim_image}")
print("="*60)


# =====================================================================
# 3. 组装 CTCLIP 宏观架构
# =====================================================================
clip = CTCLIP(
    image_encoder = image_encoder,
    text_encoder = text_encoder,
    dim_text = dim_text,  # 传入动态获取的 Qwen 维度 (4096)
    dim_image = dynamic_dim_image,  # 🚨 完美注入动态维度！
    dim_latent = 512,
    use_mlm = False,      
    extra_latent_projection = False,
    downsample_image_embeds = False,
    use_all_token_embeds = False
)

# 物理级强制全模型转为 BFloat16
clip.to(torch.bfloat16)

# =====================================================================
# 🚨 终极补丁 1：不可绕过的 FSDP Buffer 精度同步 Hook
# =====================================================================
def vq_pre_hook(module, args):
    x = args[0]
    for buffer in module.buffers():
        if buffer.is_floating_point() and buffer.dtype != x.dtype:
            buffer.data = buffer.data.to(x.dtype)
    return args

clip.visual_transformer.vq.register_forward_pre_hook(vq_pre_hook)

# =====================================================================
# 补丁 2：CTCLIP 图像输入拦截器 
# =====================================================================
original_clip_forward = CTCLIP.forward

def new_forward(self, text, image, **kwargs):
    if image is not None:
        image = image.to(torch.bfloat16)
    return original_clip_forward(self, text, image, **kwargs)

if not hasattr(CTCLIP, 'original_forward'):
    CTCLIP.original_forward = CTCLIP.forward
    CTCLIP.forward = new_forward

# =====================================================================
# 🚨 5. 绝对安全的 FSDP 梯度冻结逻辑 
# =====================================================================
for param in clip.parameters():
    param.requires_grad = True

for param in clip.text_transformer.parameters():
    param.requires_grad = False

print("✅ Qwen 权重已彻底物理冻结，视觉端与投影层梯度已唤醒。")

# =====================================================================
# 6. 实例化 Trainer 
# =====================================================================
base_csv_dir = "/home/huali/code/CT-CLIP-main/CT_CLIP/dataset_10000"

trainer = CTClipTrainer(
    clip,
    # 接入清洗后的全身数据
    data_train = f"{base_csv_dir}/filtered_thin_slices_train.csv", 
    data_valid = f"{base_csv_dir}/filtered_thin_slices_valid.csv",
    reports_file_train = f"{base_csv_dir}/filtered_train_reports.csv", 
    reports_file_valid = f"{base_csv_dir}/filtered_valid_reports.csv",
    train_meta_file = f"{base_csv_dir}/filtered_train_metadata.csv",
    valid_meta_file = f"{base_csv_dir}/filtered_valid_metadata.csv",
    
    batch_size = 2,
    results_folder="/mnt/huali/ct_dataset_10000/output_v2_body",
    
    num_train_steps = 35000,    
    save_model_every = 500,     
    save_results_every = 500,   
    num_workers = 8,            
)

trainer.train()
