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
# from transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer, AutoModel
from ct_clip import CTCLIP
from CTCLIPTrainer import CTClipTrainer

import torch
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

# tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized',do_lower_case=True)

# text_encoder = BertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")

# dim_text = text_encoder.config.hidden_size # 自动获取 9B 的 hidden size
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
# 3. 组装 CTCLIP 宏观架构
# =====================================================================
clip = CTCLIP(
    image_encoder = image_encoder,
    text_encoder = text_encoder,
    dim_text = dim_text,  # 传入动态获取的 Qwen 维度
    dim_image = 294912,
    dim_latent = 512,
    use_mlm = False,      # 必须关闭 MLM，Qwen是 Decoder-only 不支持 MLM
    extra_latent_projection = False,
    downsample_image_embeds = False,
    use_all_token_embeds = False
)
#dim_image = 131072,


# clip = CTCLIP(
#     image_encoder = image_encoder,
#     text_encoder = text_encoder,
#     dim_text = 768,
#     dim_image = 294912,
#     dim_latent = 512,
#     extra_latent_projection = False,         # whether to use separate projections for text-to-image vs image-to-text comparisons (CLOOB)
#     use_mlm=False,
#     downsample_image_embeds = False,
#     use_all_token_embeds = False

# )

# 👇 2. 物理级强制全模型转为 BFloat16
# 这一步极其关键：它把 FSDP 管不到的 VQ EMA 密码本(Buffer)永久变成了 BFloat16！
clip.to(torch.bfloat16)

# =====================================================================
# 🚨 终极补丁 1：不可绕过的 FSDP Buffer 精度同步 Hook
# =====================================================================
def vq_pre_hook(module, args):
    # args[0] 就是进入 VQ 层的输入图像张量 x (此时是 BFloat16)
    x = args[0]
    
    # 动态遍历 VQ 层及其子层的所有 Buffer (包括 EMA 密码本 embed)
    # 将它们强制对齐到输入 x 的精度
    for buffer in module.buffers():
        if buffer.is_floating_point() and buffer.dtype != x.dtype:
            buffer.data = buffer.data.to(x.dtype)
            
    return args

# 注册官方 Hook！FSDP 绝对无法绕过它！
clip.visual_transformer.vq.register_forward_pre_hook(vq_pre_hook)

# =====================================================================
# 补丁 2：CTCLIP 图像输入拦截器 (依然保留，负责把大门守好)
# =====================================================================
original_clip_forward = CTCLIP.forward

def new_forward(self, text, image, **kwargs):
    if image is not None:
        # 确保 DataLoader 进来的图像一开始就是 BFloat16
        image = image.to(torch.bfloat16)
    return original_clip_forward(self, text, image, **kwargs)

if not hasattr(CTCLIP, 'original_forward'):
    CTCLIP.original_forward = CTCLIP.forward
    CTCLIP.forward = new_forward

# =====================================================================
# 🚨 5. 绝对安全的 FSDP 梯度冻结逻辑 (在实例化之后执行！)
# =====================================================================
# 步骤 A：暴力唤醒所有参数，确保所有视觉层和投影层处于就绪状态
for param in clip.parameters():
    param.requires_grad = True

# 步骤 B：精准狙击，只冻结 Qwen 语言大脑（在 CTCLIP 中名为 text_transformer）
# 这样不仅冻结了 Qwen，还确保了 clip.to_text_latent 等投影层仍然可以被训练！
for param in clip.text_transformer.parameters():
    param.requires_grad = False

print("✅ Qwen 权重已彻底物理冻结，视觉端与投影层梯度已唤醒。")

# =====================================================================
# 6. 实例化 Trainer
# =====================================================================

# trainer = CTClipTrainer(
#     clip,
#     # reports_file_train= "path_to_train_reports_csv", #TODO: Path to train reports CSV
#     # reports_file_valid= "path_to_validation_reports_csv", #TODO: Path to validation reports CSV
#     # data_train= "path_to_preprocessed_train", #TODO: Path to preprocessed train data
#     # data_valid = "path_to_preprocessed_valid", #TODO: Path to preprocessed validation data
#     reports_file_train = "/home/huali/code/CT-CLIP-main/CT_CLIP/dataset/train_reports.csv", 
#     reports_file_valid = "/home/huali/code/CT-CLIP-main/CT_CLIP/dataset/valid_reports.csv",
#     data_train = "/home/huali/code/CT-CLIP-main/dataset/pretrain_processed_train_data", 
#     data_valid = "/home/huali/code/CT-CLIP-main/dataset/pretrain_processed_valid_data",
#     train_meta_file = "/home/huali/code/CT-CLIP-main/CT_CLIP/dataset/train_metadata.csv", #TODO: Path to train metadata CSV
#     valid_meta_file = "/home/huali/code/CT-CLIP-main/CT_CLIP/dataset/valid_metadata.csv", #TODO: Path to validation metadata CSV
#     labels = "path_to_validation_labels_csv", #TODO: Path to validation labels CSV
#     batch_size = 2,
#     results_folder="output_folder", #TODO: Path to save output results
#     # num_train_steps = 100001,
#     num_train_steps = 1000,   # 对于 100 个数据，1000 步相当于跑了将近 80 个 Epoch，足够测试收敛性
#     save_model_every = 50,    # 每 50 步保存一次（大约每 4 个 Epoch 保存一次）
#     save_results_every = 50,  # 验证集评估也同步改为 50 步一次
#     num_workers = 4,
# )
# ✅ 完全替换为你干净的初始化参数：
trainer = CTClipTrainer(
    clip,
    # reports_file_train = "/home/huali/code/CT-CLIP-main/CT_CLIP/dataset/train_reports.csv", 
    # reports_file_valid = "/home/huali/code/CT-CLIP-main/CT_CLIP/dataset/valid_reports.csv",
    # data_train = "/mnt/share_data/CT/ct_dataset_base_260316/pretrain_processed_data", 
    # data_valid = "/mnt/share_data/CT/ct_dataset_base_260316/pretrain_processed_valid_data",
    # train_meta_file = "/home/huali/code/CT-CLIP-main/CT_CLIP/dataset/train_metadata.csv",
    # valid_meta_file = "/home/huali/code/CT-CLIP-main/CT_CLIP/dataset/valid_metadata.csv",
    
    reports_file_train = "/home/huali/code/CT-CLIP-main/CT_CLIP/dataset_10000/train_reports.csv", 
    reports_file_valid = "/home/huali/code/CT-CLIP-main/CT_CLIP/dataset_10000/valid_reports.csv",
    data_train = "/mnt/huali/ct_dataset_10000/pretrain_processed_train_data", 
    data_valid = "/mnt/huali/ct_dataset_10000/pretrain_processed_valid_data",
    train_meta_file = "/home/huali/code/CT-CLIP-main/CT_CLIP/dataset_10000/train_metadata.csv",
    valid_meta_file = "/home/huali/code/CT-CLIP-main/CT_CLIP/dataset_10000/valid_metadata.csv",
    # 【彻底移除 labels, train_meta_file, valid_meta_file 的传参】
    
    batch_size = 2,
    # results_folder="/mnt/share_data/CT/ct_dataset_base_260316/output_100",
    results_folder="/mnt/huali/ct_dataset_10000/output",
    #results_folder="/data2/pretrain_ct_chest_1000_output_folder", 
    # num_train_steps = 1000,     
    # save_results_every = 50,  
    # save_model_every = 50,   
    # num_workers = 4,
    # 🚀 生产级参数
    num_train_steps = 35000,    # 大约跑 110 个 Epoch
    save_model_every = 500,     # 大约每 1.5 个 Epoch 保存一次模型
    save_results_every = 500,   # 大约每 1.5 个 Epoch 验证一次
    num_workers = 8,            # 提升数据吞吐量
)

trainer.train()


