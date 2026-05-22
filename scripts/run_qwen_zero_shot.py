import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
# 导入你的模型架构
from transformer_maskgit import CTViT
from ct_clip import CTCLIP
from transformers import AutoTokenizer, AutoModel, AutoConfig
import nibabel as nib
import torch.nn.functional as F
# 👇 🚨 物理封印 cuDNN，彻底解决 3D Conv 的 bfloat16 崩溃 Bug！
torch.backends.cudnn.enabled = False
# 👆 ========================================================

def resize_array(array, current_spacing, target_spacing):
    original_shape = array.shape[2:]
    scaling_factors = [
        current_spacing[i] / target_spacing[i] for i in range(len(original_shape))
    ]
    new_shape = [
        max(1, int(original_shape[i] * scaling_factors[i])) for i in range(len(original_shape))
    ]
    resized_array = F.interpolate(array, size=new_shape, mode='trilinear', align_corners=False).cpu().numpy()
    return resized_array

class CTLabelMatrixDataset(Dataset):
    #def __init__(self, csv_file, data_dir, meta_file, reports_full_file="/home/huali/code/CT-CLIP-main/CT_CLIP/dataset_10000/reports_full.csv", limit=100):
    def __init__(self, csv_file, data_dir, meta_file, reports_full_file="/home/huali/code/CT-CLIP-main/CT-CLIP/dataset_eval_2000/reports_full.csv", limit=3000):
        print("\n🕵️‍♂️ 启动多级数据对齐管线...")
        
        # --- (之前的数据表对齐逻辑保持不变，略...) ---
        matrix_df = pd.read_csv(csv_file, dtype={0: str})
        matrix_id_col = matrix_df.columns[0]
        matrix_df[matrix_id_col] = matrix_df[matrix_id_col].str.strip()
        self.pathologies = list(matrix_df.columns[1:])
        
        reports_df = pd.read_csv(reports_full_file, dtype=str)
        reports_df['_id'] = reports_df['_id'].str.strip()
        reports_df['folder_name'] = reports_df['dicom_path'].apply(
            lambda x: os.path.basename(str(x)).strip() if pd.notna(x) else ''
        )
        
        merged_df = pd.merge(matrix_df, reports_df[['_id', 'folder_name']], left_on=matrix_id_col, right_on='_id', how='inner')
        
        valid_folder_names = set()
        if os.path.exists(data_dir):
            for item in os.listdir(data_dir):
                if os.path.isdir(os.path.join(data_dir, item)):
                    valid_folder_names.add(str(item).strip())
                    
        self.df = merged_df[merged_df['folder_name'].isin(valid_folder_names)]
        
        if limit is not None and len(self.df) > limit:
            self.df = self.df.head(limit)
            
        self.data_dir = data_dir
        
        # 🚨 新增：加载元数据，用于 HU 换算和 Spacing
        self.meta_df = pd.read_csv(meta_file)
        
        print(f"✅ 多级数据拓扑对齐大获全胜！测评数据: {len(self.df)} 例。")
        
    def __len__(self):
        return len(self.df)

    def preprocess_image(self, file_path):
        """完全复刻预训练的黄金预处理管线"""
        nii_img = nib.load(file_path)
        img_data = nii_img.get_fdata()

        file_name = os.path.basename(file_path)
        file_name_no_ext = file_name.replace(".nii.gz", "")
        
        row = self.meta_df[self.meta_df['VolumeName'] == file_name]
        if row.empty:
            row = self.meta_df[self.meta_df['VolumeName'] == file_name_no_ext]
            
        if row.empty:
            # 容错：如果实在找不到元数据，使用最常见的默认参数
            slope, intercept = 1.0, -1024.0
            xy_spacing, z_spacing = 0.75, 1.5 
        else:
            slope = float(row["RescaleSlope"].iloc[0])
            intercept = float(row["RescaleIntercept"].iloc[0])
            xy_spacing_str = str(row["XYSpacing"].iloc[0])
            xy_spacing = float(xy_spacing_str[1:][:-2].split(",")[0]) if "[" in xy_spacing_str else float(xy_spacing_str.split(",")[0])
            z_spacing = float(row["ZSpacing"].iloc[0])

        # 转 HU
        img_data = slope * img_data + intercept

        # 降维打击结界
        while img_data.ndim > 3: img_data = img_data[..., 0]
        if img_data.ndim == 2: img_data = img_data[:, :, np.newaxis]

        img_data = img_data.transpose(2, 0, 1)
        tensor = torch.tensor(img_data.copy()).unsqueeze(0).unsqueeze(0)

        # 重采样
        current = (z_spacing, xy_spacing, xy_spacing)
        target = (1.5, 0.75, 0.75)
        img_data = resize_array(tensor, current, target)[0][0]
        img_data = np.transpose(img_data, (1, 2, 0))

        # 截断与归一化 (-1000 到 1000)
        img_data = np.clip(img_data, -1000, 1000)
        img_data = (img_data / 1000).astype(np.float32)

        tensor = torch.tensor(img_data)
        
        # 裁剪与 Pad 至严格的 480x480x240
        target_shape = (480, 480, 240)
        h, w, d = tensor.shape
        dh, dw, dd = target_shape
        h_start, h_end = max((h - dh) // 2, 0), min(max((h - dh) // 2, 0) + dh, h)
        w_start, w_end = max((w - dw) // 2, 0), min(max((w - dw) // 2, 0) + dw, w)
        d_start, d_end = max((d - dd) // 2, 0), min(max((d - dd) // 2, 0) + dd, d)

        tensor = tensor[h_start:h_end, w_start:w_end, d_start:d_end]

        pad_h_before = (dh - tensor.size(0)) // 2
        pad_h_after = dh - tensor.size(0) - pad_h_before
        pad_w_before = (dw - tensor.size(1)) // 2
        pad_w_after = dw - tensor.size(1) - pad_w_before
        pad_d_before = (dd - tensor.size(2)) // 2
        pad_d_after = dd - tensor.size(2) - pad_d_before

        tensor = torch.nn.functional.pad(tensor, (pad_d_before, pad_d_after, pad_w_before, pad_w_after, pad_h_before, pad_h_after), value=-1)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)

        # 换上 BFloat16 战袍
        return tensor.to(torch.bfloat16)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        labels = torch.tensor(row[self.pathologies].values.astype(np.float32))
        folder_name = str(row['folder_name'])
        patient_folder = os.path.join(self.data_dir, folder_name)
        
        try:
            files = [f for f in os.listdir(patient_folder) if f.endswith('.nii.gz')]
            if len(files) > 0:
                files.sort()
                file_path = os.path.join(patient_folder, files[0])
                # 🚨 核心：调用原版的黄金预处理！
                img_tensor = self.preprocess_image(file_path)
            else:
                raise FileNotFoundError
        except Exception as e:
            print(f"⚠️ 警告: 读取 {folder_name} 失败 ({e})")
            # 容错：给一个空的标准化张量
            img_tensor = torch.zeros((1, 240, 480, 480), dtype=torch.bfloat16) 
            
        return img_tensor, labels, folder_name

# ==========================================
# 2. 核心评估管线 (全矩阵高通量优化版)
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #results_folder = "./qwen_zeroshot_results/"
    results_folder = "./qwen_zeroshot_2000/"
    os.makedirs(results_folder, exist_ok=True)

    # 路径配置
    #csv_file = "/mnt/huali/ct_dataset_10000/labeled_10000_label_matrix.csv"
    #data_dir = "/mnt/huali/ct_dataset_10000/pretrain_processed_valid_data/"
    csv_file = "/home/huali/workspace/psj/evaluation_dataset/api/eval/labeled_eval_label_matrix.csv"
    data_dir = "/oss/share_data/CT/ct_dataset_eval_260514/ct_dataset_eval_260514_img/"
    qwen_path = "/home/huali/model/Qwen3.5-9B"
    pretrained_weights = "/mnt/huali/ct_dataset_10000/output/CTClip_step_34500_full.pt"
    # 注意：如果你的 CTLabelMatrixDataset 修改了参数要求，请务必传入 meta_file 和 reports_full_file
    #meta_file = "/home/huali/code/CT-CLIP-main/CT_CLIP/dataset_10000/valid_metadata.csv" # 请确保路径正确
    meta_file = "/oss/share_data/CT/ct_dataset_eval_260514/train_metadata.csv"

    # 初始化 Dataset
    dataset = CTLabelMatrixDataset(
        csv_file=csv_file, 
        data_dir=data_dir, 
        meta_file=meta_file, 
        limit=None
    )
    pathologies = dataset.pathologies
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    print(f"📊 成功加载数据集，共 {len(dataset)} 例影像，包含 {len(pathologies)} 种疾病标签。")
    # ------------------------------------------
    # 模型初始化 (🚨 严格 1:1 像素级复刻预训练配置 🚨)
    # ------------------------------------------
    print("🏗️ 正在构建 CT-CLIP-Qwen 模型架构...")

    # 1. 恢复深度为 4 的满血视觉骨干！
    ctvit = CTViT(
        dim=512,
        codebook_size=8192,
        image_size=480,
        patch_size=20,
        temporal_patch_size=10,
        spatial_depth=4,   # 👈 核心修正：恢复为 4
        temporal_depth=4,  # 👈 核心修正：恢复为 4
        dim_head=32,
        heads=8
    )

    tokenizer = AutoTokenizer.from_pretrained(qwen_path, trust_remote_code=True)
    text_model = AutoModel.from_pretrained(
        qwen_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )

    # 2. 恢复宏观架构的所有开关和维度！
    clip = CTCLIP(
        image_encoder=ctvit,
        text_encoder=text_model,
        dim_text=4096,               # Qwen 的真实维度
        dim_image=294912,            # 👈 核心修正：与预训练完全一致
        dim_latent=512,
        use_mlm=False,               # 👈 核心修正：与预训练完全一致
        extra_latent_projection=False, # 👈 核心修正：与预训练完全一致
        downsample_image_embeds=False, # 👈 核心修正：与预训练完全一致
        use_all_token_embeds=False,    # 👈 核心修正：与预训练完全一致
        tokenizer=None
    )


    # 🚨 加载预训练权重
    print(f"📥 正在加载预训练权重: {pretrained_weights}")
    state_dict = torch.load(pretrained_weights, map_location="cpu")
    clip.load_state_dict(state_dict, strict=False)
    clip.to(device)
    # 👇 🚨 终极绝招：模态精度隔离 🚨 👇
    # 文本投影层：强制使用 BFloat16，与 Qwen 完美对齐
    clip.to_text_latent.to(torch.bfloat16)

    # 视觉侧：保留纯 Float32，彻底屏蔽 VQ 层的混合精度 Bug！
    clip.visual_transformer.to(torch.float32)
    clip.to_visual_latent.to(torch.float32)
    # 👆 ============================= 👆
    clip.eval()

    # 👇 🚨 终极绝杀 2.0：偷天换日，物理封印 Decoder 🚨 👇
    # 1. 定义一个专属逃生舱，专门用来装载高维特征
    class TokenEscape(Exception):
        def __init__(self, tokens):
            self.tokens = tokens

    # 2. 部署报警器：一旦特征送进 decode，立刻将其装入逃生舱并强制弹出！
    def fake_decode(tokens, *args, **kwargs):
        raise TokenEscape(tokens)

    clip.visual_transformer.decode = fake_decode
    # 👆 =================================================== 👆

    # 👇 ========================================== 👇
    # 🚨 终极补丁：复刻预训练时的 VQ Buffer 精度同步 Hook
    # 强制把底层顽固的密码本对齐到 BFloat16
    # ==========================================
    def vq_pre_hook(module, args):
        x = args[0]
        # 动态遍历 VQ 层的所有 Buffer，强制对齐到输入图像的精度
        for buffer in module.buffers():
            if buffer.is_floating_point() and buffer.dtype != x.dtype:
                buffer.data = buffer.data.to(x.dtype)
        return args

    # 将 Hook 挂载到视觉骨干的 vq 模块上
    if hasattr(clip.visual_transformer, 'vq'):
        clip.visual_transformer.vq.register_forward_pre_hook(vq_pre_hook)
    # 👆 ========================================== 👆

    # ------------------------------------------
    # 核心优化：预计算并缓存所有疾病的文本特征
    # ------------------------------------------
    print("🧠 正在使用 Qwen-9B 预计算中文提示词特征 (Text Feature Caching)...")
    text_features_cache = {}

    with torch.no_grad():
        for pathology in tqdm(pathologies, desc="Caching Text Prompts"):
            # 👇 🚨 修复 1：使用高度贴合预训练分布的中文正负提示词
            texts = [
                f"影像所见：存在{pathology}。影像所得：{pathology}。",
                f"影像所见：未见异常，无{pathology}。影像所得：正常。"
            ]

            tokens = tokenizer(texts, padding="max_length", max_length=64, truncation=True, return_tensors="pt").to(device)
            text_outputs = text_model(**tokens)
            enc_text = text_outputs.last_hidden_state

            # 👇 🚨 修复 2：100% 还原预训练源码的 Masked Mean Pooling (精度安全版)
            # 动态对齐掩码精度，Qwen 是 bf16 这里就是 bf16
            input_mask_expanded = tokens.attention_mask.unsqueeze(-1).expand(enc_text.size()).to(enc_text.dtype)
            
            sum_embeddings = torch.sum(enc_text * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1).clamp(min=1e-9)
            text_embeds = sum_embeddings / sum_mask

            # 🚨 终极保险：强转回 BFloat16，完美喂给投影层
            text_embeds = text_embeds.to(torch.bfloat16)
            
            # 穿过文本投影层并进行 L2 归一化
            text_latent = clip.to_text_latent(text_embeds)
            text_latent = F.normalize(text_latent, dim=-1) # 形状: [2, 512]
            text_features_cache[pathology] = text_latent

    print("\n🧹 文本缓存完毕！正在销毁 Qwen 模型并强制清理显存...")
    if 'text_model' in locals(): del text_model
    if hasattr(clip, 'text_encoder'): del clip.text_encoder
    if hasattr(clip, 'text_transformer'): del clip.text_transformer
    import gc
    gc.collect()             
    torch.cuda.empty_cache() 
    print("✨ 显存清理完毕！可以安全进行大尺度 3D 视觉推理了！")

    # ==========================================
    # ⚡ 并行矩阵构建
    # ==========================================
    print("\n🔄 正在将疾病特征缝合为高性能全并行矩阵...")
    text_latent_list = []
    for pathology in pathologies:
        text_latent_list.append(text_features_cache[pathology])

    # 拼接后形状: [Num_Pathologies * 2, 512]
    stacked_text_latents = torch.cat(text_latent_list, dim=0).to(device)
    stacked_text_latents = F.normalize(stacked_text_latents, dim=-1)

    # ------------------------------------------
    # 零样本高通量推理
    # ------------------------------------------
    print("🚀 开始 3D 视觉高通量推理...")
    all_predictions = []
    all_labels = []

    temperature = clip.temperature.item() if hasattr(clip, 'temperature') else 0.07

    with torch.no_grad():
        for img_tensor, labels, patient_id in tqdm(dataloader, desc="Inference"):
            img_tensor = img_tensor.to(device, dtype=torch.float32).contiguous()

            # 👇 🚨 修复 3：100% 还原预训练源码的视觉特征流 🚨 👇
            # 1. 开启 return_encoded_tokens=True，原生跳过 VQ 和 Decode
            enc_image = clip.visual_transformer(img_tensor, return_encoded_tokens=True)

            # 2. 复刻预训练的 Mean Pooling (对 dim=1 求均值)
            enc_image = torch.mean(enc_image, dim=1)

            # 3. 强行把多维特征压平为 [Batch, 294912] 形状，完美喂给满血投影层！
            enc_image = enc_image.view(enc_image.shape[0], -1)

            # 4. 投影至 512 维潜空间
            image_latent = clip.to_visual_latent(enc_image)
            # 👆 ========================================================== 👆

            # 穿过模态桥梁！将算出的视觉向量强转为 BFloat16，准备与文本矩阵会师
            image_latent = image_latent.to(torch.bfloat16)

            # 视觉特征 L2 归一化，激活余弦相似度空间
            image_latent = F.normalize(image_latent, dim=-1)

            # 🎯 降维打击：并行矩阵乘法
            logits = (image_latent @ stacked_text_latents.T) / temperature

            # 形状重塑：压缩 Batch 维度，恢复为 [num_pathologies, 2]
            logits = logits.squeeze(0).view(len(pathologies), 2)

            # 概率激活
            probs = F.softmax(logits, dim=-1)

            # 提取正样本概率并记录
            patient_preds = probs[:, 0].cpu().to(torch.float32).numpy().tolist()
            all_predictions.append(patient_preds)
            all_labels.append(labels.numpy()[0])

    # ------------------------------------------
    # 评估与结果保存 (全指标多维分析版)
    # ------------------------------------------
    print("📈 正在计算各项评估指标 (AUROC, F1, ACC, Precision, Recall)...")

    if len(all_predictions) == 0:
        print("❌ 推理未产生任何结果，结束评估。")
        return

    all_predictions = np.array(all_predictions) # Shape: [Batch, 431]
    all_labels = np.array(all_labels)           # Shape: [Batch, 431]

    # 将概率通过 0.5 阈值二值化
    all_binary_preds = (all_predictions >= 0.5).astype(int)

    np.savez(os.path.join(results_folder, "predictions_and_labels.npz"),
             preds=all_predictions, labels=all_labels, pathologies=pathologies)

    results_list = []
    valid_metrics = {'AUROC': [], 'F1': [], 'Accuracy': [], 'Precision': [], 'Recall': []}

    for i, pathology in enumerate(pathologies):
        y_true = all_labels[:, i]
        y_prob = all_predictions[:, i]
        y_pred = all_binary_preds[:, i]

        metrics = {"Pathology": pathology}

        if len(np.unique(y_true)) > 1:
            auc = roc_auc_score(y_true, y_prob)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            acc = accuracy_score(y_true, y_pred)
            pre = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)

            metrics.update({"AUROC": auc, "F1": f1, "Accuracy": acc, "Precision": pre, "Recall": rec})

            valid_metrics['AUROC'].append(auc)
            valid_metrics['F1'].append(f1)
            valid_metrics['Accuracy'].append(acc)
            valid_metrics['Precision'].append(pre)
            valid_metrics['Recall'].append(rec)
        else:
            metrics.update({"AUROC": "N/A", "F1": "N/A", "Accuracy": "N/A", "Precision": "N/A", "Recall": "N/A"})

        results_list.append(metrics)

    avg_metrics = {"Pathology": "Average (Valid Only)"}
    for key in valid_metrics:
        avg_metrics[key] = np.mean(valid_metrics[key]) if len(valid_metrics[key]) > 0 else "N/A"

    results_list.append(avg_metrics)

    df_results = pd.DataFrame(results_list)
    df_results.to_excel(os.path.join(results_folder, "qwen_zeroshot_full_metrics.xlsx"), index=False)

    print("\n" + "="*50)
    print("🏆 零样本高通量测评最终战报 (宏平均 Macro-Avg):")
    print("="*50)

    def format_metric(val):
        return f"{val:.4f}" if isinstance(val, float) else "N/A"

    print(f"   🎯 AUROC:     {format_metric(avg_metrics['AUROC'])}")
    print(f"   🎯 F1-Score:  {format_metric(avg_metrics['F1'])}")
    print(f"   🎯 Accuracy:  {format_metric(avg_metrics['Accuracy'])}")
    print(f"   🎯 Precision: {format_metric(avg_metrics['Precision'])}")
    print(f"   🎯 Recall:    {format_metric(avg_metrics['Recall'])}")
    print("="*50)
    print("✅ 评估全部完成！详细的逐疾病报表已保存至 qwen_zeroshot_results 目录下。")


if __name__ == "__main__":
    main()
