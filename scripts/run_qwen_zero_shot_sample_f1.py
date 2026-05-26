import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
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
    def __init__(self, csv_file, data_dir, meta_file, reports_full_file="/home/huali/code/CT-CLIP-main/CT-CLIP/dataset_eval_2000/reports_full.csv", limit=3000):
        print("\n🕵️‍♂️ 启动多级数据对齐管线...")
        
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
        
        # 🚨 修正 1：消除 Pandas 混合类型加载警告
        self.meta_df = pd.read_csv(meta_file, low_memory=False)
        
        print(f"✅ 多级数据拓扑对齐大获全胜！测评数据: {len(self.df)} 例。")
        
    def __len__(self):
        return len(self.df)

    def preprocess_image(self, file_path):
        """完全复刻预训练的黄金预处理管线"""
        nii_img = nib.load(file_path)
        # 🚨 修正 2：CPU 内存防爆机制！强制使用 float32 读取 NIfTI 数据
        img_data = nii_img.get_fdata(dtype=np.float32)

        file_name = os.path.basename(file_path)
        file_name_no_ext = file_name.replace(".nii.gz", "")
        
        row = self.meta_df[self.meta_df['VolumeName'] == file_name]
        if row.empty:
            row = self.meta_df[self.meta_df['VolumeName'] == file_name_no_ext]
            
        if row.empty:
            slope, intercept = 1.0, -1024.0
            xy_spacing, z_spacing = 0.75, 1.5 
        else:
            slope = float(row["RescaleSlope"].iloc[0])
            intercept = float(row["RescaleIntercept"].iloc[0])
            xy_spacing_str = str(row["XYSpacing"].iloc[0])
            xy_spacing = float(xy_spacing_str[1:][:-2].split(",")[0]) if "[" in xy_spacing_str else float(xy_spacing_str.split(",")[0])
            z_spacing = float(row["ZSpacing"].iloc[0])

        img_data = slope * img_data + intercept

        while img_data.ndim > 3: img_data = img_data[..., 0]
        if img_data.ndim == 2: img_data = img_data[:, :, np.newaxis]

        img_data = img_data.transpose(2, 0, 1)
        tensor = torch.tensor(img_data.copy()).unsqueeze(0).unsqueeze(0)

        current = (z_spacing, xy_spacing, xy_spacing)
        target = (1.5, 0.75, 0.75)
        img_data = resize_array(tensor, current, target)[0][0]
        img_data = np.transpose(img_data, (1, 2, 0))

        img_data = np.clip(img_data, -1000, 1000)
        img_data = (img_data / 1000).astype(np.float32)

        tensor = torch.tensor(img_data)
        
        # 🚨 修正 3：动态视野升维！将裁剪中心 Z 轴目标设为 320
        target_shape = (480, 480, 320)
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
                img_tensor = self.preprocess_image(file_path)
            else:
                raise FileNotFoundError
        except Exception as e:
            print(f"⚠️ 警告: 读取 {folder_name} 失败 ({e})")
            # 🚨 修正 4：异常托底 Tensor 必须同步为 320 深度！
            img_tensor = torch.zeros((1, 320, 480, 480), dtype=torch.bfloat16) 
            
        return img_tensor, labels, folder_name

# ==========================================
# 2. 核心评估管线 (全矩阵高通量优化版)
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_folder = "./qwen_zeroshot_2000_v2/"
    os.makedirs(results_folder, exist_ok=True)

    csv_file = "/home/huali/workspace/psj/evaluation_dataset/api/eval/labeled_eval_label_matrix.csv"
    data_dir = "/oss/share_data/CT/ct_dataset_eval_260514/ct_dataset_eval_260514_img/"
    qwen_path = "/home/huali/model/Qwen3.5-9B"
    meta_file = "/oss/share_data/CT/ct_dataset_eval_260514/train_metadata.csv"

    # 🚨 修正 5：挂载我们的权重
    pretrained_weights = "/mnt/huali/ct_dataset_10000/output_v2/CTClip_step_8500_full_fixed.pt"

    dataset = CTLabelMatrixDataset(csv_file=csv_file, data_dir=data_dir, meta_file=meta_file, limit=None)
    pathologies = dataset.pathologies
    
    # 你可以安全地将 batch_size 调至 2 或 4 以加快推理速度
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=4)
    print(f"📊 成功加载数据集，共 {len(dataset)} 例影像，包含 {len(pathologies)} 种疾病标签。")

    print("🏗️ 正在构建 CT-CLIP-Qwen 模型架构...")
    ctvit = CTViT(
        dim=512, codebook_size=8192, image_size=480, patch_size=20, temporal_patch_size=10,
        spatial_depth=4, temporal_depth=4, dim_head=32, heads=8
    )

    tokenizer = AutoTokenizer.from_pretrained(qwen_path, trust_remote_code=True)
    text_model = AutoModel.from_pretrained(qwen_path, trust_remote_code=True, torch_dtype=torch.bfloat16)

    clip = CTCLIP(
        image_encoder=ctvit, text_encoder=text_model, dim_text=4096, dim_image=294912, 
        dim_latent=512, use_mlm=False, extra_latent_projection=False, 
        downsample_image_embeds=False, use_all_token_embeds=False, tokenizer=None
    )

    print(f"📥 正在加载预训练满血权重: {pretrained_weights}")
    state_dict = torch.load(pretrained_weights, map_location="cpu")
    # 虽然是满血权重，保留 strict=False 依然是个好习惯
    clip.load_state_dict(state_dict, strict=False)
    clip.to(device)

    clip.to_text_latent.to(torch.bfloat16)
    clip.visual_transformer.to(torch.float32)
    clip.to_visual_latent.to(torch.float32)
    clip.eval()

    class TokenEscape(Exception):
        def __init__(self, tokens): self.tokens = tokens

    def fake_decode(tokens, *args, **kwargs):
        raise TokenEscape(tokens)
    clip.visual_transformer.decode = fake_decode

    def vq_pre_hook(module, args):
        x = args[0]
        for buffer in module.buffers():
            if buffer.is_floating_point() and buffer.dtype != x.dtype:
                buffer.data = buffer.data.to(x.dtype)
        return args
    if hasattr(clip.visual_transformer, 'vq'):
        clip.visual_transformer.vq.register_forward_pre_hook(vq_pre_hook)

    print("🧠 正在使用 Qwen-9B 预计算中文提示词特征 (Text Feature Caching)...")
    text_features_cache = {}

    with torch.no_grad():
        for pathology in tqdm(pathologies, desc="Caching Text Prompts"):
            texts = [
                f"影像所见：存在{pathology}。影像所得：{pathology}。",
                f"影像所见：未见异常，无{pathology}。影像所得：正常。"
            ]
            tokens = tokenizer(texts, padding="max_length", max_length=64, truncation=True, return_tensors="pt").to(device)
            text_outputs = text_model(**tokens)
            enc_text = text_outputs.last_hidden_state

            input_mask_expanded = tokens.attention_mask.unsqueeze(-1).expand(enc_text.size()).to(enc_text.dtype)
            sum_embeddings = torch.sum(enc_text * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1).clamp(min=1e-9)
            text_embeds = sum_embeddings / sum_mask
            text_embeds = text_embeds.to(torch.bfloat16)
            
            text_latent = clip.to_text_latent(text_embeds)
            text_latent = F.normalize(text_latent, dim=-1) 
            text_features_cache[pathology] = text_latent

    print("\n🧹 文本缓存完毕！正在销毁 Qwen 模型并强制清理显存...")
    if 'text_model' in locals(): del text_model
    if hasattr(clip, 'text_encoder'): del clip.text_encoder
    if hasattr(clip, 'text_transformer'): del clip.text_transformer
    import gc
    gc.collect()             
    torch.cuda.empty_cache() 
    print("✨ 显存清理完毕！可以安全进行大尺度 3D 视觉推理了！")

    print("\n🔄 正在将疾病特征缝合为高性能全并行矩阵...")
    text_latent_list = []
    for pathology in pathologies:
        text_latent_list.append(text_features_cache[pathology])

    stacked_text_latents = torch.cat(text_latent_list, dim=0).to(device)
    stacked_text_latents = F.normalize(stacked_text_latents, dim=-1)

    print("🚀 开始 3D 视觉高通量推理...")
    all_predictions = []
    all_labels = []

    temperature = clip.temperature.item() if hasattr(clip, 'temperature') else 0.07

    with torch.no_grad():
        for img_tensor, labels, patient_id in tqdm(dataloader, desc="Inference"):
            img_tensor = img_tensor.to(device, dtype=torch.float32).contiguous()
            batch_size = img_tensor.shape[0]

            enc_image = clip.visual_transformer(img_tensor, return_encoded_tokens=True)
            enc_image = torch.mean(enc_image, dim=1)
            enc_image = enc_image.view(batch_size, -1)
            image_latent = clip.to_visual_latent(enc_image)
            image_latent = image_latent.to(torch.bfloat16)
            image_latent = F.normalize(image_latent, dim=-1)

            logits = (image_latent @ stacked_text_latents.T) / temperature

            # 🚨 修正 6：消除原本 squeeze(0) 带来的 Batch 崩溃隐患
            # 使其支持任意大小的 inference batch size
            logits = logits.view(batch_size, len(pathologies), 2)
            probs = F.softmax(logits, dim=-1)

            # 提取正样本 (index 0) 概率，支持 Batch 级追加
            patient_preds = probs[:, :, 0].cpu().to(torch.float32).numpy()
            all_predictions.extend(patient_preds.tolist())
            all_labels.extend(labels.numpy().tolist())

            # 保存患者 / 病例 ID，后面按“每个患者一行”计算样本级指标
            if isinstance(patient_id, (list, tuple)):
                all_patient_ids.extend([str(x) for x in patient_id])
            else:
                all_patient_ids.append(str(patient_id))

    print("📈 正在计算各项评估指标：按疾病 Macro + 按患者 Sample-Level... ")

    if len(all_predictions) == 0:
        print("❌ 推理未产生任何结果，结束评估。")
        return

    all_predictions = np.array(all_predictions, dtype=np.float32)
    all_labels = np.array(all_labels, dtype=np.int32)
    all_binary_preds = (all_predictions >= 0.5).astype(np.int32)

    # 保存原始预测，方便后续重新调阈值或复查
    np.savez(
        os.path.join(results_folder, "predictions_and_labels.npz"),
        preds=all_predictions,
        binary_preds=all_binary_preds,
        labels=all_labels,
        patient_ids=np.array(all_patient_ids),
        pathologies=np.array(pathologies),
    )

    # ==========================================================
    # A. 按疾病计算：原来的 macro，是“每一类疾病一个 F1，再平均”
    # ==========================================================
    disease_results_list = []
    valid_disease_metrics = {'AUROC': [], 'F1': [], 'Accuracy': [], 'Precision': [], 'Recall': []}

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
            valid_disease_metrics['AUROC'].append(auc)
            valid_disease_metrics['F1'].append(f1)
            valid_disease_metrics['Accuracy'].append(acc)
            valid_disease_metrics['Precision'].append(pre)
            valid_disease_metrics['Recall'].append(rec)
        else:
            metrics.update({"AUROC": "N/A", "F1": "N/A", "Accuracy": "N/A", "Precision": "N/A", "Recall": "N/A"})

        disease_results_list.append(metrics)

    disease_avg_metrics = {"Pathology": "Disease Macro Average (Valid Only)"}
    for key in valid_disease_metrics:
        disease_avg_metrics[key] = np.mean(valid_disease_metrics[key]) if len(valid_disease_metrics[key]) > 0 else "N/A"
    disease_results_list.append(disease_avg_metrics)

    df_disease_results = pd.DataFrame(disease_results_list)
    df_disease_results.to_excel(
        os.path.join(results_folder, "qwen_zeroshot_disease_macro_metrics.xlsx"),
        index=False,
    )

    # ==========================================================
    # B. 按患者/样本计算：每个患者一行，跨所有疾病标签计算 F1
    #    最终 Sample-Level F1 = 所有患者 F1 的平均值
    # ==========================================================
    patient_results_list = []
    patient_f1_list = []
    patient_precision_list = []
    patient_recall_list = []
    patient_accuracy_list = []

    for idx in range(all_labels.shape[0]):
        y_true = all_labels[idx, :]
        y_pred = all_binary_preds[idx, :]

        sample_f1 = f1_score(y_true, y_pred, zero_division=0)
        sample_pre = precision_score(y_true, y_pred, zero_division=0)
        sample_rec = recall_score(y_true, y_pred, zero_division=0)
        sample_acc = accuracy_score(y_true, y_pred)

        patient_f1_list.append(sample_f1)
        patient_precision_list.append(sample_pre)
        patient_recall_list.append(sample_rec)
        patient_accuracy_list.append(sample_acc)

        patient_results_list.append({
            "PatientID": all_patient_ids[idx] if idx < len(all_patient_ids) else idx,
            "Sample_F1": sample_f1,
            "Sample_Precision": sample_pre,
            "Sample_Recall": sample_rec,
            "Sample_Accuracy": sample_acc,
            "GT_Positive_Count": int(y_true.sum()),
            "Pred_Positive_Count": int(y_pred.sum()),
        })

    sample_avg_metrics = {
        "PatientID": "Sample-Level Average",
        "Sample_F1": float(np.mean(patient_f1_list)) if len(patient_f1_list) > 0 else "N/A",
        "Sample_Precision": float(np.mean(patient_precision_list)) if len(patient_precision_list) > 0 else "N/A",
        "Sample_Recall": float(np.mean(patient_recall_list)) if len(patient_recall_list) > 0 else "N/A",
        "Sample_Accuracy": float(np.mean(patient_accuracy_list)) if len(patient_accuracy_list) > 0 else "N/A",
        "GT_Positive_Count": "-",
        "Pred_Positive_Count": "-",
    }
    patient_results_list.append(sample_avg_metrics)

    df_patient_results = pd.DataFrame(patient_results_list)
    df_patient_results.to_excel(
        os.path.join(results_folder, "qwen_zeroshot_sample_level_metrics.xlsx"),
        index=False,
    )

    # 汇总表：把你最关心的 sample-level F1 放在这里
    summary_metrics = {
        "Disease_Macro_AUROC": disease_avg_metrics["AUROC"],
        "Disease_Macro_F1": disease_avg_metrics["F1"],
        "Disease_Macro_Accuracy": disease_avg_metrics["Accuracy"],
        "Disease_Macro_Precision": disease_avg_metrics["Precision"],
        "Disease_Macro_Recall": disease_avg_metrics["Recall"],
        "Sample_Level_F1": sample_avg_metrics["Sample_F1"],
        "Sample_Level_Accuracy": sample_avg_metrics["Sample_Accuracy"],
        "Sample_Level_Precision": sample_avg_metrics["Sample_Precision"],
        "Sample_Level_Recall": sample_avg_metrics["Sample_Recall"],
        "Num_Samples": int(all_labels.shape[0]),
        "Num_Pathologies": int(len(pathologies)),
        "Threshold": 0.5,
    }
    pd.DataFrame([summary_metrics]).to_excel(
        os.path.join(results_folder, "qwen_zeroshot_summary_metrics.xlsx"),
        index=False,
    )

    print("\n" + "="*60)
    print("🏆 零样本高通量测评最终战报")
    print("="*60)
    def format_metric(val): return f"{val:.4f}" if isinstance(val, (float, np.floating)) else "N/A"

    print("📌 按疾病 Macro-Avg：每类疾病先算一次，再对疾病取平均")
    print(f"   🎯 Disease Macro AUROC:     {format_metric(disease_avg_metrics['AUROC'])}")
    print(f"   🎯 Disease Macro F1-Score:  {format_metric(disease_avg_metrics['F1'])}")
    print(f"   🎯 Disease Macro Accuracy:  {format_metric(disease_avg_metrics['Accuracy'])}")
    print(f"   🎯 Disease Macro Precision: {format_metric(disease_avg_metrics['Precision'])}")
    print(f"   🎯 Disease Macro Recall:    {format_metric(disease_avg_metrics['Recall'])}")

    print("\n📌 按患者 Sample-Level Avg：每个患者跨所有疾病标签算 F1，再对患者取平均")
    print(f"   🎯 Sample-Level F1-Score:   {format_metric(sample_avg_metrics['Sample_F1'])}")
    print(f"   🎯 Sample-Level Accuracy:   {format_metric(sample_avg_metrics['Sample_Accuracy'])}")
    print(f"   🎯 Sample-Level Precision:  {format_metric(sample_avg_metrics['Sample_Precision'])}")
    print(f"   🎯 Sample-Level Recall:     {format_metric(sample_avg_metrics['Sample_Recall'])}")
    print("="*60)
    print("✅ 评估全部完成！")
    print(f"📄 疾病级指标: {os.path.join(results_folder, 'qwen_zeroshot_disease_macro_metrics.xlsx')}")
    print(f"📄 样本级指标: {os.path.join(results_folder, 'qwen_zeroshot_sample_level_metrics.xlsx')}")
    print(f"📄 汇总指标:   {os.path.join(results_folder, 'qwen_zeroshot_summary_metrics.xlsx')}")

if __name__ == "__main__":
    main()
