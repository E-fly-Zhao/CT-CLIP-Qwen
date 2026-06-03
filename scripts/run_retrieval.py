import os
import gc
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import nibabel as nib
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformer_maskgit import CTViT
from ct_clip import CTCLIP
from transformers import AutoTokenizer, AutoModel

# 物理封印 cuDNN 混合精度 Bug
torch.backends.cudnn.enabled = False

# ===================================================================
# 1. 全局重采样引擎
# ===================================================================
def resize_array_global(array, current_xy_spacing, target_xy_spacing, target_z_dim):
    original_shape = array.shape[2:] 
    orig_z, orig_x, orig_y = original_shape
    new_x = max(1, int(orig_x * (current_xy_spacing / target_xy_spacing)))
    new_y = max(1, int(orig_y * (current_xy_spacing / target_xy_spacing)))
    new_shape = [target_z_dim, new_x, new_y]
    resized_array = F.interpolate(array, size=new_shape, mode='trilinear', align_corners=False).cpu().numpy()
    return resized_array

# ===================================================================
# 2. 专用检索数据集 (修复绝对路径覆盖 Bug 版)
# ===================================================================
class CTRetrievalDataset(Dataset):
    def __init__(self, reports_csv, metadata_csv, data_dir):
        print("\n🕵️‍♂️ 启动跨模态图文对齐拓扑检查 (强制两级拼接版)...")
        
        self.reports_df = pd.read_csv(reports_csv, dtype=str)
        self.meta_df = pd.read_csv(metadata_csv, low_memory=False)
        self.data_dir = data_dir
        self.valid_pairs = []
        
        failed_paths_sample = [] # 用于打印死胡同路径
        
        for _, row in self.reports_df.iterrows():
            # 🚨 核心修复：强制切断 CSV 里的旧绝对路径，只提取最后一级目录名！
            raw_folder = str(row.get('SourceFolder', row.get('folder_name', row.get('dicom_path', '')))).strip()
            folder_name = os.path.basename(raw_folder.rstrip('/'))
            
            # 🚨 核心修复：强制只提取文件名！
            raw_vol = str(row.get('VolumeName', row.get('_id', ''))).strip()
            vol_name = os.path.basename(raw_vol)
            
            # 安全拼接：data_dir / folder_name / vol_name
            full_nii_path = os.path.join(data_dir, folder_name, vol_name)
            
            # 容错 1：如果没写后缀，补上后缀
            if not full_nii_path.endswith('.nii.gz') and not os.path.exists(full_nii_path):
                full_nii_path += '.nii.gz'
                
            # 容错 2：如果还是找不到，退一步只看文件夹，抓取里面的第一个 nii.gz
            if not os.path.exists(full_nii_path):
                alt_folder = os.path.join(data_dir, folder_name)
                if os.path.exists(alt_folder) and os.path.isdir(alt_folder):
                    nii_files = [f for f in os.listdir(alt_folder) if f.endswith('.nii.gz')]
                    if nii_files:
                        nii_files.sort()
                        full_nii_path = os.path.join(alt_folder, nii_files[0])
                        vol_name = nii_files[0]
            
            # 最终判定是否真的存在
            if os.path.exists(full_nii_path):
                clin_diag = str(row.get('临床诊断', '')).strip()
                findings = str(row.get('影像所见', '')).strip()
                impression = str(row.get('影像所得', '')).strip()
                
                clin_diag = "" if clin_diag == '未提及' else clin_diag
                combined_text = "".join([f"{k}：{v}。" for k, v in zip(["临床诊断", "影像所见", "影像所得"], [clin_diag, findings, impression]) if v and str(v).lower() not in ['nan', 'none']])
                
                if not combined_text:
                    combined_text = "未提供有效诊断信息。"
                
                self.valid_pairs.append({
                    "nii_path": full_nii_path,
                    "filename": vol_name,
                    "folder_name": folder_name,
                    "report_text": combined_text
                })
            else:
                if len(failed_paths_sample) < 3:
                    failed_paths_sample.append(full_nii_path)
                    
        if len(self.valid_pairs) == 0:
            print(f"\n❌ 严重警告：没有找到任何有效的 NIfTI 文件！")
            print(f"🔍 脚本拼接出的最新寻址路径失败示例：\n" + "\n".join(failed_paths_sample))
            exit(1)
        else:
            print(f"✅ 拓扑筛选大获全胜！成功挂载了 {len(self.valid_pairs)} 对标准 [图-文] 检索基准。")

    def __len__(self):
        return len(self.valid_pairs)

    def preprocess_image(self, file_path):
        try:
            # 🚨 1. 危险区：尝试加载并解析图像灰度阵列
            nii_img = nib.load(file_path)
            img_data = nii_img.get_fdata(dtype=np.float32)

            file_name = os.path.basename(file_path)
            file_name_no_ext = file_name.replace(".nii.gz", "")
            
            row = self.meta_df[self.meta_df['VolumeName'] == file_name]
            if row.empty:
                row = self.meta_df[self.meta_df['VolumeName'] == file_name_no_ext]
                
            if row.empty:
                slope, intercept = 1.0, -1024.0
                xy_spacing = 0.75
            else:
                slope = float(row["RescaleSlope"].iloc[0])
                intercept = float(row["RescaleIntercept"].iloc[0])
                xy_spacing_str = str(row["XYSpacing"].iloc[0])
                xy_spacing = float(xy_spacing_str[1:][:-2].split(",")[0]) if "[" in xy_spacing_str else float(xy_spacing_str.split(",")[0])

            TARGET_D = 320
            TARGET_XY_SPACING = 0.75
            TARGET_H, TARGET_W = 480, 480

            # 🚨 2. 危险区：物理空间缩放与维度转换
            img_data = slope * img_data + intercept

            while img_data.ndim > 3: img_data = img_data[..., 0]
            if img_data.ndim == 2: img_data = img_data[:, :, np.newaxis]

            img_data = img_data.transpose(2, 0, 1)
            tensor = torch.tensor(img_data.copy()).unsqueeze(0).unsqueeze(0)

            img_data = resize_array_global(tensor, xy_spacing, TARGET_XY_SPACING, TARGET_D)[0][0]
            img_data = np.transpose(img_data, (1, 2, 0))

            img_data = np.clip(img_data, -1000, 1000)
            img_data = (img_data / 1000).astype(np.float32)

            tensor = torch.tensor(img_data)
            h, w, d = tensor.shape 

            h_start = max((h - TARGET_H) // 2, 0)
            h_end = min(h_start + TARGET_H, h)
            w_start = max((w - TARGET_W) // 2, 0)
            w_end = min(w_start + TARGET_W, w)

            tensor = tensor[h_start:h_end, w_start:w_end, :]

            pad_h_before = (TARGET_H - tensor.size(0)) // 2
            pad_h_after = TARGET_H - tensor.size(0) - pad_h_before
            pad_w_before = (TARGET_W - tensor.size(1)) // 2
            pad_w_after = TARGET_W - tensor.size(1) - pad_w_before

            tensor = torch.nn.functional.pad(tensor, (0, 0, pad_w_before, pad_w_after, pad_h_before, pad_h_after), value=-1)
            tensor = tensor.permute(2, 0, 1).unsqueeze(0)

            return tensor.to(torch.bfloat16)
            
        except Exception as e:
            # 🛡️ 绝对防爆屏障：捕获一切异常（包含 DTypePromotionError，显存溢出，文件损坏等）
            print(f"\n⚠️ [文件坏死跳过] 无法读取影像: {file_path}")
            print(f"   -> 报错细节: {e}")
            # 返回一个标准的、全0的 3D 张量，保证 DataLoader 和后续视觉编码器不崩溃
            # 维度严格对齐：(1, Z:320, X:480, Y:480)
            return torch.zeros((1, 320, 480, 480), dtype=torch.bfloat16)

    def __getitem__(self, idx):
        item = self.valid_pairs[idx]
        img_tensor = self.preprocess_image(item["nii_path"])
        return img_tensor, item["report_text"], item["filename"], item["folder_name"]

# ===================================================================
# 3. 核心单向检索判定逻辑 (Report -> Volume)
# ===================================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 路径配置
    image_dir = "/oss/share_data/CT/ct_dataset_eval_260514/ct_dataset_eval_260514_img/"
    metadata_csv = "/oss/share_data/CT/ct_dataset_eval_260514/train_metadata_longest_chest.csv"
    reports_csv = "/oss/share_data/CT/ct_dataset_eval_260514/train_reports_longest_chest.csv"
    qwen_model_path = "/home/huali/model/Qwen3.5-9B"
    pretrained_weights = "/data4/huali/ct_dataset_10000/output_v2/CTClip_step_24500_full_fixed.pt"
    
    output_excel_path = "./CT_CLIP_Unidirectional_Retrieval_Step24500_Chest.xlsx"

    # 1. 实例化数据集与装载器
    dataset = CTRetrievalDataset(reports_csv=reports_csv, metadata_csv=metadata_csv, data_dir=image_dir)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=4)

    # 2. 构建完全匹配的空壳网络架构
    print("🏗️ 正在初始化 3D CTViT 骨干与 Qwen-9B 文本底座...")
    ctvit = CTViT(
        dim=512, codebook_size=8192, image_size=480, patch_size=20, temporal_patch_size=10,
        spatial_depth=4, temporal_depth=4, dim_head=32, heads=8
    )
    tokenizer = AutoTokenizer.from_pretrained(qwen_model_path, trust_remote_code=True)
    text_model = AutoModel.from_pretrained(qwen_model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)

    clip = CTCLIP(
        image_encoder=ctvit, text_encoder=text_model, dim_text=4096, dim_image=294912, 
        dim_latent=512, use_mlm=False, extra_latent_projection=False, 
        downsample_image_embeds=False, use_all_token_embeds=False, tokenizer=None
    )

    print(f"📥 注入混合架构满血预训练权重 (兼容部位分类头)...")
    state_dict = torch.load(pretrained_weights, map_location="cpu")
    clip.load_state_dict(state_dict, strict=False)
    clip.to(device)

    clip.to_text_latent.to(torch.bfloat16)
    clip.visual_transformer.to(torch.float32)
    clip.to_visual_latent.to(torch.float32)
    clip.eval()

    # 3. 高通量并行潜表征提取
    print("🚀 启动高通量端到端联合推理（潜表征在线提取中）...")
    all_image_latents = []
    all_text_latents = []
    meta_records = []

    with torch.no_grad():
        for img_tensor, text_list, filenames, folders in tqdm(dataloader, desc="Inference Extraction"):
            batch_size = img_tensor.shape[0]
            img_tensor = img_tensor.to(device, dtype=torch.float32).contiguous()

            # A. 提取图像高级特征
            enc_image = clip.visual_transformer(img_tensor, return_encoded_tokens=True)
            enc_image = torch.mean(enc_image, dim=1).view(batch_size, -1)
            image_latent = clip.to_visual_latent(enc_image).to(torch.float32)
            image_latent = F.normalize(image_latent, dim=-1)
            all_image_latents.append(image_latent.cpu())

            # B. 提取文本高级特征
            tokens = tokenizer(list(text_list), padding="max_length", max_length=512, truncation=True, return_tensors="pt").to(device)
            text_outputs = text_model(**tokens)
            enc_text = text_outputs.last_hidden_state
            input_mask_expanded = tokens.attention_mask.unsqueeze(-1).expand(enc_text.size()).to(enc_text.dtype)
            sum_embeddings = torch.sum(enc_text * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1).clamp(min=1e-9)
            text_embeds = (sum_embeddings / sum_mask).to(torch.bfloat16)
            
            text_latent = clip.to_text_latent(text_embeds).to(torch.float32)
            text_latent = F.normalize(text_latent, dim=-1)
            all_text_latents.append(text_latent.cpu())

            # C. 记录元数据映射信息
            for b in range(batch_size):
                meta_records.append({
                    "PatientFolder": folders[b],
                    "VolumeName": filenames[b],
                    "ReportText": text_list[b]
                })

    img_matrix = torch.cat(all_image_latents, dim=0) # [N, 512]
    txt_matrix = torch.cat(all_text_latents, dim=0) # [N, 512]
    num_samples = img_matrix.shape[0]

    print(f"\n🔮 开始单向点阵交互：构建 [Text-to-Image] 全局相似度对齐矩阵...")
    sim_matrix = (txt_matrix @ img_matrix.T).numpy()

    # ===================================================================
    # 4. 统计与解析单向检索评估指标 (Report -> Volume)
    # ===================================================================
    print("📈 正在计算核心临床评测维度...")
    list_ks = [1, 3, 5, 10, 50]
    
    t2i_ranks = []
    t2i_details = []
    
    for i in range(num_samples):
        scores = sim_matrix[i, :]
        sorted_indices = np.argsort(-scores)
        rank = np.where(sorted_indices == i)[0][0] + 1
        t2i_ranks.append(rank)
        
        top5_cands = []
        for k in range(min(5, num_samples)):
            idx = sorted_indices[k]
            top5_cands.append(f"Rank{k+1}: {meta_records[idx]['VolumeName']}(Sim:{scores[idx]:.4f})")
        
        t2i_details.append({
            "Query_Text": meta_records[i]["ReportText"],
            "True_Image": meta_records[i]["VolumeName"],
            "True_Image_Rank": rank,
            "Top_5_Predicted_Images": " | ".join(top5_cands)
        })

    def compute_retrieval_metrics(ranks):
        ranks_arr = np.array(ranks)
        metrics_dict = {}
        for k in list_ks:
            metrics_dict[f"Recall@{k}"] = float(np.mean(ranks_arr <= k))
        metrics_dict["Mean Rank"] = float(np.mean(ranks_arr))
        metrics_dict["Median Rank"] = float(np.median(ranks_arr))
        return metrics_dict

    t2i_metrics = compute_retrieval_metrics(t2i_ranks)

    summary_data = [{"Retrieval Direction": "Report-to-Volume (Text-to-Image)", **t2i_metrics}]
    summary_df = pd.DataFrame(summary_data)
    detail_df = pd.DataFrame(t2i_details)
    candidates_df = pd.DataFrame(meta_records)

    # ===================================================================
    # 5. 导出 Excel
    # ===================================================================
    print(f"💾 推理判定完成！正在创建分析报表: {output_excel_path}")
    with pd.ExcelWriter(output_excel_path) as writer:
        summary_df.to_excel(writer, sheet_name="retrieval_summary", index=False)
        detail_df.to_excel(writer, sheet_name="retrieval_details", index=False)
        candidates_df.to_excel(writer, sheet_name="candidates", index=False)

    print("\n" + "="*60)
    print("🏆 单向检索 (Report-to-Volume) 最终战报")
    print("="*60)
    print(f"   🎯 Recall@1:  {t2i_metrics['Recall@1']:.4f}  |  Recall@3:  {t2i_metrics['Recall@3']:.4f}")
    print(f"   🎯 Recall@5:  {t2i_metrics['Recall@5']:.4f}  |  Recall@10: {t2i_metrics['Recall@10']:.4f}")
    print(f"   🎯 Recall@50: {t2i_metrics['Recall@50']:.4f}")
    print(f"   🎯 Mean Rank: {t2i_metrics['Mean Rank']:.2f}  |  Median Rank: {t2i_metrics['Median Rank']:.1f}")
    print("="*60)

if __name__ == "__main__":
    main()
