import numpy as np
import json
import os
from sklearn.metrics import roc_curve

# ==========================================
# 1. 路径配置
# ==========================================
results_folder = "./qwen_zeroshot_results/"
npz_file = os.path.join(results_folder, "predictions_and_labels.npz")
output_json_path = os.path.join(results_folder, "optimal_thresholds.json")

def main():
    print("🔍 正在读取 100 例验证集的预测结果...")
    try:
        data = np.load(npz_file, allow_pickle=True)
        all_predictions = data['preds']
        all_labels = data['labels']
        pathologies = data['pathologies']
    except FileNotFoundError:
        print(f"❌ 找不到文件 {npz_file}，请确保路径正确！")
        return

    optimal_thresholds = {}
    valid_count = 0

    print("⚙️ 开始搜索 431 种疾病的最佳阈值 (Youden's J)...")
    
    for i, pathology in enumerate(pathologies):
        y_true = all_labels[:, i]
        y_prob = all_predictions[:, i]

        # 必须同时存在正样本和负样本才能画出 ROC 曲线并计算约登指数
        if len(np.unique(y_true)) > 1:
            fpr, tpr, thresholds = roc_curve(y_true, y_prob)
            
            # 计算约登指数 (TPR - FPR) 并找到最大值的索引
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
            
            optimal_thresholds[pathology] = float(optimal_threshold)
            valid_count += 1
        else:
            # 🚨 兜底机制：如果这 100 例里某罕见病全都是阴性，无法计算阈值，默认给 0.5
            optimal_thresholds[pathology] = 0.5

    # ==========================================
    # 2. 将阈值字典固化保存为 JSON
    # ==========================================
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(optimal_thresholds, f, indent=4, ensure_ascii=False)

    print("\n" + "="*50)
    print(f"✅ 阈值提取全部完成！")
    print(f"📊 共成功计算 {valid_count} 种疾病的动态阈值。")
    print(f"📉 其余 {len(pathologies) - valid_count} 种疾病因验证集中无正样本，已默认设为 0.5。")
    print(f"💾 阈值字典已永久固化至: {output_json_path}")
    print("="*50)

if __name__ == "__main__":
    main()
