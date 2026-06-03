import os
import pandas as pd
from tqdm import tqdm

# =========================
# 1. 核心映射字典 (完美继承提取脚本逻辑)
# =========================

SPECIAL_PART_MAP = {
    "头部": ["头部"], "颈部": ["颈部"], "胸部": ["胸部"], "腹部": ["腹部"],
    "盆部": ["盆腔"], "盆腔": ["盆腔"], "脊柱": ["脊柱"], "骨骼": ["骨骼"],
    "软组织": ["软组织"], "脂肪间隙": ["软组织"], "颈胸部": ["颈部", "胸部"],
    "脊柱与骨盆": ["脊柱", "盆腔", "骨骼"], "淋巴结": ["颈部", "胸部", "腹部", "盆腔"],
    "网膜": ["腹部"], "血管": ["头部", "颈部", "胸部", "腹部", "盆腔", "骨骼"],
    "上肢": ["骨骼"], "上肢-骨": ["骨骼"], "上肢-关节": ["骨骼"],
    "下肢-骨": ["骨骼"], "下肢-关节": ["骨骼"], "下肢血管": ["骨骼"],
    "踝关节": ["骨骼"], "髋关节": ["骨骼"], "膝关节": ["骨骼"], "足部": ["骨骼"],
}

EXAM_PART_RULES = [
    (["head", "headseq"], ["头部"]),
    (["thorax", "chest", "thoraxroutine"], ["胸部"]),
    (["pelvis", "pelvisroutine"], ["盆腔"]),
    (["abdomen", "abdominal"], ["腹部"]),
    (["shoulder"], ["骨骼"]),
    (["knee", "lowerextremities", "lower extremities"], ["骨骼"]),
    (["冠脉", "冠状动脉"], ["胸部"]),
    (["头颅cta", "颅脑cta", "头部cta"], ["头部"]),
    (["头颈部cta", "颈部cta", "颈动脉"], ["头部", "颈部"]),
    (["胸主动脉"], ["胸部"]),
    (["腹主动脉"], ["腹部"]),
    (["主动脉"], ["胸部", "腹部"]),
    (["下肢血管", "下肢cta"], ["骨骼"]),
    (["口腔", "口底", "舌", "牙", "牙槽", "颌面", "面颅骨", "上颌骨", "下颌骨"], ["头部", "颈部"]),
    (["头颅", "颅脑", "头部", "鼻窦", "鼻骨", "眼眶", "颞骨", "乳突", "内听道"], ["头部"]),
    (["颈部", "甲状腺", "鼻咽", "咽", "喉", "腮腺"], ["颈部"]),
    (["胸部", "肺部", "双肺", "纵隔", "纵膈", "胸膜", "胸腔", "心脏", "胸腺", "食道", "食管"], ["胸部"]),
    (["全腹", "全腹部"], ["腹部", "盆腔"]),
    (["下腹", "中下腹"], ["腹部", "盆腔"]),
    (["泌尿系", "尿路", "ctu", "输尿管"], ["腹部", "盆腔"]),
    (["腹部", "上腹", "中腹", "肝", "胆", "胰", "脾", "肾脏", "双肾", "肾上腺", "阑尾", "门脉"], ["腹部"]),
    (["盆腔", "盆部", "膀胱", "前列腺", "子宫", "卵巢", "阴囊", "睾丸"], ["盆腔"]),
    (["脊柱", "颈椎", "胸椎", "腰椎", "骶椎", "骶尾椎", "骶尾骨", "椎间盘", "腰间盘", "椎体"], ["脊柱"]),
    (["肋骨", "胸骨", "胸锁关节", "锁骨"], ["胸部", "骨骼"]),
    (["骨盆", "骶髂关节", "坐耻骨", "耻骨"], ["盆腔", "骨骼"]),
    (["颞颌关节", "颞下颌关节"], ["头部", "骨骼"]),
    ([
        "上肢", "下肢", "肩", "肘", "腕", "髋", "膝", "踝", "足", "手",
        "肱骨", "股骨", "胫腓骨", "尺桡骨", "桡骨", "尺骨",
        "跟骨", "跖骨", "趾骨", "掌骨", "指骨", "舟骨", "髌骨"
    ], ["骨骼"]),
]

PART_RULES = {
    "头部": ["头", "颅", "脑", "眼", "耳", "鼻", "鼻窦", "鼻骨", "眼眶", "鞍区", "颅底", "颅骨", "颞骨", "乳突", "内听道", "颌面", "上颌骨", "下颌骨", "颧弓"],
    "颈部": ["颈", "甲状腺", "咽", "喉", "鼻咽", "腮腺", "口腔", "口底", "舌"],
    "胸部": ["胸", "肺", "纵隔", "纵膈", "胸膜", "胸腔", "心", "心包", "气管", "支气管", "食管", "胸腺"],
    "腹部": ["腹", "肝", "胆", "胰", "脾", "胃", "肠", "阑尾", "肾", "肾上腺", "输尿管", "泌尿系", "尿路"],
    "盆腔": ["盆", "膀胱", "前列腺", "子宫", "卵巢", "下腹", "腹股沟"],
    "脊柱": ["脊柱", "颈椎", "胸椎", "腰椎", "骶椎", "骶尾椎", "骶尾骨", "胸腰段", "腰骶段", "椎间盘", "腰椎间盘", "腰间盘", "椎体"],
    "骨骼": ["关节", "上肢", "下肢", "肩", "肘", "腕", "手", "髋", "膝", "踝", "足", "肋骨", "胸骨", "锁骨", "尺桡骨", "桡骨", "尺骨", "肱骨", "股骨", "胫腓骨", "跟骨", "跖骨", "趾骨", "掌骨", "指骨", "舟骨", "髌骨", "骨盆", "骶髂关节", "坐耻骨", "耻骨"],
    "软组织": ["软组织", "皮下", "肌肉", "脂肪间隙"],
    "全身": ["全身", "多脏器"],
}

# =========================
# 2. 映射引擎
# =========================

def is_empty_value(x):
    if pd.isna(x): return True
    return str(x).strip().lower() in {"", "nan", "none", "null"}

def normalize_exam_text(text):
    text = str(text).strip().lower()
    text = text.replace(" ", "").replace("（", "(").replace("）", ")")
    text = text.replace("；", ";").replace("，", ",").replace("、", ",")
    return text

def infer_major_parts(text, use_exam_rules=False):
    raw_text = str(text).strip()
    if not raw_text or raw_text.lower() in {"nan", "none"}:
        return []

    if raw_text in SPECIAL_PART_MAP:
        return SPECIAL_PART_MAP[raw_text]

    norm_text = normalize_exam_text(raw_text)
    parts = []

    if use_exam_rules:
        for keywords, mapped_parts in EXAM_PART_RULES:
            if any(k.lower() in norm_text for k in keywords):
                parts.extend(mapped_parts)
        if parts:
            return list(dict.fromkeys(parts))

    for major_part, keywords in PART_RULES.items():
        if any(k.lower() in norm_text for k in keywords):
            parts.append(major_part)

    return list(dict.fromkeys(parts)) or ["其他"]

def process_row(row):
    """
    联合推导逻辑：
    由于当前【部位】列大多为空，核心依赖【检查项目】列进行推导。
    """
    body_text = row.get("部位", "")
    exam_text = row.get("检查项目", "")

    body_empty = is_empty_value(body_text)
    exam_empty = is_empty_value(exam_text)

    body_parts = [] if body_empty else infer_major_parts(body_text, use_exam_rules=False)
    exam_parts = [] if exam_empty else infer_major_parts(exam_text, use_exam_rules=True)

    if body_empty:
        final_parts = exam_parts if exam_parts else ["其他"]
    elif str(body_text).strip() in {"血管"}:
        final_parts = exam_parts if exam_parts and exam_parts != ["其他"] else body_parts
    elif body_parts == ["其他"]:
        final_parts = exam_parts if exam_parts else ["其他"]
    elif exam_parts and exam_parts != ["其他"]:
        final_parts = list(dict.fromkeys(body_parts + exam_parts))
    else:
        final_parts = body_parts

    # 用 "、" 拼接部位列表，例如：["颈部", "胸部"] -> "颈部、胸部"
    return "、".join(final_parts)

# =========================
# 3. 主程序：加载、处理、保存
# =========================
def main():
    #input_file = "/data1/sft/ct_workspace/ct_dataset_base_260428.xlsx"
    input_file = "/home/huali/workspace/xly/CTModel/labeled_eval.xlsx"
    #output_file = "/home/huali/code/CT-CLIP-main/full_reports_body.xlsx"
    output_file = "/home/huali/workspace/xly/CTModel/labeled_eval_body.xlsx"

    print(f"📥 正在加载源数据: {input_file}")
    try:
        df = pd.read_excel(input_file)
    except Exception as e:
        print(f"❌ 读取 Excel 失败: {e}")
        return

    # 检查必选列
    if "部位" not in df.columns:
        print("⚠️ 找不到'部位'列，正在自动创建...")
        df["部位"] = ""
    if "检查项目" not in df.columns:
        print("❌ 严重错误: 找不到'检查项目'列，无法进行推导！请检查源数据。")
        return

    print(f"🧠 开始智能推理映射，共 {len(df)} 条数据...")
    
    # 使用 tqdm 增加进度条显示
    tqdm.pandas(desc="推导部位信息")
    df["部位"] = df.progress_apply(process_row, axis=1)

    print(f"\n💾 映射完成！正在保存至: {output_file}")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_excel(output_file, index=False)
    
    print("✅ 大功告成！")

    # 打印一些统计信息方便你核对
    print("-" * 40)
    print("📊 部位推导结果统计 (前 10 种常见组合):")
    print(df["部位"].value_counts().head(10))
    print("-" * 40)

if __name__ == "__main__":
    main()
