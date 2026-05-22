import os
import re
import json
import time
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from json_repair import repair_json
import pandas as pd
from tqdm import tqdm
from openai import OpenAI


# =========================
# 路径与运行配置
# =========================

# SRC_PATH = "/mnt/share_data/CT/ct_dataset_base_260316/0316_nodesk_ct_chest_1000.xlsx"
# SRC_PATH = "/home/huali/code/CT-CLIP-main/CT_CLIP/dataset_10000/reports_full.csv"
# SRC_PATH = "/oss/share_data/CT/ct_dataset_eval_260514/ct_dataset_eval_260514.xlsx"
SRC_PATH = "/home/huali/workspace/psj/labels/score.xlsx"
# SAVE_PATH = "/home/huali/workspace/psj/labels/labeled_10000_full.csv"
# SAVE_PATH = "/mnt/huali/ct_dataset_10000/labeled_10000_full.csv"
# SAVE_PATH = "/home/huali/workspace/psj/evaluation_dataset/labeled_eval.xlsx"
SAVE_PATH = "/home/huali/workspace/psj/labels/labeled_score1.xlsx"
LABEL_SET_PATH = "/home/huali/workspace/psj/labels/label_set.xlsx"

TARGET_COL = "影像所见"
BODY_COL = "部位"
EXAM_COL = "检查项目"
OUT_COL = "labels"


# 本地 vLLM / OpenAI 兼容服务（可通过环境变量覆盖）
API_KEY = os.getenv("QWEN_API_KEY", "EMPTY")
BASE_URL = os.getenv("QWEN_BASE_URL", "http://192.168.0.3:8005/v1")
MODEL = os.getenv("QWEN_MODEL", "Qwen3___6-27B")

# 每处理完成多少条数据把当前结果保存一次
SAVE_EVERY = 50
# 每一条报告调用 Qwen API 最多重试次数
MAX_RETRY = 3
# 输出上限；需小于服务端「总上下文」，并给 system+user 提示词留出余量
MAX_COMPLETION_TOKENS = 2048

# 并发数量：同时最多多少条调用本地 Qwen
# 本地 vLLM 建议先用 5~10；若显存/连接报错再调小
CONCURRENT_WORKERS = 50

# 已经有 labels 时是否重新抽取
FORCE_REDO = True
# 测试前几条；全量运行改成 None
TEST_N = 50

# True：只保存 positive/negative/uncertain；False：把 blank 也补齐
OUTPUT_ONLY_NON_BLANK = True
VALID_STATUS = {"positive", "negative", "uncertain", "blank"}


# =========================
# 部位 / 检查项目 归一化
# =========================

# 1. 优先按 CT sheet 的“解剖部位”做精确映射
# CT sheet 中主要就是：头颈、口腔、脊柱、上肢、下肢、胸部、腹部、盆腔、血管
SPECIAL_PART_MAP = {
    # 原始大部位
    "头部": ["头部"],
    "颈部": ["颈部"],
    "胸部": ["胸部"],
    "腹部": ["腹部"],
    "盆部": ["盆腔"],
    "盆腔": ["盆腔"],
    "脊柱": ["脊柱"],
    "骨骼": ["骨骼"],
    "软组织": ["软组织"],
    "脂肪间隙": ["软组织"],

    # 复合部位
    "颈胸部": ["颈部", "胸部"],
    "脊柱与骨盆": ["脊柱", "盆腔", "骨骼"],

    # 头颈/胸腹盆泛部位
    "淋巴结": ["颈部", "胸部", "腹部", "盆腔"],
    "网膜": ["腹部"],
    "血管": ["头部", "颈部", "胸部", "腹部", "盆腔", "骨骼"],

    # 四肢/关节/骨
    "上肢": ["骨骼"],
    "上肢-骨": ["骨骼"],
    "上肢-关节": ["骨骼"],
    "下肢-骨": ["骨骼"],
    "下肢-关节": ["骨骼"],
    "下肢血管": ["骨骼"],

    "踝关节": ["骨骼"],
    "髋关节": ["骨骼"],
    "膝关节": ["骨骼"],
    "足部": ["骨骼"],
}


# 2. 检查项目只处理“会改变或补充部位”的少数特殊情况
EXAM_PART_RULES = [
    # 英文协议名 / 设备协议名
    (["head", "headseq"], ["头部"]),
    (["thorax", "chest", "thoraxroutine"], ["胸部"]),
    (["pelvis", "pelvisroutine"], ["盆腔"]),
    (["abdomen", "abdominal"], ["腹部"]),
    (["shoulder"], ["骨骼"]),
    (["knee", "lowerextremities", "lower extremities"], ["骨骼"]),

    # 血管类
    (["冠脉", "冠状动脉"], ["胸部"]),
    (["头颅cta", "颅脑cta", "头部cta"], ["头部"]),
    (["头颈部cta", "颈部cta", "颈动脉"], ["头部", "颈部"]),
    (["胸主动脉"], ["胸部"]),
    (["腹主动脉"], ["腹部"]),
    (["主动脉"], ["胸部", "腹部"]),
    (["下肢血管", "下肢cta"], ["骨骼"]),

    # 头颈 / 口腔
    (["口腔", "口底", "舌", "牙", "牙槽", "颌面", "面颅骨", "上颌骨", "下颌骨"], ["头部", "颈部"]),
    (["头颅", "颅脑", "头部", "鼻窦", "鼻骨", "眼眶", "颞骨", "乳突", "内听道"], ["头部"]),
    (["颈部", "甲状腺", "鼻咽", "咽", "喉", "腮腺"], ["颈部"]),

    # 胸部
    (["胸部", "肺部", "双肺", "纵隔", "纵膈", "胸膜", "胸腔", "心脏", "胸腺", "食道", "食管"], ["胸部"]),

    # 腹盆跨区
    (["全腹", "全腹部"], ["腹部", "盆腔"]),
    (["下腹", "中下腹"], ["腹部", "盆腔"]),
    (["泌尿系", "尿路", "ctu", "输尿管"], ["腹部", "盆腔"]),

    # 单纯腹部
    (["腹部", "上腹", "中腹", "肝", "胆", "胰", "脾", "肾脏", "双肾", "肾上腺", "阑尾", "门脉"], ["腹部"]),

    # 盆腔
    (["盆腔", "盆部", "膀胱", "前列腺", "子宫", "卵巢", "阴囊", "睾丸"], ["盆腔"]),

    # 脊柱
    (["脊柱", "颈椎", "胸椎", "腰椎", "骶椎", "骶尾椎", "骶尾骨", "椎间盘", "腰间盘", "椎体"], ["脊柱"]),

    # 骨性结构
    (["肋骨", "胸骨", "胸锁关节", "锁骨"], ["胸部", "骨骼"]),
    (["骨盆", "骶髂关节", "坐耻骨", "耻骨"], ["盆腔", "骨骼"]),
    (["颞颌关节", "颞下颌关节"], ["头部", "骨骼"]),
    ([
        "上肢", "下肢", "肩", "肘", "腕", "髋", "膝", "踝", "足", "手",
        "肱骨", "股骨", "胫腓骨", "尺桡骨", "桡骨", "尺骨",
        "跟骨", "跖骨", "趾骨", "掌骨", "指骨", "舟骨", "髌骨"
    ], ["骨骼"]),
]


# 3. 兜底规则：用于 label_set 或部位列不规范时
# 不需要写得特别细，只要能覆盖大类即可
PART_RULES = {
    "头部": [
        "头", "颅", "脑", "眼", "耳", "鼻",
        "鼻窦", "鼻骨", "眼眶", "鞍区", "颅底",
        "颅骨", "颞骨", "乳突", "内听道",
        "颌面", "上颌骨", "下颌骨", "颧弓"
    ],
    "颈部": [
        "颈", "甲状腺", "咽", "喉", "鼻咽",
        "腮腺", "口腔", "口底", "舌"
    ],
    "胸部": [
        "胸", "肺", "纵隔", "纵膈", "胸膜", "胸腔",
        "心", "心包", "气管", "支气管", "食管", "胸腺"
    ],
    "腹部": [
        "腹", "肝", "胆", "胰", "脾", "胃", "肠",
        "阑尾", "肾", "肾上腺", "输尿管", "泌尿系", "尿路"
    ],
    "盆腔": [
        "盆", "膀胱", "前列腺", "子宫", "卵巢",
        "下腹", "腹股沟"
    ],
    "脊柱": [
        "脊柱", "颈椎", "胸椎", "腰椎", "骶椎",
        "骶尾椎", "骶尾骨", "胸腰段", "腰骶段",
        "椎间盘", "腰椎间盘", "腰间盘", "椎体"
    ],
    "骨骼": [
        "关节", "上肢", "下肢",
        "肩", "肘", "腕", "手",
        "髋", "膝", "踝", "足",
        "肋骨", "胸骨", "锁骨",
        "尺桡骨", "桡骨", "尺骨", "肱骨", "股骨", "胫腓骨",
        "跟骨", "跖骨", "趾骨", "掌骨", "指骨", "舟骨", "髌骨",
        "骨盆", "骶髂关节", "坐耻骨", "耻骨"
    ],
    "软组织": ["软组织", "皮下", "肌肉", "脂肪间隙"],
    "全身": ["全身", "多脏器"],
}


def normalize_exam_text(text: str) -> str:
    text = str(text).strip().lower()
    text = text.replace(" ", "")
    text = text.replace("（", "(").replace("）", ")")
    text = text.replace("；", ";").replace("，", ",").replace("、", ",")
    return text

def infer_major_parts(text: str, use_exam_rules: bool = False) -> List[str]:
    """
    use_exam_rules=False：用于 label_set 的“部位”列，按标准部位映射。
    use_exam_rules=True：用于 Excel/CSV 的“检查项目”列，优先按检查项目规则映射。
    """
    raw_text = str(text).strip()
    if not raw_text or raw_text.lower() in {"nan", "none"}:
        return []

    # label_set 中的标准部位，先精确处理
    if raw_text in SPECIAL_PART_MAP:
        return SPECIAL_PART_MAP[raw_text]

    norm_text = normalize_exam_text(raw_text)

    parts = []

    # 只有检查项目才走 EXAM_PART_RULES
    if use_exam_rules:
        for keywords, mapped_parts in EXAM_PART_RULES:
            if any(k.lower() in norm_text for k in keywords):
                parts.extend(mapped_parts)

        # 检查项目已经匹配到时，直接返回，避免 PART_RULES 继续误加“颈部/胸部”等
        if parts:
            return list(dict.fromkeys(parts))

    # label_set 部位，或者检查项目没匹配到时，再兜底
    for major_part, keywords in PART_RULES.items():
        if any(k.lower() in norm_text for k in keywords):
            parts.append(major_part)

    return list(dict.fromkeys(parts)) or ["其他"]


GENERIC_BODY_FORCE_EXAM = {"血管"}


def infer_report_parts_from_row(row: pd.Series) -> List[str]:
    body_text = row.get(BODY_COL, "")
    exam_text = row.get(EXAM_COL, "")

    body_empty = is_empty_value(body_text)
    exam_empty = is_empty_value(exam_text)

    body_raw = "" if body_empty else str(body_text).strip()
    body_parts = [] if body_empty else infer_major_parts(body_text, use_exam_rules=False)
    exam_parts = [] if exam_empty else infer_major_parts(exam_text, use_exam_rules=True)

    # 部位为空：完全靠检查项目推断
    if body_empty:
        return exam_parts if exam_parts else ["其他"]

    # 血管太泛：优先用检查项目判断，例如冠脉CTA、头颅CTA
    if body_raw in GENERIC_BODY_FORCE_EXAM:
        return exam_parts if exam_parts and exam_parts != ["其他"] else body_parts

    # 部位无法识别：用检查项目兜底
    if body_parts == ["其他"]:
        return exam_parts if exam_parts else ["其他"]

    # 检查项目提供额外信息：取并集
    if exam_parts and exam_parts != ["其他"]:
        return list(dict.fromkeys(body_parts + exam_parts))

    return body_parts

def is_empty_value(x) -> bool:
    """判断 Excel/CSV 单元格是否为空。"""
    if pd.isna(x):
        return True
    s = str(x).strip()
    return s == "" or s.lower() in {"nan", "none", "null"}

# =========================
# label_set 读取与候选标签筛选
# =========================

def split_terms(text: str) -> List[str]:
    """拆分同义词/细分类型字段。"""
    text = str(text).strip()
    if not text or text.lower() in {"nan", "none"} or text == "无":
        return []

    # 去掉括号中的说明性长文本，避免污染 prompt
    text = re.sub(r"（.*?）", "", text)
    text = re.sub(r"\(.*?\)", "", text)

    terms = []
    for term in re.split(r"[，,、;；/]\s*", text):
        term = term.strip()
        if not term:
            continue
        if term in {"无", "左侧", "右侧", "双侧"}:
            continue
        if len(term) > 30:
            continue
        terms.append(term)

    return list(dict.fromkeys(terms))


def load_label_set(path: str) -> List[Dict[str, Any]]:
    """
    从 label_set.xlsx 读取：
    - 身体部位：筛选候选标签
    - 异常名称：label key
    - 同义词、细分类别：辅助归一化
    """
    label_map: Dict[str, Dict[str, Any]] = {}

    df = pd.read_excel(path)
    for _, row in df.iterrows():
        raw_part = "" if is_empty_value(row.get("身体部位")) else str(row["身体部位"]).strip()
        label = "" if is_empty_value(row.get("异常名称")) else str(row["异常名称"]).strip()
        alias_text = "" if is_empty_value(row.get("同义词")) else str(row["同义词"]).strip()
        subtype_text = "" if is_empty_value(row.get("细分类别")) else str(row["细分类别"]).strip()

        if not label or label in {"异常名称", "疾病", "征象", "无"}:
            continue
        if len(label) > 30:
            continue

        major_parts = infer_major_parts(raw_part)
        aliases = split_terms(alias_text) + split_terms(subtype_text)
        aliases = [x for x in aliases if x != label]
        aliases = list(dict.fromkeys(aliases))

        if label not in label_map:
            label_map[label] = {
                "label": label,
                "major_parts": major_parts,
                "aliases": aliases,
            }
        else:
            label_map[label]["major_parts"] = list(dict.fromkeys(
                label_map[label]["major_parts"] + major_parts
            ))
            label_map[label]["aliases"] = list(dict.fromkeys(
                label_map[label]["aliases"] + aliases
            ))

    return list(label_map.values())


def select_candidate_items(label_items: List[Dict[str, Any]], report_parts: List[str]) -> List[Dict[str, Any]]:
    """根据报告部位筛选候选异常标签。"""
    report_part_set = set(report_parts)
    selected = []

    for item in label_items:
        item_part_set = set(item.get("major_parts", []))
        if report_part_set & item_part_set:
            selected.append(item)

    return selected


def format_candidates_for_prompt(candidate_items: List[Dict[str, Any]]) -> str:
    """把候选标签整理成 prompt 中的短列表。"""
    lines = []
    for item in candidate_items:
        label = item["label"]
        aliases = item.get("aliases", [])
        if aliases:
            lines.append(f"- {label}：同义词/细分类型包括：{'，'.join(aliases[:12])}")
        else:
            lines.append(f"- {label}")
    return "\n".join(lines)


# =========================
# Prompt 与 Qwen 调用
# =========================

PROMPT_TEMPLATE = """你是一个中文 CT 放射报告标签抽取助手。

任务：
根据输入的“原始影像所见”，只判断候选异常标签的状态。
不要抽取候选标签之外的异常名称。

标签状态：
- positive：原文明确表示该异常存在。
- negative：原文明确否定该异常，例如“未见”“无”“未显示”“未发现”“不明显”。
- uncertain：原文表示可疑、考虑、可能、不除外、待排、倾向、建议复查等不确定表达。
- blank：原文没有提到该标签。

规则：
1. 只输出 JSON，不要输出解释、Markdown 或代码块。
2. labels 的 key 必须来自候选异常标签中的“异常名称”。
3. 如果原文出现同义词、简称、细分类型或近义表达，要归一化到候选异常标签中的异常名称。
4. 不要新增候选异常标签之外的标签。
5. 普通正常描述不要自动转成疾病标签，例如“胸廓对称”“肺纹理清晰”“纵隔居中”“气管通畅”“心影正常”。
6. “未见明显异常”“未见异常密度影”这种泛泛正常描述，不要自动把所有候选标签设为 negative。
7. 只有原文明确定名否定某个异常时，才输出 negative。
8. 如果某个候选标签为 blank，可以不输出。
9. 如果输出某个标签，必须在 evidence 中给出原文中连续的支持短语或完整短句。
10. 对于并列否定，例如“无胸腔积液及胸膜增厚”，应分别输出两个 negative 标签，evidence 都使用包含共同否定语义的完整短语。

候选异常标签：
{candidate_labels}

输出格式：
{{
  "labels": {{
    "异常名称": "positive|negative|uncertain|blank"
  }},
  "evidence": {{
    "异常名称": "支持判断的原文短语"
  }}
}}

输入文本：
{report_text}
"""


def build_prompt(report_text: str, candidate_items: List[Dict[str, Any]]) -> str:
    return PROMPT_TEMPLATE.format(
        candidate_labels=format_candidates_for_prompt(candidate_items),
        report_text=str(report_text).strip(),
    )


def extract_json(text: str) -> Dict[str, Any]:
    """兼容 thinking / markdown / 非严格 JSON 输出。"""
    text = str(text).strip()

    # 去掉 <think>...</think>
    text = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()

    # 去掉 markdown 代码块
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    # 只截取第一个 JSON 对象
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        text = match.group(0)

    # 先尝试标准 JSON
    try:
        return json.loads(text)
    except Exception:
        pass

    # 再尝试修复不严格 JSON：少逗号、尾逗号、引号异常等
    try:
        fixed = repair_json(text)
        return json.loads(fixed)
    except Exception as e:
        raise ValueError(f"无法解析 JSON: {text[:1000]}") from e


def normalize_result(
    obj: Dict[str, Any],
    candidate_items: List[Dict[str, Any]],
    report_parts: List[str],
) -> Dict[str, Any]:
    """只保留候选标签内的合法输出。"""
    if not isinstance(obj, dict):
        obj = {}

    raw_labels = obj.get("labels", {})
    raw_evidence = obj.get("evidence", {})

    if not isinstance(raw_labels, dict):
        raw_labels = {}
    if not isinstance(raw_evidence, dict):
        raw_evidence = {}

    candidate_names = [x["label"] for x in candidate_items]
    candidate_set = set(candidate_names)

    labels = {}
    evidence = {}

    for label, status in raw_labels.items():
        label = str(label).strip()
        status = str(status).strip().lower()

        # 关键约束：模型输出的 label 必须在候选异常标签内
        if label not in candidate_set:
            continue
        if status not in VALID_STATUS:
            continue
        if OUTPUT_ONLY_NON_BLANK and status == "blank":
            continue

        labels[label] = status

        ev = raw_evidence.get(label, "")
        if isinstance(ev, list):
            ev = "；".join(str(x) for x in ev)
        evidence[label] = str(ev).strip()

    if not OUTPUT_ONLY_NON_BLANK:
        for label in candidate_names:
            labels.setdefault(label, "blank")
            evidence.setdefault(label, "")

    return {
        "major_parts": report_parts,
        "labels": labels,
        "evidence": evidence,
    }


def call_qwen(
    client: OpenAI,
    report_text: str,
    candidate_items: List[Dict[str, Any]],
    report_parts: List[str],
) -> Dict[str, Any]:
    if pd.isna(report_text) or not str(report_text).strip():
        return {"major_parts": report_parts, "labels": {}, "evidence": {}}

    prompt = build_prompt(report_text, candidate_items)
    last_error = None

    for attempt in range(1, MAX_RETRY + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "你是一个中文 CT 放射报告标签抽取助手。你必须只输出合法 JSON。不要输出思考过程、解释、Markdown 或代码块。",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=MAX_COMPLETION_TOKENS,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )

            content = response.choices[0].message.content
            obj = extract_json(content)
            return normalize_result(obj, candidate_items, report_parts)

        except Exception as e:
            last_error = e
            print(f"\n第 {attempt} 次调用失败: {e}")
            time.sleep(2 * attempt)

    return {
        "major_parts": report_parts,
        "labels": {},
        "evidence": {},
        "error": str(last_error),
    }


# =========================
# 主流程
# =========================

def _read_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"不支持的文件类型: {ext}，请使用 .csv/.xlsx/.xls: {path}")


def _write_table(df: pd.DataFrame, path: str) -> None:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df.to_csv(path, index=False)
        return
    if ext in {".xlsx", ".xls"}:
        df.to_excel(path, index=False)
        return
    raise ValueError(f"不支持的文件类型: {ext}，请使用 .csv/.xlsx/.xls: {path}")


def load_dataframe() -> pd.DataFrame:
    if os.path.exists(SAVE_PATH):
        print(f"发现已有输出文件，继续处理: {SAVE_PATH}")
        return _read_table(SAVE_PATH)

    print(f"读取源文件: {SRC_PATH}")
    return _read_table(SRC_PATH)

def process_one_row(
    idx: int,
    row: pd.Series,
    label_items: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    部位列 + 检查项目列联合判断
    """
    report_text = row[TARGET_COL]

    report_parts = infer_report_parts_from_row(row)
    candidate_items = select_candidate_items(label_items, report_parts)

    if not candidate_items:
        result = {
            "major_parts": report_parts,
            "labels": {},
            "evidence": {},
            "error": "未筛选到候选异常标签",
        }
    else:
        client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        result = call_qwen(
            client=client,
            report_text=report_text,
            candidate_items=candidate_items,
            report_parts=report_parts,
        )

    return {
        "idx": idx,
        "result": result,
    }


def main():
    label_items = load_label_set(LABEL_SET_PATH)
    print(f"已加载标准异常标签数量: {len(label_items)}")
    print(f"本地 Qwen: base_url={BASE_URL} model={MODEL}")

    df = load_dataframe()

    if TARGET_COL not in df.columns:
        raise ValueError(f"找不到目标列: {TARGET_COL}，当前列名: {list(df.columns)}")
    if BODY_COL not in df.columns and EXAM_COL not in df.columns:
        raise ValueError(
            f"找不到部位列 {BODY_COL}，也找不到检查项目列 {EXAM_COL}，当前列名: {list(df.columns)}"
        )
    if OUT_COL not in df.columns:
        df[OUT_COL] = ""

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    total = len(df) if TEST_N is None else min(len(df), TEST_N)

    pending_indices = []

    for idx in range(total):
        old_value = df.at[idx, OUT_COL]
        if not FORCE_REDO and isinstance(old_value, str) and old_value.strip():
            continue
        pending_indices.append(idx)

    print(f"待处理数量: {len(pending_indices)}")
    print(f"并发数量: {CONCURRENT_WORKERS}")

    finished = 0

    with ThreadPoolExecutor(max_workers=CONCURRENT_WORKERS) as executor:
        futures = {
            executor.submit(
                process_one_row,
                idx,
                df.loc[idx].copy(),
                label_items,
            ): idx
            for idx in pending_indices
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="extracting labels"):
            idx = futures[future]

            try:
                item = future.result()
                result = item["result"]
            except Exception as e:
                result = {
                    "major_parts": [],
                    "labels": {},
                    "evidence": {},
                    "error": f"线程处理失败: {str(e)}",
                }

            df.at[idx, OUT_COL] = json.dumps(result, ensure_ascii=False)

            finished += 1
            if finished % SAVE_EVERY == 0:
                _write_table(df, SAVE_PATH)

    _write_table(df, SAVE_PATH)
    print(f"完成，保存到: {SAVE_PATH}")


if __name__ == "__main__":
    main()