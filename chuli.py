import json
import pandas as pd
from pathlib import Path

# ===== 修改成你的文件路径 =====
PATH = "mimic_annotation_all.json"
OUT_CSV = "structured.csv"
OUT_JSONL = "structured.jsonl"

def normalize_split(name: str) -> str:
    if not name:
        return "unknown"
    name = name.strip().lower()
    if name in {"val", "validation", "dev"}:
        return "val"
    if name in {"train", "test"}:
        return name
    return name

def is_empty_report(v) -> bool:
    if v is None:
        return True
    if isinstance(v, str):
        return len(v.strip()) == 0
    return False

def to_json_str(x):
    """把列表/字典安全保存为JSON字符串，便于放进CSV；标量直接返回。"""
    if isinstance(x, (list, dict)):
        return json.dumps(x, ensure_ascii=False)
    return x

def extract_item(raw: dict, split_hint: str | None = None) -> dict:
    """将原始样本统一映射为结构化记录"""
    split = normalize_split(raw.get("split", split_hint or "unknown"))
    rec = {
        "split": split,
        "id": raw.get("id"),
        "study_id": raw.get("study_id"),
        "subject_id": raw.get("subject_id"),
        "viewposition": raw.get("viewposition"),
        "indication_pure": raw.get("indication_pure"),
        "impression": raw.get("impression"),
        # report 优先用 report，否则为空；report_original 原样保留
        "report": raw.get("report"),
        "report_original": raw.get("report_original"),
        # 影像路径统一成 image_paths（原始可能是 image_path）
        "image_paths": raw.get("image_paths") or raw.get("image_path"),
        "seg_organs": raw.get("seg_organs"),
        "seg_mask_paths": raw.get("seg_mask_paths"),
    }
    # 生成空报告标记（report缺失时也记为空）
    rec["report_is_empty"] = is_empty_report(rec["report"])
    # 统计图片数量
    if isinstance(rec["image_paths"], list):
        rec["num_images"] = len(rec["image_paths"])
    else:
        rec["num_images"] = 0
    return rec

# === 读取与规范化 ===
with open(PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

records = []

if isinstance(data, dict) and any(k in data for k in ("train", "val", "test", "validation", "dev")):
    # 结构1：顶层 dict
    for split_key, items in data.items():
        if not isinstance(items, list):
            continue
        for it in items:
            if isinstance(it, dict):
                records.append(extract_item(it, split_hint=split_key))
elif isinstance(data, list):
    # 结构2：顶层 list
    for it in data:
        if isinstance(it, dict):
            records.append(extract_item(it))
else:
    raise ValueError(f"不支持的数据顶层类型：{type(data)}，期望 dict 或 list")

# === 构建 DataFrame ===
df = pd.DataFrame(records)

# 填充缺失列，保证导出列齐全且顺序一致
cols = [
    "split","id","study_id","subject_id",
    "viewposition","indication_pure","impression",
    "report","report_original","report_is_empty",
    "num_images","image_paths","seg_organs","seg_mask_paths",
]
for c in cols:
    if c not in df.columns:
        df[c] = None
df = df[cols]

# 将列表/字典列转为 JSON 字符串便于CSV保存
for c in ["image_paths","seg_organs","seg_mask_paths"]:
    df[c] = df[c].map(to_json_str)

# === 导出 CSV 与 JSONL ===
df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
with open(OUT_JSONL, "w", encoding="utf-8") as w:
    for rec in records:
        # 原样导出为逐行 JSON（列表/字典字段保持结构）
        w.write(json.dumps(rec, ensure_ascii=False) + "\n")

# === 打印一个简要汇总 ===
summary = df.groupby("split").agg(
    samples=("id","count"),
    empty_reports=("report_is_empty","sum"),
    images_total=("num_images","sum")
).reset_index()

print("=== 汇总 ===")
print(summary.to_string(index=False))
print(f"\n已导出: {OUT_CSV}  和  {OUT_JSONL}")
