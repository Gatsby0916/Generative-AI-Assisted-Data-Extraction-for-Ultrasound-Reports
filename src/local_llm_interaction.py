# src/local_llm_interaction.py
import os, sys, json, argparse, re
import requests
import config

def load_file(path):
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()

def sanitize_slug(model_id: str) -> str:
    # 与 main.py 的写法保持一致：把 / : 改成 _
    return model_id.replace('/', '_').replace(':', '_')

def robust_json_extract(text: str):
    """尽量把模型返回中的 JSON 提取出来（去掉markdown围栏等）。"""
    # 去掉 ```json ... ``` 围栏
    fenced = re.findall(r"```json\s*(\{[\s\S]*?\})\s*```", text, flags=re.IGNORECASE)
    if fenced:
        text = fenced[0]
    # 粗暴匹配第一个大括号到最后一个大括号
    start = text.find('{')
    end   = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        candidate = text[start:end+1]
        return json.loads(candidate)
    # 直接尝试整体解析
    return json.loads(text)

def build_prompt_from_template(json_template_path: str) -> str:
    # 仅使用模板（与项目内的通用提示一致地强调“只输出 JSON”）
    # 这里不依赖 FIELDS_METADATA，保证最小外部依赖
    tmpl_str = load_file(json_template_path)
    prompt = (
        "# Task\n"
        "Given a clinical ultrasound report (plain text), extract all fields and fill the JSON strictly.\n"
        "Output MUST be a single valid JSON object, no extra text.\n\n"
        "# JSON Template\n"
        "```json\n" + tmpl_str.strip() + "\n```\n\n"
        "# Rules\n"
        "- If a field is explicitly stated as present/yes/positive/etc., fill \"1\"; if absent/negative/not mentioned, fill \"0\".\n"
        "- For numeric fields (e.g., '(mm)', counts), fill the numeric value; if not found, fill \"0\".\n"
        "- For dates, use YYYY-MM-DD; if unknown, leave empty string.\n"
        "- For free-text comment fields: if no abnormality, fill \"0\" (do not copy normal phrases).\n"
        "- Use the exact keys and value formats from the template. Do not add extra keys.\n"
        "- Return ONLY the JSON object.\n"
    )
    return prompt

def main(dataset_name: str, report_id: str, model_id: str, base_url: str):
    # 1) 路径与目录
    ds = config.DATASET_CONFIGS[dataset_name]
    processed_dir = config.get_processed_data_dir(dataset_name)  # results/processed_text/sugo
    raw_json_dir  = config.get_extracted_json_raw_dir("local", sanitize_slug(model_id), dataset_name)
    os.makedirs(raw_json_dir, exist_ok=True)

    # 2) 读取输入文本与 JSON 模板
    txt_path = os.path.join(processed_dir, f"{report_id}.txt")
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"Processed text not found: {txt_path}")
    report_text = load_file(txt_path)

    json_template_path = ds["template_json"]
    if not os.path.exists(json_template_path):
        raise FileNotFoundError(f"JSON template not found: {json_template_path}")

    prompt = build_prompt_from_template(json_template_path)

    # 3) 组织 Chat Completions 请求
    url = base_url.rstrip('/') + "/chat/completions"
    headers = {"Content-Type": "application/json"}  # 本地服务通常不需要 API Key
    messages = [
        {"role": "system", "content": "You are a careful clinical information extractor."},
        {"role": "user",   "content": prompt},
        {"role": "user",   "content": "## Ultrasound Report (plain text)\n" + report_text}
    ]
    payload = {
        "model": model_id,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 3000
    }

    # 4) 调用本地服务
    r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=600)
    r.raise_for_status()
    data = r.json()
    content = data["choices"][0]["message"]["content"]

    # 5) 解析并保存为 {report_id}_extracted_data.json（供 data_validation.py 使用）
    try:
        obj = robust_json_extract(content)
    except Exception as e:
        raise RuntimeError(f"Failed to parse JSON from model output: {e}\nRaw content head:\n{content[:500]}")

    out_path = os.path.join(raw_json_dir, f"{report_id}_extracted_data.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=config.JSON_INDENT, ensure_ascii=config.ENSURE_ASCII)
    print(f"[OK] Saved extracted JSON -> {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run local LLM to extract fields for one report.")
    ap.add_argument("dataset_name", choices=list(config.DATASET_CONFIGS.keys()))
    ap.add_argument("report_id")
    ap.add_argument("model_id", help="Local model id or alias as seen by your OpenAI-compatible server.")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    args = ap.parse_args()
    main(args.dataset_name, args.report_id, args.model_id, args.base_url)
