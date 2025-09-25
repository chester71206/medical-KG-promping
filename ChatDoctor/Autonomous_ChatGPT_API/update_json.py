import json

# 從外部 JSON 檔案讀入 list of dicts
with open("NER_Gemini20.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 輸出成每行一筆的 JSONL 格式
with open("NER_Gemini20_formatted.jsonl", "w", encoding="utf-8") as f_out:
    for item in data:
        output = {
            "Q_ID": item["Q_ID"],
            "question_output": item["qustion_output"],
            "answer_output": item["anwser_output"]
        }
        json_line = json.dumps(output, ensure_ascii=False)
        f_out.write(json_line + "\n")
