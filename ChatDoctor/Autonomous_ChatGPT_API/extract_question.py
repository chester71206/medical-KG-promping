import json
import random
from tqdm import tqdm
import re
import time

from chat_gemini import csv_prompter


re1 = r"<CLS>(.*?)<SEP>"
csv_name = "disease_database_mini.csv"

json_data = []
SAVE_INTERVAL = 5
with open("/Users/yuu/GitHub/ChatDoctor/Autonomous_ChatGPT_API/NER_chatgpt.json", "r") as f:
    lines = f.readlines()
    ### 設置隨機種子
    RANDOM_SEED = 42
    random.seed(RANDOM_SEED)
    # 總行數
    total_lines = len(lines)
    # 使用隨機種子選取 100 個題號
    selected_indices = random.sample(range(total_lines), 718)
    exclude = {552, 2553, 4333, 5084}  # 這裡是題號（從 1 開始）
    selected_indices = [q for q in selected_indices if q not in exclude]
    selected_indices = selected_indices[163:400]
    print(f"隨機選取的題號: {selected_indices}")

    for count, index in enumerate(tqdm(selected_indices,desc="Processing Questions",unit="question",dynamic_ncols=True)):
        try: 
            ### Step1 --- Question output
            line = lines[index]
            x = json.loads(line)
            input = x["qustion_output"]
            input = input.replace("\n", "")
            input = input.replace("<OOS>", "<EOS>")
            input = input.replace(":", "") + "<END>"
            input_text = re.findall(re1, input)

            if input_text == []:
                continue

            print("Question:", index + 1, "\n", input_text[0])
            ### ---


            ### Step2 --- Answer output
            FinalAnswer = csv_prompter(input_text[0], csv_name)
            ### ---

    
            ### --- save for json
            json_data.append({
                "Q_ID": index + 1,
                "qustion_output": x["qustion_output"],
                "anwser_output": FinalAnswer
            })

            # 每 SAVE_INTERVAL 筆就存檔一次
            with open("/Users/yuu/GitHub/ChatDoctor/Autonomous_ChatGPT_API/NER_Gemini20_2.json","w",encoding="utf-8") as f2:
                json.dump(json_data, f2, ensure_ascii=False, indent=4)
        
        except Exception as e:
            print(f"Error processing question {index + 1}: {e}")
            continue

        time.sleep(5)

    with open("/Users/yuu/GitHub/ChatDoctor/Autonomous_ChatGPT_API/NER_Gemini20_2.json","w",encoding="utf-8") as f3:
        json.dump(json_data, f3, ensure_ascii=False, indent=4)
