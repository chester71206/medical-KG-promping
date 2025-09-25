# import json
# import os
# import csv
# import re
# from llm import gemini
# from tqdm import tqdm
# from datetime import date
# import time


# if __name__ == "__main__":

#     ### --- 1. Gemini API Connection
#     GEMINI_API_KEY = os.getenv("gemini_api_key_upgrade")
#     chat_gm = gemini.Gemini(API_KEY=GEMINI_API_KEY)
#     ### --- End of step 1


#     ### --- 2. Question Processing & Gemini Answering
#     with open(f'./output/Gemini/Final_Q714_Gemma3-1B/{date.today()}.csv', 'w', newline='') as f4: ###
#         writer = csv.writer(f4)
#         writer.writerow(['Q_ID', 'Question', 'Reference Answer', 'Answer'])

#     re3 = r"<CLS>(.*?)<SEP>"

#     with open("./data/chatdoctor5k/NER_Gemini20_formatted.jsonl", "r") as f:
#         lines = f.readlines()

#     round_count = 0
#     for line in tqdm(lines[round_count:], desc="Processing Q&A", unit="question", dynamic_ncols=True):
#         x = json.loads(line)
#         print("Q_ID from data:", x["Q_ID"])

#         input = x["question_output"]
#         input = input.replace("\n","")
#         input = input.replace("<OOS>","<EOS>")
#         input = input.replace(":","") + "<END>"
#         input_text = re.findall(re3,input)
#         question = input_text[0]
#         print('Question:', x["Q_ID"],'\n',question)

#         answer = x["answer_output"]
            
#         prompt = (
#             f"Patient input: {question}\n\n"
#             "What disease does the patient have? What tests should patient take to confirm the diagnosis? "
#             "What recommened medications can cure the disease? Think step by step.\n\n\n"
#             "Answer in the following format with three clearly separated sections:\n"
#             "1. **Most Likely Disease:**\n[State only one most likely disease based on the symptoms. Briefly explain why.]\n\n"
#             "2. **Recommended Medication(s):**\n[List multiple possible medications or treatment options for this disease, if applicable.]\n\n"
#             "3. **Suggested Medical Test(s):**\n[List multiple relevant medical tests that can help confirm or rule out the diagnosis.]\n\n"
#         )
        
#         output_all = chat_gm(prompt)
        
#         with open(f'./output/Gemini/Final_Q714_Gemma3-1B/{date.today()}.csv', 'a+', newline='') as f6:
#             writer = csv.writer(f6)
#             writer.writerow([x["Q_ID"], question, answer, output_all])
#             f6.flush()
#     ### --- End of step 2



import json
import os
import csv
import re
from llm import gemini  # 假設你的 gemini 模組路徑正確
from tqdm import tqdm
from datetime import date
import time


if __name__ == "__main__":

    ### --- 1. Gemini API Connection
    # 請確保你已經設定了環境變數 'gemini_api_key_upgrade'
    GEMINI_API_KEY = os.getenv("gemini_api_key_upgrade")
    if not GEMINI_API_KEY:
        raise ValueError("Gemini API key not found. Please set the 'gemini_api_key_upgrade' environment variable.")
    chat_gm = gemini.Gemini(API_KEY=GEMINI_API_KEY)
    ### --- End of step 1


    ### --- 2. Question Processing & Gemini Answering

    # --- START: 修改部分 ---
    # 1. 定義輸出的資料夾路徑和檔案路徑
    output_dir = './output/Gemini/Final_Q714_Gemma3-1B'
    output_filepath = os.path.join(output_dir, f'{date.today()}.csv')

    # 2. 使用 os.makedirs() 自動建立不存在的資料夾
    #    exist_ok=True 參數可以避免在資料夾已存在時發生錯誤
    os.makedirs(output_dir, exist_ok=True)
    
    # 3. 使用定義好的檔案路徑來建立並寫入標頭 (header)
    with open(output_filepath, 'w', newline='', encoding='utf-8') as f_init:
        writer = csv.writer(f_init)
        writer.writerow(['Q_ID', 'Question', 'Reference Answer', 'Answer'])
    # --- END: 修改部分 ---

    re3 = r"<CLS>(.*?)<SEP>"

    try:
        with open("./data/chatdoctor5k/NER_Gemini20_formatted.jsonl", "r", encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print("Error: The data file './data/chatdoctor5k/NER_Gemini20_formatted.jsonl' was not found.")
        exit() # 如果找不到資料來源檔，就直接結束程式

    round_count = 0
    for line in tqdm(lines[round_count:], desc="Processing Q&A", unit="question", dynamic_ncols=True):
        x = json.loads(line)
        print("\nQ_ID from data:", x["Q_ID"])

        input_str = x["question_output"]
        input_str = input_str.replace("\n", "")
        input_str = input_str.replace("<OOS>", "<EOS>")
        input_str = input_str.replace(":", "") + "<END>"
        
        # 使用 re.search 來確保找到匹配項
        match = re.search(re3, input_str)
        if not match:
            print(f"Warning: Could not find question pattern in Q_ID {x['Q_ID']}. Skipping.")
            continue # 如果找不到問題，就跳過這一輪

        question = match.group(1)
        print('Question:', x["Q_ID"], '\n', question)

        answer = x["answer_output"]
            
        prompt = (
            f"Patient input: {question}\n\n"
            "What disease does the patient have? What tests should patient take to confirm the diagnosis? "
            "What recommened medications can cure the disease? Think step by step.\n\n\n"
            "Answer in the following format with three clearly separated sections:\n"
            "1. **Most Likely Disease:**\n[State only one most likely disease based on the symptoms. Briefly explain why.]\n\n"
            "2. **Recommended Medication(s):**\n[List multiple possible medications or treatment options for this disease, if applicable.]\n\n"
            "3. **Suggested Medical Test(s):**\n[List multiple relevant medical tests that can help confirm or rule out the diagnosis.]\n\n"
        )
        
        try:
            output_all = chat_gm(prompt)
            
            # --- START: 修改部分 ---
            # 使用 'a+' (附加) 模式將結果寫入同一個檔案
            with open(output_filepath, 'a+', newline='', encoding='utf-8') as f_append:
                writer = csv.writer(f_append)
                writer.writerow([x["Q_ID"], question, answer, output_all])
                f_append.flush()
            # --- END: 修改部分 ---
        
        except Exception as e:
            print(f"An error occurred while calling the Gemini API for Q_ID {x['Q_ID']}: {e}")
            # 你可以在這裡決定是否要重試或記錄錯誤後繼續
            time.sleep(1) # 發生錯誤時稍作等待

    ### --- End of step 2

print("\nProcessing complete.")
print(f"Output saved to: {output_filepath}")