import os
import csv
import re
import json
import random

from tqdm import tqdm
from time import sleep
from llm import gemini
from dotenv import load_dotenv
from pathlib import Path

if __name__ == "__main__":
    load_dotenv()  
    ### --- 1. Gemini API Connection
    GEMINI_API_KEY = os.getenv("gemini_api_key")
    chat_gm = gemini.Gemini(API_KEY=GEMINI_API_KEY)
    ### --- End of Step 1


    ### --- 2. Question Processing & Gemini Answering
    # with open('../output/Gemini/Final_Q714/20250417_Gemini20Flash.csv', 'w', newline='') as f4:
    #     writer = csv.writer(f4)
    #     writer.writerow(['Question_ID', 'Question', 'Reference_Ans','Answer'])

    ## 從 chatdoctor5k 資料集擷取問題與答案
    re1 = r'The extracted entities are (.*?)<END>'
    re2 = r"The extracted entity is (.*?)<END>"
    re3 = r"<CLS>(.*?)<SEP>"

    docs_dir = '../data/chatdoctor5k/document'

    docs = []
    for file in os.listdir(docs_dir):
        with open(os.path.join(docs_dir, file), 'r', encoding='utf-8') as f:
            doc = f.read()
            docs.append(doc)

    with open("../data/chatdoctor5k/NER_chatgpt.json", "r") as f:
        lines = f.readlines()
        RANDOM_SEED = 42
        random.seed(RANDOM_SEED)
        ## 總行數
        total_lines = len(lines)
        ## 使用隨機種子選取 714 個題號
        selected_indices = random.sample(range(total_lines), 718)
        exclude = {552, 2553, 4333, 5084}
        selected_indices = [q for q in selected_indices if q not in exclude]
        # selected_indices = selected_indices[225:]
        print(f"隨機選取的題號: {selected_indices}")

        for count, index in enumerate(tqdm(selected_indices, desc="Processing Questions", unit="question", dynamic_ncols=True)):
            line = lines[index]
            x = json.loads(line)
            input = x["qustion_output"]
            input = input.replace("\n","")
            input = input.replace("<OOS>","<EOS>")
            input = input.replace(":","") + "<END>"
            input_text = re.findall(re3,input)
            
            if input_text == []:
                continue
            print('Question:', index+1,'\n',input_text[0])

            output = x["answer_output"]
            output = output.replace("\n","")
            output = output.replace("<OOS>","<EOS>")
            output = output.replace(":","") + "<END>"
            output_text = re.findall(re3,output)
        
            output_all = chat_gm(input_text[0])
            
            ## Q3873 input_text, output_text 會無法擷取，單獨存取
            if index+1 == 3873:
                with open('../output/Gemini/Final_Q714/20250916_Gemini20Flash.csv', 'a+', newline='') as f6:
                    writer = csv.writer(f6)
                    writer.writerow([index+1, output_all])
                    f6.flush()
            else:
                with open('../output/Gemini/Final_Q714/20250916_Gemini20Flash.csv', 'a+', newline='') as f6:
                    writer = csv.writer(f6)
                    writer.writerow([index+1, input_text[0], output_text[0], output_all])
                    f6.flush()
    ### --- End of Step 2
            