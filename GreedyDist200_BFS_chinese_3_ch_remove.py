import numpy as np
import re
import codecs
import pandas as pd
import pickle
import json
import os
import csv
import time
import random
from neo4j import GraphDatabase, basic_auth
from tqdm import tqdm
from time import perf_counter
from datetime import date
from dotenv import load_dotenv
load_dotenv()
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", message="When cdn_resources is 'local'")

## 呼叫自己的檔案
import BuildDatabase
from llm import openrouter, gemini
from mindmap import Preprocessing
from communitysearch import FindKG, GreedyDist, KGtoPath_PR, KGtoPath_BFS, PromptGenerate
from other import KG_vision_pyvis

# 指定你要運行的特定題號
TARGET_Q_IDS = {459, 3783, 995} 

# 翻譯函式 (維持不變，因為我們最後還是需要將答案翻譯成中文)
def translate_text(text_to_translate, target_language, llm_client):
    """
    使用指定的 LLM 客戶端將文字翻譯成目標語言。
    """
    if not text_to_translate or not text_to_translate.strip():
        return ""
        
    prompt = f"Please translate the following text into {target_language}. Provide only the translated text without any other explanations or context.\n\nText to translate: \"{text_to_translate}\""
    
    try:
        response_json_str = llm_client(prompt)
        translated_text = json.loads(response_json_str)
        return translated_text.strip()
    except Exception as e:
        print(f"An error occurred during translation: {e}")
        return f"[Translation Error: {text_to_translate}]"


if __name__ == "__main__":

    uri = os.getenv("neo4j_uri")
    username = os.getenv("neo4j_username")
    password = os.getenv("neo4j_password")
    print(codecs.decode(uri, 'unicode_escape'))
    driver = GraphDatabase.driver(codecs.decode(uri, 'unicode_escape'), auth=(username, password))
    
    GEMINI_API_KEY = os.getenv("gemini_api_key_upgrade")
    chat_gm = gemini.Gemini(API_KEY=GEMINI_API_KEY)

    # 定義輸出的 CSV 檔案路徑，檔名可以調整以反映這次的修改
    output_csv_path = f'./output/DenselyConnectedSubgraphRetrieval/Q714_Gemini1.5/GreedyDist200/{date.today()}_BFS_Chinese_Specific_Questions_OriginalText.csv'

    # 檢查輸出檔案是否已存在。如果不存在，就建立新檔案並寫入標頭。
    if not os.path.exists(output_csv_path):
        print(f"Output file not found. Creating a new file: {output_csv_path}")
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as f4:
            writer = csv.writer(f4)
            writer.writerow(['Q_ID', 'Chinese_Question', 'Chinese_Answer'])
    else:
        print(f"Output file found. Appending results to: {output_csv_path}")

    with open('./data/chatdoctor5k/entity_embeddings.pkl','rb') as f1:
        entity_embeddings = pickle.load(f1)
        
    with open('./data/chatdoctor5k/keyword_embeddings_new.pkl','rb') as f2:
        keyword_embeddings = pickle.load(f2)
    
    re3 = r"<CLS>(.*?)<SEP>"

    with open("./data/chatdoctor5k/NER_Gemini20_formatted.jsonl", "r") as f:
        lines = f.readlines()

    for line in tqdm(lines, desc="Scanning for Target Q&A", unit="question", dynamic_ncols=True):
        x = json.loads(line)
        
        # 如果當前問題的 Q_ID 不在我們的目標清單中，就跳過
        if x['Q_ID'] not in TARGET_Q_IDS:
            continue
            
        tqdm.write("\n" + "="*50)
        tqdm.write(f"Processing Matched Question | Q_ID: {x['Q_ID']}")
        
        original_input_with_entities = x["question_output"].replace("\n","").replace("<OOS>","<EOS>").replace(":","") + "<END>"
        input_text = re.findall(re3, original_input_with_entities)
        if not input_text: continue
        original_english_question = input_text[0]
        tqdm.write(f'Using Original English Question for processing:\n {original_english_question}')

        # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        ### <<< 修改點: 移除中英來回翻譯，只保留一次翻譯供CSV寫入 >>> ###
        # 1. 移除了 retranslated_english_question 的生成
        # 2. 生成一個中文問題版本，僅用於最後寫入 CSV 檔案
        chinese_question_for_csv = translate_text(original_english_question, "繁體中文", chat_gm)
        tqdm.write(f'Translated Chinese Question (for CSV output only):\n {chinese_question_for_csv}')

        # --- 實體提取與後續流程 ---
        # 3. 直接使用 original_english_question 進行實體提取
        tqdm.write("Dynamically extracting entities from the ORIGINAL English question...")
        re1_for_extraction = r'<CLS>.*<SEP>The extracted entities are (.*?)<EOS>'
        extracted_entities_raw = Preprocessing.prompt_extract_keyword(
            original_english_question, # <-- 使用原始英文問題
            chat_gm, 
            re1_for_extraction
        )
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        tqdm.write(f"extracted_entities_raw: {extracted_entities_raw}")

        if not extracted_entities_raw or not extracted_entities_raw[0]:
            tqdm.write(f"<Warning> Failed to dynamically extract entities for question: {original_english_question}")
            with open(output_csv_path, 'a+', newline='', encoding='utf-8') as f4:
                writer = csv.writer(f4)
                # 使用 chinese_question_for_csv 寫入
                writer.writerow([x["Q_ID"], chinese_question_for_csv, "[Error: Dynamic entity extraction failed]"])
                f4.flush()
            continue

        question_kg = [entity.strip() for entity in extracted_entities_raw[0].split(',')]
        tqdm.write(f'Dynamically Extracted question_kg:\n {question_kg}')

        match_kg = []
        entity_embeddings_emb = pd.DataFrame(entity_embeddings["embeddings"])
        for kg_entity in question_kg:
            try:
                keyword_index = keyword_embeddings["keywords"].index(kg_entity)
            except ValueError:
                tqdm.write(f"<Warning> Entity '{kg_entity}' not in keyword embeddings. Skipping.")
                continue
            kg_entity_emb = np.array(keyword_embeddings["embeddings"][keyword_index])
            cos_similarities = Preprocessing.cosine_similarity_manual(entity_embeddings_emb, kg_entity_emb)[0]
            max_index = cos_similarities.argmax()
            match_kg_i = entity_embeddings["entities"][max_index]
            while match_kg_i.replace(" ","_") in match_kg:
                cos_similarities[max_index] = 0
                max_index = cos_similarities.argmax()
                match_kg_i = entity_embeddings["entities"][max_index]
            match_kg.append(match_kg_i.replace(" ","_"))

        if not match_kg:
            tqdm.write("<Warning> No matching entities found in KG. Cannot proceed.")
            with open(output_csv_path, 'a+', newline='', encoding='utf-8') as f4:
                writer = csv.writer(f4)
                # 使用 chinese_question_for_csv 寫入
                writer.writerow([x["Q_ID"], chinese_question_for_csv, "[Error: No matching entities found in KG]"])
                f4.flush()
            continue
        
        tqdm.write(f'Question Entities:\n {match_kg}')
        
        start = perf_counter()
        graph_dict = FindKG.find_whole_KG(driver)
        condition_constraint = {'distance':5, 'size':200}
        distance, result_subgraph = GreedyDist.greedy_dist(graph_dict, match_kg, condition_constraint)
        step4_start = perf_counter()
        tqdm.write(f"Find Subgraph: { step4_start - start:.3f} 秒")

        all_paths = FindKG.subgraph_path_finding(result_subgraph, match_kg)
        top_n=10
        path_list, flag = KGtoPath_BFS.paths_in_neo4j_optimized_bfs_full(all_paths, top_n, driver)
        (path_join, path_join_list, path_nodes_count) = FindKG.combine_lists(community_search_paths=path_list, pagerank_values=None, top_n=top_n, flag=flag)
        step5_start = perf_counter()
        tqdm.write(f"Find Paths: {step5_start - step4_start:.3f} 秒")

        prompt = PromptGenerate.GeneratePathPrompt(path_join, chat_gm)
        step6_start = perf_counter()
        tqdm.write(f"Generate Paths in NL: {step6_start - step5_start:.3f} 秒")
        
        times=1
        final_english_answer = ""
        while True:
            # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
            ### <<< 修改點: 直接使用原始英文問題生成答案 >>> ###
            output_all = PromptGenerate.final_answer(original_english_question, prompt, chat_gm)
            # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            output1 = PromptGenerate.extract_final_answer(output_all)
            if len(output1) > 0:
                final_english_answer = output1[0]
                break
            elif times == 2:
                final_english_answer = "[Error: Failed to generate answer after 2 retries]"
                break
            else:
                times+=1
        tqdm.write(f"Final English Answer: {final_english_answer}")

        # 保留這一步，將最終的英文答案翻譯成中文
        final_chinese_answer = translate_text(final_english_answer, "繁體中文", chat_gm)
        tqdm.write(f"Final Chinese Answer: {final_chinese_answer}")
        
        ### --- [Step 8]. Save Results --- ###
        with open(output_csv_path, 'a+', newline='', encoding='utf-8') as f4:
            writer = csv.writer(f4)
            csv_safe_answer = final_chinese_answer.replace('\n', '\r\n')
            # 使用 chinese_question_for_csv 變數寫入
            writer.writerow([x["Q_ID"], chinese_question_for_csv, csv_safe_answer])
            f4.flush()

    driver.close()
    print("\nProcessing complete. Results saved to:", output_csv_path)