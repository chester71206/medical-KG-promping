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

# vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
### <<< 修改點 1: 定義你要運行的特定題號 >>> ###
# 使用集合 (set) 查詢效率最高
TARGET_Q_IDS = {459} 

# 註解掉不再需要的變數
# START_FROM_QUESTION=8
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


# vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
# <<< 修改點: 更新翻譯函式以適應你的 gemini.py (這部分維持不變) >>>
def translate_text(text_to_translate, target_language, llm_client):
    """
    使用指定的 LLM 客戶端將文字翻譯成目標語言。
    """
    if not text_to_translate or not text_to_translate.strip():
        return ""
        
    prompt = f"Please translate the following text into {target_language}. Provide only the translated text without any other explanations or context.\n\nText to translate: \"{text_to_translate}\""
    
    try:
        # 根據你的 gemini.py，它是一個 LangChain LLM 的子類，應該直接呼叫實例
        response_json_str = llm_client(prompt)
        # 因為你的 gemini.py 使用 json.dumps() 處理輸出，所以我們要用 json.loads() 還原
        translated_text = json.loads(response_json_str)
        return translated_text.strip()
    except Exception as e:
        print(f"An error occurred during translation: {e}")
        return f"[Translation Error: {text_to_translate}]"
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


if __name__ == "__main__":

    uri = os.getenv("neo4j_uri")
    username = os.getenv("neo4j_username")
    password = os.getenv("neo4j_password")
    print(codecs.decode(uri, 'unicode_escape'))
    driver = GraphDatabase.driver(codecs.decode(uri, 'unicode_escape'), auth=(username, password))
    ### --- End of [Step 1]
    
    ### --- [Step 2]. Gemini API Connection
    GEMINI_API_KEY = os.getenv("gemini_api_key_upgrade")
    chat_gm = gemini.Gemini(API_KEY=GEMINI_API_KEY)
    ### --- End of [Step 2]

    # 定義輸出的 CSV 檔案路徑
    output_csv_path = f'./output/DenselyConnectedSubgraphRetrieval/Q714_Gemini1.5/GreedyDist200/{date.today()}_BFS_Chinese_Specific_Questions.csv'

    # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    ### <<< 修改點 2: 簡化檔案寫入邏輯 >>> ###
    # 檢查輸出檔案是否已存在。如果不存在，就建立新檔案並寫入標頭。
    # 這種方法比依賴 START_FROM_QUESTION 更穩健。
    if not os.path.exists(output_csv_path):
        print(f"Output file not found. Creating a new file: {output_csv_path}")
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as f4:
            writer = csv.writer(f4)
            writer.writerow(['Q_ID', 'Chinese_Question', 'Chinese_Answer'])
    else:
        print(f"Output file found. Appending results to: {output_csv_path}")
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    with open('./data/chatdoctor5k/entity_embeddings.pkl','rb') as f1:
        entity_embeddings = pickle.load(f1)
        
    with open('./data/chatdoctor5k/keyword_embeddings_new.pkl','rb') as f2:
        keyword_embeddings = pickle.load(f2)
    
    ### --- [Step 3]. Extract Question Entities
    re1 = r'The extracted entities are (.*?)<END>'
    re2 = r"The extracted entity is (.*?)<END>"
    re3 = r"<CLS>(.*?)<SEP>"

    with open("./data/chatdoctor5k/NER_Gemini20_formatted.jsonl", "r") as f:
        lines = f.readlines()

    # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    ### <<< 修改點 3: 修改主迴圈以篩選指定題號 >>> ###
    # 1. 迴圈遍歷整個 `lines` 列表，而不是 `lines[START_FROM_QUESTION:]`
    # 2. 在迴圈內部，檢查 Q_ID 是否在我們的目標集合 `TARGET_Q_IDS` 中
    for line in tqdm(lines, desc="Scanning for Target Q&A", unit="question", dynamic_ncols=True):
        x = json.loads(line)
        
        # 如果當前問題的 Q_ID 不在我們的目標清單中，就用 continue 跳過
        if x['Q_ID'] not in TARGET_Q_IDS:
            continue
            
        # --- 如果 Q_ID 匹配成功，則執行以下所有處理 ---
        tqdm.write("\n" + "="*50)
        tqdm.write(f"Processing Matched Question | Q_ID: {x['Q_ID']}")
        
        original_input_with_entities = x["question_output"].replace("\n","").replace("<OOS>","<EOS>").replace(":","") + "<END>"
        input_text = re.findall(re3, original_input_with_entities)
        if not input_text: continue
        original_english_question = input_text[0]
        tqdm.write(f'Original English Question:\n {original_english_question}')

        # --- 翻譯流程 ---
        #chinese_question = translate_text(original_english_question, "繁體中文", chat_gm)
        chinese_question ="""醫生，我也不知道該怎麼說，就覺得整個人不太對勁，已經好幾個月了。

一開始只是覺得很累，不是那種沒睡飽的累，是一種...像是電池被拔掉的感覺，全身沒力，連腦袋都轉不動，重重的、霧霧的。本來以為是工作太操勞，但我休假睡了一整天，醒來還是覺得骨頭裡都是疲倦。

然後...身體就開始輪流抗議。今天可能是手腕悶悶的痛，明天換成膝蓋，後天又沒事了。但它又不是真的受傷那種尖銳的痛，比較像...關節在鬧脾氣，酸酸脹脹的，讓你沒辦法忽視它。有時候早上起來，手指會覺得僵硬，要動一動才會好。

最奇怪的是我的皮膚。有時候曬到太陽，也不是去海邊那種大曬，就只是上下班走在路上，晚上就覺得臉上熱熱的，有點發燙，照鏡子會看到兩頰有點像過敏的紅疹，不痛不癢，但過幾天又自己消了。

我家人都說我是不是壓力太大，想太多了。我自己也懷疑是不是憂鬱症，但我明明就不是心情不好，我是『身體』不舒服。腦袋常常一片空白，話到嘴邊就忘了要說什麼。真的快被搞瘋了，感覺身體好像不是我自己的，每天都有新的『驚喜』。"""
        tqdm.write(f'Translated Chinese Question:\n {chinese_question}')
        tqdm.write(f'Translated Chinese Question:\n {chinese_question}')
        retranslated_english_question = translate_text(chinese_question, "English", chat_gm)
        tqdm.write(f'Re-translated English Question:\n {retranslated_english_question}')

        # --- 實體提取與後續流程 (這部分不變) ---
        tqdm.write("Dynamically extracting entities from the re-translated question...")
        re1_for_extraction = r'<CLS>.*<SEP>The extracted entities are (.*?)<EOS>'
        extracted_entities_raw = Preprocessing.prompt_extract_keyword(
            retranslated_english_question, 
            chat_gm, 
            re1_for_extraction
        )
        tqdm.write(f"extracted_entities_raw: {extracted_entities_raw}")

        if not extracted_entities_raw or not extracted_entities_raw[0]:
            tqdm.write(f"<Warning> Failed to dynamically extract entities for question: {retranslated_english_question}")
            with open(output_csv_path, 'a+', newline='', encoding='utf-8') as f4:
                writer = csv.writer(f4)
                writer.writerow([x["Q_ID"], chinese_question, "[Error: Dynamic entity extraction failed]"])
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
                writer.writerow([x["Q_ID"], chinese_question, "[Error: No matching entities found in KG]"])
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
            output_all = PromptGenerate.final_answer(retranslated_english_question, prompt, chat_gm)
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

        final_chinese_answer = translate_text(final_english_answer, "繁體中文", chat_gm)
        tqdm.write(f"Final Chinese Answer: {final_chinese_answer}")
        
        ### --- [Step 8]. Save Results --- ###
        with open(output_csv_path, 'a+', newline='', encoding='utf-8') as f4:
            writer = csv.writer(f4)
            csv_safe_answer = final_chinese_answer.replace('\n', '\r\n')
            writer.writerow([x["Q_ID"], chinese_question, csv_safe_answer])
            f4.flush()
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    driver.close()
    print("\nProcessing complete. Results saved to:", output_csv_path)