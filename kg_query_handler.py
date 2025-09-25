# kg_query_handler.py
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

# 呼叫自己的檔案 (假設這些檔案都在同一個路徑下)
import BuildDatabase
from llm import gemini
from mindmap import Preprocessing
from communitysearch import FindKG, GreedyDist, KGtoPath_BFS, PromptGenerate
from other import KG_vision_pyvis

# 全域載入環境變數
load_dotenv()

# 將翻譯函式獨立出來，因為 app.py 也需要使用
def translate_text(text_to_translate, target_language, llm_client):
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

class KGQueryHandler:
    def __init__(self):
        """
        初始化所有必要的資源，例如資料庫連線、模型和資料檔案。
        這個方法只會在伺服器啟動時執行一次。
        """
        # --- [Step 1]. Neo4j Connection
        uri = os.getenv("neo4j_uri")
        username = os.getenv("neo4j_username")
        password = os.getenv("neo4j_password")
        self.driver = GraphDatabase.driver(codecs.decode(uri, 'unicode_escape'), auth=(username, password))
        
        # --- [Step 2]. Gemini API Connection
        GEMINI_API_KEY = os.getenv("gemini_api_key_upgrade")
        self.chat_gm = gemini.Gemini(API_KEY=GEMINI_API_KEY)
        
        # --- [Step 3]. Load Data Files
        print("Loading data files (embeddings)...")
        with open('./data/chatdoctor5k/entity_embeddings.pkl', 'rb') as f1:
            self.entity_embeddings = pickle.load(f1)
        with open('./data/chatdoctor5k/keyword_embeddings_new.pkl', 'rb') as f2:
            self.keyword_embeddings = pickle.load(f2)
        self.entity_embeddings_df = pd.DataFrame(self.entity_embeddings["embeddings"])
        print("Data files loaded.")
        
    def process_query(self, english_question):
        """
        處理單一的英文問題，執行知識圖譜檢索和答案生成。
        """
        # --- 實體提取 ---
        print("  > Extracting entities...")
        re1_for_extraction = r'<CLS>.*<SEP>The extracted entities are (.*?)<EOS>'
        extracted_entities_raw = Preprocessing.prompt_extract_keyword(
            english_question, 
            self.chat_gm, 
            re1_for_extraction
        )
        print(f"  > Raw extracted entities: {extracted_entities_raw}")

        if not extracted_entities_raw or not extracted_entities_raw[0]:
            print("  <Warning> Failed to dynamically extract entities.")
            return "[Error: 無法從您的問題中提取關鍵實體]"

        question_kg = [entity.strip() for entity in extracted_entities_raw[0].split(',')]
        print(f'  > Dynamically Extracted question_kg: {question_kg}')
        
        # --- 實體連結 (Entity Linking) ---
        print("  > Linking entities to KG...")
        match_kg = []
        for kg_entity in question_kg:
            try:
                keyword_index = self.keyword_embeddings["keywords"].index(kg_entity)
            except ValueError:
                print(f"  <Warning> Entity '{kg_entity}' not in keyword embeddings. Skipping.")
                continue
            
            kg_entity_emb = np.array(self.keyword_embeddings["embeddings"][keyword_index])
            cos_similarities = Preprocessing.cosine_similarity_manual(self.entity_embeddings_df, kg_entity_emb)[0]
            
            max_index = cos_similarities.argmax()
            match_kg_i = self.entity_embeddings["entities"][max_index]
            
            # 避免重複加入相同的實體
            while match_kg_i.replace(" ","_") in match_kg:
                cos_similarities[max_index] = 0
                max_index = cos_similarities.argmax()
                match_kg_i = self.entity_embeddings["entities"][max_index]
            
            match_kg.append(match_kg_i.replace(" ","_"))

        if not match_kg:
            print("<Warning> No matching entities found in KG.")
            return "[Error: 抱歉，知識庫中找不到與您問題相關的實體]"
        
        print(f'  > Linked KG Entities: {match_kg}')
        
        # --- 子圖搜尋 ---
        print("  > Finding subgraph...")
        start_subgraph = perf_counter()
        graph_dict = FindKG.find_whole_KG(self.driver)
        condition_constraint = {'distance': 5, 'size': 200}
        _, result_subgraph = GreedyDist.greedy_dist(graph_dict, match_kg, condition_constraint)
        print(f"  > Subgraph found in {perf_counter() - start_subgraph:.3f}s")
        
        # --- 路徑搜尋 ---
        print("  > Finding paths...")
        start_paths = perf_counter()
        all_paths = FindKG.subgraph_path_finding(result_subgraph, match_kg)
        top_n = 10
        path_list, flag = KGtoPath_BFS.paths_in_neo4j_optimized_bfs_full(all_paths, top_n, self.driver)
        (path_join, _, _) = FindKG.combine_lists(community_search_paths=path_list, pagerank_values=None, top_n=top_n, flag=flag)
        print(f"  > Paths found in {perf_counter() - start_paths:.3f}s")
        
        # --- 提示生成 & 最終問答 ---
        print("  > Generating final answer with LLM...")
        prompt = PromptGenerate.GeneratePathPrompt(path_join, self.chat_gm)
        
        final_english_answer = ""
        retries = 2
        for i in range(retries):
            output_all = PromptGenerate.final_answer(english_question, prompt, self.chat_gm)
            output1 = PromptGenerate.extract_final_answer(output_all)
            if len(output1) > 0:
                final_english_answer = output1[0]
                break
            print(f"  <Warning> Failed to generate answer on attempt {i+1}.")
            if i == retries - 1:
                final_english_answer = "[Error: LLM failed to generate a valid answer]"
                
        print(f"  > Generated English Answer: {final_english_answer}")

        if "[Error:" in final_english_answer:
            return "抱歉，AI 在生成最終答案時遇到問題，請嘗試用不同的方式提問。"

        # --- 將最終答案翻譯回中文 ---
        print("  > Translating final answer to Chinese...")
        final_chinese_answer = translate_text(final_english_answer, "繁體中文", self.chat_gm)
        
        return final_chinese_answer

    def __del__(self):
        """
        在物件被銷毀時（例如伺服器關閉時）關閉資料庫連線。
        """
        if self.driver:
            self.driver.close()
            print("Neo4j connection closed.")