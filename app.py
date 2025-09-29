# app.py

# --- 基礎與標準函式庫 ---
import os
import re
import json
import time
import random
import codecs
import pickle
import warnings
from datetime import date
from dotenv import load_dotenv

# --- Flask 相關 ---
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS

# --- 數據處理與圖譜 ---
import numpy as np
import pandas as pd
import networkx as nx
from neo4j import GraphDatabase

# --- 自訂模組 ---
from llm import gemini
from mindmap import Preprocessing
from communitysearch import FindKG, GreedyDist, KGtoPath_BFS, PromptGenerate
from other.path_visualizer import draw_paths_graph

# --- 環境與設定 ---
warnings.filterwarnings("ignore")
load_dotenv()

# --- Flask App 初始化 ---
# 將 'frontend' 設為靜態資料夾，方便服務 index.html 等檔案
app = Flask(__name__, static_folder='frontend')
CORS(app)

class MedicalQAService:
    """
    封裝所有醫療問答核心邏輯的服務類別
    """
    def __init__(self):
        """初始化資料庫連線、LLM客戶端並載入必要的資料"""
        try:
            # 初始化Neo4j連接
            self.uri = os.getenv("neo4j_uri")
            self.username = os.getenv("neo4j_username")
            self.password = os.getenv("neo4j_password")
            self.driver = GraphDatabase.driver(
                codecs.decode(self.uri, 'unicode_escape'),
                auth=(self.username, self.password)
            )
            
            # 初始化Gemini API
            self.GEMINI_API_KEY = os.getenv("gemini_api_key_upgrade")
            self.chat_gm = gemini.Gemini(API_KEY=self.GEMINI_API_KEY)
            
            # 載入嵌入向量
            self.load_embeddings()
            print("✅ MedicalQAService initialized successfully.")

        except Exception as e:
            print(f"❌ Critical error during MedicalQAService initialization: {e}")
            # 在這種嚴重錯誤下，後續操作可能會失敗
            self.driver = None
            self.chat_gm = None

    def load_embeddings(self):
        """載入實體和關鍵詞嵌入向量"""
        try:
            with open('./data/chatdoctor5k/entity_embeddings.pkl','rb') as f1:
                self.entity_embeddings = pickle.load(f1)
            with open('./data/chatdoctor5k/keyword_embeddings_new.pkl','rb') as f2:
                self.keyword_embeddings = pickle.load(f2)
            print("✅ Embeddings loaded successfully")
        except FileNotFoundError as e:
            print(f"❌ Error loading embeddings: File not found - {e}. Please check the data path.")
            self.entity_embeddings = None
            self.keyword_embeddings = None
        except Exception as e:
            print(f"❌ Error loading embeddings: {e}")

    def detect_language(self, text):
        """檢測輸入文字的語言"""
        if not text or not text.strip(): return "unknown"
        if re.search(r'[\u4e00-\u9fff]', text): return "chinese"
        if re.search(r'[a-zA-Z]', text): return "english"
        return "unknown"

    def translate_text(self, text_to_translate, target_language):
        """使用Gemini翻譯文字"""
        if not text_to_translate or not text_to_translate.strip(): return ""
        prompt = f"Please translate the following text into {target_language}. Provide only the translated text without any other explanations or context.\n\nText to translate: \"{text_to_translate}\""
        try:
            response = self.chat_gm(prompt)
            if isinstance(response, str):
                try: return json.loads(response).strip()
                except json.JSONDecodeError: return response.strip()
            return str(response).strip()
        except Exception as e:
            print(f"❌ Translation error: {e}")
            return f"[Translation Error]"

    def extract_entities(self, question):
        """從問題中提取醫療實體"""
        try:
            re_extraction = r'<CLS>.*<SEP>The extracted entities are (.*?)<EOS>'
            raw_entities = Preprocessing.prompt_extract_keyword(question, self.chat_gm, re_extraction)
            if not raw_entities or not raw_entities[0]: return []
            return [e.strip() for e in raw_entities[0].split(',')]
        except Exception as e:
            print(f"❌ Entity extraction error: {e}")
            return []

    def match_entities_to_kg(self, entities):
        """將問題實體匹配到知識圖譜實體"""
        if not self.entity_embeddings or not self.keyword_embeddings: return []
        match_kg = []
        entity_df = pd.DataFrame(self.entity_embeddings["embeddings"])
        for entity in entities:
            try:
                idx = self.keyword_embeddings["keywords"].index(entity)
                emb = np.array(self.keyword_embeddings["embeddings"][idx])
                sims = Preprocessing.cosine_similarity_manual(entity_df, emb)[0]
                sorted_indices = np.argsort(sims)[::-1]
                for index in sorted_indices:
                    matched_entity = self.entity_embeddings["entities"][index].replace(" ", "_")
                    if matched_entity not in match_kg:
                        match_kg.append(matched_entity)
                        break
            except ValueError:
                continue
        return match_kg

    def search_knowledge_graph(self, matched_entities):
        """在知識圖譜中搜索相關子圖"""
        try:
            graph_dict = FindKG.find_whole_KG(self.driver)
            condition = {'distance': 5, 'size': 200}
            _, result_subgraph = GreedyDist.greedy_dist(graph_dict, matched_entities, condition)
            return result_subgraph
        except Exception as e:
            print(f"❌ Knowledge graph search error: {e}")
            return None

    def find_and_analyze_paths(self, subgraph, matched_entities):
        """尋找並分析路徑，同時回傳用於Prompt的path_join和用於視覺化的path_list"""
        try:
            all_paths = FindKG.subgraph_path_finding(subgraph, matched_entities)
            path_list, flag = KGtoPath_BFS.paths_in_neo4j_optimized_bfs_full(all_paths, 10, self.driver)
            path_join, _, _ = FindKG.combine_lists(community_search_paths=path_list, pagerank_values=None, top_n=10, flag=flag)
            return path_join, path_list
        except Exception as e:
            print(f"❌ Path analysis error: {e}")
            return None, None

    def generate_answer(self, question, path_join):
        """根據路徑提示生成最終答案"""
        try:
            prompt = PromptGenerate.GeneratePathPrompt(path_join, self.chat_gm)
            for _ in range(2): # 重試機制
                output_all = PromptGenerate.final_answer(question, prompt, self.chat_gm)
                output1 = PromptGenerate.extract_final_answer(output_all)
                if output1 and output1[0]:
                    return output1[0]
            return "[Error: Failed to generate answer after retries]"
        except Exception as e:
            print(f"❌ Answer generation error: {e}")
            return f"[Error in answer generation]"

    def process_medical_question(self, question, question_id):
        """處理醫療問題的完整主要流程"""
        try:
            # 步驟 1 & 2: 語言檢測與翻譯
            original_language = self.detect_language(question)
            english_question = self.translate_text(question, "English") if original_language == "chinese" else question

            # 步驟 3 & 4: 實體提取與匹配
            entities = self.extract_entities(english_question)
            if not entities: return {"success": False, "error": "無法從問題中提取有效的醫療實體。"}

            matched_entities = self.match_entities_to_kg(entities)
            if not matched_entities: return {"success": False, "error": "在知識圖譜中找不到與您問題相關的實體。"}

            # 步驟 5: 知識圖譜搜索
            subgraph = self.search_knowledge_graph(matched_entities)
            if not subgraph: return {"success": False, "error": "無法在知識圖譜中找到相關的資訊子圖。"}
            
            # 步驟 6: 路徑分析
            path_join, path_list = self.find_and_analyze_paths(subgraph, matched_entities)
            if not path_join: return {"success": False, "error": "無法分析實體間的關聯路徑。"}

            # 步驟 7 & 8: 生成答案並翻譯回來
            english_answer = self.generate_answer(english_question, path_join)
            final_answer = self.translate_text(english_answer, "繁體中文") if original_language == "chinese" else english_answer

            # 步驟 9: 視覺化
            visualization_url = None
            try:
                paths_graph = nx.DiGraph()
                if isinstance(path_list, dict):
                    for paths in path_list.values():
                        for path in paths:
                            for i in range(0, len(path) - 2, 2):
                                paths_graph.add_edge(path[i], path[i+2])
                
                if paths_graph.nodes():
                    output_dir = os.path.join(app.static_folder, 'visualizations')
                    os.makedirs(output_dir, exist_ok=True)
                    filename = f"graph_{question_id}.html"
                    output_filepath = os.path.join(output_dir, filename)
                    visualization_url = f"/visualizations/{filename}"
                    draw_paths_graph(
                        graph_to_draw=paths_graph,
                        match_kg=matched_entities,
                        output_filepath=output_filepath
                    )
            except Exception as e:
                print(f"❌ Visualization generation failed: {e}")

            return {
                "success": True,
                "data": {
                    "final_answer": final_answer,
                    "detected_language": original_language,
                    "extracted_entities": entities,
                    "matched_entities": matched_entities,
                    "visualization_url": visualization_url
                }
            }
        except Exception as e:
            print(f"❌ An unexpected error occurred in process_medical_question: {e}")
            return {"success": False, "error": f"伺服器內部處理錯誤: {str(e)}"}

# --- 全局服務實例 ---
medical_service = MedicalQAService()

# --- API 路由定義 ---
@app.route('/api/process-question', methods=['POST'])
def process_question():
    """處理醫療問題的主要API"""
    data = request.get_json()
    question = data.get('question', '')
    if not question.strip():
        return jsonify({"success": False, "error": "問題不能為空"}), 400
    
    # 產生唯一ID以避免瀏覽器快取舊圖檔
    question_id = f"{int(time.time())}_{random.randint(1000, 9999)}"
    result = medical_service.process_medical_question(question, question_id)
    
    return jsonify(result) if result["success"] else (jsonify(result), 500)

@app.route('/api/direct-gemini', methods=['POST'])
def direct_gemini():
    """純粹直接調用Gemini API"""
    data = request.get_json()
    question = data.get('question', '')
    if not question.strip():
        return jsonify({"success": False, "error": "問題不能為空"}), 400
    
    try:
        raw_response = medical_service.chat_gm(question)
        final_answer = raw_response
        if isinstance(raw_response, str):
            try: final_answer = json.loads(raw_response)
            except json.JSONDecodeError: pass
        
        return jsonify({
            "success": True,
            "data": {
                "original_question": question,
                "final_answer": str(final_answer),
                "method": "pure_gemini"
            }
        })
    except Exception as e:
        print(f"❌ Pure Gemini error: {e}")
        return jsonify({"success": False, "error": f"Gemini API 呼叫失敗: {str(e)}"}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康檢查API"""
    return jsonify({"status": "healthy", "service": "Medical QA API"})

@app.route('/visualizations/<path:filename>')
def serve_visualization(filename):
    """提供視覺化HTML檔案"""
    return send_from_directory(os.path.join(app.static_folder, 'visualizations'), filename)

# --- 前端靜態檔案服務 ---
@app.route('/')
def index():
    """提供前端主頁"""
    return send_from_directory('frontend', 'index.html')

@app.route('/<path:filename>')
def static_files(filename):
    """提供靜態文件（CSS, JS等）"""
    if filename in ['styles.css', 'script.js']:
        return send_from_directory('frontend', filename)
    return "File not found", 404

# --- 伺服器啟動 ---
if __name__ == '__main__':
    print("🚀 Starting Medical QA API Server...")
    print("🔗 Access the application at: http://localhost:5000/")
    app.run(debug=True, host='0.0.0.0', port=5000)