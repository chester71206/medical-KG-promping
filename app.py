from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
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
import warnings
import re
warnings.filterwarnings("ignore")

# 載入環境變數
load_dotenv()

# 導入你的自定義模組
import BuildDatabase
from llm import openrouter, gemini
from mindmap import Preprocessing
from communitysearch import FindKG, GreedyDist, KGtoPath_PR, KGtoPath_BFS, PromptGenerate
from other import KG_vision_pyvis

app = Flask(__name__)
CORS(app)  # 允許跨域請求

class MedicalQAService:
    def __init__(self):
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
    
        
    def load_embeddings(self):
        """載入實體和關鍵詞嵌入向量"""
        try:
            with open('./data/chatdoctor5k/entity_embeddings.pkl','rb') as f1:
                self.entity_embeddings = pickle.load(f1)
            with open('./data/chatdoctor5k/keyword_embeddings_new.pkl','rb') as f2:
                self.keyword_embeddings = pickle.load(f2)
            print("Embeddings loaded successfully")
        except Exception as e:
            print(f"Error loading embeddings: {e}")
            self.entity_embeddings = None
            self.keyword_embeddings = None
    
    def detect_language(self, text):
        """檢測輸入文字的語言"""
        if not text or text.strip() == "":
            return "unknown"
        
        # 中文字符正則表達式（包含繁體和簡體）
        chinese_pattern = r'[\u4e00-\u9fff\u3400-\u4dbf]'
        english_pattern = r'[a-zA-Z]'
        
        chinese_count = len(re.findall(chinese_pattern, text))
        english_count = len(re.findall(english_pattern, text))
        
        # 只要有任何中文字符就判断为中文（不管英文字符多少）
        if chinese_count > 0:
            return "chinese"
        elif english_count > 0:
            return "english"
        else:
            return "unknown"
    
    def translate_text(self, text_to_translate, target_language):
        """使用Gemini翻譯文字"""
        if not text_to_translate or not text_to_translate.strip():
            return ""
            
        prompt = f"Please translate the following text into {target_language}. Provide only the translated text without any other explanations or context.\n\nText to translate: \"{text_to_translate}\""
        
        try:
            response_json_str = self.chat_gm(prompt)
            # Gemini返回的已經是JSON格式化的字符串，需要解析
            if isinstance(response_json_str, str):
                try:
                    # 嘗試解析JSON字符串
                    translated_text = json.loads(response_json_str)
                    if isinstance(translated_text, str):
                        return translated_text.strip()
                    else:
                        return str(translated_text).strip()
                except json.JSONDecodeError:
                    # 如果不是JSON格式，直接返回原字符串
                    return response_json_str.strip()
            else:
                return str(response_json_str).strip()
        except Exception as e:
            print(f"Translation error: {e}")
            return f"[Translation Error: {text_to_translate}]"
    
    def extract_entities(self, question):
        """從問題中提取醫療實體 - 完全按照原始程序GreedyDist200_BFS_chinese_3.py的邏輯"""
        try:
            print(f"Dynamically extracting entities from the re-translated question...")
            re1_for_extraction = r'<CLS>.*<SEP>The extracted entities are (.*?)<EOS>'
            extracted_entities_raw = Preprocessing.prompt_extract_keyword(
                question, 
                self.chat_gm, 
                re1_for_extraction
            )
            print(f"extracted_entities_raw: {extracted_entities_raw}")

            if not extracted_entities_raw or not extracted_entities_raw[0]:
                print(f"<Warning> Failed to dynamically extract entities for question: {question}")
                return []

            question_kg = [entity.strip() for entity in extracted_entities_raw[0].split(',')]
            print(f'Dynamically Extracted question_kg:\n {question_kg}')
            return question_kg
            
        except Exception as e:
            print(f"Entity extraction error: {e}")
            return []
    
    def match_entities_to_kg(self, question_entities):
        """將問題實體匹配到知識圖譜實體 - 完全按照原始程序的邏輯"""
        if not self.entity_embeddings or not self.keyword_embeddings:
            return []
        
        match_kg = []
        entity_embeddings_emb = pd.DataFrame(self.entity_embeddings["embeddings"])
        
        for kg_entity in question_entities:
            try:
                keyword_index = self.keyword_embeddings["keywords"].index(kg_entity)
            except ValueError:
                print(f"<Warning> Entity '{kg_entity}' not in keyword embeddings. Skipping.")
                continue
            
            kg_entity_emb = np.array(self.keyword_embeddings["embeddings"][keyword_index])
            cos_similarities = Preprocessing.cosine_similarity_manual(entity_embeddings_emb, kg_entity_emb)[0]
            max_index = cos_similarities.argmax()
            match_kg_i = self.entity_embeddings["entities"][max_index]
            
            while match_kg_i.replace(" ","_") in match_kg:
                cos_similarities[max_index] = 0
                max_index = cos_similarities.argmax()
                match_kg_i = self.entity_embeddings["entities"][max_index]
                
            match_kg.append(match_kg_i.replace(" ","_"))

        if not match_kg:
            print("<Warning> No matching entities found in KG. Cannot proceed.")
            return []
        
        print(f'Question Entities:\n {match_kg}')
        return match_kg
    
    def search_knowledge_graph(self, matched_entities):
        """在知識圖譜中搜索相關子圖"""
        try:
            graph_dict = FindKG.find_whole_KG(self.driver)
            condition_constraint = {'distance': 5, 'size': 200}
            distance, result_subgraph = GreedyDist.greedy_dist(
                graph_dict, matched_entities, condition_constraint
            )
            return result_subgraph
        except Exception as e:
            print(f"Knowledge graph search error: {e}")
            return None
    
    def find_and_analyze_paths(self, subgraph, matched_entities):
        """尋找並分析路徑"""
        try:
            all_paths = FindKG.subgraph_path_finding(subgraph, matched_entities)
            top_n = 10
            path_list, flag = KGtoPath_BFS.paths_in_neo4j_optimized_bfs_full(
                all_paths, top_n, self.driver
            )
            path_join, path_join_list, path_nodes_count = FindKG.combine_lists(
                community_search_paths=path_list, 
                pagerank_values=None, 
                top_n=top_n, 
                flag=flag
            )
            return path_join
        except Exception as e:
            print(f"Path analysis error: {e}")
            return None
    
    def generate_answer(self, question, path_join):
        """生成最終答案"""
        try:
            prompt = PromptGenerate.GeneratePathPrompt(path_join, self.chat_gm)
            
            # 重試機制
            for attempt in range(2):
                output_all = PromptGenerate.final_answer(question, prompt, self.chat_gm)
                output1 = PromptGenerate.extract_final_answer(output_all)
                
                if len(output1) > 0:
                    return output1[0]
            
            return "[Error: Failed to generate answer after retries]"
        except Exception as e:
            print(f"Answer generation error: {e}")
            return f"[Error: {str(e)}]"
    
    def process_medical_question(self, question):
        """處理醫療問題的主要流程"""
        try:
            # 步驟1: 語言檢測
            detected_language = self.detect_language(question)
            
            # 步驟2: 翻譯處理（只要有中文就翻譯）
            if detected_language == "chinese":
                english_question = self.translate_text(question, "English")
                original_language = "chinese"
            else:
                english_question = question
                original_language = detected_language
            
            # 步驟3: 實體提取
            entities = self.extract_entities(english_question)
            if not entities:
                return {
                    "success": False,
                    "error": "Failed to extract medical entities from the question"
                }
            
            # 步驟4: 實體匹配
            matched_entities = self.match_entities_to_kg(entities)
            if not matched_entities:
                return {
                    "success": False,
                    "error": "No matching entities found in knowledge graph"
                }
            
            # 步驟5: 知識圖譜搜索
            subgraph = self.search_knowledge_graph(matched_entities)
            if not subgraph:
                return {
                    "success": False,
                    "error": "Failed to find relevant subgraph"
                }
            
            # 步驟6: 路徑分析
            path_join = self.find_and_analyze_paths(subgraph, matched_entities)
            if not path_join:
                return {
                    "success": False,
                    "error": "Failed to analyze knowledge paths"
                }
            
            # 步驟7: 生成答案
            english_answer = self.generate_answer(english_question, path_join)
            
            # 步驟8: 翻譯回原語言
            if original_language == "chinese":
                final_answer = self.translate_text(english_answer, "繁體中文")
            else:
                final_answer = english_answer
            
            return {
                "success": True,
                "data": {
                    "original_question": question,
                    "detected_language": detected_language,
                    "processed_question": english_question,
                    "extracted_entities": entities,
                    "matched_entities": matched_entities,
                    "final_answer": final_answer
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"Processing error: {str(e)}"
            }

# 全局服務實例
medical_service = MedicalQAService()

@app.route('/api/detect-language', methods=['POST'])
def detect_language():
    """語言檢測API"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        language = medical_service.detect_language(text)
        
        return jsonify({
            "success": True,
            "language": language
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/translate', methods=['POST'])
def translate():
    """翻譯API"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        target_language = data.get('target_language', 'English')
        
        translated_text = medical_service.translate_text(text, target_language)
        
        return jsonify({
            "success": True,
            "translated_text": translated_text
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/process-question', methods=['POST'])
def process_question():
    """處理醫療問題的主要API"""
    try:
        data = request.get_json()
        question = data.get('question', '')
        
        if not question.strip():
            return jsonify({
                "success": False,
                "error": "Question cannot be empty"
            }), 400
        
        result = medical_service.process_medical_question(question)
        
        if result["success"]:
            return jsonify(result)
        else:
            return jsonify(result), 500
            
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Server error: {str(e)}"
        }), 500

@app.route('/api/direct-gemini', methods=['POST'])
def direct_gemini():
    """純粹直接調用Gemini API，不經過任何處理或翻譯"""
    try:
        data = request.get_json()
        question = data.get('question', '')
        
        if not question.strip():
            return jsonify({
                "success": False,
                "error": "Question cannot be empty"
            }), 400
        
        try:
            # 純粹直接調用Gemini，用戶輸入什麼就問什麼
            raw_response = medical_service.chat_gm(question)
            
            # 解析JSON響應
            if isinstance(raw_response, str):
                try:
                    final_answer = json.loads(raw_response)
                    if not isinstance(final_answer, str):
                        final_answer = str(final_answer)
                except json.JSONDecodeError:
                    final_answer = raw_response
            else:
                final_answer = str(raw_response)
            
            return jsonify({
                "success": True,
                "data": {
                    "original_question": question,
                    "final_answer": final_answer,
                    "method": "pure_gemini"
                }
            })
            
        except Exception as e:
            print(f"Pure Gemini error: {e}")
            return jsonify({
                "success": False,
                "error": f"Gemini API error: {str(e)}"
            }), 500
            
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Server error: {str(e)}"
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康檢查API"""
    return jsonify({
        "status": "healthy",
        "service": "Medical QA API",
        "timestamp": time.time()
    })

# 前端文件服務路由
@app.route('/')
def index():
    """提供前端主頁"""
    return send_file('frontend/index.html')

@app.route('/<path:filename>')
def static_files(filename):
    """提供靜態文件（CSS, JS等）"""
    # 檢查文件是否在frontend目錄中
    if filename in ['styles.css', 'script.js']:
        return send_from_directory('frontend', filename)
    # 如果不是前端文件，返回404
    return "File not found", 404

if __name__ == '__main__':
    print("Starting Medical QA API Server...")
    print("Swagger UI available at: http://localhost:5000/")
    app.run(debug=True, host='0.0.0.0', port=5000)