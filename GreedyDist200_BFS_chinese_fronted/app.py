# --- 1. 匯入必要的模組 ---
import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import json
import re

# --- 2. 載入 .env 檔案中的環境變數 ---
load_dotenv()


# vvvvvvvvvvvvvv 模擬你的 kg_query_handler.py 邏輯 vvvvvvvvvvvvvvvvv
# 在實際應用中，你應該 import 你真正的函式
# (此處模擬程式碼與前次相同，保持不變)
class MockGemini:
    def __init__(self, api_key):
        if not api_key:
            raise ValueError("API Key is missing. Please check your .env file.")
        print(f"Mock Gemini Initialized with API Key: {api_key[:4]}...{api_key[-4:]}")
    
    def __call__(self, prompt):
        if "translate" in prompt and "English" in prompt:
            text_to_translate = re.search(r'Text to translate: "(.*?)"', prompt).group(1)
            return json.dumps(f"[Mock Translated to English: {text_to_translate}]")
        elif "translate" in prompt and "繁體中文" in prompt:
            text_to_translate = re.search(r'Text to translate: "(.*?)"', prompt).group(1)
            return json.dumps(f"[模擬翻譯成中文: {text_to_translate}]")
        else:
            return json.dumps(f"This is a mock answer based on the processed English query.")

def translate_text(text_to_translate, target_language, llm_client):
    if not text_to_translate or not text_to_translate.strip():
        return ""
    prompt = f"Please translate the following text into {target_language}. ... Text to translate: \"{text_to_translate}\""
    response_json_str = llm_client(prompt)
    translated_text = json.loads(response_json_str)
    return translated_text.strip()

def process_query(question_text_in_english):
    print(f"--- 開始處理英文問題: {question_text_in_english} ---")
    print(">>> 步驟 1: 實體提取 (模擬)")
    print(">>> 步驟 2: 知識圖譜搜尋 (模擬)")
    print(">>> 步驟 3: 生成英文答案 (模擬)")
    final_english_answer = f"Based on the knowledge graph, the answer for '{question_text_in_english}' is related to mock entities A, B, and C."
    print(f"--- 處理完成，生成英文答案: {final_english_answer} ---")
    return final_english_answer
# --- END OF SIMULATION ---
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


# --- Flask App 設定 ---
app = Flask(__name__)

# --- 3. 從環境變數讀取 API KEY ---
# 使用 os.getenv() 讀取 .env 檔案中設定的變數
GEMINI_API_KEY = os.getenv("gemini_api_key_upgrade")

# --- 4. (推薦) 新增檢查機制 ---
if not GEMINI_API_KEY:
    # 如果找不到 API Key，程式會在這裡停止並提示使用者
    raise ValueError("GEMINI_API_KEY not found in .env file. Please check your configuration.")

# 初始化 Gemini 客戶端 (使用讀取到的金鑰)
# 在真實情況下，請取消註解 gemini.Gemini 並移除 MockGemini
# from llm import gemini 
# chat_gm = gemini.Gemini(API_KEY=GEMINI_API_KEY)
chat_gm = MockGemini(API_KEY=GEMINI_API_KEY)


@app.route('/')
def index():
    """渲染主頁面"""
    return render_template('index.html')


@app.route('/ask', methods=['POST'])
def ask():
    """處理來自前端的問答請求"""
    data = request.get_json()
    question = data.get('question')
    is_chinese = data.get('is_chinese', False)

    print("\n" + "="*50)
    print(f"收到問題: '{question}', 語言是否為中文: {is_chinese}")

    try:
        if is_chinese:
            print("執行中文處理流程...")
            english_question = translate_text(question, "English", chat_gm)
            print(f"翻譯後的英文問題: {english_question}")
            final_english_answer = process_query(english_question)
            final_answer = translate_text(final_english_answer, "繁體中文", chat_gm)
            print(f"翻譯回中文的最終答案: {final_answer}")
        else:
            print("執行英文處理流程...")
            final_english_answer = process_query(question)
            final_answer = final_english_answer

        return jsonify({'answer': final_answer})

    except Exception as e:
        print(f"處理過程中發生錯誤: {e}")
        return jsonify({'answer': f'抱歉，後端處理時發生錯誤: {e}'}), 500


if __name__ == '__main__':
    # Flask 預設會在 5000 port 運行
    app.run(debug=True)