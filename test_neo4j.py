from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

# 確保 .env 檔案被讀取
load_dotenv()

# 從環境變數讀取憑證
uri = os.getenv("neo4j_uri")
username = os.getenv("neo4j_username")
password = os.getenv("neo4j_password")

print("--- 正在驗證 Neo4j 連線資訊 ---")
print(f"[*] 嘗試連線到 URI: {uri}")
print(f"[*] 使用者名稱: {username}")
# 為了安全，不直接印出密碼
print(f"[*] 密碼: {'*' * len(password) if password else '未提供 (None)'}")
print("------------------------------------")

# 檢查是否成功讀取到變數
if not all([uri, username, password]):
    print("\n[失敗] 錯誤：.env 檔案中的 uri、username 或 password 未能成功讀取。請檢查變數名稱（需全小寫）和檔案位置。")
else:
    driver = None  # 先宣告 driver
    try:
        # 嘗試建立一個 driver 物件並驗證連線
        driver = GraphDatabase.driver(uri, auth=(username, password))
        driver.verify_connectivity()
        print("\n[成功] 認證成功！你的連線資訊完全正確。")

    except Exception as e:
        print(f"\n[失敗] 認證失敗！請檢查你的連線資訊。")
        print(f"錯誤詳情: {e}")

    finally:
        # 確保 driver 被關閉
        if driver:
            driver.close()