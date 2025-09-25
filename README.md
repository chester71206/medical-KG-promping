# 快速開始

請依照以下步驟設定並執行專案。

---

## 1. 建立 Conda 環境

首先，使用 Conda 建立一個新的虛擬環境，並指定 Python 版本：

```bash
conda create -n your_env_name python=3.10.12
```

請將 `your_env_name` 替換為您想要的環境名稱。

---

## 2. 安裝依賴套件

啟動剛剛建立的 Conda 環境：

```bash
conda activate your_env_name
```

接著，安裝 `requirements.txt` 檔案中列出的所有依賴套件：

```bash
pip install -r requirements.txt
```

---

## 3. 設定環境變數

將 `.env_template` 檔案複製一份並重新命名為 `.env`：

```bash
cp .env_template .env
```

然後，編輯 `.env` 檔案，設定以下變數：

- `neo4j_uri`  
- `neo4j_username`  
- `neo4j_password`  
- `gemini_api_key_upgrade`  

---

## 4. 執行應用程式

完成所有設定後，執行 `app.py` 檔案來啟動應用程式：

```bash
python app.py
```
