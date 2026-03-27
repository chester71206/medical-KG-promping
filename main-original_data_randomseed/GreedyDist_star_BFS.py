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


if __name__ == "__main__":

    ### --- [Step 1]. Neo4j Sandbox Connection
    uri = os.getenv("neo4j_uri")
    username = os.getenv("neo4j_username")
    password = os.getenv("neo4j_password")
    print(codecs.decode(uri, 'unicode_escape')) # 檢查用

    # --- build KG 
    # data_path = './data/chatdoctor5k/train.txt'
    # BuildDatabase.BuildNeo4j(data_path)
    # ---

    driver = GraphDatabase.driver(codecs.decode(uri, 'unicode_escape'), auth=(username, password))
    # session = driver.session()
    ### --- End of [Step 1]
    

    ### --- [Step 2]. Gemini API Connection
    GEMINI_API_KEY = os.getenv("gemini_api_key_yi")
    chat_gm = gemini.Gemini(API_KEY=GEMINI_API_KEY)
    ### --- End of [Step 2]


    with open(f'../output/DenselyConnectedSubgraphRetrieval/Q714_Gemini1.5/GreedyDist*/{date.today()}_BFS.csv', 'w', newline='') as f4:
        writer = csv.writer(f4)
        writer.writerow(['Q_ID', 'Question', 'Answer'])


    with open('../data/chatdoctor5k/entity_embeddings.pkl','rb') as f1:
        entity_embeddings = pickle.load(f1)
    
    with open('../data/chatdoctor5k/keyword_embeddings_new.pkl','rb') as f2:
        keyword_embeddings = pickle.load(f2)

    
    ### --- [Step 3]. Extract Question & Answer's Entities
    re1 = r'The extracted entities are (.*?)<END>'
    re2 = r"The extracted entity is (.*?)<END>"
    re3 = r"<CLS>(.*?)<SEP>"


    csv_rows = []
    json_data = []
    with open("../data/chatdoctor5k/NER_chatgpt.json", "r") as f:
        lines = f.readlines()
        RANDOM_SEED = 42
        random.seed(RANDOM_SEED)
        # 總行數
        total_lines = len(lines)
        # 使用隨機種子選取 714 個題號
        selected_indices = random.sample(range(total_lines), 718)
        exclude = {552, 2553, 4333, 5084}
        selected_indices = [q for q in selected_indices if q not in exclude]
        # selected_indices = selected_indices[249:250]
        print(f"隨機選取的題號: {selected_indices}")

        for count, index in enumerate(tqdm(selected_indices, desc="Processing Questions", unit="question", dynamic_ncols=True)):
            try:  
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

                    
                question_kg = re.findall(re1,input)

                if len(question_kg) == 0:
                    question_kg = re.findall(re2,input)
                    if len(question_kg) == 0:
                        print("<Warning> no entities found", input)
                        continue
                question_kg = question_kg[0].replace("<END>","").replace("<EOS>","")
                question_kg = question_kg.replace("\n","")
                question_kg = question_kg.split(", ")

                
                match_kg = []
                entity_embeddings_emb = pd.DataFrame(entity_embeddings["embeddings"])
            

                for kg_entity in question_kg:
                    
                    keyword_index = keyword_embeddings["keywords"].index(kg_entity)
                    kg_entity_emb = np.array(keyword_embeddings["embeddings"][keyword_index])

                    cos_similarities = Preprocessing.cosine_similarity_manual(entity_embeddings_emb, kg_entity_emb)[0]
                    max_index = cos_similarities.argmax()
                            
                    match_kg_i = entity_embeddings["entities"][max_index]
                    while match_kg_i.replace(" ","_") in match_kg:
                        cos_similarities[max_index] = 0
                        max_index = cos_similarities.argmax()
                        match_kg_i = entity_embeddings["entities"][max_index]

                    match_kg.append(match_kg_i.replace(" ","_"))

                print('Question Entities:\n', match_kg, "\n") ### 與問題keywords相符的entities
                ### --- End of [Step 3]
                

                start = perf_counter()
                ### --- [Step 4]. Densely Connected Subgraph Retrieval (GreedyDist*)
                graph_dict = FindKG.find_whole_KG(driver)

                condition_constraint = {'distance':5}
                result_subgraph = GreedyDist.greedy_gen(graph_dict, match_kg, condition_constraint)
                # print("result_subgraph: \n", len(result_subgraph))
                
                ## 計算執行時間用
                step4_start = perf_counter()
                print(f"Find Subgraph: { step4_start - start:.3f} 秒")
                ### --- End of [Step 4]


                ### --- [Step 5]. Path Selection - BFS
                all_paths = FindKG.subgraph_path_finding(result_subgraph, match_kg)

                top_n=10
                path_list, flag = KGtoPath_BFS.paths_in_neo4j_optimized_bfs_full(all_paths, top_n, driver)

                (path_join, 
                 path_join_list, 
                 path_nodes_count) = FindKG.combine_lists(community_search_paths=path_list, pagerank_values=None, top_n=top_n, flag=flag)
            
                step5_start = perf_counter()
                print(f"Path Selection: {step5_start - step4_start:.3f} 秒")
                 ### --- End of [Step 5]


                ### --- [Step 6]. Paths to Natural Language
                prompt = PromptGenerate.GeneratePathPrompt(path_join, chat_gm)

                step6_start = perf_counter()
                print(f"Generate Paths in NL: {step6_start - step5_start:.3f} 秒")
                ### --- End of [Step 6]

                
                ### --- [Step 7]. Final Answer Generation
                # 若沒有回答出來，讓 LLM 重新回答
                times=1
                while True:
                    output_all = PromptGenerate.final_answer(input_text[0], prompt, chat_gm)
                    output1 = PromptGenerate.extract_final_answer(output_all)
                    # output1, output2, output3 = extract_final_answer(output_all)
                    if len(output1) > 0:
                        output1 = output1[0]
                        # output2 = output2[0]
                        # output3 = output3[0]
                        break
                    elif times == 2:
                        break
                    else:
                        times+=1
                print("Final Answer: ", output1)
                # time.sleep(60)

                step6_start = perf_counter()
                print(f"Generate Final Answer: {step6_start - step5_start:.3f} 秒")
                ### --- End of [Step 7] 


                ### --- [Step 8]. Save Results
                # --- Save csv
                with open(f'../output/DenselyConnectedSubgraphRetrieval/Q714_Gemini1.5/GreedyDist*/{date.today()}_BFS.csv', 'a+', newline='') as f4:
                    writer = csv.writer(f4)
                    writer.writerow([index+1,  input_text[0], output1])
                    f4.flush()

                # # --- Save Json
                # json_data.append({
                #     "Q_ID": index+1,
                #     "question": input_text[0],
                #     "query_nodes": match_kg,
                #     "subgraph": result_subgraph,
                #     "paths-list": path_join_list,
                #     "output": output1
                # })
            
            except Exception as e:
                print(f"錯誤在第 {index+1} 題: {e}")

    # [Step 8]. JSON 統一寫入檔案
    # with open(f"../output/DenselyConnectedSubgraphRetrieval/Q714_Gemini1.5/GreedyDist*/{date.today()}_BFS.json", "w", encoding="utf-8") as f:
    #     json.dump(json_data, f, ensure_ascii=False, indent=4)
    ### --- End of [Step 8]

    driver.close()
