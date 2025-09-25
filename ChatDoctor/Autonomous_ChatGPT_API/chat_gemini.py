# import openai
# import pandas as pd

import google.generativeai as genai
import pandas as pd  # type: ignore
import os
import time

# print(os.getenv("GEMINI_API_KEY"))
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.0-flash")


def csv_prompter(question, csv_name):
    # Step1
    fulltext = (
        "A question is provided below. Given the question, extract "
        + "keywords from the text. Focus on extracting the keywords that we can use "
        + "to best lookup answers to the question. \n"
        + "---------------------\n"
        + "{}\n".format(question)
        + "---------------------\n"
        + "Provide keywords in the following comma-separated format.\nKeywords: "
    )

    response = model.generate_content(fulltext)
    keyword_list = response.text.strip().split(", ")
    # print("stwp1 - keywords: \n", keyword_list)

    # Step2
    df = pd.read_csv(csv_name)
    divided_text = []
    csvdata = df.to_dict("records")
    step_length = 15
    for csv_item in range(0, len(csvdata), step_length):
        csv_text = (
            str(csvdata[csv_item : csv_item + step_length])
            .replace("}, {", "\n\n")
            .replace('"', "")
        )  # .replace("[", "").replace("]", "")
        # print("step2 - csv_text: \n", csv_text)
        divided_text.append(csv_text)

    answer_llm = ""

    score_textlist = [0] * len(divided_text)

    for i, chunk in enumerate(divided_text):
        for t, keyw in enumerate(keyword_list):
            if keyw.lower() in chunk.lower():
                score_textlist[i] = score_textlist[i] + 1

    answer_list = []
    divided_text = [
        item for _, item in sorted(zip(score_textlist, divided_text), reverse=True)
    ]

    for i, chunk in enumerate(divided_text):
        if i > 6:
            continue

        fulltext = (
            f"{chunk}\n"
            "---------------------\n"
            f"Based on the Table above and not prior knowledge, "
            f"Select the Table Entries that will help to answer the question: {question}\n"
            'Output in the format of " Disease: <>; Symptom: <>; Medical Test: <>; Medications: <>;". '
            "If there is no useful form entries, output: 'No Entry'."
        )

        response = model.generate_content(fulltext)
        answer_llm = (
            response.text.strip()
        )  # ← 取代 OpenAI choices[0]['message']['content']

        # print("\nAnswer: " + answer_llm)
        # print("round:", i)

        if "No Entry" not in answer_llm:
            answer_list.append(answer_llm)
        
        time.sleep(5)

    final_prompt = (
        "The original question is as follows: {}\n\n".format(question)
        + "Based on this Table:\n"
        + "------------\n"
        + "{}\n".format(str("\n\n".join(answer_list)))
        + "------------\n"
        + """Please analyze the above symptoms and answer in the following format:\n

            If there is no reference table available, please answer using your own medical knowledge.\n    
        
            1. **Most Likely Disease:**\n  
            State only one most likely disease based on the symptoms. Briefly explain why.\n

            2. **Recommended Medication(s):**\n  
            List multiple possible medications by name only. Do not provide explanations.\n

            3. **Suggested Medical Test(s):**\n  
            List multiple relevant medical tests by name only. Do not provide explanations.\n\n"""
        + "Answer: "
    )

    print(final_prompt)
    final_response = model.generate_content(final_prompt)  # ← OpenAI 改為 Gemini
    answer_llm = final_response.text.strip()  # ← 同樣是取得 text

    print("\nFinal Answer: " + answer_llm)

    return answer_llm


# question = "If I have frontal headache, fever, and painful sinuses, what disease should I have, and what medical test should I take?"
# csv_name = "disease_database_mini.csv"
# FinalAnswer = csv_prompter(question, csv_name)
# print(FinalAnswer)
