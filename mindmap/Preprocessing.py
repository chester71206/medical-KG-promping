import numpy as np
import re
from langchain import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
    )

def cosine_similarity_manual(x, y):
    dot_product = np.dot(x, y.T)
    norm_x = np.linalg.norm(x, axis=-1)
    norm_y = np.linalg.norm(y, axis=-1)
    sim = dot_product / (norm_x[:, np.newaxis] * norm_y)
    return sim


def autowrap_text(text, font, max_width):

    text_lines = []
    if font.getsize(text)[0] <= max_width:
        text_lines.append(text)
    else:
        words = text.split(' ')
        i = 0
        while i < len(words):
            line = ''
            while i < len(words) and font.getsize(line + words[i])[0] <= max_width:
                line = line + words[i] + ' '
                i += 1
            if not line:
                line = words[i]
                i += 1
            text_lines.append(line)
    return text_lines


def prompt_extract_keyword(input_text, chat, re1):
    template = """
    There are some samples:
    \n\n
    ### Instruction:\n'Learn to extract entities from the following medical questions.'\n\n### Input:\n
    <CLS>Doctor, I have been having discomfort and dryness in my vagina for a while now. I also experience pain during sex. What could be the problem and what tests do I need?<SEP>The extracted entities are\n\n ### Output:
    <CLS>Doctor, I have been having discomfort and dryness in my vagina for a while now. I also experience pain during sex. What could be the problem and what tests do I need?<SEP>The extracted entities are Vaginal pain, Vaginal dryness, Pain during intercourse<EOS>
    \n\n
    Instruction:\n'Learn to extract entities from the following medical answers.'\n\n### Input:\n
    <CLS>Okay, based on your symptoms, we need to perform some diagnostic procedures to confirm the diagnosis. We may need to do a CAT scan of your head and an Influenzavirus antibody assay to rule out any other conditions. Additionally, we may need to evaluate you further and consider other respiratory therapy or physical therapy exercises to help you feel better.<SEP>The extracted entities are\n\n ### Output:
    <CLS>Okay, based on your symptoms, we need to perform some diagnostic procedures to confirm the diagnosis. We may need to do a CAT scan of your head and an Influenzavirus antibody assay to rule out any other conditions. Additionally, we may need to evaluate you further and consider other respiratory therapy or physical therapy exercises to help you feel better.<SEP>The extracted entities are CAT scan of head (Head ct), Influenzavirus antibody assay, Physical therapy exercises; manipulation; and other procedures, Other respiratory therapy<EOS>
    \n\n
    Try to output:
    ### Instruction:\n'Learn to extract entities from the following medical questions.'\n\n### Input:\n
    <CLS>{input}<SEP>The extracted entities are\n\n ### Output:
    """

    prompt = PromptTemplate(
        template = template,
        input_variables = ["input"]
    )

    system_message_prompt = SystemMessagePromptTemplate(prompt = prompt)
    system_message_prompt.format(input = input_text)

    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt,human_message_prompt])
    chat_prompt_with_values = chat_prompt.format_prompt(input = input_text,\
                                                        text={})

    #response_of_KG = chat(chat_prompt_with_values.to_messages()).content
    # 修復：直接將prompt格式化為字符串，而不是消息列表
    formatted_prompt = chat_prompt_with_values.to_string()
    response_of_KG_raw = chat.invoke(formatted_prompt)
    
    # 處理Gemini返回的JSON格式響應
    if isinstance(response_of_KG_raw, str):
        try:
            # 嘗試解析JSON字符串
            import json
            response_of_KG = json.loads(response_of_KG_raw)
            if not isinstance(response_of_KG, str):
                response_of_KG = str(response_of_KG)
        except json.JSONDecodeError:
            # 如果不是JSON格式，直接使用原字符串
            response_of_KG = response_of_KG_raw
    else:
        response_of_KG = str(response_of_KG_raw)

    print(f"DEBUG: Gemini response for entity extraction: {response_of_KG}")
    
    question_kg = re.findall(re1, response_of_KG)
    print(f"DEBUG: Regex match result: {question_kg}")
    
    # 如果正則匹配失敗，嘗試更寬鬆的匹配 - 保持和原始程序一致的額外匹配邏輯
    if not question_kg:
        print("DEBUG: Standard regex failed, trying alternative patterns...")
        # 嘗試不同的匹配模式
        alternative_patterns = [
            r'entities are (.*?)<',
            r'entities are:?\s*(.*?)(?:\n|$)',
            r'entities.*?:\s*(.*?)(?:\n|$)',
            r'(?:entities|症狀|symptoms).*?[:：]\s*(.*?)(?:\n|$|。)'
        ]
        
        for pattern in alternative_patterns:
            matches = re.findall(pattern, response_of_KG, re.IGNORECASE | re.DOTALL)
            if matches:
                print(f"DEBUG: Alternative pattern matched: {pattern} -> {matches}")
                question_kg = matches
                break
    
    return question_kg