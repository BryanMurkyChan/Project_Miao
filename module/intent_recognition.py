from module.llm_client import get_client
from module.PROMPT_TEMPLATE import OTHER_PROMPT_TEMPLATE
import json
import re
import os

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
with open(CONFIG_PATH, "r", encoding="utf-8")as f:
    config = json.load(f)

eco_mode = config["ECO_MODE"]
if eco_mode:
    MODEL_FLASHX = "glm-4-flash"
else:
    MODEL_FLASHX = "glm-4-long"

client = get_client(MODEL_FLASHX)

Intent_Recognition_prompt = OTHER_PROMPT_TEMPLATE["Intent_Recognition_prompt"]

def get_former_query(text):
    pattern = r'(\d{4}年\d{1,2}月\d{1,2}日 \d{1,2}时\d{1,2}分\d{1,2}秒)'
    data = re.split(pattern, text)
    data = [part for part in data if part]
    if len(data)>=8:
        last_two_query = []
        for data_info in data[-10:]:
            if "\n爸比：" in data_info:
                data_info = data_info.replace("\n","")
                last_two_query.append(data_info)
        last_two_query = '\n'.join(last_two_query)
        return last_two_query
    else:
        return ""

def Intent_Recognition(query):
    if query:
        prompt = Intent_Recognition_prompt+"\n"+query
        response = client.generate_result(  
        query=prompt,
    )
        return response

def IR_result(text):
    if text:
        text = text.replace("`","")
        text = text.replace("json","")
        text = text.replace("\n","")
        matches = re.findall(r"\{.*?\}", text)
        output_json = json.loads(matches[0])
        return output_json

if __name__ == '__main__':
    file_path = r'C:\code\Miao\memory_storage\miao_memory\chat_history\2024-08-18_chat_history.txt'
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    print(get_former_query(text))