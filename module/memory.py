import os
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from module.conversation import postprocess_text, preprocess_text, Conversation, Role
from module.llm_client import get_client
from module.PROMPT_TEMPLATE import RAG_PROMPT_TEMPLATE, OTHER_PROMPT_TEMPLATE
from streamlit.delta_generator import DeltaGenerator
import json
import re
from datetime import datetime, timedelta
from langchain.chains import LLMChain
import time
from openai import OpenAI
import random

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
with open(CONFIG_PATH, "r", encoding="utf-8")as f:
    config = json.load(f)

eco_mode = config["ECO_MODE"]
User_Identity = config["User_Identity"]
Miao_Name = config["Miao_Name"]

if eco_mode:
    MODEL_LONG = MODEL_FLASH = "glm-4-flash"
else:
    MODEL_LONG = "glm-4-long"
    MODEL_FLASH = "glm-4-flash"

long_client = get_client(MODEL_LONG)
flash_client = get_client(MODEL_FLASH)

API_KEY = config["OPENAI_API_KEY"]
ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
now = datetime.now()
today = datetime.now().strftime("%Y-%m-%d")
yesterday = (now - timedelta(days=1)).strftime("%Y-%m-%d")

CHAT_HISTORY_FILE = f'./memory_storage/miao_memory/chat_history/{today}_chat_history.txt'
CHAT_HISTORY_FILE_YESTERDAY = f"./memory_storage/miao_memory/chat_history/{yesterday}_chat_history.txt"
MIAO_DIARY_PATH = "./memory_storage/miao_memory/miao_diary/miao_diary.json"
INVOKE_RECORD_TXT_PATH = "./memory_storage/VBstorage/Invoke_Record.txt"
CHAT_HISTORY_FOLDER = "./memory_storage/miao_memory/chat_history"
TIME_CHINESE_FORMAT = "{0}年{1}月{2}日 {3}时{4}分{5}秒".format(
            now.year, now.month, now.day, now.hour, now.minute, now.second
        )

def activate_glm4():
    llm = ChatOpenAI(
        model_name=MODEL_LONG,
        openai_api_base=ZHIPU_BASE_URL,
        openai_api_key=API_KEY,
        streaming=False,
    ) 
    return llm
   
class Abstract():
    def __init__(self):
        pass

    def default_abstract(client,model_name,prompt):
        response = long_client.generate_result(query=prompt)
        return response

    def map_reduce_abstract(llm,prompt_template,text):
        llm = activate_glm4()
        
        text_spliter = RecursiveCharacterTextSplitter(separators=['\n\n','\n'], chunk_size = 6000, chunk_overlap = 150)
        docs = text_spliter.create_documents([text])
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
        chain = load_summarize_chain(llm=llm, chain_type="map_reduce", verbose = True, return_intermediate_steps=False, map_prompt=PROMPT, combine_prompt=PROMPT)
        output = chain({"input_documents": docs}, return_only_outputs=True)
        abstract_text = output["output_text"]
        return abstract_text

class Recall():
    def __init__(self):
        pass

    def flash_back_triger(self, prompt_text):
        random_num = random.randint(1,12)
        if random_num == 1:
            return 1
        else:
            return None

    @staticmethod
    def date_info_extraction(query):
        prompt = OTHER_PROMPT_TEMPLATE['Get_Time'].format(time = TIME_CHINESE_FORMAT, query = query)
        date = flash_client.generate_result(query=prompt)
        return date

    def date_memory_detection(self, extract_date):
        if "不" in extract_date or "没" in extract_date:
            return False
        else:
            return True

    def date_memory_call_back(self, date):
        pattern = r"\d{4}-\d{2}-\d{2}"
        matches = re.findall(pattern, date)
        date = matches[0]
        with open(MIAO_DIARY_PATH,"r",encoding="utf-8")as f:
            memory_data_json = json.load(f)

        for i in memory_data_json:
            if i["time"] == date:
                memory =  RAG_PROMPT_TEMPLATE['Remember_prompt_template_default'].format(
                    Miao_Name = config["Miao_Name"], 
                    memory = i["memory"],
                    User_Identity = config["User_Identity"],)
                return memory

    def remember(self, embedding, vector, question, k):
        info = vector.query(query=question, EmbeddingModel=embedding, k=k)
        return info

    def remember_with_vector(self, embedding, vector, question):
        info = vector.query_with_vector(query=question, EmbeddingModel=embedding, k=2)
        return info

    def self_query(self, query, embedding, vector):
        prompt_template = OTHER_PROMPT_TEMPLATE["Self_Query"]
        prompt = prompt_template.format(text=query)
        output = flash_client.generate_result(prompt)
        
        llm_cut = output.split("\n")
        query_list = []
        for i in llm_cut:
            if "-" in i:
                query_list.append(i)
        
        print(query_list)

        query_result = []
        for i in query_list:
            info = vector.query_with_vector(query=i, EmbeddingModel=embedding, k=5)
            query_result.append(info)

        return query_result


    def find_dict_by_value(self, value, lst):
        for item in lst:
            if item["memory"] == value:
                return item

class History_Management():
    def __init__(self):
        pass

    def last_history(self,history):
        history_data = []
        if len(history)>7:
            for chat in history[-6:]:
                history_data.append(chat.content)
        return history_data

    def count_history(self, history):
        history_data = ""
        for chat in history:
            history_data += chat.content
        return len(history_data)

    def save_conversation_to_file(self, history, is_document_mode=False, is_fc_mode=False):
        today = datetime.now().strftime("%Y-%m-%d")
        CHAT_HISTORY_FILE = f'./memory_storage/miao_memory/chat_history/{today}_chat_history.txt'
        with open(CHAT_HISTORY_FILE, 'a', encoding='utf-8') as file:
            now = datetime.now()
            TIME_CHINESE_FORMAT = "{0}年{1}月{2}日 {3}时{4}分{5}秒".format(
            now.year, now.month, now.day, now.hour, now.minute, now.second
        )
            file.write(TIME_CHINESE_FORMAT + "\n")
            if str(history[-1].role) == "<|user|>":
                if is_document_mode:
                    text = history[-1].content
                    start_index = text.find("<文章开始>")
                    end_index = text.find("<文章结束>")
                    if start_index != -1 and end_index != -1:
                        text = text[:start_index] + text[end_index + len("<文章结束>"):]
                    file.write(f"{User_Identity}：" + text + "\n")
                else:
                    file.write(f"{User_Identity}：" + history[-1].content + "\n")
            elif str(history[-1].role) == "<|assistant|>":
                if is_document_mode:
                    file.write(f"[文档模式]{Miao_Name}：" + history[-1].content + "\n")
                elif is_fc_mode:
                    file.write("调用工具" + history[-1].tool + f"\n[工具模式]{Miao_Name}：\n" + history[-1].content + "\n")
                else:
                    file.write(f"{Miao_Name}：" + history[-1].content + "\n")

    def append_conversation(
            self,
            conversation: Conversation,
            history: list[Conversation],
            placeholder: DeltaGenerator | None = None,
            save_and_show: bool = True,
            document_mode: bool = False,
            fc_mode: bool = False,
    ) -> None:
        history.append(conversation)
        if save_and_show == True:
            conversation.show(placeholder)
            if document_mode:
                self.save_conversation_to_file(history,is_document_mode=True)
            elif fc_mode:
                self.save_conversation_to_file(history,is_fc_mode=True)
            else:
                self.save_conversation_to_file(history)

    def save_memory_record(self,memory):
        with open(INVOKE_RECORD_TXT_PATH,"a",encoding="utf-8")as f:
            f.write( memory + "\n")

    def return_str_history(self, history):
        history_data = []
        for chat in history:
            index = history.index(chat)
            if str(history[index].role) == "<|user|>":
                history_data.append(f"{User_Identity}："+ history[index].content + "\n")
            elif str(history[index].role) == "<|assistant|>":
                history_data.append(f"{Miao_Name}："+ history[index].content + "\n")
        history_str = "\n".join(history_data)
        return history_str
    
    def get_last_three_history(self):
        if os.path.exists(CHAT_HISTORY_FILE):
            with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
                text = f.read()
        else:
            if os.path.exists(CHAT_HISTORY_FILE_YESTERDAY):
                with open(CHAT_HISTORY_FILE_YESTERDAY, "r", encoding="utf-8") as f:
                    text = f.read()
            else:
                try:
                    files = [f for f in os.listdir(CHAT_HISTORY_FOLDER) if f.endswith('.txt')]
                    files.sort(key=lambda f: os.path.getmtime(os.path.join(CHAT_HISTORY_FOLDER, f)), reverse=True)
                    
                    with open(os.path.join(CHAT_HISTORY_FOLDER, files[0]), "r", encoding="utf-8") as f:
                        text = f.read()
                except IndexError:
                    print("文件夹中没有找到txt文件。")
                    text = ""

        pattern = r'(\d{4}年\d{1,2}月\d{1,2}日 \d{1,2}时\d{1,2}分\d{1,2}秒)'
        data = re.split(pattern, text)
        data = [part for part in data if part]
        if len(data) <=12:
            last_three_memory_prompt = "记忆数量还不够呢，多聊聊吧，喵呜~"
        else:
            last_three_memory = data[-11].replace("\n","")+"\n"+data[-9].replace("\n","")+"\n"+data[-7].replace("\n","")+"\n"+data[-5].replace("\n","")+"\n"+data[-3].replace("\n","")+"\n"+data[-1].replace("\n","")
            last_three_memory_prompt = OTHER_PROMPT_TEMPLATE['Get_Last_Three_Rounds_Abstract'].format(last_three_memory = last_three_memory)

        return last_three_memory_prompt

if __name__ == "__main__":
    activate_glm4_new()
    with open(r"C:\code\Miao\memory_storage\miao_memory\chat_memory\2024-08-12_chat_memory.txt","r",encoding="utf-8")as f:
        text = f.read()
    
