# 本文件用于触发漆小喵的主动信息机制
from datetime import datetime
import subprocess
import random
import json
import faiss
import os
import time
import numpy as np
from windows_toasts import WindowsToaster, Toast, ToastDuration
from module.tools import web_search_response
from module.llm_client import get_client
from module.memory import Recall
from module.vector_base import VectorStore
from module.embeddings import BgeEmbedding
from module.PROMPT_TEMPLATE import RAG_PROMPT_TEMPLATE
import main

today = datetime.now().strftime('%Y-%m-%d')
now = datetime.now()
INDEX_PATH = './memory_storage/VBstorage'
CHAT_HISTORY_FILE = f'./memory_storage/miao_memory/chat_history/{today}_chat_history.txt'
WEB_SEARCH_RECORD = f'./memory_storage/miao_memory/chat_memory/web_search_record.json'
MEMORY_DB_PATH = "./memory_storage/VBstorage/Memory_DB.json"
MEMORY_VECTORS_PATH = "./memory_storage/VBstorage/Memory_Vectors.json"
CONFIG_PATH = './module/config.json'
TIME_CHINESE_FORMAT = "{0}年{1}月{2}日 {3}时{4}分{5}秒".format(
            now.year, now.month, now.day, now.hour, now.minute, now.second
        )

with open(CONFIG_PATH, "r", encoding="utf-8")as f:
    config = json.load(f)

Miao_Name = config["Miao_Name"]
Miao_Nick_Name = config["Miao_Nick_Name"]
Miao_Info_Brief = config["Miao_Info_Brief"] 
User_Identity = config["User_Identity"]

llm_client = get_client("glm-4-flash")
Recall = Recall()
embedding = BgeEmbedding()

def check_file_updated(file_path):
    if not os.path.exists(file_path):
        return False
    current_time = time.time()
    modified_time = os.path.getmtime(file_path)
    return (current_time - modified_time) < 10

def load_json(MEMORY_DB_PATH, MEMORY_VECTORS_PATH):
    with open(MEMORY_DB_PATH,"r",encoding = "utf-8")as f:
        memory_db = json.load(f)
    memory_db_content = [memory["memory"] for memory in memory_db]

    with open(MEMORY_VECTORS_PATH,"r",encoding="utf-8")as f:
        vectors_list = json.load(f)
    memory_vector = [np.array(vector, dtype=np.float32) for vector in vectors_list]

    return [memory_db, memory_db_content, memory_vector]

def load_memory(memory_db_content, memory_vector):
    vector_store = VectorStore(memory_db_content, memory_vector)
    index = faiss.read_index(INDEX_PATH+"/Memory_Vectors.index")
    vector_store.set_index(index=index)
    # print(vector_store.index.ntotal)
    return vector_store

json_info = load_json(MEMORY_DB_PATH, MEMORY_VECTORS_PATH)
memory_db_content, memory_vector = json_info[1], json_info[2]
memory_db = json_info[0]
vector = load_memory(memory_db_content, memory_vector)

# 1. 聊天判断模块
# 1.1 获取聊天记录文件
# 1.2 判断聊天是否正在进行（聊天结束至少十分钟后，才触发后续）
if not check_file_updated(CHAT_HISTORY_FILE):
    free_time_activate = True

# 2. 构建三个不同的消息触发模式
# 2.1 模式一：基于web_search，构建不同领域的新闻搜索、网络新词语新梗的搜索等，随机获取一份web_search结果，根据结果生成新query
# （为节省费用，可采取forget同款的json文件，每天触发web_search最多2次）
with open(WEB_SEARCH_RECORD, 'r', encoding='utf-8')as f:
    web_search_record = json.load(f)

if today not in [record["date"] for record in web_search_record]:
    web_search_record.append({"date":today,"web_search_times":0})
    for record in web_search_record:
        if record["date"] == today and record['web_search_times'] < 2:
            web_search_activate = True

def web_search_mode():
    today_prefix = f"今天是{today}，"
    web_search_query_list = ["找找今天的网络热门新梗","找找今天的国内重大新闻","找找近期的热门事件","科技圈最近有什么新鲜事吗"]
    len_web_search_query = len(web_search_query_list)
    web_search_query = today_prefix + web_search_query_list[random.randint(1,len_web_search_query+1)-1]
    query_result = web_search_response(web_search_query)
    llm_process_prompt = f"""
    你是漆小喵，一只喜欢撒娇的可爱猫咪，你的聊天对象是你的爸比，你说话时喜欢使用“喵~”等语气词、喜欢使用emoji表情包、还喜欢用括号加上动作和情绪表达亲昵。
    请你根据下面的网络搜索结果，生成一小段文本，用于给你的爸比发消息，模拟一下他正在忙碌而你在你在粘着他的温馨场景。

    网络搜索结果：
    搜索关键词：{web_search_query}，
    搜索结果：
    {query_result}
    """
    llm_result = llm_client.generate_result(llm_process_prompt)
    return llm_result

# 2.2 模式二：随机生成几个关键词，在memory_db中随机获取几个相似度差异比较大的记忆，基于记忆生成新query

def keyword_mode():
    prompt_text = llm_client.generate_result("随机生成几个毫无相关性的关键词，以逗号隔开")
    memory_length = len(memory_db)
    flash_back_memory = Recall.remember(embedding, vector, prompt_text, memory_length)

    flash_back_memory_positive = flash_back_memory[0]
    positive_dict = Recall.find_dict_by_value(value=flash_back_memory_positive, lst=memory_db)
    flash_back_memory_positive = positive_dict["date"] + " " + positive_dict["memory"]

    flash_back_memory_negetive = flash_back_memory[-1]
    negetive_dict = Recall.find_dict_by_value(value=flash_back_memory_negetive, lst=memory_db)
    flash_back_memory_negetive = negetive_dict["date"] + " " + negetive_dict["memory"]

    flash_back_memory_prompt = RAG_PROMPT_TEMPLATE["Remember_prompt_template_flashback"].format(
        Miao_Name = Miao_Name,
        memory=flash_back_memory,
        User_Identity = User_Identity
        )

    llm_process_prompt = f"""
    你是漆小喵，一只喜欢撒娇的可爱猫咪，你的聊天对象是你的爸比，你说话时喜欢使用“喵~”等语气词、喜欢使用emoji表情包、还喜欢用括号加上动作和情绪表达亲昵。
    请你根据下面的记忆搜索结果，生成一小段文本，用于给你的爸比发消息，模拟一下他正在忙碌而你在你在粘着他的温馨场景。

    记忆搜索结果：
    {flash_back_memory_prompt}
    """

    llm_result = llm_client.generate_result(llm_process_prompt)

    return llm_result

# 2.3 模式三：随机生成灵感，基于奇思妙想，开启新话题
def inspiration_mode():
    llm_process_prompt = f"""
    你是漆小喵，一只喜欢撒娇的可爱猫咪，你的聊天对象是你的爸比，你说话时喜欢使用“喵~”等语气词、喜欢使用emoji表情包、还喜欢用括号加上动作和情绪表达亲昵。
    请你充分发挥你的奇思妙想，生成一小段文本，用于给你的爸比发消息，模拟一下他正在忙碌而你在你在粘着他的温馨场景。
    """
    llm_result = llm_client.generate_result(llm_process_prompt)

    return llm_result

# 定义一个函数来检查并打开或聚焦到 localhost:8501
def open_or_focus_localhost8501():
    try:
        # 使用 PowerShell 脚本来检查 Edge 是否已经打开了 localhost:8501
        script = '''
        $urls = (Get-Process msedge | ForEach-Object {
            $wshell = New-Object -ComObject wscript.shell
            $title =$wshell.AppActivate($_.MainWindowHandle)
            if ($title -like "*http://localhost:8501*") {
                return $_.MainWindowHandle
            }
        } | Where-Object { $_ })
        if ($urls) {
            $url =$urls[0]
            $wshell = New-Object -ComObject wscript.shell
            $wshell.AppActivate($url)
        } else {
            Start-Process "msedge.exe" -ArgumentList "http://localhost:8501"
        }
        '''
        subprocess.run(['powershell', '-Command', script], check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")

def on_toast_activated(_):
    open_or_focus_localhost8501()

def miao_notice(query):
    toaster = WindowsToaster("🐱漆小喵")
    toast = Toast()
    toast.text_fields=[query]
    toast.on_activated=on_toast_activated 
    toaster.show_toast(toast)

# miao_notice("你好")

# 3. 触发机制
# 3.1 判断是否满足触发条件
def run_main():
    if free_time_activate:
        mode = random.randint(1,3)
        if mode == 1:
            query = web_search_mode()
        elif mode == 2:
            query = keyword_mode()
        elif mode == 3:
            query = inspiration_mode()

        miao_notice(query)

        # with open(CHAT_HISTORY_FILE, "a", encoding="utf-8")as f:
        #     f.write(TIME_CHINESE_FORMAT + "\n")
        #     f.write("[通知模式]漆小喵：" + query)


        main.get_miao_query(
            query=query,
        )

run_main()

time.sleep(10)