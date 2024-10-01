import streamlit as st
from module.llm_client import get_client
from module.conversation import postprocess_text, preprocess_text, Conversation, Role
from module import PROMPT_TEMPLATE
from module.intent_recognition import Intent_Recognition,IR_result
from module.vector_base import VectorStore
from module.utils import ReadFiles
from module.embeddings import BgeEmbedding
from module.memory import Abstract, History_Management, Recall
from module.PROMPT_TEMPLATE import RAG_PROMPT_TEMPLATE
from datetime import datetime, timedelta
import json
import random
import base64
import re
from tqdm import tqdm
import os
import faiss
import numpy as np
import time

Abstractor = Abstract()
Recall = Recall()
History_Manager = History_Management()
embedding = BgeEmbedding()
today = datetime.now().strftime('%Y-%m-%d')

MEMORY_DB_PATH = "./memory_storage/VBstorage/Memory_DB.json"
MEMORY_VECTORS_PATH = "./memory_storage/VBstorage/Memory_Vectors.json"
TMP_PATH = f"./memory_storage/miao_memory/chat_memory/tmp/{today}_TMP_memory.json"
INDEX_PATH = './memory_storage/VBstorage'
CONFIG_PATH = './module/config.json'
CHAT_HISTORY_FILE = f'./memory_storage/miao_memory/chat_history/{today}_chat_history.txt'

with open(CONFIG_PATH, "r", encoding="utf-8")as f:
    config = json.load(f)

Miao_Name = config["Miao_Name"]
Miao_Nick_Name = config["Miao_Nick_Name"]
Miao_Info_Brief = config["Miao_Info_Brief"] 
User_Identity = config["User_Identity"]
eco_mode = config["ECO_MODE"]

if eco_mode:
    MODEL_LONG = MODEL_FLASH = "glm-4-flash"
else:
    MODEL_LONG = "glm-4-long"
    MODEL_FLASH = "glm-4-flash"

print(MODEL_FLASH)
print(MODEL_LONG)

client = get_client(MODEL_FLASH)

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
    print(vector_store.index.ntotal)
    return vector_store

def check_file_updated(file_path,diff_time):
    if not os.path.exists(file_path):
        return False
    current_time = time.time()
    modified_time = os.path.getmtime(file_path)
    return (current_time - modified_time) < diff_time

with st.spinner(f"嗷嗷，太阳晒屁股了，{Miao_Nick_Name}正在摸爬滚打地起床，脑壳困困的~"):
    json_info = load_json(MEMORY_DB_PATH, MEMORY_VECTORS_PATH)
    memory_db_content, memory_vector = json_info[1], json_info[2]
    memory_db = json_info[0]
    vector = load_memory(memory_db_content, memory_vector)
    print("vector_reloaded")

def main(
        ir_result: None,
        prompt_text: str,
        system_prompt: str,
        top_p: float = 0.8,
        temperature: float = 0.95,
        max_tokens: int = 1024,
        retry: bool = False,
):
    global vector, embedding, memory_db
    col1, col2 = st.columns([3, 2])
    with col1: 
        placeholder = st.empty()
        with placeholder.container():
            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []

        if prompt_text == "" and retry == False:
            print("\n== Clean ==\n")
            st.session_state.chat_history = []
            with st.spinner("嗷嗷，太阳晒屁股了，喵喵正在摸爬滚打地起床，脑壳困困的~"):
                json_info = load_json(MEMORY_DB_PATH, MEMORY_VECTORS_PATH)
                memory_db_content, memory_vector = json_info[1], json_info[2]
                memory_db = json_info[0]
                vector = load_memory(memory_db_content, memory_vector)
                print("vector_reloaded")
            return

        history: list[Conversation] = st.session_state.chat_history
        for conversation in history:
            conversation.show()

        if retry:
            print("\n== Retry ==\n")
            last_user_conversation_idx = None
            for idx, conversation in enumerate(history):
                if conversation.role == Role.USER:
                    last_user_conversation_idx = idx
            if last_user_conversation_idx is not None:
                prompt_text = history[last_user_conversation_idx].content
                del history[last_user_conversation_idx:]

        first_round = len(st.session_state.chat_history) == 0
        
        if check_file_updated(TMP_PATH, 1800):
            json_info = load_json(MEMORY_DB_PATH, MEMORY_VECTORS_PATH)
            memory_db_content, memory_vector = json_info[1], json_info[2]
            memory_db = json_info[0]
            vector = load_memory(memory_db_content, memory_vector)
            print("vector_reloaded")

        if prompt_text:
            if "清空聊天记录" in prompt_text:
                json_info = load_json(MEMORY_DB_PATH, MEMORY_VECTORS_PATH)
                memory_db_content, memory_vector = json_info[1], json_info[2]
                memory_db = json_info[0]
                vector = load_memory(memory_db_content, memory_vector)
                print("vector_reloaded")
                pass
            else:
                system_prompt = system_prompt
                prompt_text = prompt_text.strip()
                now = datetime.now()
                yesterday = now - timedelta(days=1)
                time_chinese_format = "{0}年{1}月{2}日 {3}时{4}分{5}秒".format(
            now.year,now.month,now.day,now.hour,now.minute,now.second
        )       
                time_prompt = "<time>现在是{time}<time/>".format(time = time_chinese_format)
                
                History_Manager.append_conversation(
                    Conversation(Role.USER, prompt_text), 
                    history,
                    save_and_show=True)
                
                history[-1].content = time_prompt + history[-1].content
                with st.spinner("别着急哦，漆小喵的脑瓜子已经转起来啦~"):
                    if isinstance(ir_result, dict):
                        if ir_result["mode"] == "memory" and ir_result["type"] == "date_memory":
                            prompt_text = prompt_text.replace("今天", "{0}年{1}月{2}日".format(now.year, now.month, now.day))
                            prompt_text = prompt_text.replace("刚刚", "{0}年{1}月{2}日".format(now.year, now.month, now.day))
                            prompt_text = prompt_text.replace("刚才", "{0}年{1}月{2}日".format(now.year, now.month, now.day))
                            prompt_text = prompt_text.replace("昨天", "{0}年{1}月{2}日".format(yesterday.year, yesterday.month, yesterday.day))
                            date = Recall.date_info_extraction(prompt_text)
                            print(date)
                            if Recall.date_memory_detection(date) == True:
                                print(Recall.date_memory_call_back(date))
                                if Recall.date_memory_call_back(date):
                                    history[-1].content = Recall.date_memory_call_back(date) + history[-1].content
                                with st.expander("爸比，这是喵喵脑瓜子里翻到的日记哦~",expanded=False,icon="🗒️"):
                                    if Recall.date_memory_call_back(date):
                                        date_memory = Recall.date_memory_call_back(date).replace("<memory_begin>\n请注意，这是漆小喵脑瓜子中关于过去的记忆片段，可能与本次对话有关联。\n如果这段记忆有关联，请结合记忆回答，如果没有关联，请使用自身能力回答。\n以下是可能关联的记忆片段：","")
                                        date_memory = date_memory.replace("<memory_end>\n爸比的提问：","")
                                        st.write(date_memory)
                                    
                        elif ir_result["mode"] == "memory" and ir_result["type"] == "keyword_memory":
                            query_result_list = Recall.self_query(query=prompt_text, embedding=embedding,vector=vector)
                            query_result_list_expand = [item for query_result in query_result_list for item in query_result]

                            core_query_result_list = []
                            for query_result in query_result_list_expand:
                                item_dict = Recall.find_dict_by_value(value=query_result[0], lst=memory_db)
                                basic_result = item_dict["date"] + " " + item_dict["memory"]
                                core_query_result_list.append((basic_result,query_result[1]))

                            relate_query_result_list = []
                            for query_result in query_result_list_expand:
                                item_dict = Recall.find_dict_by_value(value=query_result[0], lst=memory_db)
                                if item_dict.get("attribute") and isinstance(item_dict["attribute"], dict):
                                    if item_dict["attribute"].get("relate_memory"):
                                        for relate_memory in item_dict["attribute"]["relate_memory"]:
                                            relate_query_result_list.append(relate_memory)

                            relate_memory_filtered = []
                            for memory in relate_query_result_list:
                                print(memory)
                                if memory not in relate_memory_filtered and memory not in [memory_content[0] for memory_content in core_query_result_list]:
                                    relate_memory_filtered.append(memory)

                            relate_memory_with_vector = []
                            for memory in relate_memory_filtered:
                                # print(memory)
                                embedding_result = embedding.get_embedding(memory)
                                relate_memory_with_vector.append((memory,embedding_result))

                            final_result_list = core_query_result_list + relate_memory_with_vector

                            final_result_filter = []
                            for i in final_result_list:
                                if i[0] not in [item[0] for item in final_result_filter]:
                                    final_result_filter.append(i)
                            
                            final_result_list = final_result_filter

                            prompt_text_embedding = embedding.get_embedding(prompt_text)
                            similarity_filter = []

                            for memory in final_result_list:
                                similarity = vector.get_similarity(prompt_text_embedding, memory[1])
                                if similarity >= 0.65:
                                    print(memory[0])
                                    print(similarity)
                                    similarity_filter.append((memory[0], similarity)) 

                            similarity_filter.sort(key=lambda x: x[1], reverse=True)

                            top_six_memories = [memory for memory, similarity in similarity_filter[:6]]

                            print(top_six_memories)

                            for memory in top_six_memories:
                                History_Manager.save_memory_record(memory)

                            keyword_memory = "\n".join(top_six_memories)
                            keyword_memory_prompt = RAG_PROMPT_TEMPLATE["Remember_prompt_template_default"].format(
                                Miao_Name=Miao_Name,
                                memory=keyword_memory,
                                User_Identity=User_Identity)


                            history[-1].content = keyword_memory_prompt + history[-1].content
                            with st.expander(f"{User_Identity}，{Miao_Nick_Name}脑瓜子里想起了一些记忆碎片耶~",expanded=False,icon="🍼"):
                                keyword_memory_prompt = keyword_memory_prompt.replace("<memory_begin>\n请注意，这是漆小喵脑瓜子中关于过去的记忆片段，可能与本次对话有关联。\n如果这段记忆有关联，请结合记忆回答，如果没有关联，请使用自身能力回答。\n以下是可能关联的记忆片段：","")
                                keyword_memory_prompt = keyword_memory_prompt.replace("<memory_end>\n爸比的提问：","")
                                st.write(keyword_memory_prompt)
                                                
                        elif ir_result["mode"] == "chat":
                            if Recall.flash_back_triger(prompt_text) == 1:
                                print("记忆触发！触发类型：记忆闪回！")
                                memory_length = len(memory_db)
                                flash_back_memory = Recall.remember(embedding, vector, prompt_text, memory_length)
                                print(len(flash_back_memory))
                                print(flash_back_memory[0])
                                print(flash_back_memory[-1])
                                flash_back_memory_positive = flash_back_memory[0]
                                
                                positive_dict = Recall.find_dict_by_value(value=flash_back_memory_positive, lst=memory_db)
                                flash_back_memory_positive = positive_dict["date"] + " " + positive_dict["memory"]
                                print("positive_memory")
                                print(flash_back_memory_positive)

                                History_Manager.save_memory_record(flash_back_memory_positive)

                                flash_back_memory_negetive = flash_back_memory[-1]
                                negetive_dict = Recall.find_dict_by_value(value=flash_back_memory_negetive, lst=memory_db)
                                flash_back_memory_negetive = negetive_dict["date"] + " " + negetive_dict["memory"]
                                print("negetive_memory")
                                print(flash_back_memory_negetive)

                                History_Manager.save_memory_record(flash_back_memory_negetive)

                                flash_back_memory = flash_back_memory_positive + "\n" + flash_back_memory_negetive
                                flash_back_memory_prompt = RAG_PROMPT_TEMPLATE["Remember_prompt_template_flashback"].format(
                                    Miao_Name = Miao_Name,
                                    memory=flash_back_memory,
                                    User_Identity = User_Identity
                                    )

                                print(f"Memory retrieved: {flash_back_memory}")  # 检查记忆是否被成功检索

                                if flash_back_memory:  # 确保返回的记忆不是空的
                                    history[-1].content = flash_back_memory_prompt + history[-1].content
                                    with st.expander("爸比，喵喵脑瓜子里闪回了一些记忆片段呢~", expanded=False, icon="🍦"):
                                        flash_back_memory_prompt = flash_back_memory_prompt.replace("<memory_begin>\n请注意，漆小喵脑瓜子中突然想起了一些过去的记忆片段。\n请结合闪回的记忆片段，延续或转移当前话题，这是你的奇思妙想时刻。\n以下是脑瓜子里闪回的记忆片段：", "")
                                        flash_back_memory_prompt = flash_back_memory_prompt.replace("<memory_end>\n爸比的提问：", "")
                                        st.write(flash_back_memory_prompt)
                                else:
                                    print("No memory was retrieved for the flashback trigger.")
                            
                            else:
                                default_memory = Recall.remember_with_vector(embedding, vector, prompt_text)
                                prompt_text_embedding = embedding.get_embedding(prompt_text)
                                similarity_filter = []
                                for memory in default_memory:
                                    memory_embedding = memory[1]
                                    similarity = vector.get_similarity(prompt_text_embedding,memory_embedding)
                                    if similarity >= 0.65:
                                        print(memory[0])
                                        print(similarity)
                                        similarity_filter.append(memory[0])
                                
                                final_default_memory = []
                                if similarity_filter:
                                    for memory in similarity_filter:
                                        memory_dict = Recall.find_dict_by_value(value=memory, lst=memory_db)
                                        memory_content = memory_dict["date"] + " " + memory_dict["memory"]
                                        final_default_memory.append(memory_content)
                                similarity_filter_clean = []
                                for i in final_default_memory:
                                    if i not in similarity_filter_clean:
                                        similarity_filter_clean.append(i)
                                
                                final_default_memory = similarity_filter_clean

                                for memory in final_default_memory:
                                    History_Manager.save_memory_record(memory)

                                if final_default_memory:
                                    final_default_memory = "\n".join(final_default_memory)
                                    default_memory_prompt = RAG_PROMPT_TEMPLATE["Remember_prompt_template_default"].format(
                                        Miao_Name=Miao_Name,
                                        memory=final_default_memory,
                                        User_Identity=User_Identity)

                                    history[-1].content = default_memory_prompt + history[-1].content

                    else:
                        print(f"ir_result is not a dictionary: {ir_result}")

                    placeholder = st.empty()
                    message_placeholder = placeholder.chat_message(name="assistant", avatar="😺")
                    markdown_placeholder = message_placeholder.empty()
                    output_text = ''
                    for response in client.generate_stream(
                            system_prompt,
                            tools=None,
                            history=history,
                            # max_tokens=max_tokens,
                            # temperature=temperature,
                            # top_p=top_p,
                    ):

                        token = response.token
                        if response.token.special:
                            print("\n==Output:==\n", output_text)
                            match token.text.strip():
                                case '<|user|>':
                                    break
                                case _:
                                    st.error(f'Unexpected special token: {token.text.strip()}')
                                    break
                        output_text += response.token.text
                        markdown_placeholder.markdown(postprocess_text(output_text + '▌'))

                    History_Manager.append_conversation(
                        Conversation(Role.ASSISTANT, postprocess_text(output_text),), 
                        history, 
                        markdown_placeholder,
                        save_and_show=True,
                        document_mode=False,
                        fc_mode=False)

                    print("history_length")
                    print(History_Manager.count_history(history=history))

        if History_Manager.count_history(history) > 40000:
            history_str = History_Manager.return_str_history(history)
            prompt = RAG_PROMPT_TEMPLATE["history_abstract_prompt_template"].format(
            Miao_Info_Brief=Miao_Info_Brief,
            User_Identity=User_Identity,
            Miao_Name=Miao_Name,
            text=history_str)

            abstract = Abstractor.default_abstract(model_name=MODEL_LONG,prompt=prompt)
            last_three_memory = History_Manager.last_history(history)
            history = []
            History_Manager.append_conversation(
                Conversation(Role.ASSISTANT,abstract),history,save_and_show=True)
            History_Manager.append_conversation(
                Conversation(Role.USER,last_three_memory[-6]),history,save_and_show=True)
            History_Manager.append_conversation(
                Conversation(Role.ASSISTANT,last_three_memory[-5]),history,save_and_show=True)
            History_Manager.append_conversation(
                Conversation(Role.USER,last_three_memory[-4]),history,save_and_show=True)
            History_Manager.append_conversation(
                Conversation(Role.ASSISTANT,last_three_memory[-3]),history,save_and_show=True)
            History_Manager.append_conversation(
                Conversation(Role.USER,last_three_memory[-2]),history,save_and_show=True)
            History_Manager.append_conversation(
                Conversation(Role.ASSISTANT,last_three_memory[-1]),history,save_and_show=True)
            st.session_state.chat_history = history
                    # print(history)
                

