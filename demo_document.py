import streamlit as st
from streamlit.delta_generator import DeltaGenerator
from module.llm_client import get_client
from module.conversation import postprocess_text, Conversation, Role, FILE_TEMPLATE
from module.utils import extract_pdf, extract_docx, extract_pptx, extract_text
from module.vector_base import VectorStore
from module.utils import ReadFiles
from module.memory import Abstract, History_Management
from module.PROMPT_TEMPLATE import RAG_PROMPT_TEMPLATE, OTHER_PROMPT_TEMPLATE
from module.tools import web_search_response
import json
from datetime import datetime, timedelta
import random
import base64
import os
import shutil
import tempfile
from uuid import uuid4

# 获取相关组件

today = datetime.now().strftime('%Y-%m-%d')
now = datetime.now()
yesterday = now - timedelta(days=1)


CHAT_HISTORY_FILE = f'./memory_storage/miao_memory/chat_history/{today}_chat_history.txt'
WEB_SEARCH_WORDS = "./module/web_search_words.txt"
TMP_FOLDER_PATH_DOC = "./memory_storage/rag_memory/tmp"
TIME_CHINESE_FORMAT = "{0}年{1}月{2}日 {3}时{4}分{5}秒".format(
        now.year,now.month,now.day,now.hour,now.minute,now.second
    )       

CONFIG_PATH = './module/config.json'

f = open(WEB_SEARCH_WORDS,"r",encoding="utf-8")
web_search_words = f.read().split("\n")

with open(CONFIG_PATH, "r", encoding="utf-8")as f:
    config = json.load(f)

Miao_Name = config["Miao_Name"]
Miao_Nick_Name = config["Miao_Nick_Name"]
Miao_Info_Brief = config["Miao_Info_Brief"]
User_Identity = config["User_Identity"]
eco_mode = config["ECO_MODE"]

if eco_mode:
    MODEL_LONG = "glm-4-flash"
else:
    MODEL_LONG = "glm-4-long"

client = get_client(MODEL_LONG)
Abstractor = Abstract()
History_Manager = History_Management()

def tmp_folder_clean():
    for item in os.listdir(TMP_FOLDER_PATH_DOC):
        item_path = os.path.join(TMP_FOLDER_PATH_DOC, item)
        if os.path.isfile(item_path):
            os.remove(item_path)

def save_abstract_to_chat_history(abstract):
    with open(CHAT_HISTORY_FILE, "a", encoding="utf-8")as f:
        f.write(TIME_CHINESE_FORMAT + "\n")
        f.write("[文档模式]漆小喵：" + abstract + "\n")


def main(
        uploaded_files,
        prompt_text: str,
        system_prompt: str,
        top_p: float = 0.8,
        temperature: float = 0.95,
        repetition_penalty: float = 1.0,
        max_tokens: int = 4000,
        retry: bool = False,
        clear_history: bool = False,

):
    tmp_folder_clean()
    col1, col2 = st.columns([3, 2])
    with col1: 
        if "uploader_key" not in st.session_state:
            st.session_state.uploader_key = str(random.randint(1000, 100000000))
        
        if "files_uploaded" not in st.session_state:
            st.session_state.files_uploaded = False

        if "generate_abstract" not in st.session_state:
            st.session_state.generate_abstract = False

        if "session_id" not in st.session_state:
            st.session_state.session_id = uuid4()

        placeholder = st.empty()
        with placeholder.container():
            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []
            if 'document_history' not in st.session_state:
                st.session_state.document_history = []

        if prompt_text == "" and retry == False:
            print("\n== Clean ==\n")
            st.session_state.chat_history = []
            st.session_state.document_history = []
            return

        history: list[Conversation] = st.session_state.chat_history
        document_history = st.session_state.document_history


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

        if uploaded_files and not st.session_state.files_uploaded:
            st.session_state.generate_abstract = True

        if st.session_state.generate_abstract:
            with st.spinner(f"{Miao_Nick_Name}已接收到文档啦，{User_Identity}别着急，让{Miao_Nick_Name}好好读一读哦~"):
                uploaded_texts = []
                file_name_list = []
                for uploaded_file in uploaded_files:
                    file_name = uploaded_file.name
                    tmp_file_path = os.path.join(TMP_FOLDER_PATH_DOC,file_name)
                    if not os.path.exists(tmp_file_path):
                        with open(tmp_file_path,"wb")as f:
                            f.write(uploaded_file.getbuffer())
                        if file_name.endswith(".pdf"):
                            content = extract_pdf(tmp_file_path)
                        elif file_name.endswith(".docx"):
                            content = extract_docx(tmp_file_path)
                        elif file_name.endswith(".pptx"):
                            content = extract_pptx(tmp_file_path)
                        else:
                            content = extract_text(tmp_file_path)
                        uploaded_texts.append(
                            FILE_TEMPLATE.format(file_name=file_name, file_content=content)
                        )
                        file_name_list.append(tmp_file_path)
                
                st.session_state.uploaded_texts = "\n\n".join(uploaded_texts)
                uploaded_texts = st.session_state.get("uploaded_texts")
            
            uploaded_texts = "<文章开始>" + uploaded_texts + "<文章结束>" + "文章已上传！"

            History_Manager.append_conversation(Conversation(Role.USER, uploaded_texts), document_history,save_and_show=False)

            print(history)
            print(document_history)

            with st.spinner(f"{Miao_Nick_Name}正在进行文档摘要！马上就给{User_Identity}送上一份，嘻嘻~"):
                if len(uploaded_texts) <= 100000:
                    print("文档长度不足100000字，启动默认摘要模式")
                    prompt = RAG_PROMPT_TEMPLATE["RAG_abstract_prompt_template"].format(
                        Miao_Info_Brief = Miao_Info_Brief,
                        text = uploaded_texts,
                    )
                    file_abstract = Abstractor.default_abstract(model_name=MODEL_LONG, prompt=prompt)
                else:
                    print("文本长度超过100000字，启动map_reduce摘要模式")
                    prompt_template = RAG_PROMPT_TEMPLATE["RAG_abstract_prompt_template"].format(
                        Miao_Info_Brief = Miao_Info_Brief,
                        text = "{text}"
                    )
                    file_abstract = Abstractor.map_reduce_abstract(
                        prompt_template = prompt_template,
                        text = uploaded_texts)

                History_Manager.append_conversation(
                    conversation=Conversation(Role.ASSISTANT,postprocess_text(file_abstract),), 
                    history=document_history, 
                    save_and_show=False)

                History_Manager.append_conversation(
                    conversation=Conversation(Role.ASSISTANT,postprocess_text(file_abstract),),
                    history=history,
                    save_and_show=False)

                save_abstract_to_chat_history(file_abstract)

                st.session_state.files_uploaded = True
                st.session_state.uploaded_texts = ""
                st.session_state.generate_abstract = False

        for conversation in history:
            conversation.show()

        if prompt_text:
            prompt_text = prompt_text.strip()

            History_Manager.append_conversation(
                conversation=Conversation(Role.USER, prompt_text), 
                history=document_history, 
                save_and_show=True,
                document_mode=True)

            History_Manager.append_conversation(
                conversation=Conversation(Role.USER, prompt_text), 
                history=history,
                save_and_show=False)


            time_prompt = OTHER_PROMPT_TEMPLATE["Time_Prompt"].format(time = TIME_CHINESE_FORMAT)

            if any(word in prompt_text for word in web_search_words):
                with st.spinner(f"耶嘿，{Miao_Nick_Name}正在启动网页搜索功能！"):
                    print("已启动网页搜索")
                    prompt_text = prompt_text.replace("今天", "{0}年{1}月{2}日".format(now.year, now.month, now.day))
                    prompt_text = prompt_text.replace("昨天", "{0}年{1}月{2}日".format(yesterday.year, yesterday.month, yesterday.day))
                    web_info = web_search_response(prompt_text).strip() 
                    history[-1].content = web_info + time_prompt + history[-1].content
                    with st.expander(f"{User_Identity}，{Miao_Nick_Name}在网上找到了这个",expanded=False,icon="📨"):
                        web_info_show = web_info.replace("<web_info>\n已触发联网搜索，以下是联网搜索返回结果，如果与用户提问有关，请结合搜索答案回答，如果无关，请使用模型自身能力回答，不要杜撰任何内容。","")
                        web_info_show = web_info_show.replace("<web_info/>\n关于这个网页信息的问题：","")
                        st.write(web_info_show)
                        
            with st.spinner(f"别着急哦，{Miao_Name}的脑瓜子已经转起来啦~"):
                placeholder = st.empty()
                message_placeholder = placeholder.chat_message(name="assistant", avatar="😺")
                markdown_placeholder = message_placeholder.empty()
                output_text = ''
                for response in client.generate_stream(
                        system_prompt,
                        tools=None,
                        history=document_history,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
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

                # print(history)

                History_Manager.append_conversation(
                    conversation=Conversation(Role.ASSISTANT,postprocess_text(output_text),), 
                    history=document_history, 
                    placeholder=markdown_placeholder,
                    save_and_show=True,
                    document_mode=True)

                History_Manager.append_conversation(
                    conversation=Conversation(Role.ASSISTANT,postprocess_text(output_text),), 
                    history=history,
                    save_and_show=False)
                
                print(history)
                print("history_length:")
                print(History_Manager.count_history(history))
                print("document_history_length")
                print(History_Manager.count_history(document_history))
