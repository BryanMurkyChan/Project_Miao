import streamlit as st
import base64
import psutil
import signal
import json

CONFIG_PATH = './module/config.json'

with open(CONFIG_PATH, "r", encoding="utf-8")as f:
    config = json.load(f)

Miao_Name = config["Miao_Name"]
Miao_Nick_Name = config["Miao_Nick_Name"]
Miao_Info_Brief = config["Miao_Info_Brief"]
User_Identity = config["User_Identity"]
eco_mode = config["ECO_MODE"]

st.set_page_config(
    page_title=f"{Miao_Name}",
    page_icon="😻",
    layout='wide',
    initial_sidebar_state='collapsed',
)

def stop_streamlit():
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            # 检查进程名是否匹配
            if "streamlit" in proc.info['name']:
                proc.terminate()
                st.success(f"您的小可爱{Miao_Name}已下线，呼噜呼噜~")
                break
            # 如果cmdline存在，检查命令行是否匹配
            elif proc.info['cmdline'] and any("streamlit" in part for part in proc.info['cmdline']):
                proc.terminate()
                st.success(f"您的小可爱{Miao_Name}已下线，呼噜呼噜~")
                break
        except psutil.NoSuchProcess:
            st.error(f"坏了，{Miao_Nick_Name}不见了...")
        except psutil.AccessDenied:
            st.error(f"哎呀，{Miao_Nick_Name}的脑瓜子撬不开了")

def main_bg(main_bg):
    main_bg_ext = "jpg"
    st.markdown(
        f"""
         <style>
         .stApp {{
             background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
             background-size: cover
         }}
         </style>
         """,
        unsafe_allow_html=True
    )

# main_bg('./picture/background04.png')
main_bg('./picture/01.jpg')

hide_streamlit_style = """
<style>
    .st-emotion-cache-h4xjwg{
    height: 0rem;
    background-color: transparent;
    }
    .block-container {
        padding: 2rem 5rem 2rem;
    } /*顶部内边距(因为页面顶部有一个header元素，所以要给够间距值)、左侧&右侧内边距、底部内边距*/
    .st-emotion-cache-qdbtli{
        padding: 1rem 1rem 8px;
        background:transparent;
    }
    .st-emotion-cache-vj1c9o{
        padding:1rem 4rem 1rem ;
        bottom: 10px;
        width: 69%;
        min-width:10%;
        background-color: transparent;
    }
</style>
"""

# hide_streamlit_style = """
# <style>
#     .st-emotion-cache-h4xjwg{
#     height: 0rem;
#     background-color: transparent;
#     }
#     .block-container {
#         padding: 2rem 5rem 2rem;
#     } /*顶部内边距(因为页面顶部有一个header元素，所以要给够间距值)、左侧&右侧内边距、底部内边距*/
#     .st-emotion-cache-qdbtli{
#         width ：100%;
#         padding: 1rem 32rem 2rem 5rem;
    

#     }
#     .st-emotion-cache-vj1c9o{
#         padding:1rem 4rem 1rem ;
#         bottom: 10px;
#         width: 69%;
#         min-width:10%;
#         background-color: transparent;
#     }
#     .st-emotion-cache-12fmjuu {
#     position: fixed;
#     top: 0px;
#     left: 0px;
#     right: 0px;
#     height: 3.75rem;
#     background: transparent;
#     outline: none;
#     z-index: 999990;
#     display: block;
# }
#     .st-emotion-cache-uhkwx6 {
#         position: relative;
#         bottom: 0px;
#         width: 100%;
#         min-width: 100%;
#         background-color: transparent;
#         display: flex;
#         flex-direction: column;
#         -webkit-box-align: center;
#         z-index: auto;
# }
# </style>
# """

st.html(hide_streamlit_style)

import demo_chat, demo_document, demo_fc
from enum import Enum
from module import PROMPT_TEMPLATE
from module.PROMPT_TEMPLATE import RAG_PROMPT_TEMPLATE, OTHER_PROMPT_TEMPLATE
from module.intent_recognition import Intent_Recognition, IR_result, get_former_query
from module.memory import History_Management, Abstract
import random
from datetime import datetime
import os
import time
import json

History_Manager = History_Management()
Abstractor = Abstract()

today = datetime.now().strftime("%Y-%m-%d")
HISTORY_PATH = f"./memory_storage/miao_memory/chat_history/{today}_chat_history.txt"
free_time_activate = False


def check_file_updated(file_path):
    if not os.path.exists(file_path):
        return False
    current_time = time.time()
    modified_time = os.path.getmtime(file_path)
    return (current_time - modified_time) < 10

if eco_mode:
    MODEL_FLASHX = "glm-4-flash"
else:
    MODEL_FLASHX = "glm-4-flashx"

if os.path.exists(HISTORY_PATH):
    with open(HISTORY_PATH,"r",encoding="utf-8") as f:
        text = f.read()
else:
    text = ""

if 'system_prompt' not in st.session_state:
    with st.spinner(f"{Miao_Nick_Name}正在找刚刚聊天的内容呢！"):
        last_three_memory = History_Manager.get_last_three_history()
        default_abstract_prompt = RAG_PROMPT_TEMPLATE["history_abstract_prompt_template"].format(
            Miao_Info_Brief=Miao_Info_Brief,
            User_Identity = User_Identity,
            Miao_Name = Miao_Name,
            text=last_three_memory)
        history_abstract = OTHER_PROMPT_TEMPLATE["Chat_Hisory_Abstract_Prompt"].format(
            text=Abstractor.default_abstract(
                model_name=MODEL_FLASHX,
                prompt=default_abstract_prompt))
        st.session_state.system_prompt = PROMPT_TEMPLATE.get_system_prompt().strip() + history_abstract

class Mode(str, Enum):
    CHAT = "💬 Chat"
    DOC = "📃 Doc"
    AGENT = "🔧 AGENT"

with st.sidebar:
    if st.button(f"{Miao_Name}，睡觉！"):
        stop_streamlit()

    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = str(random.randint(1000, 100000000))

    uploaded_files = st.file_uploader(f"{User_Identity}，来这里上传文档吧~", type=["pdf", "txt", "py", "docx", "pptx", "json", "cpp", "md"], accept_multiple_files=True,key=st.session_state.uploader_key)

    if uploaded_files is None or not uploaded_files:
        tab = Mode.CHAT
    else:
        tab = Mode.DOC

    cols = st.columns(2)
    export_btn = cols[0]
    clear_history = cols[1].button("Clear History", use_container_width=True)
    retry = export_btn.button("Retry", use_container_width=True)
    
    top_p = st.slider(
        'top_p', 0.0, 1.0, 0.8, step=0.01
    )
    temperature = st.slider(
        'temperature', 0.0, 1.5, 0.95, step=0.01
    )

    max_new_token = st.slider(
        'Output length', 5, 32000, 256, step=1
    )

    system_prompt = st.text_area(
        label="System Prompt",
        height=300,
        value=st.session_state.system_prompt,
    )

# Set the title of the demo
st.title(f"😽{User_Identity}，{Miao_Nick_Name}想你了")
# Add your custom text here, with smaller font size
st.markdown(
    f"😻来跟{Miao_Nick_Name}聊天吧~",
    unsafe_allow_html=True)

# if not check_file_updated(HISTORY_PATH):
#     free_time_activate = True

# if free_time_activate:
#     # random_num = random.randint(0,1)
#     random_num = 0
#     if random_num == 1:
#         miao_query = miao_query.main()
#         free_time_activate = False

# def get_miao_query(query:None):
#     if query:
#         return query

# query = None

# miao_query = get_miao_query(query)

# if miao_query:
#     demo_chat.main(
#                 ir_result=None,
#                 miao_query=miao_query,
#                 retry=retry,
#                 top_p=top_p,
#                 temperature=temperature,
#                 prompt_text=None,
#                 system_prompt=st.session_state.system_prompt,
#                 repetition_penalty=repetition_penalty,
#                 max_tokens=max_new_token,
#             )

prompt_text = st.chat_input(
    'Chat with Miao!',
    key='chat_input',
)

if text:
    former_query = get_former_query(text)
else:
    former_query = None

if prompt_text:
    if former_query:
        ir_query = PROMPT_TEMPLATE.OTHER_PROMPT_TEMPLATE['Former_Query'].format(former_query=former_query, present_query=prompt_text)
    else:
        ir_query = prompt_text
else:
    ir_query = prompt_text

print(ir_query)
ir_process = Intent_Recognition(ir_query)
ir_result = IR_result(ir_process)
print(ir_result)

if prompt_text is not None:
    if isinstance(ir_result, dict):
        if ir_result["mode"] == "agent":
            tab = Mode.AGENT
    if "退出文档模式" in prompt_text:
        uploaded_files = []
        st.session_state.files_uploaded = False
        st.session_state.uploaded_texts = ""
        st.session_state.uploaded_file_nums = 0
        st.session_state.uploader_key = str(random.randint(1000, 100000000))
        tab = Mode.CHAT
    if "清空聊天记录" in prompt_text:
        uploaded_files = []
        st.session_state.files_uploaded = False
        st.session_state.uploaded_texts = ""
        st.session_state.uploaded_file_nums = 0
        st.session_state.uploader_key = str(random.randint(1000, 100000000))
        st.session_state.chat_history = []
        st.session_state.agent_history = []
        st.session_state.document_history = []
        st.session_state.history = []
        tab = Mode.CHAT

if clear_history or retry:
    prompt_text = ""
    st.session_state.clear()
    st.session_state.files_uploaded = False
    st.session_state.uploaded_texts = ""
    st.session_state.uploaded_file_nums = 0
    st.session_state.chat_history = []
    st.session_state.agent_history = []
    st.session_state.document_history = []
    st.session_state.history = []
    with st.spinner(f"{Miao_Nick_Name}正在找刚刚聊天的内容呢！"):
        last_three_memory = History_Manager.get_last_three_history()
        default_abstract_prompt = RAG_PROMPT_TEMPLATE["history_abstract_prompt_template"].format(
            Miao_Info_Brief = Miao_Info_Brief,
            User_Identity = User_Identity,
            Miao_Name = Miao_Name,
            text=last_three_memory)
        history_abstract = OTHER_PROMPT_TEMPLATE["Chat_Hisory_Abstract_Prompt"].format(
            text=Abstractor.default_abstract(
                model_name=MODEL_FLASHX,
                prompt=default_abstract_prompt))
        st.session_state.system_prompt = PROMPT_TEMPLATE.get_system_prompt().strip() + history_abstract



match tab:
    case Mode.CHAT:
        demo_chat.main(
                ir_result=ir_result,
                # miao_query=miao_query,
                retry=retry,
                top_p=top_p,
                temperature=temperature,
                prompt_text=prompt_text,
                system_prompt=st.session_state.system_prompt,
                max_tokens=max_new_token,
                
            )
    case Mode.DOC:
        demo_document.main(
                    uploaded_files=uploaded_files,
                    retry=retry,
                    clear_history=clear_history,
                    top_p=top_p,
                    temperature=temperature,
                    prompt_text=prompt_text,
                    system_prompt=system_prompt,
                    max_tokens=max_new_token,
                )
    case Mode.AGENT:
        demo_fc.main(
                ir_result=ir_result,
                retry=retry,
                top_p=top_p,
                temperature=temperature,
                prompt_text=prompt_text,
                system_prompt=st.session_state.system_prompt,
                max_tokens=max_new_token
            )
    case _:
        st.error(f'Unexpected tab: {tab}')
