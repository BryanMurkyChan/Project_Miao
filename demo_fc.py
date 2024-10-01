import streamlit as st
from streamlit.delta_generator import DeltaGenerator
from module.llm_client import get_client
from module.conversation import postprocess_text, preprocess_text, Conversation, Role
from module.intent_recognition import Intent_Recognition, IR_result
from module.vector_base import VectorStore
from module.utils import ReadFiles
from module.embeddings import BgeEmbedding
from module.memory import History_Management
from module.tools import parse_function_call, web_search_response, translation
from module import PROMPT_TEMPLATE
import json
from datetime import datetime, timedelta
import random
import base64
import re
from tqdm import tqdm
import os
import faiss
import numpy as np
from zhipuai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall


CONFIG_PATH = './module/config.json'

with open(CONFIG_PATH, "r", encoding="utf-8")as f:
    config = json.load(f)

Miao_Nick_Name = config["Miao_Nick_Name"]
User_Identity = config["User_Identity"]
eco_mode = config["ECO_MODE"]

if eco_mode:
    MODEL_PLUS = MODEL_FLASH = "glm-4-flash"
    web_search_activate = False
else:
    MODEL_PLUS = "glm-4-long"
    MODEL_FLASH = "glm-4-flash"
    web_search_activate = True


plus_client = get_client(MODEL_PLUS)
flash_client = get_client(MODEL_FLASH)
History_Manager = History_Management()
today = datetime.now().strftime('%Y-%m-%d')

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_web_search_result",
            "description": "根据用户的输入，进行网络搜索，如果用户提供了链接，则直接使用传递用户的链接",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "description": "用户提问",
                        "type": "string"
                    }
                },
                "required": [ "query" ]
            },
        }
    }
]

tool_prompt = PROMPT_TEMPLATE.TOOL_PROMPT["TOOL_PROMPT"]

def main(
        ir_result: None,
        prompt_text: str,
        system_prompt: str,
        top_p: float = 0.8,
        temperature: float = 0.95,
        max_tokens: int = 1024,
        retry: bool = False,
        # clear_history: bool = False,
):
    col1, col2 = st.columns([3, 2])
    with col1: 
        placeholder = st.empty()
        with placeholder.container():
            if 'chat_history' not in st.session_state:
                st.session_state.chat_history = []
            if 'agent_history' not in st.session_state:
                st.session_state.agent_history = []

        if prompt_text == "" and retry == False:
            print("\n== Clean ==\n")
            st.session_state.chat_history = []
            st.session_state.agent_history = []
            return

        history: list[Conversation] = st.session_state.chat_history
        agent_history : list[Conversation] = st.session_state.agent_history
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

        if prompt_text:
            prompt_text = prompt_text.strip()
            now = datetime.now()
            time_chinese_format = "{0}年{1}月{2}日 {3}时{4}分{5}秒".format(
        now.year,now.month,now.day,now.hour,now.minute,now.second
    )       
            time_prompt = "<time>现在是{time}<time/>".format(time = time_chinese_format)

            History_Manager.append_conversation(conversation=Conversation(Role.USER, prompt_text), history=agent_history, save_and_show=True, fc_mode=True)
            History_Manager.append_conversation(conversation=Conversation(Role.USER, prompt_text), history=history,save_and_show=False)

            print(agent_history)

            history[-1].content = time_prompt + history[-1].content
            agent_history[-1].content = time_prompt + agent_history[-1].content
            
            with st.spinner(f"{Miao_Nick_Name}正在启动工具调用能力！{User_Identity}的需求很快就能解决啦~"):
                if isinstance(ir_result, dict):
                    if ir_result["type"] == "web_search_agent" and web_search_activate:
                    # with st.spinner("耶嘿，{Miao_Nick_Name}正在启动网页搜索功能！"):
                        print("已启动网页搜索")
                        prompt_text = prompt_text.replace("今天", "{0}年{1}月{2}日".format(now.year, now.month, now.day))
                        web_info = web_search_response(prompt_text).strip() 
                        agent_history[-1].content = web_info + agent_history[-1].content
                        history[-1].content = web_info + history[-1].content
                        with st.expander(f"{User_Identity}，{Miao_Nick_Name}在网上找到了这个",expanded=False,icon="📨"):
                            web_info = web_info.replace("<web_info>\n已触发联网搜索，以下是联网搜索返回结果，如果与用户提问有关，请结合搜索答案回答，如果无关，请使用模型自身能力回答，不要杜撰任何内容。","")
                            web_info = web_info.replace("根据搜索答案进行的回答，需要附上网页相关链接。","")
                            web_info = web_info.replace("<web_info/>\n关于这个网页信息的问题：","")
                            st.write(web_info)

                        placeholder = st.empty()
                        message_placeholder = placeholder.chat_message(name="assistant", avatar="🔧")
                        markdown_placeholder = message_placeholder.empty()

                        output_text = ''
                        for response in flash_client.generate_stream(
                            system_prompt,
                            tools=None,
                            history=agent_history,
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
                    
                        History_Manager.append_conversation(Conversation(
                            Role.TOOL,
                            postprocess_text(output_text), 
                            tool="web_search_agent",
                        ), 
                        history=agent_history, 
                        placeholder=markdown_placeholder,
                        save_and_show=True,
                        fc_mode=True)
                        
                        History_Manager.append_conversation(Conversation(
                            Role.TOOL,
                            postprocess_text(output_text),
                            tool="web_search_agent"
                        ), history=history, save_and_show=False)

                    elif ir_result["type"] == "translation_agent":
                        translation_result = translation(prompt_text)
                        prompt_text = PROMPT_TEMPLATE.OTHER_PROMPT_TEMPLATE['Translation_Prompt'].format(query=prompt_text,result=translation_result)


                        placeholder = st.empty()
                        message_placeholder = placeholder.chat_message(name="assistant", avatar="🔧")
                        markdown_placeholder = message_placeholder.empty()

                        output_text = ''
                        for response in flash_client.generate_stream(
                            system_prompt,
                            tools=None,
                            history=agent_history,
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
                    
                        History_Manager.append_conversation(Conversation(
                            Role.TOOL,
                            postprocess_text(output_text), 
                            tool="translation_agent",
                        ), 
                        history=agent_history, 
                        placeholder=markdown_placeholder,
                        save_and_show=True,
                        fc_mode=True)
                        
                        History_Manager.append_conversation(Conversation(
                            Role.TOOL,
                            postprocess_text(output_text),
                            tool="translation_agent"
                        ), history=history, save_and_show=False)

                    else:
                        placeholder = st.empty()
                        message_placeholder = placeholder.chat_message(name="assistant", avatar="🔧")
                        markdown_placeholder = message_placeholder.empty()
                        for tool_response in plus_client.generate_stream(
                                system=tool_prompt,
                                tools=tools,
                                history=agent_history,
                                max_tokens=max_tokens,
                                temperature=temperature,
                                top_p=top_p,
                        ):
                            print(tools)
                            print(tool_response)
                            if isinstance(tool_response, ChoiceDeltaToolCall):
                                output = tool_response
                                print(output)
                                function_call_result = parse_function_call(tool_response)
                                print(function_call_result)

                                History_Manager.append_conversation(conversation=Conversation(Role.TOOL, function_call_result[0], tool=function_call_result[1]), history=agent_history, save_and_show=False)

                                print(agent_history)

                                output_text = ''
                                for response in flash_client.generate_stream(
                                system=tool_prompt,
                                tools=tools,
                                history=agent_history,
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
                            else:
                                fail_text += "\n\n请告诉用户，工具调用失败了，可能与问题需求或工具限制有关"
                                agent_history[-1].content = agent_history[-1].content + fail_text

                                output_text = ''
                                for response in flash_client.generate_stream(
                                system=tool_prompt,
                                tools=tools,
                                history=agent_history,
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

                            History_Manager.append_conversation(conversation=Conversation(
                                Role.TOOL,
                                postprocess_text(output_text), 
                                tool=function_call_result[1],
                            ), 
                            history=agent_history, 
                            placeholder=markdown_placeholder,
                            save_and_show=True,
                            fc_mode=True)
                            
                            History_Manager.append_conversation(conversation=Conversation(
                                Role.TOOL,
                                postprocess_text(output_text),
                                tool=function_call_result[1]
                            ), history=history, save_and_show=False)

                        print(History_Manager.count_history(history=history))
                        # print(history)
                        print(agent_history)
                        print(agent_history[-1].role)
                        print(agent_history[-1].tool)
            print(History_Manager.count_history(history=history))
            print(History_Manager.count_history(history=agent_history))