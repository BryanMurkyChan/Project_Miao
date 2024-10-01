from dataclasses import dataclass
from enum import auto, Enum
import json

from PIL.Image import Image
import streamlit as st
from streamlit.delta_generator import DeltaGenerator

FILE_TEMPLATE = "[File Name]\n{file_name}\n[File Content]\n{file_content}"

class Role(Enum):
    SYSTEM = auto()
    USER = auto()
    ASSISTANT = auto()
    TOOL = auto()

    def __str__(self):
        match self:
            case Role.SYSTEM:
                return "<|system|>"
            case Role.USER:
                return "<|user|>"
            case Role.ASSISTANT | Role.TOOL:
                return "<|assistant|>"
            
    # Get the message block for the given role
    def get_message(self):
        match self.value:
            case Role.SYSTEM.value:
                return
            case Role.USER.value:
                return st.chat_message(name="user", avatar="🧑‍💻")
            case Role.ASSISTANT.value:
                return st.chat_message(name="assistant", avatar="😺")
            case Role.TOOL.value:
                return st.chat_message(name="tool", avatar="🔧")
            case _:
                st.error(f'Unexpected role: {self}')

@dataclass
class Conversation:
    role: Role
    content: str
    tool: str | None = None
    image: Image | None = None


    def __str__(self) -> str:
        print(self.role, self.content, self.tool)
        match self.role:
            case Role.SYSTEM | Role.USER | Role.ASSISTANT:
                return f'{self.role}\n{self.content}'
            case Role.TOOL:
                return f'{self.role}{self.tool}\n{self.content}'
    

    # Human readable format
    def get_text(self) -> str:
        text = postprocess_text(self.content)
        match self.role.value:
            case Role.TOOL.value:
                text = f'喵喵调用的工具是 `{self.tool}` \n\n{text}'
        return text
    
    # Display as a markdown block
    def show(self, placeholder: DeltaGenerator | None=None) -> str:
        if placeholder:
            message = placeholder
        else:
            message = self.role.get_message()
        if self.image:
            message.image(self.image)
        else:
            text = self.get_text()
            message.markdown(text)



def preprocess_text(
    system: str | None,
    tools: list[dict] | None,
    history: list[Conversation],
) -> str:
    if tools:
        tools = json.dumps(tools, indent=4, ensure_ascii=False)

    prompt = f"{Role.SYSTEM}\n"
    prompt += system
    if tools:
        tools = json.loads(tools)
        prompt += json.dumps(tools, ensure_ascii=False)
    for conversation in history:
        prompt += f'{conversation}'
    prompt += f'{Role.ASSISTANT}\n'
    return prompt

def postprocess_text(text: str) -> str:

    start_index = text.find("<translation_info>")
    end_index = text.find("<translation_info/>")
    if start_index != -1 and end_index != -1:
        text = text[:start_index] + text[end_index + len("<translation_info/>"):]

    start_index = text.find("<文章开始>")
    end_index = text.find("<文章结束>")
    if start_index != -1 and end_index != -1:
        text = text[:start_index] + text[end_index + len("<文章结束>"):]

    start_index = text.find("<memory_begin>")
    end_index = text.find("爸比的提问：")
    if start_index != -1 and end_index != -1:
        text = text[:start_index] + text[end_index + len("爸比的提问："):]
    
    start_index = text.find("<document_details>")
    end_index = text.find("关于这个文档的问题：")
    if start_index != -1 and end_index != -1:
        text = text[:start_index] + text[end_index + len("关于这个文档的问题："):]

    start_index = text.find("<web_info>")
    end_index = text.find("关于这个网页信息的问题：")
    if start_index != -1 and end_index != -1:
        text = text[:start_index] + text[end_index + len("关于这个网页信息的问题："):]

    start_index = text.find("<time>")
    end_index = text.find("<time/>")
    
    if start_index != -1 and end_index != -1:
        text = text[:start_index] + text[end_index + len("<time/>"):]

    text = text.replace("\n\n请告诉用户，工具调用失败了，可能与问题需求或工具限制有关","")
    text = text.replace("~", "~ ")
    text = text.replace("\(", "$")
    text = text.replace("\)", "$")
    text = text.replace("\[", "$$")
    text = text.replace("\]", "$$")
    text = text.replace("<|assistant|>", "")
    text = text.replace("<|system|>", "")
    text = text.replace("<|user|>", "")
    return text.strip()