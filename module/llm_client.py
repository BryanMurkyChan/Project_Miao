from __future__ import annotations

import os
import streamlit as st
import torch
from collections.abc import Iterable
from typing import Any, Protocol
from huggingface_hub.inference._text_generation import TextGenerationStreamResponse, Token
from transformers import AutoModel, AutoTokenizer, AutoConfig
from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.utils import LogitsProcessorList
from module.conversation import Conversation
from zhipuai import ZhipuAI
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM
import datetime
import json

# 创建ZhipuAI客户端实例
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
with open(CONFIG_PATH, "r", encoding="utf-8")as f:
    config = json.load(f)

API_KEY = config["OPENAI_API_KEY"]
ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
# model_name = "glm-4-flash"

@st.cache_resource
def get_client(model_name) -> Client:
    client = HFClient(api_key=API_KEY, model_name=model_name)
    return client

class Client(Protocol):
    def generate_stream(self,
                        system: str | None,
                        tools: list[dict] | None,
                        history: list[Conversation],
                        **parameters: Any
                        ) -> Iterable[TextGenerationStreamResponse]:
        ...


class HFClient(Client):
    def __init__(self, api_key: str, model_name: str):
        # self.client = ZhipuAI(api_key=api_key)
        self.client = OpenAI(api_key=API_KEY, base_url=ZHIPU_BASE_URL)
        self.model_name = model_name

    def _prepare_messages(self, system: str | None, history: list[Conversation], query: str):
        messages = [
            {"role": "system", "content": system if system else ""},
        ]

        for conversation in history:
            messages.append({
                'role': str(conversation.role).removeprefix('<|').removesuffix('|>'),
                'content': conversation.content,
            })

        messages.append({"role": "user", "content": query})

        return messages

    def generate_result(
            self,
            query: str
    )-> str:
        messages = self._prepare_messages(system=None,history=[],query=query)
        # [{"role": "user", "content": query}]

        response = self.client.chat.completions.create(
            model=self.model_name, 
            messages=messages,
            stream=False,
        )
        result = response.choices[0].message.content
        return result

    def generate_stream(
            self,
            system: str | None,
            tools: list[dict] | None,
            history: list[Conversation],
            max_tokens: int = 2048,
            temperature: float = 0.7,
            top_p: float = 0.9,
            **parameters: Any
    ) -> Iterable[TextGenerationStreamResponse]:
        # messages = [
        #     {"role": "system", "content": system if system else ""},
        # ]

        # for conversation in history:
        #     messages.append({
        #         'role': str(conversation.role).removeprefix('<|').removesuffix('|>'),
        #         'content': conversation.content,
        #     })

        # query = history[-1].content
        # messages.append({"role": "user", "content": query})

        messages = self._prepare_messages(system=system,history=history,query=history[-1].content)

        response = self.client.chat.completions.create(
            model=self.model_name, 
            messages=messages,
            stream=True,
            tools=tools,
            # max_tokens=max_tokens,
            # temperature=temperature,
            # top_p=top_p,
        )

        for chunk in response:
            if chunk.choices:
                delta_content = chunk.choices[0].delta.content
                if chunk.choices[0].delta.tool_calls:
                    yield chunk.choices[0].delta.tool_calls[0]
                if delta_content:
                    yield TextGenerationStreamResponse(
                        generated_text=delta_content,
                        token=Token(
                            id=0,
                            logprob=0,
                            text=delta_content,
                            special=False,
                        )
                    )

