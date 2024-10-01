import requests
import uuid
import json
from zhipuai import ZhipuAI
from module.PROMPT_TEMPLATE import OTHER_PROMPT_TEMPLATE
from module.llm_client import get_client
import os

flash_client = get_client("glm-4-flash")
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
with open(CONFIG_PATH, "r", encoding="utf-8")as f:
    config = json.load(f)

API_KEY = config["OPENAI_API_KEY"]
ZHIPU_TOOLS_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/tools"

def get_web_search_result(query):
    msg = [
        {
            "role": "user",
            "content":query,
        }
    ]
    tool = "web-search-pro"
    url = ZHIPU_TOOLS_BASE_URL
    request_id = str(uuid.uuid4())
    data = {
        "request_id": request_id,
        "tool": tool,
        "stream": False,
        "messages": msg
    }

    resp = requests.post(
        url,
        json=data,
        headers={'Authorization': API_KEY},
        timeout=300
    )
    output = resp.content.decode()
    output_json = json.loads(output)
    if "error" in output_json and output_json["error"]:
        return {"result": output_json}
    elif "choices" in output_json and output_json["choices"]:
        search_results = output_json["choices"][0]["message"]["tool_calls"][1]["search_result"]
        if len(search_results)>1:
            result_list = []
            for i in search_results:
                title = i.get("title")
                link = i.get("link")
                content = i.get("content", "")  
                if title and link:
                    result_dict = {"title": title, "link": link, "content": content}
                else:
                    result_dict = {"content": content}
                result_list.append(result_dict)

            search_results = json.dumps(result_list,ensure_ascii=False)
            web_search_prompt_pre_analysis = OTHER_PROMPT_TEMPLATE['Web_Search_Result_Analysis'].format(query=query,search_result=search_results)
            llm_process_result = flash_client.generate_result(web_search_prompt_pre_analysis)
            return {"result": llm_process_result}
        else:
            return {"result": json.dumps(output_json["choices"][0]["message"]["tool_calls"][1]["search_result"],ensure_ascii=False)}
    else:
        return {"result": "没有获取到搜索结果，可能权限不足或网络出故障了！"}

def web_search_response(query):
    web_search_info = get_web_search_result(query)
    web_search_prompt = OTHER_PROMPT_TEMPLATE["Web_Search_Prompt"].format(web_search_info = web_search_info)
    return web_search_prompt

def translation(query):
    translation_prompt = f"你是一名中英翻译专家，你将获得一份待翻译文本，请将其翻译为另一种语言。\n这是待翻译的文本：{query}\n这是你的翻译结果："
    llm_result = flash_client.generate_result(translation_prompt)
    return llm_result

def parse_function_call(tool_response):
    if tool_response:
        args = tool_response.function.arguments
        function_result = {}
        if tool_response.function.name == "get_web_search_result":
            function_result = get_web_search_result(**json.loads(args))
        if tool_response.function.name == "translation":
            function_result = translation(**json.loads(args))
        return json.dumps(function_result,ensure_ascii=False), tool_response.function.name


if __name__ == '__main__':
    print(get_web_search_result("看看这个https://mp.weixin.qq.com/s/sWNy-c5QT8NbzYDV6yK9Cg，都讲了啥"))