# -*- coding: utf-8 -*-

from datetime import datetime
from module.memory import Abstract
from module.llm_client import get_client
from module.PROMPT_TEMPLATE import RAG_PROMPT_TEMPLATE
from module.vector_base import VectorStore
from module.embeddings import BgeEmbedding
import numpy as np
import random
import jieba
import os
import json
import re
import faiss
import shutil

today = datetime.now().strftime('%Y-%m-%d')
# today = "2024-09-13"
now = datetime.now()
today_chinese_format = "{0}年{1}月{2}日".format(
        now.year,now.month,now.day
    )

# 定义路径
MIAO_DIARY_PATH = "./memory_storage/miao_memory/miao_diary/miao_diary.json"
HISTORY_PATH_TODAY = f"./memory_storage/miao_memory/chat_history/{today}_chat_history.txt"
HISTORY_FOLDER_PATH = f"./memory_storage/miao_memory/chat_history"
SYNC_PATH_TODAY = f"./memory_storage/miao_memory/chat_memory/tmp/{today}_SYNC.txt"
TMP_PATH_TODAY = f"./memory_storage/miao_memory/chat_memory/tmp/{today}_TMP_memory.json"
MEMORY_DB_PATH = "./memory_storage/VBstorage/Memory_DB.json"
MEMORY_VECTORS_PATH = "./memory_storage/VBstorage/Memory_Vectors.json"
INDEX_PATH = "./memory_storage/VBstorage"
INVOKE_RECORD_PATH = "./memory_storage/VBstorage/Invoke_Record.json"
INVOKE_RECORD_TXT_PATH = "./memory_storage/VBstorage/Invoke_Record.txt"
TMP_FOLDER_PATH = "./memory_storage/miao_memory/chat_memory/tmp"
BACKUP_FOLDER_PATH = "./memory_storage/VBstorage/backup"
SOURCE_FOLDER_PATH = "./memory_storage/VBstorage"
FORGET_RECORD_PATH = "./memory_storage/miao_memory/chat_memory/forget_record.json"
CONFIG_PATH = './module/config.json'

with open(CONFIG_PATH, "r", encoding="utf-8")as f:
    config = json.load(f)

eco_mode = config["ECO_MODE"]
if eco_mode:
    MODEL_PLUS = MODEL_LONG = MODEL_FLASH = "glm-4-flash"
else:
    MODEL_PLUS = "glm-4-plus"
    MODEL_LONG = "glm-4-long"
    MODEL_FLASH = "glm-4-flash"

long_client = get_client(MODEL_LONG)
flash_client = get_client(MODEL_FLASH)
plus_client = get_client(MODEL_PLUS)
embedding = BgeEmbedding()
Abstractor = Abstract()

def document_split(data, chunk_size=4000, overlap=150):
    if len(data) <= chunk_size:
        return [data]
    chunks = []
    effective_chunk_size = chunk_size - overlap
    for i in range(0, len(data), effective_chunk_size):
        end = min(i + chunk_size, len(data))
        chunks.append(data[i:end])
    return chunks

def get_rs_output(text):
    text = text.replace("\n\n","\n")
    text = text.split("\n")
    rs_output = []
    for i in text:
        if "-" in i and "*" not in i:
            rs_output.append(i.strip())
    return rs_output

def MP_prompt(query):
    pre_process_prompt = RAG_PROMPT_TEMPLATE["Memory_Preprocessing"]
    final_prompt = pre_process_prompt + query
    return final_prompt

def remove_random_elements(lst, num_elements):
    lst_copy = lst.copy()
    num_elements = min(num_elements, len(lst_copy))
    indices_to_remove = random.sample(range(len(lst_copy)), num_elements)
    for index in sorted(indices_to_remove, reverse=True):
        del lst_copy[index]
    return lst_copy

def find_dict_by_value(value,record):
    for item in record:
        if item["memory"] == value:
            return item

# 1. 实时记忆总结

# 1.1 日记写入到json中，不再生成txt文件
with open(HISTORY_PATH_TODAY,"r",encoding = "utf-8")as f:
    text = f.read()
abstract_prompt = RAG_PROMPT_TEMPLATE['daily_map_reduce_template'].format(Miao_Info_Brief=config["Miao_Info_Brief"],
User_Identity=config["User_Identity"],
Miao_Name=config["Miao_Name"],
text = "{text}")
miao_memory = Abstractor.map_reduce_abstract(prompt_template = abstract_prompt,text = text)
print(miao_memory)

with open(MIAO_DIARY_PATH,"r",encoding="utf-8")as f:
    data = json.load(f)

miao_memory_dict = {
    "time": today,
    "memory": miao_memory,
}

found_today = False
for index, item in enumerate(data): 
    if item["time"] == today:
        data[index] = miao_memory_dict
        found_today = True
        break

if not found_today:
    data.append(miao_memory_dict)

with open(MIAO_DIARY_PATH,"w",encoding="utf-8")as f:
    json.dump(data,f,indent=2,ensure_ascii=False)

# 2. 实时结构化记忆抽取
# 2.1 读取聊天记录
with open(HISTORY_PATH_TODAY,"r",encoding="utf-8")as f:
    chat_history_all = f.read().split("爸比：")

# 2.2 如果不存在同步文档，则新建同步文档，如果存在同步文档，则读取同步文档
if not os.path.exists(SYNC_PATH_TODAY):
    with open(SYNC_PATH_TODAY, 'w',encoding="utf-8") as file:
        sync_content = []
else:
    with open(SYNC_PATH_TODAY, 'r',encoding="utf-8")as file:
        sync_content = file.read().split("爸比：")

# 2.3 对于将聊天记录与同步文档中的内容相减，获取剩余的没有总结过的聊天记录
remain_content = []
for content in chat_history_all:
    if content not in sync_content:
        remain_content.append(content)
remain_content_str = "爸比：".join(remain_content)

# 2.4 使用新的记忆摘要规范，依次进行CS-RS-MP，将结果保存到TMP_memory.json中
# 2.4.1 CS-粗摘要
remain_content_list = document_split(remain_content_str)

print("remain_content_str")
print(remain_content_str)

if remain_content_str:
    cs_output_list = []
    date = HISTORY_PATH_TODAY[42:46]+"年"+HISTORY_PATH_TODAY[47:49]+"月"+HISTORY_PATH_TODAY[50:52]+"日"
    if len(remain_content_list) == 1:
        prompt = RAG_PROMPT_TEMPLATE["Coarse_Summary"].format(
            Miao_Name = config["Miao_Name"],
            User_Identity = config["User_Identity"],
            text = remain_content_list[0])
        cs_output = long_client.generate_result(query=prompt)
        cs_output_list.append(cs_output)
    else:
        for content in remain_content_list:
            prompt = RAG_PROMPT_TEMPLATE["Coarse_Summary"].format(
                        Miao_Name = config["Miao_Name"],
                        User_Identity = config["User_Identity"],
                        text = content)
            cs_output = long_client.generate_result(query=prompt)
            cs_output_list.append(cs_output)

    cs_output_final_list = []
    for i in cs_output_list:
        if "-" in i and i not in cs_output_final_list:
            cs_output_final_list.append(i)

    cs_output_final = "\n".join(cs_output_final_list)

    print("cs_output")
    print(cs_output_final)

    # 2.4.2 RS-细摘要
    rs_output_list = []

    while len(rs_output_list) <= 0:
        prompt = RAG_PROMPT_TEMPLATE["Fine_Summary"].format(
            Miao_Name = config["Miao_Name"],
            Miao_Memory_Example = config["Miao_Memory_Example"],
            text=cs_output_final)
        rs_output = plus_client.generate_result(query=prompt)
        rs_output_list_tmp = get_rs_output(rs_output)
        rs_output_list.extend(rs_output_list_tmp)

    print(rs_output_list)

    # 2.4.3 MP-记忆预处理
    mp_list = []

    for rs in rs_output_list:
        prompt = MP_prompt(rs)
        mp_output = plus_client.generate_result(query=prompt)
        mp_output = mp_output.replace("`","")
        mp_output = mp_output.replace("json","")
        mp_output = mp_output.replace("\n","")
        matches = re.findall(r"\{.*?\}", mp_output)
        mp_output_json = json.loads(matches[0])
        mp_dict = {
                "memory":rs,
                "date":date,
                "attribute": mp_output_json
            }
        mp_list.append(mp_dict)

    if os.path.exists(TMP_PATH_TODAY):
        with open(TMP_PATH_TODAY,"r",encoding="utf-8")as f:
            org_mp_list = json.load(f)

        mp_list = org_mp_list + mp_list

        with open(TMP_PATH_TODAY,"w",encoding="utf-8")as f:
            f.write(json.dumps(mp_list,indent=2,ensure_ascii=False))

    else:
        with open(TMP_PATH_TODAY,"w",encoding="utf-8")as f:
            f.write(json.dumps(mp_list,indent=2,ensure_ascii=False))

    # 2.5 更新同步文档
    sync_content += remain_content
    sync_content_str = "爸比：".join(sync_content)
    with open(SYNC_PATH_TODAY,"w",encoding="utf-8")as f:
        f.write(sync_content_str)

# 3. 数据库处理
# 3.1 读取Memory_DB.json、Memory_Vectors.index和Memory_Vectors.json
with open(MEMORY_DB_PATH,"r",encoding="utf-8")as f:
    memory_db = json.load(f)

memory_db_content = [memory["memory"] for memory in memory_db]

# print(memory_db[0])
print(len(memory_db))

with open(MEMORY_VECTORS_PATH,"r",encoding="utf-8")as f:
    vectors_list = json.load(f)
memory_vector = [np.array(vector, dtype=np.float32) for vector in vectors_list]

print(len(memory_vector))

vector_store = VectorStore(memory_db_content, memory_vector)
vector_db = faiss.read_index(INDEX_PATH+"/Memory_Vectors.index")
vector_store.set_index(index=vector_db)

print(vector_store.index.ntotal)


# # 3.2 记忆遗忘
# # 3.2.1 对于记忆召回中提到过的记忆，均不进行遗忘，如果提到的次数增加，可以增加importance的权重；

with open(INVOKE_RECORD_TXT_PATH,"r",encoding="utf-8")as f:
    record_raw = f.read().split("\n")
record_list = [memory[12:] for memory in record_raw if memory[0:12]]


print(record_list)

with open(INVOKE_RECORD_PATH,"r",encoding="utf-8")as f:
    record = json.load(f)

print(record)

with open(FORGET_RECORD_PATH,"r",encoding="utf-8")as f:
    forget_records = json.load(f)

print(forget_records)

if today not in [forget_record["date"] for forget_record in forget_records]:

    for memory in record_list:
        if memory:
            if memory not in [memory["memory"] for memory in record]:
                record_dict = {
                    "memory":memory,
                    "rounds":1
                }
                record.append(record_dict)
            else:
                for i in record:
                    if memory == i["memory"]:
                        i["rounds"] += 1

    with open(INVOKE_RECORD_PATH,"w",encoding="utf-8")as f:
        json.dump(record,f,indent=2,ensure_ascii=False)

    level_one = []
    level_two_daily = []
    level_two_work = []
    level_two_social = []
    level_three = []
    level_three_memory = []

    for memory_index, memory in enumerate(memory_db):
        if memory["memory"] in [memory["memory"] for memory in record]:
            item = find_dict_by_value(memory["memory"],record)
            if item["rounds"] >= len(record)//2 and memory["attribute"]["memory_importance"] < 3:
                memory["attribute"]["memory_importance"] += 1
        elif memory["attribute"]["memory_importance"] == 1:
            level_one.append(memory_index)
        elif memory["attribute"]["memory_importance"] == 2:
            if memory["attribute"]["memory_type"] == "personal_daily_event":
                level_two_daily.append(memory_index)
            elif memory["attribute"]["memory_type"] == "work_study_event":
                level_two_work.append(memory_index)
            elif memory["attribute"]["memory_type"] == "social_emotional_event":
                level_two_social.append(memory_index)
        elif memory["attribute"]["memory_importance"] == 3:
            level_three.append(memory_index)
            level_three_memory.append(memory["memory"])

    # 3.2.2 对于所有记忆等级为1的记忆，使用randint生成随机遗忘值，获取遗忘记忆的序号；

    if len(level_one) >= 2:
        forget_num = random.randint(2,len(level_one))//2
        remain_index = remove_random_elements(level_one,forget_num)
        level_one_forget = [num for num in level_one if num not in remain_index]
    else:
        level_one_forget = []

    # 3.2.3 对于所有记忆等级为2的记忆，如果为日常记忆，则直接1/3randint遗忘值，如果为学习工作记忆，则1/4randint遗忘，如果为社交情感记忆，则1/5randint遗忘，特殊事件不遗忘；

    if len(level_two_daily) >= 3:
        forget_num = random.randint(3,len(level_two_daily))//3
        remain_index = remove_random_elements(level_two_daily,forget_num)
        level_two_daily_forget = [num for num in level_two_daily if num not in remain_index]
    else:
        level_two_daily_forget = []

    if len(level_two_work) >= 4:
        forget_num = random.randint(4,len(level_two_work))//4
        remain_index = remove_random_elements(level_two_work,forget_num)
        level_two_work_forget = [num for num in level_two_work if num not in remain_index]
    else:
        level_two_work_forget = []

    if len(level_two_social) >= 5:
        forget_num = random.randint(5,len(level_two_social))//5
        remain_index = remove_random_elements(level_two_social,forget_num)
        level_two_social_forget = [num for num in level_two_social if num not in remain_index]
    else:
        level_two_social_forget = []

    final_forget_index = level_one_forget + level_two_daily_forget + level_two_work_forget + level_two_social_forget
    print(final_forget_index)
    print(len(final_forget_index))

    # 3.2.3 对于所有记忆等级为3的记忆，两两计算语义相似度，如果语义相似度大于0.95，则随机遗忘其中一个

    # level_three_vector = [vector for index, vector in enumerate(memory_vector) if index in level_three]
    level_three_vector = []
    for index,vector in enumerate(memory_vector):
        for i,memo_index in enumerate(level_three):
            if index == memo_index:
                level_three_vector.append(vector)

    level_three_vb = VectorStore(level_three_memory, level_three_vector)
    level_three_vb.build_index()

    # print(len(level_three_memory))
    # print(len(level_three))
    # print(len(level_three_vector))

    level_three_forget = []
    item_needs_forget = []

    for index, memory_info in enumerate(level_three_memory):
        item_vector = level_three_vector[index]
        # print(type(memory_info))
        # print(type(item_vector))
        # print(memory_info)
        if isinstance(memory_info,str):            
            most_similar = level_three_vb.query_with_vector(query=memory_info, EmbeddingModel=embedding, k=4)
            # print(most_similar)
            similarity = level_three_vb.get_similarity(item_vector, most_similar[1][1])
            if similarity > 0.95:
                item_combine = [memory_info, most_similar[1][0]]
                random_num = random.randint(0,1)
                forget_item = item_combine[random_num]
                item_needs_forget.append(forget_item)

    print(item_needs_forget)

    for index, item in enumerate(level_three_memory):
        if item in item_needs_forget:
            forget_item = level_three[index]
            level_three_forget.append(forget_item)



        # for index, item in enumerate(level_three_memory):
        #     if item == forget_item:
        #         forget_index = index
        #         level_three_forget.append(forget_index)

    # print(level_three_forget)
    # print(len(level_three_forget))

    final_forget_index = final_forget_index + level_three_forget
    print(len(final_forget_index))

    # 3.2.4 获取遗忘序号后，进行重排序，对于document、json与index三个文件，同步进行遗忘；
    final_forget_index = sorted(final_forget_index)

    memory_db_remain = []
    for memory_index,memory in enumerate(memory_db):
        if memory_index not in final_forget_index:
            memory_db_remain.append(memory)
    memory_db_remain_content = [memory["memory"] for memory in memory_db_remain]
    print(len(memory_db_remain))

    memory_vector_remain = []
    for memory_index,memory in enumerate(memory_vector):
        if memory_index not in final_forget_index:
            memory_vector_remain.append(memory)

    print(len(memory_vector_remain))

    vector_store.remove_ids(ids_to_remove=final_forget_index)

    forget_record = {
        "date": today
    }
    forget_records.append(forget_record)
    with open(FORGET_RECORD_PATH,"w",encoding="utf-8")as f:
        f.write(json.dumps(forget_records,indent=2,ensure_ascii=False))

    with open(INVOKE_RECORD_TXT_PATH,"w",encoding="utf-8")as f:
        pass

else:
    memory_db_remain_content = memory_db_content
    memory_vector_remain = memory_vector


# 3.3 加入TMP记忆
# 3.3.1 检查上次打开之后没有同步完的记忆，如果有，则进行同步；将上次的记忆文档与今天的记忆文档拼接，获取最终的tmp_memory
for file_name in os.listdir(TMP_FOLDER_PATH):
    if file_name.endswith("TMP_memory.json") and file_name != TMP_PATH_TODAY:
        file_name_path = os.path.join(TMP_FOLDER_PATH,file_name)
        with open(file_name_path,"r",encoding="utf-8")as f:
            former_tmp_memory = json.load(f)
    else:
        former_tmp_memory = []

if os.path.exists(TMP_PATH_TODAY):
    with open(TMP_PATH_TODAY,"r",encoding="utf-8")as f:
        today_tmp_memory = json.load(f)
else:
    today_tmp_memory = []

tmp_memory_raw = former_tmp_memory + today_tmp_memory
tmp_filter = []
tmp_memory_raw_filter = []

if tmp_memory_raw:
    tmp_memory_content = [memory["memory"] for memory in tmp_memory_raw]
    # 3.3.2 过滤掉TMP_memory.json中与memory_db_remain重合的部分；
    for memory in tmp_memory_raw:
        if memory["memory"] not in memory_db_remain_content and memory["memory"] not in tmp_filter:
            tmp_filter.append(memory["memory"])
            tmp_memory_raw_filter.append(memory)

tmp_memory_content = tmp_filter
tmp_memory_raw = tmp_memory_raw_filter
print(tmp_memory_content)

# # 3.3.3 对TMP_memory.json中的每一个事件，使用向量数据的query方法，进行关联事件查询，获取关联事件后，拼接到memory_db.json文件后面
# embedding = BgeEmbedding()
if tmp_memory_content:
    tmp_memory_vector = []
    for item in tmp_memory_content:
        print(item)
        item_vector = embedding.get_embedding(item)
        tmp_memory_vector.append(item_vector)
    # tmp_memory_vector = [embedding.get_embedding(memory) for memroy in tmp_memory_content]
else:
    tmp_memory_vector = []

memory_combine_raw = memory_db + tmp_memory_raw
print(memory_combine_raw[0:2])

memory_content_combine = memory_db_remain_content + tmp_memory_content
memory_vector_combine = memory_vector_remain + tmp_memory_vector

print(memory_content_combine[0:2])
vector_store_combine = VectorStore(memory_content_combine, memory_vector_combine)
vector_store_combine.build_index()

memory_db_combine = []
for index, memory in enumerate(memory_content_combine):
    memory_vector = memory_vector_combine[index]
    print(memory)
    relate_memory = vector_store_combine.query_with_vector(query=memory, EmbeddingModel=embedding, k=4)
    print(relate_memory)
    relate_memory = [memory[0] for memory in relate_memory]
    relate_memory = relate_memory[1:]
    relate_memory_with_date = [find_dict_by_value(memory,memory_combine_raw)["date"] + " " + find_dict_by_value(memory,memory_combine_raw)["memory"] for memory in relate_memory]
    org_memory_dict = find_dict_by_value(memory,memory_combine_raw)
    memory_process_dict = {
        "memory":memory,
        "date":org_memory_dict["date"],
        "attribute":
        {
            "memory_importance":org_memory_dict["attribute"]["memory_importance"],
            "memory_type":org_memory_dict["attribute"]["memory_type"],
            "relate_memory":relate_memory_with_date
        }
    }
    memory_db_combine.append(memory_process_dict)

# print(json.dumps(memory_db_combine[0:2],indent=2,ensure_ascii=False))

with open(MEMORY_DB_PATH,"w",encoding="utf-8")as f:
    json.dump(memory_db_combine,f,indent=2,ensure_ascii=False)

# 3.3.4 在Memory_Vectors.index中进行merge并保存
vector_store_combine.save_merged_index(path = "./memory_storage/VBstorage")

# 3.4 保险机制
# 3.4.1 检查今日的Memory_DB.json长度是否与Memory_Vectors.index的ntotal是否一致，如果不一致，则使用今天的Memory_DB.json重新persis一份Memory_Vectors.index，覆盖原有的index文件

len_json = len(memory_db_combine)
len_index = vector_store_combine.index.ntotal
embedding = BgeEmbedding()

# secure_memory_vector = memory_vector + tmp_vector

if len_json != len_index:
    secure_memory_db = [memory["memory"] for memory in memory_db_combine]
    vector_store_combine = VectorStore(memory_content_combine, memory_vector_combine)
    vector_store_combine.get_vector(EmbeddingModel=embedding)
    vector_store_combine.build_index()
    vector_store_combine.save_merged_index(path = "./memory_storage/VBstorage")

# 3.4.2 保险机制2-清空Backup文件夹，为今日的Memory_DB.json文件、Memory_Vector.json文件与Memory_Vectors.index文件创建备份，保存到Backup文件夹中。
for item in os.listdir(BACKUP_FOLDER_PATH):
    ITEM_PATH = os.path.join(BACKUP_FOLDER_PATH, item)
    if os.path.isfile(ITEM_PATH):
        os.remove(ITEM_PATH)

files_to_backup = ['Memory_DB.json', 'Memory_Vectors.index', 'Memory_Vectors.json']

for file_name in files_to_backup:
    source_file = os.path.join(SOURCE_FOLDER_PATH, file_name)
    backup_file = os.path.join(BACKUP_FOLDER_PATH, f"{today}_{file_name}")
    shutil.copy(source_file, backup_file)

# 4. 清除tmp文件夹中不属于今天的内容

# 清理临时文件夹，删除不是今天的.json和.txt文件
for item in os.listdir(TMP_FOLDER_PATH):
    item_path = os.path.join(TMP_FOLDER_PATH, item)
    if os.path.isfile(item_path) and not item.startswith(today) and item.endswith(('.json', '.txt')):
        os.remove(item_path)

# 清理历史文件夹，删除不是今天的.txt文件
for item in os.listdir(HISTORY_FOLDER_PATH):
    item_path = os.path.join(HISTORY_FOLDER_PATH, item)
    if os.path.isfile(item_path) and not item.startswith(today) and item.endswith('.txt'):
        os.remove(item_path)




