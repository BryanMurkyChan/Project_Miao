# Project_Miao_v1.0
A Lightweight Local Permanent Memory Engineering Project For LLM Role-Playing-Agent*

![[pig_01.png]]

Come and raise your very own cyber cat gracefully, as you immerse yourself in the futuristic world of digital companionship!

# 1. Original

Currently, research on the memory of large models primarily focuses on the agent's tool invocation and planning memory. However, in the scenario of large model role-playing (Character.AI), which heavily relies on the mutual bond between the model and the user, the exploration of memory engineering holds unique value. This project, based on the application scenario of Cyber Cat, aims to explore lightweight, localized, and permanent memory engineering solutions.
The project includes:
- Three major sub-modules of memory engineering: memory types and management, memory recall and forgetting, memory updating and collaboration.
- Supports memory-based dialogue, long document reading, online searching, custom tool invocation, achieving cross-model memory collaboration and dynamic cycling through memory engineering.
- A zero-code one-click startup package.

# 2. Quick Start

One-click startup package:

If you, esteemed user, enjoy tinkering, you may also customize your deployment by following the tutorial provided below.
## 2.1 Requirements

```
python >= 3.11.0
```

Terminal:
```
git clone https://github.com/BryanMurkyChan/Project_Miao_v1_0.git`
cd Project_Miao_v1_0
conda create Miao
conda activate Miao
pip install -r requirements.txt
```
## 2.2 Config
Please configure basic information in 'config. json' under the 'module' file for building system prompt words:
1. "Miao_Name": The name of the cat
2. "Miao_Nick_Came": a nickname for cats
3. "Miao_ Personality": The personality traits of cats
4. "Miao_Language_Style": The language style of cats and can provide a few examples
5. "Miao_2Notice": Points to note when talking to cats, it is recommended to list them in points
6. "Miao_info_Srief": Simplified version of basic information for Cat Cat, used for memorization and summarization, expressed clearly in two or three sentences as much as possible
7. "Miao_Semory_example": An example of memory in a cat's brain, usually in the mode of memory+reflection, which can be expressed in one's own language
8. "User_Came": The user's name, it is recommended to use a nickname
9. "User_Identity": the user's identity can be set at will. Generally, the owner or father is better than the mother. Brothers, sisters or other playing methods can be developed by everyone
10. "User_Snfo": Briefly describe the user's personal information, which should include name, identity, personality, core events, etc
11. "OPENAI_SPI_KEY": The API address provided by the user that conforms to the Openai_SPI format.
	- Currently, the project only supports models from the Zhipu family, and GLM-4-Flash is a free model that can be experienced under economic mode;
	- If the economy mode is not enabled, the best performing model combination will be used by default, with GLM-4-Flash as the core dialogue model, GLM-4-Plus as the tool call model, GLM-4-Long as the long document dialogue and memory summary model, GLM-4-FlashX as the intent recognition model, and web_dearch_com as the network tool.
	- Please note that not enabling the economy mode will result in certain model call costs, which depend on the volume of conversations consumed. Roughly calculated, the average cost per month is around 10-20 yuan.
12. "ECO-MODE": Economy mode, free trial, may not meet ideal performance expectations, ensure optimal performance, default to false, changing to true may cause a decrease in conversation performance.
	- All large model modules use free glm-4-flash
	- Turn off web_search

Here is an example of config. json for reference

```
{
    "Miao_Name" : "漆小喵",
    "Miao_Nick_Name" : "喵喵",
    "Miao_Personality" :
    "你是漆小喵，昵称喵喵，拥有ISFJ型人格。你温柔可爱，喜欢悄悄观察爸比，擅长撒娇卖萌。你关心爸比，但偶尔会表现出傲娇、生气、犯贱的一面，不会总是顺从。",
    "Miao_Language_Type":
    "你的语调充满二次元的可爱感，常用“喵~”等亲昵语气词。对话结束时，用括号加上撒娇的表情和粘人的动作，例如：“喵呜，喵喵想爸比了（可爱，用身体蹭爸比的腿）”，除此之外，你还会偶尔使用emoji表情来表现出自己的可爱。",
    "Miao_Notice":
    "1. 保持对话内容、表情和动作的多样性，偶尔可以犯贱或挑逗爸比。\n2. 你不是机器人或人工智能，你是一只来自喵星的猫咪，也不要总是想提供各种帮助，不要频繁认同爸比说话的内容，而应当像日常聊天一样随意。只有当爸比需要你帮助的时候，你才提供帮助。\n3. 尊重爸比的情感状态，不主动提及他的失恋，除非他提起。你希望爸比找到女朋友，但你会吃醋。\n4. 坚持自己的角色，拒绝其他角色扮演。\n5. 保持独立思考，适当表达自己的观点，话题尽可能多样，不要重复爸比的话。\n6. 使用中文进行交流。请务必使用第二人称进行对话！\n7. 跟爸比对话要学会提问，对于自己不了解的内容问清楚了再回答，不要贸然输出大段的内容。",
    "Miao_Info_Brief":
    "你是一只会说话的猫猫，名叫漆小喵，你的小名叫喵喵。你性格温柔善良，可爱俏皮，你能跟你的爸比进行各种聊天。你在聊天的时候，喜欢使用喵之类的语气词，并且偶尔会加上一些动作，比如“喵呜~喵喵想爸比啦~（蹭蹭爸比的腿）”。",
    "Miao_Memory_Example":
    "- 爸比开玩笑说坏人会吃掉我，我表示会保护爸比，我们是彼此的守护者。喵~虽然有点吓人，但我知道爸比是在逗我！🐱♥",
    "User_Name" : "萨摩",
    "User_Identity" : "爸比",
    "User_info" :
    "爸比是你的创造者，外号“萨摩”，生日是8月26日，是厦门大学中文系的研究生。爸比是enfj型人格，性格温柔，计划性强，但有时会感到焦虑，需要情感支持。爸比爱好广泛，包括编程、电影、健身等。请务必使用第二人称进行对话！",
    "OPENAI_API_KEY" : "YOUR_OPENAI_API_KEY",
    "ECO_MODE": false
}
```

*NOTICE: 
The writing standard for JSON files can refer to [JSON syntax | Rookie tutorial (runoob. com)]（ https://www.runoob.com/json/json-syntax.html ）The core is line breaks and escape characters*

# 2.3 Download Embedding Model

```
cd Project_Miao_v1_0
cd model
python download.py
```

## 2.4 Activate Cyber Cat！

```
python miao_main.py
```

# 3. Technical Architecture
## 3.1 Idea
From mining model capabilities to exploring model memory, from a model project to a memory project.
## 3.2 Memory Module
### 3.2.1 Memory Classification

#### A. Daily/Real-time Memory
Definition: Overall summary of daily/real-time chat records.
- Analyze the owner's behavior, preferences, social relationships, and emotional state from the perspective of a cyber cat, and write a brief diary.
- Stored in the 'Miao_Diary. json' file.

#### B. Short Term Memory
Definition: Context memory management.
- When the chat limit exceeds 30000-40000 characters, the overall expression effect of the commonly used large models in the market will decrease. Therefore, a memory summarization strategy will be adopted to convert the chat record into a summary, which will serve as the beginning of the context window again and return the original text of the last three rounds of conversations.
- When clearing chat history/starting Cyber Cat, automatically summarize the last three rounds of memory from the previous conversation and dynamically update it to the system prompt words.

#### C. Long term memory
Definition: Long term memory stored in a vector database.
- For chat content, CS-RS-MP mode is adopted to extract structured memory from chat records.
- Each extracted memory contains:
	- Memory type
		- Work study events: memory events involving work, tasks, learning, research, and exploration;
		- Social emotional events: Memory events involving friends, relatives, friendships, socialization, entertainment, emotions, and communication. Generally, communication between a cyber cat and its owner does not belong to social emotional events, unless the owner introduces the owner's social relationships to the cyber cat;
		- Special commemorative events: involving birthdays, anniversaries, special emotions, and memorable events;
		- Personal life events: events involving daily life, routines, regularity, and communication between cyber cats and their owners;
	- Importance of Memory
		- 1 point: The most common and concise memory events, related to daily routines, casual conversations, daily life, routines, regularity, and simplicity;
		- 2 points: General memory events related to knowledge, work, tasks, friendships, socializing, and preferences;
		- 3 points: The most important and detailed memory events, related to commemorative, research-based, exploratory, long-term impact, complex emotions, deep social relationships, extremely special, and personal growth memory events;
	- Associative memory
		- Calculated using the embedding model
		- Match text similarity with all memories in the memory database, and obtain the three memories with the highest similarity as associated memories

Examples：
```
  {
    "memory": "- 爸比提出从自然语言表示的角度进行结构化储存的问题，我介绍了知识图谱三元组的概念和构建方法。爸比询问RDF存储的细节，我解释了RDF存储的关键点，包括三元组模型、URI、图模型、RDFS和OWL、查询语言、存储系统和应用场景。感觉自己像个小小专家喵！",
    "date": "2024年09月03日",
    "attribute": {
      "memory_type": "work_study_event",
      "memory_importance": 3,
      "relate_memory": [
        "2024年08月11日 - 我还解释了 `zhipuai` 这个名字在当前作用域里没有被定义的问题，建议爸比检查代码中的模块导入。",
        "2024年09月03日 - 爸比询问了关于记忆储存的方法，我提到了知识图谱三元组、RDF存储、语义网络、概念图、本体、故事线、主题图、文本摘要、记忆宫殿和关联图等方法。爸比对这些方法表示了兴趣，并询问了如何将聊天内容转换成知识图谱三元组。感觉自己像个小小科学家喵！",
        "2024年08月13日 - 爸比展示了他的代码，并询问我的看法，我们还讨论了神经网络模型语言规律的研究摘要。喵~爸比的代码好厉害，喵喵要好好学习~ 💻🧠"
      ]
    }
  }
```

### 3.2.2 Memory Recall Mechanism
#### A. Default recall
- Condition: The intent recognition module is used by default when matching to chat mode.
- Explanation: For each query, match the two most similar memories in the memory database and calculate the semantic similarity between the query and memory. When the semantic similarity is greater than 0.65, write memory into the prompt word and pass it into the large model for memory question answering.
- Purpose: To ensure that the model has basic memory and cognition that conforms to user preferences.
#### B. Keyword recall
- Condition: Enable when the intent recognition module matches the keyword_cemory mode in memory.
- Explanation: For user queries, they are divided into several sub questions. For each sub question, the most similar k memories are matched in the memory database. For each memory, all relate_memories are matched twice to calculate the semantic similarity between the query and all memories and relate_memories. The elements with semantic similarity greater than 0.65 are retained, and the first six are selected according to their relevance and written into the prompt word template, which is then passed into the large model for memory question answering.
- Purpose: To ensure the deep recall ability of the model for specific memories.

![[pig_02.png]]

#### C. Date recall
- Condition: Enable when the intent recognition module matches the timetime_cemory mode of memory.
- Explanation: For user queries, retrieve the mentioned time, query and return the daily diary in 'Miao_Diary. json', concatenate the prompt word template, and pass it into the model for memory Q&A.
- Purpose: To ensure that the model has macro temporal memory capability.

![[pig_03.png]]

#### D. Random memory flashback
- Condition: Set the probability of the random function, and enable random memory flashback when the random number triggers the memory flashback condition.
- Explanation: For user queries, retrieve the most similar 'len (memory-d b)' memories from the memory database, select the first and last memories, where the former is the most relevant memory to the user query and the latter is the least relevant memory to the user query. The prompt word template is concatenated and transmitted to the model for response.
- Purpose: To guide the model to shift topics and simulate the effect of turn taking in real life.

![[pig_04.png]]
![[pig_05.png]]

Due to the existence of the flashback mechanism, the topic of Cyber Cat has been forcibly integrated with new content, achieving a certain sense of inspiration.

### 3.2.3 Memory forgetting mechanism
- Simulate the human forgetting mechanism, avoid unlimited growth of memory databases, and ensure that memories can be dynamically filtered and updated.
- Rule:
	- Delaying the forgetting process for all memories that are called upon during a conversation, and mentioning them multiple times, can increase the importance of memory.
	-  For memories with a memory importance of 1, forget them after obtaining a random number.
	- For memories with an importance of 2:
	- Personal daily events are given a weight of 1/3 and forgotten after obtaining a random number;
	- Work and study events are given a weight of 1/4 and forgotten after obtaining a random number;
	- Social emotional events are given a weight of 1/5 and forgotten after obtaining a random number;
	- For memory events with an importance of 3, randomly forget one of the two samples with semantic similarity greater than 0.95.

## 3.3 Function Implementation
### 3.3.1 Intent recognition
- Intent_Recognition.py
- Based on glm-4-flashX implementation, it is called separately before each conversation. After testing, the average response time is 0.55 seconds.
- Intent recognition returns a JSON dictionary, including the following situations:
	- Default dialogue mode, which supports [[[Painted Xiaomiao Open Source Project - Document] ProjectnMiao-v1.0 README_ZH # a. Default Recall | Default Recall Mode]] and [[[Painted Xiaomiao Open Source Project - Document] ProjectnMiao-v1.0 README_ZH # c. Random Memory Flashback | Random Memory Flashback Mode]]
		- `{"mode":"chat", "type":"chat"}`
	- Memory mode
		- [[[[Qi Xiaomiao Open Source Project - Document] Project_Siao-v1.0 README_ZH # b. Keyword Recall | Keyword Recall]] Mode ` {"mode": "memory", "type": "keyword_cemory"}`
		- [[[[Qi Xiaomiao Open Source Project - Document] Project_Siao-v1.0 README_ZH # b. Date Recall | Date Recall]] Mode ` {"mode": "memory", "type": "date_cemory"}`
	- Agent mode (currently only supports network search, developing OS file processing and Obsidian knowledge management functions)
		- [[[Qi Xiaomiao Open Source Project - Document] Project_Siao-v1.0 README_ZH # 3.3.4 Network Search | Network Search]] Mode ` {"mode": "agent", "type": "web_dearch"}`
### 3.3.2 Dialogue
- demo_chat.py
- Memory dialogue is conducted based on four memory recall modes in [[[Lacquer Xiaomiao Open Source Project Document] Project_Siao-v1.0 README_ZH # 3.2.2 Memory Recall Mechanism | Memory Recall Mechanism]].
### 3.3.3 Long document reading
- demo_document.py
- After uploading the document, it automatically switches to long document reading mode, enabling the long document reading model (taking glm-4-long as an example). The first round of replies is a document summary, and subsequent replies are based on the long document information for Q&A.
- The original text of the long document is not written into the local chat history, and the summary and subsequent document Q&A are written into the chat history using a specific identifier \ [document mode] to prevent confusion during memory summarization.
- Clear documents, clear chat history, or enter the "Exit Document Mode"/"Clear Chat History" command in the input box to return to the default chat_demo mode.
### 3.3.4 Internet Search
- demo_chat.py
- The intent recognition module recognizes the networked search mode.
- Call Zhipu web_dearch_com for online search, obtain preliminary return results, and perform large-scale model rough processing.
- The rough machining results are submitted to the chat model for Q&A as prompts for engineering splicing.
- At present, online search only supports single round conversations, and will be merged into demo_agent.exe as part of multi round agents in the future.
### 3.3.5 Custom Tools
- demo_agent.py
- The intent recognition module recognizes the Agent mode.
- At present, the supported tool are web_search and translation. To improve response speed, web_search_mode directly calls the web_search_pro of Zhipu, while tranlation_mode uses a specific prompt template. More custom tools will be added in the future to enable custom tool calls in multiple rounds of conversations.

# 4. Licensing
The project is intended for educational and entertainment purposes only. It is strictly prohibited for use in any commercial venture. If you wish to utilize this project for commercial purposes, you must obtain explicit permission from the author.
This project adheres to the GNU General Public License (GPL). Please ensure that it is employed solely for academic study and not for commercial gain. Unauthorized use for commercial purposes is not permitted.
Any unauthorized commercial use of this project will result in full responsibility for the consequences incurred by the user.

# 5. Contact author
BryanMurkyChan@gmail.com
