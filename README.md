# 😼 Project_Miao_v1.0
*面向大模型角色扮演的轻量化本地永久记忆解决方案*

![default](./picture/default.png)


一起来优雅地养一只属于自己的赛博猫猫吧！

## 🐱 项目缘起

当前，关于大模型的记忆的研究大多集中于Agent的工具调用与规划记忆，而大模型角色扮演（Character.AI）这一场景，极度依赖模型与用户之间的共同羁绊，其记忆工程的探索具备独特价值，本项目基于赛博猫猫这一应用场景，探索轻量化+本地化+永久记忆的工程解决方案。

项目包含：
- 记忆工程三大子模块：记忆类型与管理、记忆召回与遗忘、记忆更新与协同
- 支持记忆对话、长文档阅读、联网搜索、自定义工具调用，通过记忆工程实现跨模型记忆协同与动态循环
- 零代码一键启动包

## 🚀 快速开始

一键启动包：

阁下如果喜欢折腾，也欢迎按照下面的教程进行自定义部署

### 🖥️ 安装依赖

项目版本要求：
```
python >= 3.11.0
```

项目拷贝至本地
`git clone https://github.com/BryanMurkyChan/Project_Miao_v1_0.git`

命令行安装项目依赖
```
cd Project_Miao_v1_0
conda create Miao
conda activate Miao
pip install -r requirements.txt
```

### 🤔 配置基本信息

请在`module`文件下下面的`config.json`中配置基本信息，用于构建系统提示词：
1. "Miao_Name" : 猫猫的姓名
2. "Miao_Nick_Name" : 猫猫的昵称
3. "Miao_Personality" : 猫猫的性格特征
4. "Miao_Language_Style": 猫猫的语言风格，可以给少量示例
5. "Miao_Notice" : 猫猫对话时需要注意的事项，建议分点罗列
6. "Miao_Info_Brief" : 猫猫的简化版基础信息，用于记忆摘要，尽量两三句话表述清楚
7. "Miao_Memory_Example" : 猫猫脑子里的记忆举例，一般是记忆+感想的模式，可使用自己的语言表述
8. "User_Name" : 用户的名字，建议使用昵称
9. "User_Identity" : 用户的身份，可随意设置，一般为主人或爸比妈咪效果比较好，兄弟姐妹或者其他玩法大家可以自行开发
10. "User_Info" : 简述用户的个人信息，应包括姓名、身份、性格、核心事件等
11. "OPENAI_API_KEY" : 用户提供的符合Openai_API格式的API地址。
	1. 目前项目仅支持智谱家族的模型，GLM-4-Flash为免费模型，可在经济模式下体验本项目；
	2. 如不开启经济模式，则默认使用最佳性能的模型组合，GLM-4-Flash为核心对话模型，GLM-4-Plus为工具调用模型，GLM-4-Long为长文档对话与记忆摘要模型，GLM-4-FlashX为意图识别模型，web_search_pro为网络工具。
	3. 请注意，不开启经济模式，会造成一定的模型调用费用，具体视对话量消耗计算，粗略计算一个月平均10~20元左右。
12. "ECO_MODE": 经济模式，免费试用，可能无法达到理想性能预期，确保最佳性能，默认为false，改为true开启后可能会导致对话性能下降。
- 在经济模式下：
	1. 所有大模型模块均使用免费的glm-4-flash
	2. 关闭网络搜索


下面是一份供参考的config.json样例

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
json文件书写规范可参考[JSON 语法 | 菜鸟教程 (runoob.com)](https://www.runoob.com/json/json-syntax.html) 核心是换行符和转义符*

## 🛜 下载embedding模型

```
cd Project_Miao_v1_0
cd model
python download.py
```

### 🚀 赛博猫猫，启动！

```
python main.py
```

## 🧑‍🔧 技术架构

### 🚩 核心理念
从模型能力的挖掘，转变为模型记忆探索，从一个模型项目，转变为一个记忆项目。

### 🧐 记忆模块

#### ❤️ 记忆分类

##### 🐱 每日/实时记忆
定义：每日/实时聊天记录的整体总结。
- 以赛博猫猫的视角对主人的行为、偏好、社交关系、情感状态进行分析，并撰写简要日记。
- 储存在`Miao_Diary.json`文件中。

##### 🗒️ 短期记忆
定义：上下文记忆管理。
- 当聊天额度超过30000~40000字符时，市面通用大模型表达效果会整体下降，采取记忆摘要的策略将聊天记录转换为摘要，重新作为上下文窗口的开始，并返回最近三轮对话原文。
- 当清空聊天记录/启动赛博猫猫时，自动摘要上一次对话的最后三轮记忆，动态更新至系统提示词中。

##### 💾 长期记忆
定义：储存在向量数据库中的长期记忆。
- 对于聊天内容，采取CS-RS-MP模式，对聊天记录进行结构化记忆抽取。
- 抽取得到的每一条记忆，均包含：
	- 记忆类型
		- 工作学习事件：涉及到工作、任务、学习、研究、探索的记忆事件；
		- 社交情感事件：涉及到朋友、亲人、友谊、社交、娱乐、情感、交流的记忆事件，一般情况下，赛博猫猫与主人的交流不属于社交情感事件，除非是主人向赛博猫猫介绍主人的社交关系；
		- 特殊纪念事件：涉及到生日、纪念日、特殊情感、特别要记住的记忆事件；
		- 个人生活事件：涉及到生活、作息、规律化、日常、赛博猫猫与主人的日常交流的事件；
	- 记忆重要性
		- 1分：最普通、简略的记忆事件，关于日常化、闲聊、生活、作息、规律化、简单的记忆事件；
		- 2分：一般的记忆事件，关于知识、工作、任务、友谊、社交、喜好的记忆事件；
		- 3分：最重要、详细的记忆事件，关于纪念性、研究性、探索性、长期影响、复杂情感、深度社交关系、极其特殊、个人成长的记忆事件；
	- 关联记忆
		- 使用embedding模型计算得到
		- 与记忆数据库中所有记忆进行文本相似度匹配，获取相似度最高的三条记忆作为关联记忆

样例如下：
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

#### 🧠 记忆召回机制
##### 🐱 默认召回
- 条件：意图识别模块匹配为chat模式时默认使用。
- 解释：对于每一次query，在记忆数据库中匹配最相似的两条memory，并计算query与memory之间的语义相似度，当语义相似度大于0.65时，将memory写入提示词中，传入大模型进行记忆问答。
- 目的：确保模型具备符合用户偏好的基本记忆与认知。

##### 🔑 关键词召回 
- 条件：当意图识别模块匹配为memory下的keyword_memory模式时启用。
- 解释：对于用户query，拆分成若干子问题，对于每一个子问题，在记忆数据库中匹配最相似的k条memory，对于每一条memory，二次匹配所有relate_memory，计算query与所有memory、relate_memory的语义相似度，保留语义相似度大于0.65的元素，按相关性高低取前六条写入提示词模板中，传入大模型进行记忆问答。
- 目的：确保模型对特定记忆的深层回忆能力。

![miao_pig05](./picture/readme_05.png)

##### 📅 日期召回
- 条件：当意图识别模块匹配为memory的datetime_memory模式时启用。
- 解释：对于用户query，获取提到的时间，在`Miao_Diary.json`中查询并返回当日的日记，通过提示词模板拼接后，传入模型进行记忆问答。
- 目的：确保模型具备宏观时间记忆能力。

![miao_pig04](./picture/readme_04.png)


##### ⚡ 随机记忆闪回
- 条件：设定随机函数概率，当随机数触发记忆闪回条件时，启用随机记忆闪回。
- 解释：对于用户query，获取记忆数据库中最相似的`len(memory_db)`条memory，选取第一条与最后一条，前者是与用户query最相关的记忆，后者是与用户query最不相关的记忆，通过提示词模板拼接后传入模型进行回答。
- 目的：引导模型进行话题转移，模拟现实中话轮转换效果。

例如：
![miao_pig02](./picture/readme_02.png)

由于闪回机制的存在，赛博猫猫的话题被强制融合了新内容，实现某种意义上的灵光一闪。

#### 🗑️ 记忆遗忘机制
- 模拟人类遗忘机制，避免记忆数据库无限制增长，确保记忆能被动态过滤更新。
- 规则：
	- 对于所有在对话中被调用的记忆，延缓遗忘进度，多次提及时，可增加记忆重要性。
	- 对于记忆重要性为1的记忆，获取随机数后遗忘。
	- 对于记忆重要性为2的记忆：
		- 个人日常事件按照1/3的权重，获取随机数后遗忘；
		- 工作学习事件按照1/4的权重，获取随机数后遗忘；
		- 社交情感事件按照1/5的权重，获取随机数后遗忘；
	- 重要性为3的记忆事件，对于语义相似度大于0.95的两个样本，随机遗忘其中一个。

### 😸 功能实现
#### 🤔 意图识别
- Intent_Recognition.py
- 基于glm-4-flashX实现，每次对话前单独调用，经测试，平均响应速度为0.55秒。
- 意图识别返回json字典，包含以下情况：
	- 默认对话模式，该模式下，支持[[【漆小喵开源项目-文档】Project_Miao_v1.0 README_ZH#a. 默认召回 | 默认召回模式]]与[[【漆小喵开源项目-文档】Project_Miao_v1.0 README_ZH#c. 随机记忆闪回 | 随机记忆闪回模式]]
		- `{"mode":"chat", "type":"chat"}`
	- 记忆模式
		- [[【漆小喵开源项目-文档】Project_Miao_v1.0 README_ZH#b. 关键词召回 | 关键词召回]] 模式 `{"mode":"memory", "type":"keyword_memory"}`
		- [[【漆小喵开源项目-文档】Project_Miao_v1.0 README_ZH#b. 日期召回 | 日期召回]] 模式 `{"mode":"memory", "type":"date_memory"}`
	- Agent模式（目前仅支持网络搜索，正在开发os文件处理与obsidian知识管理功能）
		- [[【漆小喵开源项目-文档】Project_Miao_v1.0 README_ZH#3.3.4 联网搜索 | 联网搜索]] 模式 `{"mode":"agent","type":"web_search"}`

#### 🐱 对话
- demo_chat.py
- 基于[记忆召回](#-记忆召回机制)中的4种记忆召回模式进行记忆对话。

![default](./picture/default.png)

#### 📃 长文档阅读
- demo_document.py
- 上传文档后，自动转换为长文档阅读模式，启用长文档阅读模型（以glm-4-long为例），第一轮回复为文档摘要，后续回复基于长文档信息进行问答。
- 长文档原文不写入本地聊天记录历史，摘要与后续文档问答以特定标识符\[文档模式]写入聊天历史记录，以防止记忆总结时出现混乱。
- 清除文档、清空聊天记录或在输入框输入“退出文档模式”/“清空聊天记录”指令，即可返回默认的chat_demo模式。

![document01](./picture/document_mode01.png)
![document02](./picture/document_mode02.png)

#### 🏄‍ 联网搜索
- demo_chat.py
- 意图识别模块识别到联网搜索模式。
- 调用智谱web_search_pro进行联网搜索，获取初步返回结果后进行大模型粗加工。
- 粗加工结果以提示工程拼接提交给chat模型进行问答。
- 目前联网搜索仅支持单轮对话，后续将合并至demo_agent.py中成为多轮agent的一部分。

![联网搜索](./picture/web_search_example.png)

![翻译](./picture/translation_example.png)

#### 🔧 自定义工具
- demo_agent.py
- 意图识别模块识别到Agent模式。
- 目前支持工具为web_search、translation，为提高响应速度，web_search下，该模式直接调用智谱的web_search_pro，translation下，直接调用另一套系统提示词；后续将加入更多自定义tools，实现多轮对话下的自定义工具调用。

## 🎫 开源协议
本项目仅用于教育和娱乐目的。如果您希望将该项目用于商业目的，必须获得作者的明确许可。 本项目遵循GNU通用公共许可证（GPL）。请确保仅将其用于学术研究，而不用于商业盈利。未经授权的商业用途是不允许的。任何未经授权的商业使用该项目，用户将承担由此产生的全部后果。

## 📫 联系作者
BryanMurkyChan@gmail.com

## 🔗 引用
如果本项目对您的工作有所帮助，请使用以下格式引用：
```bibtex
@misc{Project_Miao,
    title={Project_Miao},
    author={Wenhui Chen},
    url={https://github.com/BryanMurkyChan/Project_Miao},
    year={2024}
}
```

## 💕 致谢
- [ChatGLM3]([https://github.com/THUDM/ChatGLM3])
- [TinyRAG]([https://github.com/KMnO4-zx/TinyRAG])
