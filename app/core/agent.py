import os
import sys
import nest_asyncio
nest_asyncio.apply()

import warnings
import logging
import asyncio
import torch
import re
import json
import datetime

warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # 只显示严重错误
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1" # 屏蔽 Windows 下的符号链接警告
logging.getLogger("langchain").setLevel(logging.ERROR)
logging.getLogger("langgraph").setLevel(logging.ERROR)

from typing import Annotated, Literal, TypedDict

# --- LlamaIndex 依赖 (用于 RAG) ---
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.elasticsearch import ElasticsearchStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.llms.openai_like import OpenAILike

# --- LangGraph & LangChain 依赖 (用于 Agent) ---
from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# 加载环境变量
from dotenv import load_dotenv
load_dotenv(override=True)
# 定义待解答问题的文件路径
UNANSWERED_FILE = "unanswered_questions.json"

# ==============================================================================
# 1. 准备 RAG 引擎
# ==============================================================================

# 配置 Embedding
GLOBAL_EMBED_MODEL = HuggingFaceEmbedding(model_name="BAAI/bge-m3")

Settings.embed_model = GLOBAL_EMBED_MODEL

# 配置llm
Settings.llm = OpenAILike(
        model="deepseek-chat",
        api_key=os.getenv('DEEPSEEK_API_KEY'),
        api_base="https://api.deepseek.com",
        is_chat_model=True,
        context_window=32768,
        temperature=0.1,
        max_tokens=1024
    )

# 配置 Reranker (核心竞争力: 重排序)
reranker = FlagEmbeddingReranker(
    model="BAAI/bge-reranker-base", 
    top_n=5,
    use_fp16=True  # 必须开启半精度，进一步省显存
)

# ==============================================================================
# 2. 定义 Agent 的工具 (Tool)
# ==============================================================================

@tool
def search_factory_knowledge(query: str) -> str:
    """
    当用户询问工厂设备故障、错误码、维修步骤或操作规程时，必须调用此工具进行查询。
    重要提示:query 参数必须是完整的中文问题句子，不要随意对用户的问题进行概括、不要提取关键词。
    :param query: 必要参数，字符串类型，用于输入用户的具体问题。
    :return: 返回查询的结果和来源文件，包含图文混排内容。
    """
    print(f"\n🔍 [Agent 动作] 正在调用知识库查询: {query}")
    vector_store = None
    try:
        # 连接 ES 数据库
        vector_store = ElasticsearchStore(
            es_url="http://localhost:9200",
            index_name="factory_knowledge",
        )
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

        # RAG Engine
        rag_engine = index.as_query_engine(
            similarity_top_k=10,  # 粗排
            node_postprocessors=[reranker], # 精排
            verbose=True
        )
        # 调用 LlamaIndex 的 RAG 引擎
        response = rag_engine.query(query)

        # ---------------------------------------------------------
        # 1. 排序：先按文件名，再按页码
        # ---------------------------------------------------------
        node_data = []
        if hasattr(response, 'source_nodes'):
            for node in response.source_nodes:
                page_str = node.metadata.get('page_label', '0')
                try:
                    page_num = int(page_str)
                except ValueError:
                    page_num = 0
                
                node_data.append({
                    "text": node.text,
                    "file_name": node.metadata.get('file_name', '未知文件'),
                    "page_label": page_num
                })

        sorted_nodes = sorted(node_data, key=lambda x: (x['file_name'], x['page_label']))

        # ---------------------------------------------------------
        # 2. 拼接：构建连续的上下文流
        # ---------------------------------------------------------
        final_context_list = []
        current_file = None
        
        for item in sorted_nodes:
            # 如果换文件了，加一个明显的大标题
            if item['file_name'] != current_file:
                final_context_list.append(f"\n\n====== 文件: {item['file_name']} (开始) ======\n")
                current_file = item['file_name']
            
            # 使用更紧凑的分页标记，并在标记中提示 LLM 注意跨页连接
            # 我们故意在分页符前后少加换行，让 LLM 感觉这是一篇连续的文章
            context_str = f"\n{item['text']}"
            final_context_list.append(context_str)

        final_response = "".join(final_context_list) # 使用空字符串连接，更紧凑
        
        if not final_response.strip():
            return "未在知识库中找到相关内容。"

        # Debug
        print("✅ [Debug] 已按页码重排检索结果")
        print("内容预览：", final_response) # 调试时可开启
        
        return final_response
    except Exception as e:
        print(f"❌ 详细错误: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return f"查询出错: {e}"
    
    finally:
        # 显式关闭 Elasticsearch 客户端连接
        if vector_store is not None:
            try:
                # 关闭 ES 客户端
                if hasattr(vector_store, 'client'):
                    asyncio.get_event_loop().run_until_complete(
                        vector_store.client.close()
                    )
            except Exception as e:
                pass  # 忽略关闭时的错误

@tool
def record_missing_knowledge(user_query: str, reason: str = "未检索到相关文档") -> str:
    """
    当 'search_factory_knowledge' 工具无法在知识库中找到答案，或者检索到的内容与用户问题不匹配时，
    **必须**调用此工具将问题记录到待解答库中。
    :param user_query: 用户的原始问题。
    :param reason: 记录原因（例如：知识库无结果、结果不相关）。
    :return: 返回记录成功的提示。
    """
    print(f"\n📝 [Agent 动作] 正在记录缺失知识: {user_query}")
    
    # 构造记录数据
    record = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "query": user_query,
        "reason": reason,
        "status": "pending" # pending=待人工处理, solved=已入库
    }

    # 读取旧数据并追加
    data = []
    if os.path.exists(UNANSWERED_FILE):
        try:
            with open(UNANSWERED_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except:
            data = []
    
    data.append(record)

    # 写入文件
    with open(UNANSWERED_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return "该问题已成功记录到‘待解答库’，请告知用户工程师将后续补充此知识。"

# 工具列表
tools = [search_factory_knowledge, record_missing_knowledge]

# ==============================================================================
# 3. 构建Agent
# ==============================================================================

llm = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key=os.getenv('DEEPSEEK_API_KEY'),
    openai_api_base="https://api.deepseek.com",
    temperature=0.1
)

system_prompt = SystemMessage(content="""
    ### 角色定义
    你是一个严谨专业的工厂智能助手。你的任务是根据知识库的内容，回答用户的故障处理或操作问题。

    ### 核心工作流 (必须严格执行以下步骤)
    
    **第1步：提取关键实体**
    - 分析用户问题，提取核心设备/系统名称（例如：“自动分拣系统”、“FANUC机器人”、“传送带”）。
    - 记住这个核心实体，它是本次回答的“主语”。

    **第2步：查询并审查 (关键一步)**
    - 调用 `search_factory_knowledge` 查询知识库。
    - **审查查询结果的主语**：
      - 仔细阅读查询到的每一段文字，寻找其中提到的设备名称。
      - **匹配检查示例**：
        - 用户问：“自动分拣系统” -> 查询内容：“机器人手动操作...” -> **不匹配！** (这是张冠李戴)
        - 用户问：“自动分拣系统” -> 查询内容：“分拣单元操作...” -> **匹配。**
        - 用户问：“自动分拣系统” -> 查询内容完全没提设备名，只说“按下红色按钮” -> **高风险！** 除非你能从上下文（如文件名）确信这是分拣系统，否则视为不匹配。

    **第3步：决策与行动**
    - **情况 A (主语匹配 且 内容相关)**：
      - 对查询的结果进行整合或提取，清晰准确地回答用户。
    - **情况 B (主语不匹配 或 查询工具返回“未在知识库中找到相关内容” 或 返回的内容与用户问题的关联性很低，不足以支撑你回答用户的问题)**：
      - **绝对禁止**强行拼凑答案。例如：不要把机器人的操作安在分拣系统头上。
      - **必须**调用 `record_missing_knowledge` 工具，将问题记录到待解答库。
      - 礼貌回复用户：“抱歉，当前知识库中暂未收录此问题。但我已将其自动记录到【待解答问题库】，工程师将在后续更新中补充该内容。”
                              
    ### 注意事项
    1. 你输出的内容务必能够**完整**地契合用户的问题，例如用户提问“自动分拣系统的手动操作流程”，查询工具返回的内容只包含“手动操作流程”，但缺少“自动分拣系统”这个关键词，也要视为无法回答用户问题，需要将该问题存入待解答问题库。
    2. 如果查询工具返回内容中的某一步骤有图，你在回答该步骤时就必须带上那张图。不要遗漏。
    3. 不允许在查询工具返回的内容上增加无中生有的内容，你只能对查询的结果进行整合或提取，然后清晰地回答用户，**严禁编造**。
    4. 如果用户的问题不清晰（例如只说了“机器坏了”），请追问具体的错误码或故障现象等问题的细节，不要瞎猜。

    ### 图文匹配逻辑
    如果查询工具返回的内容中包含 Markdown 格式的图片链接（如 `![示意图](http://...)`）。你必须严格遵守以下规则：
    你的输出必须遵循**“先说文字，后配图”**的模式。严格尊重查询工具返回的内容的原有顺序。
   **Step 1. 输出文字**
    - 当你决定引用某一段操作步骤或描述文字时，先输出这段文字。

    **Step 2. 寻找配图 (向后查找)**
    - 输出完文字后，请立刻看文字的**后面**紧跟着的内容。如果这段文字紧后面跟着一张或多张图片 `![示意图](...)`，这些图就是该文字的配图。**必须立刻输出这些图**。
    - **绝对禁止**：查询工具返回的内容中，若某一图片在某一段文字之后，你在输出的时候禁止先输出这张图片再输出这段文字

    **Step 3. 循环**
    - 图片输出完毕后，继续输出下一段文字，重复上述步骤。

    ### 输出示例
    **查询工具返回的结果**:
    > 1. 打开系统变量菜单...
    > ![p3_0.jpg]
    > 2. 点击确认按钮...

    **错误输出 (绝对禁止)**:
    > ![p3_0.jpg]  <-- 错误！莫名其妙先出图
    > 1. 打开系统变量菜单...
    > 2. 点击确认按钮...

    **正确输出**:
    > 1. 打开系统变量菜单...
    > ![p3_0.jpg]  <-- 正确！严格按照
    >
    > 2. 点击确认按钮...
                              
    ### 回答格式
    - 使用清晰的 Markdown 格式。
    - 在回答末尾列出【参考来源文件】。
    """)

# 在生产环境中，这里可以换成 PostgresSaver 或 SqliteSaver 来持久化存入硬盘
memory = MemorySaver()

graph = create_react_agent(
    model=llm, 
    tools=tools, 
    prompt=system_prompt,
    checkpointer=memory
)

print("🤖 工厂智能Agent已启动！")

# 封装一个异步生成器函数，用于流式输出
async def chat_stream(message: str, thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}

    # 使用 LangGraph 的 astream_events 方法监听所有事件
    async for event in graph.astream_events(
        {"messages": [("user", message)]}, 
        config=config,
        version="v1"
    ):
        # 过滤只返回 LLM 生成的文本内容，跳过中间步骤的日志
        if event["event"] == "on_chat_model_stream":
             content = event["data"]["chunk"].content
             if content:
                 yield content

# ==============================================================================
# 4. 交互式运行
# ==============================================================================
def main():
    print("\n你可以开始提问了 (输入 'q' 退出)")
    
    # 定义线程 ID，LangGraph 通过这个 ID 来区分不同的对话历史
    # 如果你想开启一段全新的对话（忘记过去），只需要换一个 ID (例如 "thread_2")
    config = {"configurable": {"thread_id": "factory_user_001"}}
    
    while True:
        user_input = input("\n请提问: ")
        if user_input.lower() == 'q':
            break
            
        print("\n[Agent 思考中...]")

        # 我们只把当前最新的这一句话传给 Agent
        # Agent 会根据 config 里的 thread_id 自动去 memory 里查找之前的聊天记录
        inputs = {"messages": [("user", user_input)]}
        
        # stream_mode="values" 会返回当前时刻完整的消息列表（包含历史）
        # 我们只打印最后一条新增的消息
        for event in graph.stream(inputs, config=config, stream_mode="values"):
            last_message = event["messages"][-1]
            
            # 这里的逻辑是：只打印 AI 新生成的回复
            if last_message.type == "ai" and last_message.content:
                print(f"\n[助手回答]: {last_message.content}")

if __name__ == "__main__":
    main()