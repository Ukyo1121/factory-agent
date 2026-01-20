import os
import sys
import nest_asyncio
nest_asyncio.apply()

import warnings
import logging
import asyncio
import torch
import re

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
    :return: 返回查询的结果和来源文件，包含文本和对应图片链接的结构化结果。
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
            similarity_top_k=15,  # 粗排
            node_postprocessors=[reranker], # 精排
            verbose=True
        )
        # 调用 LlamaIndex 的 RAG 引擎
        response = rag_engine.query(query)
        final_context_list = []
        
        if hasattr(response, 'source_nodes'):
            for node in response.source_nodes:
                text = node.text
                fname = node.metadata.get('file_name', '未知文件')
                page = node.metadata.get('page_label', '未知页')
                
                # 直接信任 metadata
                # 因为 kb_manager 已经帮我们把图片按页分好了
                images = node.metadata.get("image_files", [])
                
                # 组装文本
                context_str = f"--- 来源: {fname} (第 {page} 页) ---\n{text}\n"
                
                # 组装图片 (紧跟在文本后面)
                # 这样 LLM 读到这段话时，马上就能看到图片，从而在生成答案时实现图文穿插
                if images and isinstance(images, list):
                    context_str += "\n[该段落关联参考图]:\n"
                    for img_name in images:
                        img_url = f"http://localhost:8000/images/{img_name}"
                        context_str += f"![示意图]({img_url})\n"
                
                final_context_list.append(context_str)

        final_response = "\n\n".join(final_context_list)
        
        if not final_response:
            return "未在知识库中找到相关内容。"

        # Debug 打印
        if "![示意图]" in final_response:
             print("✅ [Debug] 成功检测到关联图片，已注入上下文")
        
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

# 工具列表
tools = [search_factory_knowledge]

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
    你是一个专业的工厂智能助手。
    1. 遇到故障问题或操作问题，必须优先使用 'search_factory_knowledge' 工具查询知识库。
    2. 如果查询工具返回了解决方法或操作步骤，请清晰地转述给用户，并告诉用户参考来源文件的名称。
    3. 如果工具返回的内容中包含图片链接（Markdown格式如 ![](http://...)），请务必在回答的对应位置原样展示这些图片，不要忽略它们，也不要修改链接地址。图片对于用户理解操作步骤非常重要。
    4. 不允许在查询工具返回的内容上增加无中生有的内容，你只能对查询的结果进行整合并清晰地回答用户。
    5. 如果查询结果不足以支撑你回答用户的问题或与用户的问题关联性很小，请你诚实回答查询不到相关结果，或追问用户问题的细节。
    6. 如果用户的问题不清晰（例如只说了“机器坏了”），请追问具体的错误码或故障现象，不要瞎猜。
    7.【排版严格要求】：
       - 工具会返回多段文本，每段文本后面可能会附带“[该段落关联的图片]”。
       - 请在回答时，将图片**穿插**在对应的文字描述之后。
       - 例如：先解释“第一步：按下复位按钮”，紧接着立刻展示复位按钮的图片，然后再说“第二步...”。
       - **绝对不要**把所有图片都堆在回答的最后面。
       - 图片链接必须保持原样 Markdown 格式 `![](http://...)`。
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
    
    # 使用 LangGraph 的 astream 方法
    # 注意：这里根据你 LangGraph 的具体版本，API 可能是 .stream 或 .astream
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