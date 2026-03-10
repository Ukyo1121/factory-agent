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
import base64
import uuid

warnings.filterwarnings("ignore")

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
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated, Sequence
from langgraph.graph import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool
from contextlib import asynccontextmanager
from passlib.context import CryptContext
from app.core.video_manager import load_metadata

# 加载环境变量
from dotenv import load_dotenv
load_dotenv(override=True)
# 定义待解答问题的文件路径
UNANSWERED_FILE = "unanswered_questions.json"
# 定义本地图片存储路径
IMAGES_DIR = "./factory_images"

es_url = "http://localhost:9200"
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
DB_URI = os.getenv("DB_URI", "postgresql://admin:factory_pass@localhost:5432/factory_agent")

# 创建连接池（添加更多配置参数）
pool = AsyncConnectionPool(
    conninfo=DB_URI,
    min_size=1,          # 最小连接数
    max_size=10,         # 最大连接数（降低以避免资源占用）
    open=False,          # 延迟打开
    timeout=10,          # 连接超时时间（秒）
    max_waiting=0,       # 不排队等待连接
    max_lifetime=3600,   # 连接最大生命周期（1小时）
    max_idle=600,        # 空闲连接最大时间（10分钟）
    kwargs={"autocommit": True} # 建议开启 autocommit
)

checkpointer = AsyncPostgresSaver(pool)

# ==============================================================================
# 1. 准备 RAG 引擎
# ==============================================================================

# 配置 Embedding
GLOBAL_EMBED_MODEL = HuggingFaceEmbedding(model_name="models/hub/models--BAAI--bge-m3")

Settings.embed_model = GLOBAL_EMBED_MODEL

Settings.llm = None

# 配置 Reranker (核心竞争力: 重排序)
reranker = FlagEmbeddingReranker(
    model="models/hub/models--BAAI--bge-reranker-base", 
    top_n=15,
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
            es_url=es_url,
            index_name="factory_knowledge",
        )
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

        # RAG Engine
        rag_engine = index.as_query_engine(
            similarity_top_k=20,  # 粗排
            node_postprocessors=[reranker], # 精排
            verbose=True,
            response_mode="no_text"
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

    # 读取旧数据
    data = []
    if os.path.exists(UNANSWERED_FILE):
        try:
            with open(UNANSWERED_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except:
            data = []
    
    # 去重逻辑：检查是否已存在相同的 query
    for item in data:
        # 使用 strip() 去除首尾空格，确保匹配准确
        if item.get("query", "").strip() == user_query.strip():
            print(f"⚠️ [Agent 动作] 发现待解答库中已存在该问题，跳过写入: {user_query}")
            return "该问题已成功记录到待解答问题库，请告知用户工程师将后续补充此知识。"
        
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

    return "该问题已成功记录到待解答问题库，请告知用户工程师将后续补充此知识。"

# 工具列表
tools = [search_factory_knowledge, record_missing_knowledge]

# ==============================================================================
# 多模态处理核心函数
# ==============================================================================
def convert_to_multimodal_messages(messages):
    """
    这是一个中间件函数。
    它的作用是：检查最近一条消息（通常是 ToolMessage），
    如果有 Markdown 图片链接，就把它变成 Base64 发给大模型。
    """
    processed_messages = list(messages)
    last_msg = processed_messages[-1]

    if isinstance(last_msg, ToolMessage) and "![示意图]" in str(last_msg.content):
        text_content = last_msg.content
        new_content_blocks = []
        
        # 正则匹配 Markdown 图片链接
        pattern = rf'!\[.*?\]\(({re.escape(API_BASE_URL)}/images/(.*?))\)'
        last_end = 0
        for match in re.finditer(pattern, text_content):
            start, end = match.span()
            
            # 添加图片前的文字
            if start > last_end:
                text_part = text_content[last_end:start]
                if text_part:
                    new_content_blocks.append({"type": "text", "text": text_part})
            
            img_url = match.group(1)
            filename = match.group(2)
            local_path = os.path.join(IMAGES_DIR, filename)
            
            if os.path.exists(local_path):
                try:
                    # [过滤] 忽略小于 1.5KB 的图标/噪点
                    if os.path.getsize(local_path) < 1500:
                        print(f"⚠️ [中间件] 忽略微型图片: {filename}")
                    else:
                        with open(local_path, "rb") as f:
                            b64_data = base64.b64encode(f.read()).decode("utf-8")
                        
                        new_content_blocks.append({
                            "type": "text", 
                            "text": f"\n[系统提示：图片引用链接 {img_url}]\n"
                        })
                        new_content_blocks.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64_data}"}
                        })
                except Exception as e:
                    print(f"❌ 图片处理异常: {e}")
            
            last_end = end
            
        # 添加剩余文字
        if last_end < len(text_content):
            tail = text_content[last_end:]
            if tail:
                new_content_blocks.append({"type": "text", "text": tail})
                
        last_msg.content = new_content_blocks
        
    return processed_messages

# ==============================================================================
# 3. 构建Agent
# ==============================================================================
llm = ChatOpenAI(
    model="qwen3-vl-plus", 
    openai_api_key=os.getenv('DASHSCOPE_API_KEY'),
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0.1,
    max_tokens=2048,
    model_kwargs={"stream": True} 
)

system_prompt = SystemMessage(content="""
    ### 角色定义
    你是一个严谨专业的工厂智能助手。你将面对以下三种场景，需要执行不同的任务：
    
    **场景1：**用户没有上传文件或图片，且询问工厂设备、操作规程等业务问题
    - 当用户提问故障处理或操作问题时，你需要调用`search_factory_knowledge` 工具查询知识库，根据知识库的内容来回答用户问题。当发现检索内容不足以回答问题时，**必须立即**调用 `record_missing_knowledge`记录问题到【待解答问题库】，绝对不要犹豫或尝试自我纠正。
    
    **场景2：**用户上传了文件或图片
    - 如果用户在提问时**上传了文件或图片**，需要先仔细理解用户的问题，再调用`search_factory_knowledge` 工具查询知识库
      - 如果查询到的内容与用户上传的内容和用户的问题高度相关，则需要根据用户上传的文件或图片，结合知识库中的内容回答用户的问题
      - 如果查询到的内容与用户上传的内容和用户的问题无关，则直接根据用户上传的文件或图片进行作答，**不需要**调用记录工具记录到【待解答问题库】
    
    **场景3：**用户提出代码调试、代码报错类问题（如“Python脚本报错怎么解决”、“这段代码怎么优化”等）
    - 第一步：先调用 `search_factory_knowledge` 查询知识库，看看是否有针对本工厂特定系统的报错记录或解决方案。
    - 第二步：如果查询结果能解决该报错，则直接结合知识库内容回答。
    - 第三步：如果未在知识库中找到相关内容，**请直接发挥你作为代码专家的能力，运用你自身的编程知识**，仔细分析用户的代码报错或需求，给出导致错误的原因，并提供修复后的完整代码和原理解释。
    - 【注意】：此代码场景下查不到资料时，**绝对不要**调用 record_missing_knowledge 工具，直接输出你的代码调试结果即可。
                              
    ### 场景1 详细工作流 (必须严格执行以下步骤)
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
    - **审查图片内容 (视觉能力)**：
      - 你会看到穿插在文字中的图片。
      - **请仔细看图**：判断图片内容是“设备操作示意图/电路图/实物图”还是“无意义的Logo/页眉”。
      - **决策**：只有当图片能辅助说明操作步骤时，才保留它；如果是无关图片，请直接忽略，不要输出。
                              
    **第3步：决策与行动**
    - **情况 A (主语匹配 且 内容相关)**：
      - 对查询的结果进行整合或提取，清晰准确地回答用户。
      - **图文混排规则**：
        - 你的回答必须图文并茂。
        - 引用图片时，请使用 Markdown 格式：`![示意图](图片链接)`。
        - **注意**：只能使用系统提示中给出的 `http://localhost...` 链接，**绝对不要**输出 Base64 编码。
    - **情况 B (主语不匹配 或 查询工具返回“未在知识库中找到相关内容” 或 返回的内容与用户问题的关联性很低，不足以支撑你回答用户的问题)**：
      - **绝对禁止**强行拼凑答案。例如：不要把机器人的操作安在分拣系统头上。
      - **必须**调用 `record_missing_knowledge` 工具，将问题记录到待解答库。
      - 礼貌回复用户：“抱歉，当前知识库中暂未收录此问题。但我已将其自动记录到【待解答问题库】，工程师将在后续更新中补充该内容。”
                              
    ### 注意事项
    1. **完整性**：你输出的内容务必能够**完整**地契合用户的问题，例如用户提问“自动分拣系统的手动操作流程”，查询工具返回的内容只包含“手动操作流程”，但缺少“自动分拣系统”这个关键词，也要视为无法回答用户问题，需要将该问题存入待解答问题库。
    2. **图文对应**：如果查询工具返回内容中的某一步骤有图，你在回答该步骤时就必须带上那张图。不要遗漏。图片应该紧跟在它所解释的步骤文字之后。
    3. **严禁编造**：不允许在查询工具返回的内容上增加无中生有的内容，你只能对查询的结果进行整合或提取，然后清晰地回答用户，**严禁编造**。
    4. **确定性**：如果用户的问题不清晰（例如只说了“机器坏了”），请追问具体的错误码或故障现象等问题的细节，不要瞎猜。
                              
    ### 工具调用格式规范
    **你必须使用标准的 OpenAI Function Calling 格式。**
    **严禁**输出 `<tool_call>`, `<function>` 等 XML 标签。
    **严禁**输出 Base64 编码。
    【严格限制】在输出内容时，如果提供的知识库检索结果中没有包含具体的图片 URL 链接，**严禁**自己捏造、猜测或生成任何 Markdown 格式的图片链接（如 ![图](url)）。
                              
    ### 回答格式
    - 使用清晰的 Markdown 格式。
    - 在回答末尾列出【参考来源文件】。
    """)

# 定义状态
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# 定义节点：调用模型
async def call_model(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]

    # 1. [记录工具防死循环拦截] 保持不变
    if isinstance(last_message, ToolMessage) and "该问题已成功记录到待解答问题库" in str(last_message.content):
        print("🛑 [系统拦截] 检测到刚刚执行了记录工具，强制结束对话循环。")
        return {
            "messages": [
                AIMessage(content="抱歉，当前知识库中暂未收录此问题。我已将其自动记录到【待解答问题库】，工程师将在后续更新中补充该内容。")
            ]
        }

    print("🤖 [Agent 动作] 正在思考 (调用大模型)...")
    
    # 2. 确保 SystemPrompt 在最前
    if not isinstance(messages[0], SystemMessage):
        messages = [system_prompt] + messages
    else:
        messages[0] = system_prompt

    # 3. 执行中间件：处理图片 Base64
    messages_with_images = convert_to_multimodal_messages(messages)
    
    # ==================== [智能检测搜索次数] ====================
    search_count = 0  
    
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if isinstance(msg, HumanMessage):
            break
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc["name"] == "search_factory_knowledge":
                    search_count += 1
    
    print(f"🔍 [系统检测] 本轮对话已搜索次数: {search_count}")

    current_tools = list(tools)
    if search_count >= 3:
        print(f"🛑 [系统强制] 检测到本轮对话已搜索 {search_count} 次，正在移除搜索工具...")
        current_tools = [t for t in tools if t.name != "search_factory_knowledge"]
    
    model_with_tools = llm.bind_tools(current_tools)
    # ====================================================================
    
    try:
        response = await model_with_tools.ainvoke(messages_with_images)
        content_str = str(response.content)
        
        # ==================== [XML 强力修复补丁] ====================
        # 只做格式转换，不再这里做业务拦截逻辑
        if not response.tool_calls and ("<tool_call>" in content_str or "<function=" in content_str):
            print(f"⚠️ [兼容性修复] 检测到 Qwen 返回了 XML...")
            func_pattern = r"<function=['\"]?(\w+)['\"]?>| <function=['\"]?(\w+)['\"]?>"
            func_match = re.search(func_pattern, content_str)
            
            if func_match:
                func_name = func_match.group(1) or func_match.group(2)
                args = {}
                if func_name == "search_factory_knowledge":
                    q_match = re.search(r"<parameter=query>(.*?)</parameter>", content_str, re.DOTALL)
                    if q_match: args["query"] = q_match.group(1).strip()
                        
                elif func_name == "record_missing_knowledge":
                    uq_match = re.search(r"<parameter=user_query>(.*?)</parameter>", content_str, re.DOTALL)
                    if uq_match: args["user_query"] = uq_match.group(1).strip()
                    r_match = re.search(r"<parameter=reason>(.*?)</parameter>", content_str, re.DOTALL)
                    if r_match: args["reason"] = r_match.group(1).strip()
                    else: args["reason"] = "未检索到相关文档"
                
                if args:
                    response.tool_calls = [{
                        "name": func_name,
                        "args": args,
                        "id": f"call_{uuid.uuid4().hex[:8]}"
                    }]
                    response.content = "" 
            
            if "<tool_call>" in str(response.content):
                clean_content = re.sub(r"<tool_call>.*?</tool_call>", "", str(response.content), flags=re.DOTALL)
                response.content = clean_content.strip()

        # ==================== [100% 绝对拦截器] ====================
        # 此时无论上面是原生 JSON 还是 XML 转换的，工具调用都在 response.tool_calls 里
        if response.tool_calls and search_count >= 3:
            for tc in response.tool_calls:
                # 即使模型幻觉强行输出了搜索工具，我们在这里“狸猫换太子”
                if tc["name"] == "search_factory_knowledge":
                    print(f"🛡️ [绝对拦截] 模型试图强行进行第 {search_count + 1} 次搜索，系统强制篡改为‘记录缺失知识’...")
                    tc["name"] = "record_missing_knowledge"
                    # 尝试保留它本想搜的词
                    query_val = tc["args"].get("query", "未知问题")
                    tc["args"] = {
                        "user_query": query_val,
                        "reason": "自动拦截：多次检索（3次）无果，强制转入待解答库"
                    }
        # ==========================================================
        # ==================== [视频预览卡片注入] ====================
        # 当模型没有调用工具，说明这是发给用户的最终回答
        if not response.tool_calls:
            try:
                video_titles = set()
                # 倒序遍历本轮的对话，寻找工具返回的知识库文本
                for msg in reversed(messages):
                    if isinstance(msg, HumanMessage):
                        break  # 只看本轮的
                    if getattr(msg, 'type', '') == 'tool':
                        # 从检索内容中提取《视频名字》
                        import re
                        matches = re.findall(r'来源视频：《(.*?)》', str(msg.content))
                        for m in matches:
                            video_titles.add(m)
                
                if video_titles:
                    videos = load_metadata()
                    video_blocks = []
                    for title in video_titles:
                        # 在视频元数据中寻找匹配的视频
                        matched_video = next((v for v in videos if v.get("title") == title), None)
                        if matched_video:
                            # 剔除不需要的过大字段（如果有的话），转为 JSON
                            video_info = {
                                "id": matched_video.get("id"),
                                "title": matched_video.get("title"),
                                "url": matched_video.get("url"),
                                "thumbnail": matched_video.get("thumbnail"),
                                "duration": matched_video.get("duration")
                            }
                            video_json = json.dumps(video_info, ensure_ascii=False)
                            # 生成前端专属的解析标签
                            video_blocks.append(f"<video_preview>{video_json}</video_preview>")
                    
                    if video_blocks:
                        # 把标签无缝拼接到大模型原回答的末尾
                        appendix = "\n\n---\n**🎬 参考操作演示视频：**\n" + "\n".join(video_blocks)
                        response.content = str(response.content) + appendix

            except Exception as e:
                print(f"⚠️ [增强失败] 视频链接注入出错: {e}")
        # ==========================================================
        print("✅ [Agent 动作] 大模型思考完成")
        return {"messages": [response]}
        
    except Exception as e:
        print(f"❌ [Agent 报错] 模型调用失败: {e}")
        return {"messages": [AIMessage(content=f"模型调用出错: {str(e)}")]}

# 定义边：判断是否结束
def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    
    # 终极防死循环：统计本轮连读调用工具的总次数
    tool_call_count = 0
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if isinstance(msg, HumanMessage):
            break
        if getattr(msg, 'type', '') == 'tool':
            tool_call_count += 1
            
    # 如果系统失控，单轮工具交互超过 5 次，直接拔网线结束图流转
    if tool_call_count >= 5:
        print("🛑 [终极防线] 检测到工具循环超限(5次)，强制拔网线，跳出大模型推理循环！")
        return "__end__"

    # 正常跳转逻辑
    if last_message.tool_calls:
        return "tools"
        
    return "__end__"
    
# --- 构建图 ---
workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(tools))

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "__end__": "__end__"
    }
)
workflow.add_edge("tools", "agent")

# 编译图
graph_structure = workflow

print("🤖 工厂智能Agent已启动！")

# ==============================================================================
# 4. 获取 Graph 实例
# ==============================================================================
# 用于缓存已编译的 graph，避免重复初始化
_compiled_graph = None

async def get_graph():
    """
    返回编译后的 Graph 实例
    """
    global _compiled_graph
    
    # 如果已经编译过，直接返回
    if _compiled_graph is not None:
        return _compiled_graph
    
    print("🏗️  [Graph] 正在编译 StateGraph...")
    
    # 确保连接池已打开
    if pool.closed:
        await pool.open()

    # 编译 Graph
    _compiled_graph = graph_structure.compile(checkpointer=checkpointer)
    
    print("✅ [Graph] 编译完成")
    return _compiled_graph

# =============================================================================
# chat_stream 函数
# =============================================================================

async def chat_stream(message: str, thread_id: str, temp_context: dict = None):
    """
    流式对话接口,支持临时上下文(文件/图片)
    """
    try:
        # 获取 graph
        graph = await get_graph()
        config = {"configurable": {"thread_id": thread_id}}
        has_yielded = False
        
        print(f"💬 [Chat] 开始处理消息: thread_id={thread_id}")

        # 构造增强型消息内容 (处理多文件/多图)
        
        # 基础文本部分
        final_text_content = message
        
        # 收集图片部分 (多模态)
        image_contents = [] 
        
        # 收集文本文件内容
        attached_texts = []

        # 归一化 temp_context:如果是 None 转空列表,如果是单个 dict 转列表
        if temp_context:
            if isinstance(temp_context, dict):
                temp_context = [temp_context]
            
            for ctx in temp_context:
                c_type = ctx.get("type")
                c_content = ctx.get("content")
                c_name = ctx.get("fileName", "未知文件")
                c_saved_path = ctx.get("savedPath")  # 获取保存路径

                if c_type == "text":
                    if c_saved_path:
                        attached_texts.append(f"【参考文件:{c_name}|路径:{c_saved_path}】\n{c_content}")
                    else:
                        attached_texts.append(f"【参考文件:{c_name}】\n{c_content}")
                
                elif c_type == "image":
                    # 记录图片信息时包含保存路径
                    if c_saved_path:
                        # 新格式: 包含路径信息
                        attached_texts.append(f"【参考文件:{c_name}|路径:{c_saved_path}】")
                    else:
                        # 旧格式兼容
                        attached_texts.append(f"【参考文件:{c_name}】")
                    
                    # 累积图片 (构造 LangChain/OpenAI 标准 image_url 格式)
                    image_contents.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{c_content}" 
                        }
                    })

        # 如果有附加的文本文件或图片,拼接到用户 prompt 前面
        if attached_texts:
            joined_files = "\n".join(attached_texts)  
            final_text_content = (
                f"用户上传了以下参考资料:\n"
                f"{joined_files}\n"
                f"---\n"
                f"用户的具体问题是:{message}"
            )

        # 构造最终发给 LLM 的 content
        # 如果有图片,必须使用 list 格式 [{"type": "text", ...}, {"type": "image_url", ...}]
        if image_contents:
            content_to_send = [{"type": "text", "text": final_text_content}] + image_contents
        else:
            # 如果只有文本,直接发字符串即可 (节省 token 且兼容性更好)
            content_to_send = final_text_content

        # 构造 HumanMessage
        input_message = HumanMessage(content=content_to_send)

        # 调用 graph
        async for event in graph.astream_events(
            {"messages": [input_message]}, 
            config=config,
            version="v1"
        ):
            # ------------------------------------------------------
            # 1. 捕获大模型的流式输出 (正常对话)
            # ------------------------------------------------------
            if event["event"] == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                if content:
                    # 过滤 XML 标签
                    if isinstance(content, str) and ("<tool_call>" in content or "<function=" in content): 
                        continue
                    has_yielded = True
                    yield content
            
            # ------------------------------------------------------
            # 2. 捕获大模型非流式结果
            # ------------------------------------------------------
            elif event["event"] == "on_chat_model_end" and not has_yielded:
                output = event["data"]["output"]
                if hasattr(output, "generations") and output.generations:
                    msg = output.generations[0][0].message
                    if isinstance(msg, BaseMessage) and msg.type == "ai" and msg.content:
                        if not msg.tool_calls:
                            has_yielded = True
                            yield msg.content

            # ------------------------------------------------------
            # 3. 捕获 Agent 节点的直接输出
            # ------------------------------------------------------
            elif event["event"] == "on_chain_end" and event["name"] == "agent":
                data = event["data"].get("output")
                if data and isinstance(data, dict) and "messages" in data:
                    last_msg = data["messages"][-1]
                    if (not has_yielded and 
                        isinstance(last_msg, AIMessage) and 
                        last_msg.content):
                        
                        print(f"⚡ [Chat] 捕获到系统拦截消息: {last_msg.content[:20]}...")
                        has_yielded = True
                        yield last_msg.content

    except Exception as e:
        error_msg = f"\n\n❌ 对话处理失败: {str(e)}\n"
        print(f"❌ [Chat] {error_msg}")
        import traceback
        traceback.print_exc()
        yield error_msg