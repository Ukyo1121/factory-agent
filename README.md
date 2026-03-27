# 智能分拣助手 (Factory Agent)

## 项目简介

智能分拣助手是一个面向工业分拣场景的 AI 智能平台，集成了 RAG 知识检索、多模态对话、生产日志分析、培训视频知识提取、图片采集等核心功能。系统基于 LangGraph Agent 架构构建，支持 SSE 流式响应，前端采用 React + Tailwind CSS 构建现代化交互界面。

项目路径：/data/suzhiling/factory-agent

---

## 1. 技术栈

### 后端

| 技术 | 版本 | 用途 |
|------|------|------|
| Python | 3.11 | 主语言 |
| FastAPI | 0.128 | REST API 框架 |
| Uvicorn | 0.40 | ASGI 服务器 |
| LangGraph | 1.0.7 | Agent 状态图编排 |
| LangChain | 1.2.3 | LLM 工具链 |
| LlamaIndex | 0.14.12 | RAG 检索框架 |
| Elasticsearch | 8.11.1 | 向量数据库 |
| PostgreSQL | - | 用户数据、会话管理、LangGraph 状态持久化 |
| faster-whisper | 1.2.1 | 语音转文字 |
| PyMuPDF | 1.26.7 | PDF 解析（布局感知） |
| OpenCV / Pillow | - | 图像处理、缩略图生成 |
| sse-starlette | 2.1.3 | SSE 流式推送 |
| passlib (bcrypt) | - | 用户认证 |

**AI 模型：**

- **本地 LLM：** Qwen2.5-VL-7B-Instruct/Qwen2-VL-72B-Instruct-AWQ (INT4 量化版)（通过 vLLM 部署，OpenAI 兼容接口 `localhost:8080`）
- **云端 LLM：** Qwen3-VL-Plus（通过 DashScope API，用于视频分析）
- **Embedding：** `BAAI/bge-m3`（HuggingFace 本地模型）
- **Reranker：** `BAAI/bge-reranker-base`（Top-15，fp16）

### 补充：模型选型评估与运行配置

**一、 Qwen2.5-VL-7B 模型效果欠佳**
* **参数预算与多模态能力的博弈：** 7B 级别模型的参数总量受限，作为视觉语言（VL）模型，其训练重心高度倾斜于图像与视频解析。这导致其用于复杂逻辑推理和严格格式输出（如 Function Calling 的 JSON 规范）的表征能力被严重挤压，面对高难度 Agent 任务易退化为普通对话模式。
* **长上下文注意力衰减（长 Prompt 理解受限）：** 现有的 Agent 框架（如 LangChain）会在底层封装大量详尽的工具说明与系统指令。7B 模型在处理此类超长且高密度的 System Prompt 时，指令遵循度会显著下降（出现“注意力涣散”），容易忽略后台的工具调用约束而进行自由发挥。

**二、 Qwen2.5-VL-72B 大模型当前运行瓶颈**
* **算力不足：** 72B 模型的庞大参数量对推理引擎（vLLM）提出了极高要求。当前显卡的空闲显存远低于该规模模型权重加载与 KV Cache 初始化所需的最低物理显存阈值，系统无法分配足够的连续内存块，直接导致进程启动失败。

**三、模型部署与业务层调用规范**

若后续开发中需要使用本地部署的大模型，需要进行以下配置

**1. 后端模型推理服务启动命令（以 7B 为例）**
*(注：若后续部署 72B 模型，需额外增加分布式多卡并行参数)*
```bash
CUDA_VISIBLE_DEVICES=0 vllm serve ./models/Qwen2.5-VL-7B-Instruct \
  --served-model-name qwen2.5-vl-7b-instruct \
  --host 0.0.0.0 --port 8080 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.9 \
  --trust-remote-code \
  --limit-mm-per-prompt '{"image": 10, "video": 1}' \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```

**2. 业务层 Agent 引擎实例配置 (`修改agent.py中构建Agent的llm部分的代码`)**
```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="qwen2.5-vl-7b-instruct",               # 必须和刚才启动命令里的 served-model-name 一致
    openai_api_key="EMPTY",                       # 本地调用不需要真实的 Key
    openai_api_base="http://127.0.0.1:8080/v1",   # 指向你本地刚刚启动的 8080 端口
    temperature=0.1,
    max_tokens=2048,
    model_kwargs={"stream": True} 
)
```

### 前端

| 技术 | 版本 | 用途 |
|------|------|------|
| React | 19.2 | UI 框架 |
| Vite | 7.2 | 构建工具 |
| Tailwind CSS | 3.4 | 样式框架 |
| Recharts | 3.6 | 数据可视化（饼图、柱状图） |
| react-markdown | 10.1 | Markdown 渲染 |
| lucide-react | 0.562 | 图标库 |

### 基础设施

| 组件 | 说明 |
|------|------|
| Docker Compose | 多服务编排（ES、Kibana、后端、前端） |
| PostgreSQL (Docker) | 数据持久化 |
| Elasticsearch (Docker) | 向量检索 |
| Kibana (Docker) | ES 可视化管理 |
| Systemd (用户级) | 后端/前端进程管理 |

---

## 2. 项目目录结构

```
factory-agent/
├── app/                              # 后端应用
│   ├── main.py                       # FastAPI 主入口，30+ REST 端点
│   ├── models.py                     # Pydantic 请求模型
│   └── core/                         # 核心业务模块
│       ├── agent.py                  # LangGraph Agent（RAG + 工具调用）
│       ├── history_manager.py        # 数据库初始化、用户认证、会话管理
│       ├── kb_manager.py             # 知识库管理（PDF 解析、ES 索引）
│       ├── log_core.py               # 生产日志解析引擎（正则 + OpenCV）
│       ├── video_analyzer.py         # 视频 SOP 知识提取（DashScope）
│       ├── video_manager.py          # 培训视频 CRUD 管理
│       └── image_repo.py             # 图片采集数据库操作
│
├── factory-chat-ui/                  # 前端应用
│   ├── src/
│   │   ├── main.jsx                  # React 入口
│   │   ├── App.jsx                   # 主路由（登录 + 模块选择）
│   │   ├── config.js                 # API 地址配置
│   │   ├── TrainingAssistant.jsx     # 培训助手（AI 对话）
│   │   ├── DebugAssistant.jsx        # 调试助手（AI 对话）
│   │   ├── TrainingVideoManager.jsx  # 培训视频管理
│   │   └── components/
│   │       ├── LifecycleDashboard.jsx # 生产监测看板（日志分析）
│   │       ├── KnowledgeModal.jsx     # 知识库文件管理
│   │       ├── UnansweredModal.jsx    # 待解答问题管理
│   │       └── ImageCollection.jsx    # 图片采集库
│   ├── package.json
│   ├── vite.config.js
│   ├── tailwind.config.js
│   ├── Dockerfile.frontend           # 前端 Docker 镜像（Node 构建 + Nginx）
│   └── index.html
│
├── factory_docs/                     # 知识库上传文档（PDF/Word/Excel/TXT）
├── factory_images/                   # 文档中提取的图片
├── factory_files/                    # 上传附件
├── collect_images/                   # 采集/标注图片
├── training_videos/                  # 培训视频文件
├── training_thumbnails/              # 视频缩略图
├── logtest/                          # 生产日志数据
│   ├── 2026-01-21/                   # 按日期组织的日志批次
│   │   ├── *.log                     # 原始日志文件
│   │   ├── report_*.xlsx             # 分析报告
│   │   └── .ui_thumbs_cache/         # 套料图缩略图缓存
│   └── VISUALNESTING/                # 套料图源文件目录
│
├── models/                           # 本地 AI 模型缓存
├── .env                              # 环境变量（API Key、数据库连接）
├── docker-compose.yml                # Docker 服务编排
├── Dockerfile.backend                # 后端 Docker 镜像
├── requirements.txt                  # Python 依赖（215 个包）
├── ui_config.json                    # 日志分析引擎配置
├── training_videos_metadata.json     # 视频元数据
├── unanswered_questions.json         # 待解答问题记录
├── debug_thumbs.py                   # 缩略图重新生成工具脚本
└── README.md
```

---

## 3. 功能模块

### 3.1 培训助手

- **智能问答：** 基于 LangGraph Agent + RAG 的多轮对话，支持工具调用
- **知识检索：** 通过 Elasticsearch 向量检索工厂文档，结合 Reranker 重排序
- **多模态输入：** 支持文字、图片、文件上传、语音输入
- **SSE 流式响应：** 实时逐字输出，支持 Markdown 渲染
- **会话管理：** 多会话历史，支持搜索和删除
- **文档上传：** 支持 PDF、Word、Excel、TXT 格式，自动向量化入库
- **待解答归档：** 记录 AI 无法回答的问题，支持人工补充答案

**Agent 机制：**
- 基于 LangGraph StateGraph 构建，支持 `call_model` + `ToolNode` 循环
- 两个工具：`search_factory_knowledge`（RAG 检索）和 `record_missing_knowledge`（记录缺失知识）
- 防循环保护：每轮最多 3 次检索、5 次工具调用
- 状态持久化：通过 `AsyncPostgresSaver` 存储到 PostgreSQL

### 3.2 调试助手

- 与培训助手相同的对话架构，独立的会话类型（`thread_type='debug'`）
- 针对机器故障排查、代码异常分析等调试场景优化

### 3.3 生产监测

- **日志解析：** 基于正则表达式解析工厂 `.log` 文件，识别 10+ 种日志模式
- **套料图展示：** 从 VISUALNESTING 目录匹配套料图，通过 OpenCV 生成彩色缩略图
- **KPI 看板：** 总零件数、正常/异常/警告数量统计
- **饼图分析：** 总耗时分布 + 4 个工位耗时分布，支持下钻筛选
- **零件详情：** 点击零件卡片查看完整时间线
- **Excel 导出：** 一键导出分析报告

### 3.4 培训视频库

- **视频管理：** 上传、播放、删除培训视频
- **SOP 知识提取：** 通过 DashScope Qwen3-VL-Plus API 从视频中提取结构化操作步骤
- **知识入库：** 提取的 SOP 自动存入 Elasticsearch 知识库
- **Word 报告：** 自动生成 Word 格式的 SOP 文档

### 3.5 数据采集中心

- **主动录入：** 上传 PDF/Word/Excel 文档到知识库
- **待解答归档：** 查看和处理 AI 无法回答的问题
- **图片采集库：** 上传和标注工业现场图片

---

## 4. 主要 API 端点

| 方法 | 路径 | 功能 |
|------|------|------|
| POST | `/login` | 用户登录（bcrypt 认证） |
| POST | `/chat` | SSE 流式 AI 对话 |
| POST | `/voice-to-text` | 语音转文字 |
| GET | `/threads/{user_id}` | 获取用户会话列表 |
| POST | `/threads` | 创建新会话 |
| DELETE | `/threads/{thread_id}` | 删除会话及历史 |
| GET | `/history/{thread_id}` | 获取会话历史 |
| POST | `/knowledge/upload` | 上传知识库文档 |
| GET | `/knowledge/files` | 获取已索引文件列表 |
| DELETE | `/knowledge/files/{filename}` | 删除知识库文件 |
| GET | `/admin/unanswered_questions` | 获取待解答问题列表 |
| POST | `/admin/solve_question` | 标记问题已解决 |
| GET | `/api/log_dates` | 获取可用日志日期列表 |
| GET | `/api/log_analyze` | 分析指定日期的日志数据 |
| GET | `/api/log_config` | 获取/保存日志路径配置 |
| GET | `/api/log_export` | 导出 Excel 分析报告 |
| GET | `/api/log_thumbs/{date}/{file}` | 获取套料图缩略图 |
| GET | `/training-videos` | 获取培训视频列表 |
| POST | `/training-videos/upload` | 上传培训视频 |
| POST | `/training-videos/{id}/extract` | 提取视频 SOP 知识 |
| POST | `/collect/upload` | 上传采集图片 |
| GET | `/collect/list` | 获取采集图片列表 |

---

## 5. 部署与环境准备

### 5.1 环境要求

- Linux 服务器（推荐 Ubuntu 20.04+）
- Docker & Docker Compose
- Python 3.11+
- Node.js 18+（前端构建）
- NVIDIA GPU（可选，用于本地 LLM 推理）

### 5.2 数据库部署

项目依赖 PostgreSQL 和 Elasticsearch，均通过 Docker 部署。

**启动 PostgreSQL：**

```bash
docker start factory-db
```

验证运行状态：

```bash
docker ps | grep factory-db
```

**启动 Elasticsearch 和 Kibana：**

```bash
docker start factory_es factory_kibana
```

验证运行状态：

```bash
docker ps | grep elastic
```

> Elasticsearch 默认端口 `9200`，Kibana 默认端口 `5601`。

### 5.3 环境变量配置

在项目根目录创建 `.env` 文件：

```env
DASHSCOPE_API_KEY='your_dashscope_api_key'
LANGSMITH_TRACING=true
LANGSMITH_API_KEY='your_langsmith_api_key'
LANGSMITH_PROJECT=factory_agent
LLAMA_CLOUD_API_KEY='your_llama_cloud_api_key'
API_BASE_URL=http://your_server_ip:8000
DB_URI="postgresql://admin:factory_pass@localhost:5432/factory_agent?sslmode=disable"
```

### 5.4 Python 依赖安装

```bash
pip install -r requirements.txt
```

### 5.5 前端构建

```bash
cd factory-chat-ui
npm install
npm run build
```

### 5.6 服务管理（Systemd 用户级）

项目已部署为 Systemd 用户级服务，使用以下命令管理：

**后端服务：**

```bash
# 重启后端
systemctl --user restart factory_agent

# 查看后端运行状态
systemctl --user status factory_agent

# 查看后端实时日志
journalctl --user -u factory_agent -f

# 停止后端
systemctl --user stop factory_agent
```

**前端服务：**

```bash
# 重启前端
systemctl --user restart factory-chat-ui

# 查看前端运行状态
systemctl --user status factory-chat-ui

# 查看前端实时日志
journalctl --user -u factory-chat-ui -f

# 停止前端
systemctl --user stop factory-chat-ui
```

### 5.7 完整启动流程

按以下顺序启动所有服务：

```bash
# 1. 启动数据库
docker start factory-db
docker start factory_es factory_kibana

# 2. 确认数据库正常
docker ps | grep -E "factory-db|elastic"

# 3. 启动后端
systemctl --user restart factory_agent

# 4. 启动前端
systemctl --user restart factory-chat-ui

# 5. 验证服务
systemctl --user status factory_agent
systemctl --user status factory-chat-ui
```

## 6. 默认账号

| 字段 | 值 |
|------|------|
| 用户名 | `admin` |
| 密码 | `admin123` |
| 数据库连接 | `postgresql://admin:factory_pass@localhost:5432/factory_agent` |

---

## 7. 数据库表结构

系统启动时自动创建以下表：

| 表名 | 说明 |
|------|------|
| `users` | 用户账号（user_id, username, password_hash, created_at） |
| `user_threads` | 会话管理（thread_id, user_id, title, thread_type, updated_at） |
| `collected_images` | 采集图片（id, filename, file_path, annotation, created_at） |
| `checkpoints` | LangGraph 状态检查点 |
| `checkpoint_blobs` | LangGraph 状态数据 |
| `checkpoint_writes` | LangGraph 写入记录 |
