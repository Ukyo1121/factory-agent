# app/main.py
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from app.models import ChatRequest
from app.core.agent import chat_stream  # 导入刚才封装的流式函数
from fastapi import UploadFile, File, HTTPException
from app.core.kb_manager import list_files_in_es, delete_file_from_es, ingest_file
from fastapi.staticfiles import StaticFiles # 导入 StaticFiles
from app.core.kb_manager import UPLOAD_DIR # 导入刚才定义的目录变量
from app.core.kb_manager import IMAGES_DIR

# 框架搭建: 初始化 FastAPI
app = FastAPI(title="智能问答模块 API", version="1.0")
app.mount("/files", StaticFiles(directory=UPLOAD_DIR), name="files")
app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")

# 配置 CORS (允许前端跨域访问，为阶段三做准备)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境建议修改为具体的前端域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Factory AI Agent Service is Running"}

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    与 AI 助手对话的接口，支持流式输出。
    """
    user_query = request.query
    thread_id = request.thread_id

    # 实现流式响应
    # 将 LangGraph 的生成器包装成 StreamingResponse
    return StreamingResponse(
        chat_stream(user_query, thread_id),
        media_type="text/event-stream"  # 这种格式便于前端逐字渲染
    )

@app.get("/knowledge/files")
def get_files():
    """获取知识库文件列表"""
    return list_files_in_es()

@app.delete("/knowledge/files/{filename}")
def delete_file(filename: str):
    """删除指定文件"""
    success = delete_file_from_es(filename)
    if not success:
        raise HTTPException(status_code=500, detail="删除失败")
    return {"message": f"{filename} 已删除"}

@app.post("/knowledge/upload")
async def upload_file(file: UploadFile = File(...)):
    """上传文件并进行 RAG 入库"""
    try:
        num_chunks = await ingest_file(file)
        return {"message": "入库成功", "filename": file.filename, "chunks": num_chunks}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))