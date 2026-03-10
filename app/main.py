# app/main.py
import os
import shutil
import json
import uuid
from typing import Optional
import pandas as pd
import io
from fastapi import FastAPI, UploadFile, File, HTTPException, Form,BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel
from pydantic import BaseModel
from contextlib import asynccontextmanager
import docx
import pandas as pd
import base64
from app.models import ChatRequest
from app.core.kb_manager import list_files_in_es, delete_file_from_es, ingest_file, ingest_from_local_path, UPLOAD_DIR, IMAGES_DIR,parse_pdf_with_layout
from app.core.agent import chat_stream, pool, UNANSWERED_FILE
from app.core import video_manager
from app.core.video_manager import load_metadata,mark_video_extracted
from app.core.image_repo import ImageRepository
from app.core.history_manager import init_database, db_login_user, db_get_user_threads, db_create_thread, db_update_thread_timestamp, db_update_thread_title,db_delete_thread,get_history
from app.core.video_analyzer import extract_knowledge_from_video
import logging
from pathlib import Path

# 配置日志记录器
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

IMAGES_DIR = Path("./factory_images")
COLLECT_IMAGES_DIR = Path("./collect_images")
FILES_DIR = Path("./factory_files") 
VIDEOS_DIR = Path("./training_videos")
if not os.path.exists(FILES_DIR):
    os.makedirs(FILES_DIR)
    
class LoginRequest(BaseModel):
    username: str
    password: str 

class CreateThreadRequest(BaseModel):
    user_id: str
    thread_type: str

class UpdateTitleRequest(BaseModel):
    title: str

try:
    # download_root 可以指定模型下载路径，避免每次都下
    voice_model = WhisperModel("small", device="cpu", compute_type="int8", download_root="./models/whisper")
except Exception as e:
    print(f"语音模型加载失败: {e}")
    voice_model = None
# 定义生命周期管理器
image_repo = None
@asynccontextmanager
async def lifespan(app: FastAPI):
    global image_repo
    # --- 启动时运行 ---
    print("🚀 [System] 正在启动数据库连接池...")
    try:
        await pool.open()
        # 等待连接就绪
        await pool.wait()
        await init_database()
        image_repo = ImageRepository(pool)
        print("✅ [System] 数据库连接池已就绪")
    except Exception as e:
        print(f"❌ [System] 数据库连接失败: {e}")
    
    yield # 服务运行期间保持在这里
    
    # --- 关闭时运行 ---
    print("🛑 [System] 正在关闭数据库连接池...")
    await pool.close()
    print("✅ [System] 数据库连接池已关闭")
# --------------------------------------------------------------------------
# 1. 框架配置
# --------------------------------------------------------------------------
app = FastAPI(title="智能分拣助手 API", version="2.0",lifespan=lifespan)

app.mount("/files", StaticFiles(directory=UPLOAD_DIR), name="files")
app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")
app.mount("/collect_images", StaticFiles(directory=COLLECT_IMAGES_DIR), name="collect_images")
app.mount("/chatfiles", StaticFiles(directory=FILES_DIR), name="chatfiles")

app.mount("/videos", StaticFiles(directory=video_manager.VIDEOS_DIR), name="videos")
app.mount("/thumbnails", StaticFiles(directory=video_manager.THUMBNAILS_DIR), name="thumbnails")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Factory AI Agent Service is Running"}

# --------------------------------------------------------------------------
# 2. 用户登录接口
# --------------------------------------------------------------------------
# 登录接口
@app.post("/login")
async def login(req: LoginRequest):
    result = await db_login_user(req.username, req.password)
    if "error" in result:
        # 返回 HTTP 401 未授权错误
        raise HTTPException(status_code=401, detail=result["error"])
    return result

# 新建会话接口
@app.post("/threads")
async def create_thread_route(req: CreateThreadRequest):
    try:
        # 调用 agent.py 里的数据库插入函数
        return await db_create_thread(user_id=req.user_id, title="新会话", thread_type=req.thread_type)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 获取用户的历史会话列表
@app.get("/threads/{user_id}")
async def get_user_threads_route(user_id: str, thread_type: str = "training"):
    try:
        threads = await db_get_user_threads(user_id,thread_type)
        # threads 结构应为: [{"id": "...", "title": "...", "date": "..."}]
        return threads
    except Exception as e:
        print(f"Error fetching threads: {e}")
        return []

# 更新标题的接口
@app.put("/threads/{thread_id}/title")
async def update_thread_title_route(thread_id: str, req: UpdateTitleRequest):
    try:
        await db_update_thread_title(thread_id, req.title)
        return {"status": "success", "title": req.title}
    except Exception as e:
        print(f"更新标题失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# 删除会话接口
@app.delete("/threads/{thread_id}")
async def delete_thread_route(thread_id: str, user_id: str): 
    # 注意：这里通过 query参数 ?user_id=xxx 传递用户ID进行鉴权
    try:
        success = await db_delete_thread(thread_id, user_id)
        if not success:
            raise HTTPException(status_code=404, detail="会话不存在或无权删除")
        return {"status": "success", "message": "会话已删除"}
    except Exception as e:
        print(f"删除会话失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))
# --------------------------------------------------------------------------
# 3. 核心接口
# --------------------------------------------------------------------------

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """对话接口 (流式)"""
    # 先更新数据库里的时间戳，这样列表排序才会变
    await db_update_thread_timestamp(request.thread_id)
    return StreamingResponse(
        chat_stream(request.query, request.thread_id, request.temp_context),
        media_type="text/event-stream"
    )

@app.post("/voice-to-text")
async def voice_to_text_endpoint(file: UploadFile = File(...)):
    """
    语音转文字接口 (Local Faster-Whisper)
    """
    if not voice_model:
        raise HTTPException(status_code=500, detail="语音模型未加载，请检查后台日志")

    # 1. 保存上传的临时音频文件
    temp_filename = f"temp_{file.filename}"
    try:
        with open(temp_filename, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 2. 调用模型进行识别
        # beam_size=5 提升准确率
        segments, info = voice_model.transcribe(temp_filename, beam_size=5, language="zh")
        
        # 3. 拼接结果
        full_text = "".join([segment.text for segment in segments])
        
        # 4. 删除临时文件
        os.remove(temp_filename)
        
        print(f"🎤 语音识别结果: {full_text}")
        return {"text": full_text}

    except Exception as e:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        print(f"❌ 语音识别出错: {e}")
        raise HTTPException(status_code=500, detail=f"识别失败: {str(e)}")

# --------------------------------------------------------------------------
# 4. 知识库接口
# --------------------------------------------------------------------------
@app.get("/knowledge/files")
def get_files():
    return list_files_in_es()

@app.delete("/knowledge/files/{filename}")
def delete_file(filename: str):
    if delete_file_from_es(filename):
        return {"message": f"{filename} 已删除"}
    raise HTTPException(status_code=500, detail="删除失败")

@app.post("/knowledge/upload")
async def upload_file(file: UploadFile = File(...)):
    print(f"\n📥 [Debug] 收到文件上传请求: filename={file.filename}, content_type={file.content_type}")
    
    # 1. 检查 python-multipart 是否正常工作
    try:
        file_size = 0
        # 尝试读取一点点数据，测试 UploadFile 对象是否健康
        content_sample = await file.read(1024)
        file_size = len(content_sample)
        await file.seek(0) # 读完记得指针归位
        print(f"✅ [Debug] 文件对象读取正常，前1KB已读取。")
    except Exception as e:
        print(f"❌ [Debug] UploadFile 对象损坏 (可能是 python-multipart 未安装或版本不兼容): {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="文件接收失败，请检查服务器 python-multipart 依赖")

    try:
        # 2. 调用核心逻辑
        print("🚀 [Debug] 正在调用 kb_manager.ingest_file ...")
        num = await ingest_file(file)
        print(f"✅ [Debug]入库成功，片段数: {num}")
        return {"message": "入库成功", "chunks": num}
    except Exception as e:
        # 3. 捕获核心逻辑的所有报错，并打印堆栈！
        print(f"❌ [Debug] 知识库入库过程发生致命错误: {type(e).__name__} - {e}")
        print("-" * 60)
        import traceback
        traceback.print_exc()  # <--- 这行代码会把真正的报错打印在黑色窗口里
        print("-" * 60)
        raise HTTPException(status_code=500, detail=f"后端处理失败: {str(e)}")

@app.get("/admin/unanswered_questions")
def get_unanswered_questions():
    """
    获取所有待解答的问题列表
    """
    if not os.path.exists(UNANSWERED_FILE): return {"count": 0, "questions": []}
    try:
        with open(UNANSWERED_FILE, "r", encoding="utf-8") as f: data = json.load(f)
        pending = [q for q in data if q.get("status") == "pending"]
        return {"count": len(pending), "questions": pending}
    except: return {"count": 0, "questions": []}

@app.post("/admin/solve_question")
async def solve_question(
    query: str = Form(...),
    answer_text: Optional[str] = Form(None),
    custom_filename: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
):
    """
    解决问题：接收人工回答（文字或文件），生成文档入库，并更新状态
    """
    # A. 校验
    if not answer_text and not file:
        raise HTTPException(status_code=400, detail="必须提供文字回答或上传文件")

    try:
        # B. 处理回答并入库
        ingested_filename = ""
        
        # 情况1：上传了文件 (PDF/Word等)
        if file:
            file_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # 入库
            await ingest_from_local_path(file_path, file.filename)
            ingested_filename = file.filename

        # 情况2：纯文字回答 (生成一个 .txt 文件)
        elif answer_text:
            # 确定文件名
            if custom_filename and custom_filename.strip():
                # 使用用户自定义的文件名
                safe_name = custom_filename.strip()
                # 自动补全 .txt 后缀
                if not safe_name.lower().endswith(".txt"):
                    safe_name += ".txt"
                txt_filename = safe_name
            else:
                # 默认逻辑：生成带随机ID的文件名
                short_id = str(uuid.uuid4())[:8]
                txt_filename = f"人工解答_{short_id}.txt"
            
            txt_path = os.path.join(UPLOAD_DIR, txt_filename)
            
            # 写入内容：明确的问题和答案格式
            content = f"【故障/问题】\n{query}\n\n【解决方案】\n{answer_text}"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(content)
            
            # 入库
            await ingest_from_local_path(txt_path, txt_filename)
            ingested_filename = txt_filename

        # C. 更新 JSON 状态
        if os.path.exists(UNANSWERED_FILE):
            with open(UNANSWERED_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            found = False
            for item in data:
                if item["query"] == query and item["status"] == "pending":
                    item["status"] = "solved"
                    item["solved_at"] = "now" # 简化处理
                    item["solution_source"] = ingested_filename
                    found = True
                    break
            
            # 写回
            with open(UNANSWERED_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        return {"message": "处理成功，知识已入库", "file": ingested_filename}

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    
# --------------------------------------------------------------------------
# 5. 零件生命周期数据可视化接口
# --------------------------------------------------------------------------

@app.post("/api/upload_lifecycle")
async def upload_lifecycle_data(file: UploadFile = File(...)):
    """
    接收 CSV 或 Excel 文件，解析为 JSON 数据供前端可视化使用
    """
    try:
        contents = await file.read()
        filename = file.filename.lower()
        
        df = None
        
        # --- 分支 1: 处理 Excel (.xlsx) ---
        if filename.endswith(".xlsx") or filename.endswith(".xls"):
            # read_excel 需要二进制流 (BytesIO)，不需要 decode
            df = pd.read_excel(io.BytesIO(contents))
            
        # --- 分支 2: 处理 CSV (.csv) ---
        else:
            # read_csv 需要文本流，尝试 utf-8 和 gbk 解码
            try:
                df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
            except UnicodeDecodeError:
                df = pd.read_csv(io.StringIO(contents.decode('gbk')))
        
        # --- 通用数据清洗逻辑 ---
        # 确保数值列是数字类型，如果为空则填0
        numeric_cols = ['总耗时(分钟)', '坐标 X', '坐标 Y']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # 填充空字符串，防止前端报错
        df = df.fillna("")

        # 将 DataFrame 转为字典列表返回
        data = df.to_dict(orient="records")
        return {"data": data, "count": len(data)}
        
    except Exception as e:
        print(f"文件解析错误: {e}")
        return {"error": f"解析失败: {str(e)}"}


# ==========================================================================
# 6. 培训视频管理接口 (新增)
# ==========================================================================

@app.get("/training-videos")
def get_training_videos():
    """获取所有培训视频列表"""
    try:
        return video_manager.list_videos()
    except Exception as e:
        print(f"获取视频列表失败: {e}")
        raise HTTPException(status_code=500, detail="获取视频列表失败")

@app.post("/training-videos/upload")
async def upload_training_video(file: UploadFile = File(...)):
    """
    上传培训视频
    """
    print(f"\n📹 [Debug] 收到视频上传请求: filename={file.filename}, content_type={file.content_type}")
    
    try:
        # 调用核心逻辑保存视频
        video_metadata = video_manager.save_video(file.file, file.filename)
        
        print(f"✅ [Debug] 视频上传成功: {video_metadata['title']}")
        return JSONResponse(
            status_code=200,
            content={"message": "视频上传成功", "video": video_metadata}
        )
        
    except ValueError as e:
        # 业务逻辑错误（如文件格式不支持、文件过大）
        print(f"❌ [Debug] 视频上传被拒绝: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        # 系统错误
        print(f"❌ [Debug] 视频上传失败: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"上传失败: {str(e)}")

@app.get("/training-videos/{video_id}")
def get_training_video(video_id: str):
    """获取单个视频的元数据"""
    try:
        video = video_manager.get_video_by_id(video_id)
        if video:
            return video
        raise HTTPException(status_code=404, detail="视频不存在")
    except HTTPException:
        raise
    except Exception as e:
        print(f"获取视频信息失败: {e}")
        raise HTTPException(status_code=500, detail="获取视频信息失败")

@app.delete("/training-videos/{video_id}")
def delete_training_video(video_id: str):
    """删除视频"""
    try:
        if video_manager.delete_video(video_id):
            return JSONResponse(
                status_code=200,
                content={"message": "视频删除成功"}
            )
        raise HTTPException(status_code=404, detail="视频不存在")
    except HTTPException:
        raise
    except Exception as e:
        print(f"删除视频失败: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"删除失败: {str(e)}")

@app.put("/training-videos/{video_id}")
async def update_training_video_metadata(
    video_id: str, 
    title: Optional[str] = Form(None), 
    description: Optional[str] = Form(None)
):
    """更新视频元数据（标题、描述）"""
    try:
        if video_manager.update_video_metadata(video_id, title, description):
            return JSONResponse(
                status_code=200,
                content={"message": "更新成功"}
            )
        raise HTTPException(status_code=404, detail="视频不存在")
    except HTTPException:
        raise
    except Exception as e:
        print(f"更新视频信息失败: {e}")
        raise HTTPException(status_code=500, detail=f"更新失败: {str(e)}")

@app.post("/training-videos/{video_id}/extract")
async def extract_video_knowledge(video_id: str, background_tasks: BackgroundTasks):
    videos = load_metadata()
    video_info = next((v for v in videos if v["id"] == video_id), None)
    
    if not video_info:
        raise HTTPException(status_code=404, detail="视频未找到")
        
    video_path = os.path.join(VIDEOS_DIR, video_info["filename"])
    if not os.path.exists(video_path):
        raise HTTPException(status_code=404, detail="视频文件不存在")

    # 定义后台任务函数
    async def process_task():
        success = await extract_knowledge_from_video(video_path, video_info.get("title", "未命名操作视频"))
        if success:
            mark_video_extracted(video_id)

    # 视频处理极其耗时，放入后台执行，立即给前端返回响应
    background_tasks.add_task(process_task)
    
    return {"status": "success", "message": "视频分析任务已提交后台处理"}

@app.get("/history/{thread_id}")
async def fetch_history(thread_id: str):
    """
    前端加载页面时调用此接口，恢复历史记录
    """
    try:
        history = await get_history(thread_id)
        return {"history": history}
    except Exception as e:
        print(f"获取历史记录失败: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
    
@app.post("/upload-temp-file")
async def upload_temp_file(file: UploadFile = File(...)):
    """
    处理临时上传的文件，提取文本内容返回给前端
    """
    ext = file.filename.split(".")[-1].lower()
    content = ""
    
    try:
        # 1. 处理 PDF
        if ext == "pdf":
            # A. 保存文件
            saved_filename = f"doc_{uuid.uuid4().hex}.pdf"
            saved_path = os.path.join(FILES_DIR, saved_filename)
            
            file_data = await file.read()
            with open(saved_path, "wb") as f:
                f.write(file_data)
            
            # B. 解析文本
            docs = parse_pdf_with_layout(saved_path, file.filename)
            text_content = "\n".join([doc.text for doc in docs])
            
            # C. 返回 JSON (包含文本和路径)
            return JSONResponse({
                "type": "text",            # 前端识别为文本类文件
                "content": text_content,   # 解析出的文字内容
                "fileName": file.filename, # 原文件名
                "savedPath": saved_filename # 返回保存的文件名
            })

        # --------------------------------------------------------------------------
        # 2. 处理 Word
        # --------------------------------------------------------------------------
        elif ext in ["doc", "docx"]:
            saved_filename = f"doc_{uuid.uuid4().hex}.{ext}"
            saved_path = os.path.join(FILES_DIR, saved_filename)
            
            file_data = await file.read()
            with open(saved_path, "wb") as f:
                f.write(file_data)

            doc = docx.Document(saved_path)
            text_content = "\n".join([para.text for para in doc.paragraphs])
            
            return JSONResponse({
                "type": "text",
                "content": text_content,
                "fileName": file.filename,
                "savedPath": saved_filename
            })

        # --------------------------------------------------------------------------
        # 3. 处理 Excel
        # --------------------------------------------------------------------------
        elif ext in ["xls", "xlsx"]:
            saved_filename = f"doc_{uuid.uuid4().hex}.{ext}"
            saved_path = os.path.join(FILES_DIR, saved_filename)
            
            file_data = await file.read()
            with open(saved_path, "wb") as f:
                f.write(file_data)

            df = pd.read_excel(saved_path)
            text_content = df.to_csv(index=False)
            
            return JSONResponse({
                "type": "text",
                "content": text_content,
                "fileName": file.filename,
                "savedPath": saved_filename
            })
        # --------------------------------------------------------------------------
        # 4. 处理图片 (Base64 编码) - 保持原逻辑不变
        # --------------------------------------------------------------------------
        
        elif ext in ["jpg", "jpeg", "png"]:
            file_data = await file.read()
            base64_image = base64.b64encode(file_data).decode('utf-8')
            image_filename = f"chat_{uuid.uuid4().hex}.{ext}"
            image_path = os.path.join(IMAGES_DIR, image_filename)
            with open(image_path, "wb") as f:
                f.write(file_data)
        
            return JSONResponse({
                "type": "image",
                "content": base64_image,
                "fileName": file.filename,
                "savedPath": image_filename 
            })

        return JSONResponse({
            "type": "text",
            "content": content,
            "fileName": file.filename
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件解析失败: {str(e)}")

# 图片采集
@app.post("/collect/upload")
async def upload_and_annotate_image(
    file: UploadFile = File(...),
    annotation: str = Form(...) # 使用 Form 来接收同一次请求中的文本数据 Let users provide a text annotation
):
    """
    上传图片并保存标注信息到数据库。
    """
    if not image_repo:
        raise HTTPException(status_code=503, detail="Image repository not initialized")

    try:
        # 1. 保存文件到磁盘 (复用原有的上传目录)
        file_path = COLLECT_IMAGES_DIR / file.filename
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"Collected image saved to: {file_path}")

        # 2. 生成相对路径供前端访问
        # 注意：这里存的是 URL 路径部分，不是操作系统的绝对路径
        relative_url_path = f"collect_images/{file.filename}"

        # 3. 保存元数据和标注到数据库
        image_id = await image_repo.save_image_annotation(
            filename=file.filename,
            file_path=relative_url_path,
            annotation=annotation
        )

        return JSONResponse(content={
            "message": "Image and annotation collected successfully",
            "id": image_id,
            "file_path": relative_url_path
        }, status_code=200)

    except Exception as e:
        logger.error(f"Error during image collection: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/collect/list")
async def list_collected_images():
    """
    获取最近采集的图片列表。
    """
    if not image_repo:
        raise HTTPException(status_code=503, detail="Image repository not initialized")
    
    images = await image_repo.get_recent_images(limit=30)
    return JSONResponse(content={"images": images}, status_code=200)

@app.delete("/collect/image/{image_id}")
async def delete_collected_image(image_id: int):
    """
    删除采集的图片（包含数据库记录和物理文件）
    """
    if not image_repo:
        raise HTTPException(status_code=503, detail="Image repository not initialized")

    try:
        # 1. 从数据库中删除记录，并获取文件的相对路径
        # 注意：这里调用的是我们在 image_repo.py 中新增的 delete_image 方法
        file_path = await image_repo.delete_image(image_id)
        
        if not file_path:
            raise HTTPException(status_code=404, detail="Image not found or already deleted")

        # 2. 删除服务器上的物理文件
        # 从 file_path (例如 "collect_images/xxx.jpg") 中提取出文件名
        from pathlib import Path
        filename = Path(file_path).name 
        physical_path = COLLECT_IMAGES_DIR / filename
        
        # 检查物理文件是否存在，存在则删除
        if physical_path.exists():
            physical_path.unlink() 
            logger.info(f"Deleted physical image file: {physical_path}")
        else:
            logger.warning(f"Physical file not found for deletion: {physical_path}")

        return JSONResponse(content={"message": "Image deleted successfully"}, status_code=200)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during image deletion: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))