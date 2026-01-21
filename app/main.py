# app/main.py
import os
import shutil
import json
import uuid
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel

from app.models import ChatRequest
from app.core.agent import chat_stream, UNANSWERED_FILE
from app.core.kb_manager import list_files_in_es, delete_file_from_es, ingest_file, ingest_from_local_path, UPLOAD_DIR, IMAGES_DIR

# --------------------------------------------------------------------------
# 1. 初始化本地语音模型 (Faster-Whisper)
# --------------------------------------------------------------------------
# 为了防止显存(VRAM)溢出，强制使用 "cpu" 和 "int8" 量化
# "small" 模型对中文识别效果很好，且在 CPU 上运行速度也很快
print("🎤 正在加载本地语音模型 (faster-whisper-small)...")
try:
    # download_root 可以指定模型下载路径，避免每次都下
    voice_model = WhisperModel("small", device="cpu", compute_type="int8", download_root="./models/whisper")
    print("✅ 语音模型加载完成！")
except Exception as e:
    print(f"❌ 语音模型加载失败: {e}")
    voice_model = None

# --------------------------------------------------------------------------
# 2. 框架配置
# --------------------------------------------------------------------------
app = FastAPI(title="工厂智能助手 API", version="1.0")

app.mount("/files", StaticFiles(directory=UPLOAD_DIR), name="files")
app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")

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
# 3. 核心接口
# --------------------------------------------------------------------------

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """对话接口 (流式)"""
    return StreamingResponse(
        chat_stream(request.query, request.thread_id),
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
    try:
        num = await ingest_file(file)
        return {"message": "入库成功", "chunks": num}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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