# app/core/video_manager.py

import os
import json
import shutil
from datetime import datetime
from typing import List, Optional, Dict
import uuid

# 配置目录
VIDEOS_DIR = "./training_videos"
THUMBNAILS_DIR = "./training_thumbnails"
METADATA_FILE = "./training_videos_metadata.json"

# 确保目录存在
os.makedirs(VIDEOS_DIR, exist_ok=True)
os.makedirs(THUMBNAILS_DIR, exist_ok=True)

# 支持的视频格式
ALLOWED_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB

# -----------------------------------------------------------
# 辅助函数
# -----------------------------------------------------------

def load_metadata() -> List[Dict]:
    """加载视频元数据"""
    if not os.path.exists(METADATA_FILE):
        return []
    try:
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"加载视频元数据失败: {e}")
        return []

def save_metadata(videos: List[Dict]):
    """保存视频元数据"""
    try:
        with open(METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(videos, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"保存视频元数据失败: {e}")
        raise Exception("保存元数据失败")

def generate_thumbnail(video_path: str, thumbnail_path: str) -> bool:
    """
    使用ffmpeg生成视频缩略图（可选功能）
    需要系统安装了ffmpeg
    """
    try:
        import subprocess
        subprocess.run([
            'ffmpeg', '-i', video_path, '-ss', '00:00:01.000',
            '-vframes', '1', thumbnail_path, '-y'
        ], check=True, capture_output=True, timeout=30)
        return True
    except Exception as e:
        print(f"生成缩略图失败（这不影响核心功能）: {e}")
        return False

def get_video_duration(video_path: str) -> Optional[float]:
    """
    使用ffprobe获取视频时长（可选功能）
    需要系统安装了ffmpeg
    """
    try:
        import subprocess
        result = subprocess.run([
            'ffprobe', '-v', 'error', '-show_entries',
            'format=duration', '-of',
            'default=noprint_wrappers=1:nokey=1', video_path
        ], capture_output=True, text=True, check=True, timeout=10)
        return float(result.stdout)
    except Exception as e:
        print(f"获取视频时长失败（这不影响核心功能）: {e}")
        return None

# -----------------------------------------------------------
# 核心业务逻辑函数
# -----------------------------------------------------------

def list_videos() -> Dict:
    """获取所有视频列表"""
    videos = load_metadata()
    return {"videos": videos, "total": len(videos)}

def get_video_by_id(video_id: str) -> Optional[Dict]:
    """根据ID获取单个视频信息"""
    videos = load_metadata()
    for video in videos:
        if video.get("id") == video_id:
            return video
    return None

def save_video(file_content, original_filename: str) -> Dict:
    """
    保存视频文件并创建元数据
    
    参数:
        file_content: 文件内容（file-like对象）
        original_filename: 原始文件名
    
    返回:
        视频元数据字典
    """
    # 验证文件扩展名
    file_ext = os.path.splitext(original_filename)[1].lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"不支持的文件格式。支持: {', '.join(ALLOWED_EXTENSIONS)}")
    
    # 生成唯一ID和文件名
    video_id = str(uuid.uuid4())
    safe_filename = f"{video_id}{file_ext}"
    video_path = os.path.join(VIDEOS_DIR, safe_filename)
    
    # 保存视频文件
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file_content, buffer)
    
    # 获取文件大小
    file_size = os.path.getsize(video_path)
    
    # 验证文件大小
    if file_size > MAX_FILE_SIZE:
        os.remove(video_path)
        raise ValueError("文件大小超过500MB限制")
    
    # 尝试生成缩略图
    thumbnail_filename = f"{video_id}.jpg"
    thumbnail_path = os.path.join(THUMBNAILS_DIR, thumbnail_filename)
    thumbnail_generated = generate_thumbnail(video_path, thumbnail_path)
    
    # 尝试获取视频时长
    duration = get_video_duration(video_path)
    
    # 创建元数据
    video_metadata = {
        "id": video_id,
        "title": os.path.splitext(original_filename)[0],
        "description": None,
        "filename": safe_filename,
        "url": f"/videos/{safe_filename}",
        "thumbnail": f"/thumbnails/{thumbnail_filename}" if thumbnail_generated else None,
        "size": file_size,
        "duration": duration,
        "uploadedAt": datetime.now().isoformat()
    }
    
    # 更新元数据列表
    videos = load_metadata()
    videos.append(video_metadata)
    save_metadata(videos)
    
    return video_metadata

def delete_video(video_id: str) -> bool:
    """
    删除视频文件和元数据
    
    参数:
        video_id: 视频ID
    
    返回:
        是否删除成功
    """
    videos = load_metadata()
    
    # 查找要删除的视频
    video_to_delete = None
    for video in videos:
        if video.get("id") == video_id:
            video_to_delete = video
            break
    
    if not video_to_delete:
        return False
    
    try:
        # 删除视频文件
        video_path = os.path.join(VIDEOS_DIR, video_to_delete["filename"])
        if os.path.exists(video_path):
            os.remove(video_path)
        
        # 删除缩略图
        if video_to_delete.get("thumbnail"):
            thumbnail_path = os.path.join(
                THUMBNAILS_DIR, 
                os.path.basename(video_to_delete["thumbnail"])
            )
            if os.path.exists(thumbnail_path):
                os.remove(thumbnail_path)
        
        # 更新元数据
        videos = [v for v in videos if v.get("id") != video_id]
        save_metadata(videos)
        
        return True
        
    except Exception as e:
        print(f"删除视频失败: {e}")
        return False

def update_video_metadata(video_id: str, title: Optional[str] = None, description: Optional[str] = None) -> bool:
    """
    更新视频元数据（标题、描述）
    
    参数:
        video_id: 视频ID
        title: 新标题（可选）
        description: 新描述（可选）
    
    返回:
        是否更新成功
    """
    videos = load_metadata()
    
    updated = False
    for video in videos:
        if video.get("id") == video_id:
            if title:
                video["title"] = title
            if description is not None:
                video["description"] = description
            updated = True
            break
    
    if not updated:
        return False
    
    save_metadata(videos)
    return True

def mark_video_extracted(video_id: str) -> bool:
    """标记视频已经被提取过知识"""
    videos = load_metadata()
    for v in videos:
        if v.get("id") == video_id:
            v["is_extracted"] = True
            save_metadata(videos)
            return True
    return False