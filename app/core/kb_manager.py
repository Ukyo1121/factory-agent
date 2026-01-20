# app/core/kb_manager.py

import os
import shutil
import fitz  # PyMuPDF
import requests
import nest_asyncio
from typing import List, Dict
from fastapi import UploadFile
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.elasticsearch import ElasticsearchStore
from app.core.agent import GLOBAL_EMBED_MODEL 
# 加载环境变量
from dotenv import load_dotenv
load_dotenv(override=True)


# 应用异步补丁
nest_asyncio.apply()

ES_URL = "http://localhost:9200"
INDEX_NAME = "factory_knowledge"
UPLOAD_DIR = "./factory_docs"       # 存放 PDF 原文
IMAGES_DIR = "./factory_images"     # 存放抠出来的图片

# 确保目录存在
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

# -----------------------------------------------------------
# 1. 辅助函数：本地暴力提取 PDF 图片
# -----------------------------------------------------------
def extract_images_from_pdf(pdf_path, output_dir):
    """
    使用 PyMuPDF 从 PDF 中提取所有图片，并返回图片的文件名列表。
    """
    image_files = []
    try:
        doc = fitz.open(pdf_path)
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        
        print(f"🖼️  开始从 {base_name} 中提取图片...")
        
        for i in range(len(doc)):
            page = doc[i]
            image_list = page.get_images(full=True)
            
            if image_list:
                print(f"    - 第 {i+1} 页发现 {len(image_list)} 张图片")
            
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]  # png 或 jpeg
                
                # 生成唯一文件名: 文件名_页码_图片索引.png
                image_filename = f"{base_name}_p{i+1}_{img_index}.{image_ext}"
                image_filepath = os.path.join(output_dir, image_filename)
                
                # 保存图片到硬盘
                with open(image_filepath, "wb") as f:
                    f.write(image_bytes)
                
                image_files.append(image_filename)
                
        print(f"✅ 图片提取完成，共 {len(image_files)} 张，存入 {output_dir}")
        return image_files
    except Exception as e:
        print(f"❌ 图片提取失败: {e}")
        return []

# -----------------------------------------------------------
# 2. ES 操作函数
# -----------------------------------------------------------
def list_files_in_es() -> List[Dict]:
    search_url = f"{ES_URL}/{INDEX_NAME}/_search"
    payload = {
        "size": 0,
        "aggs": {
            "unique_files": {
                "terms": {
                    "field": "metadata.file_name.keyword",
                    "size": 1000
                }
            }
        }
    }
    try:
        response = requests.get(search_url, json=payload)
        if response.status_code == 200:
            buckets = response.json().get('aggregations', {}).get('unique_files', {}).get('buckets', [])
            return [{"name": b['key'], "chunks": b['doc_count']} for b in buckets]
        return []
    except Exception as e:
        print(f"查询失败: {e}")
        return []

def delete_file_from_es(filename: str) -> bool:
    url = f"{ES_URL}/{INDEX_NAME}/_delete_by_query"
    payload = {
        "query": {
            "term": {
                "metadata.file_name.keyword": filename
            }
        }
    }
    try:
        response = requests.post(url, json=payload)
        return response.status_code == 200
    except Exception as e:
        print(f"删除失败: {e}")
        return False

# -----------------------------------------------------------
# 3. 核心入库逻辑
# -----------------------------------------------------------
async def ingest_file(file: UploadFile):
    # 1. 保存 PDF 原文件
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    print(f"📂 处理文件: {file.filename}")

    # 2. 先进行本地图片提取
    extracted_images = []
    if file.filename.lower().endswith(".pdf"):
        extracted_images = extract_images_from_pdf(file_path, IMAGES_DIR)

    # 3. 配置 LlamaParse
    
    parser = LlamaParse(
        api_key=os.getenv('LLAMA_CLOUD_API_KEY'),
        result_type="markdown",
        language="ch_sim",
        verbose=True,
        premium_mode=False, 
        take_screenshot=False,
        split_by_page=True,
    )
    
    file_extractor = {".pdf": parser, ".docx": parser, ".doc": parser}

    # 4. 使用全局模型
    Settings.embed_model = GLOBAL_EMBED_MODEL
    Settings.chunk_size = 512

    # 5. 读取并解析文字
    documents = SimpleDirectoryReader(
        input_files=[file_path],
        file_extractor=file_extractor
    ).load_data()

    # 6. 精准分配图片到每一页
    if extracted_images:
        print(f"🔗 正在将图片精确匹配到对应页码...")
        
        for doc in documents:
            # LlamaParse 会自动在 metadata 里放入 'page_label' (通常是 "1", "2" 字符串)
            page_label = doc.metadata.get("page_label")
            
            if page_label:
                # 构造匹配特征，例如 "_p1_" (对应第1页)
                # 我们的图片命名格式是: base_name_p{页码}_{索引}.ext
                match_str = f"_p{page_label}_"
                
                # 筛选属于这一页的图片
                page_images = [img for img in extracted_images if match_str in img]
                
                # 只把这一页的图片挂载到当前文档
                if page_images:
                    doc.metadata["image_files"] = page_images
            else:
                # 如果是 Word/Excel 没有页码概念，或者 LlamaParse 没返回页码
                # 可以选择挂载所有图片，或者不挂载
                pass

    # 7. 存入 ES
    vector_store = ElasticsearchStore(
        es_url=ES_URL,
        index_name=INDEX_NAME,
    )
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    print(f"⏳ 准备开始向量化，共有 {len(documents)} 个文档对象等待处理...")
    print("   (这一步需要调用显卡 BGE-M3 模型，如果文档很长，可能需要几分钟，请耐心等待...)")
    try:
        # 这里是最容易卡住的地方
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            show_progress=True  # 开启内置进度条！
        )
        print("✅ 向量化完成！")
    except Exception as e:
        print(f"❌ 向量化过程中出错: {e}")
        raise e

    print("🎉 全量入库完成！")
    
    return len(documents)