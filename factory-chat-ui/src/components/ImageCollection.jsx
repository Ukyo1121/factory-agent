import React, { useState, useEffect, useRef } from 'react';
import {
    Upload, Type, Image as ImageIcon, Save, RefreshCw, ArrowLeft, Trash2, Download
} from 'lucide-react';
import { API_BASE_URL } from '../config';

const ImageCollection = ({ onBack }) => {
    const [selectedFile, setSelectedFile] = useState(null);
    const [previewUrl, setPreviewUrl] = useState(null);
    const [annotation, setAnnotation] = useState('');
    const [isUploading, setIsUploading] = useState(false);
    const [collectedImages, setCollectedImages] = useState([]);
    const fileInputRef = useRef(null);

    useEffect(() => {
        fetchCollectedImages();
    }, []);

    useEffect(() => {
        if (!selectedFile) {
            setPreviewUrl(null);
            return;
        }
        const objectUrl = URL.createObjectURL(selectedFile);
        setPreviewUrl(objectUrl);
        return () => URL.revokeObjectURL(objectUrl);
    }, [selectedFile]);

    const fetchCollectedImages = async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/collect/list`);
            if (response.ok) {
                const data = await response.json();
                setCollectedImages(data.images || []);
            }
        } catch (error) {
            console.error("Failed to fetch collected images:", error);
        }
    };

    const handleFileChange = (e) => {
        if (e.target.files && e.target.files.length > 0) {
            setSelectedFile(e.target.files[0]);
        }
    };

    const handleDelete = async (id) => {
        // 弹窗确认，防止误删
        if (!window.confirm("确定要删除这张图片吗？删除后将无法恢复。")) {
            return;
        }

        try {
            const response = await fetch(`${API_BASE_URL}/collect/image/${id}`, {
                method: 'DELETE',
            });

            if (response.ok) {
                // 删除成功后，直接从当前列表中过滤掉被删的图片，无需重新请求整个列表
                setCollectedImages(prev => prev.filter(img => img.id !== id));
            } else {
                const errorData = await response.json();
                alert(`删除失败: ${errorData.detail}`);
            }
        } catch (error) {
            console.error("Error deleting image:", error);
            alert("删除过程中发生错误");
        }
    };

    const handleExport = async () => {
        try {
            const response = await fetch(`${API_BASE_URL}/collect/export`);
            if (response.ok) {
                const data = await response.json();
                const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `collected_images_${new Date().toISOString().slice(0, 10)}.json`;
                a.click();
                URL.revokeObjectURL(url);
            } else {
                alert("导出失败");
            }
        } catch (error) {
            console.error("Export failed:", error);
            alert("导出过程中发生错误");
        }
    };

    const handleUpload = async () => {
        if (!selectedFile || !annotation.trim()) {
            alert("请选择图片并填写标注内容");
            return;
        }

        setIsUploading(true);
        const formData = new FormData();
        formData.append('file', selectedFile);
        formData.append('annotation', annotation);

        try {
            const response = await fetch(`${API_BASE_URL}/collect/upload`, {
                method: 'POST',
                body: formData,
            });

            if (response.ok) {
                // 重置表单
                setSelectedFile(null);
                setAnnotation('');
                if (fileInputRef.current) fileInputRef.current.value = '';
                // 刷新列表
                fetchCollectedImages();
                alert("采集成功！");
            } else {
                const errorData = await response.json();
                alert(`采集失败: ${errorData.detail}`);
            }
        } catch (error) {
            console.error("Error uploading:", error);
            alert("上传过程中发生错误");
        } finally {
            setIsUploading(false);
        }
    };

    return (
        <div className="flex flex-col h-screen bg-gray-100 p-4 overflow-hidden">
            {/* 顶部标题栏 - 保持与主界面一致的风格 */}
            <header className="flex justify-between items-center mb-4 bg-white p-4 rounded-lg shadow-sm shrink-0">
                <div className="flex items-center gap-4">
                    {/* 返回按钮 */}
                    <button
                        onClick={onBack}
                        className="p-2 hover:bg-gray-100 rounded-full transition-colors text-gray-600"
                        title="返回主页"
                    >
                        <ArrowLeft size={24} />
                    </button>

                    <h1 className="text-2xl font-bold text-gray-800 flex items-center gap-2">
                        <ImageIcon className="w-8 h-8 text-emerald-600" /> {/* 改成绿色配合卡片 */}
                        图片采集标注库
                    </h1>
                </div>

                <div className="flex items-center gap-2">
                    <button
                        onClick={handleExport}
                        className="p-2 text-gray-600 hover:text-emerald-600 hover:bg-emerald-50 rounded-full transition-colors"
                        title="导出 JSON"
                    >
                        <Download size={20} />
                    </button>
                    <button
                        onClick={fetchCollectedImages}
                        className="p-2 text-gray-600 hover:text-emerald-600 hover:bg-emerald-50 rounded-full transition-colors"
                        title="刷新列表"
                    >
                        <RefreshCw size={20} />
                    </button>
                </div>
            </header>

            <div className="flex flex-1 gap-4 overflow-hidden">
                {/* 左侧：上传和标注区域 */}
                <div className="w-1/3 bg-white p-6 rounded-lg shadow-sm flex flex-col overflow-y-auto">
                    <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
                        <Upload size={20} /> 新增采集
                    </h2>

                    {/* 图片预览与选择区域 */}
                    <div
                        className={`flex flex-col items-center justify-center border-2 border-dashed rounded-lg h-64 mb-4 cursor-pointer transition-colors relative bg-gray-50 hover:bg-gray-100 ${previewUrl ? 'border-blue-400' : 'border-gray-300'}`}
                        onClick={() => fileInputRef.current.click()}
                    >
                        {previewUrl ? (
                            <img src={previewUrl} alt="Preview" className="h-full w-full object-contain rounded-lg" />
                        ) : (
                            <div className="text-center text-gray-500">
                                <ImageIcon size={40} className="mx-auto mb-2 opacity-50" />
                                <p>点击选择图片</p>
                                <p className="text-sm opacity-70">支持 JPG, PNG</p>
                            </div>
                        )}
                        <input
                            type="file"
                            ref={fileInputRef}
                            onChange={handleFileChange}
                            accept="image/*"
                            className="hidden"
                        />
                    </div>

                    {/* 标注文本输入区域 */}
                    <div className="flex-1 flex flex-col min-h-[150px]">
                        <label className="block text-sm font-medium text-gray-700 mb-2 flex items-center gap-1">
                            <Type size={16} /> 图片标注 (必填)
                        </label>
                        <textarea
                            className="flex-1 w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
                            placeholder="请描述这张图片的内容、用途或关键信息..."
                            value={annotation}
                            onChange={(e) => setAnnotation(e.target.value)}
                        />
                    </div>

                    {/* 提交按钮 */}
                    <button
                        onClick={handleUpload}
                        disabled={isUploading || !selectedFile || !annotation.trim()}
                        className={`mt-4 py-3 px-6 rounded-lg text-white font-medium flex items-center justify-center gap-2 transition-colors ${isUploading || !selectedFile || !annotation.trim()
                            ? 'bg-gray-400 cursor-not-allowed'
                            : 'bg-green-600 hover:bg-green-700 shadow-md'
                            }`}
                    >
                        {isUploading ? <RefreshCw className="animate-spin" size={20} /> : <Save size={20} />}
                        {isUploading ? "正在入库..." : "保存到采集库"}
                    </button>
                </div>

                {/* 右侧：已采集图片展示区域 */}
                <div className="flex-1 bg-white p-6 rounded-lg shadow-sm overflow-y-auto">
                    <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
                        <ImageIcon size={20} /> 最近采集
                    </h2>
                    <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
                        {collectedImages.map((img) => (
                            <div key={img.id} className="relative border rounded-lg overflow-hidden shadow-sm hover:shadow-md transition-shadow bg-gray-50 flex flex-col group">
                                {/* 删除按钮 */}
                                <button
                                    onClick={() => handleDelete(img.id)}
                                    className="absolute top-2 right-2 p-1.5 bg-red-500 text-white rounded-md opacity-0 group-hover:opacity-100 transition-opacity duration-200 z-10 hover:bg-red-600 shadow-sm"
                                    title="删除图片"
                                >
                                    <Trash2 size={16} />
                                </button>
                                {/* 图片缩略图区域 */}
                                <div className="h-48 bg-gray-200 flex items-center justify-center overflow-hidden">
                                    <img
                                        src={`${API_BASE_URL}/${img.file_path}`}
                                        alt={img.filename}
                                        className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-300" // 可选：鼠标悬停图片微微放大
                                        onError={(e) => { e.target.onerror = null; e.target.src = "https://via.placeholder.com/300x200?text=Image+Load+Error" }}
                                    />
                                </div>
                                {/* 标注信息区域 */}
                                <div className="p-4 flex-1 flex flex-col">
                                    <p className="text-gray-800 text-sm mb-3 flex-1 whitespace-pre-wrap break-words" style={{ maxHeight: '100px', overflowY: 'auto' }}>{img.annotation}</p>
                                    <div className="flex justify-between items-center text-xs text-gray-500 mt-auto pt-2 border-t">
                                        <span className="truncate pr-2" title={img.filename}>{img.filename}</span>
                                        <span className="whitespace-nowrap">{new Date(img.created_at).toLocaleDateString()}</span>
                                    </div>
                                </div>
                            </div>
                        ))}
                        {collectedImages.length === 0 && (
                            <div className="col-span-full text-center text-gray-500 py-10">
                                暂无采集数据
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
};

export default ImageCollection;