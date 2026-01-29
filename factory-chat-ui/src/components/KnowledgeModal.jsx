import { useState, useEffect } from 'react';
import { X, Trash2, Upload, FileText, RefreshCw, Loader2 } from 'lucide-react';
import { API_BASE_URL } from "../config";
const API_BASE = API_BASE_URL;

// 定义点击打开文件的函数
const handleOpenFile = (filename) => {
    // 使用 encodeURIComponent 处理中文文件名和空格
    const fileUrl = `${API_BASE}/files/${encodeURIComponent(filename)}`;
    // 在新标签页打开
    window.open(fileUrl, '_blank');
};

export default function KnowledgeModal({ isOpen, onClose }) {
    const [files, setFiles] = useState([]);
    const [loading, setLoading] = useState(false);
    const [uploading, setUploading] = useState(false);

    // 加载文件列表
    const fetchFiles = async () => {
        setLoading(true);
        try {
            const res = await fetch(`${API_BASE}/knowledge/files`);
            const data = await res.json();
            setFiles(data);
        } catch (err) {
            console.error("加载失败", err);
        } finally {
            setLoading(false);
        }
    };

    // 每次打开弹窗时刷新列表
    useEffect(() => {
        if (isOpen) fetchFiles();
    }, [isOpen]);

    // 删除文件
    const handleDelete = async (filename) => {
        if (!confirm(`确定要删除 "${filename}" 吗？这会清除相关的所有知识。`)) return;

        try {
            await fetch(`${API_BASE}/knowledge/files/${filename}`, { method: 'DELETE' });
            fetchFiles(); // 刷新列表
        } catch (err) {
            alert("删除失败");
        }
    };

    // 上传文件
    const handleUpload = async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        setUploading(true);
        const formData = new FormData();
        formData.append('file', file);

        try {
            const res = await fetch(`${API_BASE}/knowledge/upload`, {
                method: 'POST',
                body: formData
            });
            if (!res.ok) throw new Error("上传失败");
            const data = await res.json();
            alert(`成功入库！生成了 ${data.chunks} 个知识片段`);
            fetchFiles(); // 刷新
        } catch (err) {
            alert("上传处理失败，请检查后端日志");
            console.error(err);
        } finally {
            setUploading(false);
            e.target.value = ''; // 清空 input
        }
    };

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
            <div className="bg-white w-[600px] max-h-[80vh] rounded-xl shadow-2xl flex flex-col overflow-hidden animate-fade-in">

                {/* 标题栏 */}
                <div className="p-4 border-b flex justify-between items-center bg-gray-50">
                    <h2 className="text-lg font-bold text-gray-700 flex items-center gap-2">
                        <span className="bg-blue-100 p-1.5 rounded-lg text-blue-600"><FileText size={20} /></span>
                        知识库管理
                    </h2>
                    <button onClick={onClose} className="p-1 hover:bg-gray-200 rounded-full transition">
                        <X size={20} className="text-gray-500" />
                    </button>
                </div>

                {/* 文件列表区 */}
                <div className="flex-1 overflow-y-auto p-4 space-y-2 bg-gray-50/50 min-h-[300px]">
                    {loading ? (
                        <div className="flex justify-center items-center h-full text-gray-400 gap-2">
                            <Loader2 className="animate-spin" /> 加载中...
                        </div>
                    ) : files.length === 0 ? (
                        <div className="text-center py-20 text-gray-400">
                            <p>知识库是空的</p>
                            <p className="text-sm">请上传 PDF 手册或 Excel 故障表</p>
                        </div>
                    ) : (
                        files.map((file, idx) => (
                            <div
                                key={idx}
                                // 增加点击事件
                                onClick={() => handleOpenFile(file.name)}
                                // 优化样式：
                                // cursor-pointer: 鼠标变成手型
                                // hover:bg-blue-50: 悬停变成浅蓝色
                                // hover:border-blue-200: 悬停边框变蓝
                                className="flex justify-between items-center bg-white p-3 rounded-lg border border-gray-100 hover:shadow-md hover:bg-blue-50 hover:border-blue-200 transition group cursor-pointer"
                            >
                                <div className="flex items-center gap-3 overflow-hidden">
                                    {/* 图标区域 */}
                                    <div className="w-8 h-8 bg-orange-50 rounded flex items-center justify-center text-orange-500 flex-shrink-0 group-hover:bg-white group-hover:text-blue-600 transition">
                                        {file.name.endsWith('.pdf') ? 'PDF' :
                                            file.name.endsWith('.xlsx') ? 'XLS' : 'DOC'}
                                    </div>

                                    {/* 文件名区域 */}
                                    <div className="truncate">
                                        <p className="font-medium text-gray-700 truncate text-sm group-hover:text-blue-700 transition" title={file.name}>
                                            {file.name}
                                        </p>
                                        <p className="text-xs text-gray-400 group-hover:text-blue-400">
                                            {file.chunks} 个片段 · 点击预览
                                        </p>
                                    </div>
                                </div>

                                {/* 删除按钮 (阻止冒泡) */}
                                <button
                                    onClick={(e) => {
                                        e.stopPropagation(); // 【关键】防止点击删除时触发打开文件
                                        handleDelete(file.name);
                                    }}
                                    className="p-2 text-gray-300 hover:text-red-500 hover:bg-red-50 rounded-lg transition z-10"
                                    title="删除文件"
                                >
                                    <Trash2 size={16} />
                                </button>
                            </div>
                        ))
                    )}
                </div>

                {/* 底部操作区 */}
                <div className="p-4 border-t bg-white flex justify-between items-center">
                    <div className="text-xs text-gray-400">
                        支持 PDF, Word, Excel (自动向量化)
                    </div>
                    <div className="flex gap-3">
                        <button
                            onClick={fetchFiles}
                            className="p-2 text-gray-500 hover:bg-gray-100 rounded-lg transition"
                            title="刷新列表"
                        >
                            <RefreshCw size={20} />
                        </button>

                        <label className={`flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg cursor-pointer hover:bg-blue-700 transition shadow-md ${uploading ? 'opacity-70 cursor-wait' : ''}`}>
                            {uploading ? <Loader2 size={18} className="animate-spin" /> : <Upload size={18} />}
                            <span>{uploading ? '正在解析入库...' : '上传新文件'}</span>
                            <input
                                type="file"
                                className="hidden"
                                accept=".pdf,.docx,.xlsx,.csv,.txt"
                                onChange={handleUpload}
                                disabled={uploading}
                            />
                        </label>
                    </div>
                </div>
            </div>
        </div>
    );
}