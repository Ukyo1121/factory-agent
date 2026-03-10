import { useState, useRef, useEffect } from 'react';
import {
    Upload, Video, Play, Trash2, ArrowLeft, X,
    Clock, FileVideo, CheckCircle, Loader2, Search, Brain, FileCheck
} from 'lucide-react';
import { API_BASE_URL } from "./config";

const VIDEO_API_URL = `${API_BASE_URL}/training-videos`;

export default function TrainingVideoManager({ onBack }) {
    const [videos, setVideos] = useState([]);
    const [isUploading, setIsUploading] = useState(false);
    const [uploadProgress, setUploadProgress] = useState(0);
    const [selectedVideo, setSelectedVideo] = useState(null);
    const [searchTerm, setSearchTerm] = useState("");
    const fileInputRef = useRef(null);
    const [extractingIds, setExtractingIds] = useState(new Set());

    // 加载视频列表
    useEffect(() => {
        fetchVideos();
    }, []);

    const fetchVideos = async () => {
        try {
            const response = await fetch(VIDEO_API_URL);
            if (response.ok) {
                const data = await response.json();
                setVideos(data.videos || []);
            }
        } catch (error) {
            console.error('获取视频列表失败:', error);
        }
    };

    // 处理文件上传
    const handleFileSelect = async (event) => {
        const file = event.target.files?.[0];
        if (!file) return;

        // 验证文件类型
        if (!file.type.startsWith('video/')) {
            alert('请选择视频文件');
            return;
        }

        // 验证文件大小 (限制500MB)
        if (file.size > 500 * 1024 * 1024) {
            alert('视频文件不能超过500MB');
            return;
        }

        setIsUploading(true);
        setUploadProgress(0);

        const formData = new FormData();
        formData.append('file', file);

        try {
            const xhr = new XMLHttpRequest();

            // 监听上传进度
            xhr.upload.addEventListener('progress', (e) => {
                if (e.lengthComputable) {
                    const percent = Math.round((e.loaded / e.total) * 100);
                    setUploadProgress(percent);
                }
            });

            xhr.addEventListener('load', () => {
                if (xhr.status === 200) {
                    fetchVideos(); // 刷新列表
                    alert('视频上传成功！');
                } else {
                    alert('上传失败，请重试');
                }
                setIsUploading(false);
                setUploadProgress(0);
            });

            xhr.addEventListener('error', () => {
                alert('上传失败，请检查网络连接');
                setIsUploading(false);
                setUploadProgress(0);
            });

            xhr.open('POST', `${VIDEO_API_URL}/upload`);
            xhr.send(formData);

        } catch (error) {
            console.error('上传失败:', error);
            alert('上传失败，请重试');
            setIsUploading(false);
        }
    };

    // 删除视频
    const handleDelete = async (videoId) => {
        if (!confirm('确定要删除这个视频吗？')) return;

        try {
            const response = await fetch(`${VIDEO_API_URL}/${videoId}`, {
                method: 'DELETE'
            });
            if (response.ok) {
                fetchVideos();
                alert('删除成功');
            }
        } catch (error) {
            console.error('删除失败:', error);
            alert('删除失败，请重试');
        }
    };

    // 过滤视频
    const filteredVideos = videos.filter(video =>
        video.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
        video.description?.toLowerCase().includes(searchTerm.toLowerCase())
    );

    // 格式化文件大小
    const formatSize = (bytes) => {
        if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
        return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    };

    // 格式化时长
    const formatDuration = (seconds) => {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    };

    // 触发视频知识提取
    const handleExtractKnowledge = async (video) => {
        if (extractingIds.has(video.id)) return;

        setExtractingIds(prev => new Set(prev).add(video.id));

        try {
            const response = await fetch(`${VIDEO_API_URL}/${video.id}/extract`, {
                method: 'POST'
            });

            if (response.ok) {
                alert("已提交后台开始提取！这可能需要几分钟时间，请稍后刷新查看状态。");
                // 可以选择在这里重新 fetchVideos() 刷新状态
            } else {
                const data = await response.json();
                alert(data.detail || "提取任务提交失败");
                setExtractingIds(prev => {
                    const newSet = new Set(prev);
                    newSet.delete(video.id);
                    return newSet;
                });
            }
        } catch (error) {
            console.error('触发提取失败:', error);
            alert("网络错误");
            setExtractingIds(prev => {
                const newSet = new Set(prev);
                newSet.delete(video.id);
                return newSet;
            });
        }
    };

    return (
        <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50/30 to-purple-50/40 relative overflow-hidden">
            {/* 背景装饰 */}
            <div className="absolute inset-0 bg-[linear-gradient(to_right,#8882_1px,transparent_1px),linear-gradient(to_bottom,#8882_1px,transparent_1px)] bg-[size:64px_64px]"></div>
            <div className="absolute top-0 right-0 w-[500px] h-[500px] bg-blue-200/40 rounded-full blur-[120px]"></div>

            {/* 返回按钮 */}
            <button
                onClick={onBack}
                className="absolute top-8 left-8 z-20 flex items-center gap-3 px-5 py-2.5 bg-white/80 backdrop-blur-sm rounded-full border border-slate-200 text-slate-600 hover:text-blue-600 hover:border-blue-300 hover:shadow-md transition-all group"
            >
                <ArrowLeft size={18} className="group-hover:-translate-x-1 transition-transform" />
                <span className="font-semibold">返回</span>
            </button>

            <div className="relative z-10 p-8 pt-24">
                {/* 页面标题 */}
                <div className="max-w-7xl mx-auto mb-12 text-center">
                    <div className="w-20 h-20 bg-gradient-to-br from-blue-500 to-blue-600 rounded-2xl mx-auto flex items-center justify-center mb-6 shadow-lg">
                        <Video size={36} className="text-white" />
                    </div>
                    <h1 className="text-5xl font-black text-slate-800 mb-4">
                        <span className="bg-clip-text text-transparent bg-gradient-to-r from-blue-600 to-purple-600">
                            培训视频库
                        </span>
                    </h1>
                    <p className="text-slate-600 text-lg">上传和管理您的培训视频资料</p>
                </div>

                {/* 工具栏 */}
                <div className="max-w-7xl mx-auto mb-8 flex flex-col sm:flex-row gap-4 items-center justify-between">
                    {/* 搜索框 */}
                    <div className="relative w-full sm:w-96">
                        <Search size={20} className="absolute left-4 top-1/2 -translate-y-1/2 text-gray-400" />
                        <input
                            type="text"
                            placeholder="搜索视频标题或描述..."
                            value={searchTerm}
                            onChange={(e) => setSearchTerm(e.target.value)}
                            className="w-full pl-12 pr-4 py-3 rounded-xl border border-gray-200 bg-white/80 backdrop-blur-sm focus:ring-2 focus:ring-blue-500/20 focus:border-blue-400 transition-all"
                        />
                    </div>

                    {/* 上传按钮 */}
                    <button
                        onClick={() => fileInputRef.current?.click()}
                        disabled={isUploading}
                        className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-blue-600 to-blue-700 text-white rounded-xl hover:shadow-lg hover:shadow-blue-200 transition-all disabled:opacity-50 disabled:cursor-not-allowed font-semibold"
                    >
                        {isUploading ? (
                            <>
                                <Loader2 size={20} className="animate-spin" />
                                上传中 {uploadProgress}%
                            </>
                        ) : (
                            <>
                                <Upload size={20} />
                                上传视频
                            </>
                        )}
                    </button>
                    <input
                        ref={fileInputRef}
                        type="file"
                        accept="video/*"
                        onChange={handleFileSelect}
                        className="hidden"
                    />
                </div>

                {/* 上传进度条 */}
                {isUploading && (
                    <div className="max-w-7xl mx-auto mb-6">
                        <div className="bg-white/80 backdrop-blur-sm rounded-xl p-4 border border-blue-200">
                            <div className="flex items-center justify-between mb-2">
                                <span className="text-sm text-gray-600">正在上传...</span>
                                <span className="text-sm font-semibold text-blue-600">{uploadProgress}%</span>
                            </div>
                            <div className="w-full bg-gray-200 rounded-full h-2 overflow-hidden">
                                <div
                                    className="h-full bg-gradient-to-r from-blue-500 to-blue-600 transition-all duration-300"
                                    style={{ width: `${uploadProgress}%` }}
                                />
                            </div>
                        </div>
                    </div>
                )}

                {/* 视频网格 */}
                <div className="max-w-7xl mx-auto">
                    {filteredVideos.length === 0 ? (
                        <div className="text-center py-20">
                            <div className="w-24 h-24 bg-gray-100 rounded-full mx-auto flex items-center justify-center mb-6">
                                <FileVideo size={40} className="text-gray-400" />
                            </div>
                            <p className="text-gray-500 text-lg">
                                {searchTerm ? '未找到匹配的视频' : '还没有上传任何视频'}
                            </p>
                            {!searchTerm && (
                                <p className="text-gray-400 text-sm mt-2">点击上方"上传视频"按钮开始添加培训资料</p>
                            )}
                        </div>
                    ) : (
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                            {filteredVideos.map((video) => (
                                <VideoCard
                                    key={video.id}
                                    video={video}
                                    onPlay={() => setSelectedVideo(video)}
                                    onDelete={() => handleDelete(video.id)}
                                    onExtract={() => handleExtractKnowledge(video)}
                                    isExtracting={extractingIds.has(video.id)}
                                    formatSize={formatSize}
                                    formatDuration={formatDuration}
                                />
                            ))}
                        </div>
                    )}
                </div>
            </div>

            {/* 视频播放器模态框 */}
            {selectedVideo && (
                <VideoPlayerModal
                    video={selectedVideo}
                    onClose={() => setSelectedVideo(null)}
                />
            )}
        </div>
    );
}

// 视频卡片组件
function VideoCard({ video, onPlay, onDelete, onExtract, formatSize, formatDuration, isExtracting }) {
    return (
        <div className="group bg-white/80 backdrop-blur-sm rounded-2xl overflow-hidden border border-gray-200 hover:shadow-xl transition-all duration-300 hover:-translate-y-1">
            {/* 缩略图区域 */}
            <div className="relative aspect-video bg-gradient-to-br from-gray-100 to-gray-200 overflow-hidden">
                {video.thumbnail ? (
                    <img
                        src={`${API_BASE_URL}${video.thumbnail}`}
                        alt={video.title}
                        className="w-full h-full object-cover"
                    />
                ) : (
                    <div className="w-full h-full flex items-center justify-center">
                        <Video size={48} className="text-gray-400" />
                    </div>
                )}

                {/* 播放按钮覆盖层 */}
                <div className="absolute inset-0 bg-black/0 group-hover:bg-black/40 transition-all flex items-center justify-center">
                    <button
                        onClick={onPlay}
                        className="opacity-0 group-hover:opacity-100 w-16 h-16 bg-white rounded-full flex items-center justify-center text-blue-600 hover:scale-110 transition-all shadow-lg"
                    >
                        <Play size={28} fill="currentColor" />
                    </button>
                </div>

                {/* 时长标签 */}
                {video.duration && (
                    <div className="absolute bottom-2 right-2 bg-black/70 text-white text-xs px-2 py-1 rounded">
                        {formatDuration(video.duration)}
                    </div>
                )}
            </div>

            {/* 信息区域 */}
            <div className="p-4">
                <h3 className="font-bold text-gray-800 mb-2 line-clamp-2 leading-snug">
                    {video.title}
                </h3>

                {video.description && (
                    <p className="text-sm text-gray-600 mb-3 line-clamp-2">
                        {video.description}
                    </p>
                )}

                <div className="flex items-center justify-between text-xs text-gray-500 mb-3">
                    <span className="flex items-center gap-1">
                        <Clock size={14} />
                        {new Date(video.uploadedAt).toLocaleDateString()}
                    </span>
                    <span>{formatSize(video.size)}</span>
                </div>

                <div className="flex gap-2">
                    <button
                        onClick={onPlay}
                        className="flex-1 flex items-center justify-center gap-1 px-2 py-2 bg-blue-50 text-blue-600 rounded-lg hover:bg-blue-100 transition-all font-semibold text-sm"
                    >
                        <Play size={16} />
                        播放
                    </button>

                    {/* 知识提取按钮 */}
                    {video.is_extracted ? (
                        <div className="flex-[1.2] flex items-center justify-center gap-1 px-2 py-2 bg-green-50 text-green-600 rounded-lg text-sm font-semibold cursor-default">
                            <FileCheck size={16} />
                            已入库
                        </div>
                    ) : (
                        <button
                            onClick={(e) => { e.stopPropagation(); onExtract(); }}
                            disabled={isExtracting}
                            className={`flex-[1.2] flex items-center justify-center gap-1 px-2 py-2 rounded-lg transition-all font-semibold text-sm ${isExtracting
                                ? 'bg-purple-50 text-purple-400 cursor-not-allowed'
                                : 'bg-purple-50 text-purple-600 hover:bg-purple-100'
                                }`}
                        >
                            {isExtracting ? (
                                <><Loader2 size={16} className="animate-spin" /> 解析中...</>
                            ) : (
                                <><Brain size={16} /> 提取知识</>
                            )}
                        </button>
                    )}

                    {/* 删除按钮 */}
                    <button
                        onClick={(e) => { e.stopPropagation(); onDelete(); }}
                        className="px-3 py-2 bg-red-50 text-red-600 rounded-lg hover:bg-red-100 transition-all shrink-0 flex items-center justify-center"
                        title="删除视频"
                    >
                        <Trash2 size={16} />
                    </button>
                </div>
            </div>
        </div>
    );
}

// 视频播放器模态框
function VideoPlayerModal({ video, onClose }) {
    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4">
            <div className="relative w-full max-w-5xl bg-white rounded-2xl overflow-hidden shadow-2xl">
                {/* 关闭按钮 */}
                <button
                    onClick={onClose}
                    className="absolute top-4 right-4 z-10 w-10 h-10 bg-black/50 hover:bg-black/70 text-white rounded-full flex items-center justify-center transition-all"
                >
                    <X size={20} />
                </button>

                {/* 视频播放器 */}
                <div className="aspect-video bg-black">
                    <video
                        src={`${API_BASE_URL}${video.url}`}
                        controls
                        autoPlay
                        className="w-full h-full"
                    />
                </div>

                {/* 视频信息 */}
                <div className="p-6 bg-gray-50">
                    <h2 className="text-2xl font-bold text-gray-800 mb-2">{video.title}</h2>
                    {video.description && (
                        <p className="text-gray-600 mb-4">{video.description}</p>
                    )}
                    <div className="flex gap-4 text-sm text-gray-500">
                        <span className="flex items-center gap-1">
                            <Clock size={16} />
                            上传时间: {new Date(video.uploadedAt).toLocaleString()}
                        </span>
                        <span className="flex items-center gap-1">
                            <FileVideo size={16} />
                            文件大小: {(video.size / (1024 * 1024)).toFixed(1)} MB
                        </span>
                    </div>
                </div>
            </div>
        </div>
    );
}
