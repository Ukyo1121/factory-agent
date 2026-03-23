import { useState, useRef, useEffect } from 'react';
import {
    Send, Plus, MessageSquare, User, Bot, Loader2, StopCircle,
    Mic, ArrowLeft, GraduationCap, Trash2, Wrench, AlertTriangle, Paperclip, X, Play, Video
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { API_BASE_URL } from "./config";
import { FileText, Image as ImageIcon } from 'lucide-react';
import remarkGfm from 'remark-gfm';

const API_URL = `${API_BASE_URL}/chat`;
const VOICE_API_URL = `${API_BASE_URL}/voice`;

function ChatMiniVideoCard({ video, onPlay }) {
    return (
        <div
            onClick={onPlay}
            className="mt-3 w-102 bg-white border border-blue-100 rounded-xl overflow-hidden shadow-sm hover:shadow-md transition-all cursor-pointer group"
        >
            <div className="relative aspect-video bg-slate-100">
                {video.thumbnail ? (
                    <img src={video.thumbnail} alt={video.title} className="w-full h-full object-cover" />
                ) : (
                    <div className="flex items-center justify-center w-full h-full">
                        <Video className="text-gray-300" size={32} />
                    </div>
                )}
                {/* 悬停播放遮罩 */}
                <div className="absolute inset-0 bg-black/10 group-hover:bg-black/30 transition-colors flex items-center justify-center">
                    <div className="w-12 h-12 bg-white/90 rounded-full flex items-center justify-center text-blue-600 shadow-lg group-hover:scale-110 transition-transform">
                        <Play size={24} className="ml-1" fill="currentColor" />
                    </div>
                </div>
            </div>
            <div className="p-3">
                <p className="text-sm font-semibold text-gray-800 truncate">{video.title}</p>
                <p className="text-xs text-gray-500 mt-1 flex items-center gap-1">
                    <Play size={12} /> 点击播放
                </p>
            </div>
        </div>
    );
}
// 🎬 视频播放模态框组件
const VideoPlayerModal = ({ video, onClose }) => {
    if (!video) return null;

    return (
        <div
            className="fixed inset-0 z-[999] flex items-center justify-center bg-black/80 backdrop-blur-sm p-4 transition-opacity"
            onClick={onClose} // 点击黑色背景时关闭
        >
            <div
                className="relative w-full max-w-4xl bg-black rounded-xl shadow-2xl overflow-hidden animate-in fade-in zoom-in-95 duration-200"
                onClick={(e) => e.stopPropagation()} // 防止点击视频本身时触发关闭
            >
                {/* 顶部悬浮栏：标题 + 关闭按钮 */}
                <div className="absolute top-0 left-0 right-0 z-10 flex justify-between items-center p-4 bg-gradient-to-b from-black/80 to-transparent">
                    <h3 className="text-white font-medium text-sm md:text-base truncate pr-8">
                        {video.title || '视频演示'}
                    </h3>
                    <button
                        onClick={onClose}
                        className="text-white/80 hover:text-white bg-black/40 hover:bg-black/60 rounded-full p-1.5 transition-all"
                        title="关闭"
                    >
                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12"></path>
                        </svg>
                    </button>
                </div>

                {/* 核心视频播放器 */}
                <video
                    src={video.url}
                    controls
                    autoPlay
                    className="w-full h-auto max-h-[85vh] outline-none"
                    poster={video.thumbnail}
                >
                    您的浏览器不支持 HTML5 视频播放。
                </video>
            </div>
        </div>
    );
};
// 接收 onBack 属性用于返回主页
export default function TrainingAssistant({ onBack, userId }) {
    const [threads, setThreads] = useState([]);
    const [activeThreadId, setActiveThreadId] = useState(null);
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState("");
    const [isLoading, setIsLoading] = useState(false);

    // 打字机状态
    const [streamBuffer, setStreamBuffer] = useState("");
    const [displayedContent, setDisplayedContent] = useState("");
    const [isTyping, setIsTyping] = useState(false);

    // 语音状态
    const [isRecording, setIsRecording] = useState(false);
    const [isProcessingVoice, setIsProcessingVoice] = useState(false);
    const mediaRecorderRef = useRef(null);
    const audioChunksRef = useRef([]);

    const messagesEndRef = useRef(null);
    const abortControllerRef = useRef(null);
    // 存储临时文件解析后的内容
    const [attachedFiles, setAttachedFiles] = useState([]);
    const [playingVideo, setPlayingVideo] = useState(null);

    // 内容解析器
    const renderMessageContent = (text, role) => {
        if (!text) return null;

        // 匹配包含换行符的视频标签
        const parts = text.split(/(<video_preview>[\s\S]*?<\/video_preview>)/g);

        return parts.map((part, index) => {
            // --- 1. 完整视频卡片 ---
            if (part.trim().startsWith('<video_preview>')) {
                const jsonStr = part.replace('<video_preview>', '').replace('</video_preview>', '');
                try {
                    const videoObj = JSON.parse(jsonStr);

                    // 修复路径
                    if (videoObj.url && videoObj.url.startsWith('/')) {
                        videoObj.url = `${API_BASE_URL}${videoObj.url}`;
                    }
                    if (videoObj.thumbnail && videoObj.thumbnail.startsWith('/')) {
                        videoObj.thumbnail = `${API_BASE_URL}${videoObj.thumbnail}`;
                    }

                    return (
                        <div key={`video-${index}`} className="my-4">
                            <ChatMiniVideoCard
                                video={videoObj}
                                onPlay={() => setPlayingVideo(videoObj)}
                            />
                        </div>
                    );
                } catch (e) {
                    console.error("解析视频数据失败:", e);
                    return <div key={`err-${index}`} className="text-red-500 text-xs border border-red-200 p-2 rounded">视频解析失败: {jsonStr}</div>;
                }
            }

            // --- 2. 视频流式加载中 ---
            if (part.includes('<video_preview>') && !part.includes('</video_preview>')) {
                return (
                    <span key={`loading-${index}`} className="inline-flex items-center text-blue-500 text-xs animate-pulse ml-2">
                        <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-blue-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>
                        正在生成视频组件...
                    </span>
                );
            }

            // --- 3. Markdown 普通文本 ---
            if (!part) return null;

            return (
                <ReactMarkdown
                    key={`md-${index}`}
                    remarkPlugins={[remarkGfm]}
                    components={{
                        img: ({ node, ...props }) => {
                            let imgSrc = props.src;
                            if (imgSrc) {
                                // 🛠️ 修复 8080 端口拒绝连接：正则匹配任意 localhost 端口并替换
                                imgSrc = imgSrc.replace(/http:\/\/localhost:\d+/g, API_BASE_URL);
                                if (imgSrc.startsWith('/images')) {
                                    imgSrc = `${API_BASE_URL}${imgSrc}`;
                                }
                            }
                            return <img {...props} src={imgSrc} className="max-w-full h-auto rounded-lg shadow-md my-4 border border-gray-200 cursor-zoom-in hover:shadow-lg transition-shadow" onClick={() => window.open(imgSrc, '_blank')} />
                        },
                        code({ node, className, children, ...props }) {
                            // 🛠️ 修复 DOM 嵌套报错：摒弃 inline，改用 className 匹配 language-xxx 来判断是不是代码块
                            const match = /language-(\w+)/.exec(className || '');
                            return match ? (
                                <div className="bg-gray-800 text-gray-100 p-2 rounded-md my-2 overflow-x-auto">
                                    <code className={className} {...props}>{children}</code>
                                </div>
                            ) : (
                                <code className={`${role === 'user' ? 'bg-blue-700' : 'bg-gray-100 text-red-500'} px-1 rounded`} {...props}>
                                    {children}
                                </code>
                            );
                        },
                        table: ({ node, ...props }) => <div className="overflow-x-auto my-2 rounded-lg border border-gray-200"><table className="min-w-full divide-y divide-gray-200 text-sm" {...props} /></div>,
                        thead: ({ node, ...props }) => <thead className="bg-blue-50" {...props} />,
                        tbody: ({ node, ...props }) => <tbody className="bg-white divide-y divide-gray-200" {...props} />,
                        tr: ({ node, ...props }) => <tr className="hover:bg-gray-50 transition-colors" {...props} />,
                        th: ({ node, ...props }) => <th className="px-4 py-3 text-left text-xs font-medium text-blue-800 uppercase tracking-wider font-bold" {...props} />,
                        td: ({ node, ...props }) => <td className="px-4 py-2 whitespace-nowrap text-gray-700" {...props} />,
                        p: ({ node, ...props }) => <p className="mb-2 last:mb-0" {...props} />,
                        a: ({ node, ...props }) => <a className="text-blue-600 hover:underline" target="_blank" {...props} />,
                        ul: ({ node, ...props }) => <ul className="list-disc list-inside mb-2" {...props} />,
                        ol: ({ node, ...props }) => <ol className="list-decimal list-inside mb-2" {...props} />,
                        li: ({ node, ...props }) => <li className="mb-1" {...props} />,
                    }}
                >
                    {part}
                </ReactMarkdown>
            );
        });
    };
    // -----------------------------------------------------------------------
    // 1. 获取历史会话列表的函数
    // -----------------------------------------------------------------------
    const fetchUserThreads = async () => {
        if (!userId) return;
        try {
            const res = await fetch(`${API_BASE_URL}/threads/${userId}?thread_type=training`);
            if (res.ok) {
                const data = await res.json();
                setThreads(data);

                if (data.length > 0) {
                    // 如果有历史记录，默认选中第一个
                    if (!activeThreadId) {
                        switchThread(data[0].id);
                    }
                } else {
                    // 如果列表为空，自动调用上面改写过的、带持久化的新建函数
                    // 注意：这里可能会导致组件加载时自动发一次 POST 请求，是正常行为
                    createNewThread();
                }
            }
        } catch (error) {
            console.error("获取会话列表失败:", error);
        }
    };

    // -----------------------------------------------------------------------
    // 2. 初始化 Effect：当 userId 变化时，去后端拉取列表
    // -----------------------------------------------------------------------
    useEffect(() => {
        if (userId) {
            fetchUserThreads();
        }
    }, [userId]);

    // 自动滚动
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [messages, displayedContent, isLoading]);

    // 打字机逻辑
    useEffect(() => {
        if (streamBuffer.length > displayedContent.length) {
            setIsTyping(true);
            const timer = setTimeout(() => {
                setDisplayedContent(prev => streamBuffer.slice(0, prev.length + 1));
            }, 10);
            return () => clearTimeout(timer);
        } else {
            setIsTyping(false);
            if (!isLoading && streamBuffer) {
                setMessages(prev => {
                    const newMsgs = [...prev];
                    if (newMsgs.length > 0 && newMsgs[newMsgs.length - 1].role === 'ai') {
                        newMsgs[newMsgs.length - 1].content = streamBuffer;
                    }
                    return newMsgs;
                });
            }
        }
    }, [streamBuffer, displayedContent, isLoading]);

    // -----------------------------------------------------------------------
    // 3. 加载单条会话的历史消息
    // -----------------------------------------------------------------------
    const getFileIcon = (fileName, type) => {
        const ext = fileName ? fileName.split('.').pop().toLowerCase() : '';
        const isImage = type === 'image' || ['jpg', 'jpeg', 'png', 'gif', 'webp'].includes(ext);

        if (isImage) {
            return <ImageIcon size={20} className="text-purple-500" />;
        }
        if (['pdf'].includes(ext)) {
            return <FileText size={20} className="text-red-500" />;
        }
        if (['xls', 'xlsx', 'csv'].includes(ext)) {
            return <FileText size={20} className="text-green-600" />; // 假装有个 Excel 图标，用绿色区分
        }
        if (['doc', 'docx'].includes(ext)) {
            return <FileText size={20} className="text-blue-600" />;
        }
        return <Paperclip size={20} className="text-gray-500" />;
    };
    const loadHistory = async (threadId) => {
        setIsLoading(true);
        setMessages([]); // 切换时先清空当前显示
        setStreamBuffer("");
        setDisplayedContent("");

        try {
            // 调用后端接口获取具体聊天记录
            const res = await fetch(`${API_BASE_URL}/history/${threadId}`);
            const data = await res.json();

            // 假设后端返回格式: { history: [{role: 'user', content: '...'}, ...] }
            if (data.history && Array.isArray(data.history)) {
                setMessages(data.history);
            }
        } catch (err) {
            console.error("加载历史记录失败", err);
        } finally {
            setIsLoading(false);
        }
    };

    // -----------------------------------------------------------------------
    // 4. 创建新会话
    // -----------------------------------------------------------------------
    const createNewThread = async () => {
        if (isLoading) return;
        setIsLoading(true); // 加个简单的 loading 锁防止重复点击

        try {
            // 1. 向后端发送 POST 请求，在数据库创建记录
            const res = await fetch(`${API_BASE_URL}/threads`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ user_id: userId, thread_type: 'training' }) // 告诉后端是谁创建的
            });

            if (!res.ok) {
                throw new Error("创建会话失败");
            }

            // 2. 拿到后端返回的真实 DB 数据 (包含 thread_id, title 等)
            const newThreadData = await res.json();
            // 预期格式: { "id": "uuid...", "title": "新对话", "messages": [] }

            // 3. 构建前端对象
            const newThread = {
                id: newThreadData.id,
                title: newThreadData.title || "新会话",
                history: []
            };

            // 4. 更新前端列表 (插到最前面)
            setThreads(prev => [newThread, ...prev]);

            // 5. 自动选中新会话
            setActiveThreadId(newThread.id);
            setMessages([]); // 清空右侧消息
            resetTyper();    // 重置打字机状态

        } catch (error) {
            console.error("新建会话失败:", error);
            alert("无法创建新会话，请检查网络或后端服务");
        } finally {
            setIsLoading(false);
        }
    };

    // -----------------------------------------------------------------------
    // 5. 切换会话
    // -----------------------------------------------------------------------
    const switchThread = (id) => {
        if (isLoading && activeThreadId === id) return; // 如果正在加载当前会话，忽略

        setActiveThreadId(id);

        // 查找这个会话是“本地新建的空会话”还是“已有的历史会话”
        const targetThread = threads.find(t => t.id === id);

        // 如果是新创建的空会话（通常没有后端数据），直接清空界面即可
        if (targetThread && targetThread.title === "新会话" && (!targetThread.history || targetThread.history.length === 0)) {
            setMessages([]);
            setStreamBuffer("");
            setDisplayedContent("");
        } else {
            // 如果是历史会话，去后端加载消息
            loadHistory(id);
        }
    };

    const resetTyper = () => {
        setStreamBuffer("");
        setDisplayedContent("");
        setIsTyping(false);
    };
    // 处理删除逻辑
    const handleDeleteThread = async (e, threadId) => {
        // 1. 阻止事件冒泡：防止触发外层 div 的 onClick (切换会话)
        e.stopPropagation();

        // 2. 确认提示
        if (!window.confirm("确定要删除这条历史记录吗？删除后无法恢复。")) {
            return;
        }

        try {
            // 3. 调用后端 API
            const res = await fetch(`${API_BASE_URL}/threads/${threadId}?user_id=${userId}`, {
                method: 'DELETE',
            });

            if (res.ok) {
                // 4. 更新前端状态
                const newThreads = threads.filter(t => t.id !== threadId);
                setThreads(newThreads);

                // 5. 如果删除的是当前选中的会话
                if (activeThreadId === threadId) {
                    if (newThreads.length > 0) {
                        // 切换到剩下的第一个
                        setActiveThreadId(newThreads[0].id);
                        // 这里可以选做：重新拉取该会话的消息，或者简单清空
                        setMessages([]);
                        // 触发一次切换逻辑最好，这里简化处理，手动清理
                    } else {
                        // 如果删光了，就创建一个新的
                        createNewThread();
                    }
                }
            } else {
                alert("删除失败，请重试");
            }
        } catch (error) {
            console.error("删除出错:", error);
            alert("网络错误");
        }
    };
    // 处理临时文件上传函数
    const handleFileUpload = async (e) => {
        // 获取选中的所有文件
        const files = Array.from(e.target.files);
        if (files.length === 0) return;

        // 遍历处理每个文件(为了用户体验,这里并行上传)
        const uploadPromises = files.map(async (file) => {
            const formData = new FormData();
            formData.append("file", file);

            try {
                const res = await fetch(`${API_BASE_URL}/upload-temp-file`, {
                    method: "POST",
                    body: formData,
                });
                return await res.json(); // 返回解析后的数据 {type, content, fileName, savedPath}
            } catch (err) {
                console.error(`文件 ${file.name} 上传失败`, err);
                return null;
            }
        });

        setIsLoading(true);
        try {
            const results = await Promise.all(uploadPromises);
            // 过滤掉失败的(null),并将新文件追加到现有列表中
            const successfulFiles = results.filter(f => f !== null);
            setAttachedFiles(prev => [...prev, ...successfulFiles]);
        } finally {
            setIsLoading(false);
            e.target.value = ''; // 清空 input,允许重复上传同名文件
        }
    };

    // 删除单个文件的函数
    const removeFile = (indexToRemove) => {
        setAttachedFiles(prev => prev.filter((_, index) => index !== indexToRemove));
    };

    const handleSend = async (manualInput = null) => {
        const textToSend = manualInput || input;
        if (!textToSend.trim() || isLoading) return;

        const currentFiles = [...attachedFiles]; // 复制当前的文件列表

        // 构建包含路径信息的 temp_context
        const finalTempContext = currentFiles.length > 0
            ? currentFiles.map(f => ({
                type: f.type,
                content: f.content,
                fileName: f.fileName || f.name,
                savedPath: f.savedPath  // 传递保存路径
            }))
            : null;

        // 1. 界面立即显示用户消息
        // 构建消息对象
        const userMessage = {
            role: "user",
            content: textToSend,
            files: currentFiles.map(f => ({
                name: f.fileName || f.name,
                type: f.type,
                base64: f.type === "image" ? f.content : undefined,
                savedPath: f.savedPath  // 保存路径(用于历史记录)
            }))
        };
        setMessages(prev => [...prev, userMessage]);
        setInput("");
        setIsLoading(true);
        setAttachedFiles([]);
        resetTyper();

        // 2. 预占位 AI 消息
        setMessages(prev => [...prev, { role: 'ai', content: "" }]);
        abortControllerRef.current = new AbortController();

        try {
            const currentThread = threads.find(t => t.id === activeThreadId);

            // 判断条件:如果有当前会话,且标题是默认值 "新会话"
            if (currentThread && (currentThread.title === "新会话" || currentThread.title === "New Thread")) {

                const newTitle = textToSend.length > 15
                    ? textToSend.substring(0, 15) + "..."
                    : textToSend;

                // A. 更新前端显示
                setThreads(prev => prev.map(t =>
                    t.id === activeThreadId ? { ...t, title: newTitle } : t
                ));

                // B. 异步请求后端更新数据库 (fire-and-forget,不阻塞聊天)
                fetch(`${API_BASE_URL}/threads/${activeThreadId}/title`, {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ title: newTitle })
                }).catch(err => console.warn("标题自动更新失败:", err));
            }
        } catch (err) {
            console.error("标题逻辑出错,已跳过:", err);
        }

        try {
            const response = await fetch(API_URL, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    query: textToSend,
                    thread_id: activeThreadId,
                    user_id: userId,
                    temp_context: finalTempContext  // 使用新的 temp_context
                }),
                signal: abortControllerRef.current.signal
            });

            if (!response.ok) throw new Error("API Error");
            const reader = response.body.getReader();
            const decoder = new TextDecoder();

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                const chunk = decoder.decode(value, { stream: true });
                setStreamBuffer(prev => prev + chunk);
            }

        } catch (error) {
            if (error.name !== 'AbortError') {
                setStreamBuffer(prev => prev + "\n\n⚠️ 连接服务器失败,请检查后端。");
            }
        } finally {
            setIsLoading(false);
            abortControllerRef.current = null;
        }
    };

    const handleStop = () => {
        if (abortControllerRef.current) {
            abortControllerRef.current.abort();
            setIsLoading(false);
        }
    };

    // --- 语音逻辑 ---
    const startRecording = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorderRef.current = new MediaRecorder(stream);
            audioChunksRef.current = [];

            mediaRecorderRef.current.ondataavailable = (event) => {
                if (event.data.size > 0) audioChunksRef.current.push(event.data);
            };

            mediaRecorderRef.current.onstop = async () => {
                const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
                await sendAudioToBackend(audioBlob);
                stream.getTracks().forEach(track => track.stop());
            };

            mediaRecorderRef.current.start();
            setIsRecording(true);
        } catch (error) {
            console.error("无法访问麦克风:", error);
            alert("无法访问麦克风，请检查权限设置。");
        }
    };

    const stopRecording = () => {
        if (mediaRecorderRef.current && isRecording) {
            mediaRecorderRef.current.stop();
            setIsRecording(false);
            setIsProcessingVoice(true);
        }
    };

    const sendAudioToBackend = async (audioBlob) => {
        const formData = new FormData();
        formData.append("file", audioBlob, "voice_input.webm");

        try {
            const response = await fetch(VOICE_API_URL, { method: "POST", body: formData });
            if (!response.ok) throw new Error("识别失败");
            const data = await response.json();
            if (data.text) setInput(prev => prev + data.text);
        } catch (error) {
            console.error("语音识别错误:", error);
            alert("语音识别失败，请重试");
        } finally {
            setIsProcessingVoice(false);
        }
    };

    return (
        <div className="flex h-screen bg-gray-50 text-gray-800 font-sans animate-fade-in">
            {/* 侧边栏 */}
            <div className="w-64 bg-gray-900 text-white flex flex-col flex-shrink-0 shadow-xl z-20">
                {/* 顶部返回区域 */}
                <div className="p-4 border-b border-gray-800 flex items-center gap-3">
                    <button onClick={onBack} className="p-2 hover:bg-gray-800 rounded-lg text-gray-400 hover:text-white transition-colors">
                        <ArrowLeft size={20} />
                    </button>
                    <h1 className="font-bold text-lg flex items-center gap-2">
                        <GraduationCap className="text-blue-500" size={24} />
                        培训助手
                    </h1>
                </div>

                <div className="p-4">
                    <button
                        onClick={() => createNewThread("新会话")}
                        disabled={isLoading}
                        className="w-full flex items-center gap-2 bg-blue-600 hover:bg-blue-700 p-3 rounded-lg text-sm transition-all shadow-md group"
                    >
                        <Plus size={16} className="group-hover:rotate-90 transition-transform" /> 新建会话
                    </button>
                </div>

                <div className="flex-1 overflow-y-auto px-2 custom-scrollbar">
                    {threads.map(thread => (
                        <button
                            key={thread.id}
                            onClick={() => switchThread(thread.id)}
                            disabled={isLoading}
                            // 在 className 中添加了 'group'，这是为了让里面的 group-hover 生效
                            className={`
                group w-full text-left p-3 rounded-lg mb-1 text-sm flex items-center gap-2 transition-colors 
                ${activeThreadId === thread.id
                                    ? 'bg-gray-800 text-white border-l-2 border-blue-500'
                                    : 'text-gray-400 hover:bg-gray-800'
                                }
            `}
                        >
                            <MessageSquare size={14} className="flex-shrink-0" />

                            {/* 添加了 'flex-1'。这会让文字撑满中间的空间，把后面的按钮推到最右边 */}
                            <span className="truncate flex-1">{thread.title}</span>

                            {/* 删除按钮 */}
                            <div
                                role="button"
                                onClick={(e) => handleDeleteThread(e, thread.id)}
                                className={`
                    p-1.5 rounded-md text-slate-400 hover:text-red-400 hover:bg-gray-700 transition-all 
                    opacity-0 group-hover:opacity-100 flex-shrink-0
                    ${activeThreadId === thread.id ? 'opacity-100' : ''} 
                `}
                                title="删除会话"
                            >
                                <Trash2 size={16} />
                            </div>
                        </button>
                    ))}
                </div>
            </div>

            {/* 主界面 */}
            <div className="flex-1 flex flex-col relative bg-white">
                <div className="h-14 border-b flex items-center px-6 shadow-sm z-10 bg-white/80 backdrop-blur-md justify-between">
                    <h1 className="font-semibold text-gray-700 flex items-center gap-2">
                        <GraduationCap className="text-blue-600" size={20} />
                        <span>交互式操作培训</span>
                    </h1>
                    <div className="text-xs text-gray-400 flex items-center gap-1">
                        <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                        知识库已连接
                    </div>
                </div>

                {/* 聊天区域 */}
                <div className="flex-1 overflow-y-auto p-4 pb-32 custom-scrollbar">
                    <div className="max-w-3xl mx-auto space-y-6 min-h-full flex flex-col">
                        {messages.length === 0 && (
                            <div className="flex-1 flex flex-col items-center justify-center text-center mt-10">
                                <div className="w-20 h-20 bg-blue-50 rounded-2xl shadow-sm border border-blue-100 flex items-center justify-center mb-6">
                                    <Bot size={40} className="text-blue-600" />
                                </div>
                                <h2 className="text-2xl font-bold text-gray-800 mb-3">开启您的操作培训</h2>
                                <p className="text-gray-500 mb-10 max-w-md">
                                    您可以询问具体的设备操作步骤，我会通过图文并茂的方式指导您完成任务。
                                </p>
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 w-full max-w-3xl px-4 font-sans">
                                    <button
                                        onClick={() => handleSend("教我使用自动分拣系统的桁架机械手的主控界面")}
                                        className="flex items-center gap-3 p-4 bg-white border border-gray-200 rounded-xl hover:border-blue-400 hover:shadow-md transition-all text-left group"
                                    >
                                        <div className="w-10 h-10 bg-blue-50 rounded-lg flex items-center justify-center group-hover:bg-blue-100 transition-colors">
                                            <Wrench size={20} className="text-blue-600" />
                                        </div>
                                        <div>
                                            <div className="font-semibold text-gray-700 group-hover:text-blue-700">设备操作培训</div>
                                            <div className="text-xs text-gray-400">例：教我使用自动分拣系统的桁架机械手主控界面</div>
                                        </div>
                                    </button>

                                    <button
                                        onClick={() => handleSend("自动分拣系统的设备作业工艺流程是什么？")}
                                        className="flex items-center gap-3 p-4 bg-white border border-gray-200 rounded-xl hover:border-blue-400 hover:shadow-md transition-all text-left group"
                                    >
                                        <div className="w-10 h-10 bg-blue-50 rounded-lg flex items-center justify-center group-hover:bg-blue-100 transition-colors">
                                            <AlertTriangle size={20} className="text-blue-600" />
                                        </div>
                                        <div>
                                            <div className="font-semibold text-gray-700 group-hover:text-blue-700">操作规范查询</div>
                                            <div className="text-xs text-gray-400">例：自动分拣系统的设备作业工艺流程是什么？</div>
                                        </div>
                                    </button>
                                </div>
                            </div>
                        )}

                        {messages.map((msg, idx) => {
                            const isLastAiMessage = msg.role === 'ai' && idx === messages.length - 1;
                            // 判断是否正在思考（Loading 且 还没有内容显示）
                            const isThinking = isLastAiMessage && isLoading && !displayedContent;

                            // 决定显示的内容
                            const contentToShow = isLastAiMessage && (isLoading || isTyping) ? displayedContent : msg.content;

                            return (
                                <div key={idx} className={`flex gap-4 mb-6 w-full ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>

                                    {/* 1. 左侧：AI 头像 (只有 AI 消息才显示在左边) */}
                                    {msg.role === 'ai' && (
                                        <div className="w-8 h-8 rounded-full bg-blue-50 border border-blue-100 flex items-center justify-center flex-shrink-0 mt-1">
                                            <Bot size={16} className={`text-blue-600 ${isThinking || isTyping ? 'animate-pulse' : ''}`} />
                                        </div>
                                    )}

                                    {/* 2. 中间核心区：垂直排列 (文件在上，气泡在下) */}
                                    {/* max-w-[85%] 移到这里，控制整体宽度 */}
                                    {/* items-end 让用户的文件和气泡都靠右，items-start 让 AI 的都靠左 */}
                                    <div className={`flex flex-col gap-2 max-w-[85%] ${msg.role === 'user' ? 'items-end' : 'items-start'}`}>
                                        {/* 2.1 文件列表 */}
                                        {msg.files && msg.files.length > 0 && (
                                            <div className={`flex flex-wrap gap-2 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                                                {msg.files.map((file, fIndex) => {
                                                    // 1. 宽松判断图片类型
                                                    const isImage = file.type && (file.type === 'image' || file.type.startsWith('image'));

                                                    const getFixedUrl = (originalUrl) => {
                                                        if (!originalUrl) return null;
                                                        if (originalUrl.startsWith('http')) return originalUrl; // 已经是完整链接

                                                        // 如果已经是 /images/ 开头，直接拼接 BaseUrl
                                                        if (originalUrl.startsWith('/images/') || originalUrl.startsWith('images/')) {
                                                            return `${API_BASE_URL}${originalUrl.startsWith('/') ? originalUrl : '/' + originalUrl}`;
                                                        }

                                                        if (isImage || /\.(jpg|jpeg|png|gif|webp|bmp)$/i.test(file.name || '')) {
                                                            // 防止 originalUrl 自带 '/' 导致双斜杠，先去掉开头的 '/'
                                                            const cleanName = originalUrl.startsWith('/') ? originalUrl.slice(1) : originalUrl;
                                                            return `${API_BASE_URL}/images/${cleanName}`;
                                                        }

                                                        // 其他情况（如 /files/），按原样拼接
                                                        return `${API_BASE_URL}${originalUrl.startsWith('/') ? originalUrl : '/' + originalUrl}`;
                                                    };

                                                    // 2. 智能获取图片预览地址 (imgSrc)
                                                    let imgSrc = null;
                                                    if (isImage) {
                                                        if (file.content) {
                                                            imgSrc = file.content; // Base64
                                                        } else if (file.base64) {
                                                            imgSrc = `data:image/jpeg;base64,${file.base64}`;
                                                        } else if (file instanceof File) {
                                                            imgSrc = URL.createObjectURL(file); // 本地预览
                                                        } else if (file.url) {
                                                            // 使用修复后的 URL
                                                            imgSrc = getFixedUrl(file.url);
                                                        }
                                                    }

                                                    // 3. 智能获取文件下载地址 (downloadUrl)
                                                    // 逻辑: 优先用 file.url (历史记录), 其次用 file.savedPath (刚发送成功), 最后是 null
                                                    let downloadUrl = null;
                                                    if (file.url) {
                                                        // 如果是相对路径 (/files/...), 拼上 API_BASE_URL
                                                        downloadUrl = file.url.startsWith('http') ? file.url : `${API_BASE_URL}${file.url}`;
                                                    } else if (file.savedPath) {
                                                        // 刚上传成功，后端返回了 savedPath (例如 "doc_xxx.pdf")
                                                        // 假设后端挂载点是 /files/，手动拼接完整 URL
                                                        downloadUrl = `${API_BASE_URL}/files/${file.savedPath}`;
                                                    }

                                                    // 4. 渲染逻辑
                                                    if (isImage && imgSrc) {
                                                        // --- 图片渲染 ---
                                                        return (
                                                            <div
                                                                key={fIndex}
                                                                className="rounded-xl overflow-hidden border border-gray-200 shadow-sm hover:shadow-md transition-shadow bg-white relative group"
                                                                style={{ maxWidth: '300px' }}
                                                            >
                                                                <img
                                                                    src={imgSrc}
                                                                    alt={file.name || 'image'}
                                                                    className="w-full h-auto bg-gray-100"
                                                                    style={{ maxHeight: '300px', objectFit: 'contain', display: 'block' }}
                                                                    onError={(e) => { e.target.style.display = 'none'; }}
                                                                />
                                                                {/* 图片底部文件名 */}
                                                                <div className="px-3 py-2 bg-gray-50/90 border-t border-gray-100 backdrop-blur-sm flex justify-between items-center">
                                                                    <span className="text-xs text-gray-600 truncate block max-w-[150px]">
                                                                        {file.name || "图片预览"}
                                                                    </span>
                                                                    {/* 如果有下载链接，给图片也加个下载按钮 */}
                                                                    {downloadUrl && (
                                                                        <a href={downloadUrl} download={file.name} target="_blank" rel="noopener noreferrer" className="text-blue-500 hover:text-blue-700">
                                                                            <FileText size={14} />
                                                                        </a>
                                                                    )}
                                                                </div>
                                                            </div>
                                                        );
                                                    } else {
                                                        // --- 普通文件渲染 (带下载功能) ---
                                                        return (
                                                            <a
                                                                key={fIndex}
                                                                href={downloadUrl || "#"}
                                                                target={downloadUrl ? "_blank" : undefined}
                                                                rel="noopener noreferrer"
                                                                download={file.name} // 尝试触发浏览器下载
                                                                className={`flex items-center gap-2 bg-white border border-gray-200 px-3 py-2 rounded-xl shadow-sm transition-all cursor-default ${downloadUrl ? "hover:shadow-md hover:border-blue-300 cursor-pointer group" : "opacity-80"
                                                                    }`}
                                                                onClick={(e) => {
                                                                    if (!downloadUrl) e.preventDefault();
                                                                }}
                                                            >
                                                                {/* 图标 */}
                                                                <div className={`shrink-0 p-1 rounded-lg ${downloadUrl ? "bg-blue-50 text-blue-500 group-hover:text-blue-600" : "bg-gray-50 text-gray-400"}`}>
                                                                    <FileText size={20} />
                                                                </div>

                                                                {/* 文件名信息 */}
                                                                <div className="flex flex-col min-w-0 text-left">
                                                                    <span className={`text-xs font-medium truncate max-w-[180px] ${downloadUrl ? "text-blue-700 group-hover:text-blue-800" : "text-gray-700"}`}>
                                                                        {file.name || "未知文件"}
                                                                    </span>
                                                                    <span className="text-[10px] text-gray-400">
                                                                        {/* 状态显示: 如果有链接显示'点击下载'，否则显示'上传中/文件' */}
                                                                        {file.type === 'file' ? (downloadUrl ? '点击下载' : '处理中...') : file.type}
                                                                    </span>
                                                                </div>
                                                            </a>
                                                        );
                                                    }
                                                })}
                                            </div>
                                        )}

                                        {/* 2.2 消息气泡 (放在文件下方) */}
                                        <div className={`p-4 rounded-2xl text-sm leading-7 shadow-sm transition-all duration-300 w-fit ${msg.role === 'user'
                                            ? 'bg-blue-600 text-white rounded-tr-sm' // 用户：右上角直角
                                            : 'bg-white border border-gray-100 text-gray-800 rounded-tl-sm' // AI：左上角直角
                                            }`}>
                                            {/* --- 核心内容显示逻辑 --- */}
                                            {isThinking ? (
                                                <div className="flex items-center gap-1.5 h-6 px-2">
                                                    {/* 三个跳动的小球动画 */}
                                                    <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce [animation-delay:-0.3s]"></div>
                                                    <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce [animation-delay:-0.15s]"></div>
                                                    <div className="w-2 h-2 bg-blue-400 rounded-full animate-bounce"></div>
                                                    <span className="text-xs text-gray-400 ml-2">正在查阅知识库...</span>
                                                </div>
                                            ) : (
                                                /* 添加一个 div 包裹 ReactMarkdown，并将 className 放在这里 */
                                                <div className={`prose prose-sm max-w-none ${msg.role === 'user' ? 'prose-invert' : ''}`}>
                                                    {renderMessageContent(contentToShow || msg.content, msg.role)}
                                                </div>
                                            )}
                                        </div>
                                    </div>

                                    {/* 3. 右侧：用户头像 (只有用户消息才显示在右边) */}
                                    {msg.role === 'user' && (
                                        <div className="w-8 h-8 rounded-full bg-gray-200 flex items-center justify-center flex-shrink-0 mt-1">
                                            <User size={16} className="text-gray-500" />
                                        </div>
                                    )}
                                </div>
                            );
                        })}
                        <div ref={messagesEndRef} />
                    </div>
                </div>

                {/* 输入框区域 */}
                <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-white via-white to-transparent pt-4 pb-12 px-4">
                    <div className="max-w-3xl mx-auto relative group flex flex-col justify-end">

                        {/* --- 0. 录音状态提示 --- */}
                        {isRecording && (
                            <div className="absolute -top-16 left-1/2 transform -translate-x-1/2 bg-red-500 text-white px-4 py-1.5 rounded-full text-xs animate-pulse shadow-md flex items-center gap-2 z-20">
                                <span className="w-2 h-2 bg-white rounded-full animate-ping"></span>
                                正在录音... 点击麦克风结束
                            </div>
                        )}

                        {/* --- 1. 上层：文件预览区域 --- */}
                        {attachedFiles.length > 0 && (
                            <div className="mb-2 w-full flex gap-2 overflow-x-auto custom-scrollbar pb-1">
                                {attachedFiles.map((file, index) => (
                                    <div
                                        key={index}
                                        className="relative flex-shrink-0 flex items-center gap-2 bg-white border border-blue-200 text-blue-700 px-3 py-2 rounded-xl shadow-sm animate-in slide-in-from-bottom-2 fade-in duration-300"
                                    >
                                        {/* 左侧图标 */}
                                        <div className="bg-blue-50 p-1.5 rounded-full text-blue-500 flex-shrink-0">
                                            {file.type === 'image' ? <Paperclip size={14} /> : <GraduationCap size={14} />}
                                        </div>

                                        {/* 中间文件名 */}
                                        <div className="flex flex-col max-w-[120px]">
                                            <span className="text-xs font-medium truncate" title={file.fileName}>
                                                {file.fileName}
                                            </span>
                                            <span className="text-[10px] text-blue-400 uppercase leading-none">
                                                {file.type === 'image' ? '图片' : '文档'}
                                            </span>
                                        </div>

                                        {/* 删除按钮 */}
                                        <button
                                            onClick={() => removeFile(index)}
                                            className="ml-1 p-1 text-gray-400 hover:text-red-500 hover:bg-red-50 rounded-full transition-all duration-200"
                                            title="移除此文件"
                                        >
                                            <X size={14} />
                                        </button>
                                    </div>
                                ))}
                            </div>
                        )}

                        {/* --- 2. 下层：输入框主体 (Textarea + 按钮) --- */}
                        <div className="bg-white border border-gray-300 rounded-xl shadow-lg flex items-end p-2 gap-2 focus-within:ring-2 focus-within:ring-blue-500/20 focus-within:border-blue-400 z-10">
                            <textarea
                                value={input}
                                onChange={(e) => setInput(e.target.value)}
                                onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend(); } }}
                                placeholder={isRecording ? "正在听你说话..." : "输入您想学习的操作内容..."}
                                className="w-full max-h-32 bg-transparent border-none focus:ring-0 resize-none p-3 text-gray-700 placeholder-gray-400 text-sm"
                                rows={1}
                                disabled={isLoading || isRecording || isProcessingVoice}
                            />

                            {/* 按钮工具栏 */}
                            <div className="flex items-center mb-1 gap-1">
                                {/* 隐藏的 input */}
                                <input
                                    type="file"
                                    id="file-upload"
                                    multiple
                                    className="hidden"
                                    onChange={handleFileUpload}
                                    disabled={isLoading}
                                    accept=".pdf,.doc,.docx,.xls,.xlsx,.png,.jpg,.jpeg"
                                />
                                <label
                                    htmlFor="file-upload"
                                    className={`p-2 rounded-lg transition-all mr-1 cursor-pointer flex items-center justify-center
                                    ${isLoading
                                            ? 'bg-gray-50 text-gray-300 cursor-not-allowed'
                                            : 'bg-gray-100 text-gray-500 hover:bg-gray-200 active:scale-95'
                                        }`}
                                >
                                    <Paperclip size={20} />
                                </label>

                                {/* 语音按钮 */}
                                {isProcessingVoice ? (
                                    <div className="p-2 mr-1"><Loader2 size={20} className="animate-spin text-blue-500" /></div>
                                ) : (
                                    <button
                                        onClick={isRecording ? stopRecording : startRecording}
                                        disabled={isLoading}
                                        className={`p-2 rounded-lg transition-all mr-1 ${isRecording ? 'bg-red-100 text-red-600 animate-pulse' : 'bg-gray-100 text-gray-500 hover:bg-gray-200'}`}
                                    >
                                        {isRecording ? <StopCircle size={20} /> : <Mic size={20} />}
                                    </button>
                                )}

                                {/* 发送按钮 */}
                                {isLoading ? (
                                    <button onClick={handleStop} className="p-2 rounded-lg bg-red-50 text-red-500 hover:bg-red-100"><StopCircle size={20} /></button>
                                ) : (
                                    <button
                                        onClick={() => handleSend()}
                                        disabled={!input.trim() && attachedFiles.length === 0} // 只要有文件或有文字就可以发送
                                        className={`p-2 rounded-lg transition-all ${input.trim() || attachedFiles.length > 0 ? 'bg-blue-600 text-white hover:bg-blue-700' : 'bg-gray-100 text-gray-300 cursor-not-allowed'}`}
                                    >
                                        <Send size={18} />
                                    </button>
                                )}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            {playingVideo && (
                <VideoPlayerModal
                    video={playingVideo}
                    onClose={() => setPlayingVideo(null)}
                />
            )}
        </div >
    );
}