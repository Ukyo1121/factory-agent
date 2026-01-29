import { useState, useRef, useEffect } from 'react';
import { v4 as uuidv4 } from 'uuid';
import {
    Send, Plus, MessageSquare, User, Bug, Loader2,
    Terminal, ArrowLeft, Mic, StopCircle
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { API_BASE_URL } from "./config";

const API_URL = `${API_BASE_URL}/chat`;
const VOICE_API_URL = `${API_BASE_URL}/voice`;

export default function DebugAssistant({ onBack }) {
    const [threads, setThreads] = useState([]);
    const [activeThreadId, setActiveThreadId] = useState(null);
    const [messages, setMessages] = useState([]);
    const [input, setInput] = useState("");
    const [isLoading, setIsLoading] = useState(false);

    // 打字机状态
    const [streamBuffer, setStreamBuffer] = useState("");
    const [displayedContent, setDisplayedContent] = useState("");
    const [isTyping, setIsTyping] = useState(false);

    // --- 语音状态 ---
    const [isRecording, setIsRecording] = useState(false);
    const [isProcessingVoice, setIsProcessingVoice] = useState(false);
    const mediaRecorderRef = useRef(null);
    const audioChunksRef = useRef([]);

    const messagesEndRef = useRef(null);
    const abortControllerRef = useRef(null);
    const isInitializedRef = useRef(false);

    useEffect(() => {
        if (!isInitializedRef.current) {
            isInitializedRef.current = true;
            if (threads.length === 0) createNewThread();
        }
    }, []);

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [messages, displayedContent, isLoading]);

    // 打字机效果
    useEffect(() => {
        if (streamBuffer.length > displayedContent.length) {
            setIsTyping(true);
            const timer = setTimeout(() => {
                setDisplayedContent(prev => streamBuffer.slice(0, prev.length + 1));
            }, 10); // 调试模式语速稍快
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

    const createNewThread = (title = "新调试会话") => {
        const newId = uuidv4();
        const newThread = { id: newId, title: title, history: [] };
        setThreads(prev => [newThread, ...prev]);
        setActiveThreadId(newId);
        setMessages([]);
        setStreamBuffer("");
        setDisplayedContent("");
    };

    const switchThread = (id) => {
        if (isLoading) return;
        const targetThread = threads.find(t => t.id === id);
        if (targetThread) {
            setActiveThreadId(id);
            setMessages(targetThread.history || []);
            setStreamBuffer("");
            setDisplayedContent("");
        }
    };

    const handleSend = async (manualInput = null) => {
        const textToSend = manualInput || input;
        if (!textToSend.trim() || isLoading) return;

        setMessages(prev => [...prev, { role: 'user', content: textToSend }]);
        setInput("");
        setIsLoading(true);
        setStreamBuffer("");
        setDisplayedContent("");
        setMessages(prev => [...prev, { role: 'ai', content: "" }]);
        abortControllerRef.current = new AbortController();

        // 关键：自动添加 [调试模式] 前缀，帮助 Agent 识别意图
        const queryWithPrefix = `[调试模式] ${textToSend}`;

        try {
            const response = await fetch(API_URL, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    query: queryWithPrefix,
                    thread_id: activeThreadId
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

            setThreads(prev => prev.map(t =>
                t.id === activeThreadId && t.title === "新调试会话"
                    ? { ...t, title: textToSend.slice(0, 15) } // 截取前15个字作为标题
                    : t
            ));
        } catch (error) {
            if (error.name !== 'AbortError') {
                setStreamBuffer(prev => prev + "\n\n⚠️ 连接服务器失败，请检查后端。");
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
            {/* 紫色系侧边栏 */}
            <div className="w-64 bg-gray-900 text-white flex flex-col flex-shrink-0 shadow-xl z-20">
                <div className="p-4 border-b border-gray-800 flex items-center gap-3">
                    <button onClick={onBack} className="p-2 hover:bg-gray-800 rounded-lg text-gray-400 hover:text-white transition-colors">
                        <ArrowLeft size={20} />
                    </button>
                    <h1 className="font-bold text-lg flex items-center gap-2">
                        <Terminal className="text-purple-500" size={24} />
                        调试助手
                    </h1>
                </div>

                <div className="p-4">
                    <button
                        onClick={() => createNewThread("新调试会话")}
                        disabled={isLoading}
                        className="w-full flex items-center gap-2 bg-purple-600 hover:bg-purple-700 p-3 rounded-lg text-sm transition-all shadow-md group border border-purple-500"
                    >
                        <Plus size={16} className="group-hover:rotate-90 transition-transform" /> 新建调试
                    </button>
                </div>

                <div className="flex-1 overflow-y-auto px-2 custom-scrollbar">
                    {threads.map(thread => (
                        <button
                            key={thread.id}
                            onClick={() => switchThread(thread.id)}
                            disabled={isLoading}
                            className={`w-full text-left p-3 rounded-lg mb-1 text-sm flex items-center gap-2 truncate transition-colors ${activeThreadId === thread.id
                                ? 'bg-gray-800 text-white border-l-2 border-purple-500'
                                : 'text-gray-400 hover:bg-gray-800'
                                }`}
                        >
                            <Terminal size={14} className="flex-shrink-0 text-purple-400" />
                            <span className="truncate">{thread.title}</span>
                        </button>
                    ))}
                </div>
            </div>

            {/* 主界面 */}
            <div className="flex-1 flex flex-col relative bg-white">
                <div className="h-14 border-b flex items-center px-6 shadow-sm z-10 bg-white/80 backdrop-blur-md justify-between">
                    <h1 className="font-semibold text-gray-700 flex items-center gap-2 font-sans">
                        <Bug className="text-purple-600" size={20} />
                        <span>代码与机器故障排查</span>
                    </h1>
                    <div className="text-xs text-purple-500 border border-purple-200 px-2 py-0.5 rounded-full bg-purple-50">
                        Debug Mode Active
                    </div>
                </div>

                <div className="flex-1 overflow-y-auto p-4 pb-32 custom-scrollbar">
                    <div className="max-w-4xl mx-auto space-y-6 min-h-full flex flex-col">
                        {messages.length === 0 && (
                            <div className="flex-1 flex flex-col items-center justify-center text-center mt-10">
                                <div className="w-20 h-20 bg-purple-50 rounded-2xl shadow-sm border border-purple-100 flex items-center justify-center mb-6">
                                    <Terminal size={40} className="text-purple-600" />
                                </div>
                                <h2 className="text-2xl font-bold text-gray-800 mb-3 font-sans">遇到机器故障或代码报错了吗？</h2>
                                <p className="text-gray-500 mb-10 max-w-md font-sans">
                                    请告诉我故障现象或粘贴错误日志、报错代码，我将协助您进行定位和修复。
                                </p>

                                {/* 仅保留2个快捷卡片 (一软一硬) */}
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 w-full max-w-3xl px-4 font-sans">

                                    {/* 1. 机器故障维修 */}
                                    <button
                                        onClick={() => handleSend("操作FANUC机器人时急停了怎么处理")}
                                        className="flex items-center gap-3 p-4 bg-white border border-gray-200 rounded-xl hover:border-purple-400 hover:shadow-md transition-all text-left group"
                                    >
                                        <div className="w-10 h-10 bg-purple-50 rounded-lg flex items-center justify-center group-hover:bg-purple-100 transition-colors">
                                            <Terminal size={20} className="text-purple-600" />
                                        </div>
                                        <div>
                                            <div className="font-semibold text-gray-700 group-hover:text-purple-700">机器故障维修</div>
                                            <div className="text-xs text-gray-400">例：操作FANUC机器人时急停了怎么处理</div>
                                        </div>
                                    </button>

                                    {/* 2. 代码异常修复 */}
                                    <button
                                        onClick={() => handleSend("Python脚本报 KeyError: 'status'")}
                                        className="flex items-center gap-3 p-4 bg-white border border-gray-200 rounded-xl hover:border-purple-400 hover:shadow-md transition-all text-left group"
                                    >
                                        <div className="w-10 h-10 bg-purple-50 rounded-lg flex items-center justify-center group-hover:bg-purple-100 transition-colors">
                                            <Bug size={20} className="text-purple-600" />
                                        </div>
                                        <div>
                                            <div className="font-semibold text-gray-700 group-hover:text-purple-700">代码异常修复</div>
                                            <div className="text-xs text-gray-400">例：Python脚本报 KeyError: 'status'</div>
                                        </div>
                                    </button>

                                </div>
                            </div>
                        )}

                        {messages.map((msg, idx) => {
                            const isLastAiMessage = msg.role === 'ai' && idx === messages.length - 1;
                            const contentToShow = isLastAiMessage && (isLoading || isTyping) ? displayedContent : msg.content;
                            return (
                                <div key={idx} className={`flex gap-4 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                                    {msg.role === 'ai' && <div className="w-8 h-8 rounded-full bg-purple-50 border border-purple-100 flex items-center justify-center flex-shrink-0 mt-1"><Bug size={16} className="text-purple-600" /></div>}
                                    <div className={`max-w-[90%] p-4 rounded-2xl text-sm leading-7 shadow-sm ${msg.role === 'user' ? 'bg-purple-600 text-white rounded-br-none' : 'bg-gray-50 border border-gray-100 text-gray-800 rounded-bl-none font-mono'}`}>
                                        <ReactMarkdown
                                            components={{
                                                code: ({ node, inline, className, children, ...props }) => {
                                                    return !inline ? (
                                                        <pre className="bg-gray-900 text-green-400 p-3 rounded-lg overflow-x-auto my-2 text-xs font-mono border border-gray-700">
                                                            <code {...props}>{children}</code>
                                                        </pre>
                                                    ) : (
                                                        <code className="bg-purple-100 text-purple-700 px-1 py-0.5 rounded text-xs" {...props}>{children}</code>
                                                    )
                                                }
                                            }}
                                        >{contentToShow}</ReactMarkdown>
                                    </div>
                                    {msg.role === 'user' && <div className="w-8 h-8 rounded-full bg-gray-200 flex items-center justify-center flex-shrink-0 mt-1"><User size={16} className="text-gray-500" /></div>}
                                </div>
                            )
                        })}
                        <div ref={messagesEndRef} />
                    </div>
                </div>

                {/* 输入框 */}
                <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-white via-white to-transparent pt-12 pb-6 px-4">
                    <div className="max-w-4xl mx-auto relative group">

                        {isRecording && (
                            <div className="absolute -top-12 left-1/2 transform -translate-x-1/2 bg-red-500 text-white px-4 py-1.5 rounded-full text-xs animate-pulse shadow-md flex items-center gap-2 z-20">
                                <span className="w-2 h-2 bg-white rounded-full animate-ping"></span>
                                正在录音... 点击麦克风结束
                            </div>
                        )}

                        <div className="bg-white border border-gray-300 rounded-xl shadow-lg flex items-end p-2 gap-2 focus-within:ring-2 focus-within:ring-purple-500/20 focus-within:border-purple-400">
                            <textarea
                                value={input}
                                onChange={(e) => setInput(e.target.value)}
                                onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend(); } }}
                                placeholder={isRecording ? "正在听你说话..." : "描述故障现象或粘贴代码片段与错误日志..."}
                                className="w-full max-h-32 bg-transparent border-none focus:ring-0 resize-none p-3 text-gray-700 placeholder-gray-400 text-sm"
                                rows={1}
                                disabled={isLoading || isRecording || isProcessingVoice}
                            />

                            <div className="flex items-center mb-1 gap-1">
                                {/* 语音按钮 */}
                                {isProcessingVoice ? (
                                    <div className="p-2 mr-1"><Loader2 size={20} className="animate-spin text-purple-500" /></div>
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
                                        disabled={!input.trim()}
                                        className={`p-2 rounded-lg transition-all ${input.trim()
                                            ? 'bg-purple-600 text-white hover:bg-purple-700'
                                            : 'bg-gray-100 text-gray-300 cursor-not-allowed'
                                            }`}
                                    >
                                        <Send size={18} />
                                    </button>
                                )}
                            </div>
                        </div>
                        <p className="text-center text-xs text-gray-400 mt-2 opacity-0 group-hover:opacity-100 transition-opacity">
                            调试模式下建议提供完整 Log 以便分析
                        </p>
                    </div>
                </div>
            </div>
        </div>
    );
}