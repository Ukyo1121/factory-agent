import { useState, useRef, useEffect } from 'react';
import { v4 as uuidv4 } from 'uuid';
import {
  Send, Plus, MessageSquare, User, Bot, Loader2, StopCircle,
  Zap, Wrench, AlertTriangle, Database, Mic, ClipboardList
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import KnowledgeModal from './components/KnowledgeModal';
import UnansweredModal from './components/UnansweredModal';

const API_URL = "http://localhost:8000/chat";
// 语音接口地址
const VOICE_API_URL = "http://localhost:8000/voice-to-text";

function App() {
  // --- 状态管理 ---
  const [threads, setThreads] = useState([]);
  const [activeThreadId, setActiveThreadId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isKbOpen, setIsKbOpen] = useState(false);
  // --- 打字机效果专用状态 ---
  const [streamBuffer, setStreamBuffer] = useState("");
  const [displayedContent, setDisplayedContent] = useState("");
  const [isTyping, setIsTyping] = useState(false);

  // 语音相关状态
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessingVoice, setIsProcessingVoice] = useState(false);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);

  // 待解答问题相关状态
  const [isUnansweredOpen, setIsUnansweredOpen] = useState(false);
  const [unansweredCount, setUnansweredCount] = useState(0);

  const messagesEndRef = useRef(null);
  const abortControllerRef = useRef(null);

  // --- 初始化与获取未读数量 ---
  useEffect(() => {
    if (threads.length === 0) createNewThread();
    fetchUnansweredCount(); // 初始化时获取一次
  }, []);

  // 当弹窗关闭时，重新获取一次数量（因为可能刚刚处理完问题）
  useEffect(() => {
    if (!isUnansweredOpen) {
      fetchUnansweredCount();
    }
  }, [isUnansweredOpen]);

  const fetchUnansweredCount = async () => {
    try {
      const res = await fetch("http://localhost:8000/admin/unanswered_questions");
      const data = await res.json();
      setUnansweredCount(data.count || 0);
    } catch (e) {
      console.error("获取待解答数量失败", e);
    }
  };

  // --- 自动滚动 ---
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, displayedContent, isLoading]);

  // --- 打字机定时器 ---
  useEffect(() => {
    if (streamBuffer.length > displayedContent.length) {
      setIsTyping(true);
      const timer = setTimeout(() => {
        setDisplayedContent(prev => streamBuffer.slice(0, prev.length + 1));
      }, 20);
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

  // --- 创建新会话 ---
  const createNewThread = () => {
    const newId = uuidv4();
    const newThread = { id: newId, title: "新对话", history: [] };
    setThreads(prev => [newThread, ...prev]);
    setActiveThreadId(newId);
    setMessages([]);
    resetTyper();
  };

  // --- 切换会话 ---
  const switchThread = (id) => {
    if (isLoading) return;
    if (activeThreadId) {
      setThreads(prev => prev.map(t =>
        t.id === activeThreadId ? { ...t, history: messages } : t
      ));
    }
    const targetThread = threads.find(t => t.id === id);
    if (targetThread) {
      setActiveThreadId(id);
      setMessages(targetThread.history || []);
      resetTyper();
    }
  };

  const resetTyper = () => {
    setStreamBuffer("");
    setDisplayedContent("");
    setIsTyping(false);
  };

  // --- 发送消息 ---
  const handleSend = async (manualInput = null) => {
    const textToSend = manualInput || input;
    if (!textToSend.trim() || isLoading) return;

    setMessages(prev => [...prev, { role: 'user', content: textToSend }]);
    setInput("");
    setIsLoading(true);
    resetTyper();
    setMessages(prev => [...prev, { role: 'ai', content: "" }]);
    abortControllerRef.current = new AbortController();

    try {
      const response = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: textToSend,
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
        t.id === activeThreadId && t.title === "新对话"
          ? { ...t, title: textToSend }
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

  // 录音功能函数
  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorderRef.current = new MediaRecorder(stream);
      audioChunksRef.current = [];

      mediaRecorderRef.current.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorderRef.current.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        await sendAudioToBackend(audioBlob);

        // 停止所有轨道
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
      setIsProcessingVoice(true); // 开始转圈圈等待识别
    }
  };

  const sendAudioToBackend = async (audioBlob) => {
    const formData = new FormData();
    // 添加文件，文件名后缀很重要，webm 是浏览器录音的标准格式
    formData.append("file", audioBlob, "voice_input.webm");

    try {
      const response = await fetch(VOICE_API_URL, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) throw new Error("识别失败");

      const data = await response.json();
      if (data.text) {
        setInput(prev => prev + data.text); // 将识别结果追加到输入框
      }
    } catch (error) {
      console.error("语音识别错误:", error);
      alert("语音识别失败，请重试");
    } finally {
      setIsProcessingVoice(false);
    }
  };

  return (
    <div className="flex h-screen bg-gray-50 text-gray-800 font-sans">
      {/* 侧边栏 */}
      <div className="w-64 bg-gray-900 text-white flex flex-col flex-shrink-0">
        <div className="p-4">
          <button
            onClick={createNewThread}
            disabled={isLoading}
            className={`w-full flex items-center gap-2 bg-gray-800 p-3 rounded-md border border-gray-700 text-sm transition-colors ${isLoading ? 'opacity-50 cursor-not-allowed' : 'hover:bg-gray-700'}`}
          >
            <Plus size={16} /> 新建对话
          </button>
        </div>
        <div className="flex-1 overflow-y-auto px-2 custom-scrollbar">
          {threads.map(thread => (
            <button
              key={thread.id}
              onClick={() => switchThread(thread.id)}
              disabled={isLoading}
              className={`w-full text-left p-3 rounded-md mb-1 text-sm flex items-center gap-2 truncate transition-colors ${activeThreadId === thread.id ? 'bg-gray-800 text-white' : 'text-gray-400 hover:bg-gray-800'
                }`}
            >
              <MessageSquare size={14} className="flex-shrink-0" />
              <span className="truncate">{thread.title}</span>
            </button>
          ))}
        </div>
        {/* 底部按钮区域 */}
        <div className="p-4 border-t border-gray-800 space-y-2">
          {/* 1. 知识库按钮 */}
          <button onClick={() => setIsKbOpen(true)} className="w-full flex items-center gap-2 text-gray-400 hover:text-white hover:bg-gray-800 p-2 rounded-md transition-colors text-sm">
            <Database size={16} /> 管理知识库
          </button>

          {/* 2. 待解答问题按钮 (动态样式) */}
          <button
            onClick={() => setIsUnansweredOpen(true)}
            className={`w-full flex items-center gap-2 p-2 rounded-md transition-all text-sm group ${unansweredCount > 0
              ? "text-orange-400 hover:text-orange-300 hover:bg-gray-800 font-medium"  // 有问题：高亮橙色
              : "text-gray-400 hover:text-white hover:bg-gray-800"                      // 无问题：普通灰色
              }`}
          >
            <ClipboardList size={16} className={unansweredCount > 0 ? "animate-pulse" : ""} />
            待解答问题库

            {/* 数字徽标 */}
            {unansweredCount > 0 && (
              <span className="ml-auto bg-orange-500 text-white text-[10px] font-bold px-2 py-0.5 rounded-full shadow-sm group-hover:bg-orange-400">
                {unansweredCount}
              </span>
            )}
          </button>
        </div>
      </div>

      {/* 主界面 */}
      <div className="flex-1 flex flex-col relative bg-white">
        <div className="h-14 border-b flex items-center px-6 shadow-sm z-10 bg-white">
          <h1 className="font-semibold text-gray-700 flex items-center gap-2">
            <Bot className="text-blue-600" size={20} />
            工厂智能助手
            <span className="text-xs text-gray-400 font-normal px-2 py-0.5 bg-gray-100 rounded-full">Pro</span>
          </h1>
        </div>

        <div className="flex-1 overflow-y-auto p-4 pb-32 custom-scrollbar">
          <div className="max-w-3xl mx-auto space-y-6 min-h-full flex flex-col">
            {/* 欢迎界面 */}
            {messages.length === 0 && (
              <div className="flex-1 flex flex-col items-center justify-center text-center mt-10">
                <div className="w-20 h-20 bg-white rounded-2xl shadow-sm border border-gray-100 flex items-center justify-center mb-6">
                  <Bot size={40} className="text-blue-600" />
                </div>
                <h2 className="text-2xl font-bold text-gray-800 mb-3">有什么可以帮你的吗？</h2>
                <p className="text-gray-500 mb-10 max-w-md">
                  我可以帮你查询工厂设备故障、解析错误码、搜索PDF手册或提供详细的维修步骤。
                </p>

                {/* 快捷提示词卡片 */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 w-full max-w-2xl px-4">
                  <button
                    onClick={() => handleSend("错误码303是什么意思")}
                    className="flex items-center gap-3 p-4 bg-white border border-gray-200 rounded-xl hover:border-blue-400 hover:shadow-md transition-all text-left group"
                  >
                    <div className="w-10 h-10 bg-red-50 rounded-lg flex items-center justify-center group-hover:bg-blue-50 transition-colors">
                      <AlertTriangle size={20} className="text-red-500 group-hover:text-blue-600" />
                    </div>
                    <div>
                      <div className="font-semibold text-gray-700 group-hover:text-blue-700">查询错误码</div>
                      <div className="text-xs text-gray-400">错误码303是什么意思</div>
                    </div>
                  </button>

                  <button
                    onClick={() => handleSend("操作FANUC机器人时急停了怎么办")}
                    className="flex items-center gap-3 p-4 bg-white border border-gray-200 rounded-xl hover:border-blue-400 hover:shadow-md transition-all text-left group"
                  >
                    <div className="w-10 h-10 bg-yellow-50 rounded-lg flex items-center justify-center group-hover:bg-blue-50 transition-colors">
                      <Zap size={20} className="text-yellow-600 group-hover:text-blue-600" />
                    </div>
                    <div>
                      <div className="font-semibold text-gray-700 group-hover:text-blue-700">紧急故障</div>
                      <div className="text-xs text-gray-400">操作FANUC机器人时急停了怎么办</div>
                    </div>
                  </button>

                  <button
                    onClick={() => handleSend("自动分拣系统的操作步骤")}
                    className="flex items-center gap-3 p-4 bg-white border border-gray-200 rounded-xl hover:border-blue-400 hover:shadow-md transition-all text-left group"
                  >
                    <div className="w-10 h-10 bg-green-50 rounded-lg flex items-center justify-center group-hover:bg-blue-50 transition-colors">
                      <Wrench size={20} className="text-green-600 group-hover:text-blue-600" />
                    </div>
                    <div>
                      <div className="font-semibold text-gray-700 group-hover:text-blue-700">操作规程</div>
                      <div className="text-xs text-gray-400">自动分拣系统的操作步骤</div>
                    </div>
                  </button>

                  <button
                    onClick={() => handleSend("FAUNC机器人开机零点校准故障报警怎么处理")}
                    className="flex items-center gap-3 p-4 bg-white border border-gray-200 rounded-xl hover:border-blue-400 hover:shadow-md transition-all text-left group"
                  >
                    <div className="w-10 h-10 bg-purple-50 rounded-lg flex items-center justify-center group-hover:bg-blue-50 transition-colors">
                      <Bot size={20} className="text-purple-600 group-hover:text-blue-600" />
                    </div>
                    <div>
                      <div className="font-semibold text-gray-700 group-hover:text-blue-700">维修指导</div>
                      <div className="text-xs text-gray-400">FAUNC机器人开机零点校准故障报警怎么处理</div>
                    </div>
                  </button>
                </div>
              </div>
            )}

            {/* 消息列表 */}
            {messages.map((msg, idx) => {
              const isLastAiMessage = msg.role === 'ai' && idx === messages.length - 1;
              const contentToShow = isLastAiMessage && (isLoading || isTyping) ? displayedContent : msg.content;
              return (
                <div key={idx} className={`flex gap-4 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                  {msg.role === 'ai' && (
                    <div className="w-8 h-8 rounded-full bg-blue-50 border border-blue-100 flex items-center justify-center flex-shrink-0 mt-1">
                      <Bot size={16} className="text-blue-600" />
                    </div>
                  )}
                  <div className={`max-w-[85%] p-4 rounded-2xl text-sm leading-7 shadow-sm ${msg.role === 'user'
                    ? 'bg-blue-600 text-white rounded-br-none'
                    : 'bg-white border border-gray-100 text-gray-800 rounded-bl-none'
                    }`}>
                    {msg.role === 'ai' ? (
                      <div>
                        {(!contentToShow && isLoading) ? (
                          <div className="flex items-center gap-2 text-gray-400 py-1">
                            <Loader2 size={16} className="animate-spin" />
                            <span className="text-xs">正在思考并检索知识库...</span>
                          </div>
                        ) : (
                          <ReactMarkdown
                            components={{
                              ul: ({ node, ...props }) => <ul className="list-disc pl-4 my-2 space-y-1" {...props} />,
                              ol: ({ node, ...props }) => <ol className="list-decimal pl-4 my-2 space-y-1" {...props} />,
                              strong: ({ node, ...props }) => <span className="font-bold text-blue-700 bg-blue-50 px-1 rounded" {...props} />,
                              h1: ({ node, ...props }) => <h1 className="text-xl font-bold my-3 border-b pb-2" {...props} />,
                              h2: ({ node, ...props }) => <h2 className="text-lg font-bold my-2" {...props} />,
                              code: ({ node, inline, className, children, ...props }) => {
                                return inline ? (
                                  <code className="bg-gray-100 text-red-500 px-1 py-0.5 rounded text-xs font-mono" {...props}>{children}</code>
                                ) : (
                                  <pre className="bg-gray-800 text-gray-100 p-3 rounded-lg overflow-x-auto my-2 text-xs font-mono">
                                    <code {...props}>{children}</code>
                                  </pre>
                                )
                              },
                              img: ({ node, ...props }) => (
                                <img
                                  {...props}
                                  className="max-w-full h-auto rounded-lg shadow-md my-4 border border-gray-200 cursor-zoom-in"
                                  onClick={() => window.open(props.src, '_blank')}
                                  alt="操作示意图"
                                />
                              )
                            }}
                          >
                            {contentToShow}
                          </ReactMarkdown>
                        )}
                        {isLastAiMessage && (isLoading || isTyping) && (
                          <span className="inline-block w-1.5 h-4 ml-1 align-middle bg-blue-600 animate-pulse"></span>
                        )}
                      </div>
                    ) : (
                      <div className="whitespace-pre-wrap">{msg.content}</div>
                    )}
                  </div>
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
        <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-white via-white to-transparent pt-12 pb-6 px-4">
          <div className="max-w-3xl mx-auto relative group">
            {/* 语音识别中的提示条 */}
            {isRecording && (
              <div className="absolute -top-12 left-1/2 transform -translate-x-1/2 bg-red-500 text-white px-4 py-1.5 rounded-full text-xs animate-pulse shadow-md flex items-center gap-2 z-20">
                <span className="w-2 h-2 bg-white rounded-full animate-ping"></span>
                正在录音... 点击麦克风结束
              </div>
            )}

            <div className="bg-white border border-gray-300 rounded-xl shadow-lg flex items-end p-2 gap-2 focus-within:ring-2 focus-within:ring-blue-500/20 focus-within:border-blue-400 transition-all">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleSend(null);
                  }
                }}
                placeholder={isRecording ? "正在听你说话..." : (isProcessingVoice ? "正在识别语音..." : "描述故障现象 (如: 机械臂抖动) 或输入错误码...")}
                className="w-full max-h-32 bg-transparent border-none focus:ring-0 resize-none p-3 text-gray-700 placeholder-gray-400 text-sm"
                rows={1}
                disabled={isLoading || isRecording || isProcessingVoice}
              />

              {/* 添加麦克风按钮 */}
              <div className="flex items-center mb-1 gap-1">
                {/* 如果正在处理语音，显示 Loading */}
                {isProcessingVoice ? (
                  <div className="p-2 mr-1">
                    <Loader2 size={20} className="animate-spin text-blue-500" />
                  </div>
                ) : (
                  <button
                    onClick={isRecording ? stopRecording : startRecording}
                    disabled={isLoading}
                    className={`p-2 rounded-lg transition-all mr-1 ${isRecording
                      ? 'bg-red-100 text-red-600 hover:bg-red-200 animate-pulse'
                      : 'bg-gray-100 text-gray-500 hover:bg-gray-200 hover:text-gray-700'
                      } ${isLoading ? 'opacity-50 cursor-not-allowed' : ''}`}
                    title={isRecording ? "停止录音" : "语音输入"}
                  >
                    {isRecording ? <StopCircle size={20} /> : <Mic size={20} />}
                  </button>
                )}

                {isLoading ? (
                  <button
                    onClick={handleStop}
                    className="p-2 rounded-lg bg-red-50 text-red-500 hover:bg-red-100 transition-colors"
                  >
                    <StopCircle size={20} />
                  </button>
                ) : (
                  <button
                    onClick={() => handleSend(null)}
                    disabled={!input.trim()}
                    className={`p-2 rounded-lg transition-all ${input.trim()
                      ? 'bg-blue-600 text-white hover:bg-blue-700 shadow-md'
                      : 'bg-gray-100 text-gray-300 cursor-not-allowed'
                      }`}
                  >
                    <Send size={18} />
                  </button>
                )}
              </div>
            </div>
            <p className="text-center text-xs text-gray-400 mt-2 opacity-0 group-hover:opacity-100 transition-opacity">
              AI 内容由 Factory Agent 生成，仅供参考
            </p>
          </div>
        </div>
      </div>
      <KnowledgeModal isOpen={isKbOpen} onClose={() => setIsKbOpen(false)} />
      <UnansweredModal
        isOpen={isUnansweredOpen}
        onClose={() => setIsUnansweredOpen(false)}
      />
    </div>
  );
}

export default App;